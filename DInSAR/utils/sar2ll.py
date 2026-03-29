#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
sar2ll.py - 将 SAR 坐标系数据重采样到地理坐标系（经纬度或 UTM）

输入：
- SAR 数据（GeoTIFF/.npy 或直接从 geosar HDF 读取 slc/sar_dem）
- geosar.py 输出的 HDF（包含 DEM->SAR 映射表 range_pixel/azimuth_pixel、dem_geotransform、dem_shape、SAR 尺寸等）

输出：
- GeoTIFF（经纬度或 UTM），支持单波段/多波段/复数据

插值：
- 实数：nearest / bilinear
- 复数：默认 sinc（Lanczos-windowed sinc），也可选 nearest/bilinear（不推荐）

实现方式（避免 KDTree / griddata）：
对每个输出栅格像素中心的 (lon,lat)：
  1) 反解 DEM geotransform -> (dem_row, dem_col)
  2) 对 (range_pixel_grid, azimuth_pixel_grid) 做双线性采样 -> (sar_rg, sar_az)
  3) 在 SAR 图像上按 (sar_az, sar_rg) 采样得到输出值（复数用 sinc）
"""

import argparse
import os
from typing import Tuple, Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _require_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as _np  # type: ignore
        np = _np  # type: ignore
        return np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少依赖：numpy。请在你的工作环境中安装后再运行。") from e


def _require_h5py():
    try:
        import h5py  # type: ignore
        return h5py
    except Exception as e:
        raise RuntimeError("缺少依赖：h5py。请在你的工作环境中安装后再运行。") from e


def _try_import_gdal():
    try:
        from osgeo import gdal, osr
        return gdal, osr
    except Exception:
        return None, None


def _try_import_numba():
    try:
        from numba import njit, prange
        return njit, prange
    except Exception:
        return None, None


def _load_hdf_mapping(hdf_path: str):
    _require_numpy()
    h5py = _require_h5py()
    with h5py.File(hdf_path, "r") as f:
        dem_gt = tuple(np.asarray(f["dem_geotransform"][:], dtype=np.float64).tolist())
        dem_shape = tuple(np.asarray(f["dem_shape"][:], dtype=np.int64).tolist())
        nrows, ncols = dem_shape

        rg = f["range_pixel"][:]
        az = f["azimuth_pixel"][:]
        if rg.ndim == 1:
            rg = rg.reshape((nrows, ncols))
        if az.ndim == 1:
            az = az.reshape((nrows, ncols))
        rg = np.asarray(rg, dtype=np.float32, order="C")
        az = np.asarray(az, dtype=np.float32, order="C")

        sar_az = None
        sar_rg = None
        if "sar_azimuth_lines" in f and "sar_range_samples" in f:
            sar_az = int(f["sar_azimuth_lines"][()])
            sar_rg = int(f["sar_range_samples"][()])
            if sar_az <= 0 or sar_rg <= 0:
                sar_az = None
                sar_rg = None
        if sar_az is None and "sar_az_size" in f and "sar_range_size" in f:
            sar_az = int(f["sar_az_size"][()])
            sar_rg = int(f["sar_range_size"][()])
        if sar_az is None:
            sar_az = int(np.nanmax(az)) + 1
        if sar_rg is None:
            sar_rg = int(np.nanmax(rg)) + 1

    return dem_gt, dem_shape, rg, az, (sar_az, sar_rg)


def _load_sar_input(sar_path: str, geosar_hdf: Optional[str] = None):
    """
    返回 (data, is_complex)
    - GeoTIFF:
      - 1 band: float32
      - 2 band: 默认解释为 (real, imag) -> complex64
      - >2 band: float32 多波段
    - .npy: 支持 real 或 complex
    - HDF:<name>：从 geosar_hdf 读取 dataset，例如 HDF:slc / HDF:sar_dem
    """
    _require_numpy()
    if sar_path.upper().startswith("HDF:"):
        if not geosar_hdf:
            raise ValueError("使用 HDF:<name> 输入时必须提供 --geosar-hdf")
        name = sar_path.split(":", 1)[1]
        h5py = _require_h5py()
        with h5py.File(geosar_hdf, "r") as f:
            if name == "slc":
                re = np.asarray(f["slc_real"][:], dtype=np.float32)
                im = np.asarray(f["slc_imag"][:], dtype=np.float32)
                return (re + 1j * im).astype(np.complex64, copy=False), True
            if name == "slc_update":
                re = np.asarray(f["slc_update_real"][:], dtype=np.float32)
                im = np.asarray(f["slc_update_imag"][:], dtype=np.float32)
                return (re + 1j * im).astype(np.complex64, copy=False), True
            arr = f[name][:]
            arr = np.asarray(arr)
            return arr, np.iscomplexobj(arr)

    ext = os.path.splitext(sar_path)[1].lower()
    if ext == ".npy":
        arr = np.load(sar_path)
        return arr, np.iscomplexobj(arr)

    if ext in [".tif", ".tiff"]:
        gdal, _ = _try_import_gdal()
        if gdal is None:
            raise RuntimeError("读取 GeoTIFF 需要 osgeo.gdal")
        ds = gdal.Open(sar_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"无法打开 SAR 文件: {sar_path}")
        bands = ds.RasterCount
        if bands == 1:
            b1 = ds.GetRasterBand(1)
            dt = b1.DataType
            # 支持单波段 complex GeoTIFF
            if dt in (gdal.GDT_CFloat32, gdal.GDT_CFloat64):
                arr = b1.ReadAsArray()
                arr = np.asarray(arr)
                return arr.astype(np.complex64, copy=False), True
            arr = b1.ReadAsArray().astype(np.float32)
            return arr, False
        if bands == 2:
            re = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            im = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
            return (re + 1j * im).astype(np.complex64, copy=False), True
        # 多波段
        stack = []
        for i in range(1, bands + 1):
            stack.append(ds.GetRasterBand(i).ReadAsArray().astype(np.float32))
        arr = np.stack(stack, axis=-1)
        return arr, False

    raise ValueError(f"不支持的 SAR 输入格式: {sar_path}")


def _inv_geo_transform(gt: Tuple[float, float, float, float, float, float]):
    _require_numpy()
    gt0, gt1, gt2, gt3, gt4, gt5 = gt
    det = gt1 * gt5 - gt2 * gt4
    if abs(det) < 1e-20:
        raise ValueError("DEM geotransform 不可逆（det≈0）")
    ic11 = gt5 / det
    ic12 = -gt2 / det
    ic21 = -gt4 / det
    ic22 = gt1 / det
    return gt0, gt3, (ic11, ic12, ic21, ic22)


def lonlat_to_dem_rc(gt, lon, lat):
    """(lon,lat) -> (row,col) 连续像素坐标（像素中心体系）"""
    _require_numpy()
    gt0, gt3, inv = _inv_geo_transform(gt)
    ic11, ic12, ic21, ic22 = inv
    dx = lon - gt0
    dy = lat - gt3
    c0 = ic11 * dx + ic12 * dy
    r0 = ic21 * dx + ic22 * dy
    return r0 - 0.5, c0 - 0.5


def _bilinear(grid, row, col, fill=float("nan")):
    _require_numpy()
    nrows, ncols = grid.shape
    r = row
    c = col
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1
    ok = (r0 >= 0) & (c0 >= 0) & (r1 < nrows) & (c1 < ncols) & np.isfinite(r) & np.isfinite(c)
    out = np.full(r.shape, fill, dtype=np.float64)
    if not np.any(ok):
        return out
    dr = r[ok] - r0[ok]
    dc = c[ok] - c0[ok]
    v00 = grid[r0[ok], c0[ok]].astype(np.float64, copy=False)
    v01 = grid[r0[ok], c1[ok]].astype(np.float64, copy=False)
    v10 = grid[r1[ok], c0[ok]].astype(np.float64, copy=False)
    v11 = grid[r1[ok], c1[ok]].astype(np.float64, copy=False)
    out[ok] = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )
    return out


def _sample_bilinear_src(src2d: np.ndarray, yy: np.ndarray, xx: np.ndarray, fill=0.0):
    """src[y,x] 双线性采样；yy/xx 为 float64 1D"""
    _require_numpy()
    h, w = src2d.shape
    y = yy
    x = xx
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    ok = (y0 >= 0) & (x0 >= 0) & (y1 < h) & (x1 < w) & np.isfinite(y) & np.isfinite(x)
    out = np.full(y.shape, fill, dtype=np.float32)
    if not np.any(ok):
        return out
    dy = (y[ok] - y0[ok]).astype(np.float32)
    dx = (x[ok] - x0[ok]).astype(np.float32)
    v00 = src2d[y0[ok], x0[ok]]
    v01 = src2d[y0[ok], x1[ok]]
    v10 = src2d[y1[ok], x0[ok]]
    v11 = src2d[y1[ok], x1[ok]]
    out[ok] = (
        v00 * (1 - dy) * (1 - dx)
        + v01 * (1 - dy) * dx
        + v10 * dy * (1 - dx)
        + v11 * dy * dx
    )
    return out


def _lanczos_kernel(x: np.ndarray, a: int):
    # Lanczos: sinc(x) * sinc(x/a)
    # 这里 x 是“像素偏移”，sinc 使用 np.sinc(pi*x)/(pi*x) 的归一化形式 => np.sinc(x)
    x = x.astype(np.float64, copy=False)
    out = np.sinc(x) * np.sinc(x / float(a))
    out[np.abs(x) >= a] = 0.0
    return out


def _sample_lanczos_numba(src2d: np.ndarray, yy: np.ndarray, xx: np.ndarray, a: int, fill: float):
    _require_numpy()
    njit, prange = _try_import_numba()
    if njit is None:
        return None

    # numba 版本：yy/xx 为 1D float64
    @njit(parallel=True, fastmath=True, cache=True)
    def _core(src, y, x, a, fill):
        h, w = src.shape
        out = np.empty(y.shape[0], dtype=np.float32)
        for i in prange(y.shape[0]):
            yi = y[i]
            xi = x[i]
            if not np.isfinite(yi) or not np.isfinite(xi):
                out[i] = fill
                continue
            y0 = int(np.floor(yi))
            x0 = int(np.floor(xi))
            acc = 0.0
            wsum = 0.0
            for dy in range(-a + 1, a + 1):
                yy0 = y0 + dy
                if yy0 < 0 or yy0 >= h:
                    continue
                wy = np.sinc(yi - yy0) * np.sinc((yi - yy0) / a)
                if abs(yi - yy0) >= a:
                    wy = 0.0
                for dx in range(-a + 1, a + 1):
                    xx0 = x0 + dx
                    if xx0 < 0 or xx0 >= w:
                        continue
                    wx = np.sinc(xi - xx0) * np.sinc((xi - xx0) / a)
                    if abs(xi - xx0) >= a:
                        wx = 0.0
                    ww = wy * wx
                    acc += float(src[yy0, xx0]) * ww
                    wsum += ww
            if wsum == 0.0:
                out[i] = fill
            else:
                out[i] = acc / wsum
        return out

    return _core(src2d, yy, xx, int(a), float(fill))


def _sample_lanczos_numpy(src2d: np.ndarray, yy: np.ndarray, xx: np.ndarray, a: int, fill: float, *, chunk: int = 20000):
    """
    纯 numpy 的 Lanczos-windowed sinc（比 numba 慢，但不依赖 numba）。
    - yy/xx: 1D float64
    """
    _require_numpy()
    src = src2d
    h, w = src.shape
    n = yy.shape[0]
    out = np.full(n, float(fill), dtype=np.float32)
    offs = np.arange(-a + 1, a + 1, dtype=np.int64)

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        y = yy[i0:i1].astype(np.float64, copy=False)
        x = xx[i0:i1].astype(np.float64, copy=False)
        ok0 = np.isfinite(y) & np.isfinite(x)
        if not np.any(ok0):
            continue

        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        ys = y0[:, None] + offs[None, :]
        xs = x0[:, None] + offs[None, :]

        my = (ys >= 0) & (ys < h)
        mx = (xs >= 0) & (xs < w)
        ys_i = np.clip(ys, 0, h - 1)
        xs_i = np.clip(xs, 0, w - 1)

        dy = y[:, None] - ys.astype(np.float64)
        dx = x[:, None] - xs.astype(np.float64)

        wy = np.sinc(dy) * np.sinc(dy / float(a))
        wx = np.sinc(dx) * np.sinc(dx / float(a))
        wy[np.abs(dy) >= a] = 0.0
        wx[np.abs(dx) >= a] = 0.0
        wy[~my] = 0.0
        wx[~mx] = 0.0

        vals = src[ys_i[:, :, None], xs_i[:, None, :]].astype(np.float32, copy=False)
        w2 = (wy[:, :, None] * wx[:, None, :]).astype(np.float64, copy=False)
        wsum = np.sum(w2, axis=(1, 2))
        acc = np.sum(vals.astype(np.float64) * w2, axis=(1, 2))
        good = ok0 & (wsum > 0.0)
        out[i0:i1][good] = (acc[good] / wsum[good]).astype(np.float32)

    return out


def _sample_lanczos(src2d: np.ndarray, yy: np.ndarray, xx: np.ndarray, a: int, fill: float):
    """优先用 numba；不可用则用 numpy 版本。"""
    _require_numpy()
    out = _sample_lanczos_numba(src2d, yy, xx, a, fill)
    if out is not None:
        return out
    return _sample_lanczos_numpy(src2d, yy, xx, a, fill)

def _write_geotiff(path: str, arr, gt, wkt: str):
    _require_numpy()
    gdal, _ = _try_import_gdal()
    if gdal is None:
        raise RuntimeError("写 GeoTIFF 需要 osgeo.gdal")
    driver = gdal.GetDriverByName("GTiff")
    if arr.ndim == 2:
        h, w = arr.shape
        ds = driver.Create(path, w, h, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
        ds.SetGeoTransform(gt)
        if wkt:
            ds.SetProjection(wkt)
        ds.GetRasterBand(1).WriteArray(arr.astype(np.float32, copy=False))
        ds.FlushCache()
        ds = None
        return
    if arr.ndim == 3:
        h, w, b = arr.shape
        ds = driver.Create(path, w, h, b, gdal.GDT_Float32, options=["COMPRESS=LZW"])
        ds.SetGeoTransform(gt)
        if wkt:
            ds.SetProjection(wkt)
        for i in range(b):
            ds.GetRasterBand(i + 1).WriteArray(arr[:, :, i].astype(np.float32, copy=False))
        ds.FlushCache()
        ds = None
        return
    raise ValueError("只支持 2D 或 3D 数组写入 GeoTIFF")


def _make_output_grid_from_dem(dem_gt, dem_shape, out_crs: str, res_m: float, *, lonlat_bbox=None):
    """
    基于 DEM 覆盖范围（或给定 lon/lat bbox）生成输出网格（LATLON 或 UTM/EPSG）
    - lonlat_bbox: (min_lon, max_lon, min_lat, max_lat)；若提供则不再用 DEM 四角推 bbox
    返回：out_gt, out_wkt, (out_h, out_w), out_crs_tag
    """
    _require_numpy()
    gdal, osr = _try_import_gdal()
    if osr is None:
        raise RuntimeError("生成 UTM 网格需要 osgeo.osr")

    nrows, ncols = dem_shape

    if lonlat_bbox is None:
        # DEM 四角（用像素中心近似）
        def corner(rc):
            r, c = rc
            c0 = c + 0.5
            r0 = r + 0.5
            lon = dem_gt[0] + dem_gt[1] * c0 + dem_gt[2] * r0
            lat = dem_gt[3] + dem_gt[4] * c0 + dem_gt[5] * r0
            return float(lon), float(lat)

        corners = [
            corner((0, 0)),
            corner((0, ncols - 1)),
            corner((nrows - 1, 0)),
            corner((nrows - 1, ncols - 1)),
        ]
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
    else:
        min_lon, max_lon, min_lat, max_lat = [float(x) for x in lonlat_bbox]
        corners = [
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
        ]

    out_crs_u = out_crs.upper()
    if out_crs_u in ["LATLON", "LL", "EPSG:4326", "WGS84"]:
        # 米 -> 度：用中心纬度近似
        lat0 = 0.5 * (min_lat + max_lat)
        deg_lat = res_m / 111320.0
        deg_lon = res_m / (111320.0 * max(1e-6, np.cos(np.deg2rad(lat0))))
        out_w = int(np.ceil((max_lon - min_lon) / deg_lon))
        out_h = int(np.ceil((max_lat - min_lat) / deg_lat))
        out_gt = (min_lon, deg_lon, 0.0, max_lat, 0.0, -deg_lat)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        return out_gt, srs.ExportToWkt(), (out_h, out_w), "LATLON"

    # UTM / EPSG
    if out_crs_u == "UTM":
        # 自动分带
        lon0 = 0.5 * (min_lon + max_lon)
        lat0 = 0.5 * (min_lat + max_lat)
        zone = int((lon0 + 180.0) / 6.0) + 1
        north = lat0 >= 0
        epsg = 32600 + zone if north else 32700 + zone
        out_crs_u = f"EPSG:{epsg}"

    if out_crs_u.startswith("EPSG:"):
        epsg = int(out_crs_u.split(":", 1)[1])
        srs_out = osr.SpatialReference()
        srs_out.ImportFromEPSG(epsg)
        srs_ll = osr.SpatialReference()
        srs_ll.ImportFromEPSG(4326)
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            srs_out.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            srs_ll.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tx = osr.CoordinateTransformation(srs_ll, srs_out)
        pts = tx.TransformPoints([(lon, lat) for lon, lat in corners])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        out_w = int(np.ceil((max_x - min_x) / res_m))
        out_h = int(np.ceil((max_y - min_y) / res_m))
        out_gt = (min_x, res_m, 0.0, max_y, 0.0, -res_m)
        return out_gt, srs_out.ExportToWkt(), (out_h, out_w), out_crs_u

    raise ValueError(f"不支持的 out_crs: {out_crs}")


def _compute_lonlat_bbox_from_mapping(
    dem_gt,
    dem_shape,
    range_grid: np.ndarray,
    az_grid: np.ndarray,
    sar_shape: Tuple[int, int],
    *,
    chunk_rows: int = 256,
):
    """
    仅用“能落到 SAR 有效像素范围内”的 DEM 点计算经纬度 bbox。
    - DEM 点的经纬度由 dem_gt + (row,col) 计算得到（像素中心）
    - 有效点条件：finite(rg/az) 且 0<=az<sar_h 且 0<=rg<sar_w
    目的：避免输出网格范围过大（例如与 DEM 覆盖一致，但 SAR 只覆盖其中一小部分）。
    """
    _require_numpy()
    nrows, ncols = dem_shape
    sar_h, sar_w = sar_shape

    # 预生成列向量（像素中心）
    cols = (np.arange(ncols, dtype=np.float64) + 0.5)[None, :]

    min_lon = float("inf")
    max_lon = float("-inf")
    min_lat = float("inf")
    max_lat = float("-inf")
    any_ok = False

    for r0 in range(0, nrows, int(chunk_rows)):
        r1 = min(nrows, r0 + int(chunk_rows))
        rows = (np.arange(r0, r1, dtype=np.float64) + 0.5)[:, None]

        rg = np.asarray(range_grid[r0:r1, :], dtype=np.float64)
        az = np.asarray(az_grid[r0:r1, :], dtype=np.float64)

        ok = np.isfinite(rg) & np.isfinite(az) & (az >= 0.0) & (az < float(sar_h)) & (rg >= 0.0) & (rg < float(sar_w))
        if not np.any(ok):
            continue

        lon = dem_gt[0] + dem_gt[1] * cols + dem_gt[2] * rows
        lat = dem_gt[3] + dem_gt[4] * cols + dem_gt[5] * rows

        lon_ok = lon[ok]
        lat_ok = lat[ok]
        if lon_ok.size == 0:
            continue

        any_ok = True
        lo0 = float(np.min(lon_ok))
        lo1 = float(np.max(lon_ok))
        la0 = float(np.min(lat_ok))
        la1 = float(np.max(lat_ok))
        if lo0 < min_lon:
            min_lon = lo0
        if lo1 > max_lon:
            max_lon = lo1
        if la0 < min_lat:
            min_lat = la0
        if la1 > max_lat:
            max_lat = la1

    if not any_ok:
        return None
    return (min_lon, max_lon, min_lat, max_lat)


def warp_sar_to_grid(
    sar,
    is_complex: bool,
    dem_gt,
    dem_shape,
    range_grid,
    az_grid,
    sar_shape,
    out_crs: str,
    res_m: float,
    interp: str,
    sinc_a: int,
    block_rows: int,
    extent: str = "sar",
    extent_chunk_rows: int = 256,
):
    _require_numpy()
    gdal, osr = _try_import_gdal()
    if osr is None:
        raise RuntimeError("需要 osgeo.osr")

    lonlat_bbox = None
    if str(extent).lower() == "sar":
        lonlat_bbox = _compute_lonlat_bbox_from_mapping(
            dem_gt,
            dem_shape,
            range_grid,
            az_grid,
            sar_shape,
            chunk_rows=int(extent_chunk_rows),
        )
        if lonlat_bbox is not None:
            print(
                "输出范围(bbox,由映射表裁剪): "
                f"lon[{lonlat_bbox[0]:.6f},{lonlat_bbox[1]:.6f}]  lat[{lonlat_bbox[2]:.6f},{lonlat_bbox[3]:.6f}]"
            )
        else:
            print("警告：未找到落在 SAR 有效范围内的 DEM 点，输出范围将退回 DEM 四角范围。")

    out_gt, out_wkt, (out_h, out_w), out_crs_tag = _make_output_grid_from_dem(
        dem_gt,
        dem_shape,
        out_crs,
        res_m,
        lonlat_bbox=lonlat_bbox,
    )

    # 输出数组
    if is_complex:
        out_re = np.zeros((out_h, out_w), dtype=np.float32)
        out_im = np.zeros((out_h, out_w), dtype=np.float32)
    elif sar.ndim == 3:
        out = np.zeros((out_h, out_w, sar.shape[2]), dtype=np.float32)
    else:
        out = np.zeros((out_h, out_w), dtype=np.float32)

    # UTM -> LL 需要变换
    need_utm_to_ll = out_crs_tag != "LATLON"
    if need_utm_to_ll:
        srs_out = osr.SpatialReference()
        if out_crs_tag.startswith("EPSG:"):
            srs_out.ImportFromEPSG(int(out_crs_tag.split(":", 1)[1]))
        else:
            raise ValueError("内部错误：UTM CRS tag 非 EPSG")
        srs_ll = osr.SpatialReference()
        srs_ll.ImportFromEPSG(4326)
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            srs_out.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            srs_ll.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tx = osr.CoordinateTransformation(srs_out, srs_ll)

    # 准备 SAR sampling
    sar_h, sar_w = sar_shape
    if is_complex:
        if np.iscomplexobj(sar):
            # 单波段 complex GeoTIFF/.npy/HDF:slc 走这里
            sar_re = np.real(sar).astype(np.float32, copy=False)
            sar_im = np.imag(sar).astype(np.float32, copy=False)
        elif sar.ndim == 3 and sar.shape[2] >= 2:
            # 兼容少数情况下传入 (H, W, 2) 的复数表示
            sar_re = np.asarray(sar[:, :, 0], dtype=np.float32)
            sar_im = np.asarray(sar[:, :, 1], dtype=np.float32)
        else:
            raise ValueError(f"is_complex=True 但 SAR 数组格式不支持: shape={sar.shape}, dtype={sar.dtype}")
    elif sar.ndim == 2:
        sar_re = sar.astype(np.float32, copy=False)
        sar_im = None
    else:
        sar_re = None
        sar_im = None

    interp_u = interp.lower()
    use_sinc = (interp_u == "sinc") and is_complex
    if use_sinc and _try_import_numba()[0] is None:
        print("提示：未检测到 numba，sinc 将使用纯 numpy 实现（会更慢）。")

    for r0 in range(0, out_h, block_rows):
        r1 = min(out_h, r0 + block_rows)
        rr = np.arange(r0, r1, dtype=np.float64) + 0.5
        cc = np.arange(0, out_w, dtype=np.float64) + 0.5
        ccv, rrv = np.meshgrid(cc, rr, indexing="xy")

        # 输出网格坐标
        xs = out_gt[0] + out_gt[1] * ccv + out_gt[2] * rrv
        ys = out_gt[3] + out_gt[4] * ccv + out_gt[5] * rrv

        if need_utm_to_ll:
            pts = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).tolist()
            out_pts = tx.TransformPoints(pts)
            lon = np.array([p[0] for p in out_pts], dtype=np.float64).reshape(xs.shape)
            lat = np.array([p[1] for p in out_pts], dtype=np.float64).reshape(xs.shape)
        else:
            lon = xs
            lat = ys

        # (lon,lat) -> DEM rc
        dem_r, dem_c = lonlat_to_dem_rc(dem_gt, lon, lat)
        dem_r = np.asarray(dem_r, dtype=np.float64)
        dem_c = np.asarray(dem_c, dtype=np.float64)

        # DEM rc -> SAR az/rg
        sar_rg = _bilinear(range_grid, dem_r, dem_c, fill=np.nan)
        sar_az = _bilinear(az_grid, dem_r, dem_c, fill=np.nan)

        # SAR sampling coords (y=az, x=rg)
        yy = sar_az.reshape(-1)
        xx = sar_rg.reshape(-1)

        if is_complex:
            if use_sinc:
                sampled_re = _sample_lanczos(sar_re, yy, xx, sinc_a, 0.0)
                sampled_im = _sample_lanczos(sar_im, yy, xx, sinc_a, 0.0)
            else:
                sampled_re = _sample_bilinear_src(sar_re, yy, xx, fill=0.0)
                sampled_im = _sample_bilinear_src(sar_im, yy, xx, fill=0.0)
            out_re[r0:r1, :] = sampled_re.reshape((r1 - r0, out_w))
            out_im[r0:r1, :] = sampled_im.reshape((r1 - r0, out_w))
        else:
            if sar.ndim == 3:
                for b in range(sar.shape[2]):
                    out_b = _sample_bilinear_src(sar[:, :, b].astype(np.float32, copy=False), yy, xx, fill=0.0)
                    out[r0:r1, :, b] = out_b.reshape((r1 - r0, out_w))
            else:
                out_block = _sample_bilinear_src(sar_re, yy, xx, fill=0.0)
                out[r0:r1, :] = out_block.reshape((r1 - r0, out_w))

    if is_complex:
        out_arr = np.stack([out_re, out_im], axis=-1)
    else:
        out_arr = out
    return out_arr, out_gt, out_wkt, out_crs_tag


def export_hdf_products(geosar_hdf: str, out_dir: str, prefix: str):
    _require_numpy()
    gdal, osr = _try_import_gdal()
    if gdal is None:
        raise RuntimeError("导出 GeoTIFF 需要 osgeo.gdal")
    os.makedirs(out_dir, exist_ok=True)

    h5py = _require_h5py()
    with h5py.File(geosar_hdf, "r") as f:
        dem_h = np.asarray(f["dem_h"][:], dtype=np.float32)
        dem_gt = tuple(np.asarray(f["dem_geotransform"][:], dtype=np.float64).tolist())
        dem_wkt = ""
        # 默认把 DEM 当作 EPSG:4326（如果你的 DEM 不是经纬度，请自行修改）
        if osr is not None:
            s = osr.SpatialReference()
            s.ImportFromEPSG(4326)
            dem_wkt = s.ExportToWkt()

        sar_dem = np.asarray(f["sar_dem"][:], dtype=np.float32) if "sar_dem" in f else None
        slc_re = np.asarray(f["slc_real"][:], dtype=np.float32) if "slc_real" in f else None
        slc_im = np.asarray(f["slc_imag"][:], dtype=np.float32) if "slc_imag" in f else None
        slc_u_re = np.asarray(f["slc_update_real"][:], dtype=np.float32) if "slc_update_real" in f else None
        slc_u_im = np.asarray(f["slc_update_imag"][:], dtype=np.float32) if "slc_update_imag" in f else None
        sim_re = np.asarray(f["sim_slc_real"][:], dtype=np.float32) if "sim_slc_real" in f else None
        sim_im = np.asarray(f["sim_slc_imag"][:], dtype=np.float32) if "sim_slc_imag" in f else None
        sim_amp = np.asarray(f["sim_amp"][:], dtype=np.float32) if "sim_amp" in f else None

    # DEM（带地理参考）
    _write_geotiff(os.path.join(out_dir, f"{prefix}_dem_h.tif"), dem_h, dem_gt, dem_wkt)

    # SAR 域产品（无地理参考：写 identity geotransform）
    id_gt = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if sar_dem is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_sar_dem.tif"), sar_dem, id_gt, "")
    if slc_re is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_slc_real.tif"), slc_re, id_gt, "")
    if slc_im is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_slc_imag.tif"), slc_im, id_gt, "")
    if slc_u_re is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_slc_update_real.tif"), slc_u_re, id_gt, "")
    if slc_u_im is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_slc_update_imag.tif"), slc_u_im, id_gt, "")
    if sim_re is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_sim_slc_real.tif"), sim_re, id_gt, "")
    if sim_im is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_sim_slc_imag.tif"), sim_im, id_gt, "")
    if sim_amp is not None:
        _write_geotiff(os.path.join(out_dir, f"{prefix}_sim_amp.tif"), sim_amp, id_gt, "")


def main():
    parser = argparse.ArgumentParser(
        prog="sar2ll.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "SAR -> 经纬度/UTM 地理编码（基于 geosar HDF 的 DEM<->SAR 几何映射）\n"
            "\n"
            "子命令：\n"
            "  warp   把 SAR 坐标系影像重采样到经纬度(LATLON)或 UTM 米制网格\n"
            "  export 从 geosar HDF 导出 SAR 域/DEM 域产品为 GeoTIFF\n"
            "\n"
            "重要约束（外部 SAR 数据时尤其要注意）：\n"
            "  1) warp 的 <sar_in> 必须与 <geosar_hdf> 里记录的 SAR 网格一致（尺寸/方向/像元定义）。\n"
            "  2) out_crs=UTM 时会自动计算投影带并打印最终 EPSG（例如 EPSG:32646）。\n"
            "\n"
            "插值说明：\n"
            "  - 实数：bilinear/nearest\n"
            "  - 复数：默认 sinc(Lanczos-windowed sinc)；无 numba 时自动使用纯 numpy 实现（更慢）。\n"
        ),
        epilog=(
            "示例：\n"
            "  # 外部实数 SAR -> UTM 10m\n"
            "  python3 sar2ll.py warp master_intensity.tif geosar.h5 out_utm.tif UTM 10 --interp auto\n"
            "\n"
            "  # 外部复数 SAR(两波段 real/imag) -> UTM 5m（复数自动 sinc）\n"
            "  python3 sar2ll.py warp slc_complex.tif geosar.h5 out_utm.tif UTM 5 --interp auto\n"
            "\n"
            "  # 从 HDF 直接读 slc/slc_update/sar_dem\n"
            "  python3 sar2ll.py warp HDF:slc geosar.h5 out_ll.tif LATLON 5 --interp auto\n"
            "  python3 sar2ll.py warp HDF:slc_update geosar.h5 out_ll.tif LATLON 5 --interp auto\n"
            "\n"
            "  # 导出 HDF 内产品为 tif\n"
            "  python3 sar2ll.py export geosar.h5 out_dir --prefix geosar\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_warp = sub.add_parser(
        "warp",
        help="将 SAR 数据重采样到地理/UTM 网格",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_warp.add_argument(
        "sar_in",
        help=(
            "SAR 输入：\n"
            "  - GeoTIFF: 单波段实数，或两波段(real,imag)，或单波段 complex(GDT_CFloat32/64)\n"
            "  - .npy: 实数或复数数组\n"
            "  - HDF:<dataset>: 从 geosar_hdf 中读取（例如 HDF:slc, HDF:slc_update, HDF:sar_dem）\n"
        ),
    )
    p_warp.add_argument("geosar_hdf", help="geosar.py 输出的 HDF5（提供几何映射与 DEM 覆盖范围）")
    p_warp.add_argument("out_tif", help="输出 GeoTIFF 路径")
    p_warp.add_argument(
        "out_crs",
        help=(
            "输出 CRS：\n"
            "  - LATLON: 输出经纬度网格（等价 EPSG:4326）\n"
            "  - UTM: 自动计算投影带（EPSG:326xx/327xx）\n"
            "  - EPSG:326XX / EPSG:327XX: 指定 UTM 投影带（北/南半球）\n"
        ),
    )
    p_warp.add_argument(
        "res_m",
        type=float,
        help=(
            "输出分辨率（单位：米）\n"
            "  - UTM/EPSG:326xx/327xx：单位=米\n"
            "  - LATLON：内部会按场景中心纬度把“米”近似换算成“度”（便于统一用米指定分辨率）\n"
        ),
    )
    p_warp.add_argument(
        "--interp",
        default="auto",
        choices=["auto", "nearest", "bilinear", "sinc"],
        help=(
            "插值方法：\n"
            "  - auto: 实数用 bilinear；复数用 sinc（推荐）\n"
            "  - nearest/bilinear: 实数插值\n"
            "  - sinc: 复数插值（Lanczos-windowed sinc；无 numba 时使用纯 numpy 实现，速度较慢）\n"
        ),
    )
    p_warp.add_argument("--sinc-radius", type=int, default=4, help="Lanczos sinc 半径 a（核大小约 2a；a 越大越慢）")
    p_warp.add_argument("--block-rows", type=int, default=256, help="按行分块处理的块高（像素；越大越快但更占内存）")
    p_warp.add_argument(
        "--extent",
        default="sar",
        choices=["sar", "dem"],
        help=(
            "输出范围（bbox）决定方式：\n"
            "  - sar: 用 range_pixel/azimuth_pixel 选出落在 SAR 有效像素范围内的 DEM 点，并取其经纬度 min/max（推荐，避免范围过大）\n"
            "  - dem: 直接使用 DEM 四角范围（可能会比 SAR 覆盖大很多）\n"
        ),
    )
    p_warp.add_argument("--extent-chunk-rows", type=int, default=256, help="extent=sar 时扫描 DEM 的行分块大小（像素）")

    p_exp = sub.add_parser(
        "export",
        help="从 geosar HDF 导出 sar_dem/slc_real/slc_imag/dem_h 为 GeoTIFF",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_exp.add_argument("geosar_hdf", help="geosar.py 输出的 HDF5")
    p_exp.add_argument("out_dir", help="输出目录")
    p_exp.add_argument("--prefix", default="geosar", help="输出文件名前缀")

    args = parser.parse_args()

    _require_numpy()
    if args.cmd == "export":
        export_hdf_products(args.geosar_hdf, args.out_dir, args.prefix)
        print("导出完成。")
        return

    dem_gt, dem_shape, range_grid, az_grid, sar_shape = _load_hdf_mapping(args.geosar_hdf)
    sar, is_complex = _load_sar_input(args.sar_in, geosar_hdf=args.geosar_hdf)
    interp = args.interp
    if interp == "auto":
        interp = "sinc" if is_complex else "bilinear"

    # 外部 SAR 输入必须与 HDF 的 SAR 尺寸一致，否则映射会错位
    h0, w0 = sar_shape
    if sar.ndim == 2:
        sh = sar.shape
        if tuple(sh) != tuple((h0, w0)):
            raise SystemExit(f"SAR 输入尺寸 {sh} 与 HDF 中 SAR 尺寸 {sar_shape} 不一致，无法正确转换。")
    elif sar.ndim == 3:
        sh = sar.shape[:2]
        if tuple(sh) != tuple((h0, w0)):
            raise SystemExit(f"SAR 输入尺寸 {sar.shape} 与 HDF 中 SAR 尺寸 {sar_shape} 不一致，无法正确转换。")

    out_arr, out_gt, out_wkt, out_crs_tag = warp_sar_to_grid(
        sar,
        is_complex,
        dem_gt,
        dem_shape,
        range_grid,
        az_grid,
        sar_shape,
        args.out_crs,
        float(args.res_m),
        interp,
        int(args.sinc_radius),
        int(args.block_rows),
        extent=args.extent,
        extent_chunk_rows=int(args.extent_chunk_rows),
    )
    if args.out_crs.upper() == "UTM":
        print(f"UTM 自动分带结果: {out_crs_tag}")

    # 写 GeoTIFF
    _write_geotiff(args.out_tif, out_arr, out_gt, out_wkt)
    print(f"完成: {args.out_tif}")


if __name__ == "__main__":
    main()
