#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ll2sar.py - 经纬度/UTM/投影栅格 -> SAR 栅格（以及点查询：地理坐标 -> SAR 像素坐标）

核心思路（避免 KDTree / griddata）：
- geosar.py 的 HDF 输出里通常会保存：
  1) DEM 网格每个像素对应的 (range_pixel, azimuth_pixel) 映射表（DEM->SAR，geo2rdr）
  2) SAR 网格每个像素对应的 (lat, lon) 网格（SAR->LL，rdr2geo 的结果）

本脚本提供两个能力：
1) pixels：给定一个点 (lat,lon) 或 UTM(E,N)，输出对应的 SAR 像素坐标 (az, rg)（连续值）。
   该功能使用 DEM->SAR 映射表，局部双线性近似。
2) warp：给定一幅“有地理参考”的 GeoTIFF（经纬度或任意投影，例如 UTM），把它重采样到 SAR 网格。
   该功能使用 SAR->LL 的 lat/lon 网格，对输入影像做反查采样（gather），复数默认用 sinc（Lanczos）。

注意：
- 该映射来自 DEM->SAR（geo2rdr），对叠置/阴影区域并非一一对应；点转换是“局部近似”。
- pixels 模式默认把 DEM geotransform 的坐标系当作经纬度度（EPSG:4326）。
  如果你的 DEM 不是经纬度，需要先重投影或自行扩展。
- warp 模式依赖 HDF 内存在 SAR 尺寸一致的 lat/lon 网格（例如 lat_grid/lon_grid 或 sar_lat/sar_lon 等）。
  若你的 HDF 没有保存该网格，请先在 geosar.py/dem2sar_full.py 里开启输出。
"""

import argparse
import os
from typing import Tuple, Optional, List, Dict, Any

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
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少依赖：h5py。请在你的工作环境中安装后再运行。") from e


def _load_geosar_mapping(hdf_path: str):
    _require_numpy()
    h5py = _require_h5py()
    with h5py.File(hdf_path, "r") as f:
        dem_gt = tuple(np.asarray(f["dem_geotransform"][:], dtype=np.float64).tolist())
        dem_shape = tuple(np.asarray(f["dem_shape"][:], dtype=np.int64).tolist())
        nrows, ncols = dem_shape

        r = f["range_pixel"][:]
        a = f["azimuth_pixel"][:]
        if r.ndim == 1:
            if r.size != nrows * ncols:
                raise ValueError(f"range_pixel size={r.size} 与 dem_shape={dem_shape} 不匹配")
            r = r.reshape((nrows, ncols))
        if a.ndim == 1:
            if a.size != nrows * ncols:
                raise ValueError(f"azimuth_pixel size={a.size} 与 dem_shape={dem_shape} 不匹配")
            a = a.reshape((nrows, ncols))

        # SAR 尺寸：优先用 YAML 读到的真实尺寸；否则用 geosar 计算结果；再否则用映射表估计
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
            sar_az = int(np.nanmax(a)) + 1
        if sar_rg is None:
            sar_rg = int(np.nanmax(r)) + 1

    return dem_gt, dem_shape, r.astype(np.float32, copy=False), a.astype(np.float32, copy=False), (sar_az, sar_rg)


def _inv_geo_transform(gt: Tuple[float, float, float, float, float, float]):
    """
    反解 geotransform（解线性方程得到 (col+0.5, row+0.5)）
    x = gt0 + gt1*c + gt2*r
    y = gt3 + gt4*c + gt5*r
    """
    gt0, gt1, gt2, gt3, gt4, gt5 = gt
    det = gt1 * gt5 - gt2 * gt4
    if abs(det) < 1e-20:
        raise ValueError("DEM geotransform 不可逆（det≈0）")
    inv = (gt5 / det, -gt2 / det, -gt4 / det, gt1 / det)  # (ic11, ic12, ic21, ic22)
    return gt0, gt3, inv


def _xy_to_rc(gt, x, y):
    """(x,y) -> (row,col) 连续像素坐标（像素中心体系）"""
    _require_numpy()
    gt0, gt3, inv = _inv_geo_transform(gt)
    ic11, ic12, ic21, ic22 = inv
    dx = np.asarray(x, dtype=np.float64) - gt0
    dy = np.asarray(y, dtype=np.float64) - gt3
    c0 = ic11 * dx + ic12 * dy
    r0 = ic21 * dx + ic22 * dy
    # c0/r0 对应 (col+0.5, row+0.5)
    col = c0 - 0.5
    row = r0 - 0.5
    return row, col


def lonlat_to_dem_rc(gt, lon, lat):
    """(lon,lat) -> (row,col) 连续像素坐标（像素中心体系）"""
    return _xy_to_rc(gt, lon, lat)


def dem_rc_to_lonlat(gt, row, col):
    """(row,col) 连续像素坐标（像素中心体系）-> (lon,lat)"""
    _require_numpy()
    gt0, gt1, gt2, gt3, gt4, gt5 = gt
    c0 = np.asarray(col, dtype=np.float64) + 0.5
    r0 = np.asarray(row, dtype=np.float64) + 0.5
    lon = gt0 + gt1 * c0 + gt2 * r0
    lat = gt3 + gt4 * c0 + gt5 * r0
    return lon, lat


def bilinear_sample(grid: np.ndarray, row, col, *, fill=float("nan")):
    """对 2D grid 在连续 (row,col) 上做双线性采样；返回 float32/float64"""
    _require_numpy()
    grid = np.asarray(grid)
    nrows, ncols = grid.shape
    r = np.asarray(row, dtype=np.float64)
    c = np.asarray(col, dtype=np.float64)

    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1

    # 有效范围要求落在 cell 内：r0 in [0,nrows-2], c0 in [0,ncols-2]
    ok = (r0 >= 0) & (r1 < nrows) & (c0 >= 0) & (c1 < ncols) & np.isfinite(r) & np.isfinite(c)
    out = np.full(np.broadcast(r, c).shape, fill, dtype=np.float64)
    if not np.any(ok):
        return out

    rr = r[ok] - r0[ok]
    cc = c[ok] - c0[ok]

    v00 = grid[r0[ok], c0[ok]].astype(np.float64, copy=False)
    v01 = grid[r0[ok], c1[ok]].astype(np.float64, copy=False)
    v10 = grid[r1[ok], c0[ok]].astype(np.float64, copy=False)
    v11 = grid[r1[ok], c1[ok]].astype(np.float64, copy=False)

    out_ok = (
        v00 * (1 - rr) * (1 - cc)
        + v01 * (1 - rr) * cc
        + v10 * rr * (1 - cc)
        + v11 * rr * cc
    )
    out[ok] = out_ok
    return out


def ll_to_sar_pixels(dem_gt, range_grid, az_grid, lat, lon):
    """(lat,lon) -> (az_pix, rg_pix)"""
    _require_numpy()
    row, col = lonlat_to_dem_rc(dem_gt, lon, lat)
    rg = bilinear_sample(range_grid, row, col, fill=np.nan)
    az = bilinear_sample(az_grid, row, col, fill=np.nan)
    return az, rg


def _try_utm_to_ll(lat_or_northing, lon_or_easting, *, zone: int, hemisphere: str):
    """
    UTM (northing,easting) -> (lat,lon)；依赖 GDAL/OSR。
    这里输入命名沿用旧脚本习惯：lat_array=Northing, lon_array=Easting。
    """
    try:
        from osgeo import osr
    except Exception as e:
        raise RuntimeError("需要 osgeo 才能做 UTM<->LL 转换") from e
    _require_numpy()

    utm = osr.SpatialReference()
    utm.SetUTM(int(zone), hemisphere.upper() == "N")
    wgs84 = osr.SpatialReference()
    wgs84.SetWellKnownGeogCS("WGS84")
    # GDAL 3+ 可能采用 EPSG 轴顺序（lat,lon）；这里强制传统 GIS 顺序（lon,lat）
    if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tx = osr.CoordinateTransformation(utm, wgs84)

    n = np.asarray(lat_or_northing, dtype=np.float64).reshape(-1)
    e = np.asarray(lon_or_easting, dtype=np.float64).reshape(-1)
    pts = list(zip(e.tolist(), n.tolist()))
    out = tx.TransformPoints(pts)
    lon = np.array([p[0] for p in out], dtype=np.float64).reshape(np.asarray(lat_or_northing).shape)
    lat = np.array([p[1] for p in out], dtype=np.float64).reshape(np.asarray(lat_or_northing).shape)
    return lat, lon


def _auto_utm_zone_from_dem(dem_gt, dem_shape):
    """
    根据 DEM 覆盖范围的中心经纬度估计 UTM 分带与半球。
    说明：
    - 仅当用户未显式提供 --utm-zone 时使用。
    - 对于极端情况（跨带/跨赤道）可能不准确；此时建议手动指定 --utm-zone/--hemisphere。
    """
    nrows, ncols = dem_shape

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
    lons = [p[0] for p in corners]
    lats = [p[1] for p in corners]
    lon0 = 0.5 * (min(lons) + max(lons))
    lat0 = 0.5 * (min(lats) + max(lats))
    zone = int((lon0 + 180.0) / 6.0) + 1
    zone = max(1, min(60, zone))
    hemi = "N" if lat0 >= 0 else "S"
    return zone, hemi


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


def _walk_hdf_datasets(g: Any, prefix: str = "") -> Dict[str, Any]:
    h5py = _require_h5py()
    out: Dict[str, Any] = {}
    for k, v in g.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Dataset):
            out[p] = v
        elif isinstance(v, h5py.Group):
            out.update(_walk_hdf_datasets(v, p))
    return out


def _check_reshape_like(ds: Any, shape2: Tuple[int, int]) -> None:
    """只检查 dataset 的 shape/size 是否能视作 (shape2[0],shape2[1])，避免把整张数据读入内存。"""
    if len(ds.shape) == 2 and tuple(ds.shape) == tuple(shape2):
        return
    if len(ds.shape) == 1 and int(ds.size) == int(shape2[0] * shape2[1]):
        return
    raise ValueError(f"dataset shape={ds.shape} size={ds.size} 无法视作 {shape2}")


def _find_sar_latlon_in_hdf(hdf_path: str, sar_shape: Tuple[int, int], lat_path: Optional[str], lon_path: Optional[str]):
    """
    在 HDF 中定位 SAR 网格下的 lat/lon（shape 与 sar_shape 一致）
    - 若用户显式提供 --lat-ds/--lon-ds，则直接使用
    - 否则自动在常见命名中查找
    返回： (lat_ds_path, lon_ds_path)
    """
    _require_numpy()
    h5py = _require_h5py()
    with h5py.File(hdf_path, "r") as f:
        dsets = _walk_hdf_datasets(f)
        if lat_path and lon_path:
            if lat_path not in dsets or lon_path not in dsets:
                raise ValueError(f"指定的 lat/lon dataset 不存在: {lat_path}, {lon_path}")
            # 只做 shape 兼容性检查（允许 1D flatten）
            _check_reshape_like(dsets[lat_path], sar_shape)
            _check_reshape_like(dsets[lon_path], sar_shape)
            return lat_path, lon_path

        # 常见命名（优先级从高到低）
        candidates = [
            ("sar_lat", "sar_lon"),
            ("sar_lat_grid", "sar_lon_grid"),
            ("lat_grid", "lon_grid"),
            ("geo_lat", "geo_lon"),
            ("lat", "lon"),
        ]

        # 支持 dataset 在任意 group 下（用末尾名匹配）
        def by_suffix(suf: str) -> List[str]:
            return [p for p in dsets.keys() if p.split("/")[-1] == suf]

        for lat_name, lon_name in candidates:
            lat_ps = by_suffix(lat_name)
            lon_ps = by_suffix(lon_name)
            for lp in lat_ps:
                for op in lon_ps:
                    try:
                        _check_reshape_like(dsets[lp], sar_shape)
                        _check_reshape_like(dsets[op], sar_shape)
                        return lp, op
                    except Exception:
                        continue

    raise ValueError(
        "HDF 中未找到与 SAR 尺寸一致的 lat/lon 网格。\n"
        "可尝试：\n"
        "1) 在生成 HDF 时保存 sar_lat/sar_lon 或 lat_grid/lon_grid；\n"
        "2) 或在本脚本中显式传入 --lat-ds/--lon-ds 指定 dataset 路径。"
    )


def _read_tif_as_array(path: str):
    """
    返回 (arr, gt, wkt, nodata)
    - 1 band -> (H,W) float32
    - 2 band -> complex64（real+imag）
    - >2 band -> (H,W,B) float32
    """
    gdal, _ = _try_import_gdal()
    if gdal is None:
        raise RuntimeError("warp 需要 osgeo.gdal（用于读取/写入 GeoTIFF）")
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开 GeoTIFF: {path}")
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection() or ""
    bands = ds.RasterCount
    nodata = ds.GetRasterBand(1).GetNoDataValue() if bands >= 1 else None
    if bands == 1:
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        return arr, gt, wkt, nodata
    if bands == 2:
        re = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        im = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        return (re + 1j * im).astype(np.complex64, copy=False), gt, wkt, nodata
    stack = []
    for i in range(1, bands + 1):
        stack.append(ds.GetRasterBand(i).ReadAsArray().astype(np.float32))
    arr = np.stack(stack, axis=-1)
    return arr, gt, wkt, nodata


def _create_sar_tif(out_tif: str, sar_shape: Tuple[int, int], bands: int):
    gdal, _ = _try_import_gdal()
    if gdal is None:
        raise RuntimeError("写 GeoTIFF 需要 osgeo.gdal")
    h, w = sar_shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_tif, w, h, bands, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    # SAR 坐标没有真实地理参考：写 identity geotransform + 空投影
    ds.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    ds.SetProjection("")
    return ds


def _sample_bilinear_src(src2d: np.ndarray, yy: np.ndarray, xx: np.ndarray, fill=0.0):
    """src[y,x] 双线性采样；yy/xx 为 1D float64"""
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


def _sample_nearest_src(src2d: np.ndarray, yy: np.ndarray, xx: np.ndarray, fill=0.0):
    """src[y,x] 最近邻采样；yy/xx 为 1D float64"""
    _require_numpy()
    h, w = src2d.shape
    y = np.rint(yy).astype(np.int64)
    x = np.rint(xx).astype(np.int64)
    ok = (y >= 0) & (x >= 0) & (y < h) & (x < w) & np.isfinite(yy) & np.isfinite(xx)
    out = np.full(yy.shape, fill, dtype=np.float32)
    if not np.any(ok):
        return out
    out[ok] = src2d[y[ok], x[ok]]
    return out


def _sample_lanczos_numba(src2d: np.ndarray, yy: np.ndarray, xx: np.ndarray, a: int, fill: float):
    """
    Lanczos-windowed sinc（需要 numba；否则返回 None）
    - yy/xx: 1D float64
    """
    njit, prange = _try_import_numba()
    if njit is None:
        return None

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
    offs = np.arange(-a + 1, a + 1, dtype=np.int64)  # length=2a

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        y = yy[i0:i1].astype(np.float64, copy=False)
        x = xx[i0:i1].astype(np.float64, copy=False)
        ok0 = np.isfinite(y) & np.isfinite(x)
        if not np.any(ok0):
            continue

        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        ys = y0[:, None] + offs[None, :]  # (m,2a)
        xs = x0[:, None] + offs[None, :]  # (m,2a)

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

        # (m,2a,2a)
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

def _transform_lonlat_to_src_xy(lon: np.ndarray, lat: np.ndarray, src_wkt: str):
    """
    把 WGS84 (lon,lat) 转到输入 GeoTIFF 的坐标系 (x,y)。
    - 若 src_wkt 为空，默认认为输入是 EPSG:4326，经纬度度
    - 依赖 osgeo.osr
    """
    if not src_wkt:
        return lon.astype(np.float64, copy=False), lat.astype(np.float64, copy=False)
    gdal, osr = _try_import_gdal()
    if osr is None:
        raise RuntimeError("需要 osgeo.osr 做坐标转换")
    s_ll = osr.SpatialReference()
    s_ll.ImportFromEPSG(4326)
    s_src = osr.SpatialReference()
    s_src.ImportFromWkt(src_wkt)
    if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        s_ll.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        s_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # 如果输入本身是经纬度，可直接返回
    try:
        if s_src.IsGeographic() and s_src.IsSameGeogCS(s_ll):
            return lon.astype(np.float64, copy=False), lat.astype(np.float64, copy=False)
    except Exception:
        pass
    tx = osr.CoordinateTransformation(s_ll, s_src)

    lon1 = lon.reshape(-1).astype(np.float64, copy=False)
    lat1 = lat.reshape(-1).astype(np.float64, copy=False)
    x = np.empty(lon1.shape[0], dtype=np.float64)
    y = np.empty(lon1.shape[0], dtype=np.float64)

    # 分块避免一次性创建超大 list
    chunk = 200000
    for i0 in range(0, lon1.shape[0], chunk):
        i1 = min(lon1.shape[0], i0 + chunk)
        pts = list(zip(lon1[i0:i1].tolist(), lat1[i0:i1].tolist()))
        out = tx.TransformPoints(pts)
        x[i0:i1] = [p[0] for p in out]
        y[i0:i1] = [p[1] for p in out]
    return x.reshape(lon.shape), y.reshape(lat.shape)


def warp_geo_to_sar(
    geo_tif: str,
    geosar_hdf: str,
    out_sar_tif: str,
    *,
    interp: str,
    sinc_a: int,
    block_lines: int,
    lat_ds: Optional[str],
    lon_ds: Optional[str],
):
    """
    读取输入 GeoTIFF（经纬度/UTM/任意投影），重采样到 SAR 网格并输出 GeoTIFF（identity geotransform）。
    复数（2 band）默认用 sinc；无 numba 时自动降级为 bilinear 并提示。
    """
    gdal, _ = _try_import_gdal()
    if gdal is None:
        raise RuntimeError("warp 需要 osgeo.gdal")
    _require_numpy()

    # 打开输入影像
    src_ds = gdal.Open(geo_tif, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"无法打开输入 GeoTIFF: {geo_tif}")
    src_gt = src_ds.GetGeoTransform()
    src_wkt = src_ds.GetProjection() or ""
    src_h = src_ds.RasterYSize
    src_w = src_ds.RasterXSize
    src_b = src_ds.RasterCount

    is_complex = (src_b == 2)
    if src_b == 0:
        raise ValueError("输入 GeoTIFF 不包含波段")

    # 读取 HDF 中的 SAR 尺寸与 lat/lon 网格路径
    dem_gt, dem_shape, range_grid, az_grid, sar_shape = _load_geosar_mapping(geosar_hdf)
    sar_h, sar_w = sar_shape
    lat_path, lon_path = _find_sar_latlon_in_hdf(geosar_hdf, sar_shape, lat_ds, lon_ds)

    # 创建输出 SAR tif
    out_bands = 2 if is_complex else src_b
    out_ds = _create_sar_tif(out_sar_tif, (sar_h, sar_w), out_bands)

    # auto：复数默认 sinc，实数默认 bilinear
    if interp.lower() == "auto":
        interp = "sinc" if is_complex else "bilinear"
    interp_u = interp.lower()
    if is_complex and interp_u != "sinc":
        # 用户显式指定也允许，但给出提示
        print(f"提示：输入为复数(2 band)，当前 interp={interp_u}。推荐使用 sinc。")

    use_sinc = is_complex and (interp_u == "sinc")
    if use_sinc and _try_import_numba()[0] is None:
        print("提示：未检测到 numba，sinc 将使用纯 numpy 实现（会更慢）。")

    # 逐块处理 SAR 行，避免一次性加载整张 lat/lon
    h5py = _require_h5py()
    with h5py.File(geosar_hdf, "r") as f:
        lat_ds_h = f[lat_path]
        lon_ds_h = f[lon_path]

        for az0 in range(0, sar_h, block_lines):
            az1 = min(sar_h, az0 + block_lines)

            # 支持 2D 或 1D(flatten) 存储
            if lat_ds_h.ndim == 2:
                lat = np.asarray(lat_ds_h[az0:az1, :], dtype=np.float64)
            else:
                lat = np.asarray(lat_ds_h[az0 * sar_w : az1 * sar_w], dtype=np.float64).reshape((az1 - az0, sar_w))
            if lon_ds_h.ndim == 2:
                lon = np.asarray(lon_ds_h[az0:az1, :], dtype=np.float64)
            else:
                lon = np.asarray(lon_ds_h[az0 * sar_w : az1 * sar_w], dtype=np.float64).reshape((az1 - az0, sar_w))

            # WGS84 -> src CRS
            x, y = _transform_lonlat_to_src_xy(lon, lat, src_wkt)

            # src CRS (x,y) -> src pixel (row,col)
            src_r, src_c = _xy_to_rc(src_gt, x, y)

            # 计算需要读取的 src window：
            # - 先过滤 NaN/Inf
            # - 再过滤“明显落在输入栅格范围外”的点
            #   否则极端但 finite 的 src_r/src_c 会把 rmin/rmax 拉到整幅 DEM，导致 IO/内存暴涨。
            valid = np.isfinite(src_r) & np.isfinite(src_c)
            if not np.any(valid):
                # 全无效：直接写 0
                for b in range(out_bands):
                    out_ds.GetRasterBand(b + 1).WriteArray(
                        np.zeros((az1 - az0, sar_w), dtype=np.float32),
                        xoff=0,
                        yoff=az0,
                    )
                continue

            # pad：nearest=0；bilinear=1；sinc=a
            if interp_u == "nearest":
                pad = 0
            elif use_sinc:
                pad = max(1, int(sinc_a))
            else:
                pad = 1

            inside = (
                valid
                & (src_r >= -float(pad)) & (src_r <= float(src_h - 1 + pad))
                & (src_c >= -float(pad)) & (src_c <= float(src_w - 1 + pad))
            )
            if not np.any(inside):
                # 全在范围外：直接写 0（避免读整幅输入）
                for b in range(out_bands):
                    out_ds.GetRasterBand(b + 1).WriteArray(
                        np.zeros((az1 - az0, sar_w), dtype=np.float32),
                        xoff=0,
                        yoff=az0,
                    )
                continue

            rv = src_r[inside]
            cv = src_c[inside]
            rmin = int(np.floor(np.min(rv))) - pad
            rmax = int(np.ceil(np.max(rv))) + pad
            cmin = int(np.floor(np.min(cv))) - pad
            cmax = int(np.ceil(np.max(cv))) + pad

            rmin = max(0, rmin)
            cmin = max(0, cmin)
            rmax = min(src_h - 1, rmax)
            cmax = min(src_w - 1, cmax)
            win_h = max(1, rmax - rmin + 1)
            win_w = max(1, cmax - cmin + 1)

            # 读 window
            # GDAL 的 ReadAsArray(xoff,yoff,xsize,ysize) 返回 (ysize,xsize) 或 (bands,ysize,xsize)
            src_block = src_ds.ReadAsArray(cmin, rmin, win_w, win_h)
            if src_b == 1:
                src_block = np.asarray(src_block, dtype=np.float32)
            else:
                src_block = np.asarray(src_block, dtype=np.float32)

            # 坐标转到 window 局部
            rr = (src_r - float(rmin)).reshape(-1)
            cc = (src_c - float(cmin)).reshape(-1)

            if is_complex:
                re = src_block[0] if src_block.ndim == 3 else src_block
                im = src_block[1] if src_block.ndim == 3 else None
                if im is None:
                    raise RuntimeError("内部错误：复数输入应为 2 band")
                if use_sinc:
                    out_re = _sample_lanczos(re, rr, cc, int(sinc_a), 0.0)
                    out_im = _sample_lanczos(im, rr, cc, int(sinc_a), 0.0)
                elif interp_u == "nearest":
                    out_re = _sample_nearest_src(re, rr, cc, fill=0.0)
                    out_im = _sample_nearest_src(im, rr, cc, fill=0.0)
                else:
                    out_re = _sample_bilinear_src(re, rr, cc, fill=0.0)
                    out_im = _sample_bilinear_src(im, rr, cc, fill=0.0)
                out_ds.GetRasterBand(1).WriteArray(out_re.reshape((az1 - az0, sar_w)), xoff=0, yoff=az0)
                out_ds.GetRasterBand(2).WriteArray(out_im.reshape((az1 - az0, sar_w)), xoff=0, yoff=az0)
            else:
                if src_b == 1:
                    if interp_u == "nearest":
                        out_val = _sample_nearest_src(src_block, rr, cc, fill=0.0)
                    else:
                        out_val = _sample_bilinear_src(src_block, rr, cc, fill=0.0)
                    out_ds.GetRasterBand(1).WriteArray(out_val.reshape((az1 - az0, sar_w)), xoff=0, yoff=az0)
                else:
                    # 多波段：逐 band 双线性
                    for bi in range(src_b):
                        band_arr = src_block[bi]
                        if interp_u == "nearest":
                            out_val = _sample_nearest_src(band_arr, rr, cc, fill=0.0)
                        else:
                            out_val = _sample_bilinear_src(band_arr, rr, cc, fill=0.0)
                        out_ds.GetRasterBand(bi + 1).WriteArray(out_val.reshape((az1 - az0, sar_w)), xoff=0, yoff=az0)

            out_ds.FlushCache()

    out_ds = None
    src_ds = None


def main():
    parser = argparse.ArgumentParser(
        prog="ll2sar.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "LL/UTM/投影栅格 -> SAR 网格（以及点查询：LL/UTM -> SAR 像素坐标）\n"
            "\n"
            "子命令：\n"
            "  pixels  点查询：输入一个 LL 或 UTM 坐标，输出连续的 SAR 像素坐标 (az,rg)\n"
            "  warp    栅格：把带地理参考的 GeoTIFF(经纬度/UTM/任意投影) 重采样到 SAR 网格\n"
            "\n"
            "pixels 的实现：\n"
            "  使用 HDF 内的 DEM->SAR 映射表（range_pixel/azimuth_pixel），在 DEM 上双线性插值得到 (az,rg)。\n"
            "\n"
            "warp 的实现：\n"
            "  使用 HDF 内的 SAR->LL lat/lon 网格（例如 lat_grid/lon_grid 或 sar_lat/sar_lon），\n"
            "  对输入 GeoTIFF 做反查采样（gather），输出尺寸严格等于 HDF 记录的 SAR 尺寸。\n"
            "\n"
            "重要说明：\n"
            "  - pixels 默认假设 DEM geotransform 坐标系为经纬度(EPSG:4326)。\n"
            "  - pixels 的 UTM 输入如果不填 --utm-zone，会用 DEM 覆盖范围中心经纬度自动估计分带。\n"
            "  - warp 需要 HDF 内存在 SAR 尺寸一致的 lat/lon 网格；找不到时可用 --lat-ds/--lon-ds 指定。\n"
        ),
        epilog=(
            "示例：\n"
            "  # 点查询：经纬度 -> SAR 像素\n"
            "  python3 ll2sar.py pixels geosar.h5 --lat 29.123456 --lon 102.123456\n"
            "\n"
            "  # 点查询：UTM -> SAR 像素（zone 不填则自动估计）\n"
            "  python3 ll2sar.py pixels geosar.h5 --utm-e 675000 --utm-n 3298000\n"
            "\n"
            "  # 栅格：UTM/经纬度 GeoTIFF -> SAR GeoTIFF（复数自动 sinc）\n"
            "  python3 ll2sar.py warp geo_utm.tif geosar.h5 out_sar.tif --interp auto\n"
            "\n"
            "  # 如果 HDF 里 lat/lon 网格不在默认命名下，显式指定 dataset 路径\n"
            "  python3 ll2sar.py warp geo_utm.tif geosar.h5 out_sar.tif --lat-ds /path/to/lat_grid --lon-ds /path/to/lon_grid\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pix = sub.add_parser(
        "pixels",
        help="点查询：LL/UTM -> SAR 像素坐标 (az,rg)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_pix.add_argument("geosar_hdf", help="geosar.py 输出的 HDF5 文件路径")
    p_pix.add_argument("--lat", type=float, help="查询点纬度（度）")
    p_pix.add_argument("--lon", type=float, help="查询点经度（度）")
    p_pix.add_argument("--utm-e", type=float, help="查询点 UTM Easting（米）")
    p_pix.add_argument("--utm-n", type=float, help="查询点 UTM Northing（米）")
    p_pix.add_argument("--utm-zone", type=int, help="UTM 分带编号（1-60）")
    p_pix.add_argument("--hemisphere", default=None, choices=["N", "S"], help="UTM 半球 (N/S)，不填则自动估计")

    p_warp = sub.add_parser(
        "warp",
        help="栅格转换：地理/UTM GeoTIFF -> SAR GeoTIFF",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_warp.add_argument(
        "geo_tif",
        help="输入 GeoTIFF（必须带 geotransform；CRS 可为经纬度或 UTM 等投影；会按 SAR 有效覆盖自动裁剪读取窗口）",
    )
    p_warp.add_argument("geosar_hdf", help="geosar.py 输出的 HDF5（需要包含 SAR lat/lon 网格）")
    p_warp.add_argument("out_sar_tif", help="输出 SAR GeoTIFF（SAR 坐标无真实地理参考，写入 identity geotransform）")
    p_warp.add_argument(
        "--interp",
        default="auto",
        choices=["auto", "nearest", "bilinear", "sinc"],
        help=(
            "插值方法：\n"
            "  - auto: 实数用 bilinear；复数用 sinc（推荐）\n"
            "  - nearest/bilinear: 实数插值\n"
            "  - sinc: 复数插值（Lanczos-windowed sinc；无 numba 时使用纯 numpy，实现较慢）\n"
        ),
    )
    p_warp.add_argument("--sinc-radius", type=int, default=4, help="Lanczos sinc 半径 a（核大小约 2a；a 越大越慢）")
    p_warp.add_argument("--block-lines", type=int, default=256, help="按 SAR 行分块处理的块高（像素；越大越快但更占内存）")
    p_warp.add_argument("--lat-ds", default=None, help="HDF 内 SAR 纬度网格 dataset 路径（可选，自动识别失败时使用）")
    p_warp.add_argument("--lon-ds", default=None, help="HDF 内 SAR 经度网格 dataset 路径（可选）")

    args = parser.parse_args()

    if args.cmd == "warp":
        warp_geo_to_sar(
            args.geo_tif,
            args.geosar_hdf,
            args.out_sar_tif,
            interp=args.interp,
            sinc_a=int(args.sinc_radius),
            block_lines=int(args.block_lines),
            lat_ds=args.lat_ds,
            lon_ds=args.lon_ds,
        )
        print(f"完成: {args.out_sar_tif}")
        return

    dem_gt, dem_shape, range_grid, az_grid, sar_shape = _load_geosar_mapping(args.geosar_hdf)
    sar_az, sar_rg = sar_shape
    print(f"DEM shape: {dem_shape}")
    print(f"SAR shape: (az={sar_az}, rg={sar_rg})")

    if (args.lat is None or args.lon is None) and (args.utm_e is None or args.utm_n is None):
        raise SystemExit("需要提供 --lat/--lon 或 --utm-e/--utm-n")

    if args.utm_e is not None or args.utm_n is not None:
        zone = args.utm_zone
        hemi = args.hemisphere
        if zone is None:
            zone, hemi0 = _auto_utm_zone_from_dem(dem_gt, dem_shape)
            # hemisphere 未显式指定时，也跟随自动估计
            if hemi is None:
                hemi = hemi0
            print(f"UTM 自动分带: zone={zone} hemisphere={hemi}")
        if hemi is None:
            hemi = "N"
        lat, lon = _try_utm_to_ll(args.utm_n, args.utm_e, zone=int(zone), hemisphere=str(hemi))
        lat = float(np.asarray(lat))
        lon = float(np.asarray(lon))
    else:
        lat, lon = float(args.lat), float(args.lon)

    az, rg = ll_to_sar_pixels(dem_gt, range_grid, az_grid, lat, lon)
    az = float(np.asarray(az))
    rg = float(np.asarray(rg))

    ok = np.isfinite(az) and np.isfinite(rg) and (0 <= az < sar_az) and (0 <= rg < sar_rg)
    print(f"Input LL: lat={lat:.8f} lon={lon:.8f}")
    print(f"SAR pixel: az={az:.3f} rg={rg:.3f}  in_bounds={ok}")


if __name__ == "__main__":
    main()
