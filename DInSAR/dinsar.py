#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DInSAR 一体化流程脚本（极简入口）

仅需输入：
1) master（可传文件名基名，如 master；或完整路径）
2) slave（可传文件名基名，如 slave；或完整路径）
3) 输出目录（--output-dir）

自动行为：
- 自动查找 master/slave 对应的 tiff/vrt 与 yaml
- 支持通过 --dem 指定 DEM
- 未指定时自动优先使用已有 DEM（dem_latlon.tif / dem.tif）
- 若仍无可用 DEM，则自动调用 mkdem 下载/生成
- multilook 需显式输入（不使用默认值）
- 默认解缠模式 snaphu defo，地理编码 CRS 为 UTM
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from osgeo import gdal, osr

# 允许直接 import 仓库内模块
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from interferometry.interferometry import (  # noqa: E402
    InterferogramFilter,
    TopographicPhaseRemoval,
    load_baseline_hdf,
    load_geosar_geometry_hdf,
)


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    cmd_show = " ".join(str(x) for x in cmd)
    print(f"\n[RUN] {cmd_show}")
    if cwd is not None:
        print(f"[CWD] {cwd}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _ensure_regular_file(path: Path) -> None:
    """
    若 path 为软链接，则将其就地替换为独立普通文件（复制链接目标内容）。
    用于确保 stage 内 YAML 可独立更新，不会回写到上游原始文件。
    """
    if not path.exists() and (not path.is_symlink()):
        raise FileNotFoundError(path)
    if not path.is_symlink():
        return
    src = path.resolve()
    path.unlink()
    shutil.copy2(src, path)


def _snapshot_dem_to_stage(dem_path: Path, dem_stage_dir: Path) -> Path:
    """
    在 02_dem 目录下保留“最终采用 DEM”的快照（软链接或拷贝）。
    返回快照路径。
    """
    src = dem_path.expanduser().resolve()
    dst = dem_stage_dir / f"dem_used{src.suffix.lower()}"
    _safe_symlink_or_copy(src, dst)
    return dst


def _read_sar_bbox_from_yaml(yaml_path: Path) -> Tuple[float, float, float, float]:
    """
    从 YAML corner_coordinates 读取 SAR 覆盖边界:
    返回 (min_lon, max_lon, min_lat, max_lat)。
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    corners = cfg.get("corner_coordinates", {}) or {}
    if not corners:
        raise ValueError(f"YAML 缺少 corner_coordinates: {yaml_path}")

    lons = []
    lats = []
    for key, val in corners.items():
        if not isinstance(val, dict) or ("lon" not in val) or ("lat" not in val):
            raise ValueError(f"corner_coordinates.{key} 缺少 lon/lat: {val!r}")
        lons.append(float(val["lon"]))
        lats.append(float(val["lat"]))

    return (min(lons), max(lons), min(lats), max(lats))


def _pixel_to_geo(gt: Tuple[float, float, float, float, float, float], px: float, py: float) -> Tuple[float, float]:
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return float(x), float(y)


def _axis_traditional(srs: osr.SpatialReference) -> None:
    # GDAL3 起默认遵循 EPSG 轴顺序，这里统一使用传统 GIS 顺序 lon/lat。
    if hasattr(srs, "SetAxisMappingStrategy") and hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)


def _get_dataset_bbox_wgs84(path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    计算栅格范围并转换到 WGS84，返回 (min_lon, max_lon, min_lat, max_lat)。
    无法获取地理参考时返回 None。
    """
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        return None

    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        ds = None
        return None

    w, h = ds.RasterXSize, ds.RasterYSize
    corners_xy = [
        _pixel_to_geo(gt, 0, 0),
        _pixel_to_geo(gt, w, 0),
        _pixel_to_geo(gt, 0, h),
        _pixel_to_geo(gt, w, h),
    ]

    src_wkt = ds.GetProjection()
    ds = None

    # 没有投影信息时无法可靠转换，返回 None 交由上层处理
    if not src_wkt:
        return None

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_wkt)
    _axis_traditional(src_srs)

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    _axis_traditional(dst_srs)

    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    lons: list[float] = []
    lats: list[float] = []
    for x, y in corners_xy:
        lon, lat, _ = ct.TransformPoint(x, y)
        lons.append(float(lon))
        lats.append(float(lat))

    return (min(lons), max(lons), min(lats), max(lats))


def _bbox_contains(
    outer: Tuple[float, float, float, float],
    inner: Tuple[float, float, float, float],
    eps: float = 1e-7,
) -> bool:
    omin_lon, omax_lon, omin_lat, omax_lat = outer
    imin_lon, imax_lon, imin_lat, imax_lat = inner
    return (
        omin_lon <= imin_lon + eps
        and omax_lon >= imax_lon - eps
        and omin_lat <= imin_lat + eps
        and omax_lat >= imax_lat - eps
    )


def _fmt_bbox(name: str, bbox: Tuple[float, float, float, float]) -> str:
    min_lon, max_lon, min_lat, max_lat = bbox
    return (
        f"{name}[lon:{min_lon:.6f}~{max_lon:.6f}, "
        f"lat:{min_lat:.6f}~{max_lat:.6f}]"
    )


def _ensure_dem_covers_sar(
    *,
    dem_path: Path,
    sar_bbox: Tuple[float, float, float, float],
) -> Tuple[bool, Optional[Tuple[float, float, float, float]], str]:
    dem_bbox = _get_dataset_bbox_wgs84(dem_path)
    if dem_bbox is None:
        return (
            False,
            None,
            f"无法读取 DEM 地理范围（缺少有效地理参考）: {dem_path}",
        )

    ok = _bbox_contains(dem_bbox, sar_bbox)
    if ok:
        return (
            True,
            dem_bbox,
            f"覆盖检查通过: {_fmt_bbox('DEM', dem_bbox)} 覆盖 {_fmt_bbox('SAR', sar_bbox)}",
        )
    return (
        False,
        dem_bbox,
        f"覆盖不足: {_fmt_bbox('DEM', dem_bbox)} 无法覆盖 {_fmt_bbox('SAR', sar_bbox)}",
    )


def _parse_multilook(spec: str) -> Tuple[int, int]:
    s = str(spec).strip().lower().replace(",", ":").replace("x", ":")
    parts = [p for p in s.split(":") if p]
    if len(parts) != 2:
        raise ValueError(f"multilook 格式错误: {spec}，应为 'az:rg'，例如 4:4")
    nalks = int(parts[0])
    nrlks = int(parts[1])
    if nalks <= 0 or nrlks <= 0:
        raise ValueError(f"multilook 必须为正整数: {spec}")
    return nalks, nrlks


def _auto_snaphu_tile_layout(height: int, width: int, max_pixels_per_tile: int) -> Tuple[int, int]:
    total_pixels = int(height) * int(width)
    if total_pixels <= int(max_pixels_per_tile):
        return 1, 1

    target = max(1, int(max_pixels_per_tile))
    ntiles = int(math.ceil(total_pixels / float(target)))
    aspect = float(height) / float(max(width, 1))
    rows = max(1, int(round(math.sqrt(ntiles * aspect))))
    cols = max(1, int(math.ceil(ntiles / float(rows))))
    while rows * cols < ntiles:
        rows += 1
    return rows, cols


def _normalize_snaphu_overlap(tile_size: int, overlap: int, min_recommended: int) -> int:
    if tile_size <= 2:
        return 0
    ov = int(overlap)
    ov = max(0, ov)
    ov = min(ov, tile_size - 1)
    if ov < min_recommended:
        ov = min(min_recommended, tile_size - 1)
    return ov


def _resolve_snaphu_runtime_params(shape: Tuple[int, int], args: argparse.Namespace) -> Dict[str, Any]:
    h, w = int(shape[0]), int(shape[1])
    manual_tile = (args.snaphu_tile_rows is not None) and (args.snaphu_tile_cols is not None)
    warnings: list[str] = []

    if manual_tile:
        tile_rows = int(args.snaphu_tile_rows)
        tile_cols = int(args.snaphu_tile_cols)
        tile_source = "manual"
    else:
        if bool(args.snaphu_disable_auto_tile):
            tile_rows, tile_cols = 1, 1
            tile_source = "auto_disabled"
        else:
            tile_rows, tile_cols = _auto_snaphu_tile_layout(
                height=h,
                width=w,
                max_pixels_per_tile=int(args.snaphu_auto_tile_max_pixels),
            )
            tile_source = "auto"

    use_tile = (tile_rows > 1) or (tile_cols > 1)
    if use_tile:
        tile_h = int(math.ceil(h / float(tile_rows)))
        tile_w = int(math.ceil(w / float(tile_cols)))
        tile_row_overlap = _normalize_snaphu_overlap(
            tile_size=tile_h,
            overlap=int(args.snaphu_tile_row_overlap),
            min_recommended=64 if tile_rows > 1 else 0,
        )
        tile_col_overlap = _normalize_snaphu_overlap(
            tile_size=tile_w,
            overlap=int(args.snaphu_tile_col_overlap),
            min_recommended=64 if tile_cols > 1 else 0,
        )
    else:
        tile_row_overlap = None
        tile_col_overlap = None

    if args.snaphu_nproc is not None:
        if use_tile:
            nproc = int(args.snaphu_nproc)
            nproc_source = "manual"
        else:
            nproc = None
            nproc_source = "ignored"
            warnings.append("已设置 --snaphu-nproc，但当前为单块解缠，nproc 不生效。")
    elif use_tile:
        cpu_total = max(1, int(os.cpu_count() or 1))
        reserve = max(0, int(args.snaphu_auto_reserve_cores))
        available = max(1, cpu_total - reserve)
        # 默认将并行上限限制为 CPU 核心数的 80%（向下取整，至少为 1）
        default_cap = max(1, int(math.floor(cpu_total * 0.8)))
        max_nproc_cap = int(args.snaphu_auto_max_nproc) if args.snaphu_auto_max_nproc is not None else default_cap
        available = min(available, max_nproc_cap)
        nproc = max(1, min(available, int(tile_rows) * int(tile_cols)))
        nproc_source = "auto"
    else:
        nproc = None
        nproc_source = "none"

    return {
        "use_tile": use_tile,
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "tile_row_overlap": tile_row_overlap,
        "tile_col_overlap": tile_col_overlap,
        "tile_source": tile_source,
        "nproc": nproc,
        "nproc_source": nproc_source,
        "warnings": warnings,
    }


def _resolve_image_path(identifier: str) -> Path:
    """
    支持以下输入：
    - 带后缀的完整路径：/path/master.tiff /path/master.vrt
    - 不带后缀的基名：master（自动查 master.tiff/.tif/.vrt）
    """
    p = Path(identifier).expanduser()
    cands = []
    if p.suffix:
        cands.append(p)
    else:
        cands.extend([p.with_suffix(".tiff"), p.with_suffix(".tif"), p.with_suffix(".vrt")])
    for c in cands:
        if c.exists():
            return c.resolve()
    cand_txt = ", ".join(str(x) for x in cands)
    raise FileNotFoundError(f"未找到影像文件: {identifier}（尝试: {cand_txt}）")


def _resolve_yaml_for_image(image_path: Path, identifier: str) -> Path:
    """
    自动匹配 YAML：
    - 优先同目录同 stem：<stem>.yaml/.yml
    - 若 identifier 无后缀，尝试 <identifier>.yaml/.yml
    """
    cands = [image_path.with_suffix(".yaml"), image_path.with_suffix(".yml")]
    ip = Path(identifier).expanduser()
    if not ip.suffix:
        cands.append((ip.parent / f"{ip.name}.yaml"))
        cands.append((ip.parent / f"{ip.name}.yml"))

    seen = set()
    for c in cands:
        cc = c.resolve()
        if cc in seen:
            continue
        seen.add(cc)
        if cc.exists():
            return cc
    raise FileNotFoundError(
        f"未找到 YAML: image={image_path}，尝试={', '.join(str(x) for x in cands)}"
    )


def _auto_find_dem(search_dirs: list[Path]) -> Optional[Path]:
    """
    自动查找现有 DEM。优先 dem_latlon，其次 dem。
    """
    priorities = [
        "dem_latlon.tif",
        "dem_latlon.vrt",
        "dem.tif",
        "dem.vrt",
    ]
    for d in search_dirs:
        dd = d.expanduser().resolve()
        for name in priorities:
            p = dd / name
            if p.exists():
                return p

    # 兜底：模糊匹配
    patterns = ["*dem_latlon*.tif", "*dem_latlon*.vrt", "*dem*.tif", "*dem*.vrt"]
    for d in search_dirs:
        dd = d.expanduser().resolve()
        for pat in patterns:
            hits = sorted(dd.glob(pat))
            if hits:
                return hits[0].resolve()
    return None


def _ensure_vrt(tif_path: Path) -> Path:
    vrt = tif_path.with_suffix(".vrt")
    if vrt.exists():
        return vrt
    _run(["gdal_translate", "-of", "VRT", str(tif_path), str(vrt)])
    return vrt


def _cleanup_legacy_overlay_pngs(out_dir: Path) -> int:
    """
    清理旧版遗留的相位叠加强度 PNG，避免与当前“纯相位”结果混淆。
    """
    removed = 0
    patterns = ("*_overlay_master.png", "*_overlay_amplitude.png")
    for pat in patterns:
        for p in sorted(out_dir.glob(pat)):
            if not p.is_file():
                continue
            try:
                p.unlink()
                removed += 1
            except Exception as e:
                print(f"警告: 清理旧 overlay PNG 失败: {p} ({e})")
    return removed


def _read_yaml_params(yaml_path: Path) -> Dict[str, float]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    radar = cfg.get("radar_parameters", {}) or {}

    def _pick(*keys: str, default: float) -> float:
        for k in keys:
            if k in radar:
                return float(radar[k])
        for k in keys:
            if k in cfg:
                return float(cfg[k])
        return float(default)

    out = {
        "range_spacing": _pick("range_spacing", "range_pixel_spacing", default=5.0),
        "azimuth_spacing": _pick("azimuth_spacing", default=5.0),
        "wavelength": _pick("wavelength", default=0.0555),
    }
    return out


def _read_complex_tiff(tif_path: Path) -> np.ndarray:
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开影像: {tif_path}")
    bands = ds.RasterCount
    if bands == 1:
        arr = ds.GetRasterBand(1).ReadAsArray()
        if np.iscomplexobj(arr):
            return np.asarray(arr, dtype=np.complex64)
        return np.asarray(arr, dtype=np.float32).astype(np.complex64)
    if bands >= 2:
        re = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        im = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        return (re + 1j * im).astype(np.complex64)
    raise RuntimeError(f"影像波段数异常: {tif_path}")


def _write_float_tiff(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), w, h, 1, gdal.GDT_Float32)
    if ds is None:
        raise RuntimeError(f"无法创建 GeoTIFF: {path}")
    ds.GetRasterBand(1).WriteArray(arr)
    ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    ds = None


def _create_raw_vrt(bin_file: Path, shape: Tuple[int, int], complex_data: bool) -> Path:
    vrt = bin_file.with_suffix(".vrt")
    width, height = shape[1], shape[0]
    if complex_data:
        content = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">
    <SourceFilename relativeToVRT="1">{bin_file.name}</SourceFilename>
    <ImageOffset>0</ImageOffset>
    <PixelOffset>8</PixelOffset>
    <LineOffset>{width * 8}</LineOffset>
    <ByteOrder>LSB</ByteOrder>
  </VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2" subClass="VRTRawRasterBand">
    <SourceFilename relativeToVRT="1">{bin_file.name}</SourceFilename>
    <ImageOffset>4</ImageOffset>
    <PixelOffset>8</PixelOffset>
    <LineOffset>{width * 8}</LineOffset>
    <ByteOrder>LSB</ByteOrder>
  </VRTRasterBand>
</VRTDataset>
"""
    else:
        content = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">
    <SourceFilename relativeToVRT="1">{bin_file.name}</SourceFilename>
    <ImageOffset>0</ImageOffset>
    <PixelOffset>4</PixelOffset>
    <LineOffset>{width * 4}</LineOffset>
    <ByteOrder>LSB</ByteOrder>
  </VRTRasterBand>
</VRTDataset>
"""
    vrt.write_text(content, encoding="utf-8")
    return vrt


def _bin_vrt_to_tif(bin_file: Path, out_tif: Path, shape: Tuple[int, int], complex_data: bool) -> Path:
    vrt = bin_file.with_suffix(".vrt")
    if not vrt.exists():
        vrt = _create_raw_vrt(bin_file, shape=shape, complex_data=complex_data)
    _run(["gdal_translate", "-of", "GTiff", str(vrt), str(out_tif)])
    return out_tif


def _load_coherence_mask(
    coherence_tif: Optional[Path],
    target_shape: Tuple[int, int],
    threshold: float = 0.3,
) -> Optional[np.ndarray]:
    if coherence_tif is None:
        return None
    if not coherence_tif.exists():
        print(f"警告: 相干图不存在，跳过掩膜: {coherence_tif}")
        return None
    ds = gdal.Open(str(coherence_tif), gdal.GA_ReadOnly)
    if ds is None:
        print(f"警告: 无法读取相干图，跳过掩膜: {coherence_tif}")
        return None
    coh = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    if coh is None:
        return None
    coh = np.asarray(coh, dtype=np.float32)
    if tuple(coh.shape) != tuple(target_shape):
        print(f"警告: 相干图尺寸不匹配，跳过掩膜: {coh.shape} vs {target_shape}")
        return None
    return (~np.isfinite(coh)) | (coh < float(threshold))


def _robust_percentile_limits(
    data: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> Tuple[Optional[float], Optional[float]]:
    """
    计算稳健显示区间，自动忽略空值区域：
    - 非有限值与 invalid_mask
    - 若 0 值占比过高（常见于空值填充），优先用非零样本计算分位数
    """
    arr = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(arr)
    if invalid_mask is not None:
        valid &= (~invalid_mask)

    vals = arr[valid]
    if vals.size == 0:
        return None, None

    zero = np.isclose(vals, 0.0, atol=1e-12)
    nz = vals[~zero]
    # 空值区域常被写为 0：若 0 占比很高且非零样本足够，则排除 0 再计算
    if (vals.size > 0) and (np.mean(zero) > 0.20) and (nz.size >= 1024):
        vals_used = nz
    else:
        vals_used = vals

    if vals_used.size == 0:
        vals_used = vals

    vmin, vmax = np.percentile(vals_used, [p_lo, p_hi])
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = float(np.nanmin(vals_used))
        vmax = float(np.nanmax(vals_used))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            return None, None
    return float(vmin), float(vmax)


def _phase_to_gamma_rgba(phase: np.ndarray, invalid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    GAMMA InSAR 常用循环色带近似：红->绿->蓝->红。
    返回 RGBA uint8，invalid_mask 位置 alpha=0。
    """
    p = (np.asarray(phase, dtype=np.float32) + np.pi) / (2.0 * np.pi)
    p = np.mod(p, 1.0)

    r = np.zeros_like(p, dtype=np.float32)
    g = np.zeros_like(p, dtype=np.float32)
    b = np.zeros_like(p, dtype=np.float32)

    m1 = p < (1.0 / 3.0)
    m2 = (p >= (1.0 / 3.0)) & (p < (2.0 / 3.0))
    m3 = p >= (2.0 / 3.0)

    r[m1] = 1.0 - p[m1] * 3.0
    g[m1] = p[m1] * 3.0
    b[m1] = 0.0

    r[m2] = 0.0
    g[m2] = 1.0 - (p[m2] - 1.0 / 3.0) * 3.0
    b[m2] = (p[m2] - 1.0 / 3.0) * 3.0

    r[m3] = (p[m3] - 2.0 / 3.0) * 3.0
    g[m3] = 0.0
    b[m3] = 1.0 - (p[m3] - 2.0 / 3.0) * 3.0

    rgba = np.zeros((phase.shape[0], phase.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255

    if invalid_mask is not None:
        rgba[invalid_mask, 3] = 0

    return rgba


def _save_complex_phase_products(
    ifg_tif: Path,
    pure_png: Path,
    coherence_tif: Optional[Path] = None,
    coherence_threshold: float = 0.3,
    phase_palette: str = "gamma",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"警告: matplotlib 不可用，跳过 PNG 预览: {pure_png}")
        return

    ds = gdal.Open(str(ifg_tif), gdal.GA_ReadOnly)
    if ds is None or ds.RasterCount < 2:
        print(f"警告: 干涉图读取失败或波段不足，跳过: {ifg_tif}")
        return
    re = ds.GetRasterBand(1).ReadAsArray()
    im = ds.GetRasterBand(2).ReadAsArray()
    ds = None
    if re is None or im is None:
        return
    phase = np.angle(re.astype(np.float32) + 1j * im.astype(np.float32))

    invalid = ~np.isfinite(phase)
    coh_mask = _load_coherence_mask(
        coherence_tif=coherence_tif,
        target_shape=phase.shape,
        threshold=coherence_threshold,
    )
    if coh_mask is not None:
        invalid |= coh_mask

    palette = str(phase_palette).strip().lower()
    if palette == "jet":
        norm = np.clip((phase + np.pi) / (2.0 * np.pi), 0.0, 1.0)
        rgba = (plt.get_cmap("jet")(norm) * 255.0).astype(np.uint8)
        rgba[invalid, 3] = 0
    else:
        rgba = _phase_to_gamma_rgba(phase, invalid_mask=invalid)

    # 纯相位图：输出为不透明 PNG，避免在 GIS/查看器中透出底图而被误判为“叠合”。
    pure_rgba = rgba.copy()
    pure_rgba[invalid, 0:3] = 0
    pure_rgba[:, :, 3] = 255
    plt.imsave(str(pure_png), pure_rgba)


def _save_preview_png(
    tif_path: Path,
    png_path: Path,
    mode: str,
    coherence_tif: Optional[Path] = None,
    coherence_threshold: float = 0.3,
    phase_palette: str = "gamma",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"警告: matplotlib 不可用，跳过 PNG 预览: {png_path}")
        return

    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        print(f"警告: 无法读取 GeoTIFF，跳过 PNG: {tif_path}")
        return
    bands = ds.RasterCount
    band1 = ds.GetRasterBand(1)
    nodata = band1.GetNoDataValue()
    arr = band1.ReadAsArray()
    if arr is None:
        ds = None
        return

    if mode == "complex_phase" and bands >= 2:
        re = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        im = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        phase = np.angle(re + 1j * im)
        invalid = ~np.isfinite(phase)
        coh_mask = _load_coherence_mask(
            coherence_tif=coherence_tif,
            target_shape=phase.shape,
            threshold=coherence_threshold,
        )
        if coh_mask is not None:
            invalid = invalid | coh_mask

        palette = str(phase_palette).strip().lower()
        if palette == "jet":
            norm = np.clip((phase + np.pi) / (2.0 * np.pi), 0.0, 1.0)
            rgba = (plt.get_cmap("jet")(norm) * 255.0).astype(np.uint8)
            rgba[invalid, 3] = 0
            plt.imsave(str(png_path), rgba)
        else:
            rgba = _phase_to_gamma_rgba(phase, invalid_mask=invalid)
            plt.imsave(str(png_path), rgba)
        ds = None
        return

    data = np.asarray(arr, dtype=np.float32)
    invalid = ~np.isfinite(data)
    if nodata is not None:
        invalid |= np.isclose(data, float(nodata), atol=1e-12)

    if mode == "amplitude":
        data = np.log(np.maximum(data, 1e-6))
        vmin, vmax = _robust_percentile_limits(data, invalid_mask=invalid, p_lo=2.0, p_hi=98.0)
        if vmin is None or vmax is None:
            ds = None
            return
        plt.imsave(str(png_path), data, cmap="gray", vmin=vmin, vmax=vmax)
    elif mode == "coherence":
        data = np.clip(data, 0.0, 1.0)
        plt.imsave(str(png_path), data, cmap="gray", vmin=0.0, vmax=1.0)
    elif mode in ("los", "unwrapped"):
        vmin, vmax = _robust_percentile_limits(data, invalid_mask=invalid, p_lo=2.0, p_hi=98.0)
        if vmin is None or vmax is None:
            plt.imsave(str(png_path), np.zeros_like(data), cmap="viridis")
            ds = None
            return
        plt.imsave(str(png_path), data, cmap="viridis", vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = _robust_percentile_limits(data, invalid_mask=invalid, p_lo=2.0, p_hi=98.0)
        if vmin is None or vmax is None:
            plt.imsave(str(png_path), np.zeros_like(data), cmap="gray")
            ds = None
            return
        plt.imsave(str(png_path), data, cmap="gray", vmin=vmin, vmax=vmax)
    ds = None


def _find_required(path: Path, desc: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="DInSAR 一体化流程")
    parser.add_argument("master", help="主影像标识：可传 master 或 master.tiff/master.vrt")
    parser.add_argument("slave", help="辅影像标识：可传 slave 或 slave.tiff/slave.vrt")
    parser.add_argument("--output-dir", default="dinsar_output", help="输出目录")
    parser.add_argument(
        "--dem",
        default=None,
        help="可选，手动指定 DEM 路径（tif/vrt）",
    )
    parser.add_argument(
        "--dem-on-mismatch",
        choices=["mkdem", "exit"],
        default="mkdem",
        help="当已有 DEM 无法覆盖 SAR 范围时：mkdem=自动重建（默认），exit=报错退出",
    )
    parser.add_argument(
        "--multilook",
        required=True,
        help="多视参数，格式 az:rg，例如 4:4",
    )
    parser.add_argument("--snaphu-tile-rows", type=int, default=None, help="snaphu tile 行数（启用分块解缠）")
    parser.add_argument("--snaphu-tile-cols", type=int, default=None, help="snaphu tile 列数（启用分块解缠）")
    parser.add_argument("--snaphu-tile-row-overlap", type=int, default=512, help="snaphu tile 行重叠（复杂地形建议 >= 512）")
    parser.add_argument("--snaphu-tile-col-overlap", type=int, default=512, help="snaphu tile 列重叠（复杂地形建议 >= 512）")
    parser.add_argument("--snaphu-nproc", type=int, default=None, help="snaphu tile 模式并行进程数")
    parser.add_argument(
        "--snaphu-auto-tile-max-pixels",
        type=int,
        default=4_000_000,
        help="生产默认：自动分块时每 tile 最大像素数（默认 4,000,000）",
    )
    parser.add_argument(
        "--snaphu-auto-reserve-cores",
        type=int,
        default=1,
        help="生产默认：自动并行时预留 CPU 核心数（默认 1）",
    )
    parser.add_argument(
        "--snaphu-auto-max-nproc",
        type=int,
        default=None,
        help="生产默认：自动并行 nproc 上限（默认=CPU核心数的80%%）",
    )
    parser.add_argument(
        "--snaphu-disable-auto-tile",
        action="store_true",
        help="禁用自动分块建议；未手动指定 tile 时使用单块解缠",
    )
    parser.add_argument(
        "--skip-geosar-coreg",
        action="store_true",
        help="在 geosar 阶段跳过模拟 SAR 与真实 SAR 的粗配准（不传 --real_sar）",
    )
    parser.add_argument(
        "--registration-esd",
        action="store_true",
        help="在 registration 阶段启用 ESD 残余方位向配准（推荐 Sentinel-1 TOPS）",
    )
    parser.add_argument(
        "--registration-esd-apply-low-reliability",
        action="store_true",
        help="当 registration-esd 启用时，即便 ESD 结果 reliability=low 也强制应用",
    )
    parser.add_argument(
        "--stop-after-wrapped-phase",
        action="store_true",
        help="仅处理到 wrapped phase（含配准/干涉/滤波），跳过 snaphu 解缠与地理编码",
    )
    parser.add_argument("--force", action="store_true", help="强制重跑各步骤（默认存在结果则复用）")
    args = parser.parse_args()
    try:
        nalks, nrlks = _parse_multilook(args.multilook)
    except ValueError as e:
        parser.error(str(e))
    if (args.snaphu_tile_rows is None) ^ (args.snaphu_tile_cols is None):
        parser.error("--snaphu-tile-rows 和 --snaphu-tile-cols 需要同时设置")
    if args.snaphu_tile_rows is not None and int(args.snaphu_tile_rows) <= 0:
        parser.error("--snaphu-tile-rows 必须为正整数")
    if args.snaphu_tile_cols is not None and int(args.snaphu_tile_cols) <= 0:
        parser.error("--snaphu-tile-cols 必须为正整数")
    if int(args.snaphu_tile_row_overlap) <= 0 or int(args.snaphu_tile_col_overlap) <= 0:
        parser.error("--snaphu-tile-row-overlap/--snaphu-tile-col-overlap 必须为正整数")
    if args.snaphu_nproc is not None and int(args.snaphu_nproc) <= 0:
        parser.error("--snaphu-nproc 必须为正整数")
    if int(args.snaphu_auto_tile_max_pixels) <= 0:
        parser.error("--snaphu-auto-tile-max-pixels 必须为正整数")
    if int(args.snaphu_auto_reserve_cores) < 0:
        parser.error("--snaphu-auto-reserve-cores 必须为非负整数")
    if args.snaphu_auto_max_nproc is not None and int(args.snaphu_auto_max_nproc) <= 0:
        parser.error("--snaphu-auto-max-nproc 必须为正整数")

    master = _resolve_image_path(args.master)
    slave = _resolve_image_path(args.slave)
    master_yaml = _resolve_yaml_for_image(master, args.master)
    slave_yaml = _resolve_yaml_for_image(slave, args.slave)
    out_crs = "UTM"
    snaphu_cost_mode = "defo"

    out_root = Path(args.output_dir).expanduser().resolve()
    stage = {
        "input": out_root / "00_input",
        "ml": out_root / "01_multilook",
        "dem": out_root / "02_dem",
        "reg": out_root / "03_registration",
        "geo": out_root / "04_geometry",
        "base": out_root / "05_baseline",
        "intf": out_root / "06_interferometry",
        "gcd": out_root / "07_geocode",
    }
    for p in stage.values():
        p.mkdir(parents=True, exist_ok=True)

    # 输入快照（软链接）
    master_src_tif = stage["input"] / "master_src.tiff"
    slave_src_tif = stage["input"] / "slave_src.tiff"
    _safe_symlink_or_copy(master, master_src_tif)
    _safe_symlink_or_copy(slave, slave_src_tif)
    _safe_symlink_or_copy(master_yaml, stage["input"] / "master_src.yaml")
    _safe_symlink_or_copy(slave_yaml, stage["input"] / "slave_src.yaml")
    master_src_vrt = _ensure_vrt(master_src_tif)
    slave_src_vrt = _ensure_vrt(slave_src_tif)

    master_name = "master_ml"
    slave_name = "slave_ml"
    master_ml_tif = stage["ml"] / f"{master_name}.tiff"
    slave_ml_tif = stage["ml"] / f"{slave_name}.tiff"
    master_ml_yaml = stage["ml"] / f"{master_name}.yaml"
    slave_ml_yaml = stage["ml"] / f"{slave_name}.yaml"

    # 1) 多视
    if args.force or (not master_ml_tif.exists()) or (not slave_ml_tif.exists()):
        if nalks == 1 and nrlks == 1:
            print("\n[1/8] 多视: 1:1，直接复用原始影像")
            _safe_symlink_or_copy(master, master_ml_tif)
            _safe_symlink_or_copy(slave, slave_ml_tif)
            _safe_symlink_or_copy(master_yaml, master_ml_yaml)
            _safe_symlink_or_copy(slave_yaml, slave_ml_yaml)
            _ensure_vrt(master_ml_tif)
            _ensure_vrt(slave_ml_tif)
        else:
            print(f"\n[1/8] 多视: az={nalks}, rg={nrlks}")
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "utils" / "multilook.py"),
                    str(master_src_vrt),
                    str(stage["ml"] / master_name),
                    "--nalks",
                    str(nalks),
                    "--nrlks",
                    str(nrlks),
                    "--input-yaml",
                    str(master_yaml),
                    "--output-yaml",
                    str(master_ml_yaml),
                    "--preserve-phase",
                ]
            )
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "utils" / "multilook.py"),
                    str(slave_src_vrt),
                    str(stage["ml"] / slave_name),
                    "--nalks",
                    str(nalks),
                    "--nrlks",
                    str(nrlks),
                    "--input-yaml",
                    str(slave_yaml),
                    "--output-yaml",
                    str(slave_ml_yaml),
                    "--preserve-phase",
                ]
            )
    else:
        print("\n[1/8] 多视: 已存在结果，跳过")

    # 约束：后续流程（如 geosar 几何回写）必须作用在 multilook 阶段 YAML 本地副本，
    # 不能通过软链接回写到原始输入 YAML。
    _ensure_regular_file(master_ml_yaml)
    _ensure_regular_file(slave_ml_yaml)

    master_ml_vrt = _ensure_vrt(master_ml_tif)
    _ = _ensure_vrt(slave_ml_tif)

    # 2) DEM
    print("\n[2/8] DEM 准备")
    sar_bbox = _read_sar_bbox_from_yaml(master_yaml)
    print(_fmt_bbox("SAR", sar_bbox))

    if args.dem:
        dem_latlon = Path(args.dem).expanduser().resolve()
        if not dem_latlon.exists():
            raise FileNotFoundError(f"--dem 指定文件不存在: {dem_latlon}")
        print(f"使用用户指定 DEM: {dem_latlon}")
    else:
        dem_latlon = _auto_find_dem(
            [
                master.parent,
                slave.parent,
                Path.cwd(),
                out_root,
                stage["dem"],
            ]
        )
        if dem_latlon is None:
            dem_latlon_candidate = stage["dem"] / "dem_latlon.tif"
            if args.force or (not dem_latlon_candidate.exists()):
                _run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "utils" / "mkdem.py"),
                        str(master_ml_yaml),
                        "-o",
                        str(stage["dem"]),
                        "-C",
                        "latlon",
                    ]
                )
            dem_latlon = _auto_find_dem([stage["dem"]])
            if dem_latlon is None:
                raise FileNotFoundError(f"自动 DEM 生成失败，未在 {stage['dem']} 找到 DEM 结果")

    ok_dem, dem_bbox, dem_msg = _ensure_dem_covers_sar(dem_path=dem_latlon, sar_bbox=sar_bbox)
    print(dem_msg)
    if not ok_dem:
        print("警告: 当前 DEM 不能完整覆盖 SAR 范围。")
        if args.dem_on_mismatch == "exit":
            raise RuntimeError(
                "DEM 覆盖检查失败，按 --dem-on-mismatch=exit 终止。"
            )

        print("尝试调用 mkdem 重新生成可覆盖 SAR 范围的 DEM...")
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "utils" / "mkdem.py"),
                str(master_ml_yaml),
                "-o",
                str(stage["dem"]),
                "-C",
                "latlon",
            ]
        )
        dem_retry = _auto_find_dem([stage["dem"]])
        if dem_retry is None:
            raise FileNotFoundError(f"mkdem 执行后，未在 {stage['dem']} 找到 DEM")
        ok_dem2, _, dem_msg2 = _ensure_dem_covers_sar(dem_path=dem_retry, sar_bbox=sar_bbox)
        print(dem_msg2)
        if not ok_dem2:
            raise RuntimeError("mkdem 生成的 DEM 仍无法覆盖 SAR 范围，请检查输入参数。")
        dem_latlon = dem_retry
        print(f"已切换使用 mkdem DEM: {dem_latlon}")

    dem_snapshot = _snapshot_dem_to_stage(dem_latlon, stage["dem"])
    print(f"DEM 快照: {dem_snapshot}")
    print(f"DEM: {dem_latlon}")

    # 3) 配准
    print("\n[3/8] 配准与重采样")
    slave_resamp_tif = stage["reg"] / f"{slave_name}_resamp.tiff"
    slave_resamp_yaml = stage["reg"] / f"{slave_name}_resamp.yaml"
    if args.force or (not slave_resamp_tif.exists()) or (not slave_resamp_yaml.exists()):
        regist_cmd = [
            sys.executable,
            str(REPO_ROOT / "registration" / "regist.py"),
            master_name,
            slave_name,
            "--output-dir",
            str(stage["reg"]),
        ]
        if args.registration_esd:
            regist_cmd.append("--esd")
            if args.registration_esd_apply_low_reliability:
                regist_cmd.append("--esd-apply-low-reliability")
        _run(regist_cmd, cwd=stage["ml"])
    else:
        print("配准结果已存在，跳过")
    _find_required(slave_resamp_tif, "重采样 slave")
    _find_required(slave_resamp_yaml, "重采样 slave YAML")
    slave_resamp_vrt = _ensure_vrt(slave_resamp_tif)

    # 4) geosar
    print("\n[4/8] 几何建模（geosar）")
    geosar_h5 = stage["geo"] / "geosar.h5"
    if args.force or (not geosar_h5.exists()):
        geosar_cmd = [
            sys.executable,
            str(REPO_ROOT / "utils" / "geosar.py"),
            "--yaml",
            str(master_ml_yaml),
            "--dem",
            str(dem_latlon),
            "--output",
            str(geosar_h5),
        ]
        if args.skip_geosar_coreg:
            print("geosar: 已禁用模拟 SAR 与真实 SAR 的粗配准（skip-geosar-coreg）")
        else:
            geosar_cmd += ["--real_sar", str(master_ml_tif)]
        _run(geosar_cmd)
    else:
        print("geosar 结果已存在，跳过")
    _find_required(geosar_h5, "geosar HDF")

    # 5) baseline
    print("\n[5/8] LOS 基线")
    baseline_hdf = stage["base"] / "baseline.hdf"
    base_stage = stage["base"] / "stage"
    base_stage.mkdir(parents=True, exist_ok=True)
    _safe_symlink_or_copy(master_ml_yaml, base_stage / "master_ref.yaml")
    _safe_symlink_or_copy(slave_resamp_yaml, base_stage / "slave_ref.yaml")
    _safe_symlink_or_copy(master_ml_vrt, base_stage / "master_ref.vrt")

    if args.force or (not baseline_hdf.exists()):
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "baseline.py"),
                "master_ref",
                "slave_ref",
                "--output",
                str(baseline_hdf),
                "--vrt",
                "master_ref.vrt",
                "--geosar-hdf",
                str(geosar_h5),
            ],
            cwd=base_stage,
        )
    else:
        print("baseline 结果已存在，跳过")
    _find_required(baseline_hdf, "baseline HDF")

    # 6) interferometry：raw/coherence -> flat-only -> flat+topo -> snaphu
    print("\n[6/8] 干涉处理与解缠")
    int_main = stage["intf"] / "main"
    int_flat = stage["intf"] / "flat"
    int_comb = stage["intf"] / "combined"
    int_main.mkdir(parents=True, exist_ok=True)
    int_flat.mkdir(parents=True, exist_ok=True)
    int_comb.mkdir(parents=True, exist_ok=True)

    pair_prefix = f"{master_ml_tif.stem}_{slave_resamp_tif.stem}"
    raw_ifg_bin = int_main / f"{pair_prefix}_interferogram.bin"
    coherence_bin = int_main / f"{pair_prefix}_coherence.bin"
    flat_only_ifg_bin = int_flat / f"{pair_prefix}_interferogram_flat_only.bin"
    both_raw_ifg_bin = int_comb / f"{pair_prefix}_interferogram_flat_topo_removed_raw.bin"
    both_ifg_bin = int_comb / f"{pair_prefix}_interferogram_flat_topo_removed.bin"
    both_filtered_phase_bin = int_comb / f"{pair_prefix}_filtered_phase_flat_topo_removed_goldstein.bin"
    wrapped_both_phase_bin = int_comb / f"{pair_prefix}_wrapped_phase_flat_topo_removed.bin"
    snaphu_prefix = f"{pair_prefix}_flat_topo_removed"
    unwrap_bin = int_comb / f"{snaphu_prefix}_unwrapped_phase.bin"
    los_m_bin = int_comb / f"{snaphu_prefix}_los_deformation_m.bin"

    ds = gdal.Open(str(master_ml_tif), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开主影像: {master_ml_tif}")
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None

    radar_par = _read_yaml_params(master_ml_yaml)
    snaphu_runtime = _resolve_snaphu_runtime_params(shape=shape, args=args)
    if snaphu_runtime["use_tile"]:
        print(
            "snaphu 参数(生产默认): "
            f"tile={snaphu_runtime['tile_rows']}x{snaphu_runtime['tile_cols']} "
            f"overlap=({snaphu_runtime['tile_row_overlap']},{snaphu_runtime['tile_col_overlap']}) "
            f"nproc={snaphu_runtime['nproc']} "
            f"[tile={snaphu_runtime['tile_source']}, nproc={snaphu_runtime['nproc_source']}]"
        )
    else:
        print(f"snaphu 参数(生产默认): 单块解缠 [tile={snaphu_runtime['tile_source']}]")
    for wmsg in snaphu_runtime["warnings"]:
        print(f"snaphu 提示: {wmsg}")

    # 6.0) 先生成原始干涉图与相干性（不做解缠）
    if args.force or (not raw_ifg_bin.exists()) or (not coherence_bin.exists()):
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "interferometry" / "interferometry.py"),
                str(master_ml_tif),
                str(slave_resamp_tif),
                "--output-dir",
                str(int_main),
                "--master-yaml",
                str(master_ml_yaml),
                "--geosar-hdf",
                str(geosar_h5),
                "--baseline-hdf",
                str(baseline_hdf),
                "--dem",
                str(dem_latlon),
                "--wavelength",
                str(radar_par["wavelength"]),
            ]
        )
    else:
        print("raw/coherence 已存在，跳过")
    _find_required(raw_ifg_bin, "原始干涉图")
    _find_required(coherence_bin, "相干图")

    # 6.1) flat-only（仅去平地）并使用专用命名
    flat_generated_bin = int_flat / f"{pair_prefix}_interferogram_flat.bin"
    if args.force or (not flat_generated_bin.exists()):
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "interferometry" / "interferometry.py"),
                str(master_ml_tif),
                str(slave_resamp_tif),
                "--output-dir",
                str(int_flat),
                "--master-yaml",
                str(master_ml_yaml),
                "--geosar-hdf",
                str(geosar_h5),
                "--baseline-hdf",
                str(baseline_hdf),
                "--dem",
                str(dem_latlon),
                "--wavelength",
                str(radar_par["wavelength"]),
                "--remove-flat",
            ]
        )
    else:
        print("flat-only 干涉结果已存在，跳过")
    _find_required(flat_generated_bin, "flat-only 干涉图")

    if args.force or (not flat_only_ifg_bin.exists()):
        _safe_symlink_or_copy(flat_generated_bin, flat_only_ifg_bin)
        _create_raw_vrt(flat_only_ifg_bin, shape=shape, complex_data=True)
    _find_required(flat_only_ifg_bin, "flat-only（重命名）干涉图")

    # 6.2) 复用 flat-only 结果作为 flat+topo(raw)
    # 说明：按当前处理链定义，flat-only 阶段已同步去除平地与地形相位，
    # 这里不再重复执行地形相位移除，只做统一命名供后续 Goldstein/snaphu 使用。
    both_tif_sar = int_comb / f"{pair_prefix}_interferogram_flat_topo_removed.tif"
    both_png_sar = int_comb / f"{pair_prefix}_interferogram_flat_topo_removed.png"
    if args.force or (not both_raw_ifg_bin.exists()):
        print("复用 flat-only 结果作为去平地+地形干涉图（raw）...")
        flat_ifg = np.fromfile(str(flat_only_ifg_bin), dtype=np.complex64)
        if flat_ifg.size != shape[0] * shape[1]:
            raise RuntimeError(
                f"flat-only interferogram 尺寸异常: {flat_ifg.size} vs {shape[0] * shape[1]}"
            )
        _safe_symlink_or_copy(flat_only_ifg_bin, both_raw_ifg_bin)
        _create_raw_vrt(both_raw_ifg_bin, shape=shape, complex_data=True)
    else:
        print("去平地+地形干涉图（raw）已存在，跳过")

    _find_required(both_raw_ifg_bin, "平地+地形均去除干涉图（raw）")

    # 6.3) 对 flat+topo(raw) 执行 Goldstein 滤波，得到后续统一使用的 flat+topo(filtered)
    if args.force or (not both_ifg_bin.exists()) or (not both_filtered_phase_bin.exists()):
        print("对 flat+topo 干涉相位执行 Goldstein 滤波...")
        both_raw_ifg = np.fromfile(str(both_raw_ifg_bin), dtype=np.complex64)
        if both_raw_ifg.size != shape[0] * shape[1]:
            raise RuntimeError(
                f"flat+topo(raw) interferogram 尺寸异常: {both_raw_ifg.size} vs {shape[0] * shape[1]}"
            )
        both_raw_ifg = both_raw_ifg.reshape(shape)

        coherence = np.fromfile(str(coherence_bin), dtype=np.float32)
        if coherence.size != shape[0] * shape[1]:
            raise RuntimeError(
                f"coherence 尺寸异常: {coherence.size} vs {shape[0] * shape[1]}"
            )
        coherence = coherence.reshape(shape)

        raw_phase = np.angle(both_raw_ifg).astype(np.float32, copy=False)
        goldstein_alpha = 0.5
        filter_obj = InterferogramFilter(method="goldstein")
        filtered_phase = filter_obj.filter(raw_phase, coherence, alpha=goldstein_alpha)
        filtered_phase = filtered_phase.astype(np.float32, copy=False)
        both_filtered_ifg = np.abs(both_raw_ifg).astype(np.float32) * np.exp(1j * filtered_phase)
        both_filtered_ifg = both_filtered_ifg.astype(np.complex64, copy=False)

        filtered_phase.tofile(str(both_filtered_phase_bin))
        _create_raw_vrt(both_filtered_phase_bin, shape=shape, complex_data=False)
        both_filtered_ifg.tofile(str(both_ifg_bin))
        _create_raw_vrt(both_ifg_bin, shape=shape, complex_data=True)

        coh_tif_for_mask = int_main / f"{pair_prefix}_coherence.tif"
        _bin_vrt_to_tif(coherence_bin, coh_tif_for_mask, shape=shape, complex_data=False)
        _bin_vrt_to_tif(both_ifg_bin, both_tif_sar, shape=shape, complex_data=True)
        _save_complex_phase_products(
            ifg_tif=both_tif_sar,
            pure_png=both_png_sar,
            coherence_tif=coh_tif_for_mask,
            coherence_threshold=0.3,
            phase_palette="gamma",
        )
        print(f"Goldstein alpha={goldstein_alpha}")
    else:
        print("flat+topo Goldstein 滤波结果已存在，跳过")

    _find_required(both_ifg_bin, "平地+地形均去除干涉图（Goldstein后）")
    _find_required(both_filtered_phase_bin, "平地+地形均去除相位（Goldstein后）")

    # 为 snaphu 准备包裹相位（来自去平地+地形干涉图）
    if args.force or (not wrapped_both_phase_bin.exists()):
        both_ifg = np.fromfile(str(both_ifg_bin), dtype=np.complex64)
        if both_ifg.size != shape[0] * shape[1]:
            raise RuntimeError(
                f"flat+topo interferogram 尺寸异常: {both_ifg.size} vs {shape[0] * shape[1]}"
            )
        both_ifg = both_ifg.reshape(shape)
        wrapped_phase = np.angle(both_ifg).astype(np.float32, copy=False)
        wrapped_phase.tofile(str(wrapped_both_phase_bin))
        _create_raw_vrt(wrapped_both_phase_bin, shape=shape, complex_data=False)
    _find_required(wrapped_both_phase_bin, "snaphu 包裹相位（flat+topo）")

    if bool(args.stop_after_wrapped_phase):
        print("按 --stop-after-wrapped-phase 设置，已在 wrapped phase 阶段停止。")
        return 0

    # 6.4) 对去平地+地形(并经 Goldstein)后的相位执行 snaphu 解缠
    if args.force or (not unwrap_bin.exists()) or (not los_m_bin.exists()):
        cmd_unwrap = [
            sys.executable,
            str(REPO_ROOT / "interferometry" / "snaphu.py"),
            "--wrapped-phase",
            str(wrapped_both_phase_bin),
            "--coherence",
            str(coherence_bin),
            "--width",
            str(shape[1]),
            "--height",
            str(shape[0]),
            "--output-dir",
            str(int_comb),
            "--prefix",
            str(snaphu_prefix),
            "--wavelength",
            str(radar_par["wavelength"]),
            "--cost-mode",
            str(snaphu_cost_mode),
        ]
        if snaphu_runtime["use_tile"]:
            cmd_unwrap.extend(
                [
                    "--tile-rows",
                    str(int(snaphu_runtime["tile_rows"])),
                    "--tile-cols",
                    str(int(snaphu_runtime["tile_cols"])),
                    "--tile-row-overlap",
                    str(int(snaphu_runtime["tile_row_overlap"])),
                    "--tile-col-overlap",
                    str(int(snaphu_runtime["tile_col_overlap"])),
                ]
            )
            if snaphu_runtime["nproc"] is not None:
                cmd_unwrap.extend(["--nproc", str(int(snaphu_runtime["nproc"]))])
        _run(cmd_unwrap)
    else:
        print("snaphu 解缠结果已存在，跳过")
    _find_required(unwrap_bin, "解缠相位")
    _find_required(los_m_bin, "LOS 形变(m)")

    # 7) 地理编码
    print("\n[7/8] 地理编码与产品输出")
    sar_prod_dir = stage["gcd"] / "sar_products"
    geo_prod_dir = stage["gcd"] / "geo_products"
    png_dir = stage["gcd"] / "png"
    sar_prod_dir.mkdir(parents=True, exist_ok=True)
    geo_prod_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    removed_overlay = _cleanup_legacy_overlay_pngs(png_dir)
    if removed_overlay > 0:
        print(f"已清理旧 overlay PNG: {removed_overlay} 个")

    # 7.1 几何均值幅度（SAR 域）
    amp_geom_sar_tif = sar_prod_dir / "geom_mean_amplitude_sar.tif"
    if args.force or (not amp_geom_sar_tif.exists()):
        m = _read_complex_tiff(master_ml_tif)
        s = _read_complex_tiff(slave_resamp_tif)
        if m.shape != s.shape:
            raise RuntimeError(f"主辅影像尺寸不一致: {m.shape} vs {s.shape}")
        amp_geom = np.sqrt(np.abs(m) * np.abs(s)).astype(np.float32)
        _write_float_tiff(amp_geom_sar_tif, amp_geom)
    _save_preview_png(amp_geom_sar_tif, png_dir / "geom_mean_amplitude_sar.png", mode="amplitude")

    master_amp_sar_tif = sar_prod_dir / "master_amplitude_sar.tif"
    if args.force or (not master_amp_sar_tif.exists()):
        m = _read_complex_tiff(master_ml_tif)
        master_amp = np.abs(m).astype(np.float32)
        _write_float_tiff(master_amp_sar_tif, master_amp)
    _save_preview_png(master_amp_sar_tif, png_dir / "master_amplitude_sar.png", mode="amplitude")

    # 7.2 关键 SAR 域产品 bin -> tif
    ifg_sar_tif = _bin_vrt_to_tif(raw_ifg_bin, sar_prod_dir / "interferogram_sar.tif", shape, complex_data=True)
    coh_sar_tif = _bin_vrt_to_tif(coherence_bin, sar_prod_dir / "coherence_sar.tif", shape, complex_data=False)
    both_sar_tif = _bin_vrt_to_tif(
        both_ifg_bin, sar_prod_dir / "interferogram_flat_topo_removed_sar.tif", shape, complex_data=True
    )
    unw_sar_tif = _bin_vrt_to_tif(unwrap_bin, sar_prod_dir / "unwrapped_phase_sar.tif", shape, complex_data=False)
    los_sar_tif = _bin_vrt_to_tif(los_m_bin, sar_prod_dir / "los_displacement_m_sar.tif", shape, complex_data=False)

    _save_preview_png(coh_sar_tif, png_dir / "coherence_sar.png", mode="coherence")
    _save_complex_phase_products(
        ifg_tif=ifg_sar_tif,
        pure_png=png_dir / "interferogram_sar.png",
        coherence_tif=coh_sar_tif,
        coherence_threshold=0.3,
        phase_palette="gamma",
    )
    _save_complex_phase_products(
        ifg_tif=both_sar_tif,
        pure_png=png_dir / "interferogram_flat_topo_removed_sar.png",
        coherence_tif=coh_sar_tif,
        coherence_threshold=0.3,
        phase_palette="gamma",
    )
    _save_preview_png(unw_sar_tif, png_dir / "unwrapped_phase_sar.png", mode="unwrapped")
    _save_preview_png(los_sar_tif, png_dir / "los_displacement_m_sar.png", mode="los")

    # 默认 geocode 分辨率（米）
    geocode_res_m = max(
        float(radar_par["range_spacing"]) * float(nrlks),
        float(radar_par["azimuth_spacing"]) * float(nalks),
    )
    print(f"geocode 输出分辨率: {geocode_res_m:.3f} m")

    geocode_tasks = [
        ("geom_mean_amplitude", amp_geom_sar_tif, "amplitude"),
        ("master_amplitude", master_amp_sar_tif, "amplitude"),
        ("coherence", coh_sar_tif, "coherence"),
        ("interferogram", ifg_sar_tif, "complex_phase"),
        ("interferogram_flat_topo_removed", both_sar_tif, "complex_phase"),
        ("unwrapped_phase", unw_sar_tif, "unwrapped"),
        ("los_displacement_m", los_sar_tif, "los"),
    ]

    geo_outputs: Dict[str, str] = {}
    for name, sar_tif, mode in geocode_tasks:
        geo_tif = geo_prod_dir / f"{name}_geo.tif"
        if args.force or (not geo_tif.exists()):
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "utils" / "sar2ll.py"),
                    "warp",
                    str(sar_tif),
                    str(geosar_h5),
                    str(geo_tif),
                    str(out_crs),
                    str(geocode_res_m),
                    "--interp",
                    "auto",
                    "--extent",
                    "sar",
                ]
            )
        if mode != "complex_phase":
            _save_preview_png(geo_tif, png_dir / f"{name}_geo.png", mode=mode)
        geo_outputs[name] = str(geo_tif)

    # 干涉图（Geo域）：仅输出纯相位图
    coherence_geo_tif = geo_prod_dir / "coherence_geo.tif"
    ifg_geo_tif = geo_prod_dir / "interferogram_geo.tif"
    both_geo_tif = geo_prod_dir / "interferogram_flat_topo_removed_geo.tif"

    if ifg_geo_tif.exists():
        _save_complex_phase_products(
            ifg_tif=ifg_geo_tif,
            pure_png=png_dir / "interferogram_geo.png",
            coherence_tif=coherence_geo_tif if coherence_geo_tif.exists() else None,
            coherence_threshold=0.3,
            phase_palette="gamma",
        )
    if both_geo_tif.exists():
        _save_complex_phase_products(
            ifg_tif=both_geo_tif,
            pure_png=png_dir / "interferogram_flat_topo_removed_geo.png",
            coherence_tif=coherence_geo_tif if coherence_geo_tif.exists() else None,
            coherence_threshold=0.3,
            phase_palette="gamma",
        )

    # 8) 清单
    print("\n[8/8] 写出结果清单")
    manifest: Dict[str, Any] = {
        "inputs": {
            "master": str(master),
            "slave": str(slave),
            "master_yaml": str(master_yaml),
            "slave_yaml": str(slave_yaml),
            "multilook": f"{nalks}:{nrlks}",
            "skip_geosar_coreg": bool(args.skip_geosar_coreg),
            "registration_esd": bool(args.registration_esd),
            "registration_esd_apply_low_reliability": bool(args.registration_esd_apply_low_reliability),
        },
        "snaphu_runtime_params": {
            "use_tile": bool(snaphu_runtime["use_tile"]),
            "tile_rows": int(snaphu_runtime["tile_rows"]),
            "tile_cols": int(snaphu_runtime["tile_cols"]),
            "tile_row_overlap": (
                int(snaphu_runtime["tile_row_overlap"]) if snaphu_runtime["tile_row_overlap"] is not None else None
            ),
            "tile_col_overlap": (
                int(snaphu_runtime["tile_col_overlap"]) if snaphu_runtime["tile_col_overlap"] is not None else None
            ),
            "nproc": int(snaphu_runtime["nproc"]) if snaphu_runtime["nproc"] is not None else None,
            "tile_source": str(snaphu_runtime["tile_source"]),
            "nproc_source": str(snaphu_runtime["nproc_source"]),
        },
        "core_outputs": {
            "dem_latlon_tif": str(dem_latlon),
            "slave_resamp_tiff": str(slave_resamp_tif),
            "geosar_h5": str(geosar_h5),
            "baseline_hdf": str(baseline_hdf),
            "raw_interferogram_bin": str(raw_ifg_bin),
            "coherence_bin": str(coherence_bin),
            "flat_only_interferogram_bin": str(flat_only_ifg_bin),
            "flat_topo_removed_raw_interferogram_bin": str(both_raw_ifg_bin),
            "flat_topo_removed_interferogram_bin": str(both_ifg_bin),
            "flat_topo_removed_filtered_phase_bin": str(both_filtered_phase_bin),
            "flat_topo_removed_wrapped_phase_bin": str(wrapped_both_phase_bin),
            "unwrapped_phase_bin": str(unwrap_bin),
            "los_displacement_m_bin": str(los_m_bin),
        },
        "geocoded_geotiff": geo_outputs,
        "preview_png_dir": str(png_dir),
    }
    manifest_path = out_root / "dinsar_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    print(f"Manifest: {manifest_path}")

    print("\n=== DInSAR 全流程完成 ===")
    print(f"输出目录: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
