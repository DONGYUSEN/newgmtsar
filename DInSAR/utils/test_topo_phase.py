#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试：验证地形相位信息是否能被正确生成。

功能：
1) 基于 dem + geosar + baseline 生成地形相位（raw / wrapped）
2) 输出统计信息与 PASS/FAIL 诊断
3) 可选：对比 flat-only 与 flat+topo 干涉图的相位差，验证地形去除一致性
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
from osgeo import gdal
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from interferometry.interferometry import (  # noqa: E402
    _prepare_phase_model_grids,
    _wrap_phase,
    load_baseline_hdf,
    load_geosar_geometry_hdf,
)


def _read_yaml_params(yaml_path: Path) -> dict:
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

    return {
        "range_spacing": _pick("range_spacing", "range_pixel_spacing", default=5.0),
        "azimuth_spacing": _pick("azimuth_spacing", default=10.0),
        "wavelength": _pick("wavelength", "radar_wavelength", default=0.0555),
    }


def _read_dem(dem_path: Path) -> np.ndarray:
    ds = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开 DEM: {dem_path}")
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    if arr is None:
        raise RuntimeError(f"DEM 读取失败: {dem_path}")
    return arr.astype(np.float32, copy=False)


def _read_shape_and_geo(master_tif: Path) -> Tuple[Tuple[int, int], Optional[tuple], str]:
    ds = gdal.Open(str(master_tif), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开 master: {master_tif}")
    shape = (ds.RasterYSize, ds.RasterXSize)
    gt = ds.GetGeoTransform(can_return_null=True)
    prj = ds.GetProjection()
    ds = None
    return shape, gt, prj


def _write_float_tif(path: Path, arr: np.ndarray, gt: Optional[tuple], prj: str) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(path),
        w,
        h,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if ds is None:
        raise RuntimeError(f"无法创建 GeoTIFF: {path}")
    if gt is not None:
        ds.SetGeoTransform(gt)
    if prj:
        ds.SetProjection(prj)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(np.nan)
    ds.FlushCache()
    ds = None


def _write_uint8_tif(path: Path, arr_u8: np.ndarray, gt: Optional[tuple], prj: str) -> None:
    arr_u8 = np.asarray(arr_u8, dtype=np.uint8)
    h, w = arr_u8.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(path),
        w,
        h,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if ds is None:
        raise RuntimeError(f"无法创建 GeoTIFF: {path}")
    if gt is not None:
        ds.SetGeoTransform(gt)
    if prj:
        ds.SetProjection(prj)
    ds.GetRasterBand(1).WriteArray(arr_u8)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.FlushCache()
    ds = None


def _save_phase_png(path: Path, phase_wrapped: np.ndarray) -> None:
    plt.figure(figsize=(8, 6), dpi=150)
    plt.imshow(phase_wrapped, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label="wrapped phase (rad)")
    plt.title("topographic_phase_wrapped (twilight)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _read_complex_bin(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    arr = np.fromfile(str(path), dtype=np.complex64)
    exp = shape[0] * shape[1]
    if arr.size != exp:
        raise RuntimeError(f"复杂干涉图尺寸异常: {path}, size={arr.size}, expected={exp}")
    return arr.reshape(shape)


def _stat(name: str, arr: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> str:
    x = np.asarray(arr, dtype=np.float32)
    if valid_mask is None:
        m = np.isfinite(x)
    else:
        m = np.isfinite(x) & valid_mask
    if np.count_nonzero(m) == 0:
        return f"{name}: no valid pixels"
    v = x[m]
    p2, p98 = np.percentile(v, [2, 98])
    return (
        f"{name}: min={float(np.min(v)):.6f}, max={float(np.max(v)):.6f}, "
        f"mean={float(np.mean(v)):.6f}, std={float(np.std(v)):.6f}, "
        f"p2={float(p2):.6f}, p98={float(p98):.6f}, valid={v.size}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="测试地形相位是否可正确生成")
    parser.add_argument("--master-tif", required=True, help="master multilook 影像（用于形状和坐标）")
    parser.add_argument("--master-yaml", required=True, help="master yaml（读取 spacing/wavelength）")
    parser.add_argument("--dem", required=True, help="DEM tif/vrt")
    parser.add_argument("--geosar-hdf", required=True, help="geosar.h5")
    parser.add_argument("--baseline-hdf", required=True, help="baseline.hdf")
    parser.add_argument("--output-dir", default="topo_phase_test", help="输出目录")
    parser.add_argument("--wavelength", type=float, default=None, help="覆盖 yaml 里的波长")
    parser.add_argument(
        "--flat-only-ifg-bin",
        default=None,
        help="可选：flat-only 干涉图（complex64 bin），用于与 flat+topo 做一致性对比",
    )
    parser.add_argument(
        "--flat-topo-ifg-bin",
        default=None,
        help="可选：flat+topo 干涉图（complex64 bin），用于与模型地形相位对比",
    )
    args = parser.parse_args()

    master_tif = Path(args.master_tif).expanduser().resolve()
    master_yaml = Path(args.master_yaml).expanduser().resolve()
    dem_path = Path(args.dem).expanduser().resolve()
    geosar_hdf = Path(args.geosar_hdf).expanduser().resolve()
    baseline_hdf = Path(args.baseline_hdf).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    params = _read_yaml_params(master_yaml)
    wavelength = float(args.wavelength) if args.wavelength is not None else float(params["wavelength"])
    range_spacing = float(params["range_spacing"])

    shape, gt, prj = _read_shape_and_geo(master_tif)
    dem_arr = _read_dem(dem_path)

    geom = {}
    geom.update(load_geosar_geometry_hdf(str(geosar_hdf), target_shape=shape))
    geom.update(load_baseline_hdf(str(baseline_hdf), target_shape=shape))
    geom["range_spacing"] = range_spacing

    grids = _prepare_phase_model_grids(
        target_shape=shape,
        dem=dem_arr,
        range_spacing=range_spacing,
        incidence_angle=23.0,
        geometry=geom,
    )

    b_perp = grids["b_perp"]
    inc = grids["incidence_angle"]
    slant = grids["slant_range"]
    h_grid = grids["height"]
    if b_perp is None or inc is None or slant is None or h_grid is None:
        raise RuntimeError("地形相位模型缺关键网格：b_perp/inc/slant/height")

    inc_rad = np.deg2rad(inc.astype(np.float32, copy=False))
    den = slant.astype(np.float32, copy=False) * np.sin(inc_rad)
    valid = np.isfinite(den) & (np.abs(den) > 1e-6) & np.isfinite(h_grid) & np.isfinite(b_perp)

    topo_raw = np.zeros(shape, dtype=np.float32)
    np.divide(
        b_perp.astype(np.float32, copy=False) * h_grid.astype(np.float32, copy=False),
        den,
        out=topo_raw,
        where=valid,
    )
    topo_raw *= np.float32(4.0 * math.pi / wavelength)
    topo_wrapped = _wrap_phase(np.nan_to_num(topo_raw, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32)

    _write_float_tif(out_dir / "topographic_phase_raw.tif", topo_raw, gt, prj)
    _write_float_tif(out_dir / "topographic_phase_wrapped.tif", topo_wrapped, gt, prj)
    topo_raw.tofile(str(out_dir / "topographic_phase_raw.bin"))
    topo_wrapped.tofile(str(out_dir / "topographic_phase_wrapped.bin"))

    # 便于快速查看：输出伪彩 PNG 与 8-bit quicklook tif
    _save_phase_png(out_dir / "topographic_phase_wrapped_preview.png", topo_wrapped)
    wrapped_u8 = ((topo_wrapped + np.pi) / (2.0 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)
    _write_uint8_tif(out_dir / "topographic_phase_wrapped_u8.tif", wrapped_u8, gt, prj)

    valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)
    std_raw = float(np.std(topo_raw[valid])) if np.any(valid) else 0.0
    std_wrapped = float(np.std(topo_wrapped[valid])) if np.any(valid) else 0.0

    print("=== Topographic Phase Test ===")
    print(f"shape={shape}, wavelength={wavelength:.9f}, range_spacing={range_spacing:.6f}")
    print(_stat("topo_raw", topo_raw, valid))
    print(_stat("topo_wrapped", topo_wrapped, valid))
    print(f"valid_ratio={valid_ratio*100:.3f}%")

    passed = True
    reasons: list[str] = []
    if valid_ratio < 0.90:
        passed = False
        reasons.append("valid_ratio < 90%")
    if std_raw < 1e-3:
        passed = False
        reasons.append("topo_raw std too small")
    if std_wrapped < 1e-3:
        passed = False
        reasons.append("topo_wrapped std too small")

    # 可选一致性检查：delta_phase(flat-only -> flat+topo) 是否接近模型 topo_wrapped
    if args.flat_only_ifg_bin and args.flat_topo_ifg_bin:
        flat_only_path = Path(args.flat_only_ifg_bin).expanduser().resolve()
        flat_topo_path = Path(args.flat_topo_ifg_bin).expanduser().resolve()
        if (not flat_only_path.exists()) or (not flat_topo_path.exists()):
            print(
                "WARN: 可选一致性对比跳过，输入文件不存在: "
                f"flat_only={flat_only_path.exists()}, flat_topo={flat_topo_path.exists()}"
            )
        else:
            flat_only = _read_complex_bin(flat_only_path, shape)
            flat_topo = _read_complex_bin(flat_topo_path, shape)
            delta_phase = _wrap_phase(np.angle(flat_only) - np.angle(flat_topo)).astype(np.float32)
            diff = _wrap_phase(delta_phase - topo_wrapped).astype(np.float32)
            rmse = float(np.sqrt(np.mean((diff[valid]) ** 2))) if np.any(valid) else float("nan")
            print(_stat("delta_phase(flat_only-flat_topo)", delta_phase, valid))
            print(_stat("delta_vs_model_diff", diff, valid))
            print(f"delta_vs_model_rmse={rmse:.6f} rad")
            if np.isfinite(rmse) and rmse > 0.8:
                passed = False
                reasons.append(f"delta/model rmse too large ({rmse:.3f} rad)")

    print(f"output_dir={out_dir}")
    if passed:
        print("RESULT: PASS - 地形相位已成功生成，且统计特征正常。")
        return 0

    print("RESULT: FAIL - " + "; ".join(reasons))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
