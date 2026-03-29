#!/usr/bin/env python3
"""
snaphu 解缠与 LOS 形变输出工具。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# 允许脚本以 `python interferometry/snaphu.py` 方式运行时导入 utils 模块
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.snaphu_utils import run_snaphu_unwrap

MAX_SNAPHU_NPROC = 4


def _normalize_nproc(nproc: Optional[int]) -> int:
    """
    强制限制 snaphu 并行进程数不超过 MAX_SNAPHU_NPROC。
    """
    if nproc is None:
        return MAX_SNAPHU_NPROC
    n = int(nproc)
    if n <= 0:
        raise ValueError(f"nproc 必须为正整数，当前={n}")
    return min(n, MAX_SNAPHU_NPROC)


def phase_to_los_deformation(unwrapped_phase: np.ndarray, wavelength: float) -> np.ndarray:
    """
    将解缠相位转换为 LOS 方向形变（米）。
    disp_los = phase * wavelength / (4*pi)
    """
    return (np.asarray(unwrapped_phase, dtype=np.float64) * float(wavelength) / (4.0 * np.pi)).astype(np.float32)


def _create_vrt(bin_file: Path, shape: Tuple[int, int], dtype: np.dtype) -> Path:
    """为原始二进制输出创建 VRT。"""
    vrt_file = bin_file.with_suffix(".vrt")
    width, height = shape[1], shape[0]

    gdal_type = "Float32"
    bands = 1
    pixel_size = 4
    line_offset = width * 4

    if np.dtype(dtype) == np.complex64:
        bands = 2
        pixel_size = 8
        line_offset = width * 8

    vrt_content = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">']
    for i in range(bands):
        vrt_content.extend(
            [
                f'  <VRTRasterBand dataType="{gdal_type}" band="{i + 1}" subClass="VRTRawRasterBand">',
                f"    <SourceFilename relativeToVRT=\"1\">{bin_file.name}</SourceFilename>",
                f"    <ImageOffset>{i * 4}</ImageOffset>",
                f"    <PixelOffset>{pixel_size}</PixelOffset>",
                f"    <LineOffset>{line_offset}</LineOffset>",
                "    <ByteOrder>LSB</ByteOrder>",
                "  </VRTRasterBand>",
            ]
        )
    vrt_content.append("</VRTDataset>")
    vrt_file.write_text("\n".join(vrt_content) + "\n", encoding="utf-8")
    return vrt_file


def run_unwrap_and_los(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray,
    output_dir: Path | str,
    prefix: str,
    wavelength: float,
    *,
    snaphu_bin: str = "snaphu",
    cost_mode: str = "defo",
    init_method: str = "mst",
    corr_thresh: Optional[float] = None,
    keep_temp: bool = False,
    tile_rows: Optional[int] = None,
    tile_cols: Optional[int] = None,
    tile_row_overlap: int = 512,
    tile_col_overlap: int = 512,
    nproc: Optional[int] = None,
) -> Dict[str, object]:
    """
    调用 snaphu 解缠并输出 LOS 形变。

    Returns:
        {
            "unwrapped_phase": np.ndarray,
            "los_deformation_m": np.ndarray,
            "los_deformation_mm": np.ndarray,
            "unwrapped_phase_file": str,
            "los_deformation_m_file": str,
            "los_deformation_mm_file": str,
            "snaphu_cmd": List[str],
            "snaphu_log_file": str,
            "snaphu_version": str (optional),
            "snaphu_warnings": List[str] (optional),
            "snaphu_attempt_logs": List[str] (optional),
            "tile_plan": Dict[str, object] (optional),
        }
    """
    if float(wavelength) <= 0:
        raise ValueError(f"wavelength 必须 > 0，当前={wavelength}")
    effective_nproc = _normalize_nproc(nproc)
    if (nproc is None) or (int(nproc) != effective_nproc):
        print(
            f"snaphu nproc 已限制为 {effective_nproc} "
            f"(requested={nproc}, max={MAX_SNAPHU_NPROC})"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_unw = output_dir / f"{prefix}_unwrapped_phase.bin"
    out_los_m = output_dir / f"{prefix}_los_deformation_m.bin"
    out_los_mm = output_dir / f"{prefix}_los_deformation_mm.bin"
    unwrap_result = run_snaphu_unwrap(
        wrapped_phase=wrapped_phase,
        coherence=coherence,
        output_dir=output_dir,
        prefix=prefix,
        snaphu_bin=snaphu_bin,
        cost_mode=cost_mode,
        init_method=init_method,
        corr_thresh=corr_thresh,
        keep_temp=keep_temp,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_row_overlap=tile_row_overlap,
        tile_col_overlap=tile_col_overlap,
        nproc=effective_nproc,
    )

    unwrapped = np.asarray(unwrap_result["unwrapped_phase"], dtype=np.float32)
    los_m = phase_to_los_deformation(unwrapped, wavelength=float(wavelength))
    los_mm = (los_m * 1000.0).astype(np.float32, copy=False)
    los_m.tofile(out_los_m)
    los_mm.tofile(out_los_mm)

    _create_vrt(out_unw, unwrapped.shape, unwrapped.dtype)
    _create_vrt(out_los_m, los_m.shape, los_m.dtype)
    _create_vrt(out_los_mm, los_mm.shape, los_mm.dtype)

    result = {
        "unwrapped_phase": unwrapped,
        "los_deformation_m": los_m,
        "los_deformation_mm": los_mm,
        "unwrapped_phase_file": str(out_unw),
        "los_deformation_m_file": str(out_los_m),
        "los_deformation_mm_file": str(out_los_mm),
        "snaphu_log_file": str(unwrap_result["snaphu_log_file"]),
        "snaphu_cmd": list(unwrap_result["snaphu_cmd"]),
    }
    for key in ("snaphu_version", "snaphu_warnings", "snaphu_attempt_logs", "tile_plan"):
        if key in unwrap_result:
            result[key] = unwrap_result[key]
    return result


def _load_2d_float32(file_path: Path, width: int, height: Optional[int]) -> np.ndarray:
    data = np.fromfile(file_path, dtype=np.float32)
    if width <= 0:
        raise ValueError(f"width 必须 > 0, 当前={width}")

    if height is None:
        if data.size % width != 0:
            raise ValueError(
                f"无法从 width={width} 推断高度，文件元素数={data.size} 不能整除 width。"
            )
        height = data.size // width

    expected = int(width) * int(height)
    if data.size != expected:
        raise ValueError(f"文件尺寸不匹配: 期望元素数={expected}, 实际={data.size}, 文件={file_path}")
    return data.reshape(int(height), int(width)).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="snaphu 解缠并输出 LOS 形变")
    parser.add_argument("--wrapped-phase", required=True, help="包裹相位 bin 文件（float32）")
    parser.add_argument("--coherence", required=True, help="相干性 bin 文件（float32）")
    parser.add_argument("--width", type=int, required=True, help="影像宽度（列数）")
    parser.add_argument("--height", type=int, default=None, help="影像高度（行数），不传则自动推断")
    parser.add_argument("--output-dir", default=".", help="输出目录")
    parser.add_argument("--prefix", default="snaphu", help="输出文件前缀")
    parser.add_argument("--wavelength", type=float, default=0.0555, help="雷达波长 (m)")
    parser.add_argument("--snaphu-bin", default="snaphu", help="snaphu 可执行文件名/路径")
    parser.add_argument("--cost-mode", choices=["topo", "defo", "smooth"], default="defo", help="snaphu 代价模式")
    parser.add_argument("--init-method", choices=["mst", "mcf"], default="mst", help="snaphu 初始化方法")
    parser.add_argument("--corr-thresh", type=float, default=None, help="相干性阈值，低于阈值置零")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    parser.add_argument("--tile-rows", type=int, default=None, help="snaphu tile 行数（启用分块解缠）")
    parser.add_argument("--tile-cols", type=int, default=None, help="snaphu tile 列数（启用分块解缠）")
    parser.add_argument("--tile-row-overlap", type=int, default=512, help="snaphu tile 行重叠")
    parser.add_argument("--tile-col-overlap", type=int, default=512, help="snaphu tile 列重叠")
    parser.add_argument("--nproc", type=int, default=None, help="snaphu tile 模式并行进程数")
    args = parser.parse_args()

    wrapped_phase = _load_2d_float32(Path(args.wrapped_phase), width=args.width, height=args.height)
    coherence = _load_2d_float32(Path(args.coherence), width=args.width, height=args.height)

    result = run_unwrap_and_los(
        wrapped_phase=wrapped_phase,
        coherence=coherence,
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        wavelength=args.wavelength,
        snaphu_bin=args.snaphu_bin,
        cost_mode=args.cost_mode,
        init_method=args.init_method,
        corr_thresh=args.corr_thresh,
        keep_temp=args.keep_temp,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        tile_row_overlap=args.tile_row_overlap,
        tile_col_overlap=args.tile_col_overlap,
        nproc=args.nproc,
    )

    print("snaphu 解缠完成")
    print(f"  unwrapped_phase: {result['unwrapped_phase_file']}")
    print(f"  los_deformation_m: {result['los_deformation_m_file']}")
    print(f"  los_deformation_mm: {result['los_deformation_mm_file']}")
    if "snaphu_version" in result:
        print(f"  snaphu_version: {result['snaphu_version']}")
    if "snaphu_warnings" in result:
        for w in result["snaphu_warnings"]:
            print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
