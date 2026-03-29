#!/usr/bin/env python3
"""
偏移估计模块
功能：读取配准点，进行鲁棒多项式拟合，并写回统一的 offset_estimate.txt 格式。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def _parse_header_line(line: str) -> Dict[str, Any]:
    header: Dict[str, Any] = {}
    parts = line.strip().split()
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in ("WINDOW_SIZE", "SEARCH_RANGE"):
            parsed = _safe_float(value)
            if parsed is not None:
                header[key.lower()] = int(round(parsed))
        elif key == "METHOD":
            header["method"] = value
    return header


def _parse_point_line(line: str) -> Optional[Dict[str, float]]:
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    values: List[float] = []
    for token in parts[:5]:
        parsed = _safe_float(token)
        if parsed is None:
            return None
        values.append(parsed)

    # 兼容 x dx y dy [corr] 格式
    if len(values) < 4:
        return None
    corr = values[4] if len(values) >= 5 else 0.0
    return {
        "col": values[0],       # x
        "range": values[1],     # dx
        "row": values[2],       # y
        "azimuth": values[3],   # dy
        "correlation": corr,
    }


def _read_offset_file_with_metadata(filename: str) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    points: List[Dict[str, float]] = []
    metadata: Dict[str, Any] = {
        "window_size": 256,
        "search_range": 64,
        "method": "unknown",
    }
    if not os.path.exists(filename):
        return points, metadata

    with open(filename, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("WINDOW_SIZE="):
                metadata.update(_parse_header_line(line))
                continue
            if upper in {"FITTING_FORMULA", "PARAMETERS", "NORMALIZATION", "REGISTRATION_QUALITY"}:
                # 点列表在此处之前结束
                break
            parsed = _parse_point_line(line)
            if parsed is not None:
                points.append(parsed)
    return points, metadata


def read_offsets_from_file(filename: str) -> List[Dict[str, float]]:
    points, _ = _read_offset_file_with_metadata(filename)
    return points


def _read_fine_offsets(filename: str = "offset_estimate.txt") -> List[Dict[str, float]]:
    return read_offsets_from_file(filename)


def _build_design_matrix(rows_norm: np.ndarray, cols_norm: np.ndarray, polynomial_order: int) -> np.ndarray:
    if polynomial_order <= 1:
        return np.column_stack(
            [
                np.ones_like(rows_norm),
                rows_norm,
                cols_norm,
            ]
        )
    return np.column_stack(
        [
            np.ones_like(rows_norm),
            rows_norm,
            cols_norm,
            rows_norm * cols_norm,
            rows_norm * rows_norm,
            cols_norm * cols_norm,
        ]
    )


def _fit_once(
    rows: np.ndarray,
    cols: np.ndarray,
    az: np.ndarray,
    rg: np.ndarray,
    polynomial_order: int,
) -> Dict[str, Any]:
    rows_mean = float(np.mean(rows))
    cols_mean = float(np.mean(cols))
    rows_std = float(np.std(rows))
    cols_std = float(np.std(cols))
    rows_std = rows_std if rows_std > 1e-8 else 1.0
    cols_std = cols_std if cols_std > 1e-8 else 1.0

    rows_norm = (rows - rows_mean) / rows_std
    cols_norm = (cols - cols_mean) / cols_std
    A = _build_design_matrix(rows_norm, cols_norm, polynomial_order)

    coeff_az, _, _, _ = np.linalg.lstsq(A, az, rcond=None)
    coeff_rg, _, _, _ = np.linalg.lstsq(A, rg, rcond=None)

    pred_az = A @ coeff_az
    pred_rg = A @ coeff_rg
    res_az = az - pred_az
    res_rg = rg - pred_rg

    return {
        "coeff_az": coeff_az,
        "coeff_rg": coeff_rg,
        "rows_mean": rows_mean,
        "rows_std": rows_std,
        "cols_mean": cols_mean,
        "cols_std": cols_std,
        "pred_az": pred_az,
        "pred_rg": pred_rg,
        "res_az": res_az,
        "res_rg": res_rg,
    }


def _coeff_to_params(coeff_az: np.ndarray, coeff_rg: np.ndarray, polynomial_order: int) -> Dict[str, float]:
    coeff_az = np.asarray(coeff_az, dtype=np.float64)
    coeff_rg = np.asarray(coeff_rg, dtype=np.float64)
    if polynomial_order <= 1:
        coeff_az = np.pad(coeff_az, (0, max(0, 6 - coeff_az.size)))
        coeff_rg = np.pad(coeff_rg, (0, max(0, 6 - coeff_rg.size)))
    params = {
        "a0": float(coeff_az[0]),
        "a1": float(coeff_az[1]),
        "a2": float(coeff_az[2]),
        "a3": float(coeff_az[3]),
        "a4": float(coeff_az[4]),
        "a5": float(coeff_az[5]),
        "b0": float(coeff_rg[0]),
        "b1": float(coeff_rg[1]),
        "b2": float(coeff_rg[2]),
        "b3": float(coeff_rg[3]),
        "b4": float(coeff_rg[4]),
        "b5": float(coeff_rg[5]),
    }
    return params


class IterativeOffsetEstimator:
    """与旧接口兼容的迭代偏移估计器封装。"""

    def __init__(
        self,
        polynomial_order: int = 2,
        max_iterations: int = 10,
        initial_outlier_threshold: float = 2.0,
    ):
        self.polynomial_order = polynomial_order
        self.max_iterations = max_iterations
        self.initial_outlier_threshold = initial_outlier_threshold

    def fit(
        self,
        offsets: List[Dict[str, float]],
        use_segmentation: bool = False,
        num_segments: int = 5,
    ) -> Optional[Dict[str, Any]]:
        return estimate_offsets(
            offsets=offsets,
            polynomial_order=self.polynomial_order,
            max_iterations=self.max_iterations,
            initial_outlier_threshold=self.initial_outlier_threshold,
            use_segmentation=use_segmentation,
            num_segments=num_segments,
            output_file=None,
        )


def estimate_offsets(
    offsets: Optional[List[Dict[str, float]]] = None,
    polynomial_order: int = 2,
    max_iterations: int = 10,
    initial_outlier_threshold: float = 2.0,
    use_segmentation: bool = False,
    num_segments: int = 5,
    input_file: str = "offset_estimate.txt",
    output_file: Optional[str] = None,
    min_correlation: float = 0.0,
) -> Optional[Dict[str, Any]]:
    del use_segmentation
    del num_segments

    metadata: Dict[str, Any]
    if offsets is None:
        offsets, metadata = _read_offset_file_with_metadata(input_file)
    else:
        _, metadata = _read_offset_file_with_metadata(input_file)

    if not offsets:
        return None

    arr_cols = np.array([p["col"] for p in offsets], dtype=np.float64)
    arr_rows = np.array([p["row"] for p in offsets], dtype=np.float64)
    arr_rg = np.array([p["range"] for p in offsets], dtype=np.float64)
    arr_az = np.array([p["azimuth"] for p in offsets], dtype=np.float64)
    arr_corr = np.array([p.get("correlation", 0.0) for p in offsets], dtype=np.float64)

    valid_mask = np.isfinite(arr_cols) & np.isfinite(arr_rows) & np.isfinite(arr_rg) & np.isfinite(arr_az)
    valid_mask &= arr_corr >= float(min_correlation)
    if np.count_nonzero(valid_mask) < 3:
        return None

    arr_cols = arr_cols[valid_mask]
    arr_rows = arr_rows[valid_mask]
    arr_rg = arr_rg[valid_mask]
    arr_az = arr_az[valid_mask]
    arr_corr = arr_corr[valid_mask]

    polynomial_order = 2 if polynomial_order >= 2 else 1
    min_points = 6 if polynomial_order == 2 else 3
    if arr_rows.size < min_points:
        polynomial_order = 1
        min_points = 3
    if arr_rows.size < min_points:
        return None

    inlier_mask = np.ones(arr_rows.size, dtype=bool)
    fit: Optional[Dict[str, Any]] = None
    for _ in range(max(1, int(max_iterations))):
        if np.count_nonzero(inlier_mask) < min_points:
            break
        fit = _fit_once(
            arr_rows[inlier_mask],
            arr_cols[inlier_mask],
            arr_az[inlier_mask],
            arr_rg[inlier_mask],
            polynomial_order,
        )

        # 使用当前模型在全体点上评估残差，做离群点剔除
        rows_mean = fit["rows_mean"]
        rows_std = fit["rows_std"]
        cols_mean = fit["cols_mean"]
        cols_std = fit["cols_std"]
        rows_norm_all = (arr_rows - rows_mean) / rows_std
        cols_norm_all = (arr_cols - cols_mean) / cols_std
        A_all = _build_design_matrix(rows_norm_all, cols_norm_all, polynomial_order)
        pred_az_all = A_all @ fit["coeff_az"]
        pred_rg_all = A_all @ fit["coeff_rg"]
        res_az_all = arr_az - pred_az_all
        res_rg_all = arr_rg - pred_rg_all

        az_sigma = float(np.std(res_az_all[inlier_mask])) if np.count_nonzero(inlier_mask) > 1 else 0.0
        rg_sigma = float(np.std(res_rg_all[inlier_mask])) if np.count_nonzero(inlier_mask) > 1 else 0.0
        az_sigma = max(az_sigma, 1e-6)
        rg_sigma = max(rg_sigma, 1e-6)
        az_thr = float(initial_outlier_threshold) * az_sigma
        rg_thr = float(initial_outlier_threshold) * rg_sigma

        new_mask = (np.abs(res_az_all) <= az_thr) & (np.abs(res_rg_all) <= rg_thr)
        if np.count_nonzero(new_mask) < min_points:
            break
        if np.array_equal(new_mask, inlier_mask):
            inlier_mask = new_mask
            break
        inlier_mask = new_mask

    if fit is None:
        fit = _fit_once(arr_rows, arr_cols, arr_az, arr_rg, polynomial_order)
        inlier_mask = np.ones(arr_rows.size, dtype=bool)

    # 最终模型：使用最终 inlier 重新拟合
    fit = _fit_once(
        arr_rows[inlier_mask],
        arr_cols[inlier_mask],
        arr_az[inlier_mask],
        arr_rg[inlier_mask],
        polynomial_order,
    )

    params = _coeff_to_params(fit["coeff_az"], fit["coeff_rg"], polynomial_order)
    az_rms = float(np.sqrt(np.mean(fit["res_az"] ** 2))) if fit["res_az"].size else 0.0
    rg_rms = float(np.sqrt(np.mean(fit["res_rg"] ** 2))) if fit["res_rg"].size else 0.0

    inlier_indices = np.where(inlier_mask)[0]
    inlier_points: List[Dict[str, float]] = []
    for idx in inlier_indices:
        inlier_points.append(
            {
                "col": float(arr_cols[idx]),
                "row": float(arr_rows[idx]),
                "range": float(arr_rg[idx]),
                "azimuth": float(arr_az[idx]),
                "correlation": float(arr_corr[idx]),
            }
        )

    result: Dict[str, Any] = {
        "polynomial_order": int(polynomial_order),
        "initial_points": int(arr_rows.size),
        "final_points": int(np.count_nonzero(inlier_mask)),
        "outliers_removed": int(arr_rows.size - np.count_nonzero(inlier_mask)),
        "parameters": params,
        "fitted_params": params.copy(),
        "normalization": {
            "rows_mean": float(fit["rows_mean"]),
            "rows_std": float(fit["rows_std"]),
            "cols_mean": float(fit["cols_mean"]),
            "cols_std": float(fit["cols_std"]),
        },
        "rows_mean": float(fit["rows_mean"]),
        "rows_std": float(fit["rows_std"]),
        "cols_mean": float(fit["cols_mean"]),
        "cols_std": float(fit["cols_std"]),
        "final_residuals": {
            "azimuth_rms": az_rms,
            "range_rms": rg_rms,
        },
        "azimuth_rms": az_rms,
        "range_rms": rg_rms,
        "points": inlier_points,
        "metadata": metadata,
    }

    if output_file is None and offsets is not None:
        output_file = None
    elif output_file is None:
        output_file = input_file
    if output_file:
        write_offset_estimate_result(result, output_file)
    return result


def write_offset_estimate_result(result: Dict[str, Any], output_file: str) -> None:
    metadata = result.get("metadata", {})
    window_size = int(metadata.get("window_size", 256))
    search_range = int(metadata.get("search_range", 64))
    method = str(metadata.get("method", "fine_fit"))
    if not method:
        method = "fine_fit"

    points = result.get("points", [])
    params = result.get("parameters", {})
    norm = result.get("normalization", {})
    fitted_params = result.get("fitted_params", {})
    polynomial_order = int(result.get("polynomial_order", 2))
    az_rms = float(result.get("azimuth_rms", result.get("final_residuals", {}).get("azimuth_rms", 0.0)))
    rg_rms = float(result.get("range_rms", result.get("final_residuals", {}).get("range_rms", 0.0)))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"WINDOW_SIZE={window_size} SEARCH_RANGE={search_range} METHOD={method}\n")
        for p in points:
            f.write(
                f"{int(round(p['col'])):8d} {float(p['range']):8.3f} "
                f"{int(round(p['row'])):8d} {float(p['azimuth']):8.3f} {float(p.get('correlation', 0.0)):8.2f}\n"
            )

        f.write("FITTING_FORMULA\n")
        f.write(f"polynomial_order: {polynomial_order}\n")
        f.write(f"rows_mean: {float(norm.get('rows_mean', result.get('rows_mean', 0.0))):.10f}\n")
        f.write(f"rows_std: {float(norm.get('rows_std', result.get('rows_std', 1.0))):.10f}\n")
        f.write(f"cols_mean: {float(norm.get('cols_mean', result.get('cols_mean', 0.0))):.10f}\n")
        f.write(f"cols_std: {float(norm.get('cols_std', result.get('cols_std', 1.0))):.10f}\n")
        f.write("fitted_params:\n")
        for key in ("a0", "a1", "a2", "a3", "a4", "a5", "b0", "b1", "b2", "b3", "b4", "b5"):
            value = fitted_params.get(key, params.get(key, 0.0))
            f.write(f"{key}: {float(value):.12e}\n")

        f.write("PARAMETERS\n")
        for key in ("a0", "a1", "a2", "a3", "a4", "a5", "b0", "b1", "b2", "b3", "b4", "b5"):
            f.write(f"{key}={float(params.get(key, 0.0)):.12e}\n")

        f.write("NORMALIZATION\n")
        f.write(f"rows_mean={float(norm.get('rows_mean', 0.0)):.12e}\n")
        f.write(f"rows_std={float(norm.get('rows_std', 1.0)):.12e}\n")
        f.write(f"cols_mean={float(norm.get('cols_mean', 0.0)):.12e}\n")
        f.write(f"cols_std={float(norm.get('cols_std', 1.0)):.12e}\n")

        f.write("REGISTRATION_QUALITY\n")
        f.write(f"azimuth_rms: {az_rms:.10f}\n")
        f.write(f"range_rms: {rg_rms:.10f}\n")
        f.write(f"num_points: {int(result.get('final_points', len(points)))}\n")
