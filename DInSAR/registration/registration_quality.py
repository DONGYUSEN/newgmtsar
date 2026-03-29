#!/usr/bin/env python3
"""
配准质量评估模块
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from sar_utils import compute_correlation, compute_rmse, compute_snr  # noqa: E402


def _read_offset_estimation(filename: str = "offset_estimate.txt") -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not os.path.exists(filename):
        return data

    in_parameters = False
    in_normalization = False
    with open(filename, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line == "PARAMETERS":
                in_parameters = True
                in_normalization = False
                data["parameters"] = {}
                continue
            if line == "NORMALIZATION":
                in_parameters = False
                in_normalization = True
                data["normalization"] = {}
                continue
            if line == "REGISTRATION_QUALITY":
                in_parameters = False
                in_normalization = False
                continue

            if in_parameters and "=" in line:
                key, value = line.split("=", 1)
                try:
                    data["parameters"][key.strip()] = float(value.strip())
                except Exception:
                    pass
                continue
            if in_normalization and "=" in line:
                key, value = line.split("=", 1)
                try:
                    data["normalization"][key.strip()] = float(value.strip())
                except Exception:
                    pass
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in ("azimuth_rms", "range_rms", "rows_mean", "rows_std", "cols_mean", "cols_std"):
                    try:
                        data[key] = float(value)
                    except Exception:
                        pass
                elif key == "polynomial_order":
                    try:
                        data[key] = int(float(value))
                    except Exception:
                        pass
                elif key == "num_points":
                    try:
                        data[key] = int(float(value))
                    except Exception:
                        pass
    return data


class RegistrationQualityAssessment:
    def __init__(self, excellent_rms: float = 0.2, good_rms: float = 0.5, fair_rms: float = 1.0):
        self.excellent_rms = excellent_rms
        self.good_rms = good_rms
        self.fair_rms = fair_rms

    def assess_from_offsets(self, offset_data: Dict[str, Any]) -> Dict[str, Any]:
        az_rms = float(offset_data.get("azimuth_rms", 999.0))
        rg_rms = float(offset_data.get("range_rms", 999.0))
        total_rms = float(np.sqrt(az_rms * az_rms + rg_rms * rg_rms))
        quality = self._grade(total_rms)
        confidence = self._confidence(total_rms)
        return {
            "quality": quality,
            "confidence": confidence,
            "azimuth_rms": az_rms,
            "range_rms": rg_rms,
            "total_rms": total_rms,
            "num_points": int(offset_data.get("num_points", 0)),
            "details": {
                "excellent_rms_threshold": self.excellent_rms,
                "good_rms_threshold": self.good_rms,
                "fair_rms_threshold": self.fair_rms,
            },
        }

    def assess_from_images(self, reference: np.ndarray, registered: np.ndarray) -> Dict[str, Any]:
        ref = np.abs(reference).astype(np.float64)
        reg = np.abs(registered).astype(np.float64)

        corr = float(compute_correlation(ref, reg))
        snr_db = float(compute_snr(ref, reg))
        rmse = float(compute_rmse(ref, reg))

        # 使用 RMSE 做等级，相关系数和 SNR 做置信度加权
        quality = self._grade(rmse)
        confidence = min(
            1.0,
            max(0.0, 0.5 * (corr + 1.0) + 0.5 * (1.0 / (1.0 + rmse))),
        )
        return {
            "quality": quality,
            "confidence": confidence,
            "correlation": corr,
            "snr_db": snr_db,
            "rmse": rmse,
        }

    def _grade(self, rms: float) -> str:
        if rms <= self.excellent_rms:
            return "excellent"
        if rms <= self.good_rms:
            return "good"
        if rms <= self.fair_rms:
            return "fair"
        return "poor"

    @staticmethod
    def _confidence(rms: float) -> float:
        return float(np.exp(-rms))


def assess_registration_quality(
    reference: Optional[np.ndarray] = None,
    registered: Optional[np.ndarray] = None,
    from_file: bool = False,
    offset_file: str = "offset_estimate.txt",
) -> Optional[Dict[str, Any]]:
    assessor = RegistrationQualityAssessment()
    if from_file:
        data = _read_offset_estimation(offset_file)
        if not data:
            return None
        return assessor.assess_from_offsets(data)

    if reference is None or registered is None:
        return None
    return assessor.assess_from_images(reference, registered)


def write_quality_result(result: Dict[str, Any], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("REGISTRATION_QUALITY\n")
        for key in ("quality", "confidence", "azimuth_rms", "range_rms", "total_rms", "correlation", "snr_db", "rmse"):
            if key in result:
                f.write(f"{key}: {result[key]}\n")
        if "details" in result and isinstance(result["details"], dict):
            f.write("DETAILS\n")
            for key, value in result["details"].items():
                f.write(f"{key}: {value}\n")

