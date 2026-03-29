#!/usr/bin/env python3
"""
snaphu 解缠稳健执行工具。
"""

from __future__ import annotations

import math
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SNAPHU_MIN_STABLE_VERSION = (2, 0, 7)
_VERSION_PATTERN = re.compile(r"\b(?:v)?(\d+\.\d+\.\d+)\b")


def _ensure_snaphu_binary(snaphu_bin: str = "snaphu") -> str:
    exe = shutil.which(snaphu_bin)
    if exe is None:
        raise FileNotFoundError(
            f"未找到 snaphu 可执行文件: {snaphu_bin}. "
            "请确认已安装 snaphu 并在 PATH 中可见。"
        )
    return exe


def _parse_version_tuple(version_text: str) -> Optional[Tuple[int, int, int]]:
    candidates = []
    for m in _VERSION_PATTERN.finditer(version_text):
        try:
            candidates.append(tuple(int(x) for x in m.group(1).split(".")))
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates)


def _query_snaphu_version(exe: str) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    for args in ([exe, "--info"], [exe, "-h"], [exe, "--help"]):
        try:
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except Exception:
            continue
        text = proc.stdout or ""
        ver_tuple = _parse_version_tuple(text)
        if ver_tuple is not None:
            return ".".join(str(v) for v in ver_tuple), ver_tuple
    return None, None


def _tail_text_file(path: Path, max_lines: int = 120) -> str:
    if (not path.exists()) or (not path.is_file()):
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _validate_2d_float32(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须为二维数组，当前 shape={arr.shape}")
    return arr


def _sanitize_inputs(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray,
    corr_thresh: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrapped_phase = _validate_2d_float32(wrapped_phase, "wrapped_phase")
    coherence = _validate_2d_float32(coherence, "coherence")
    if wrapped_phase.shape != coherence.shape:
        raise ValueError(
            f"wrapped_phase 与 coherence 形状不一致: {wrapped_phase.shape} vs {coherence.shape}"
        )

    coh = np.where(np.isfinite(coherence), coherence, 0.0).astype(np.float32, copy=False)
    coh = np.clip(coh, 0.0, 1.0).astype(np.float32, copy=False)

    valid = np.isfinite(wrapped_phase) & np.isfinite(coherence)
    if corr_thresh is not None:
        thr = float(corr_thresh)
        if (thr < 0.0) or (thr > 1.0):
            raise ValueError(f"corr_thresh 必须在 [0,1]，当前={corr_thresh}")
        valid = valid & (coh >= thr)
        coh = np.where(coh >= thr, coh, 0.0).astype(np.float32, copy=False)

    phase = np.where(np.isfinite(wrapped_phase), wrapped_phase, 0.0).astype(np.float64, copy=False)
    # 官方文档要求 FLOAT_DATA 输入相位在 [0, 2*pi)。
    phase = np.mod(phase, 2.0 * np.pi).astype(np.float32, copy=False)
    phase = np.where(valid, phase, 0.0).astype(np.float32, copy=False)

    mask = valid.astype(np.uint8, copy=False)
    if int(mask.sum()) <= 0:
        raise ValueError("所有像素都被掩膜，无法执行 snaphu 解缠。")

    return phase, coh, mask


@dataclass(frozen=True)
class _TilePlan:
    enabled: bool
    rows: int = 1
    cols: int = 1
    row_overlap: int = 0
    col_overlap: int = 0
    auto_selected: bool = False


def _auto_tile_layout(height: int, width: int, max_pixels_per_tile: int) -> Tuple[int, int]:
    total = int(height) * int(width)
    if total <= int(max_pixels_per_tile):
        return 1, 1
    target = max(1, int(max_pixels_per_tile))
    ntiles = int(math.ceil(total / float(target)))
    aspect = float(height) / float(max(width, 1))
    rows = max(1, int(round(math.sqrt(ntiles * aspect))))
    cols = max(1, int(math.ceil(ntiles / float(rows))))
    while rows * cols < ntiles:
        rows += 1
    return rows, cols


def _normalize_overlap(
    tile_size: int,
    overlap: int,
    min_recommended: int,
) -> int:
    if tile_size <= 2:
        return 0
    ov = int(overlap)
    ov = max(0, ov)
    ov = min(ov, tile_size - 1)
    if ov < min_recommended:
        ov = min(min_recommended, tile_size - 1)
    return ov


def _build_tile_plan(
    height: int,
    width: int,
    tile_rows: Optional[int],
    tile_cols: Optional[int],
    tile_row_overlap: int,
    tile_col_overlap: int,
    auto_tile_max_pixels: int,
) -> _TilePlan:
    if (tile_rows is None) ^ (tile_cols is None):
        raise ValueError("tile_rows 与 tile_cols 需要同时设置")

    auto_selected = False
    rows, cols = 1, 1

    if (tile_rows is not None) and (tile_cols is not None):
        rows = int(tile_rows)
        cols = int(tile_cols)
    else:
        rows, cols = _auto_tile_layout(height, width, auto_tile_max_pixels)
        auto_selected = (rows > 1) or (cols > 1)

    if (rows <= 1) and (cols <= 1):
        return _TilePlan(enabled=False)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"非法 tile 参数: rows={rows}, cols={cols}")

    tile_h = int(math.ceil(height / float(rows)))
    tile_w = int(math.ceil(width / float(cols)))
    row_ov = _normalize_overlap(tile_h, tile_row_overlap, min_recommended=64 if rows > 1 else 0)
    col_ov = _normalize_overlap(tile_w, tile_col_overlap, min_recommended=64 if cols > 1 else 0)

    return _TilePlan(
        enabled=True,
        rows=rows,
        cols=cols,
        row_overlap=row_ov,
        col_overlap=col_ov,
        auto_selected=auto_selected,
    )


def _write_runtime_config(
    config_file: Path,
    *,
    cost_mode: str,
    init_method: str,
    keep_temp: bool,
) -> None:
    cost_mode_u = cost_mode.upper()
    init_method_u = init_method.upper()
    lines = [
        "# Auto-generated by utils/snaphu_utils.py",
        "INFILEFORMAT FLOAT_DATA",
        "CORRFILEFORMAT FLOAT_DATA",
        "OUTFILEFORMAT FLOAT_DATA",
        f"STATCOSTMODE {cost_mode_u}",
        f"INITMETHOD {init_method_u}",
        "VERBOSE TRUE",
        f"RMTMPTILE {'FALSE' if keep_temp else 'TRUE'}",
    ]
    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cost_flag(cost_mode: str) -> str:
    m = cost_mode.lower()
    if m == "topo":
        return "-t"
    if m == "defo":
        return "-d"
    if m == "smooth":
        return "-s"
    raise ValueError(f"不支持的 cost_mode: {cost_mode}")


def _init_flag(init_method: str) -> str:
    m = init_method.lower()
    if m == "mst":
        return "--mst"
    if m == "mcf":
        return "--mcf"
    raise ValueError(f"不支持的 init_method: {init_method}")


def _rc_to_text(return_code: int) -> str:
    if return_code >= 0:
        return str(return_code)
    sig = -int(return_code)
    try:
        return f"{return_code} (signal={signal.Signals(sig).name})"
    except Exception:
        return f"{return_code} (signal={sig})"


def _build_attempts(
    *,
    tile_plan: _TilePlan,
    nproc: Optional[int],
    init_method: str,
    has_mask: bool,
) -> List[Dict[str, object]]:
    attempts: List[Dict[str, object]] = []
    seen = set()

    def add(
        name: str,
        use_tile: bool,
        nproc_value: Optional[int],
        init: str,
        use_mask: bool,
    ) -> None:
        key = (use_tile, int(nproc_value) if nproc_value is not None else None, init.lower(), use_mask)
        if key in seen:
            return
        seen.add(key)
        attempts.append(
            {
                "name": name,
                "use_tile": use_tile,
                "nproc": nproc_value,
                "init_method": init.lower(),
                "use_mask": use_mask,
            }
        )

    base_nproc = int(nproc) if nproc is not None else None
    add("primary", tile_plan.enabled, base_nproc, init_method, has_mask)

    if tile_plan.enabled and (base_nproc is not None) and (base_nproc > 1):
        add("retry_nproc1", True, 1, init_method, has_mask)

    if init_method.lower() == "mcf":
        add("retry_mst", tile_plan.enabled, base_nproc, "mst", has_mask)
        if tile_plan.enabled and (base_nproc is not None) and (base_nproc > 1):
            add("retry_mst_nproc1", True, 1, "mst", has_mask)

    if tile_plan.enabled:
        add("retry_notile", False, None, init_method, has_mask)
        if init_method.lower() == "mcf":
            add("retry_notile_mst", False, None, "mst", has_mask)

    if has_mask:
        add("retry_nomask", tile_plan.enabled, base_nproc, init_method, False)
        if tile_plan.enabled and (base_nproc is not None) and (base_nproc > 1):
            add("retry_nomask_nproc1", True, 1, init_method, False)
        add("retry_nomask_notile", False, None, init_method, False)
        if init_method.lower() == "mcf":
            add("retry_nomask_mst", tile_plan.enabled, base_nproc, "mst", False)
            add("retry_nomask_notile_mst", False, None, "mst", False)

    return attempts


def run_snaphu_unwrap(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray,
    output_dir: Path | str,
    prefix: str,
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
    auto_tile_max_pixels: int = 4_000_000,
) -> Dict[str, object]:
    """
    运行 snaphu 解缠（仅输出解缠相位），包含多级稳定性回退策略。
    """
    phase, coh, bytemask = _sanitize_inputs(
        wrapped_phase=wrapped_phase,
        coherence=coherence,
        corr_thresh=corr_thresh,
    )
    h, w = phase.shape

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exe = _ensure_snaphu_binary(snaphu_bin)

    version_text, version_tuple = _query_snaphu_version(exe)
    warnings: List[str] = []
    if (version_tuple is not None) and (version_tuple < _SNAPHU_MIN_STABLE_VERSION):
        warnings.append(
            "检测到 snaphu 版本 "
            f"{version_text} (< 2.0.7)。官方 release notes 提到旧版本存在崩溃/死循环/内存相关问题，"
            "建议升级到 2.0.7 或更高版本。"
        )

    tile_plan = _build_tile_plan(
        height=h,
        width=w,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_row_overlap=int(tile_row_overlap),
        tile_col_overlap=int(tile_col_overlap),
        auto_tile_max_pixels=int(auto_tile_max_pixels),
    )
    if tile_plan.auto_selected:
        warnings.append(
            "未显式指定 tile 参数，已根据影像尺寸自动启用分块: "
            f"{tile_plan.rows}x{tile_plan.cols}, overlap=({tile_plan.row_overlap},{tile_plan.col_overlap})"
        )

    out_unw = output_dir / f"{prefix}_unwrapped_phase.bin"
    legacy_primary_log = output_dir / f"{prefix}_snaphu.log"

    tmp_ctx = None
    if keep_temp:
        tmp_dir = Path(tempfile.mkdtemp(dir=output_dir, prefix=f"{prefix}_snaphu_tmp_"))
    else:
        tmp_ctx = tempfile.TemporaryDirectory(dir=output_dir, prefix=f"{prefix}_snaphu_tmp_")
        tmp_dir = Path(tmp_ctx.name)

    attempt_logs: List[str] = []
    attempt_cmds: List[List[str]] = []
    failure_summaries: List[str] = []

    try:
        wrapped_file = tmp_dir / f"{prefix}_wrapped_phase.bin"
        corr_file = tmp_dir / f"{prefix}_coherence.bin"
        mask_file = tmp_dir / f"{prefix}_bytemask.bin"
        config_file = tmp_dir / f"{prefix}_snaphu.conf"

        phase.astype(np.float32, copy=False).tofile(wrapped_file)
        coh.astype(np.float32, copy=False).tofile(corr_file)
        bytemask.astype(np.uint8, copy=False).tofile(mask_file)
        _write_runtime_config(
            config_file=config_file,
            cost_mode=cost_mode,
            init_method=init_method,
            keep_temp=keep_temp,
        )

        has_mask = (int(bytemask.min()) == 0) and (int(bytemask.max()) == 1)
        attempts = _build_attempts(
            tile_plan=tile_plan,
            nproc=nproc,
            init_method=init_method,
            has_mask=has_mask,
        )

        for attempt in attempts:
            attempt_name = str(attempt["name"])
            use_tile = bool(attempt["use_tile"])
            attempt_nproc = attempt["nproc"]
            attempt_init = str(attempt["init_method"])
            use_mask = bool(attempt["use_mask"])

            if attempt_name == "primary":
                log_file = legacy_primary_log
            else:
                log_file = output_dir / f"{prefix}_snaphu_{attempt_name}.log"
            attempt_logs.append(str(log_file))

            tile_dir = output_dir / f"{prefix}_snaphu_tiles_{attempt_name}"
            cmd: List[str] = [
                exe,
                str(wrapped_file),
                str(w),
                "-o",
                str(out_unw),
                "-c",
                str(corr_file),
                "-f",
                str(config_file),
                _cost_flag(cost_mode),
                _init_flag(attempt_init),
            ]
            if use_mask:
                cmd.extend(["-M", str(mask_file)])
            if use_tile:
                cmd.extend(
                    [
                        "--tile",
                        str(tile_plan.rows),
                        str(tile_plan.cols),
                        str(tile_plan.row_overlap),
                        str(tile_plan.col_overlap),
                        "--tiledir",
                        str(tile_dir),
                    ]
                )
                if (attempt_nproc is not None) and (int(attempt_nproc) > 0):
                    cmd.extend(["--nproc", str(int(attempt_nproc))])

            attempt_cmds.append(cmd)

            if out_unw.exists():
                try:
                    out_unw.unlink()
                except OSError:
                    pass

            with open(log_file, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    # snaphu 在异常退出时可能向当前进程组发送终止信号；
                    # 将其放到独立会话，避免连带终止 Python 主流程，保证可进入回退重试。
                    start_new_session=True,
                )

            rc = int(proc.returncode)
            out_ok = out_unw.exists() and (out_unw.stat().st_size == (h * w * 4))
            if (rc == 0) and out_ok:
                unwrapped = np.fromfile(out_unw, dtype=np.float32).reshape(h, w)
                result: Dict[str, object] = {
                    "unwrapped_phase": unwrapped,
                    "unwrapped_phase_file": str(out_unw),
                    "snaphu_log_file": str(log_file),
                    "snaphu_cmd": cmd,
                    "snaphu_attempt_logs": attempt_logs,
                    "snaphu_attempt_cmds": attempt_cmds,
                    "tile_plan": {
                        "enabled": tile_plan.enabled,
                        "rows": tile_plan.rows,
                        "cols": tile_plan.cols,
                        "row_overlap": tile_plan.row_overlap,
                        "col_overlap": tile_plan.col_overlap,
                        "auto_selected": tile_plan.auto_selected,
                    },
                }
                if version_text is not None:
                    result["snaphu_version"] = version_text
                if warnings:
                    result["snaphu_warnings"] = warnings
                return result

            failure_summaries.append(
                f"[{attempt_name}] returncode={_rc_to_text(rc)}, log={log_file}\n"
                f"{_tail_text_file(log_file, max_lines=80)}"
            )

        msg = (
            "snaphu 执行失败（所有重试均失败）\n"
            f"输出文件: {out_unw}\n"
            f"最近日志: {attempt_logs[-1] if attempt_logs else legacy_primary_log}\n"
        )
        if warnings:
            msg += "警告:\n" + "\n".join(f"- {w}" for w in warnings) + "\n"
        msg += "失败详情:\n" + "\n\n".join(failure_summaries[-5:])
        raise RuntimeError(msg)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
