#!/usr/bin/env python3
"""
粗配准多窗口评估脚本
- 针对真实SAR数据批量测试(window_size, search_range)
- 输出统计表和汇报图件
- 给出推荐窗口参数
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import shift as nd_shift

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from coarse_regist import CoarseRegistration
from sar_utils import read_image


@dataclass
class WindowResult:
    window_size: int
    search_range: int
    runtime_sec: float
    azimuth_offset: float
    range_offset: float
    average_correlation: float
    azimuth_std: float
    range_std: float
    consistency: float
    num_windows: int
    window_offsets: List[Tuple[float, float]]
    correlations: List[float]
    score: float = 0.0


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def parse_configs(config_text: str) -> List[Tuple[int, int]]:
    configs: List[Tuple[int, int]] = []
    for item in config_text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            ws_str, sr_str = item.split(":", 1)
            ws = int(ws_str)
            sr = int(sr_str)
        else:
            ws = int(item)
            sr = ws // 4
        configs.append((ws, sr))
    if not configs:
        raise ValueError("No valid window configs")
    return configs


def evaluate_configs(master: np.ndarray, slave: np.ndarray, configs: List[Tuple[int, int]], corr_thr: float) -> List[WindowResult]:
    results: List[WindowResult] = []

    for ws, sr in configs:
        t0 = time.perf_counter()
        registrar = CoarseRegistration(window_size=ws, search_range=sr, correlation_threshold=corr_thr)
        res = registrar.register(master, slave, output_file=None)
        dt = time.perf_counter() - t0

        offsets = np.array(res.get("window_offsets", []), dtype=np.float64)
        cors = np.array(res.get("correlations", []), dtype=np.float64)
        if offsets.size == 0:
            az_std = np.nan
            rg_std = np.nan
        else:
            az_std = float(np.std(offsets[:, 0])) if offsets.shape[0] > 1 else 0.0
            rg_std = float(np.std(offsets[:, 1])) if offsets.shape[0] > 1 else 0.0

        results.append(
            WindowResult(
                window_size=ws,
                search_range=sr,
                runtime_sec=float(dt),
                azimuth_offset=float(res.get("azimuth_offset", np.nan)),
                range_offset=float(res.get("range_offset", np.nan)),
                average_correlation=float(res.get("average_correlation", np.nan)),
                azimuth_std=az_std,
                range_std=rg_std,
                consistency=float(az_std + rg_std) if np.isfinite(az_std) and np.isfinite(rg_std) else np.nan,
                num_windows=int(len(res.get("window_offsets", []))),
                window_offsets=[(float(a), float(r)) for a, r in res.get("window_offsets", [])],
                correlations=[float(c) for c in res.get("correlations", [])],
            )
        )

    # 评分：高相关、低一致性误差、低耗时
    avg_corr = np.array([r.average_correlation for r in results], dtype=np.float64)
    consistency = np.array([r.consistency for r in results], dtype=np.float64)
    runtime = np.array([r.runtime_sec for r in results], dtype=np.float64)

    corr_n = minmax_norm(avg_corr)
    cons_n = minmax_norm(consistency)
    time_n = minmax_norm(runtime)

    score = 0.55 * corr_n + 0.35 * (1.0 - cons_n) + 0.10 * (1.0 - time_n)

    for i, r in enumerate(results):
        r.score = float(score[i])

    return results


def choose_recommended(results: List[WindowResult]) -> Tuple[WindowResult, WindowResult]:
    # 最高综合得分
    best = max(results, key=lambda x: x.score)

    # 工程推荐：在接近最优性能的前提下选最小窗口
    near_best = [
        r for r in results
        if r.average_correlation >= 0.97 * best.average_correlation
        and r.score >= 0.95 * best.score
        and np.isfinite(r.consistency)
    ]
    if not near_best:
        recommended = best
    else:
        near_best = sorted(near_best, key=lambda x: (x.window_size, -x.score))
        recommended = near_best[0]

    return best, recommended


def save_plots(results: List[WindowResult], out_dir: Path, title_prefix: str) -> None:
    labels = [f"{r.window_size}:{r.search_range}" for r in results]
    x = np.arange(len(results))

    avg_corr = np.array([r.average_correlation for r in results])
    consistency = np.array([r.consistency for r in results])
    runtime = np.array([r.runtime_sec for r in results])
    score = np.array([r.score for r in results])

    # 图1：总览
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=180)

    axs[0, 0].bar(x, avg_corr, color="#1f77b4")
    axs[0, 0].set_title("Average Correlation")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, rotation=30, ha="right")

    axs[0, 1].bar(x, consistency, color="#ff7f0e")
    axs[0, 1].set_title("Offset Consistency (az_std + rg_std)")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels, rotation=30, ha="right")

    axs[1, 0].bar(x, runtime, color="#2ca02c")
    axs[1, 0].set_title("Runtime (s)")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels, rotation=30, ha="right")

    axs[1, 1].bar(x, score, color="#9467bd")
    axs[1, 1].set_title("Composite Score")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels, rotation=30, ha="right")

    fig.suptitle(f"{title_prefix} - Coarse Registration Window Sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "window_sweep_overview.png")
    plt.close(fig)

    # 图2：四窗口偏移散点
    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows), dpi=180)
    axs = np.array(axs).reshape(rows, cols)

    for i, r in enumerate(results):
        rr = i // cols
        cc = i % cols
        ax = axs[rr, cc]
        if r.window_offsets:
            offs = np.array(r.window_offsets)
            az = offs[:, 0]
            rg = offs[:, 1]
            ax.scatter(rg, az, c=np.array(r.correlations), cmap="viridis", s=60, edgecolor="k")
            ax.scatter([r.range_offset], [r.azimuth_offset], marker="*", s=180, c="red", label="mean")
            ax.axvline(r.range_offset, ls="--", c="gray", lw=0.8)
            ax.axhline(r.azimuth_offset, ls="--", c="gray", lw=0.8)
            ax.set_title(f"W={r.window_size}, S={r.search_range}\nC={r.average_correlation:.2f}, Cons={r.consistency:.2f}")
            ax.set_xlabel("Range Offset (px)")
            ax.set_ylabel("Azimuth Offset (px)")
        else:
            ax.set_title(f"W={r.window_size}, S={r.search_range} (No points)")

    for j in range(n, rows * cols):
        rr = j // cols
        cc = j % cols
        axs[rr, cc].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "window_offsets_scatter.png")
    plt.close(fig)

    # 图3：各窗口相关系数箱线图
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    data = [r.correlations if len(r.correlations) > 0 else [np.nan] for r in results]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title("Per-window Correlation Distribution")
    ax.set_ylabel("Correlation")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "window_correlation_boxplot.png")
    plt.close(fig)


def save_effect_plot(
    results: List[WindowResult],
    master: np.ndarray,
    slave: np.ndarray,
    out_dir: Path,
    selected_windows: Tuple[int, ...] = (128, 256, 384, 512),
) -> None:
    master_amp = np.abs(master).astype(np.float64)
    slave_amp = np.abs(slave).astype(np.float64)

    h = min(master_amp.shape[0], slave_amp.shape[0])
    w = min(master_amp.shape[1], slave_amp.shape[1])
    master_amp = master_amp[:h, :w]
    slave_amp = slave_amp[:h, :w]

    roi_h = min(900, h)
    roi_w = min(900, w)
    r0 = max(0, h // 2 - roi_h // 2)
    c0 = max(0, w // 2 - roi_w // 2)
    r1 = r0 + roi_h
    c1 = c0 + roi_w
    m_roi = master_amp[r0:r1, c0:c1]

    selected = [r for r in results if r.window_size in set(selected_windows)]
    selected = sorted(selected, key=lambda x: x.window_size)
    if not selected:
        return

    n = len(selected)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), dpi=180)
    axs = np.array(axs).reshape(rows, cols).ravel()

    for i, r in enumerate(selected):
        ax = axs[i]
        # 使用粗配准估计的平均偏移对slave幅度图平移，展示配准后残差
        s_aligned = nd_shift(slave_amp, shift=(r.azimuth_offset, r.range_offset), order=1, mode="nearest", prefilter=False)
        s_roi = s_aligned[r0:r1, c0:c1]
        diff = np.abs(m_roi - s_roi)
        v = np.nanpercentile(diff, 98)
        mad = float(np.nanmean(diff))
        im = ax.imshow(diff, cmap="magma", vmin=0, vmax=v)
        ax.set_title(f"W={r.window_size}, S={r.search_range} | mean|Δ|={mad:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, rows * cols):
        axs[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "window_registration_effect.png")
    plt.close(fig)


def write_markdown_report(results: List[WindowResult], best: WindowResult, recommended: WindowResult, out_dir: Path, dataset_tag: str) -> None:
    lines: List[str] = []
    lines.append(f"# 粗配准多窗口评估报告（{dataset_tag}）")
    lines.append("")
    lines.append("## 结论")
    lines.append(f"- 最高综合得分窗口：`window_size={best.window_size}, search_range={best.search_range}`")
    lines.append(f"- 工程推荐窗口：`window_size={recommended.window_size}, search_range={recommended.search_range}`")
    lines.append("- 推荐原则：在接近最优性能条件下优先选择更小窗口，以减少计算开销并保持稳定性。")
    lines.append("")
    lines.append("## 指标表")
    lines.append("")
    lines.append("| window_size | search_range | avg_corr | az_offset | rg_offset | az_std | rg_std | consistency | runtime_s | score |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.window_size} | {r.search_range} | {r.average_correlation:.3f} | {r.azimuth_offset:.3f} | {r.range_offset:.3f} | {r.azimuth_std:.3f} | {r.range_std:.3f} | {r.consistency:.3f} | {r.runtime_sec:.3f} | {r.score:.3f} |"
        )

    lines.append("")
    lines.append("## 图件")
    lines.append("- `window_sweep_overview.png`：相关性/一致性/耗时/综合评分总览")
    lines.append("- `window_offsets_scatter.png`：不同窗口下4个子窗口偏移离散情况")
    lines.append("- `window_correlation_boxplot.png`：不同窗口下相关系数分布")
    lines.append("- `window_registration_effect.png`：不同窗口参数下配准后残差图（同一ROI）")
    lines.append("")
    lines.append("## 备注")
    lines.append("- 本报告使用真实SAR数据运行得到，可用于项目汇报中的“粗配准窗口优化”页面。")

    (out_dir / "coarse_window_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coarse registration under multiple windows")
    parser.add_argument("--master", required=True, help="master image path (.vrt/.tif)")
    parser.add_argument("--slave", required=True, help="slave image path (.vrt/.tif)")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--configs", default="128:32,192:48,256:64,384:96,512:128,640:160,768:192",
                        help="comma-separated window configs. format ws:sr or ws")
    parser.add_argument("--corr-threshold", type=float, default=0.0, help="coarse registration correlation threshold")
    parser.add_argument("--dataset-tag", default="LT1B Tianyi-like real SAR", help="dataset tag for report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading images...")
    master = read_image(args.master)
    slave = read_image(args.slave)
    if master is None or slave is None:
        raise RuntimeError("Failed to load master/slave images")

    print(f"Master shape: {master.shape}, Slave shape: {slave.shape}")
    configs = parse_configs(args.configs)
    print(f"Testing configs: {configs}")

    results = evaluate_configs(master, slave, configs, corr_thr=args.corr_threshold)
    best, recommended = choose_recommended(results)

    save_plots(results, out_dir, args.dataset_tag)
    save_effect_plot(results, master, slave, out_dir)

    json_data = {
        "dataset_tag": args.dataset_tag,
        "master": args.master,
        "slave": args.slave,
        "best": asdict(best),
        "recommended": asdict(recommended),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "coarse_window_metrics.json").write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    write_markdown_report(results, best, recommended, out_dir, args.dataset_tag)

    print("\n=== DONE ===")
    print(f"best: window={best.window_size}, search={best.search_range}, score={best.score:.3f}")
    print(f"recommended: window={recommended.window_size}, search={recommended.search_range}, score={recommended.score:.3f}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
