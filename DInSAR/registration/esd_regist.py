#!/usr/bin/env python3
"""
Sentinel-1 TOPS 严格版 ESD（burst overlap + annotation XML）残余方位向配准估计。

实现要点：
1) 从 master/slave YAML 读取 annotation XML 来源（zip + member）。
2) 解析 burstList、linesPerBurst、azimuthTimeInterval、dcEstimateList。
3) 在相邻 burst overlap 区域分别构建干涉量，计算 ESD 相位差。
4) 使用 Doppler centroid 差值将相位差转换为残余方位向偏移（像素）。
5) 多 overlap 对结果做鲁棒融合，输出全局残余方位向修正。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from osgeo import gdal


C_LIGHT = 299792458.0


@dataclass
class DcEstimate:
    az_time: dt.datetime
    t0: float
    coeff: np.ndarray


@dataclass
class BurstInfo:
    az_time: dt.datetime
    first_valid: np.ndarray
    last_valid: np.ndarray


@dataclass
class AnnotationInfo:
    first_line_time: dt.datetime
    az_dt: float
    range_spacing: float
    near_range: float
    lines_per_burst: int
    samples_per_burst: int
    bursts: List[BurstInfo]
    dc_list: List[DcEstimate]


@dataclass
class PairResult:
    pair_index: int
    overlap_lines_full: int
    overlap_lines_ml: int
    phase: float
    delta_fd_hz: float
    az_offset_ml_px: float
    pair_coherence: float
    esd_quality: float


def _parse_iso8601(s: str) -> dt.datetime:
    t = str(s).strip()
    if t.startswith("UTC="):
        t = t[4:]
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    ts = dt.datetime.fromisoformat(t)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _ftext(root: ET.Element, path: str, default: Optional[str] = None) -> Optional[str]:
    n = root.find(path)
    if n is None or n.text is None:
        return default
    return n.text.strip()


def _parse_int_list(txt: str, expected_len: int) -> np.ndarray:
    vals = [int(x) for x in str(txt).replace(",", " ").split()]
    if len(vals) < expected_len:
        vals = vals + ([-1] * (expected_len - len(vals)))
    if len(vals) > expected_len:
        vals = vals[:expected_len]
    return np.asarray(vals, dtype=np.int32)


def _parse_float_list(txt: str) -> np.ndarray:
    vals = [float(x) for x in str(txt).replace(",", " ").split()]
    return np.asarray(vals, dtype=np.float64)


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_annotation_xml_from_yaml(yaml_path: Path) -> str:
    cfg = _load_yaml(yaml_path)
    meta = cfg.get("metadata", {}) or {}

    src_zip = meta.get("source_zip")
    ann_member = meta.get("annotation_member")
    if src_zip and ann_member:
        z = Path(str(src_zip)).expanduser().resolve()
        if not z.exists():
            raise FileNotFoundError(f"source_zip 不存在: {z} (yaml={yaml_path})")
        with zipfile.ZipFile(z, "r") as zf:
            return zf.read(str(ann_member)).decode("utf-8")

    # 兼容手动传本地 XML
    ann_path = meta.get("annotation_xml")
    if ann_path:
        p = Path(str(ann_path)).expanduser().resolve()
        return p.read_text(encoding="utf-8")

    raise RuntimeError(
        f"YAML 缺少 annotation 源信息（metadata.source_zip + metadata.annotation_member）: {yaml_path}"
    )


def _multilook_factors(yaml_path: Path) -> Tuple[int, int]:
    cfg = _load_yaml(yaml_path)
    ml = (cfg.get("processing_parameters", {}) or {}).get("multilook", {}) or {}
    nalks = int(ml.get("nalks", 1))
    nrlks = int(ml.get("nrlks", 1))
    return max(nalks, 1), max(nrlks, 1)


def _parse_annotation(xml_text: str) -> AnnotationInfo:
    root = ET.fromstring(xml_text)
    az_dt = float(_ftext(root, ".//imageInformation/azimuthTimeInterval", "0"))
    if az_dt <= 0:
        raise ValueError("annotation 缺少有效 azimuthTimeInterval")
    range_spacing = float(_ftext(root, ".//imageInformation/rangePixelSpacing", "0"))
    slant_t = float(_ftext(root, ".//imageInformation/slantRangeTime", "0"))
    near_range = float(slant_t * C_LIGHT / 2.0)
    lines_per_burst = int(float(_ftext(root, ".//swathTiming/linesPerBurst", "0")))
    samples_per_burst = int(float(_ftext(root, ".//swathTiming/samplesPerBurst", "0")))
    if lines_per_burst <= 0 or samples_per_burst <= 0:
        raise ValueError("annotation 缺少有效 linesPerBurst/samplesPerBurst")

    first_line = _ftext(root, ".//imageInformation/productFirstLineUtcTime")
    if first_line is None:
        first_line = _ftext(root, ".//downlinkInformation/firstLineSensingTime")
    if first_line is None:
        raise ValueError("annotation 缺少 first line sensing time")
    first_line_time = _parse_iso8601(first_line)

    bursts: List[BurstInfo] = []
    bl = root.find(".//swathTiming/burstList")
    if bl is None:
        raise ValueError("annotation 缺少 swathTiming/burstList")
    for b in bl.findall("./burst"):
        az_time_txt = _ftext(b, "./azimuthTime")
        fvs_txt = _ftext(b, "./firstValidSample")
        lvs_txt = _ftext(b, "./lastValidSample")
        if az_time_txt is None or fvs_txt is None or lvs_txt is None:
            continue
        bursts.append(
            BurstInfo(
                az_time=_parse_iso8601(az_time_txt),
                first_valid=_parse_int_list(fvs_txt, lines_per_burst),
                last_valid=_parse_int_list(lvs_txt, lines_per_burst),
            )
        )
    if len(bursts) < 2:
        raise ValueError("burst 数量不足（<2），无法做 overlap ESD")

    dc_list: List[DcEstimate] = []
    dcl = root.find(".//dopplerCentroid/dcEstimateList")
    if dcl is not None:
        for d in dcl.findall("./dcEstimate"):
            az_txt = _ftext(d, "./azimuthTime")
            t0_txt = _ftext(d, "./t0")
            poly_txt = _ftext(d, "./dataDcPolynomial")
            if az_txt is None or t0_txt is None or poly_txt is None:
                continue
            coeff = _parse_float_list(poly_txt)
            if coeff.size == 0:
                continue
            dc_list.append(DcEstimate(az_time=_parse_iso8601(az_txt), t0=float(t0_txt), coeff=coeff))
    if not dc_list:
        raise ValueError("annotation 缺少 dopplerCentroid/dataDcPolynomial")

    return AnnotationInfo(
        first_line_time=first_line_time,
        az_dt=az_dt,
        range_spacing=range_spacing,
        near_range=near_range,
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
        bursts=bursts,
        dc_list=dc_list,
    )


def _poly_eval(coeff: np.ndarray, x: float) -> float:
    out = 0.0
    p = 1.0
    for c in coeff:
        out += float(c) * p
        p *= float(x)
    return float(out)


def _eval_doppler(ann: AnnotationInfo, t: dt.datetime, col_full: float) -> float:
    est = min(ann.dc_list, key=lambda e: abs((e.az_time - t).total_seconds()))
    slant_range = ann.near_range + float(col_full) * ann.range_spacing
    tau = 2.0 * slant_range / C_LIGHT
    x = float(tau - est.t0)
    return _poly_eval(est.coeff, x)


def _doppler_slope_hz_per_s(ann: AnnotationInfo, col_full: float) -> float:
    """
    利用 dcEstimateList 拟合 f_dc(az_time) 的斜率（Hz/s）。
    这比“相邻时刻取最近 dcEstimate 再做差”更稳定，避免退化为 0。
    """
    if len(ann.dc_list) < 2:
        return 0.0
    t0 = ann.dc_list[0].az_time
    ts: List[float] = []
    fs: List[float] = []
    for d in ann.dc_list:
        slant_range = ann.near_range + float(col_full) * ann.range_spacing
        tau = 2.0 * slant_range / C_LIGHT
        x = float(tau - d.t0)
        fdc = _poly_eval(d.coeff, x)
        ts.append(float((d.az_time - t0).total_seconds()))
        fs.append(float(fdc))
    if len(ts) < 2:
        return 0.0
    t = np.asarray(ts, dtype=np.float64)
    f = np.asarray(fs, dtype=np.float64)
    if (not np.isfinite(t).all()) or (not np.isfinite(f).all()):
        return 0.0
    if float(np.std(t)) <= 1e-9:
        return 0.0
    # 一次线性拟合，返回 Hz/s
    slope = float(np.polyfit(t, f, deg=1)[0])
    return slope


def _read_complex(ds: gdal.Dataset, row0: int, row1: int) -> np.ndarray:
    h = int(row1 - row0)
    w = int(ds.RasterXSize)
    if h <= 0:
        raise ValueError("空窗口")
    if ds.RasterCount >= 2:
        re = ds.GetRasterBand(1).ReadAsArray(0, row0, w, h)
        im = ds.GetRasterBand(2).ReadAsArray(0, row0, w, h)
        if re is None or im is None:
            raise RuntimeError("读取复数窗口失败")
        return re.astype(np.float32, copy=False) + 1j * im.astype(np.float32, copy=False)
    arr = ds.GetRasterBand(1).ReadAsArray(0, row0, w, h)
    if arr is None:
        raise RuntimeError("读取窗口失败")
    return np.asarray(arr, dtype=np.complex64)


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cw = np.cumsum(ws)
    if cw[-1] <= 0:
        return float(np.median(xs))
    k = int(np.searchsorted(cw, 0.5 * cw[-1]))
    k = min(max(k, 0), xs.size - 1)
    return float(xs[k])


def _robust_keep(x: np.ndarray, w: np.ndarray, max_dev_px: float) -> np.ndarray:
    med = _weighted_median(x, w)
    abs_dev = np.abs(x - med)
    mad = float(np.median(abs_dev))
    sigma = max(1.4826 * mad, 1e-4)
    thr = max(3.0 * sigma, float(max_dev_px))
    return abs_dev <= thr


def _pair_overlap_lines(ann: AnnotationInfo, i: int) -> int:
    dt_lines = (ann.bursts[i + 1].az_time - ann.bursts[i].az_time).total_seconds() / ann.az_dt
    ov = int(round(ann.lines_per_burst - dt_lines))
    return max(0, min(ov, ann.lines_per_burst))


def _pair_cols_ml(
    bi: BurstInfo,
    bj: BurstInfo,
    lines_per_burst: int,
    ov_full: int,
    nrlks: int,
    width_ml: int,
    ov_ml: int,
    nalks: int,
) -> Tuple[np.ndarray, np.ndarray]:
    c0 = np.full((ov_ml,), -1, dtype=np.int32)
    c1 = np.full((ov_ml,), -2, dtype=np.int32)
    for r in range(ov_ml):
        lf = min(ov_full - 1, int(r * nalks + nalks // 2))
        li = lines_per_burst - ov_full + lf
        lj = lf
        a0 = int(bi.first_valid[li])
        a1 = int(bi.last_valid[li])
        b0 = int(bj.first_valid[lj])
        b1 = int(bj.last_valid[lj])
        lo = max(a0, b0)
        hi = min(a1, b1)
        if lo < 0 or hi < lo:
            continue
        lo_ml = int(lo // nrlks)
        hi_ml = int(hi // nrlks)
        lo_ml = max(0, min(lo_ml, width_ml - 1))
        hi_ml = max(0, min(hi_ml, width_ml - 1))
        if hi_ml < lo_ml:
            continue
        c0[r] = lo_ml
        c1[r] = hi_ml
    return c0, c1


def _coherence(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    xa = a[mask]
    xb = b[mask]
    if xa.size == 0:
        return 0.0
    num = np.sum(xa * np.conj(xb), dtype=np.complex128)
    den = math.sqrt(
        float(np.sum(np.abs(xa) ** 2, dtype=np.float64))
        * float(np.sum(np.abs(xb) ** 2, dtype=np.float64))
    )
    if den <= 0:
        return 0.0
    return float(np.abs(num) / den)


def _estimate_pairs(
    master_ds: gdal.Dataset,
    slave_ds: gdal.Dataset,
    ann_m: AnnotationInfo,
    ann_s: AnnotationInfo,
    *,
    nalks: int,
    nrlks: int,
    min_overlap_lines_ml: int,
    min_pair_coh: float,
    min_delta_fd_hz: float,
    prior_az_offset_ml: float,
) -> List[PairResult]:
    h = int(master_ds.RasterYSize)
    w = int(master_ds.RasterXSize)
    n_pairs = min(len(ann_m.bursts), len(ann_s.bursts)) - 1
    out: List[PairResult] = []
    if n_pairs <= 0:
        return out

    for i in range(n_pairs):
        ov_full = _pair_overlap_lines(ann_m, i)
        if ov_full <= 0:
            continue

        lpb = ann_m.lines_per_burst
        # 按 S1 TOPS 产品行组织：burst 顺序拼接，overlap 对应于前 burst 尾部与后 burst 首部。
        fi0 = i * lpb + (lpb - ov_full)
        fi1 = i * lpb + lpb
        fj0 = (i + 1) * lpb
        fj1 = (i + 1) * lpb + ov_full

        mi0 = fi0 // nalks
        mi1 = (fi1 + nalks - 1) // nalks
        mj0 = fj0 // nalks
        mj1 = (fj1 + nalks - 1) // nalks
        if min(mi1, mj1) <= max(mi0, mj0):
            continue

        ov_ml = min(mi1 - mi0, mj1 - mj0)
        if ov_ml < int(min_overlap_lines_ml):
            continue

        # 限制在影像有效范围
        if mi0 < 0 or mj0 < 0 or mi0 + ov_ml > h or mj0 + ov_ml > h:
            continue

        c0, c1 = _pair_cols_ml(
            ann_m.bursts[i],
            ann_m.bursts[i + 1],
            lines_per_burst=lpb,
            ov_full=ov_full,
            nrlks=nrlks,
            width_ml=w,
            ov_ml=ov_ml,
            nalks=nalks,
        )
        valid_rows = (c1 >= c0)
        if int(np.count_nonzero(valid_rows)) < max(4, ov_ml // 4):
            continue

        m_i = _read_complex(master_ds, mi0, mi0 + ov_ml)
        s_i = _read_complex(slave_ds, mi0, mi0 + ov_ml)
        m_j = _read_complex(master_ds, mj0, mj0 + ov_ml)
        s_j = _read_complex(slave_ds, mj0, mj0 + ov_ml)

        mask = np.zeros((ov_ml, w), dtype=bool)
        for r in range(ov_ml):
            if c1[r] >= c0[r]:
                mask[r, c0[r] : c1[r] + 1] = True
        if int(np.count_nonzero(mask)) < 1024:
            continue

        pair_coh_i = _coherence(m_i, s_i, mask)
        pair_coh_j = _coherence(m_j, s_j, mask)
        pair_coh = min(pair_coh_i, pair_coh_j)
        if pair_coh < float(min_pair_coh):
            continue

        ifg_i = m_i * np.conj(s_i)
        ifg_j = m_j * np.conj(s_j)
        prod = ifg_j[mask] * np.conj(ifg_i[mask])
        if prod.size == 0:
            continue
        z = np.sum(prod, dtype=np.complex128)
        phase = float(np.angle(z))
        esd_q = float(np.abs(z) / (np.sum(np.abs(prod), dtype=np.float64) + 1e-12))

        # representative az/range for Doppler evaluation
        col_full_center = float(np.median((c0[valid_rows] + c1[valid_rows]) * 0.5) * nrlks)
        t_m_i = ann_m.bursts[i].az_time + dt.timedelta(seconds=(lpb - 0.5 * ov_full) * ann_m.az_dt)
        t_m_j = ann_m.bursts[i + 1].az_time + dt.timedelta(seconds=(0.5 * ov_full) * ann_m.az_dt)
        t_s_i = ann_s.bursts[i].az_time + dt.timedelta(seconds=(lpb - 0.5 * ov_full) * ann_s.az_dt)
        t_s_j = ann_s.bursts[i + 1].az_time + dt.timedelta(seconds=(0.5 * ov_full) * ann_s.az_dt)

        dt_sep_m = float((t_m_j - t_m_i).total_seconds())
        dt_sep_s = float((t_s_j - t_s_i).total_seconds())
        slope_m = _doppler_slope_hz_per_s(ann_m, col_full_center)
        slope_s = _doppler_slope_hz_per_s(ann_s, col_full_center)
        delta_fd = 0.5 * (slope_m * dt_sep_m + slope_s * dt_sep_s)
        if abs(delta_fd) < float(min_delta_fd_hz):
            continue

        # phase = 2*pi*delta_fd*dt
        dt_err = phase / (2.0 * np.pi * delta_fd)
        az_full_px = dt_err / ann_m.az_dt
        az_ml_px = az_full_px / float(nalks)

        # 周期解缠到先验附近
        period_ml = 1.0 / (abs(delta_fd) * ann_m.az_dt * float(nalks) + 1e-12)
        az_ml_px = az_ml_px - period_ml * np.round((az_ml_px - float(prior_az_offset_ml)) / period_ml)

        out.append(
            PairResult(
                pair_index=i,
                overlap_lines_full=int(ov_full),
                overlap_lines_ml=int(ov_ml),
                phase=phase,
                delta_fd_hz=float(delta_fd),
                az_offset_ml_px=float(az_ml_px),
                pair_coherence=float(pair_coh),
                esd_quality=float(esd_q),
            )
        )
    return out


def _patch_offset_file(src: Path, dst: Path, az_corr_px: float) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    in_points = True
    in_fitted = False
    in_params = False
    for raw in lines:
        s = raw.strip()
        if s == "FITTING_FORMULA":
            in_points = False
            in_fitted = False
            in_params = False
            out.append(raw)
            continue
        if s == "fitted_params:":
            in_fitted = True
            in_params = False
            out.append(raw)
            continue
        if s == "PARAMETERS":
            in_fitted = False
            in_params = True
            out.append(raw)
            continue
        if s in ("NORMALIZATION", "REGISTRATION_QUALITY"):
            in_fitted = False
            in_params = False
            out.append(raw)
            continue

        if in_points and s:
            p = s.split()
            if len(p) >= 5:
                try:
                    col = int(float(p[0]))
                    rg = float(p[1])
                    row = int(float(p[2]))
                    az = float(p[3]) + float(az_corr_px)
                    cc = float(p[4])
                    out.append(f"{col:8d} {rg:8.3f} {row:8d} {az:8.3f} {cc:8.2f}")
                    continue
                except ValueError:
                    pass

        if in_fitted and s.startswith("a0:"):
            try:
                v = float(s.split(":", 1)[1].strip())
                out.append(f"a0: {v + float(az_corr_px):.12e}")
                continue
            except ValueError:
                pass
        if in_params and s.startswith("a0="):
            try:
                v = float(s.split("=", 1)[1].strip())
                out.append(f"a0={v + float(az_corr_px):.12e}")
                continue
            except ValueError:
                pass

        out.append(raw)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sentinel-1 TOPS 严格版 ESD 残余方位向估计")
    ap.add_argument("master", help="master 复数影像（建议 master_ml.vrt）")
    ap.add_argument("slave", help="slave 复数影像（建议 slave_resamp.vrt）")
    ap.add_argument("--master-yaml", required=True, help="master YAML（需包含 source_zip + annotation_member）")
    ap.add_argument("--slave-yaml", required=True, help="slave YAML（需包含 source_zip + annotation_member）")
    ap.add_argument("--prior-az-offset", type=float, default=0.0, help="先验残余方位向偏移（ml像素）")
    ap.add_argument("--min-overlap-lines-ml", type=int, default=16, help="最小 overlap 行数（ml网格）")
    ap.add_argument("--min-pair-coh", type=float, default=0.08, help="最小 pair coherence")
    ap.add_argument("--min-delta-fd-hz", type=float, default=200.0, help="最小 |delta Doppler| (Hz)")
    ap.add_argument("--max-dev-px", type=float, default=0.08, help="鲁棒筛选最大偏差（ml像素）")
    ap.add_argument("--output-json", default="esd_result.json", help="输出 JSON")
    ap.add_argument("--offset-file", default=None, help="输入 offset_estimate.txt（可选）")
    ap.add_argument("--output-offset-file", default=None, help="输出修正后的 offset_estimate.txt")
    args = ap.parse_args()

    master_path = Path(args.master).expanduser().resolve()
    slave_path = Path(args.slave).expanduser().resolve()
    my = Path(args.master_yaml).expanduser().resolve()
    sy = Path(args.slave_yaml).expanduser().resolve()

    mds = gdal.Open(str(master_path), gdal.GA_ReadOnly)
    sds = gdal.Open(str(slave_path), gdal.GA_ReadOnly)
    if mds is None or sds is None:
        raise RuntimeError("无法打开 master/slave 影像")
    if int(mds.RasterYSize) != int(sds.RasterYSize) or int(mds.RasterXSize) != int(sds.RasterXSize):
        raise RuntimeError("master/slave 尺寸不一致，无法执行 ESD")

    nalks_m, nrlks_m = _multilook_factors(my)
    nalks_s, nrlks_s = _multilook_factors(sy)
    if (nalks_m != nalks_s) or (nrlks_m != nrlks_s):
        raise RuntimeError(f"master/slave multilook 因子不一致: master={nalks_m}:{nrlks_m}, slave={nalks_s}:{nrlks_s}")
    nalks, nrlks = nalks_m, nrlks_m

    ann_m = _parse_annotation(_read_annotation_xml_from_yaml(my))
    ann_s = _parse_annotation(_read_annotation_xml_from_yaml(sy))

    used_min_pair_coh = float(args.min_pair_coh)
    used_min_delta_fd_hz = float(args.min_delta_fd_hz)

    pairs = _estimate_pairs(
        mds,
        sds,
        ann_m,
        ann_s,
        nalks=nalks,
        nrlks=nrlks,
        min_overlap_lines_ml=int(args.min_overlap_lines_ml),
        min_pair_coh=used_min_pair_coh,
        min_delta_fd_hz=used_min_delta_fd_hz,
        prior_az_offset_ml=float(args.prior_az_offset),
    )
    if (not pairs) and (used_min_pair_coh > 0.0):
        print(f"提示: min_pair_coh={used_min_pair_coh:.4f} 无有效 pair，自动回退到 0.0")
        used_min_pair_coh = 0.0
        pairs = _estimate_pairs(
            mds,
            sds,
            ann_m,
            ann_s,
            nalks=nalks,
            nrlks=nrlks,
            min_overlap_lines_ml=int(args.min_overlap_lines_ml),
            min_pair_coh=used_min_pair_coh,
            min_delta_fd_hz=used_min_delta_fd_hz,
            prior_az_offset_ml=float(args.prior_az_offset),
        )
    if (not pairs) and (used_min_delta_fd_hz > 0.0):
        print(f"提示: min_delta_fd_hz={used_min_delta_fd_hz:.3f} 无有效 pair，自动回退到 0.0")
        used_min_delta_fd_hz = 0.0
        pairs = _estimate_pairs(
            mds,
            sds,
            ann_m,
            ann_s,
            nalks=nalks,
            nrlks=nrlks,
            min_overlap_lines_ml=int(args.min_overlap_lines_ml),
            min_pair_coh=used_min_pair_coh,
            min_delta_fd_hz=used_min_delta_fd_hz,
            prior_az_offset_ml=float(args.prior_az_offset),
        )
    if not pairs:
        out_obj = {
            "method": "s1_tops_esd_burst_overlap_strict",
            "multilook": {"nalks": int(nalks), "nrlks": int(nrlks)},
            "num_pairs_used": 0,
            "num_pairs_kept": 0,
            "esd_azimuth_offset_px": 0.0,
            "esd_azimuth_offset_px_weighted_median": 0.0,
            "esd_azimuth_std_px": None,
            "pair_coherence_median": 0.0,
            "esd_quality_median": 0.0,
            "reliability": "low",
            "target_precision_1e3_px_reached": False,
            "used_min_pair_coh": used_min_pair_coh,
            "used_min_delta_fd_hz": used_min_delta_fd_hz,
            "pairs": [],
            "message": "无有效 burst overlap 对可用于 ESD",
        }
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("=== 严格版 ESD 完成（无有效 pair） ===")
        print("result: 无有效 burst overlap 对，已输出 low reliability JSON，不应用修正。")
        print(f"result: {out_json}")
        return 0

    x = np.asarray([p.az_offset_ml_px for p in pairs], dtype=np.float64)
    w = np.asarray(
        [
            max(p.pair_coherence, 1e-6) ** 2
            * max(p.esd_quality, 1e-6) ** 2
            * min(abs(p.delta_fd_hz) / 1000.0, 5.0)
            for p in pairs
        ],
        dtype=np.float64,
    )
    keep = _robust_keep(x, w, max_dev_px=float(args.max_dev_px))
    if int(np.count_nonzero(keep)) < 2:
        raise RuntimeError("ESD 鲁棒筛选后有效 pair 不足")

    xk = x[keep]
    wk = w[keep]
    az = float(np.average(xk, weights=wk))
    std = float(np.sqrt(np.average((xk - az) ** 2, weights=wk)))
    med = float(_weighted_median(xk, wk))
    coh_med = float(np.median(np.asarray([p.pair_coherence for p in pairs], dtype=np.float64)))
    q_med = float(np.median(np.asarray([p.esd_quality for p in pairs], dtype=np.float64)))

    reliability = "low"
    if (coh_med >= 0.25) and (q_med >= 0.15) and (std <= 0.01):
        reliability = "high"
    elif (coh_med >= 0.12) and (q_med >= 0.06) and (std <= 0.03):
        reliability = "medium"

    out_obj = {
        "method": "s1_tops_esd_burst_overlap_strict",
        "multilook": {"nalks": int(nalks), "nrlks": int(nrlks)},
        "num_pairs_used": len(pairs),
        "num_pairs_kept": int(np.count_nonzero(keep)),
        "esd_azimuth_offset_px": az,
        "esd_azimuth_offset_px_weighted_median": med,
        "esd_azimuth_std_px": std,
        "pair_coherence_median": coh_med,
        "esd_quality_median": q_med,
        "reliability": reliability,
        "target_precision_1e3_px_reached": bool(std <= 0.001),
        "used_min_pair_coh": used_min_pair_coh,
        "used_min_delta_fd_hz": used_min_delta_fd_hz,
        "pairs": [
            {
                "pair_index": int(p.pair_index),
                "overlap_lines_full": int(p.overlap_lines_full),
                "overlap_lines_ml": int(p.overlap_lines_ml),
                "phase": float(p.phase),
                "delta_fd_hz": float(p.delta_fd_hz),
                "az_offset_ml_px": float(p.az_offset_ml_px),
                "pair_coherence": float(p.pair_coherence),
                "esd_quality": float(p.esd_quality),
            }
            for p in pairs
        ],
    }

    out_json = Path(args.output_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== 严格版 ESD 完成（burst overlap + XML Doppler） ===")
    print(f"pair: used={len(pairs)}, kept={int(np.count_nonzero(keep))}")
    print(f"ESD az offset: {az:.6f} px (median={med:.6f}, std={std:.6f})")
    print(f"pair coherence median={coh_med:.4f}, esd quality median={q_med:.4f}, reliability={reliability}")
    print(f"result: {out_json}")
    if reliability == "low":
        print("警告: 可靠性低，默认不建议应用该 ESD 修正。")

    if args.offset_file:
        src = Path(args.offset_file).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"offset_file 不存在: {src}")
        dst = (
            Path(args.output_offset_file).expanduser().resolve()
            if args.output_offset_file
            else src.with_name(f"{src.stem}_esd{src.suffix or '.txt'}")
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        _patch_offset_file(src, dst, az)
        print(f"patched offset file: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
