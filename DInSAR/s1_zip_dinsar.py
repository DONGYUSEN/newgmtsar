#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel-1 ZIP -> DInSAR 全流程编排器

目标：
1) 自动下载精确轨道(POEORB，必要时回退 RESORB)
2) 不解压 SAFE 到目录，直接从 ZIP 内读取 measurement/annotation
3) 为 IW1/IW2/IW3 分别生成输入并调用 dinsar.py
4) 将各 IW 的 Geo 产品进行拼接

依赖：
- GDAL(gdal_translate / gdalbuildvrt)
- 本仓库 dinsar.py
- Python: requests, PyYAML, osgeo.gdal
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import io
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import requests
import yaml
import numpy as np
from osgeo import gdal, osr


REPO_ROOT = Path(__file__).resolve().parent
DINSAR_PY = REPO_ROOT / "dinsar.py"
C_LIGHT = 299_792_458.0

POEORB_BASE = "https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB"
RESORB_BASE = "https://step.esa.int/auxdata/orbits/Sentinel-1/RESORB"

S1_NAME_RE = re.compile(
    r"^(S1[AB])_IW_SLC__([0-9A-Z]{4})_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{6})_([0-9A-F]{6})_([0-9A-F]{4})$",
    re.IGNORECASE,
)
ORBIT_RE = re.compile(
    r"^(S1[AB])_OPER_AUX_(POEORB|RESORB)_OPOD_(\d{8}T\d{6})_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF(?:\.zip)?$",
    re.IGNORECASE,
)
UTC_PREFIX_RE = re.compile(r"^[A-Z]+=")

gdal.UseExceptions()


@dataclass
class Scene:
    zip_path: Path
    scene_id: str
    sat: str
    start: dt.datetime
    stop: dt.datetime
    abs_orbit: int
    pols_tag: str


@dataclass
class SwathInput:
    swath: str
    vrt_path: Path
    yaml_path: Path
    measurement_member: str
    annotation_member: str
    source_zip: Path


@dataclass
class BurstLayout:
    lines_per_burst: int
    azimuth_time_interval: float
    first_line_time: dt.datetime
    burst_times: List[dt.datetime]
    geogrid_points: List[Tuple[float, float, float, float]]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    _log("[RUN] " + " ".join(str(x) for x in cmd))
    if cwd is not None:
        _log(f"[CWD] {cwd}")
    subprocess.run(list(cmd), check=True, cwd=str(cwd) if cwd else None)


def _tail_text_file(path: Path, max_lines: int = 60) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def _read_yaml(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _iso_utc(ts: dt.datetime) -> str:
    t = ts.astimezone(dt.timezone.utc)
    return t.isoformat().replace("+00:00", "Z")


def _pixel_to_geo(gt: Tuple[float, float, float, float, float, float], px: float, py: float) -> Tuple[float, float]:
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return float(x), float(y)


def _axis_traditional(srs: osr.SpatialReference) -> None:
    if hasattr(srs, "SetAxisMappingStrategy") and hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)


def _dataset_bbox_wgs84(path: Path) -> Optional[Tuple[float, float, float, float]]:
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
    if not src_wkt:
        # 缺失投影时退化为“已是经纬度”假设，尽量不中断流程。
        lons = [float(x) for x, _ in corners_xy]
        lats = [float(y) for _, y in corners_xy]
        return (min(lons), max(lons), min(lats), max(lats))

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_wkt)
    _axis_traditional(src_srs)

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    _axis_traditional(dst_srs)

    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    lons: List[float] = []
    lats: List[float] = []
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
    return f"{name}[lon:{min_lon:.6f}~{max_lon:.6f}, lat:{min_lat:.6f}~{max_lat:.6f}]"


def _parse_iso8601(t: str) -> dt.datetime:
    s = str(t).strip()
    s = UTC_PREFIX_RE.sub("", s)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    v = dt.datetime.fromisoformat(s)
    if v.tzinfo is None:
        v = v.replace(tzinfo=dt.timezone.utc)
    return v.astimezone(dt.timezone.utc)


def _scene_from_zip(zip_path: Path) -> Scene:
    stem = zip_path.stem
    m = S1_NAME_RE.match(stem)
    if not m:
        raise ValueError(f"不符合 Sentinel-1 IW SLC 命名: {zip_path.name}")
    sat, mode_pol, t0, t1, abs_orb, _, _ = m.groups()
    pols = mode_pol[-2:]
    return Scene(
        zip_path=zip_path.resolve(),
        scene_id=stem,
        sat=sat.upper(),
        start=_parse_iso8601(t0),
        stop=_parse_iso8601(t1),
        abs_orbit=int(abs_orb),
        pols_tag=pols.upper(),
    )


def _parse_orbit_name(name: str) -> Optional[Tuple[str, str, dt.datetime, dt.datetime, dt.datetime]]:
    mm = ORBIT_RE.match(Path(name).name)
    if not mm:
        return None
    sat, kind, created, v0, v1 = mm.groups()
    return (
        sat.upper(),
        kind.upper(),
        _parse_iso8601(created),
        _parse_iso8601(v0),
        _parse_iso8601(v1),
    )


def _orbit_covers(path_or_name: str, sat: str, start: dt.datetime, stop: dt.datetime) -> bool:
    p = _parse_orbit_name(Path(path_or_name).name)
    if p is None:
        return False
    sat0, _, _, v0, v1 = p
    return sat0 == sat.upper() and v0 <= start and v1 >= stop


def _find_local_orbit(orbit_dir: Path, sat: str, start: dt.datetime, stop: dt.datetime) -> Optional[Path]:
    if not orbit_dir.exists():
        return None
    best: Optional[Tuple[int, dt.datetime, Path]] = None
    for p in orbit_dir.glob("*.EOF"):
        info = _parse_orbit_name(p.name)
        if info is None:
            continue
        sat0, kind, created, v0, v1 = info
        if sat0 != sat.upper() or not (v0 <= start and v1 >= stop):
            continue
        kind_score = 1 if kind == "POEORB" else 0
        cand = (kind_score, created, p.resolve())
        if best is None or cand > best:
            best = cand
    return best[2] if best else None


class OrbitDownloader:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self._listing_cache: Dict[str, List[str]] = {}

    def _list_month(self, base: str, sat: str, year: int, month: int) -> List[str]:
        url = f"{base}/{sat}/{year:04d}/{month:02d}/"
        if url in self._listing_cache:
            return self._listing_cache[url]
        r = self.session.get(url, timeout=60)
        r.raise_for_status()
        names = re.findall(r'href="([^"]+)"', r.text)
        files = [n for n in names if ORBIT_RE.match(Path(n).name)]
        self._listing_cache[url] = files
        return files

    def _download_url(self, url: str) -> bytes:
        r = self.session.get(url, timeout=180)
        r.raise_for_status()
        return r.content

    def _find_remote_candidate(
        self,
        base: str,
        sat: str,
        start: dt.datetime,
        stop: dt.datetime,
    ) -> Optional[Tuple[str, str, dt.datetime]]:
        # 为避免月边界漏检，覆盖观测时间前后 3 天所涉及的月份
        days = set()
        d0 = (start - dt.timedelta(days=3)).date()
        d1 = (stop + dt.timedelta(days=3)).date()
        dd = d0
        while dd <= d1:
            days.add((dd.year, dd.month))
            dd += dt.timedelta(days=1)

        candidates: List[Tuple[str, str, dt.datetime]] = []
        for y, m in sorted(days):
            try:
                files = self._list_month(base, sat, y, m)
            except Exception:
                continue
            month_url = f"{base}/{sat}/{y:04d}/{m:02d}/"
            for fn in files:
                info = _parse_orbit_name(fn)
                if info is None:
                    continue
                sat0, _, created, v0, v1 = info
                if sat0 == sat.upper() and v0 <= start and v1 >= stop:
                    candidates.append((month_url, fn, created))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0]

    def ensure_orbit(
        self,
        orbit_dir: Path,
        sat: str,
        start: dt.datetime,
        stop: dt.datetime,
        allow_resorb: bool = True,
        keep_zip: bool = False,
    ) -> Path:
        orbit_dir.mkdir(parents=True, exist_ok=True)
        local = _find_local_orbit(orbit_dir, sat, start, stop)
        if local is not None:
            _log(f"复用本地轨道: {local.name}")
            return local

        plans = [(POEORB_BASE, "POEORB")]
        if allow_resorb:
            plans.append((RESORB_BASE, "RESORB"))

        for base, kind in plans:
            cand = self._find_remote_candidate(base, sat, start, stop)
            if cand is None:
                continue
            month_url, fn, _ = cand
            url = month_url + fn
            _log(f"下载 {kind} 轨道: {fn}")
            data = self._download_url(url)

            if fn.lower().endswith(".zip"):
                zpath = orbit_dir / fn
                zpath.write_bytes(data)
                with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                    eofs = [n for n in zf.namelist() if n.upper().endswith(".EOF")]
                    if not eofs:
                        raise RuntimeError(f"轨道压缩包不含 EOF: {fn}")
                    eof_name = Path(eofs[0]).name
                    eof_path = orbit_dir / eof_name
                    eof_path.write_bytes(zf.read(eofs[0]))
                if not keep_zip:
                    try:
                        zpath.unlink()
                    except Exception:
                        pass
            else:
                eof_path = orbit_dir / Path(fn).name
                eof_path.write_bytes(data)

            if not _orbit_covers(eof_path.name, sat=sat, start=start, stop=stop):
                raise RuntimeError(f"下载轨道不覆盖观测时段: {eof_path.name}")
            _log(f"轨道就绪: {eof_path}")
            return eof_path

        raise RuntimeError(
            f"未找到可用轨道: sat={sat}, start={start.isoformat()}, stop={stop.isoformat()}"
        )


def _read_text_from_zip(zip_path: Path, member: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member, "r") as fp:
            return fp.read().decode("utf-8")


def _find_members_for_pol(zip_path: Path, pol: str) -> Dict[str, Tuple[str, str]]:
    pol = pol.lower()
    out: Dict[str, Tuple[str, str]] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    meas_re = re.compile(
        rf"^(.+\.SAFE)/measurement/(s1[ab]-iw([123])-slc-{pol}-[a-z0-9\-]+\.tiff)$",
        re.IGNORECASE,
    )
    anno_re = re.compile(
        rf"^(.+\.SAFE)/annotation/(s1[ab]-iw([123])-slc-{pol}-[a-z0-9\-]+\.xml)$",
        re.IGNORECASE,
    )
    meas: Dict[str, str] = {}
    anno: Dict[str, str] = {}
    for n in names:
        mm = meas_re.match(n)
        if mm:
            meas[f"iw{mm.group(3)}"] = n
            continue
        aa = anno_re.match(n)
        if aa:
            anno[f"iw{aa.group(3)}"] = n

    for sw in ("iw1", "iw2", "iw3"):
        if sw in meas and sw in anno:
            out[sw] = (meas[sw], anno[sw])
    return out


def _vsizip_member(zip_path: Path, member: str) -> str:
    return f"/vsizip/{zip_path.resolve()}/{member}"


def _create_vrt(src: str, dst_vrt: Path) -> None:
    ds = gdal.Open(src, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开源影像: {src}")
    drv = gdal.GetDriverByName("VRT")
    if drv is None:
        ds = None
        raise RuntimeError("GDAL 缺少 VRT 驱动")
    out = drv.CreateCopy(str(dst_vrt), ds, strict=0)
    ds = None
    if out is None:
        raise RuntimeError(f"创建 VRT 失败: {dst_vrt}")
    out = None


def _ftext(root: ET.Element, path: str, default: Optional[str] = None) -> Optional[str]:
    node = root.find(path)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _parse_geolocation_grid_points(root: ET.Element) -> List[Tuple[float, float, float, float]]:
    pts: List[Tuple[float, float, float, float]] = []
    for gp in root.findall(".//geolocationGridPoint"):
        try:
            line = float(_ftext(gp, "./line", "nan"))
            pixel = float(_ftext(gp, "./pixel", "nan"))
            lat = float(_ftext(gp, "./latitude", "nan"))
            lon = float(_ftext(gp, "./longitude", "nan"))
            if not (line == line and pixel == pixel and lat == lat and lon == lon):
                continue
            pts.append((line, pixel, lat, lon))
        except Exception:
            continue
    return pts


def _parse_annotation_fields(xml_text: str) -> Dict[str, object]:
    root = ET.fromstring(xml_text)
    image_info = root.find(".//imageInformation")
    if image_info is None:
        raise ValueError("annotation XML 缺少 imageInformation")

    def g(path: str, default: Optional[str] = None) -> Optional[str]:
        return _ftext(root, path, default=default)

    mission = g("./adsHeader/missionId", "S1A")
    swath = g("./adsHeader/swath", "IW2")
    pol = g("./adsHeader/polarisation", "VV")
    orbit_dir = g(".//generalAnnotation/productInformation/pass", "Ascending")
    abs_orbit = int(g("./adsHeader/absoluteOrbitNumber", "0"))

    first_line = (
        g(".//imageInformation/productFirstLineUtcTime")
        or g(".//downlinkInformation/firstLineSensingTime")
        or g("./adsHeader/startTime")
    )
    last_line = (
        g(".//imageInformation/productLastLineUtcTime")
        or g(".//downlinkInformation/lastLineSensingTime")
        or g("./adsHeader/stopTime")
    )
    if first_line is None or last_line is None:
        raise ValueError("annotation XML 缺少观测起止时间")

    radar_freq = float(g(".//generalAnnotation/productInformation/radarFrequency", "5405000000.0"))
    range_spacing = float(g(".//imageInformation/rangePixelSpacing", "2.329562"))
    az_spacing = float(g(".//imageInformation/azimuthPixelSpacing", "13.9"))
    slant_t = float(g(".//imageInformation/slantRangeTime", "0.0"))
    prf_txt = g(".//downlinkInformation/prf")
    az_dt_txt = g(".//imageInformation/azimuthTimeInterval")
    if prf_txt:
        prf = float(prf_txt)
    elif az_dt_txt:
        az_dt = float(az_dt_txt)
        prf = 1.0 / az_dt if az_dt > 0 else 0.0
    else:
        prf = 0.0

    az_dt = float(g(".//imageInformation/azimuthTimeInterval", "0.0"))
    corners = _parse_annotation_corners(root)

    return {
        "mission": mission,
        "swath": swath,
        "pol": pol,
        "orbit_direction": orbit_dir,
        "abs_orbit": abs_orbit,
        "first_line": first_line,
        "last_line": last_line,
        "range_spacing": range_spacing,
        "az_spacing": az_spacing,
        "slant_range_time": slant_t,
        "near_range": slant_t * C_LIGHT / 2.0,
        "radar_frequency": radar_freq,
        "wavelength": C_LIGHT / radar_freq if radar_freq > 0 else 0.0555,
        "prf": prf,
        "azimuth_time_interval": az_dt,
        "corners": corners,
    }


def _parse_annotation_corners(root: ET.Element) -> Dict[str, Dict[str, float]]:
    pts = _parse_geolocation_grid_points(root)

    if len(pts) < 4:
        return {
            "upper_left": {"lat": 0.0, "lon": 0.0},
            "upper_right": {"lat": 0.0, "lon": 0.0},
            "lower_left": {"lat": 0.0, "lon": 0.0},
            "lower_right": {"lat": 0.0, "lon": 0.0},
        }

    lines = [p[0] for p in pts]
    pixels = [p[1] for p in pts]
    min_l, max_l = min(lines), max(lines)
    min_p, max_p = min(pixels), max(pixels)

    targets = {
        "upper_left": (min_l, min_p),
        "upper_right": (min_l, max_p),
        "lower_left": (max_l, min_p),
        "lower_right": (max_l, max_p),
    }
    out: Dict[str, Dict[str, float]] = {}
    for key, (tl, tp) in targets.items():
        best = min(pts, key=lambda p: (p[0] - tl) ** 2 + (p[1] - tp) ** 2)
        out[key] = {"lat": float(best[2]), "lon": float(best[3])}
    return out


def _parse_eof_points(eof_path: Path) -> List[Dict[str, object]]:
    tree = ET.parse(str(eof_path))
    root = tree.getroot()
    out: List[Dict[str, object]] = []
    for osv in root.findall(".//OSV"):
        try:
            utc = _ftext(osv, "./UTC")
            x = float(_ftext(osv, "./X", "nan"))
            y = float(_ftext(osv, "./Y", "nan"))
            z = float(_ftext(osv, "./Z", "nan"))
            vx = float(_ftext(osv, "./VX", "nan"))
            vy = float(_ftext(osv, "./VY", "nan"))
            vz = float(_ftext(osv, "./VZ", "nan"))
            if utc is None:
                continue
            _ = _parse_iso8601(utc)  # 仅用于校验
            out.append(
                {
                    "time": UTC_PREFIX_RE.sub("", utc),
                    "position": {"x": x, "y": y, "z": z},
                    "velocity": {"vx": vx, "vy": vy, "vz": vz},
                }
            )
        except Exception:
            continue
    if not out:
        raise RuntimeError(f"EOF 未解析到 OSV: {eof_path}")
    return out


def _subset_orbit_points(
    orbit_points: List[Dict[str, object]],
    t0: dt.datetime,
    t1: dt.datetime,
    margin_hours: float = 2.0,
) -> List[Dict[str, object]]:
    margin = dt.timedelta(hours=float(margin_hours))
    lo = t0 - margin
    hi = t1 + margin
    keep: List[Dict[str, object]] = []
    for p in orbit_points:
        try:
            tp = _parse_iso8601(str(p["time"]))
        except Exception:
            continue
        if lo <= tp <= hi:
            keep.append(p)
    return keep if len(keep) >= 16 else orbit_points


def _build_yaml_config(
    scene: Scene,
    swath: str,
    shape: Tuple[int, int],
    ann: Dict[str, object],
    orbit_points: List[Dict[str, object]],
    measurement_member: str,
    annotation_member: str,
) -> Dict[str, object]:
    first_line = str(ann["first_line"])
    last_line = str(ann["last_line"])
    orbit_dir = str(ann["orbit_direction"])
    ascending = orbit_dir.strip().lower().startswith("asc")
    y, x = shape

    cfg = {
        "metadata": {
            "mission": scene.sat,
            "scene_id": scene.scene_id,
            "swath": swath.upper(),
            "polarization": str(ann["pol"]),
            "product_type": "SLC",
            "first_line_sensing_time": first_line,
            "last_line_sensing_time": last_line,
            "start_time": scene.start.isoformat(),
            "stop_time": scene.stop.isoformat(),
            "absolute_orbit_number": int(ann["abs_orbit"]),
            "source_zip": str(scene.zip_path.resolve()),
            "measurement_member": str(measurement_member),
            "annotation_member": str(annotation_member),
        },
        "image_parameters": {
            "nrows": int(y),
            "ncols": int(x),
            "bands": 1,
            "data_format": "CInt16",
            "byte_order": "LSB",
        },
        "radar_parameters": {
            "near_range": float(ann["near_range"]),
            "range_spacing": float(ann["range_spacing"]),
            "range_pixel_spacing": float(ann["range_spacing"]),
            "azimuth_spacing": float(ann["az_spacing"]),
            "azimuth_time_interval": float(ann.get("azimuth_time_interval", 0.0)),
            "prf": float(ann["prf"]),
            "radar_frequency": float(ann["radar_frequency"]),
            "wavelength": float(ann["wavelength"]),
            "look_dir": "right",
        },
        "orbit_parameters": {
            "orbit_direction": orbit_dir,
        },
        "orbit_data": {
            "orbit_points": orbit_points,
        },
        "orbit_ascending": bool(ascending),
        "corner_coordinates": ann["corners"],
        # 兼容旧字段读取
        "near_range": float(ann["near_range"]),
        "range_spacing": float(ann["range_spacing"]),
        "azimuth_pixel_spacing": float(ann["az_spacing"]),
        "azimuth_time_interval": float(ann.get("azimuth_time_interval", 0.0)),
        "wavelength": float(ann["wavelength"]),
    }
    return cfg


def _prepare_scene_inputs(
    scene: Scene,
    orbit_path: Path,
    out_dir: Path,
    pol: str,
    swaths: Iterable[str],
    orbit_margin_hours: float,
) -> Dict[str, SwathInput]:
    sw_members = _find_members_for_pol(scene.zip_path, pol=pol)
    need_swaths = [s.lower() for s in swaths]
    missing = [s for s in need_swaths if s not in sw_members]
    if missing:
        raise RuntimeError(f"{scene.zip_path.name} 缺少 swath: {missing}, pol={pol}")

    orbit_all = _parse_eof_points(orbit_path)
    scene_dir = out_dir / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, SwathInput] = {}
    for sw in need_swaths:
        meas_member, ann_member = sw_members[sw]
        src_vsizip = _vsizip_member(scene.zip_path, meas_member)

        stem = f"{scene.scene_id}_{sw}_{pol.lower()}"
        vrt_path = scene_dir / f"{stem}.vrt"
        yaml_path = scene_dir / f"{stem}.yaml"
        _log(f"[VRT] {src_vsizip} -> {vrt_path}")
        _create_vrt(src_vsizip, vrt_path)

        ds = gdal.Open(str(vrt_path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"无法打开 VRT: {vrt_path}")
        shape = (int(ds.RasterYSize), int(ds.RasterXSize))
        ds = None

        ann_xml = _read_text_from_zip(scene.zip_path, ann_member)
        ann = _parse_annotation_fields(ann_xml)
        t0 = _parse_iso8601(str(ann["first_line"]))
        t1 = _parse_iso8601(str(ann["last_line"]))
        orbit_subset = _subset_orbit_points(orbit_all, t0=t0, t1=t1, margin_hours=orbit_margin_hours)
        cfg = _build_yaml_config(
            scene=scene,
            swath=sw,
            shape=shape,
            ann=ann,
            orbit_points=orbit_subset,
            measurement_member=meas_member,
            annotation_member=ann_member,
        )
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        out[sw] = SwathInput(
            swath=sw,
            vrt_path=vrt_path,
            yaml_path=yaml_path,
            measurement_member=meas_member,
            annotation_member=ann_member,
            source_zip=scene.zip_path,
        )
    return out


def _load_bbox_from_yaml(yaml_path: Path) -> Tuple[float, float, float, float]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        ycfg = yaml.safe_load(f) or {}
    cc = ycfg.get("corner_coordinates", {}) or {}
    vals = []
    for k in ("upper_left", "upper_right", "lower_left", "lower_right"):
        v = cc.get(k, {})
        if "lon" in v and "lat" in v:
            vals.append((float(v["lon"]), float(v["lat"])))
    if not vals:
        raise ValueError(f"YAML 缺少 corner_coordinates: {yaml_path}")
    lons = [x[0] for x in vals]
    lats = [x[1] for x in vals]
    return min(lons), max(lons), min(lats), max(lats)


def _build_union_bbox(yaml_paths: Sequence[Path]) -> Tuple[float, float, float, float]:
    mins = []
    maxs = []
    minlat = []
    maxlat = []
    for p in yaml_paths:
        a, b, c, d = _load_bbox_from_yaml(p)
        mins.append(a)
        maxs.append(b)
        minlat.append(c)
        maxlat.append(d)
    return min(mins), max(maxs), min(minlat), max(maxlat)


def _ensure_dem(
    dem_path: Optional[Path],
    dem_dir: Path,
    ref_yaml: Path,
    union_bbox: Tuple[float, float, float, float],
    dem_on_mismatch: str = "mkdem",
) -> Path:
    def _build_dem() -> Path:
        dem_dir.mkdir(parents=True, exist_ok=True)
        bbox = ",".join(f"{x:.8f}" for x in union_bbox)
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "utils" / "mkdem.py"),
                str(ref_yaml),
                "--bbox",
                bbox,
                "--output-dir",
                str(dem_dir),
                "--out-crs",
                "latlon",
            ]
        )
        dem_latlon = dem_dir / "dem_latlon.tif"
        dem_fallback = dem_dir / "dem.tif"
        if dem_latlon.exists():
            return dem_latlon.resolve()
        if dem_fallback.exists():
            return dem_fallback.resolve()
        raise RuntimeError(f"mkdem 未产出 DEM: {dem_dir}")

    candidate: Optional[Path] = None
    if dem_path is not None:
        candidate = dem_path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"DEM 不存在: {candidate}")
    else:
        candidate = _build_dem()

    dem_bbox = _dataset_bbox_wgs84(candidate)
    if dem_bbox is None:
        raise RuntimeError(f"无法读取 DEM 地理范围（缺少有效地理参考）: {candidate}")

    if _bbox_contains(dem_bbox, union_bbox):
        _log(f"DEM 覆盖检查通过: {_fmt_bbox('DEM', dem_bbox)} 覆盖 {_fmt_bbox('ALL_SWATH', union_bbox)}")
        return candidate

    _log(f"DEM 覆盖不足: {_fmt_bbox('DEM', dem_bbox)} 无法覆盖 {_fmt_bbox('ALL_SWATH', union_bbox)}")
    if dem_on_mismatch == "exit":
        raise RuntimeError("DEM 覆盖检查失败，按 --dem-on-mismatch=exit 终止。")

    _log("尝试调用 mkdem 重新生成可覆盖全部 swath 的 DEM...")
    rebuilt = _build_dem()
    rebuilt_bbox = _dataset_bbox_wgs84(rebuilt)
    if rebuilt_bbox is None:
        raise RuntimeError(f"重建 DEM 后仍无法读取地理范围: {rebuilt}")
    if not _bbox_contains(rebuilt_bbox, union_bbox):
        raise RuntimeError(
            f"mkdem 生成的 DEM 仍覆盖不足: {_fmt_bbox('DEM', rebuilt_bbox)} vs {_fmt_bbox('ALL_SWATH', union_bbox)}"
        )
    _log(f"已切换使用重建 DEM: {rebuilt}")
    return rebuilt


def _parse_burst_layout_from_xml(xml_text: str) -> Optional[BurstLayout]:
    root = ET.fromstring(xml_text)
    az_dt = float(_ftext(root, ".//imageInformation/azimuthTimeInterval", "0.0"))
    lines_per_burst = int(float(_ftext(root, ".//swathTiming/linesPerBurst", "0")))
    bl = root.find(".//swathTiming/burstList")
    if az_dt <= 0.0 or lines_per_burst <= 0 or bl is None:
        return None

    first_line_txt = (
        _ftext(root, ".//imageInformation/productFirstLineUtcTime")
        or _ftext(root, ".//downlinkInformation/firstLineSensingTime")
        or _ftext(root, "./adsHeader/startTime")
    )
    if first_line_txt is None:
        return None

    burst_times: List[dt.datetime] = []
    for b in bl.findall("./burst"):
        az_t = _ftext(b, "./azimuthTime")
        if az_t is None:
            continue
        try:
            burst_times.append(_parse_iso8601(az_t))
        except Exception:
            continue
    if not burst_times:
        return None

    return BurstLayout(
        lines_per_burst=lines_per_burst,
        azimuth_time_interval=az_dt,
        first_line_time=_parse_iso8601(first_line_txt),
        burst_times=burst_times,
        geogrid_points=_parse_geolocation_grid_points(root),
    )


def _median_interval_seconds(times: Sequence[dt.datetime]) -> Optional[float]:
    if len(times) < 2:
        return None
    vals = []
    for i in range(len(times) - 1):
        vals.append((times[i + 1] - times[i]).total_seconds())
    vals = [v for v in vals if v > 0]
    if not vals:
        return None
    vals.sort()
    return float(vals[len(vals) // 2])


def _match_common_bursts(
    master_times: Sequence[dt.datetime],
    slave_times: Sequence[dt.datetime],
    max_dt_seconds: float,
) -> List[Tuple[int, int, float]]:
    i = 0
    j = 0
    out: List[Tuple[int, int, float]] = []
    while i < len(master_times) and j < len(slave_times):
        d = (master_times[i] - slave_times[j]).total_seconds()
        ad = abs(d)
        if ad <= max_dt_seconds:
            out.append((i, j, ad))
            i += 1
            j += 1
            continue
        if d < 0:
            i += 1
        else:
            j += 1
    return out


def _match_common_bursts_offset_contiguous(
    master_times: Sequence[dt.datetime],
    slave_times: Sequence[dt.datetime],
    seed_matches: Sequence[Tuple[int, int, float]],
) -> List[Tuple[int, int, float]]:
    """
    基于 seed 匹配估计 burst 序号偏移（j-i），构建连续且无空洞的匹配对。
    目的：避免“时间容差匹配后遗漏部分 burst”导致 burst-group 窗口出现空隙。
    """
    if (not master_times) or (not slave_times):
        return []
    if seed_matches:
        offs = sorted(int(j) - int(i) for i, j, _ in seed_matches)
        off = int(offs[len(offs) // 2])
    else:
        off = 0

    i0 = max(0, -off)
    i1 = min(len(master_times), len(slave_times) - off)
    out: List[Tuple[int, int, float]] = []
    for i in range(i0, i1):
        j = i + off
        d = abs((master_times[i] - slave_times[j]).total_seconds())
        out.append((int(i), int(j), float(d)))
    return out


def _matches_have_index_gaps(matches: Sequence[Tuple[int, int, float]]) -> bool:
    if len(matches) < 2:
        return False
    for k in range(1, len(matches)):
        i0, j0, _ = matches[k - 1]
        i1, j1, _ = matches[k]
        if (int(i1) - int(i0) != 1) or (int(j1) - int(j0) != 1):
            return True
    return False


def _normalize_corner_coordinates(cc: object) -> Dict[str, Dict[str, float]]:
    zero = {
        "upper_left": {"lat": 0.0, "lon": 0.0},
        "upper_right": {"lat": 0.0, "lon": 0.0},
        "lower_left": {"lat": 0.0, "lon": 0.0},
        "lower_right": {"lat": 0.0, "lon": 0.0},
    }
    if not isinstance(cc, dict):
        return zero
    out: Dict[str, Dict[str, float]] = {}
    for k in ("upper_left", "upper_right", "lower_left", "lower_right"):
        v = cc.get(k, {})
        if isinstance(v, dict) and ("lat" in v) and ("lon" in v):
            out[k] = {"lat": float(v["lat"]), "lon": float(v["lon"])}
        else:
            out[k] = {"lat": 0.0, "lon": 0.0}
    return out


def _subset_corners_from_geogrid(
    geogrid_points: Sequence[Tuple[float, float, float, float]],
    row0: int,
    row1: int,
    fallback: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    if row1 <= row0:
        return fallback
    pts = [p for p in geogrid_points if (row0 - 2) <= p[0] <= (row1 + 2)]
    if len(pts) < 4:
        return fallback

    lines = [p[0] for p in pts]
    pixels = [p[1] for p in pts]
    min_l, max_l = min(lines), max(lines)
    min_p, max_p = min(pixels), max(pixels)
    targets = {
        "upper_left": (min_l, min_p),
        "upper_right": (min_l, max_p),
        "lower_left": (max_l, min_p),
        "lower_right": (max_l, max_p),
    }
    out: Dict[str, Dict[str, float]] = {}
    for key, (tl, tp) in targets.items():
        best = min(pts, key=lambda p: (p[0] - tl) ** 2 + (p[1] - tp) ** 2)
        out[key] = {"lat": float(best[2]), "lon": float(best[3])}
    return out


def _create_window_vrt(src_vrt: Path, dst_vrt: Path, row0: int, row1: int) -> Tuple[int, int]:
    ds = gdal.Open(str(src_vrt), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开源 VRT: {src_vrt}")
    width = int(ds.RasterXSize)
    height = int(ds.RasterYSize)
    ds = None
    if row0 < 0 or row1 > height or row1 <= row0:
        raise ValueError(f"非法窗口: row0={row0}, row1={row1}, height={height}")

    dst_vrt.parent.mkdir(parents=True, exist_ok=True)
    win_h = int(row1 - row0)
    out = gdal.Translate(
        str(dst_vrt),
        str(src_vrt),
        format="VRT",
        srcWin=[0, int(row0), int(width), int(win_h)],
    )
    if out is None:
        raise RuntimeError(f"创建窗口 VRT 失败: {dst_vrt}")
    out = None
    return win_h, width


def _build_subset_yaml(
    src_yaml: Path,
    dst_yaml: Path,
    row0: int,
    row1: int,
    ncols: int,
    corners: Dict[str, Dict[str, float]],
    t0: dt.datetime,
    t1: dt.datetime,
    orbit_margin_hours: float,
) -> None:
    cfg = copy.deepcopy(_read_yaml(src_yaml))
    cfg.setdefault("image_parameters", {})
    cfg.setdefault("metadata", {})
    cfg.setdefault("orbit_data", {})
    cfg["image_parameters"]["nrows"] = int(row1 - row0)
    cfg["image_parameters"]["ncols"] = int(ncols)
    cfg["metadata"]["first_line_sensing_time"] = _iso_utc(t0)
    cfg["metadata"]["last_line_sensing_time"] = _iso_utc(t1)
    cfg["metadata"]["subset_row_start"] = int(row0)
    cfg["metadata"]["subset_row_stop"] = int(row1)
    cfg["corner_coordinates"] = corners

    try:
        orbit_points = cfg.get("orbit_data", {}).get("orbit_points", [])
        if isinstance(orbit_points, list) and orbit_points:
            cfg["orbit_data"]["orbit_points"] = _subset_orbit_points(
                orbit_points,
                t0=t0,
                t1=t1,
                margin_hours=float(orbit_margin_hours),
            )
    except Exception:
        pass

    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(dst_yaml, cfg)


def _prepare_burst_group_inputs(
    master: SwathInput,
    slave: SwathInput,
    out_dir: Path,
    bursts_per_group: int,
    burst_margin_lines: int,
    orbit_margin_hours: float,
    min_common_bursts: int,
    az_looks: int = 1,
    id_prefix: str = "g",
) -> List[Dict[str, object]]:
    m_xml = _read_text_from_zip(master.source_zip, master.annotation_member)
    s_xml = _read_text_from_zip(slave.source_zip, slave.annotation_member)
    m_layout = _parse_burst_layout_from_xml(m_xml)
    s_layout = _parse_burst_layout_from_xml(s_xml)
    if m_layout is None or s_layout is None:
        return []

    m_ds = gdal.Open(str(master.vrt_path), gdal.GA_ReadOnly)
    s_ds = gdal.Open(str(slave.vrt_path), gdal.GA_ReadOnly)
    if m_ds is None or s_ds is None:
        raise RuntimeError(f"无法读取 swath 尺寸: {master.vrt_path} / {slave.vrt_path}")
    m_rows = int(m_ds.RasterYSize)
    s_rows = int(s_ds.RasterYSize)
    m_cols = int(m_ds.RasterXSize)
    s_cols = int(s_ds.RasterXSize)
    m_ds = None
    s_ds = None

    burst_dt_m = _median_interval_seconds(m_layout.burst_times) or (
        m_layout.lines_per_burst * m_layout.azimuth_time_interval
    )
    burst_dt_s = _median_interval_seconds(s_layout.burst_times) or (
        s_layout.lines_per_burst * s_layout.azimuth_time_interval
    )
    match_tol = max(0.5, min(float(burst_dt_m), float(burst_dt_s)) * 0.5)
    strict_matches = _match_common_bursts(m_layout.burst_times, s_layout.burst_times, max_dt_seconds=match_tol)
    contig_matches = _match_common_bursts_offset_contiguous(
        m_layout.burst_times,
        s_layout.burst_times,
        strict_matches,
    )
    matches = list(strict_matches)
    if len(matches) >= int(min_common_bursts):
        # strict 匹配存在漏配（索引不连续）时，优先切换到连续 offset 匹配，避免 burst-group 出现空隙。
        if _matches_have_index_gaps(matches) and len(contig_matches) >= len(matches):
            matches = contig_matches
    elif len(contig_matches) >= int(min_common_bursts):
        # strict 匹配数量不足时，回退到连续 offset 匹配（仍保持主/从序号单调对应）。
        matches = contig_matches

    if len(matches) < int(min_common_bursts):
        # 回退到按序号对齐（保守模式），避免严格时间匹配导致无法运行。
        k = min(len(m_layout.burst_times), len(s_layout.burst_times))
        matches = []
        for idx in range(k):
            d = abs((m_layout.burst_times[idx] - s_layout.burst_times[idx]).total_seconds())
            matches.append((idx, idx, d))

    if len(matches) < int(min_common_bursts):
        return []

    m_yaml = _read_yaml(master.yaml_path)
    s_yaml = _read_yaml(slave.yaml_path)
    m_corner_fallback = _normalize_corner_coordinates(m_yaml.get("corner_coordinates", {}))
    s_corner_fallback = _normalize_corner_coordinates(s_yaml.get("corner_coordinates", {}))

    plans: List[Dict[str, object]] = []
    m_lpb = int(m_layout.lines_per_burst)
    s_lpb = int(s_layout.lines_per_burst)
    step = max(1, int(bursts_per_group))
    margin = max(0, int(burst_margin_lines))
    az_lks = max(1, int(az_looks))
    gid = 0
    for start in range(0, len(matches), step):
        chunk = matches[start : start + step]
        if not chunk:
            continue
        gid += 1
        gname = f"{str(id_prefix)}{gid:03d}"
        m_ids = [x[0] for x in chunk]
        s_ids = [x[1] for x in chunk]
        m_row0 = max(0, min(m_ids) * m_lpb - margin)
        m_row1 = min(m_rows, (max(m_ids) + 1) * m_lpb + margin)
        s_row0 = max(0, min(s_ids) * s_lpb - margin)
        s_row1 = min(s_rows, (max(s_ids) + 1) * s_lpb + margin)
        if az_lks > 1:
            m_row0 = (m_row0 // az_lks) * az_lks
            s_row0 = (s_row0 // az_lks) * az_lks
            m_row1 = (m_row1 // az_lks) * az_lks
            s_row1 = (s_row1 // az_lks) * az_lks
        if m_row1 <= m_row0 or s_row1 <= s_row0:
            continue

        group_dir = out_dir / gname
        m_vrt = group_dir / "master.vrt"
        s_vrt = group_dir / "slave.vrt"
        _create_window_vrt(master.vrt_path, m_vrt, row0=m_row0, row1=m_row1)
        _create_window_vrt(slave.vrt_path, s_vrt, row0=s_row0, row1=s_row1)

        m_t0 = m_layout.first_line_time + dt.timedelta(seconds=float(m_row0) * m_layout.azimuth_time_interval)
        m_t1 = m_layout.first_line_time + dt.timedelta(seconds=float(max(m_row0, m_row1 - 1)) * m_layout.azimuth_time_interval)
        s_t0 = s_layout.first_line_time + dt.timedelta(seconds=float(s_row0) * s_layout.azimuth_time_interval)
        s_t1 = s_layout.first_line_time + dt.timedelta(seconds=float(max(s_row0, s_row1 - 1)) * s_layout.azimuth_time_interval)

        m_corners = _subset_corners_from_geogrid(m_layout.geogrid_points, m_row0, m_row1, m_corner_fallback)
        s_corners = _subset_corners_from_geogrid(s_layout.geogrid_points, s_row0, s_row1, s_corner_fallback)

        m_yaml_out = group_dir / "master.yaml"
        s_yaml_out = group_dir / "slave.yaml"
        _build_subset_yaml(
            src_yaml=master.yaml_path,
            dst_yaml=m_yaml_out,
            row0=m_row0,
            row1=m_row1,
            ncols=m_cols,
            corners=m_corners,
            t0=m_t0,
            t1=m_t1,
            orbit_margin_hours=orbit_margin_hours,
        )
        _build_subset_yaml(
            src_yaml=slave.yaml_path,
            dst_yaml=s_yaml_out,
            row0=s_row0,
            row1=s_row1,
            ncols=s_cols,
            corners=s_corners,
            t0=s_t0,
            t1=s_t1,
            orbit_margin_hours=orbit_margin_hours,
        )

        plans.append(
            {
                "group_id": gname,
                "master": SwathInput(
                    swath=master.swath,
                    vrt_path=m_vrt,
                    yaml_path=m_yaml_out,
                    measurement_member=master.measurement_member,
                    annotation_member=master.annotation_member,
                    source_zip=master.source_zip,
                ),
                "slave": SwathInput(
                    swath=slave.swath,
                    vrt_path=s_vrt,
                    yaml_path=s_yaml_out,
                    measurement_member=slave.measurement_member,
                    annotation_member=slave.annotation_member,
                    source_zip=slave.source_zip,
                ),
                "master_row_range": [int(m_row0), int(m_row1)],
                "slave_row_range": [int(s_row0), int(s_row1)],
                "matched_burst_pairs": [[int(a), int(b)] for a, b, _ in chunk],
            }
        )
    return plans


def _parse_multilook_lks(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in str(s).split(":") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"multilook 格式非法: {s}，应为 az:rg")
    az, rg = int(parts[0]), int(parts[1])
    if az <= 0 or rg <= 0:
        raise ValueError(f"multilook 视数必须为正整数: {s}")
    return az, rg


def _read_image_shape_from_yaml(yaml_path: Path) -> Tuple[int, int]:
    cfg = _read_yaml(yaml_path)
    ip = cfg.get("image_parameters", {}) if isinstance(cfg, dict) else {}
    if not isinstance(ip, dict):
        raise ValueError(f"YAML 缺少 image_parameters: {yaml_path}")
    nr = int(ip.get("nrows", 0))
    nc = int(ip.get("ncols", 0))
    if nr <= 0 or nc <= 0:
        raise ValueError(f"YAML 中 nrows/ncols 非法: {yaml_path}")
    return nr, nc


def _read_radar_params(yaml_path: Path) -> Tuple[float, float, float]:
    cfg = _read_yaml(yaml_path)
    rp = cfg.get("radar_parameters", {}) if isinstance(cfg, dict) else {}
    if not isinstance(rp, dict):
        rp = {}
    wl = float(rp.get("wavelength", cfg.get("wavelength", 0.0555)))
    rg = float(rp.get("range_spacing", cfg.get("range_spacing", 1.0)))
    az = float(rp.get("azimuth_spacing", cfg.get("azimuth_pixel_spacing", 1.0)))
    if wl <= 0:
        wl = 0.0555
    if rg <= 0:
        rg = 1.0
    if az <= 0:
        az = 1.0
    return wl, rg, az


def _load_float32_2d(path: Path, shape: Tuple[int, int], desc: str) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"{desc} shape 非法: {shape}")
    arr = np.fromfile(str(path), dtype=np.float32)
    if arr.size != h * w:
        raise RuntimeError(f"{desc} 尺寸异常: {path} size={arr.size}, expect={h*w}")
    return arr.reshape((h, w)).astype(np.float32, copy=False)


def _write_float32_tiff(path: Path, arr: np.ndarray) -> Path:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"仅支持 2D float32 写出: {path}, shape={a.shape}")
    h, w = int(a.shape[0]), int(a.shape[1])
    drv = gdal.GetDriverByName("GTiff")
    if drv is None:
        raise RuntimeError("GDAL 缺少 GTiff 驱动")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = drv.Create(str(path), w, h, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    if ds is None:
        raise RuntimeError(f"创建 GeoTIFF 失败: {path}")
    b = ds.GetRasterBand(1)
    b.WriteArray(a)
    b.FlushCache()
    ds.FlushCache()
    ds = None
    return path.resolve()


def _run_swath_mosaic_from_burst_groups(
    swath: str,
    swath_out_dir: Path,
    master_input: SwathInput,
    dem: Path,
    group_exec: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    nalks, nrlks = _parse_multilook_lks(args.multilook)
    full_rows, full_cols = _read_image_shape_from_yaml(master_input.yaml_path)
    ml_rows = int(full_rows // nalks)
    ml_cols = int(full_cols // nrlks)
    if ml_rows <= 0 or ml_cols <= 0:
        raise RuntimeError(f"{swath} 多视后尺寸非法: {(ml_rows, ml_cols)}")

    phase_acc = np.zeros((ml_rows, ml_cols), dtype=np.complex64)
    weight_acc = np.zeros((ml_rows, ml_cols), dtype=np.float32)
    coh_blend_sum = np.zeros((ml_rows, ml_cols), dtype=np.float32)
    coh_blend_w = np.zeros((ml_rows, ml_cols), dtype=np.float32)

    merged_groups: List[str] = []
    skipped_groups: Dict[str, str] = {}
    group_stitch_meta: Dict[str, Dict[str, object]] = {}

    # ISCE2 风格的 overlap 处理：先估计常量相位偏置，再在 overlap 区做 taper feather。
    phase_align_min_coh = 0.12
    phase_align_min_pixels = 2048
    eps = 1e-8

    def _group_row_start(item: Tuple[str, Dict[str, object]]) -> int:
        rr = item[1].get("master_row_range", [])
        if isinstance(rr, list) and len(rr) == 2:
            try:
                return int(rr[0])
            except Exception:
                return 10**12
        return 10**12

    for gid, rec in sorted(group_exec.items(), key=_group_row_start):
        if not bool(rec.get("success", False)):
            continue
        run_dir = Path(str(rec.get("run_dir", ""))).expanduser().resolve()
        rr = rec.get("master_row_range", [])
        if not isinstance(rr, list) or len(rr) != 2:
            skipped_groups[gid] = "invalid_master_row_range"
            continue
        row0 = int(rr[0])
        row1 = int(rr[1])
        if row1 <= row0:
            skipped_groups[gid] = "empty_master_row_range"
            continue

        ml_yaml = run_dir / "01_multilook" / "master_ml.yaml"
        if ml_yaml.exists():
            gh, gw = _read_image_shape_from_yaml(ml_yaml)
        else:
            gh = int((row1 - row0) // nalks)
            gw = ml_cols
        if gh <= 0 or gw <= 0:
            skipped_groups[gid] = "invalid_group_ml_shape"
            continue

        y0 = int(row0 // nalks)
        y1 = min(ml_rows, y0 + gh)
        x1 = min(ml_cols, gw)
        if y1 <= y0 or x1 <= 0:
            skipped_groups[gid] = "group_outside_swath_ml_grid"
            continue

        coh_bin = run_dir / "06_interferometry" / "main" / "master_ml_slave_ml_resamp_coherence.bin"
        wrp_bin = (
            run_dir
            / "06_interferometry"
            / "combined"
            / "master_ml_slave_ml_resamp_wrapped_phase_flat_topo_removed.bin"
        )
        if (not coh_bin.exists()) or (not wrp_bin.exists()):
            skipped_groups[gid] = "missing_group_interferometry_bins"
            continue

        coh = _load_float32_2d(coh_bin, (gh, gw), f"{swath}:{gid}:coherence")
        wrp = _load_float32_2d(wrp_bin, (gh, gw), f"{swath}:{gid}:wrapped_phase")
        coh = coh[: (y1 - y0), :x1]
        wrp = wrp[: (y1 - y0), :x1]

        valid = np.isfinite(coh) & np.isfinite(wrp)
        if not np.any(valid):
            skipped_groups[gid] = "no_valid_pixels"
            continue

        coh_clip = np.where(valid, np.clip(coh, 0.0, 1.0), 0.0).astype(np.float32, copy=False)
        z = np.exp(1j * wrp.astype(np.float32, copy=False)).astype(np.complex64, copy=False)

        # 1) overlap 相位偏置校正（常量项）
        curr_w = coh_clip.copy()
        exist_w = weight_acc[y0:y1, :x1]
        overlap0 = valid & (exist_w > eps) & (curr_w > phase_align_min_coh)
        phase_bias_rad = 0.0
        phase_bias_applied = False
        if int(np.count_nonzero(overlap0)) >= int(phase_align_min_pixels):
            exist_complex = phase_acc[y0:y1, :x1]
            exist_unit = np.zeros_like(exist_complex, dtype=np.complex64)
            mk = np.abs(exist_complex) > eps
            exist_unit[mk] = (exist_complex[mk] / np.abs(exist_complex[mk])).astype(np.complex64, copy=False)
            ww = np.sqrt(np.clip(exist_w[overlap0], 0.0, None) * np.clip(curr_w[overlap0], 0.0, None)).astype(np.float32)
            cross = exist_unit[overlap0] * np.conj(z[overlap0])
            csum = np.sum(cross * ww)
            if np.isfinite(csum.real) and np.isfinite(csum.imag) and (abs(csum) > 0.0):
                phase_bias_rad = float(np.angle(csum))
                z = z * np.exp(1j * np.float32(phase_bias_rad))
                phase_bias_applied = True

        # 2) overlap feather（行向线性 taper）
        exist_w2 = weight_acc[y0:y1, :x1]
        overlap1 = valid & (exist_w2 > eps) & (curr_w > 0.0)
        blend_alpha = np.ones((y1 - y0, x1), dtype=np.float32)
        ov_rows = np.where(np.any(overlap1, axis=1))[0]
        feather_rows = 0
        if ov_rows.size > 0:
            r0 = int(ov_rows.min())
            r1 = int(ov_rows.max())
            feather_rows = int(r1 - r0 + 1)
            if feather_rows > 0:
                ramp = (np.arange(feather_rows, dtype=np.float32) + 1.0) / (float(feather_rows) + 1.0)
                blend_alpha[r0 : r1 + 1, :] = ramp[:, None]

        phase_w = (curr_w * blend_alpha).astype(np.float32, copy=False)
        phase_w = np.where(valid, phase_w, 0.0).astype(np.float32, copy=False)
        coh_w = np.where(valid, blend_alpha, 0.0).astype(np.float32, copy=False)

        phase_acc[y0:y1, :x1] += z * phase_w
        weight_acc[y0:y1, :x1] += phase_w
        coh_blend_sum[y0:y1, :x1] += coh_clip * coh_w
        coh_blend_w[y0:y1, :x1] += coh_w

        group_stitch_meta[gid] = {
            "phase_bias_applied": bool(phase_bias_applied),
            "phase_bias_rad": float(phase_bias_rad),
            "overlap_pixels": int(np.count_nonzero(overlap1)),
            "feather_rows": int(feather_rows),
            "phase_align_min_coh": float(phase_align_min_coh),
            "phase_align_min_pixels": int(phase_align_min_pixels),
        }
        if phase_bias_applied:
            _log(
                f"{swath.upper()}:{gid} overlap 校正: phase_bias={phase_bias_rad:.6f} rad, "
                f"overlap_pixels={int(np.count_nonzero(overlap1))}, feather_rows={feather_rows}"
            )
        merged_groups.append(gid)

    if not merged_groups:
        raise RuntimeError(f"{swath} 无可用于 swath 后处理的 burst-group")

    wrapped_mosaic = np.zeros((ml_rows, ml_cols), dtype=np.float32)
    mk = weight_acc > 1e-8
    wrapped_mosaic[mk] = np.angle(phase_acc[mk]).astype(np.float32, copy=False)

    coherence_mosaic = np.zeros((ml_rows, ml_cols), dtype=np.float32)
    ck = coh_blend_w > 0.0
    coherence_mosaic[ck] = (coh_blend_sum[ck] / coh_blend_w[ck]).astype(np.float32, copy=False)
    coherence_mosaic = np.clip(coherence_mosaic, 0.0, 1.0)

    post_dir = swath_out_dir / "08_swath_post"
    intf_dir = post_dir / "interferometry"
    intf_dir.mkdir(parents=True, exist_ok=True)
    wrapped_bin = intf_dir / "swath_wrapped_phase_flat_topo_removed.bin"
    coherence_bin = intf_dir / "swath_coherence.bin"
    wrapped_mosaic.tofile(str(wrapped_bin))
    coherence_mosaic.tofile(str(coherence_bin))

    wavelength, range_spacing, az_spacing = _read_radar_params(master_input.yaml_path)

    swml_dir = post_dir / "swath_ml"
    swml_dir.mkdir(parents=True, exist_ok=True)
    swml_base = swml_dir / "master_ml"
    swml_tif = swml_dir / "master_ml.tiff"
    swml_yaml = swml_dir / "master_ml.yaml"
    if bool(args.force) or (not swml_tif.exists()) or (not swml_yaml.exists()):
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "utils" / "multilook.py"),
                str(master_input.vrt_path),
                str(swml_base),
                "--nalks",
                str(nalks),
                "--nrlks",
                str(nrlks),
                "--input-yaml",
                str(master_input.yaml_path),
                "--output-yaml",
                str(swml_yaml),
                "--preserve-phase",
            ]
        )
    if (not swml_tif.exists()) or (not swml_yaml.exists()):
        raise RuntimeError(f"{swath} swath master multilook 产物缺失")

    geosar_dir = post_dir / "geometry"
    geosar_dir.mkdir(parents=True, exist_ok=True)
    geosar_h5 = geosar_dir / "geosar.h5"
    if bool(args.force) or (not geosar_h5.exists()):
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "utils" / "geosar.py"),
                "--yaml",
                str(swml_yaml),
                "--dem",
                str(dem),
                "--output",
                str(geosar_h5),
            ]
        )
    if not geosar_h5.exists():
        raise RuntimeError(f"{swath} geosar 产物缺失")

    geocode_res_m = max(float(range_spacing) * float(nrlks), float(az_spacing) * float(nalks))

    return {
        "merged_groups": merged_groups,
        "skipped_groups": skipped_groups,
        "group_stitch_meta": group_stitch_meta,
        "ml_shape": [int(ml_rows), int(ml_cols)],
        "wrapped_bin": str(wrapped_bin),
        "coherence_bin": str(coherence_bin),
        "swml_tif": str(swml_tif),
        "swml_yaml": str(swml_yaml),
        "geosar_h5": str(geosar_h5),
        "wavelength": float(wavelength),
        "range_spacing": float(range_spacing),
        "azimuth_spacing": float(az_spacing),
        "nalks": int(nalks),
        "nrlks": int(nrlks),
        "geocode_res_m": float(geocode_res_m),
    }


def _run_swath_unwrap_geocode(
    swath: str,
    swath_out_dir: Path,
    mosaic_info: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    post_dir = swath_out_dir / "08_swath_post"
    intf_dir = post_dir / "interferometry"
    intf_dir.mkdir(parents=True, exist_ok=True)

    ml_shape = mosaic_info.get("ml_shape", [])
    if (not isinstance(ml_shape, list)) or len(ml_shape) != 2:
        raise RuntimeError(f"{swath} 缺少有效 ml_shape")
    ml_rows, ml_cols = int(ml_shape[0]), int(ml_shape[1])
    if ml_rows <= 0 or ml_cols <= 0:
        raise RuntimeError(f"{swath} ml_shape 非法: {ml_shape}")

    wrapped_bin = Path(str(mosaic_info.get("wrapped_bin", ""))).expanduser().resolve()
    coherence_bin = Path(str(mosaic_info.get("coherence_bin", ""))).expanduser().resolve()
    geosar_h5 = Path(str(mosaic_info.get("geosar_h5", ""))).expanduser().resolve()
    wavelength = float(mosaic_info.get("wavelength", 0.0555))
    geocode_res_m = float(mosaic_info.get("geocode_res_m", 30.0))
    if (not wrapped_bin.exists()) or (not coherence_bin.exists()) or (not geosar_h5.exists()):
        raise RuntimeError(f"{swath} 缺少 unwrap/geocode 所需输入")

    snaphu_prefix = "swath_flat_topo_removed"
    unwrap_bin = intf_dir / f"{snaphu_prefix}_unwrapped_phase.bin"
    los_bin = intf_dir / f"{snaphu_prefix}_los_deformation_m.bin"
    snaphu_cmd = [
        sys.executable,
        str(REPO_ROOT / "interferometry" / "snaphu.py"),
        "--wrapped-phase",
        str(wrapped_bin),
        "--coherence",
        str(coherence_bin),
        "--width",
        str(ml_cols),
        "--height",
        str(ml_rows),
        "--output-dir",
        str(intf_dir),
        "--prefix",
        str(snaphu_prefix),
        "--wavelength",
        str(wavelength),
    ]
    if args.snaphu_tile_rows is not None and args.snaphu_tile_cols is not None:
        snaphu_cmd += ["--tile-rows", str(args.snaphu_tile_rows), "--tile-cols", str(args.snaphu_tile_cols)]
    if args.snaphu_nproc is not None:
        snaphu_cmd += ["--nproc", str(args.snaphu_nproc)]
    if bool(args.force) or (not unwrap_bin.exists()) or (not los_bin.exists()):
        _run(snaphu_cmd)
    else:
        _log(f"{swath.upper()} swath snaphu 结果已存在，跳过重算")
    if (not unwrap_bin.exists()) or (not los_bin.exists()):
        raise RuntimeError(f"{swath} swath snaphu 结果缺失")

    sar_dir = swath_out_dir / "07_geocode" / "sar_products"
    geo_dir = swath_out_dir / "07_geocode" / "geo_products"
    sar_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    coherence_sar_tif = _write_float32_tiff(
        sar_dir / "coherence_sar.tif",
        _load_float32_2d(coherence_bin, (ml_rows, ml_cols), f"{swath}:coherence_mosaic"),
    )
    wrapped_sar_tif = _write_float32_tiff(
        sar_dir / "wrapped_phase_flat_topo_removed_sar.tif",
        _load_float32_2d(wrapped_bin, (ml_rows, ml_cols), f"{swath}:wrapped_mosaic"),
    )
    unwrap_sar_tif = _write_float32_tiff(
        sar_dir / "unwrapped_phase_sar.tif",
        _load_float32_2d(unwrap_bin, (ml_rows, ml_cols), f"{swath}:unwrapped_phase"),
    )
    los_sar_tif = _write_float32_tiff(
        sar_dir / "los_displacement_m_sar.tif",
        _load_float32_2d(los_bin, (ml_rows, ml_cols), f"{swath}:los"),
    )

    geocode_tasks = [
        ("coherence", coherence_sar_tif),
        ("wrapped_phase_flat_topo_removed", wrapped_sar_tif),
        ("unwrapped_phase", unwrap_sar_tif),
        ("los_displacement_m", los_sar_tif),
    ]
    geocode_outputs: Dict[str, str] = {}
    for name, sar_tif in geocode_tasks:
        geo_tif = geo_dir / f"{name}_geo.tif"
        if bool(args.force) or (not geo_tif.exists()):
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "utils" / "sar2ll.py"),
                    "warp",
                    str(sar_tif),
                    str(geosar_h5),
                    str(geo_tif),
                    "UTM",
                    str(geocode_res_m),
                    "--interp",
                    "auto",
                    "--extent",
                    "sar",
                ]
            )
        geocode_outputs[name] = str(geo_tif)

    return {
        "unwrapped_bin": str(unwrap_bin),
        "los_bin": str(los_bin),
        "geocode_outputs": geocode_outputs,
    }


def _run_dinsar_swath(
    master: SwathInput,
    slave: SwathInput,
    swath_out_dir: Path,
    dem: Path,
    multilook: str,
    force: bool = False,
    dem_on_mismatch: str = "mkdem",
    snaphu_tile_rows: Optional[int] = None,
    snaphu_tile_cols: Optional[int] = None,
    snaphu_nproc: Optional[int] = None,
    registration_esd: bool = True,
    registration_esd_force_apply_low_reliability: bool = True,
    log_file: Optional[Path] = None,
    stop_after_wrapped_phase: bool = False,
) -> Tuple[List[str], int]:
    swath_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(DINSAR_PY),
        str(master.vrt_path),
        str(slave.vrt_path),
        "--output-dir",
        str(swath_out_dir),
        "--dem",
        str(dem),
        "--dem-on-mismatch",
        str(dem_on_mismatch),
        "--multilook",
        str(multilook),
        "--skip-geosar-coreg",
    ]
    if registration_esd:
        cmd.append("--registration-esd")
        if registration_esd_force_apply_low_reliability:
            cmd.append("--registration-esd-apply-low-reliability")
    if force:
        cmd.append("--force")
    if snaphu_tile_rows is not None and snaphu_tile_cols is not None:
        cmd += ["--snaphu-tile-rows", str(snaphu_tile_rows), "--snaphu-tile-cols", str(snaphu_tile_cols)]
    if snaphu_nproc is not None:
        cmd += ["--snaphu-nproc", str(snaphu_nproc)]
    if bool(stop_after_wrapped_phase):
        cmd.append("--stop-after-wrapped-phase")

    if log_file is None:
        _run(cmd)
        return list(cmd), 0

    log_file.parent.mkdir(parents=True, exist_ok=True)
    _log("[RUN] " + " ".join(str(x) for x in cmd))
    _log(f"[LOG] {log_file}")
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"# CMD: {' '.join(str(x) for x in cmd)}\n")
        lf.write(f"# CWD: {swath_out_dir}\n")
        lf.write(f"# START: {dt.datetime.now().isoformat()}\n\n")
        proc = subprocess.run(
            list(cmd),
            check=False,
            cwd=str(swath_out_dir),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lf.write(f"\n# END: {dt.datetime.now().isoformat()}\n")
        lf.write(f"# RETURNCODE: {proc.returncode}\n")
    return list(cmd), int(proc.returncode)


def _mosaic_geo_products(swath_dirs: Dict[str, Path], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prod_map: Dict[str, List[Path]] = {}
    for sw, wd in swath_dirs.items():
        gp_candidates = [
            wd / "07_geocode" / "geo_products",  # 与当前 dinsar.py 一致
            wd / "gcd" / "geo_products",          # 兼容旧目录结构
        ]
        gp = next((p for p in gp_candidates if p.exists()), None)
        if gp is None:
            _log(f"警告: {sw} 无 geo_products 目录，跳过拼接")
            continue
        for tif in sorted(gp.glob("*_geo.tif")):
            prod_map.setdefault(tif.name, []).append(tif.resolve())

    out_files: Dict[str, Path] = {}
    for name, files in sorted(prod_map.items()):
        if not files:
            continue
        out_tif = out_dir / name
        if len(files) == 1:
            shutil.copy2(files[0], out_tif)
            out_files[name] = out_tif.resolve()
            continue
        vrt = out_dir / f"{Path(name).stem}.vrt"
        gdal.BuildVRT(
            str(vrt),
            [str(x) for x in files],
            options=gdal.BuildVRTOptions(resolution="highest"),
        )
        ds = gdal.Translate(
            str(out_tif),
            str(vrt),
            format="GTiff",
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
        )
        if ds is None:
            raise RuntimeError(f"拼接输出失败: {out_tif}")
        ds = None
        out_files[name] = out_tif.resolve()
    return out_files


def _run_with_retries(
    run_label: str,
    master: SwathInput,
    slave: SwathInput,
    run_dir: Path,
    dem: Path,
    args: argparse.Namespace,
    stop_after_wrapped_phase: bool = False,
    registration_esd_force_apply_low_reliability: bool = True,
) -> Tuple[bool, List[Dict[str, object]], Optional[str]]:
    max_attempts = int(args.swath_retries) + 1
    attempts: List[Dict[str, object]] = []
    for idx in range(1, max_attempts + 1):
        log_file = run_dir / "logs" / f"dinsar_attempt_{idx}.log"
        t0 = time.time()
        cmd, rc = _run_dinsar_swath(
            master=master,
            slave=slave,
            swath_out_dir=run_dir,
            dem=dem,
            multilook=args.multilook,
            force=bool(args.force),
            dem_on_mismatch=args.dem_on_mismatch,
            snaphu_tile_rows=args.snaphu_tile_rows,
            snaphu_tile_cols=args.snaphu_tile_cols,
            snaphu_nproc=args.snaphu_nproc,
            registration_esd=(not bool(args.no_registration_esd)),
            registration_esd_force_apply_low_reliability=bool(registration_esd_force_apply_low_reliability),
            log_file=log_file,
            stop_after_wrapped_phase=bool(stop_after_wrapped_phase),
        )
        dt_sec = float(time.time() - t0)
        tail = _tail_text_file(log_file, max_lines=40)
        attempts.append(
            {
                "attempt": idx,
                "success": (rc == 0),
                "return_code": int(rc),
                "duration_sec": dt_sec,
                "log_file": str(log_file),
                "cmd": cmd,
                "log_tail": tail,
            }
        )
        if rc == 0:
            _log(f"{run_label} 成功（attempt {idx}/{max_attempts}）")
            return True, attempts, None

        _log(f"{run_label} 失败（attempt {idx}/{max_attempts}, rc={rc}）")
        if idx < max_attempts:
            backoff = float(args.swath_retry_backoff_seconds) * idx
            if backoff > 0:
                _log(f"{run_label} 将在 {backoff:.1f}s 后重试")
                time.sleep(backoff)

    return False, attempts, f"return_code={attempts[-1].get('return_code') if attempts else 'unknown'}"


def _choose_pair(scenes: List[Scene], master_name: Optional[str], slave_name: Optional[str]) -> Tuple[Scene, Scene]:
    by_name = {s.zip_path.name: s for s in scenes}
    by_stem = {s.zip_path.stem: s for s in scenes}

    def _pick(name: Optional[str], arg_name: str) -> Optional[Scene]:
        if not name:
            return None
        key = Path(name).name
        if key in by_name:
            return by_name[key]
        stem = Path(key).stem
        if stem in by_stem:
            return by_stem[stem]
        raise ValueError(f"{arg_name} 未找到: {name}")

    m = _pick(master_name, "--master-zip")
    s = _pick(slave_name, "--slave-zip")

    if m is not None and s is not None:
        return m, s

    ordered = sorted(scenes, key=lambda x: x.start)
    if len(ordered) < 2:
        raise ValueError("至少需要两个 Sentinel-1 ZIP")
    if m is None and s is None:
        return ordered[0], ordered[-1]
    if m is None:
        cands = [x for x in ordered if x.start < s.start]
        return (cands[-1] if cands else ordered[0]), s
    cands = [x for x in ordered if x.start > m.start]
    return m, (cands[0] if cands else ordered[-1])


def _parse_swaths(sw: str) -> List[str]:
    vals = [x.strip().lower() for x in sw.split(",") if x.strip()]
    ok = {"iw1", "iw2", "iw3"}
    bad = [x for x in vals if x not in ok]
    if bad:
        raise ValueError(f"--swaths 非法: {bad}")
    if not vals:
        raise ValueError("--swaths 不能为空")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sentinel-1 ZIP 直读 DInSAR 全流程（轨道下载/分swath处理/拼接）"
    )
    parser.add_argument("--data-dir", required=True, help="包含 Sentinel-1 .zip 的目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--master-zip", default=None, help="指定 master zip 文件名（位于 data-dir）")
    parser.add_argument("--slave-zip", default=None, help="指定 slave zip 文件名（位于 data-dir）")
    parser.add_argument("--pol", default="vv", choices=["vv", "vh", "hh", "hv"], help="极化通道")
    parser.add_argument("--swaths", default="iw1,iw2,iw3", help="处理的 swath 列表，逗号分隔")
    parser.add_argument("--multilook", default="4:1", help="传给 dinsar.py 的 multilook，例如 4:1")
    parser.add_argument("--dem", default=None, help="可选 DEM 路径；不传则自动 mkdem")
    parser.add_argument("--dem-on-mismatch", default="mkdem", choices=["mkdem", "exit"])
    parser.add_argument("--orbit-dir", default=None, help="轨道目录，默认 output-dir/orbits")
    parser.add_argument("--no-resorb", action="store_true", help="POEORB 不可用时不回退 RESORB")
    parser.add_argument("--keep-orbit-zip", action="store_true", help="保留下载的 .EOF.zip")
    parser.add_argument("--orbit-margin-hours", type=float, default=2.0, help="写入 YAML 的轨道时间缓冲小时数")
    parser.add_argument("--prepare-only", action="store_true", help="仅准备 VRT/YAML/轨道，不执行 dinsar")
    parser.add_argument("--skip-stitch", action="store_true", help="执行 swath 处理但跳过最终拼接")
    parser.add_argument("--snaphu-tile-rows", type=int, default=None)
    parser.add_argument("--snaphu-tile-cols", type=int, default=None)
    parser.add_argument("--snaphu-nproc", type=int, default=None)
    parser.add_argument("--swath-retries", type=int, default=1, help="每个 swath 的失败重试次数（默认 1，即最多执行 2 次）")
    parser.add_argument("--swath-retry-backoff-seconds", type=float, default=10.0, help="swath 重试前等待秒数（线性回退）")
    parser.add_argument("--allow-partial-success", action="store_true", help="即便有 swath 最终失败也返回 0（默认返回非零）")
    parser.add_argument(
        "--burst-split-mode",
        choices=["off", "auto", "force"],
        default="force",
        help="按 burst-group 处理模式：off=关闭，auto=可用则启用（必要时可回退），force=必须启用（默认）",
    )
    parser.add_argument(
        "--burst-processing-style",
        choices=["isce2", "group"],
        default="isce2",
        help="burst 处理风格：isce2=每个 burst 独立处理并后续拼接（默认），group=按 burst-group 聚合处理",
    )
    parser.add_argument("--bursts-per-group", type=int, default=3, help="burst 分组大小（group 风格生效；isce2 风格固定为 1）")
    parser.add_argument("--burst-margin-lines", type=int, default=64, help="burst-group 上下额外扩展行数（group 风格生效；isce2 风格固定为 0）")
    parser.add_argument("--burst-min-common", type=int, default=2, help="master/slave 最小公共 burst 数，低于该值时 burst 模式不可用")
    parser.add_argument(
        "--allow-swath-fallback",
        action="store_true",
        help="允许回退整 swath 处理（默认不允许，避免误用整 swath）",
    )
    parser.add_argument("--no-registration-esd", action="store_true", help="关闭 S1 默认 ESD 配准微调")
    parser.add_argument(
        "--no-registration-esd-force-apply",
        action="store_true",
        help="默认会强制应用 ESD（即便 reliability=low）；使用该参数可关闭强制应用",
    )
    parser.add_argument(
        "--registration-esd-force-apply",
        action="store_true",
        help="显式强制应用 low-reliability ESD（会覆盖 --no-registration-esd-force-apply）",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if (args.snaphu_tile_rows is None) ^ (args.snaphu_tile_cols is None):
        raise ValueError("--snaphu-tile-rows 和 --snaphu-tile-cols 必须同时设置")
    if args.snaphu_tile_rows is not None and int(args.snaphu_tile_rows) <= 0:
        raise ValueError("--snaphu-tile-rows 必须为正整数")
    if args.snaphu_tile_cols is not None and int(args.snaphu_tile_cols) <= 0:
        raise ValueError("--snaphu-tile-cols 必须为正整数")
    if args.snaphu_nproc is not None and int(args.snaphu_nproc) <= 0:
        raise ValueError("--snaphu-nproc 必须为正整数")
    if int(args.swath_retries) < 0:
        raise ValueError("--swath-retries 必须为非负整数")
    if float(args.swath_retry_backoff_seconds) < 0:
        raise ValueError("--swath-retry-backoff-seconds 必须为非负数")
    if int(args.bursts_per_group) <= 0:
        raise ValueError("--bursts-per-group 必须为正整数")
    if int(args.burst_margin_lines) < 0:
        raise ValueError("--burst-margin-lines 必须为非负整数")
    if int(args.burst_min_common) <= 0:
        raise ValueError("--burst-min-common 必须为正整数")

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir 不存在或不是目录: {data_dir}")
    out_root = Path(args.output_dir).expanduser().resolve()
    orbit_dir = Path(args.orbit_dir).expanduser().resolve() if args.orbit_dir else (out_root / "orbits")
    prep_dir = out_root / "prepared"
    runs_dir = out_root / "swath_runs"
    mosaic_dir = out_root / "mosaic_geo_products"
    dem_dir = out_root / "dem"

    out_root.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(data_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"{data_dir} 未发现 .zip")
    scenes = [_scene_from_zip(z) for z in zips]

    master_scene, slave_scene = _choose_pair(scenes, args.master_zip, args.slave_zip)
    if master_scene.scene_id == slave_scene.scene_id:
        raise ValueError("master 和 slave 不能相同")

    swaths = _parse_swaths(args.swaths)
    nalks, nrlks = _parse_multilook_lks(args.multilook)
    burst_style = str(args.burst_processing_style).lower()
    effective_bursts_per_group = int(args.bursts_per_group)
    effective_burst_margin_lines = int(args.burst_margin_lines)
    effective_burst_min_common = int(args.burst_min_common)
    # 兼容历史行为：group 风格默认强制应用 low-reliability ESD；isce2 风格默认不强制。
    effective_esd_force_apply_low_reliability = (not bool(args.no_registration_esd_force_apply))
    if burst_style == "isce2":
        effective_bursts_per_group = 1
        effective_burst_margin_lines = 0
        effective_burst_min_common = 1
        effective_esd_force_apply_low_reliability = False
    if bool(args.registration_esd_force_apply):
        effective_esd_force_apply_low_reliability = True

    _log(f"选择 master: {master_scene.zip_path.name}")
    _log(f"选择 slave : {slave_scene.zip_path.name}")
    _log(f"处理 swaths: {swaths}, pol={args.pol}, multilook={nalks}:{nrlks}")
    _log(
        "burst 处理策略: "
        f"style={burst_style}, bursts_per_group={effective_bursts_per_group}, "
        f"burst_margin_lines={effective_burst_margin_lines}, burst_min_common={effective_burst_min_common}, "
        f"registration_esd_force_apply_low_reliability={effective_esd_force_apply_low_reliability}"
    )

    downloader = OrbitDownloader()
    allow_resorb = not bool(args.no_resorb)
    m_orbit = downloader.ensure_orbit(
        orbit_dir=orbit_dir,
        sat=master_scene.sat,
        start=master_scene.start,
        stop=master_scene.stop,
        allow_resorb=allow_resorb,
        keep_zip=bool(args.keep_orbit_zip),
    )
    s_orbit = downloader.ensure_orbit(
        orbit_dir=orbit_dir,
        sat=slave_scene.sat,
        start=slave_scene.start,
        stop=slave_scene.stop,
        allow_resorb=allow_resorb,
        keep_zip=bool(args.keep_orbit_zip),
    )

    master_inputs = _prepare_scene_inputs(
        scene=master_scene,
        orbit_path=m_orbit,
        out_dir=prep_dir,
        pol=args.pol,
        swaths=swaths,
        orbit_margin_hours=float(args.orbit_margin_hours),
    )
    slave_inputs = _prepare_scene_inputs(
        scene=slave_scene,
        orbit_path=s_orbit,
        out_dir=prep_dir,
        pol=args.pol,
        swaths=swaths,
        orbit_margin_hours=float(args.orbit_margin_hours),
    )

    union_bbox = _build_union_bbox(
        [master_inputs[s].yaml_path for s in swaths] + [slave_inputs[s].yaml_path for s in swaths]
    )
    _log(f"ALL_SWATH bbox: {_fmt_bbox('ALL_SWATH', union_bbox)}")
    dem_path = _ensure_dem(
        dem_path=Path(args.dem).expanduser() if args.dem else None,
        dem_dir=dem_dir,
        ref_yaml=master_inputs[swaths[0]].yaml_path,
        union_bbox=union_bbox,
        dem_on_mismatch=args.dem_on_mismatch,
    )
    _log(f"DEM: {dem_path}")

    swath_run_dirs: Dict[str, Path] = {}
    swath_exec: Dict[str, Dict[str, object]] = {}
    if not args.prepare_only:
        burst_unit_dirname = "bursts" if burst_style == "isce2" else "burst_groups"
        burst_id_prefix = "b" if burst_style == "isce2" else "g"
        burst_processing_mode = "burst" if burst_style == "isce2" else "burst_group"
        for sw in swaths:
            _log(f"\n=== 处理 {sw.upper()} ===")
            sw_out = runs_dir / sw
            if dem_path is None:
                raise RuntimeError("内部错误：dem_path 为空")
            use_burst_mode = (args.burst_split_mode != "off")
            if not use_burst_mode:
                if not bool(args.allow_swath_fallback):
                    swath_exec[sw] = {
                        "success": False,
                        "attempts": [],
                        "final_error": "swath_mode_disabled(use_burst_mode_or_allow_swath_fallback)",
                        "run_dir": str(sw_out),
                        "processing_mode": "swath_blocked",
                    }
                    _log(f"{sw.upper()} 已阻止整 swath 模式（未开启 --allow-swath-fallback）")
                    continue
                success, attempts, final_error = _run_with_retries(
                    run_label=sw.upper(),
                    master=master_inputs[sw],
                    slave=slave_inputs[sw],
                    run_dir=sw_out,
                    dem=dem_path,
                    args=args,
                    registration_esd_force_apply_low_reliability=effective_esd_force_apply_low_reliability,
                )
                swath_exec[sw] = {
                    "success": bool(success),
                    "attempts": attempts,
                    "final_error": final_error if not success else None,
                    "run_dir": str(sw_out),
                    "processing_mode": "swath",
                }
                if success:
                    swath_run_dirs[sw] = sw_out
                else:
                    _log(f"{sw.upper()} 最终失败，已记录日志并继续处理其他 swath")
                continue

            burst_prep_dir = prep_dir / burst_unit_dirname / sw
            try:
                group_plans = _prepare_burst_group_inputs(
                    master=master_inputs[sw],
                    slave=slave_inputs[sw],
                    out_dir=burst_prep_dir,
                    bursts_per_group=int(effective_bursts_per_group),
                    burst_margin_lines=int(effective_burst_margin_lines),
                    orbit_margin_hours=float(args.orbit_margin_hours),
                    min_common_bursts=int(effective_burst_min_common),
                    az_looks=int(nalks),
                    id_prefix=burst_id_prefix,
                )
            except Exception as e:
                if args.burst_split_mode == "force":
                    swath_exec[sw] = {
                        "success": False,
                        "attempts": [],
                        "final_error": f"{burst_processing_mode}_prepare_failed: {e}",
                        "run_dir": str(sw_out),
                        "processing_mode": burst_processing_mode,
                        "burst_group_count": 0,
                    }
                    _log(f"{sw.upper()} burst 分组准备失败（force 模式）：{e}")
                    continue
                if not bool(args.allow_swath_fallback):
                    swath_exec[sw] = {
                        "success": False,
                        "attempts": [],
                        "final_error": f"{burst_processing_mode}_prepare_failed(no_swath_fallback): {e}",
                        "run_dir": str(sw_out),
                        "processing_mode": burst_processing_mode,
                        "burst_group_count": 0,
                    }
                    _log(f"{sw.upper()} burst 分组准备失败，且未开启 swath 回退：{e}")
                    continue
                _log(f"{sw.upper()} burst 分组准备失败，回退整 swath: {e}")
                group_plans = []

            if not group_plans:
                if args.burst_split_mode == "force":
                    swath_exec[sw] = {
                        "success": False,
                        "attempts": [],
                        "final_error": f"{burst_processing_mode}_unavailable",
                        "run_dir": str(sw_out),
                        "processing_mode": burst_processing_mode,
                        "burst_group_count": 0,
                    }
                    _log(f"{sw.upper()} 未找到可用 burst-group（force 模式），记为失败")
                    continue
                if not bool(args.allow_swath_fallback):
                    swath_exec[sw] = {
                        "success": False,
                        "attempts": [],
                        "final_error": f"{burst_processing_mode}_unavailable(no_swath_fallback)",
                        "run_dir": str(sw_out),
                        "processing_mode": burst_processing_mode,
                        "burst_group_count": 0,
                    }
                    _log(f"{sw.upper()} 无可用 burst-group，且未开启 swath 回退，记为失败")
                    continue
                _log(f"{sw.upper()} 无可用 burst-group，回退整 swath 处理")
                success, attempts, final_error = _run_with_retries(
                    run_label=sw.upper(),
                    master=master_inputs[sw],
                    slave=slave_inputs[sw],
                    run_dir=sw_out,
                    dem=dem_path,
                    args=args,
                    registration_esd_force_apply_low_reliability=effective_esd_force_apply_low_reliability,
                )
                swath_exec[sw] = {
                    "success": bool(success),
                    "attempts": attempts,
                    "final_error": final_error if not success else None,
                    "run_dir": str(sw_out),
                    "processing_mode": "swath_fallback",
                }
                if success:
                    swath_run_dirs[sw] = sw_out
                else:
                    _log(f"{sw.upper()} 最终失败，已记录日志并继续处理其他 swath")
                continue

            if burst_style == "isce2":
                _log(f"{sw.upper()} 使用 ISCE2 风格 burst 模式，共 {len(group_plans)} 个 burst")
            else:
                _log(f"{sw.upper()} 使用 burst-group 模式，共 {len(group_plans)} 组")
            group_runs: Dict[str, Path] = {}
            group_exec: Dict[str, Dict[str, object]] = {}
            for gp in group_plans:
                gid = str(gp["group_id"])
                g_master = gp["master"]
                g_slave = gp["slave"]
                if not isinstance(g_master, SwathInput) or not isinstance(g_slave, SwathInput):
                    raise RuntimeError("内部错误：burst-group 输入类型异常")
                if burst_style == "isce2" and len(gp.get("matched_burst_pairs", []) or []) != 1:
                    raise RuntimeError(f"ISCE2 风格要求每个处理单元仅含 1 个 burst，对象 {gid} 不满足")
                g_run = sw_out / burst_unit_dirname / gid
                ok, attempts, ferr = _run_with_retries(
                    run_label=f"{sw.upper()}:{gid}",
                    master=g_master,
                    slave=g_slave,
                    run_dir=g_run,
                    dem=dem_path,
                    args=args,
                    stop_after_wrapped_phase=True,
                    registration_esd_force_apply_low_reliability=effective_esd_force_apply_low_reliability,
                )
                group_exec[gid] = {
                    "success": bool(ok),
                    "attempts": attempts,
                    "final_error": ferr if not ok else None,
                    "run_dir": str(g_run),
                    "master_row_range": gp.get("master_row_range", []),
                    "slave_row_range": gp.get("slave_row_range", []),
                    "matched_burst_pairs": gp.get("matched_burst_pairs", []),
                }
                if ok:
                    group_runs[gid] = g_run

            final_error: Optional[str] = None
            partial = False
            post_info: Dict[str, object] = {}
            failed_groups = [g for g in group_exec.keys() if g not in group_runs]
            if failed_groups:
                partial = True
                _log(f"{sw.upper()} 存在失败 burst-group: {failed_groups}")

            if not group_runs:
                sw_success = False
                final_error = f"all_groups_failed({len(group_plans)})"
            else:
                try:
                    post_info = _run_swath_mosaic_from_burst_groups(
                        swath=sw,
                        swath_out_dir=sw_out,
                        master_input=master_inputs[sw],
                        dem=dem_path,
                        group_exec=group_exec,
                        args=args,
                    )
                    sw_success = True
                    _log(f"{sw.upper()} swath 级 mosaic 完成（wrapped/coherence）")
                    if partial:
                        final_error = f"partial_groups_failed({len(failed_groups)}/{len(group_plans)})"
                except Exception as e:
                    sw_success = False
                    ferr = f"swath_mosaic_failed: {e}"
                    if partial:
                        ferr += f"; partial_groups_failed({len(failed_groups)}/{len(group_plans)})"
                    final_error = ferr
                    _log(f"{sw.upper()} swath 级 mosaic 失败: {e}")

            swath_exec[sw] = {
                "success": bool(sw_success),
                "attempts": [],
                "final_error": final_error if (not sw_success or partial) else None,
                "run_dir": str(sw_out),
                "processing_mode": burst_processing_mode,
                "burst_processing_style": burst_style,
                "burst_group_count": len(group_plans),
                "burst_group_success_count": len(group_runs),
                "partial_group_failure": bool(partial),
                "failed_groups": failed_groups,
                "burst_groups": group_exec,
                "swath_mosaic": post_info,
            }
            if sw_success:
                swath_run_dirs[sw] = sw_out
            else:
                _log(f"{sw.upper()} 所有 burst-group 均失败，已继续处理其他 swath")
    else:
        for sw in swaths:
            swath_exec[sw] = {
                "success": False,
                "skipped": True,
                "attempts": [],
                "final_error": "prepare_only",
                "run_dir": str(runs_dir / sw),
            }

    if not args.prepare_only:
        burst_swaths = [
            s
            for s in swaths
            if bool(swath_exec.get(s, {}).get("success", False))
            and str(swath_exec.get(s, {}).get("processing_mode", "")) in ("burst_group", "burst")
        ]
        if burst_swaths:
            _log("\n=== 全部 swath burst-mosaic 完成，开始统一执行解缠与地理编码 ===")
        for sw in burst_swaths:
            rec = swath_exec.get(sw, {})
            sw_out = Path(str(rec.get("run_dir", runs_dir / sw))).expanduser().resolve()
            mosaic_info = rec.get("swath_mosaic", {})
            if not isinstance(mosaic_info, dict) or not mosaic_info:
                rec["success"] = False
                prev = str(rec.get("final_error") or "").strip()
                ferr = "missing_swath_mosaic_info_for_unwrap"
                rec["final_error"] = (prev + "; " + ferr) if prev else ferr
                swath_run_dirs.pop(sw, None)
                _log(f"{sw.upper()} 缺少 swath_mosaic 信息，无法执行解缠/地理编码")
                continue
            try:
                ug = _run_swath_unwrap_geocode(
                    swath=sw,
                    swath_out_dir=sw_out,
                    mosaic_info=mosaic_info,
                    args=args,
                )
                rec["swath_unwrap_geocode"] = ug
                _log(f"{sw.upper()} 解缠与地理编码完成（mosaic 之后）")
            except Exception as e:
                rec["success"] = False
                prev = str(rec.get("final_error") or "").strip()
                ferr = f"unwrap_after_mosaic_failed: {e}"
                rec["final_error"] = (prev + "; " + ferr) if prev else ferr
                swath_run_dirs.pop(sw, None)
                _log(f"{sw.upper()} 解缠/地理编码失败: {e}")

    succeeded_swaths = [s for s in swaths if bool(swath_exec.get(s, {}).get("success", False))]
    failed_swaths = [s for s in swaths if s not in succeeded_swaths and not bool(swath_exec.get(s, {}).get("skipped", False))]
    partial_swaths = [s for s in swaths if bool(swath_exec.get(s, {}).get("partial_group_failure", False))]

    mosaic_outputs: Dict[str, str] = {}
    if (not args.prepare_only) and (not args.skip_stitch):
        if not swath_run_dirs:
            _log("警告: 没有成功的 swath 结果，跳过拼接")
        else:
            if failed_swaths:
                _log(f"警告: 存在失败 swath（{failed_swaths}），将仅拼接成功 swath")
            if partial_swaths:
                _log(f"警告: 存在部分失败 swath（{partial_swaths}），结果可能为部分覆盖")
        mosaicked = _mosaic_geo_products(swath_run_dirs, mosaic_dir)
        mosaic_outputs = {k: str(v) for k, v in mosaicked.items()}
        _log(f"拼接完成，产品数: {len(mosaic_outputs)}")

    swath_report = {
        "created_at": dt.datetime.now().isoformat(),
        "policy": {
            "swath_retries": int(args.swath_retries),
            "swath_retry_backoff_seconds": float(args.swath_retry_backoff_seconds),
            "allow_partial_success": bool(args.allow_partial_success),
            "burst_split_mode": str(args.burst_split_mode),
            "burst_processing_style": burst_style,
            "allow_swath_fallback": bool(args.allow_swath_fallback),
            "unwrap_after_mosaic": True,
            "bursts_per_group": int(effective_bursts_per_group),
            "burst_margin_lines": int(effective_burst_margin_lines),
            "burst_min_common": int(effective_burst_min_common),
            "registration_esd": (not bool(args.no_registration_esd)),
            "registration_esd_force_apply_low_reliability": bool(effective_esd_force_apply_low_reliability),
        },
        "summary": {
            "total_swaths": len(swaths),
            "succeeded_swaths": succeeded_swaths,
            "failed_swaths": failed_swaths,
            "partial_swaths": partial_swaths,
            "prepare_only": bool(args.prepare_only),
        },
        "swaths": swath_exec,
    }
    swath_report_path = out_root / "swath_execution_report.yaml"
    with open(swath_report_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(swath_report, f, allow_unicode=True, sort_keys=False)

    swath_summary_txt = out_root / "swath_failure_summary.txt"
    lines: List[str] = []
    lines.append("Sentinel-1 swath execution summary")
    lines.append(f"Created at: {swath_report['created_at']}")
    lines.append(f"Swaths: {', '.join(swaths)}")
    lines.append(f"Succeeded: {', '.join(succeeded_swaths) if succeeded_swaths else '(none)'}")
    lines.append(f"Failed: {', '.join(failed_swaths) if failed_swaths else '(none)'}")
    lines.append(f"Partial: {', '.join(partial_swaths) if partial_swaths else '(none)'}")
    lines.append("")
    for sw in swaths:
        rec = swath_exec.get(sw, {})
        lines.append(f"[{sw}] success={rec.get('success', False)}")
        lines.append(f"  mode: {rec.get('processing_mode', 'swath')}")
        if rec.get("skipped", False):
            lines.append("  skipped: prepare_only")
            lines.append("")
            continue
        if rec.get("processing_mode") in ("burst_group", "burst"):
            lines.append(f"  burst_style: {rec.get('burst_processing_style', burst_style)}")
            lines.append(
                f"  burst_groups: total={rec.get('burst_group_count', 0)} "
                f"success={rec.get('burst_group_success_count', 0)} "
                f"partial_failure={rec.get('partial_group_failure', False)}"
            )
            lines.append(f"  unwrap_after_mosaic: {('swath_unwrap_geocode' in rec)}")
            bgs = rec.get("burst_groups", {})
            if isinstance(bgs, dict):
                for gid, grec in sorted(bgs.items()):
                    lines.append(f"  [{gid}] success={grec.get('success', False)} run_dir={grec.get('run_dir')}")
                    lines.append(
                        f"    master_rows={grec.get('master_row_range')} "
                        f"slave_rows={grec.get('slave_row_range')} "
                        f"pairs={len(grec.get('matched_burst_pairs', []))}"
                    )
                    for a in grec.get("attempts", []):
                        lines.append(
                            f"    attempt={a.get('attempt')} rc={a.get('return_code')} "
                            f"duration={float(a.get('duration_sec', 0.0)):.2f}s log={a.get('log_file')}"
                        )
                    if grec.get("final_error"):
                        lines.append(f"    final_error: {grec.get('final_error')}")
            sm = rec.get("swath_mosaic", {})
            if isinstance(sm, dict) and sm:
                gsm = sm.get("group_stitch_meta", {})
                align_cnt = 0
                if isinstance(gsm, dict):
                    align_cnt = sum(1 for _gid, _m in gsm.items() if bool((_m or {}).get("phase_bias_applied", False)))
                lines.append(
                    f"  swath_mosaic: merged_groups={len(sm.get('merged_groups', []))} "
                    f"skipped_groups={len(sm.get('skipped_groups', {}))} "
                    f"ml_shape={sm.get('ml_shape')} "
                    f"phase_aligned_groups={align_cnt}"
                )
            su = rec.get("swath_unwrap_geocode", {})
            if isinstance(su, dict) and su:
                lines.append(f"  swath_unwrap: unwrapped_bin={su.get('unwrapped_bin')}")
        else:
            for a in rec.get("attempts", []):
                lines.append(
                    f"  attempt={a.get('attempt')} rc={a.get('return_code')} "
                    f"duration={float(a.get('duration_sec', 0.0)):.2f}s log={a.get('log_file')}"
                )
        ferr = rec.get("final_error")
        if ferr:
            lines.append(f"  final_error: {ferr}")
            last_attempts = rec.get("attempts", [])
            if last_attempts:
                tail = str(last_attempts[-1].get("log_tail", "")).strip()
                if tail:
                    lines.append("  last_log_tail:")
                    for tline in tail.splitlines():
                        lines.append("    " + tline)
        lines.append("")
    swath_summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Swath执行报告: {swath_report_path}")
    _log(f"Swath错误摘要: {swath_summary_txt}")

    manifest = {
        "data_dir": str(data_dir),
        "output_dir": str(out_root),
        "master_zip": str(master_scene.zip_path),
        "slave_zip": str(slave_scene.zip_path),
        "orbit_master": str(m_orbit),
        "orbit_slave": str(s_orbit),
        "dem": str(dem_path) if dem_path is not None else None,
        "pol": args.pol,
        "swaths": swaths,
        "prepare_only": bool(args.prepare_only),
        "skip_stitch": bool(args.skip_stitch),
        "swath_retries": int(args.swath_retries),
        "swath_retry_backoff_seconds": float(args.swath_retry_backoff_seconds),
        "allow_partial_success": bool(args.allow_partial_success),
        "burst_split_mode": str(args.burst_split_mode),
        "burst_processing_style": burst_style,
        "allow_swath_fallback": bool(args.allow_swath_fallback),
        "unwrap_after_mosaic": True,
        "bursts_per_group": int(effective_bursts_per_group),
        "burst_margin_lines": int(effective_burst_margin_lines),
        "burst_min_common": int(effective_burst_min_common),
        "registration_esd": (not bool(args.no_registration_esd)),
        "registration_esd_force_apply_low_reliability": bool(effective_esd_force_apply_low_reliability),
        "swath_status": swath_exec,
        "swath_report_yaml": str(swath_report_path),
        "swath_report_text": str(swath_summary_txt),
        "prepared_inputs": {
            "master": {
                s: {
                    "vrt": str(master_inputs[s].vrt_path),
                    "yaml": str(master_inputs[s].yaml_path),
                    "measurement_member": master_inputs[s].measurement_member,
                    "annotation_member": master_inputs[s].annotation_member,
                    "source_zip": str(master_inputs[s].source_zip),
                }
                for s in swaths
            },
            "slave": {
                s: {
                    "vrt": str(slave_inputs[s].vrt_path),
                    "yaml": str(slave_inputs[s].yaml_path),
                    "measurement_member": slave_inputs[s].measurement_member,
                    "annotation_member": slave_inputs[s].annotation_member,
                    "source_zip": str(slave_inputs[s].source_zip),
                }
                for s in swaths
            },
        },
        "swath_run_dirs": {k: str(v) for k, v in swath_run_dirs.items()},
        "mosaic_geo_products": mosaic_outputs,
    }
    manifest_path = out_root / "s1_zip_dinsar_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    _log(f"Manifest: {manifest_path}")

    problem_swaths = sorted(set(failed_swaths + partial_swaths))
    if problem_swaths and not bool(args.allow_partial_success):
        _log(
            f"存在失败/部分失败 swath 且未开启 --allow-partial-success，返回非零退出码。"
            f"problem_swaths={problem_swaths}"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
