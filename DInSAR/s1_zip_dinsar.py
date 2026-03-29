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
import datetime as dt
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import requests
import yaml
from osgeo import gdal


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


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    _log("[RUN] " + " ".join(str(x) for x in cmd))
    if cwd is not None:
        _log(f"[CWD] {cwd}")
    subprocess.run(list(cmd), check=True, cwd=str(cwd) if cwd else None)


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
        "corners": corners,
    }


def _parse_annotation_corners(root: ET.Element) -> Dict[str, Dict[str, float]]:
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
) -> Path:
    if dem_path is not None:
        d = dem_path.expanduser().resolve()
        if not d.exists():
            raise FileNotFoundError(f"DEM 不存在: {d}")
        return d
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
) -> None:
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
    if force:
        cmd.append("--force")
    if snaphu_tile_rows is not None and snaphu_tile_cols is not None:
        cmd += ["--snaphu-tile-rows", str(snaphu_tile_rows), "--snaphu-tile-cols", str(snaphu_tile_cols)]
    if snaphu_nproc is not None:
        cmd += ["--snaphu-nproc", str(snaphu_nproc)]
    _run(cmd)


def _mosaic_geo_products(swath_dirs: Dict[str, Path], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prod_map: Dict[str, List[Path]] = {}
    for sw, wd in swath_dirs.items():
        gp = wd / "gcd" / "geo_products"
        if not gp.exists():
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
        cmd_vrt = ["gdalbuildvrt", "-overwrite", str(vrt)] + [str(x) for x in files]
        _run(cmd_vrt)
        _run(
            [
                "gdal_translate",
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=LZW",
                "-co",
                "TILED=YES",
                str(vrt),
                str(out_tif),
            ]
        )
        out_files[name] = out_tif.resolve()
    return out_files


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
    parser.add_argument("--no-registration-esd", action="store_true", help="关闭 S1 默认 ESD 配准微调")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
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
    _log(f"选择 master: {master_scene.zip_path.name}")
    _log(f"选择 slave : {slave_scene.zip_path.name}")
    _log(f"处理 swaths: {swaths}, pol={args.pol}")

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

    union_bbox = _build_union_bbox([master_inputs[s].yaml_path for s in swaths])
    dem_path = _ensure_dem(
        dem_path=Path(args.dem).expanduser() if args.dem else None,
        dem_dir=dem_dir,
        ref_yaml=master_inputs[swaths[0]].yaml_path,
        union_bbox=union_bbox,
    )
    _log(f"DEM: {dem_path}")

    swath_run_dirs: Dict[str, Path] = {}
    if not args.prepare_only:
        for sw in swaths:
            _log(f"\n=== 处理 {sw.upper()} ===")
            sw_out = runs_dir / sw
            if dem_path is None:
                raise RuntimeError("内部错误：dem_path 为空")
            _run_dinsar_swath(
                master=master_inputs[sw],
                slave=slave_inputs[sw],
                swath_out_dir=sw_out,
                dem=dem_path,
                multilook=args.multilook,
                force=bool(args.force),
                dem_on_mismatch=args.dem_on_mismatch,
                snaphu_tile_rows=args.snaphu_tile_rows,
                snaphu_tile_cols=args.snaphu_tile_cols,
                snaphu_nproc=args.snaphu_nproc,
                registration_esd=(not bool(args.no_registration_esd)),
            )
            swath_run_dirs[sw] = sw_out

    mosaic_outputs: Dict[str, str] = {}
    if (not args.prepare_only) and (not args.skip_stitch):
        mosaicked = _mosaic_geo_products(swath_run_dirs, mosaic_dir)
        mosaic_outputs = {k: str(v) for k, v in mosaicked.items()}
        _log(f"拼接完成，产品数: {len(mosaic_outputs)}")

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
        "registration_esd": (not bool(args.no_registration_esd)),
        "prepared_inputs": {
            "master": {
                s: {
                    "vrt": str(master_inputs[s].vrt_path),
                    "yaml": str(master_inputs[s].yaml_path),
                    "measurement_member": master_inputs[s].measurement_member,
                    "annotation_member": master_inputs[s].annotation_member,
                }
                for s in swaths
            },
            "slave": {
                s: {
                    "vrt": str(slave_inputs[s].vrt_path),
                    "yaml": str(slave_inputs[s].yaml_path),
                    "measurement_member": slave_inputs[s].measurement_member,
                    "annotation_member": slave_inputs[s].annotation_member,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
