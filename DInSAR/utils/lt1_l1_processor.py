#!/usr/bin/env python3
"""
LuTan-1 (LT1) L1A importer for DInSAR.

It reads LT1 `*.meta.xml` + complex TIFF and exports a DInSAR-ready triplet:
  - <name>.tiff
  - <name>.vrt
  - <name>.yaml
"""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from osgeo import gdal

C_LIGHT = 299792458.0

# Keep current behavior and silence GDAL 4.0 future warning in runtime.
gdal.DontUseExceptions()


@dataclass
class ProductFiles:
    product_dir: Path
    base_name: str
    meta_xml: Path
    tiff: Path
    incidence_xml: Optional[Path]
    rpc: Optional[Path]


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return -1


def _validate_input_product(product: ProductFiles) -> None:
    meta_sz = _file_size(product.meta_xml)
    tif_sz = _file_size(product.tiff)

    errs: list[str] = []
    if meta_sz <= 0:
        errs.append(f"meta.xml is empty: {product.meta_xml}")
    if tif_sz <= 0:
        errs.append(f"tiff is empty: {product.tiff}")

    # Optional files: if empty, ignore them later.
    if product.incidence_xml is not None and _file_size(product.incidence_xml) <= 0:
        product.incidence_xml = None
    if product.rpc is not None and _file_size(product.rpc) <= 0:
        product.rpc = None

    if not errs:
        # Make sure source TIFF itself is readable before creating output links.
        ds = gdal.Open(str(product.tiff), gdal.GA_ReadOnly)
        if ds is None:
            errs.append(f"tiff cannot be opened by GDAL: {product.tiff}")
        ds = None

    if errs:
        tgz = product.product_dir.parent / f"{product.base_name}.tar.gz"
        msg = "Invalid LT1 input product:\n  - " + "\n  - ".join(errs)
        if tgz.exists() and _file_size(tgz) > 0:
            msg += (
                f"\nPossible fix: extract original archive first:\n"
                f"  tar -xzf {tgz} -C {product.product_dir.parent}"
            )
        raise RuntimeError(msg)


def _find_text(root: ET.Element, paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        node = root.find(p)
        if node is not None and node.text is not None:
            val = node.text.strip()
            if val:
                return val
    return None


def _to_float(v: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Optional[str], default: Optional[int] = None) -> Optional[int]:
    fv = _to_float(v, None)
    if fv is None:
        return default
    return int(round(fv))


def _base_from_meta_name(meta_name: str) -> str:
    if meta_name.endswith(".meta.xml"):
        return meta_name[:-9]
    if meta_name.endswith(".xml"):
        return meta_name[:-4]
    return Path(meta_name).stem


def _discover_product(input_path: Path) -> ProductFiles:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    meta_xml: Optional[Path] = None
    tiff: Optional[Path] = None
    product_dir: Path
    base_name: str

    if input_path.is_dir():
        product_dir = input_path
        metas = sorted(product_dir.glob("*.meta.xml"))
        if not metas:
            raise FileNotFoundError(f"No *.meta.xml found in: {product_dir}")
        if len(metas) > 1:
            raise RuntimeError(
                f"Multiple *.meta.xml found in {product_dir}; please pass explicit meta.xml path."
            )
        meta_xml = metas[0]
        base_name = _base_from_meta_name(meta_xml.name)
        for ext in (".tiff", ".tif"):
            cand = product_dir / f"{base_name}{ext}"
            if cand.exists():
                tiff = cand
                break
        if tiff is None:
            tifs = sorted(list(product_dir.glob("*.tiff")) + list(product_dir.glob("*.tif")))
            if len(tifs) == 1:
                tiff = tifs[0]
            else:
                raise FileNotFoundError(f"Cannot uniquely determine TIFF in: {product_dir}")
    else:
        product_dir = input_path.parent
        name_lower = input_path.name.lower()
        if name_lower.endswith(".meta.xml"):
            meta_xml = input_path
            base_name = _base_from_meta_name(meta_xml.name)
            for ext in (".tiff", ".tif"):
                cand = product_dir / f"{base_name}{ext}"
                if cand.exists():
                    tiff = cand
                    break
        elif input_path.suffix.lower() in (".tif", ".tiff"):
            tiff = input_path
            base_name = input_path.stem
            for cand in (
                product_dir / f"{base_name}.meta.xml",
                product_dir / f"{base_name}.xml",
            ):
                if cand.exists():
                    meta_xml = cand
                    break
        else:
            raise RuntimeError(
                "Input must be product directory, *.meta.xml, or *.tif/*.tiff file."
            )

    if meta_xml is None or not meta_xml.exists():
        raise FileNotFoundError("meta.xml not found")
    if tiff is None or not tiff.exists():
        raise FileNotFoundError("TIFF not found")

    incidence_xml = product_dir / f"{base_name}.incidence.xml"
    rpc = product_dir / f"{base_name}.rpc"

    return ProductFiles(
        product_dir=product_dir,
        base_name=base_name,
        meta_xml=meta_xml,
        tiff=tiff,
        incidence_xml=incidence_xml if incidence_xml.exists() else None,
        rpc=rpc if rpc.exists() else None,
    )


def _safe_link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir():
            raise IsADirectoryError(dst)
        dst.unlink()

    # If output already points to the same path, keep it.
    try:
        if src.resolve() == dst.resolve():
            return
    except Exception:
        pass

    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except Exception:
            pass
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src.resolve())
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def _create_vrt(src_tiff: Path, out_vrt: Path) -> None:
    ds = gdal.Open(str(src_tiff), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open TIFF: {src_tiff}")
    out_ds = gdal.Translate(str(out_vrt), ds, format="VRT")
    ds = None
    if out_ds is None:
        raise RuntimeError(f"Failed to create VRT: {out_vrt}")
    out_ds = None


def _read_raster_info(path: Path) -> Dict[str, Any]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    try:
        band1 = ds.GetRasterBand(1)
        dt = gdal.GetDataTypeName(band1.DataType) if band1 is not None else "Unknown"
        info = {
            "width": int(ds.RasterXSize),
            "height": int(ds.RasterYSize),
            "bands": int(ds.RasterCount),
            "dtype": dt,
        }
    finally:
        ds = None
    return info


def _parse_corners(scene_info: ET.Element, nrows: int, ncols: int) -> tuple[Dict[str, Dict[str, float]], list[dict]]:
    key_map = {
        "topLeft": "top_left",
        "topRight": "top_right",
        "bottomLeft": "bottom_left",
        "bottomRight": "bottom_right",
    }
    px_map = {
        "top_left": (0.0, 0.0),
        "top_right": (float(ncols - 1), 0.0),
        "bottom_left": (0.0, float(nrows - 1)),
        "bottom_right": (float(ncols - 1), float(nrows - 1)),
    }

    corners: Dict[str, Dict[str, float]] = {}
    geo_grid_tmp: Dict[str, dict] = {}

    for c in scene_info.findall("sceneCornerCoord"):
        raw_name = c.attrib.get("name", "")
        key = key_map.get(raw_name)
        if not key:
            continue
        lat = _to_float(_find_text(c, ("lat",)), None)
        lon = _to_float(_find_text(c, ("lon",)), None)
        if lat is None or lon is None:
            continue
        px, py = px_map[key]
        corners[key] = {"lat": float(lat), "lon": float(lon), "x": px, "y": py}

        geo_grid_tmp[key] = {
            "line": int(py),
            "pixel": int(px),
            "latitude": float(lat),
            "longitude": float(lon),
            "azimuth_time": _find_text(c, ("azimuthTimeUTC",)),
            "slant_range_time": _to_float(_find_text(c, ("rangeTime",)), None),
            "incidence_angle": _to_float(_find_text(c, ("incidenceAngle",)), None),
        }

    # Keep output structure stable.
    ordered_keys = ("top_left", "top_right", "bottom_left", "bottom_right")
    corners_out = {k: corners[k] for k in ordered_keys if k in corners}
    geo_grid_out = [geo_grid_tmp[k] for k in ordered_keys if k in geo_grid_tmp]
    return corners_out, geo_grid_out


def _parse_orbit(root: ET.Element) -> list[dict]:
    orbit_points: list[dict] = []
    for sv in root.findall("./platform/orbit/stateVec"):
        t = _find_text(sv, ("timeUTC",))
        px = _to_float(_find_text(sv, ("posX",)), None)
        py = _to_float(_find_text(sv, ("posY",)), None)
        pz = _to_float(_find_text(sv, ("posZ",)), None)
        vx = _to_float(_find_text(sv, ("velX",)), None)
        vy = _to_float(_find_text(sv, ("velY",)), None)
        vz = _to_float(_find_text(sv, ("velZ",)), None)
        if t is None or None in (px, py, pz, vx, vy, vz):
            continue
        orbit_points.append(
            {
                "time": t,
                "position": {"x": float(px), "y": float(py), "z": float(pz)},
                "velocity": {"vx": float(vx), "vy": float(vy), "vz": float(vz)},
            }
        )

    def _time_key(pt: dict) -> float:
        tv = pt.get("time")
        try:
            return datetime.fromisoformat(str(tv)).timestamp()
        except Exception:
            return math.inf

    orbit_points.sort(key=_time_key)
    return orbit_points


def _parse_lt1_metadata(meta_xml: Path, raster_info: Dict[str, Any], output_data_file: str) -> Dict[str, Any]:
    root = ET.parse(meta_xml).getroot()

    nrows_xml = _to_int(_find_text(root, ("./productInfo/imageDataInfo/imageRaster/numberOfRows",)), None)
    ncols_xml = _to_int(_find_text(root, ("./productInfo/imageDataInfo/imageRaster/numberOfColumns",)), None)
    nrows = nrows_xml if nrows_xml is not None else int(raster_info["height"])
    ncols = ncols_xml if ncols_xml is not None else int(raster_info["width"])

    row_spacing = _to_float(
        _find_text(root, ("./productInfo/imageDataInfo/imageRaster/rowSpacing",)),
        0.0,
    )
    col_spacing = _to_float(
        _find_text(root, ("./productInfo/imageDataInfo/imageRaster/columnSpacing",)),
        0.0,
    )
    first_pixel_time = _to_float(
        _find_text(root, ("./productInfo/sceneInfo/rangeTime/firstPixel",)),
        None,
    )
    last_pixel_time = _to_float(
        _find_text(root, ("./productInfo/sceneInfo/rangeTime/lastPixel",)),
        None,
    )

    near_range = C_LIGHT * first_pixel_time / 2.0 if first_pixel_time is not None else None
    far_range = C_LIGHT * last_pixel_time / 2.0 if last_pixel_time is not None else None

    prf = _to_float(
        _find_text(root, ("./instrument/settings/settingRecord/PRF",)),
        None,
    )
    center_frequency = _to_float(
        _find_text(
            root,
            (
                "./instrument/radarParameters/centerFrequency",
                "./processing/processingParameter/rangeCompression/chirps/referenceChirp/centerFrequency",
            ),
        ),
        None,
    )
    wavelength = C_LIGHT / center_frequency if center_frequency and center_frequency > 0 else None

    pulse_duration = _to_float(
        _find_text(
            root,
            ("./processing/processingParameter/rangeCompression/chirps/referenceChirp/pulseLength",),
        ),
        None,
    )
    pulse_bandwidth = _to_float(
        _find_text(
            root,
            ("./processing/processingParameter/rangeCompression/chirps/referenceChirp/pulseBandwidth",),
        ),
        None,
    )
    chirp_slope_tag = _find_text(
        root,
        ("./processing/processingParameter/rangeCompression/chirps/referenceChirp/chirpSlope",),
    )
    chirp_slope = None
    if pulse_duration and pulse_duration > 0 and pulse_bandwidth is not None:
        chirp_slope = pulse_bandwidth / pulse_duration
        if str(chirp_slope_tag or "").strip().upper() == "DOWN":
            chirp_slope = -abs(chirp_slope)

    range_sampling_rate = None
    if col_spacing and col_spacing > 0:
        range_sampling_rate = C_LIGHT / (2.0 * col_spacing)

    first_line_time = _find_text(root, ("./productInfo/sceneInfo/start/timeUTC",))
    last_line_time = _find_text(root, ("./productInfo/sceneInfo/stop/timeUTC",))
    abs_orbit = _find_text(root, ("./productInfo/missionInfo/absOrbit",))
    orbit_direction = _find_text(root, ("./productInfo/missionInfo/orbitDirection",))
    look_direction = _find_text(root, ("./productInfo/acquisitionInfo/lookDirection",))
    mission = _find_text(root, ("./productInfo/missionInfo/mission", "./generalHeader/mission")) or "LT1"
    sensor_mode = _find_text(root, ("./productInfo/acquisitionInfo/imagingMode", "./generalHeader/sensorMode"))
    polarization = _find_text(
        root,
        (
            "./productInfo/acquisitionInfo/polarisationMode",
            "./productInfo/acquisitionInfo/polarisationList/polLayer",
        ),
    )

    scene_info = root.find("./productInfo/sceneInfo")
    corners: Dict[str, Dict[str, float]] = {}
    geolocation_grid: list[dict] = []
    if scene_info is not None:
        corners, geolocation_grid = _parse_corners(scene_info, nrows=nrows, ncols=ncols)

    orbit_points = _parse_orbit(root)

    looks_range = _to_int(_find_text(root, ("./processing/processingParameter/rangeLooks",)), 1)
    looks_az = _to_int(_find_text(root, ("./processing/processingParameter/azimuthLooks",)), 1)

    radar_params = {
        "wavelength": float(wavelength) if wavelength is not None else 0.0,
        "prf": float(prf) if prf is not None else 0.0,
        "pulse_duration": float(pulse_duration) if pulse_duration is not None else 0.0,
        "near_range": float(near_range) if near_range is not None else 0.0,
        "far_range": float(far_range) if far_range is not None else 0.0,
        "range_spacing": float(col_spacing) if col_spacing is not None else 0.0,
        "range_pixel_spacing": float(col_spacing) if col_spacing is not None else 0.0,
        "azimuth_spacing": float(row_spacing) if row_spacing is not None else 0.0,
        "range_sampling_rate": float(range_sampling_rate) if range_sampling_rate is not None else 0.0,
        "chirp_slope": float(chirp_slope) if chirp_slope is not None else 0.0,
        "look_dir": (look_direction or "").upper(),
        "fd1": 0.0,
        "fdd1": 0.0,
        "fddd1": 0.0,
    }

    image_parameters = {
        "nrows": int(nrows),
        "ncols": int(ncols),
        "data_format": "complex_int16" if int(raster_info.get("bands", 1)) >= 2 else "float32",
        "bands": ["real", "imaginary"] if int(raster_info.get("bands", 1)) >= 2 else ["amplitude"],
        "byte_order": "little_endian",
    }

    metadata = {
        "satellite": mission,
        "sensor": sensor_mode or "SAR",
        "absolute_orbit_number": str(abs_orbit or ""),
        "creation_time": datetime.now().astimezone().isoformat(),
        "data_file": output_data_file,
        "data_type": "SLC",
        "first_line_sensing_time": first_line_time or "",
        "last_line_sensing_time": last_line_time or "",
        "sensor_start": first_line_time or "",
        "sensor_end": last_line_time or "",
        "polarization": polarization or "",
        "version": "1.0",
    }

    orbit_parameters = {
        "orbit_direction": orbit_direction or "",
        "look_direction": (look_direction or "").upper(),
    }

    ycfg: Dict[str, Any] = {
        "metadata": metadata,
        "sensor_start": first_line_time or "",
        "sensor_end": last_line_time or "",
        "image_parameters": image_parameters,
        "radar_parameters": radar_params,
        "orbit_parameters": orbit_parameters,
        "processing_parameters": {
            "range_looks": int(looks_range or 1),
            "azimuth_looks": int(looks_az or 1),
        },
        "orbit_data": {"orbit_points": orbit_points},
        "corner_coordinates": corners,
        "geolocation_grid": geolocation_grid,
        "orbit_ascending": str(orbit_direction or "").upper() == "ASCENDING",
        "range_pixel_spacing": float(col_spacing) if col_spacing is not None else 0.0,
        "azimuth_pixel_spacing": float(row_spacing) if row_spacing is not None else 0.0,
    }
    return ycfg


def convert_lt1(
    input_path: Path,
    output_dir: Path,
    output_name: Optional[str],
    link_mode: str,
    overwrite: bool,
    include_rpc: bool = False,
) -> Dict[str, Path]:
    product = _discover_product(input_path)
    _validate_input_product(product)
    out_dir = output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = output_name if output_name else product.base_name
    out_tiff = out_dir / f"{base}.tiff"
    out_vrt = out_dir / f"{base}.vrt"
    out_yaml = out_dir / f"{base}.yaml"
    out_meta = out_dir / f"{base}.meta.xml"
    out_rpc = out_dir / f"{base}.rpc"

    # RPC is optional and some LT1 RPC sidecars are incomplete.
    # Remove stale rpc first to avoid GDAL sidecar warning when opening TIFF.
    if (not include_rpc) and overwrite and (out_rpc.exists() or out_rpc.is_symlink()):
        out_rpc.unlink()

    _safe_link_or_copy(product.tiff, out_tiff, mode=link_mode, overwrite=overwrite)
    _safe_link_or_copy(product.meta_xml, out_meta, mode=link_mode, overwrite=overwrite)

    if product.incidence_xml is not None:
        _safe_link_or_copy(
            product.incidence_xml,
            out_dir / f"{base}.incidence.xml",
            mode=link_mode,
            overwrite=overwrite,
        )
    if overwrite and out_vrt.exists():
        out_vrt.unlink()
    _create_vrt(out_tiff, out_vrt)

    raster_info = _read_raster_info(out_tiff)
    ycfg = _parse_lt1_metadata(out_meta, raster_info=raster_info, output_data_file=out_tiff.name)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(ycfg, f, allow_unicode=False, sort_keys=False)

    if include_rpc and product.rpc is not None:
        _safe_link_or_copy(product.rpc, out_rpc, mode=link_mode, overwrite=overwrite)

    return {
        "source_meta": product.meta_xml,
        "source_tiff": product.tiff,
        "output_tiff": out_tiff,
        "output_vrt": out_vrt,
        "output_yaml": out_yaml,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import LuTan-1 L1A product and generate DInSAR-ready TIFF/VRT/YAML."
    )
    parser.add_argument("input", help="LT1 product directory, or *.meta.xml, or *.tiff")
    parser.add_argument("output_dir", help="Output directory for converted product")
    parser.add_argument(
        "--name",
        default=None,
        help="Output base name (default: input product base name)",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to place source files into output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--with-rpc",
        action="store_true",
        help="Also export *.rpc sidecar (default: disabled)",
    )
    args = parser.parse_args()

    outputs = convert_lt1(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        output_name=args.name,
        link_mode=args.link_mode,
        overwrite=bool(args.overwrite),
        include_rpc=bool(args.with_rpc),
    )

    print("LT1 import completed:")
    print(f"  source_meta : {outputs['source_meta']}")
    print(f"  source_tiff : {outputs['source_tiff']}")
    print(f"  output_tiff : {outputs['output_tiff']}")
    print(f"  output_vrt  : {outputs['output_vrt']}")
    print(f"  output_yaml : {outputs['output_yaml']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
