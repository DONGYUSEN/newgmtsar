#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAXIA_DIR="${YAXIA_DIR:-$HOME/Temp/yaxia}"
GMTSAR_BIN="${GMTSAR_BIN:-$HOME/Software/GMTSAR/bin}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/work}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results}"

MASTER_PRM="${MASTER_PRM:-20231110.PRM}"
SLAVE_PRM="${SLAVE_PRM:-20231121.PRM}"
COH_THR="${COH_THR:-0.35}"
RUN_GMTSAR_INTF="${RUN_GMTSAR_INTF:-0}"

IFG_UNW_VRT="${IFG_UNW_VRT:-$YAXIA_DIR/test/interferogram/filt_topophase.unw.geo.vrt}"
COH_VRT="${COH_VRT:-$YAXIA_DIR/test/interferogram/phsig.cor.geo.vrt}"
LOS_VRT="${LOS_VRT:-$YAXIA_DIR/test/geometry/los.rdr.geo.vrt}"
DEM_FILE="${DEM_FILE:-$YAXIA_DIR/topo/dem.grd}"
IFG_UNW_BAND="${IFG_UNW_BAND:-2}"
COH_BAND="${COH_BAND:-1}"

ATM_MASTER="${ATM_MASTER:-}"
ATM_SLAVE="${ATM_SLAVE:-}"
ATM_SCALE="${ATM_SCALE:-1.0}"
RADAR_WAVELENGTH="${RADAR_WAVELENGTH:-}"

mkdir -p "$WORK_DIR" "$OUT_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing file: $1" >&2
    exit 1
  fi
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command: $1" >&2
    exit 1
  fi
}

need_cmd python3
need_file "$IFG_UNW_VRT"
need_file "$COH_VRT"
need_file "$DEM_FILE"
need_file "$YAXIA_DIR/SLC/$MASTER_PRM"
need_file "$YAXIA_DIR/SLC/$SLAVE_PRM"

if [[ -d "$GMTSAR_BIN" ]]; then
  export PATH="$GMTSAR_BIN:$PATH"
fi

if [[ "$RUN_GMTSAR_INTF" == "1" ]]; then
  need_cmd intf.csh
  log "RUN_GMTSAR_INTF=1, using GMTSAR intf.csh to (re)generate interferogram in $YAXIA_DIR/SLC"
  (
    cd "$YAXIA_DIR/SLC"
    intf.csh "$MASTER_PRM" "$SLAVE_PRM"
  )
else
  log "Skip GMTSAR intf generation (RUN_GMTSAR_INTF=$RUN_GMTSAR_INTF)."
  log "Using existing real interferogram products from: $YAXIA_DIR/test/interferogram"
fi

if [[ -n "$ATM_MASTER" || -n "$ATM_SLAVE" ]]; then
  need_file "$ATM_MASTER"
  need_file "$ATM_SLAVE"
  log "External atmospheric rasters detected: ATM_MASTER=$ATM_MASTER ATM_SLAVE=$ATM_SLAVE"
else
  log "No external atmospheric rasters provided; running DEM-based elevation stratification correction only."
fi

export IFG_UNW_VRT COH_VRT LOS_VRT DEM_FILE IFG_UNW_BAND COH_BAND OUT_DIR WORK_DIR COH_THR ATM_MASTER ATM_SLAVE ATM_SCALE
export RADAR_WAVELENGTH YAXIA_DIR MASTER_PRM

python3 - <<'PY'
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from osgeo import gdal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gdal.UseExceptions()


def env(name, default=None):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def read_array(path, band=1):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open {path}")
    arr = ds.GetRasterBand(band).ReadAsArray()
    return ds, arr


def parse_wavelength(prm_path):
    text = Path(prm_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"radar_wavelength\s*=\s*([0-9eE+\-.]+)", text)
    if not m:
        raise RuntimeError(f"radar_wavelength not found in {prm_path}")
    return float(m.group(1))


def warp_to_ref(src_path, ref_ds, out_path, resample="bilinear"):
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + xsize * gt[1]
    ymin = ymax + ysize * gt[5]

    options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(xmin, ymin, xmax, ymax),
        width=xsize,
        height=ysize,
        dstSRS=proj if proj else None,
        resampleAlg=resample,
        dstNodata=np.nan,
        multithread=True,
    )
    out = gdal.Warp(out_path, src_path, options=options)
    if out is None:
        raise RuntimeError(f"Failed to warp {src_path} to {out_path}")
    out = None


def robust_linear_fit(x, y, w, sigma_clip=3.0):
    if x.size < 1000:
        raise RuntimeError("Too few samples for robust fit")

    def fit_once(x0, y0, w0):
        A = np.vstack([x0, np.ones_like(x0)]).T
        sw = np.sqrt(w0)
        Aw = A * sw[:, None]
        yw = y0 * sw
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        return float(coef[0]), float(coef[1])

    a0, b0 = fit_once(x, y, w)
    res = y - (a0 * x + b0)
    med = np.median(res)
    mad = np.median(np.abs(res - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-8:
        return a0, b0

    keep = np.abs(res - med) <= sigma_clip * sigma
    if keep.sum() < 1000:
        return a0, b0

    return fit_once(x[keep], y[keep], w[keep])


def wrap_phase(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def write_geotiff(path, ref_ds, array):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW"],
    )
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    band = ds.GetRasterBand(1)
    band.WriteArray(array.astype(np.float32))
    band.SetNoDataValue(np.nan)
    band.FlushCache()
    ds.FlushCache()
    ds = None


def write_png(path, array, title):
    plt.figure(figsize=(10, 6), dpi=160)
    img = np.array(array, dtype=np.float64)
    valid = np.isfinite(img)
    if valid.any():
        v = np.nanpercentile(np.abs(img[valid]), 98)
        v = max(v, 1e-3)
    else:
        v = np.pi
    plt.imshow(img, cmap="twilight_shifted", vmin=-v, vmax=v)
    plt.colorbar(label="phase (rad)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


ifg_unw_vrt = env("IFG_UNW_VRT")
coh_vrt = env("COH_VRT")
los_vrt = env("LOS_VRT")
dem_file = env("DEM_FILE")
ifg_unw_band = int(env("IFG_UNW_BAND", "2"))
coh_band = int(env("COH_BAND", "1"))
out_dir = Path(env("OUT_DIR"))
work_dir = Path(env("WORK_DIR"))
coh_thr = float(env("COH_THR", "0.35"))
atm_master = env("ATM_MASTER", "")
atm_slave = env("ATM_SLAVE", "")
atm_scale = float(env("ATM_SCALE", "1.0"))

prm_path = Path(env("YAXIA_DIR")) / "SLC" / env("MASTER_PRM")
if env("RADAR_WAVELENGTH") is None:
    radar_wavelength = parse_wavelength(str(prm_path))
else:
    radar_wavelength = float(env("RADAR_WAVELENGTH"))

ref_ds, phi_unw = read_array(ifg_unw_vrt, ifg_unw_band)
_, coh = read_array(coh_vrt, coh_band)

# DEM -> IFG grid
dem_match = work_dir / "dem_to_ifg.tif"
warp_to_ref(dem_file, ref_ds, str(dem_match), resample="bilinear")
_, dem = read_array(str(dem_match), 1)

phi_unw = phi_unw.astype(np.float64)
coh = coh.astype(np.float64)
dem = dem.astype(np.float64)

valid = np.isfinite(phi_unw) & np.isfinite(coh) & np.isfinite(dem) & (coh >= coh_thr)
if valid.sum() < 1000:
    raise RuntimeError("Insufficient valid pixels after masking; check coherence threshold or inputs")

x = dem[valid]
y = phi_unw[valid]
w = np.clip(coh[valid], 1e-3, 1.0) ** 2

a, b = robust_linear_fit(x, y, w)
phi_elev = a * dem + b

phi_meteo = np.zeros_like(phi_unw, dtype=np.float64)
meteo_used = False

if atm_master and atm_slave:
    m_warp = work_dir / "atm_master_to_ifg.tif"
    s_warp = work_dir / "atm_slave_to_ifg.tif"
    warp_to_ref(atm_master, ref_ds, str(m_warp), resample="bilinear")
    warp_to_ref(atm_slave, ref_ds, str(s_warp), resample="bilinear")
    _, ztd_m = read_array(str(m_warp), 1)
    _, ztd_s = read_array(str(s_warp), 1)

    delta_ztd = (ztd_s.astype(np.float64) - ztd_m.astype(np.float64)) * atm_scale

    if los_vrt and Path(los_vrt).exists():
        _, inc = read_array(los_vrt, 1)
        inc = inc.astype(np.float64)
        cos_inc = np.cos(np.deg2rad(inc))
        cos_inc = np.where(np.abs(cos_inc) < 0.1, np.nan, cos_inc)
    else:
        cos_inc = 1.0

    delta_los = delta_ztd / cos_inc
    phi_meteo = -4.0 * np.pi / radar_wavelength * delta_los
    meteo_used = True

phi_corr = phi_unw - phi_elev - phi_meteo

before_wrap = wrap_phase(phi_unw)
after_wrap = wrap_phase(phi_corr)

# Write products
before_tif = out_dir / "before_wrapped_phase.tif"
after_tif = out_dir / "after_wrapped_phase.tif"
elev_tif = out_dir / "atm_elevation_phase.tif"
meteo_tif = out_dir / "atm_meteo_phase.tif"

write_geotiff(str(before_tif), ref_ds, before_wrap)
write_geotiff(str(after_tif), ref_ds, after_wrap)
write_geotiff(str(elev_tif), ref_ds, phi_elev)
write_geotiff(str(meteo_tif), ref_ds, phi_meteo)

write_png(str(out_dir / "before_wrapped_phase.png"), before_wrap, "Before Atmospheric Correction (wrapped phase)")
write_png(str(out_dir / "after_wrapped_phase.png"), after_wrap, "After Atmospheric Correction (wrapped phase)")
write_png(str(out_dir / "atm_elevation_phase.png"), phi_elev, "Elevation-Stratified Atmospheric Phase")

# Metrics on valid pixels
valid2 = valid & np.isfinite(phi_corr)
if valid2.sum() > 1000:
    std_before = float(np.nanstd(phi_unw[valid2]))
    std_after = float(np.nanstd(phi_corr[valid2]))
    corr_before = float(np.corrcoef(phi_unw[valid2], dem[valid2])[0, 1])
    corr_after = float(np.corrcoef(phi_corr[valid2], dem[valid2])[0, 1])
else:
    std_before = std_after = corr_before = corr_after = float("nan")

metrics = {
    "coh_threshold": coh_thr,
    "radar_wavelength_m": radar_wavelength,
    "fit": {"a_rad_per_m": a, "b_rad": b},
    "meteo_used": meteo_used,
    "valid_pixels": int(valid2.sum()),
    "std_before_rad": std_before,
    "std_after_rad": std_after,
    "corr_phase_dem_before": corr_before,
    "corr_phase_dem_after": corr_after,
    "outputs": {
        "before_tif": str(before_tif),
        "after_tif": str(after_tif),
        "before_png": str(out_dir / "before_wrapped_phase.png"),
        "after_png": str(out_dir / "after_wrapped_phase.png"),
        "elev_tif": str(elev_tif),
        "meteo_tif": str(meteo_tif),
    },
}

(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
PY

log "Done. Results are in: $OUT_DIR"
log "Key outputs: before_wrapped_phase.png and after_wrapped_phase.png"
