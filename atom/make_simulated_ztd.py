#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

ROOT = Path('/home/ysdong/Software/GMTSAR/atom')
OUT = ROOT / 'simulated_atm'
OUT.mkdir(parents=True, exist_ok=True)

phi_vrt = Path('/home/ysdong/Temp/yaxia/test/interferogram/filt_topophase.unw.geo.vrt')
coh_vrt = Path('/home/ysdong/Temp/yaxia/test/interferogram/phsig.cor.geo.vrt')
los_vrt = Path('/home/ysdong/Temp/yaxia/test/geometry/los.rdr.geo.vrt')

# Sentinel-1 C band wavelength from PRM
radar_wavelength = 0.0555171
coh_thr = 0.35
sigma_px = 60.0
phase_scale = 1.0
base_ztd = 2.30  # meters, typical zenith delay magnitude


def read(path, band=1):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'cannot open {path}')
    arr = ds.GetRasterBand(band).ReadAsArray().astype(np.float64)
    return ds, arr


def write_like(path, ref_ds, arr):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(
        str(path),
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW'],
    )
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    b = ds.GetRasterBand(1)
    b.WriteArray(arr.astype(np.float32))
    b.SetNoDataValue(np.nan)
    b.FlushCache()
    ds.FlushCache()
    ds = None


def weighted_lowpass(field, valid_mask, sigma):
    x = np.where(np.isfinite(field), field, 0.0)
    w = np.where(valid_mask, 1.0, 0.0)

    ny, nx = field.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    h = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fx * fx + fy * fy))

    num = np.real(np.fft.ifft2(np.fft.fft2(x) * h))
    den = np.real(np.fft.ifft2(np.fft.fft2(w) * h))
    low = np.where(den > 1e-3, num / den, np.nan)
    return low


phi_ds, phi_unw = read(phi_vrt, band=2)
_, coh = read(coh_vrt, band=1)
_, inc = read(los_vrt, band=1)

valid = np.isfinite(phi_unw) & np.isfinite(coh) & np.isfinite(inc) & (coh >= coh_thr)
if valid.sum() < 1000:
    raise RuntimeError('insufficient valid pixels for simulation')

phi_low = weighted_lowpass(phi_unw, valid, sigma_px)
phi_low = phi_low - np.nanmedian(phi_low[valid])
phi_sim = phase_scale * phi_low

# Convert desired atmospheric phase to ZTD difference.
cos_inc = np.cos(np.deg2rad(inc))
cos_inc = np.where(np.abs(cos_inc) < 0.1, np.nan, cos_inc)
delta_ztd = -(phi_sim * radar_wavelength * cos_inc) / (4.0 * np.pi)

# Build master/slave ZTD fields.
master_ztd = np.full_like(delta_ztd, base_ztd, dtype=np.float64)
slave_ztd = master_ztd + delta_ztd

master_ztd[~np.isfinite(delta_ztd)] = np.nan
slave_ztd[~np.isfinite(delta_ztd)] = np.nan

master_path = OUT / 'simulated_master_ztd.tif'
slave_path = OUT / 'simulated_slave_ztd.tif'
delta_path = OUT / 'simulated_delta_ztd.tif'
phase_path = OUT / 'simulated_atm_phase_for_report.tif'

write_like(master_path, phi_ds, master_ztd)
write_like(slave_path, phi_ds, slave_ztd)
write_like(delta_path, phi_ds, delta_ztd)
write_like(phase_path, phi_ds, phi_sim)

stats = {
    'coh_threshold': coh_thr,
    'sigma_px': sigma_px,
    'phase_scale': phase_scale,
    'radar_wavelength_m': radar_wavelength,
    'valid_pixels': int(valid.sum()),
    'phi_sim_rad_p05_p95': [float(np.nanpercentile(phi_sim[valid], 5)), float(np.nanpercentile(phi_sim[valid], 95))],
    'delta_ztd_m_min_max': [float(np.nanmin(delta_ztd[valid])), float(np.nanmax(delta_ztd[valid]))],
    'outputs': {
        'master': str(master_path),
        'slave': str(slave_path),
        'delta': str(delta_path),
        'phase': str(phase_path),
    },
    'note': 'Simulated product for presentation only; not physically retrieved atmospheric data.'
}
(OUT / 'simulated_atm_stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(stats, indent=2, ensure_ascii=False))
