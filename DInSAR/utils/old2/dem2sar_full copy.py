import numpy as np
import yaml
import rasterio
from scipy.interpolate import CubicSpline
from datetime import datetime

C = 299792458.0


############################################################
# 向量维度统一
############################################################

def ensure_vec3(x):

    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(1, 3)

    if x.shape[-1] != 3:
        raise ValueError("Vector must be (...,3)")

    return x


############################################################
# 时间
############################################################

def parse_time(t):
    return datetime.fromisoformat(t).timestamp()


############################################################
# 坐标转换
############################################################

def llh_to_xyz(lat, lon, h):

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    h = np.asarray(h)

    a = 6378137.0
    e2 = 0.00669437999014

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    N = a / np.sqrt(1 - e2*np.sin(lat)**2)

    x = (N+h)*np.cos(lat)*np.cos(lon)
    y = (N+h)*np.cos(lat)*np.sin(lon)
    z = (N*(1-e2)+h)*np.sin(lat)

    P = np.stack([x, y, z], axis=-1)

    return ensure_vec3(P)


############################################################
# 轨道模型
############################################################

class Orbit:

    def __init__(self, yaml_data):

        orbit = yaml_data["orbit_data"]["orbit_points"]

        t = []
        pos = []
        vel = []

        for p in orbit:

            t.append(parse_time(p["time"]))

            pos.append([
                p["position"]["x"],
                p["position"]["y"],
                p["position"]["z"]
            ])

            vel.append([
                p["velocity"]["vx"],
                p["velocity"]["vy"],
                p["velocity"]["vz"]
            ])

        t = np.array(t)
        pos = ensure_vec3(pos)
        vel = ensure_vec3(vel)

        self.pos = CubicSpline(t, pos)
        self.vel = CubicSpline(t, vel)

    def position(self, t):

        S = self.pos(t)
        return ensure_vec3(S)

    def velocity(self, t):

        V = self.vel(t)
        return ensure_vec3(V)


############################################################
# Geo2Rdr
############################################################

class Geo2Rdr:

    def __init__(self, yaml_data):

        self.orbit = Orbit(yaml_data)

        radar = yaml_data["radar_parameters"]
        meta = yaml_data["metadata"]

        self.prf = radar["prf"]
        self.near_range = radar["near_range"]
        self.range_spacing = radar["range_spacing"]
        self.wavelength = radar["wavelength"]

        self.t0 = parse_time(meta["first_line_sensing_time"])
        self.t1 = parse_time(meta["last_line_sensing_time"])

        img = yaml_data["image_parameters"]

        self.nr = img["ncols"]
        self.na = img["nrows"]


############################################################
# Zero Doppler 求解
############################################################

    def solve_time(self, P):

        P = ensure_vec3(P)

        N = P.shape[0]

        tgrid = np.linspace(self.t0, self.t1, 2000)

        S = self.orbit.position(tgrid)
        V = self.orbit.velocity(tgrid)

        dr = P[:, None, :] - S[None, :, :]

        f = np.sum(dr * V[None, :, :], axis=2)

        sign = np.sign(f)

        idx = np.argmax(np.diff(sign, axis=1) != 0, axis=1)

        t1 = tgrid[idx]
        t2 = tgrid[idx+1]

        f1 = f[np.arange(N), idx]
        f2 = f[np.arange(N), idx+1]

        t = t1 - f1*(t2-t1)/(f2-f1)

        return t


############################################################
# DEM → SAR
############################################################

    def geo2rdr(self, P):

        P = ensure_vec3(P)

        t = self.solve_time(P)

        S = self.orbit.position(t)

        dr = P - S

        R = np.linalg.norm(dr, axis=1)

        range_pixel = (R-self.near_range)/self.range_spacing
        az_pixel = (t-self.t0)*self.prf

        return range_pixel, az_pixel, R, t


############################################################
# DEM读取
############################################################

def load_dem(fname):

    with rasterio.open(fname) as ds:

        dem = ds.read(1)

        transform = ds.transform

        rows, cols = np.indices(dem.shape)

        lon, lat = rasterio.transform.xy(transform, rows, cols)

        lat = np.array(lat)
        lon = np.array(lon)

    return lat, lon, dem


############################################################
# DEM → SAR DEM
############################################################

def dem_to_sar_dem(geo, lat, lon, h):

    P = llh_to_xyz(lat.flatten(), lon.flatten(), h.flatten())

    r, a, R, t = geo.geo2rdr(P)

    sar_dem = np.full((geo.na, geo.nr), np.nan)

    r = np.round(r).astype(int)
    a = np.round(a).astype(int)

    mask = (r >= 0) & (r < geo.nr) & (a >= 0) & (a < geo.na)

    sar_dem[a[mask], r[mask]] = h.flatten()[mask]

    return sar_dem, r, a, R


############################################################
# DEM → SLC 模拟
############################################################

def simulate_slc(geo, r, a, R):

    slc = np.zeros((geo.na, geo.nr), dtype=np.complex64)

    phase = 4*np.pi*R/geo.wavelength

    signal = np.exp(1j*phase)

    r = np.round(r).astype(int)
    a = np.round(a).astype(int)

    mask = (r >= 0) & (r < geo.nr) & (a >= 0) & (a < geo.na)

    slc[a[mask], r[mask]] = signal[mask]

    return slc


############################################################
# Phase correlation
############################################################

def estimate_shift(img1, img2):

    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)

    cross = f1*np.conj(f2)

    cross /= np.abs(cross)+1e-12

    corr = np.fft.ifft2(cross)

    corr = np.abs(corr)

    y, x = np.unravel_index(np.argmax(corr), corr.shape)

    if y > img1.shape[0]//2:
        y -= img1.shape[0]

    if x > img1.shape[1]//2:
        x -= img1.shape[1]

    return x, y


############################################################
# 几何更新
############################################################

def update_geometry(geo, range_shift, az_shift):

    geo.near_range += range_shift*geo.range_spacing

    geo.t0 += az_shift/geo.prf

    print("Geometry updated")


############################################################
# SAR仿真流程
############################################################

def simulate_pipeline(geo, lat, lon, h):

    sar_dem, r, a, R = dem_to_sar_dem(geo, lat, lon, h)

    slc = simulate_slc(geo, r, a, R)

    amp = np.abs(slc)

    return sar_dem, slc, amp


############################################################
# 主流程
############################################################

def run_pipeline(yaml_file, dem_file, real_sar_file):

    yaml_data = yaml.safe_load(open(yaml_file))

    geo = Geo2Rdr(yaml_data)

    lat, lon, h = load_dem(dem_file)

    with rasterio.open(real_sar_file) as ds:
        real_sar = ds.read(1)

    print("第一次模拟")

    sar_dem, slc, sim_amp = simulate_pipeline(geo, lat, lon, h)

    print("估计误差")

    shift_x, shift_y = estimate_shift(sim_amp, real_sar)

    print("range shift:", shift_x)
    print("azimuth shift:", shift_y)

    update_geometry(geo, shift_x, shift_y)

    print("重新模拟")

    sar_dem, slc, sim_amp = simulate_pipeline(geo, lat, lon, h)

    return sar_dem, slc, sim_amp


############################################################

if __name__ == "__main__":

    import sys

    sar_dem, slc, sim_amp = run_pipeline(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3]
    )