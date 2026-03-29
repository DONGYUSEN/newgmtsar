
#!/usr/bin/env python3
"""
dem2sar_full.py - DEM到SAR转换与几何误差校正工具

数据处理流程：
DEM
 ↓
geo2rdr
 ↓
模拟SAR
 ↓
与真实SAR匹配
 ↓
估计几何误差
 ↓
更新几何模型
 ↓
重新 geo2rdr
 ↓
最终 SAR 模拟
"""

import numpy as np
import yaml
import rasterio
import rasterio.windows
import rasterio.warp
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from datetime import datetime
from multiprocessing import Pool
import argparse
import os
import time
from numba import jit, prange
from multiprocessing import shared_memory
import atexit

C = 299792458.0


def ensure_1d_array(arr, name="array"):
    """确保数组是一维，否则抛出明确的维度错误"""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D array, got {arr.ndim}D (shape: {arr.shape})")
    return arr


def ensure_vec3(x):
    """确保向量是 (..., 3) 形状"""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, 3)
    if x.shape[-1] != 3:
        raise ValueError(f"vector must be (...,3), got {x.shape}")
    return x


############################################################
# 时间
############################################################

def parse_time(t):
    return datetime.fromisoformat(t).timestamp()


############################################################
# 坐标转换
############################################################

@jit(nopython=True, fastmath=True, parallel=True)
def llh_to_xyz(lat, lon, h):

    a = 6378137.0
    e2 = 0.00669437999014

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    N = a / np.sqrt(1 - e2*np.sin(lat)**2)

    x = (N+h)*np.cos(lat)*np.cos(lon)
    y = (N+h)*np.cos(lat)*np.sin(lon)
    z = (N*(1-e2)+h)*np.sin(lat)

    # 只处理数组情况，因为在我们的使用场景中，lat、lon、h都是数组
    result = np.empty((x.shape[0], 3), dtype=np.float64)
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = z
    return result


def xyz_to_llh(x,y,z):
    # 强制转为标量（避免数组输入）
    if isinstance(x, np.ndarray):
        x = x.item() if x.size == 1 else x
        y = y.item() if y.size == 1 else y
        z = z.item() if z.size == 1 else z
    
    a = 6378137.0
    e2 = 0.00669437999014

    lon = np.arctan2(y,x)
    p = np.sqrt(x*x + y*y)

    lat = np.arctan2(z,p*(1-e2))

    for _ in range(6):
        N = a/np.sqrt(1-e2*np.sin(lat)**2)
        h = p/np.cos(lat)-N
        lat = np.arctan2(z,p*(1-e2*N/(N+h)))

    N = a/np.sqrt(1-e2*np.sin(lat)**2)
    h = p/np.cos(lat)-N

    return np.rad2deg(lat),np.rad2deg(lon),h


@jit(nopython=True, fastmath=True, parallel=True)
def xyz_to_llh_batch(xyz):
    """批量 ECEF(xyz) -> (lat, lon, h)，输入 xyz 为 (N,3)"""
    a = 6378137.0
    e2 = 0.00669437999014

    Np = xyz.shape[0]
    lat_out = np.empty(Np, dtype=np.float64)
    lon_out = np.empty(Np, dtype=np.float64)
    h_out = np.empty(Np, dtype=np.float64)

    for i in prange(Np):
        x = xyz[i, 0]
        y = xyz[i, 1]
        z = xyz[i, 2]

        lon = np.arctan2(y, x)
        p = np.sqrt(x * x + y * y)

        lat = np.arctan2(z, p * (1.0 - e2))

        for _ in range(6):
            sin_lat = np.sin(lat)
            N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
            h = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1.0 - e2 * N / (N + h)))

        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        h = p / np.cos(lat) - N

        lat_out[i] = lat * 180.0 / np.pi
        lon_out[i] = lon * 180.0 / np.pi
        h_out[i] = h

    return lat_out, lon_out, h_out


############################################################
# 轨道
############################################################

class Orbit:

    def __init__(self,yaml_data):

        orbit = yaml_data["orbit_data"]["orbit_points"]

        t=[]
        pos=[]
        vel=[]

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

        t=np.array(t)
        pos=np.array(pos, dtype=np.float32)
        vel=np.array(vel, dtype=np.float32)

        self.tmin=t.min()
        self.tmax=t.max()

        self.pos=CubicSpline(t,pos)
        self.vel=CubicSpline(t,vel)
        # 加速度（用于更稳定的零多普勒牛顿迭代）
        self.acc = self.vel.derivative(1)
        
        # 动态调整预计算点数
        time_range = self.tmax - self.tmin
        self.precompute_steps = max(1000, min(10000, int(time_range * 100)))
        self.precompute_times = np.linspace(self.tmin, self.tmax, self.precompute_steps)
        # 使用 float64，避免后续反复 upcast/downcast
        self.precompute_positions = np.asarray(self.pos(self.precompute_times), dtype=np.float64)
        self.precompute_velocities = np.asarray(self.vel(self.precompute_times), dtype=np.float64)
        self.precompute_accelerations = np.asarray(self.acc(self.precompute_times), dtype=np.float64)

    def _interp_precompute(self, t, values):
        """对预计算轨道数据做批量线性插值；t 为标量或 (N,)"""
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 0:
            idx = int(np.searchsorted(self.precompute_times, float(t)))
            if idx <= 0:
                out = values[0]
            elif idx >= self.precompute_steps:
                out = values[-1]
            else:
                t1 = self.precompute_times[idx - 1]
                t2 = self.precompute_times[idx]
                v1 = values[idx - 1]
                v2 = values[idx]
                frac = (float(t) - t1) / (t2 - t1)
                out = v1 + (v2 - v1) * frac
            return np.asarray(out, dtype=np.float64).reshape(1, 3)

        idx = np.searchsorted(self.precompute_times, t)
        idx = np.clip(idx, 1, self.precompute_steps - 1)
        t1 = self.precompute_times[idx - 1]
        t2 = self.precompute_times[idx]
        v1 = values[idx - 1]
        v2 = values[idx]
        frac = ((t - t1) / (t2 - t1)).reshape(-1, 1)
        out = v1 + (v2 - v1) * frac
        return np.asarray(out, dtype=np.float64).reshape(-1, 3)

    def position(self,t):
        return self._interp_precompute(t, self.precompute_positions)

    def velocity(self,t):
        return self._interp_precompute(t, self.precompute_velocities)

    def acceleration(self, t):
        return self._interp_precompute(t, self.precompute_accelerations)


############################################################
# Geo2Rdr
############################################################

class Geo2Rdr:

    def __init__(self,yaml_data):

        self.orbit=Orbit(yaml_data)

        radar=yaml_data["radar_parameters"]
        meta=yaml_data["metadata"]

        self.prf=radar["prf"]
        self.near_range=radar["near_range"]
        self.range_spacing=radar["range_spacing"]
        self.wavelength=radar["wavelength"]

        self.t0=parse_time(meta["first_line_sensing_time"])
        self.t1=parse_time(meta["last_line_sensing_time"])

        self.nr=yaml_data["image_parameters"]["ncols"]
        self.na=yaml_data["image_parameters"]["nrows"]

        # 用于 rdr2geo 的初值：优先使用 geolocation_grid，其次 corner_coordinates
        self._corner_latlon = None  # (lat_tl, lon_tl, lat_tr, lon_tr, lat_bl, lon_bl, lat_br, lon_br)
        self._init_latlon_from_yaml(yaml_data)

    def _init_latlon_from_yaml(self, yaml_data):
        """初始化像素(行/列)->经纬度的双线性角点参数（仅用于初始猜测）"""
        lat_tl = lon_tl = lat_tr = lon_tr = lat_bl = lon_bl = lat_br = lon_br = None

        glg = yaml_data.get("geolocation_grid")
        if isinstance(glg, list) and len(glg) >= 4:
            # 从 geolocation_grid 找到四个角点：按 (line, pixel) 的 min/max 组合挑选
            lines = np.array([p.get("line", 0) for p in glg], dtype=np.int64)
            pixels = np.array([p.get("pixel", 0) for p in glg], dtype=np.int64)
            latv = np.array([p.get("latitude", np.nan) for p in glg], dtype=np.float64)
            lonv = np.array([p.get("longitude", np.nan) for p in glg], dtype=np.float64)

            def pick(target_line, target_pixel):
                m = (lines == target_line) & (pixels == target_pixel) & np.isfinite(latv) & np.isfinite(lonv)
                if np.any(m):
                    idx = int(np.where(m)[0][0])
                    return float(latv[idx]), float(lonv[idx])
                return None, None

            l0 = int(lines.min())
            l1 = int(lines.max())
            p0 = int(pixels.min())
            p1 = int(pixels.max())

            lat_tl, lon_tl = pick(l0, p0)
            lat_tr, lon_tr = pick(l0, p1)
            lat_bl, lon_bl = pick(l1, p0)
            lat_br, lon_br = pick(l1, p1)

        if lat_tl is None:
            cc = yaml_data.get("corner_coordinates") or {}
            # corner_coordinates 里 x/y 语义不一定是 line/pixel，这里仅取 lat/lon
            tl = cc.get("top_left") or {}
            tr = cc.get("top_right") or {}
            bl = cc.get("bottom_left") or {}
            br = cc.get("bottom_right") or {}
            try:
                lat_tl, lon_tl = float(tl["lat"]), float(tl["lon"])
                lat_tr, lon_tr = float(tr["lat"]), float(tr["lon"])
                lat_bl, lon_bl = float(bl["lat"]), float(bl["lon"])
                lat_br, lon_br = float(br["lat"]), float(br["lon"])
            except Exception:
                lat_tl = lon_tl = lat_tr = lon_tr = lat_bl = lon_bl = lat_br = lon_br = None

        if lat_tl is not None:
            self._corner_latlon = (lat_tl, lon_tl, lat_tr, lon_tr, lat_bl, lon_bl, lat_br, lon_br)

    def approx_latlon(self, r, a):
        """
        使用四角点双线性插值给 (range_pixel=r, az_pixel=a) 提供 (lat, lon) 初始值。
        r/a 支持标量或数组。
        """
        if self._corner_latlon is None:
            raise ValueError("YAML 中缺少 geolocation_grid 或 corner_coordinates，无法生成初始 lat/lon")

        lat_tl, lon_tl, lat_tr, lon_tr, lat_bl, lon_bl, lat_br, lon_br = self._corner_latlon

        r = np.asarray(r, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        u = r / max(1.0, float(self.nr - 1))
        v = a / max(1.0, float(self.na - 1))
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        lat_top = lat_tl * (1.0 - u) + lat_tr * u
        lat_bot = lat_bl * (1.0 - u) + lat_br * u
        lat0 = lat_top * (1.0 - v) + lat_bot * v

        lon_top = lon_tl * (1.0 - u) + lon_tr * u
        lon_bot = lon_bl * (1.0 - u) + lon_br * u
        lon0 = lon_top * (1.0 - v) + lon_bot * v

        return lat0, lon0


    ########################################################
    # Zero Doppler 求解
    ########################################################

    def solve_time(self, P, t_init=None):
        """Zero Doppler求解 - 牛顿迭代法"""
        N=P.shape[0]
        
        # 初始 azimuth 时间
        if t_init is None:
            t0 = (self.orbit.tmin + self.orbit.tmax) / 2
            t = np.full(N, t0, dtype=np.float64)
        else:
            t_init = np.asarray(t_init, dtype=np.float64)
            if t_init.ndim == 0:
                t = np.full(N, float(t_init), dtype=np.float64)
            else:
                if t_init.size != N:
                    raise ValueError(f"t_init size mismatch: {t_init.size} vs N={N}")
                t = t_init.copy()
            t = np.clip(t, self.orbit.tmin, self.orbit.tmax)

        # 牛顿迭代求解 Zero-Doppler 方程
        max_iter=15
        tol=1e-6
        
        for k in range(max_iter):
            # 计算卫星位置和速度
            S = ensure_vec3(self.orbit.position(t))
            V = ensure_vec3(self.orbit.velocity(t))
            A = ensure_vec3(self.orbit.acceleration(t))
            
            # 计算 dr = P - S(t)
            dr = P - S
            
            # 计算 f(t) = (P - S(t))·V(t)
            f = np.sum(dr * V, axis=1)
            
            # 计算 f'(t) = -||V||² + (P - S(t))·A(t)
            fprime = -np.sum(V**2, axis=1) + np.sum(dr * A, axis=1)
            # 防止除零
            fprime = np.where(np.abs(fprime) < 1e-12, np.sign(fprime) * 1e-12 + 1e-12, fprime)
            
            # 牛顿迭代更新
            t_new = t - f / fprime
            
            # 检查收敛
            if np.max(np.abs(t_new - t)) < tol:
                t = t_new
                break
            
            t = t_new

        # 限制时间在轨道范围内
        t = np.clip(t, self.orbit.tmin, self.orbit.tmax)

        return t


    ########################################################
    # DEM → SAR
    ########################################################

    def geo2rdr(self, P, dem=None, dem_lat=None, dem_lon=None, t_init=None):
        """
        DEM → SAR 坐标转换
        输入:
            P: 地面点坐标 (N,3)
            dem: DEM 高程 ndarray (可选)
            dem_lat, dem_lon: DEM 经纬度 ndarray (可选)
        输出:
            range_pixel, az_pixel, R, t, look_dir
        """
        # 1. 初始 azimuth 时间: 使用轨道中心时间作为初始猜测
        t = self.solve_time(P, t_init=t_init)

        # 2. 计算卫星位置和速度
        S = ensure_vec3(self.orbit.position(t))
        V = ensure_vec3(self.orbit.velocity(t))

        # 3. 计算 dr = P - S(t)
        dr = P - S

        # 4. 计算精确斜距 R = ||P - S(t)||
        R = np.linalg.norm(dr, axis=1)

        # 5. DEM 高程修正
        if dem is not None and dem_lat is not None and dem_lon is not None:
            from scipy.interpolate import RegularGridInterpolator
            # 将 P 转换为经纬度，用于 DEM 插值
            # 这里需要批量转换，暂时简化处理
            # 实际应用中应该使用批量坐标转换
            pass

        # 6. 更新 look-direction 左/右
        # 计算轨道法线
        orbit_normal = np.cross(S, V)
        orbit_normal = orbit_normal / np.linalg.norm(orbit_normal, axis=1, keepdims=True)
        
        # 计算视线方向与轨道法线的点积，确定左右视
        look_dir = np.sign(np.sum(dr * orbit_normal, axis=1))

        # 7. 计算像素坐标
        range_pixel = (R - self.near_range) / self.range_spacing
        az_pixel = (t - self.t0) * self.prf

        # 返回: 像素坐标, 斜距, 方位时间, 视线方向
        return range_pixel, az_pixel, R, t, look_dir


############################################################
# Rdr2Geo
############################################################

class Rdr2Geo:
    """
    科研级 Rdr2Geo (SAR → DEM)
    功能：
        - Zero-Doppler 精确迭代
        - DEM 高程修正
        - Shadow / Layover 检测
        - 批量向量化处理
    """
    def __init__(self, geo, dem=None, dem_lat=None, dem_lon=None):
        """
        geo: Geo2Rdr 实例
        dem: DEM 高程 ndarray
        dem_lat, dem_lon: DEM 经纬度 ndarray (同 dem 维度)
        """
        self.geo = geo
        if dem is not None:
            self.dem = dem
            self.dem_lat = dem_lat
            self.dem_lon = dem_lon
            # 为 DEM 准备插值器
            self._prepare_dem_interpolators()
        else:
            self.dem = None
            self.dem_interp = None
            self.normal_interp = None

    def _prepare_dem_interpolators(self):
        """准备 DEM 高程和法向量插值器"""
        if self.dem is None:
            return
        
        # 提取 DEM 坐标
        dem_height, dem_width = self.dem.shape
        
        # 处理 dem_lat 和 dem_lon，确保它们是二维数组
        if self.dem_lat.ndim == 1:
            # 如果是一维数组，重新构造为二维数组
            self.dem_lat = self.dem_lat.reshape(dem_height, dem_width)
        if self.dem_lon.ndim == 1:
            self.dem_lon = self.dem_lon.reshape(dem_height, dem_width)
        
        # 提取坐标值
        try:
            # 尝试从第一列和第一行提取坐标
            lat_coords = np.unique(self.dem_lat[:, 0])
            lon_coords = np.unique(self.dem_lon[0, :])
        except IndexError:
            # 如果提取失败，使用整个数组的唯一值
            lat_coords = np.unique(self.dem_lat)
            lon_coords = np.unique(self.dem_lon)
        
        # 确保坐标是单调的
        if len(lat_coords) > 1:
            if lat_coords[0] > lat_coords[-1]:
                lat_coords = lat_coords[::-1]
                # 如果纬度是递减的，反转 DEM 数据
                if self.dem.ndim == 2:
                    self.dem = self.dem[::-1, :]
                    if self.dem_lat.ndim == 2:
                        self.dem_lat = self.dem_lat[::-1, :]
        
        if len(lon_coords) > 1:
            if lon_coords[0] > lon_coords[-1]:
                lon_coords = lon_coords[::-1]
                # 如果经度是递减的，反转 DEM 数据
                if self.dem.ndim == 2:
                    self.dem = self.dem[:, ::-1]
                    if self.dem_lon.ndim == 2:
                        self.dem_lon = self.dem_lon[:, ::-1]
        
        # 创建高程插值器
        try:
            self.dem_interp = RegularGridInterpolator(
                (lat_coords, lon_coords),
                self.dem,
                bounds_error=False,
                fill_value=0
            )
            
            # 计算法向量
            dzdx = np.gradient(self.dem, axis=1)
            dzdy = np.gradient(self.dem, axis=0)
            nx = -dzdx
            ny = -dzdy
            nz = np.ones_like(self.dem)
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            nx /= norm
            ny /= norm
            nz /= norm
            
            # 创建法向量插值器
            normals = np.stack([nx, ny, nz], axis=2)
            self.normal_interp = RegularGridInterpolator(
                (lat_coords, lon_coords),
                normals,
                bounds_error=False,
                fill_value=(0, 0, 1)
            )
        except Exception as e:
            print(f"准备 DEM 插值器失败: {e}")
            self.dem_interp = None
            self.normal_interp = None

    def _get_dem_height(self, lat, lon):
        """从 DEM 获取高程"""
        if self.dem_interp is None:
            return 0
        try:
            return self.dem_interp(np.column_stack([lat, lon]))
        except:
            return 0

    def _get_dem_normal(self, lat, lon):
        """从 DEM 获取法向量"""
        if self.normal_interp is None:
            return np.array([0, 0, 1])
        try:
            return self.normal_interp(np.column_stack([lat, lon]))
        except:
            return np.array([0, 0, 1])

    def zero_doppler_iter(self, r, a, max_iter=15, tol=1e-6):
        """
        SAR(r,a) -> 地理点 P 的迭代求解（固定方位时间 t）
        输入:
            r, a: range / azimuth pixels
        输出:
            P: 地面点 (N,3)
            shadow_mask, layover_mask: bool (N,)
        """
        return self.solve(r, a, max_iter=max_iter, tol=tol)

    def _solve_zero_doppler_time(self, P):
        """求解 Zero-Doppler 时间"""
        N = P.shape[0]
        t = np.full(N, (self.geo.orbit.tmin + self.geo.orbit.tmax) / 2, dtype=np.float64)
        max_iter = 10
        tol = 1e-9
        
        for k in range(max_iter):
            S = ensure_vec3(self.geo.orbit.position(t))
            V = ensure_vec3(self.geo.orbit.velocity(t))
            
            dr = P - S
            f = np.sum(dr * V, axis=1)
            fprime = -np.sum(V**2, axis=1) + 1e-12
            
            t_new = t - f / fprime
            t_new = np.clip(t_new, self.geo.orbit.tmin, self.geo.orbit.tmax)
            
            if np.max(np.abs(t_new - t)) < tol:
                return t_new
            
            t = t_new
        
        return t

    def solve_with_geo2rdr_output(self, R, t):
        """
        使用 Geo2Rdr 输出的 slant_range 和 azimuth_time 进行反算
        输入:
            R: 斜距 (N,)
            t: 方位时间 (N,)
        输出:
            P: 地面点 (N,3)
            shadow_mask, layover_mask: bool (N,)
        """
        R = np.atleast_1d(R).astype(np.float64)
        t = np.atleast_1d(t).astype(np.float64)
        r = (R - self.geo.near_range) / self.geo.range_spacing
        a = (t - self.geo.t0) * self.geo.prf
        return self.solve(r, a, max_iter=15, tol=1e-4, R_override=R, t_override=t)

    def compute_shadow_layover(self, P, S):
        """
        Shadow / Layover 检测
        方法：
            - Shadow: DEM 高度阻挡视线
            - Layover: 卫星方向与法向量夹角 < 0
        """
        N = P.shape[0]
        shadow_mask = np.zeros(N, dtype=bool)
        layover_mask = np.zeros(N, dtype=bool)

        if self.dem is None:
            return shadow_mask, layover_mask

        # 计算视线方向
        dr = P - S
        dr_unit = dr / np.linalg.norm(dr, axis=1)[:, None]

        # 计算地面点的经纬度（批量）
        lat, lon, _ = xyz_to_llh_batch(np.asarray(P, dtype=np.float64))

        # Shadow: 真实遮挡需要射线检测，这里不做（避免给出误导性的 mask）
        shadow_mask[:] = False

        # Layover 检测：法向量与视线方向点积 < 0
        normals_P = self._get_dem_normal(lat, lon)
        if normals_P.ndim == 1:
            normals_P = np.repeat(normals_P.reshape(1, 3), N, axis=0)
        dotp = np.sum(normals_P * dr_unit, axis=1)
        layover_mask[:] = dotp < 0.0

        return shadow_mask, layover_mask

    def solve(self, r, a, max_iter=30, tol=1e-3, R_override=None, t_override=None):
        """
        主接口：固定方位时间 t(a) 求解地面点
        输入:
            r, a: range/azimuth pixels（标量或数组）
        输出:
            P: (N,3) 或 (3,)
            shadow_mask, layover_mask: (N,) 或标量
        """
        r_arr = np.atleast_1d(r).astype(np.float64)
        a_arr = np.atleast_1d(a).astype(np.float64)
        N = r_arr.size

        if R_override is None:
            R = self.geo.near_range + r_arr * self.geo.range_spacing
        else:
            R = np.atleast_1d(R_override).astype(np.float64)
        if t_override is None:
            t = self.geo.t0 + a_arr / self.geo.prf
        else:
            t = np.atleast_1d(t_override).astype(np.float64)

        # 卫星状态（固定 t）
        S = ensure_vec3(self.geo.orbit.position(t))
        V = ensure_vec3(self.geo.orbit.velocity(t))

        # 初值策略：
        # 1) 物理一致初值：把“指向地心”的方向投影到 V 的法平面上（满足 u·V=0），再用 R 得到 P0
        # 2) 若 YAML 角点可用，则用角点双线性 lat/lon 作为初值（通常更接近真实几何）
        Vhat = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        u0 = -S
        u0 = u0 - np.sum(u0 * Vhat, axis=1, keepdims=True) * Vhat
        u0 = u0 / (np.linalg.norm(u0, axis=1, keepdims=True) + 1e-12)
        P0 = S + R.reshape(-1, 1) * u0
        lat, lon, _ = xyz_to_llh_batch(np.asarray(P0, dtype=np.float64))

        try:
            lat_yaml, lon_yaml = self.geo.approx_latlon(r_arr, a_arr)
            lat = np.asarray(lat_yaml, dtype=np.float64).reshape(-1)
            lon = np.asarray(lon_yaml, dtype=np.float64).reshape(-1)
        except Exception:
            pass

        # 牛顿迭代（变量：lat, lon；高度由 DEM 或 0 给定）
        dlat = 5e-6
        dlon = 5e-6
        max_step = 5e-2  # deg，上限约 5.5km；配合线搜索避免发散
        # 缩放：f1 单位 m^2/s，典型量级约 R*|V|；f2 单位 m
        scale_f1 = (np.linalg.norm(V, axis=1) * (R + 1.0)).astype(np.float64) + 1e-6
        scale_f2 = 1.0

        def objective(f1v, f2v):
            return (f1v / scale_f1) ** 2 + (f2v / scale_f2) ** 2

        for _ in range(max_iter):
            if self.dem is not None:
                h = np.asarray(self._get_dem_height(lat, lon), dtype=np.float64).reshape(-1)
            else:
                h = np.zeros(N, dtype=np.float64)

            P = llh_to_xyz(lat, lon, h)
            dr = P - S
            f1 = np.sum(dr * V, axis=1)  # zero-doppler
            rng = np.linalg.norm(dr, axis=1)
            f2 = rng - R

            # 收敛：range 误差（m）+ zero-doppler 相对误差
            if np.max(np.abs(f2)) < tol and np.max(np.abs(f1) / (scale_f1 + 1e-12)) < 1e-6:
                break

            # 数值 Jacobian（中心差分）
            lat_p = lat + dlat
            lat_m = lat - dlat
            lon_p = lon + dlon
            lon_m = lon - dlon

            if self.dem is not None:
                h_lat_p = np.asarray(self._get_dem_height(lat_p, lon), dtype=np.float64).reshape(-1)
                h_lat_m = np.asarray(self._get_dem_height(lat_m, lon), dtype=np.float64).reshape(-1)
                h_lon_p = np.asarray(self._get_dem_height(lat, lon_p), dtype=np.float64).reshape(-1)
                h_lon_m = np.asarray(self._get_dem_height(lat, lon_m), dtype=np.float64).reshape(-1)
            else:
                h_lat_p = h_lat_m = h_lon_p = h_lon_m = np.zeros(N, dtype=np.float64)

            P_lat_p = llh_to_xyz(lat_p, lon, h_lat_p)
            P_lat_m = llh_to_xyz(lat_m, lon, h_lat_m)
            P_lon_p = llh_to_xyz(lat, lon_p, h_lon_p)
            P_lon_m = llh_to_xyz(lat, lon_m, h_lon_m)

            dr_lat_p = P_lat_p - S
            dr_lat_m = P_lat_m - S
            dr_lon_p = P_lon_p - S
            dr_lon_m = P_lon_m - S

            f1_lat_p = np.sum(dr_lat_p * V, axis=1)
            f1_lat_m = np.sum(dr_lat_m * V, axis=1)
            f1_lon_p = np.sum(dr_lon_p * V, axis=1)
            f1_lon_m = np.sum(dr_lon_m * V, axis=1)

            rng_lat_p = np.linalg.norm(dr_lat_p, axis=1)
            rng_lat_m = np.linalg.norm(dr_lat_m, axis=1)
            rng_lon_p = np.linalg.norm(dr_lon_p, axis=1)
            rng_lon_m = np.linalg.norm(dr_lon_m, axis=1)

            f2_lat_p = rng_lat_p - R
            f2_lat_m = rng_lat_m - R
            f2_lon_p = rng_lon_p - R
            f2_lon_m = rng_lon_m - R

            df1_dlat = (f1_lat_p - f1_lat_m) / (2.0 * dlat)
            df1_dlon = (f1_lon_p - f1_lon_m) / (2.0 * dlon)
            df2_dlat = (f2_lat_p - f2_lat_m) / (2.0 * dlat)
            df2_dlon = (f2_lon_p - f2_lon_m) / (2.0 * dlon)

            det = df1_dlat * df2_dlon - df1_dlon * df2_dlat
            good = np.abs(det) > 1e-12
            if not np.any(good):
                break

            dlat_upd = np.zeros(N, dtype=np.float64)
            dlon_upd = np.zeros(N, dtype=np.float64)
            dlat_upd[good] = (-f1[good] * df2_dlon[good] + f2[good] * df1_dlon[good]) / det[good]
            dlon_upd[good] = (f1[good] * df2_dlat[good] - f2[good] * df1_dlat[good]) / det[good]

            dlat_upd = np.clip(dlat_upd, -max_step, max_step)
            dlon_upd = np.clip(dlon_upd, -max_step, max_step)

            # 阻尼/线搜索：确保目标函数下降（提高鲁棒性）
            obj0 = objective(f1, f2)
            alpha = 1.0
            accepted = np.zeros(N, dtype=bool)
            lat_new = lat.copy()
            lon_new = lon.copy()
            for _ls in range(10):
                lat_try = lat + alpha * dlat_upd
                lon_try = lon + alpha * dlon_upd
                lon_try = (lon_try + 180.0) % 360.0 - 180.0
                if self.dem is not None:
                    h_try = np.asarray(self._get_dem_height(lat_try, lon_try), dtype=np.float64).reshape(-1)
                else:
                    h_try = np.zeros(N, dtype=np.float64)
                P_try = llh_to_xyz(lat_try, lon_try, h_try)
                dr_try = P_try - S
                f1_try = np.sum(dr_try * V, axis=1)
                f2_try = np.linalg.norm(dr_try, axis=1) - R
                obj_try = objective(f1_try, f2_try)

                better = obj_try < obj0
                if np.any(better):
                    lat_new[better] = lat_try[better]
                    lon_new[better] = lon_try[better]
                    accepted[better] = True

                if np.all(accepted):
                    break
                alpha *= 0.5

            # 没有任何点接受更新就退出，避免死循环
            if not np.any(accepted):
                break
            lat = lat_new
            lon = lon_new

        # 最终点
        if self.dem is not None:
            h = np.asarray(self._get_dem_height(lat, lon), dtype=np.float64).reshape(-1)
        else:
            h = np.zeros(N, dtype=np.float64)
        P = llh_to_xyz(lat, lon, h)

        shadow_mask, layover_mask = self.compute_shadow_layover(P, S)

        if np.isscalar(r) and np.isscalar(a):
            return P.reshape(-1), bool(shadow_mask[0]), bool(layover_mask[0])
        return P, shadow_mask, layover_mask


############################################################
# DEM读取
############################################################

def load_dem(fname):
    return load_dem_windowed(fname, crop_bounds_ll=None)


def _xy_grid_from_transform(transform, height, width):
    """根据仿射变换生成像素中心的 x/y 网格（2D）。"""
    # Affine: x = c + a*col + b*row; y = f + d*col + e*row
    # 像素中心用 col+0.5,row+0.5
    if transform.b == 0.0 and transform.d == 0.0:
        cols = (np.arange(width, dtype=np.float64) + 0.5)
        rows = (np.arange(height, dtype=np.float64) + 0.5)
        xs = transform.c + transform.a * cols
        ys = transform.f + transform.e * rows
        x2d = np.tile(xs[None, :], (height, 1))
        y2d = np.tile(ys[:, None], (1, width))
        return x2d, y2d

    rows, cols = np.indices((height, width), dtype=np.float64)
    cols = cols + 0.5
    rows = rows + 0.5
    x2d = transform.c + transform.a * cols + transform.b * rows
    y2d = transform.f + transform.d * cols + transform.e * rows
    return x2d, y2d


def load_dem_windowed(fname, crop_bounds_ll=None):
    """
    读取 DEM，并可选按经纬度范围裁剪以减少数据量。

    参数：
    - crop_bounds_ll: None 或 (lat_min, lon_min, lat_max, lon_max)，单位：度（EPSG:4326）
      说明：若 DEM 本身不是 EPSG:4326，会先把裁剪范围变换到 DEM CRS 后裁剪；
           输出的 lat/lon 仍然统一返回 EPSG:4326。

    返回：
    - lat, lon: 2D float32（与 dem 同 shape），单位：度
    - dem: 2D float32
    """
    try:
        with rasterio.open(fname) as ds:
            window = None
            if crop_bounds_ll is not None:
                lat_min, lon_min, lat_max, lon_max = crop_bounds_ll
                # 变换裁剪范围到 DEM CRS（如果需要）
                if ds.crs is None:
                    # 没 CRS 时只能假定是 EPSG:4326
                    left, bottom, right, top = lon_min, lat_min, lon_max, lat_max
                else:
                    if str(ds.crs).upper() in ("EPSG:4326", "WGS84") or ds.crs.is_geographic:
                        left, bottom, right, top = lon_min, lat_min, lon_max, lat_max
                    else:
                        left, bottom, right, top = rasterio.warp.transform_bounds(
                            "EPSG:4326",
                            ds.crs,
                            lon_min,
                            lat_min,
                            lon_max,
                            lat_max,
                            densify_pts=21,
                        )

                window = rasterio.windows.from_bounds(left, bottom, right, top, transform=ds.transform)
                window = window.round_offsets().round_lengths()
                full = rasterio.windows.Window(0, 0, ds.width, ds.height)
                window = window.intersection(full)

            dem = ds.read(1, window=window).astype(np.float32)
            transform = ds.window_transform(window) if window is not None else ds.transform

            x2d, y2d = _xy_grid_from_transform(transform, dem.shape[0], dem.shape[1])

            # 输出统一为 EPSG:4326 的 lat/lon
            if ds.crs is None or ds.crs.is_geographic or str(ds.crs).upper() in ("EPSG:4326", "WGS84"):
                lon = x2d.astype(np.float32)
                lat = y2d.astype(np.float32)
            else:
                # 投影坐标转经纬度（分块避免一次性过大）
                xs = x2d.reshape(-1)
                ys = y2d.reshape(-1)
                lon = np.empty_like(xs, dtype=np.float64)
                lat = np.empty_like(ys, dtype=np.float64)

                chunk = 1_000_000
                for i0 in range(0, xs.size, chunk):
                    i1 = min(i0 + chunk, xs.size)
                    lo, la = rasterio.warp.transform(ds.crs, "EPSG:4326", xs[i0:i1].tolist(), ys[i0:i1].tolist())
                    lon[i0:i1] = np.asarray(lo, dtype=np.float64)
                    lat[i0:i1] = np.asarray(la, dtype=np.float64)

                lon = lon.reshape(dem.shape).astype(np.float32)
                lat = lat.reshape(dem.shape).astype(np.float32)

        return lat, lon, dem
    except Exception as e:
        print(f"读取DEM文件失败: {e}")
        raise


def _yaml_corner_bounds_ll(geo, margin_km=0.0):
    """
    从 YAML 四角点估计覆盖范围（经纬度外包矩形），并可加缓冲。
    返回：(lat_min, lon_min, lat_max, lon_max)
    """
    if getattr(geo, "_corner_latlon", None) is None:
        return None

    lat_tl, lon_tl, lat_tr, lon_tr, lat_bl, lon_bl, lat_br, lon_br = geo._corner_latlon
    lat_min = min(lat_tl, lat_tr, lat_bl, lat_br)
    lat_max = max(lat_tl, lat_tr, lat_bl, lat_br)
    lon_min = min(lon_tl, lon_tr, lon_bl, lon_br)
    lon_max = max(lon_tl, lon_tr, lon_bl, lon_br)

    margin_km = float(margin_km)
    if margin_km > 0.0:
        lat_c = 0.5 * (lat_min + lat_max)
        dlat = margin_km / 111.32
        dlon = margin_km / (111.32 * max(0.1, np.cos(np.deg2rad(lat_c))))
        lat_min -= dlat
        lat_max += dlat
        lon_min -= dlon
        lon_max += dlon

    return lat_min, lon_min, lat_max, lon_max


############################################################
# 块处理DEM数据
############################################################

def process_dem_in_blocks_parallel(geo, lat, lon, h, block_size=None):
    """并行分块处理DEM数据"""
    total_points = lat.size
    
    # 自适应块大小
    if block_size is None:
        # 根据可用内存和DEM大小计算块大小
        import psutil
        available_memory = psutil.virtual_memory().available
        # 每个点大约需要 24 bytes (3 * float64)
        estimated_memory_per_point = 24
        max_block_size = available_memory // (estimated_memory_per_point * 10)  # 留10倍余量
        block_size = min(100000, max(10000, max_block_size))
    
    num_blocks = (total_points + block_size - 1) // block_size
    
    sar_dem = np.full((geo.na, geo.nr), np.nan, dtype=np.float32)
    all_r = []
    all_a = []
    all_R = []
    all_t = []
    
    print(f"并行分块处理DEM数据，共{num_blocks}块，每块{block_size}点")
    
    # 准备任务
    tasks = []
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    h_flat = h.flatten()
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, total_points)
        tasks.append((lat_flat[start:end], lon_flat[start:end], h_flat[start:end], start, end))
    
    # 并行处理
    from multiprocessing import Pool
    cpu_count = os.cpu_count()
    # 使用90%的CPU核心
    used_cpu_count = max(1, int(cpu_count * 0.9))
    print(f"使用 {used_cpu_count} 个CPU核心并行处理")
    
    with Pool(processes=used_cpu_count, initializer=_init_dem_worker, initargs=(geo,)) as pool:
        results = pool.map(process_block_worker, tasks)
    
    # 收集结果
    for i, (r, a, R, t, start, end) in enumerate(results):
        # 限制时间在轨道范围内
        t = np.clip(t, geo.orbit.tmin, geo.orbit.tmax)
        
        # 更新SAR DEM
        r_rounded = np.round(r).astype(int)
        a_rounded = np.round(a).astype(int)
        mask = (r_rounded >= 0) & (r_rounded < geo.nr) & (a_rounded >= 0) & (a_rounded < geo.na)
        
        if np.any(mask):
            sar_dem[a_rounded[mask], r_rounded[mask]] = h_flat[start:end][mask]
        
        # 收集结果
        all_r.extend(r)
        all_a.extend(a)
        all_R.extend(R)
        all_t.extend(t)
    
    return sar_dem, np.array(all_r), np.array(all_a), np.array(all_R), np.array(all_t)

_DEM_WORKER_GEO = None
_DEM_WORKER_SHM = None
_DEM_WORKER_LAT = None
_DEM_WORKER_LON = None
_DEM_WORKER_H = None

def _init_dem_worker(geo):
    global _DEM_WORKER_GEO
    _DEM_WORKER_GEO = geo

def process_block_worker(args):
    """处理单个DEM块（worker 侧）"""
    lat_block, lon_block, h_block, start, end = args
    P = llh_to_xyz(lat_block, lon_block, h_block)
    r, a, R, t, look_dir = _DEM_WORKER_GEO.geo2rdr(P)
    return r, a, R, t, start, end


def _init_dem_worker_shm(yaml_file, geo_overrides, lat_name, lon_name, h_name, n_points):
    """
    SharedMemory worker 初始化：
    - 连接 DEM 的 lat/lon/h 共享内存
    - 在 worker 内构建 Geo2Rdr（避免 pickle 复杂对象）
    """
    global _DEM_WORKER_GEO, _DEM_WORKER_SHM, _DEM_WORKER_LAT, _DEM_WORKER_LON, _DEM_WORKER_H

    shm_lat = shared_memory.SharedMemory(name=lat_name)
    shm_lon = shared_memory.SharedMemory(name=lon_name)
    shm_h = shared_memory.SharedMemory(name=h_name)
    _DEM_WORKER_SHM = (shm_lat, shm_lon, shm_h)

    n = int(n_points)
    _DEM_WORKER_LAT = np.ndarray((n,), dtype=np.float32, buffer=shm_lat.buf)
    _DEM_WORKER_LON = np.ndarray((n,), dtype=np.float32, buffer=shm_lon.buf)
    _DEM_WORKER_H = np.ndarray((n,), dtype=np.float32, buffer=shm_h.buf)

    def _cleanup():
        try:
            for s in _DEM_WORKER_SHM or ():
                try:
                    s.close()
                except Exception:
                    pass
        except Exception:
            pass

    atexit.register(_cleanup)

    yaml_data = yaml.safe_load(open(yaml_file))
    geo = Geo2Rdr(yaml_data)
    if isinstance(geo_overrides, dict):
        if "near_range" in geo_overrides and geo_overrides["near_range"] is not None:
            geo.near_range = float(geo_overrides["near_range"])
        if "t0" in geo_overrides and geo_overrides["t0"] is not None:
            geo.t0 = float(geo_overrides["t0"])
    _DEM_WORKER_GEO = geo


def process_block_worker_shm(args):
    """SharedMemory worker：只传 start/end，避免大数组 pickle。"""
    start, end = args
    lat_block = _DEM_WORKER_LAT[start:end]
    lon_block = _DEM_WORKER_LON[start:end]
    h_block = _DEM_WORKER_H[start:end]

    P = llh_to_xyz(lat_block, lon_block, h_block)
    r, a, R, t, look_dir = _DEM_WORKER_GEO.geo2rdr(P)
    # 父进程只需要 r/a/R；t/look_dir 不用于流式落像
    return r.astype(np.float32, copy=False), a.astype(np.float32, copy=False), R.astype(np.float32, copy=False), start, end

def process_dem_in_blocks(geo, lat, lon, h, block_size=None):
    """分块处理DEM数据"""
    total_points = lat.size
    
    # 自适应块大小
    if block_size is None:
        # 根据可用内存和DEM大小计算块大小
        import psutil
        available_memory = psutil.virtual_memory().available
        # 每个点大约需要 24 bytes (3 * float64)
        estimated_memory_per_point = 24
        max_block_size = available_memory // (estimated_memory_per_point * 10)  # 留10倍余量
        block_size = min(100000, max(10000, max_block_size))
    
    num_blocks = (total_points + block_size - 1) // block_size
    
    sar_dem = np.full((geo.na, geo.nr), np.nan, dtype=np.float32)
    all_r = []
    all_a = []
    all_R = []
    all_t = []
    
    print(f"分块处理DEM数据，共{num_blocks}块，每块{block_size}点")
    
    t_guess = None
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, total_points)
        
        # print(f"处理块 {i+1}/{num_blocks}，点范围: {start}-{end}")
        print('+', end=' ', flush=True) 
        
        # 处理当前块
        P = llh_to_xyz(lat.flatten()[start:end], lon.flatten()[start:end], h.flatten()[start:end])
        r, a, R, t, look_dir = geo.geo2rdr(P, t_init=t_guess)
        # 用当前块的中位数时间作为下一块初值（通常能减少迭代次数）
        if t.size > 0:
            t_guess = float(np.median(t))
        
        # 限制时间在轨道范围内
        t = np.clip(t, geo.orbit.tmin, geo.orbit.tmax)
        
        # 更新SAR DEM
        r_rounded = np.round(r).astype(int)
        a_rounded = np.round(a).astype(int)
        mask = (r_rounded >= 0) & (r_rounded < geo.nr) & (a_rounded >= 0) & (a_rounded < geo.na)
        
        if np.any(mask):
            sar_dem[a_rounded[mask], r_rounded[mask]] = h.flatten()[start:end][mask]
        
        # 增量收集结果，避免一次性存储所有数据
        all_r.extend(r)
        all_a.extend(a)
        all_R.extend(R)
        all_t.extend(t)
        
        # 显式垃圾回收
        del P, r, a, R, t, r_rounded, a_rounded, mask
    
    return sar_dem, np.array(all_r), np.array(all_a), np.array(all_R), np.array(all_t)


def dem_to_sar_products_streaming(
    geo,
    lat,
    lon,
    h,
    block_size=100000,
    parallel=True,
    yaml_file=None,
    geo_overrides=None,
    use_shared_memory=True,
):
    """
    流式 DEM->SAR：分块计算并直接填充 sar_dem/slc，不再收集全量 r/a/R/t 列表。
    适用于大 DEM（上千万点），显著降低内存开销。

    返回：
    - sar_dem: (na,nr) float32，像素对应 DEM 高程（无点为 NaN）
    - slc: (na,nr) complex64，按 R 生成的相位信号（未命中为 0）
    """
    total_points = lat.size
    if block_size is None or block_size <= 0:
        block_size = total_points

    sar_dem = np.full((geo.na, geo.nr), np.nan, dtype=np.float32)
    slc = np.zeros((geo.na, geo.nr), dtype=np.complex64)

    lat_flat = np.asarray(lat, dtype=np.float32).reshape(-1)
    lon_flat = np.asarray(lon, dtype=np.float32).reshape(-1)
    h_flat = np.asarray(h, dtype=np.float32).reshape(-1)

    num_blocks = (total_points + block_size - 1) // block_size
    print(f"DEM→SAR 流式处理：共{num_blocks}块，每块{block_size}点")

    def _apply_block(r, a, R, start, end):
        r_rounded = np.round(r).astype(np.int64)
        a_rounded = np.round(a).astype(np.int64)
        mask = (r_rounded >= 0) & (r_rounded < geo.nr) & (a_rounded >= 0) & (a_rounded < geo.na)
        if not np.any(mask):
            return 0
        hh = h_flat[start:end]
        sar_dem[a_rounded[mask], r_rounded[mask]] = hh[mask].astype(np.float32, copy=False)
        phase = 4.0 * np.pi * R[mask] / geo.wavelength
        slc[a_rounded[mask], r_rounded[mask]] = np.exp(1j * phase).astype(np.complex64, copy=False)
        return int(np.sum(mask))

    if not parallel:
        t_guess = None
        hit = 0
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, total_points)
            if i % 10 == 0:
                print('+', end=' ', flush=True)

            P = llh_to_xyz(lat_flat[start:end], lon_flat[start:end], h_flat[start:end])
            r, a, R, t, _ = geo.geo2rdr(P, t_init=t_guess)
            if t.size > 0:
                t_guess = float(np.median(t))
            hit += _apply_block(r, a, R, start, end)
            del P, r, a, R, t
        print("")
        print(f"流式命中像素数（写入次数）: {hit}")
        return sar_dem, slc

    # 并行：worker 只负责 geo2rdr，parent 负责落数组（避免跨进程共享写）
    from multiprocessing import Pool
    cpu_count = os.cpu_count() or 1
    used_cpu_count = max(1, int(cpu_count * 0.9))
    print(f"使用 {used_cpu_count} 个CPU核心并行处理")

    hit = 0
    done = 0
    if use_shared_memory:
        if yaml_file is None:
            raise ValueError("use_shared_memory=True 时必须提供 yaml_file（用于 worker 内构建 Geo2Rdr）")

        # 建共享内存（避免把 lat/lon/h 切片 pickle 给子进程）
        shm_lat = shared_memory.SharedMemory(create=True, size=lat_flat.nbytes)
        shm_lon = shared_memory.SharedMemory(create=True, size=lon_flat.nbytes)
        shm_h = shared_memory.SharedMemory(create=True, size=h_flat.nbytes)

        try:
            lat_sh = np.ndarray(lat_flat.shape, dtype=np.float32, buffer=shm_lat.buf)
            lon_sh = np.ndarray(lon_flat.shape, dtype=np.float32, buffer=shm_lon.buf)
            h_sh = np.ndarray(h_flat.shape, dtype=np.float32, buffer=shm_h.buf)
            lat_sh[:] = lat_flat
            lon_sh[:] = lon_flat
            h_sh[:] = h_flat

            def _task_iter_idx():
                for i in range(num_blocks):
                    start = i * block_size
                    end = min((i + 1) * block_size, total_points)
                    yield (start, end)

            with Pool(
                processes=used_cpu_count,
                initializer=_init_dem_worker_shm,
                initargs=(yaml_file, geo_overrides or {}, shm_lat.name, shm_lon.name, shm_h.name, total_points),
            ) as pool:
                for r, a, R, start, end in pool.imap_unordered(process_block_worker_shm, _task_iter_idx(), chunksize=4):
                    done += 1
                    if done % 10 == 0:
                        print('+', end=' ', flush=True)
                    hit += _apply_block(r, a, R, start, end)
                    del r, a, R
        finally:
            # 释放共享内存（先 close，再 unlink）
            for s in (shm_lat, shm_lon, shm_h):
                try:
                    s.close()
                except Exception:
                    pass
            for s in (shm_lat, shm_lon, shm_h):
                try:
                    s.unlink()
                except Exception:
                    pass

        print("")
        print(f"流式命中像素数（写入次数）: {hit}")
        return sar_dem, slc

    # 兼容旧并行方式（仍会 pickle 切片，速度较慢）
    def _task_iter():
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, total_points)
            yield (lat_flat[start:end], lon_flat[start:end], h_flat[start:end], start, end)

    with Pool(processes=used_cpu_count, initializer=_init_dem_worker, initargs=(geo,)) as pool:
        for r, a, R, t, start, end in pool.imap_unordered(process_block_worker, _task_iter(), chunksize=1):
            done += 1
            if done % 10 == 0:
                print('+', end=' ', flush=True)
            hit += _apply_block(r, a, R, start, end)
            del r, a, R, t
    print("")
    print(f"流式命中像素数（写入次数）: {hit}")
    return sar_dem, slc


############################################################
# DEM → SAR DEM
############################################################

def dem_to_sar_dem(geo,lat,lon,h):

    P=llh_to_xyz(lat.flatten(),lon.flatten(),h.flatten())

    r,a,R,t,look_dir=geo.geo2rdr(P)

    # 限制时间在轨道范围内
    t = np.clip(t, geo.orbit.tmin, geo.orbit.tmax)

    sar_dem=np.full((geo.na,geo.nr),np.nan, dtype=np.float32)

    r=np.round(r).astype(int)
    a=np.round(a).astype(int)

    mask=(r>=0)&(r<geo.nr)&(a>=0)&(a<geo.na)

    sar_dem[a[mask],r[mask]]=h.flatten()[mask]

    return sar_dem,r,a,R,t


def scientific_processing_pipeline(geo, lat, lon, h, dem=None, dem_lat=None, dem_lon=None):
    """
    科研级处理流程
    输入:
        geo: Geo2Rdr 实例
        lat, lon, h: DEM 经纬度和高程
        dem, dem_lat, dem_lon: 用于高程修正的 DEM 数据 (可选)
    输出:
        P: 地面坐标 (N,3)
        shadow_mask, layover_mask: 阴影和叠掩掩码
        range_pixel, az_pixel, R, t, look_dir: Geo2Rdr 输出
    """
    # 1. DEM → XYZ
    P_xyz = llh_to_xyz(lat.flatten(), lon.flatten(), h.flatten())
    
    # 2. Geo2Rdr (高精度)
    range_pixel, az_pixel, R, t, look_dir = geo.geo2rdr(P_xyz, dem, dem_lat, dem_lon)
    
    # 3. 限制时间在轨道范围内
    t = np.clip(t, geo.orbit.tmin, geo.orbit.tmax)
    
    # 4. Rdr2Geo (使用 Geo2Rdr 输出的 slant_range 和 azimuth_time)
    rdr2geo = Rdr2Geo(geo, dem, dem_lat, dem_lon)
    P, shadow_mask, layover_mask = rdr2geo.solve_with_geo2rdr_output(R, t)
    
    return P, shadow_mask, layover_mask, range_pixel, az_pixel, R, t, look_dir


def sar_to_latlon_grid_from_yaml(geo, step=1):
    """
    基于 YAML 四角点的双线性插值生成整幅 SAR 像素的 (lat, lon) 网格。
    这是“元数据一致”的快速方法，不做基于轨道的严格 rdr2geo 反解。
    """
    if getattr(geo, "_corner_latlon", None) is None:
        raise ValueError("Geo2Rdr 缺少角点经纬度信息，无法从 YAML 生成 lat/lon 网格")

    lat_tl, lon_tl, lat_tr, lon_tr, lat_bl, lon_bl, lat_br, lon_br = geo._corner_latlon

    step = int(step)
    if step <= 0:
        raise ValueError("step must be >= 1")

    r_out = compute_sample_indices(geo.nr, step).astype(np.float64)
    a_out = compute_sample_indices(geo.na, step).astype(np.float64)

    u = r_out / max(1.0, float(geo.nr - 1))
    v = a_out / max(1.0, float(geo.na - 1))

    lat_top = lat_tl * (1.0 - u) + lat_tr * u
    lat_bot = lat_bl * (1.0 - u) + lat_br * u
    lon_top = lon_tl * (1.0 - u) + lon_tr * u
    lon_bot = lon_bl * (1.0 - u) + lon_br * u

    lat_grid = (lat_top[None, :] * (1.0 - v)[:, None] + lat_bot[None, :] * v[:, None]).astype(np.float32)
    lon_grid = (lon_top[None, :] * (1.0 - v)[:, None] + lon_bot[None, :] * v[:, None]).astype(np.float32)
    return lat_grid, lon_grid


def compute_sample_indices(size, step):
    """生成 0..size-1 的抽样索引，保证包含最后一个索引。"""
    step = int(step)
    if step <= 0:
        raise ValueError("step must be >= 1")
    idx = np.arange(0, int(size), step, dtype=np.int64)
    if idx.size == 0:
        idx = np.array([0], dtype=np.int64)
    if idx[-1] != int(size) - 1:
        idx = np.unique(np.r_[idx, int(size) - 1]).astype(np.int64)
    return idx


def _fill_nan_nearest_2d(arr):
    """用最近邻把 2D 数组中的 NaN 填满（用于插值前的控制点修复）"""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("_fill_nan_nearest_2d expects 2D array")
    mask = np.isfinite(arr)
    if np.all(mask):
        return arr
    if not np.any(mask):
        raise ValueError("control points are all NaN")

    from scipy.ndimage import distance_transform_edt
    _, indices = distance_transform_edt(~mask, return_indices=True)
    filled = arr[tuple(indices)]
    return filled


def sar_to_latlon_grid_sparse_rdr2geo(
    geo,
    dem_lat,
    dem_lon,
    dem_h,
    sparse_step=50,
    output_step=1,
    max_iter=20,
    tol=1e-3,
    chunk_points=20000,
):
    """
    稀疏点 rdr2geo + 2D 插值生成 SAR 像素地理坐标网格。

    思路：
    - 在 SAR 像素网格上按 sparse_step 抽样少量像素点；
    - 对这些像素点做严谨的 rdr2geo 反解得到控制点 (lat/lon)；
    - 对控制点用规则网格插值，补全到输出网格（由 output_step 控制输出分辨率）。

    参数：
    - sparse_step: 控制点间隔（像素），越大越快但越粗糙，建议 30~200
    - output_step: 输出网格步长（像素），1 表示输出全分辨率；>1 输出降采样网格
    - chunk_points: 每批求解的控制点数量，控制内存峰值
    """
    sparse_step = int(sparse_step)
    output_step = int(output_step)
    if sparse_step <= 0 or output_step <= 0:
        raise ValueError("sparse_step/output_step must be >= 1")

    # 控制点像素坐标（行=a, 列=r）
    a_idx = compute_sample_indices(geo.na, sparse_step)
    r_idx = compute_sample_indices(geo.nr, sparse_step)

    A, Rr = np.meshgrid(a_idx, r_idx, indexing="ij")
    a_samp = A.reshape(-1).astype(np.float64)
    r_samp = Rr.reshape(-1).astype(np.float64)

    rdr2geo = Rdr2Geo(geo, dem=dem_h, dem_lat=dem_lat, dem_lon=dem_lon)

    lat_cp = np.full(a_samp.shape[0], np.nan, dtype=np.float64)
    lon_cp = np.full(a_samp.shape[0], np.nan, dtype=np.float64)

    # 分批求解控制点，避免一次性分配过大
    n_pts = a_samp.size
    for start in range(0, n_pts, int(chunk_points)):
        end = min(start + int(chunk_points), n_pts)
        try:
            P, _, _ = rdr2geo.solve(r_samp[start:end], a_samp[start:end], max_iter=max_iter, tol=tol)
            la, lo, _ = xyz_to_llh_batch(np.asarray(P, dtype=np.float64))
            lat_cp[start:end] = la
            lon_cp[start:end] = lo
        except Exception:
            # 保持 NaN，后续用最近邻填充
            pass

    lat_cp = lat_cp.reshape(A.shape)
    lon_cp = lon_cp.reshape(A.shape)

    # 修复控制点 NaN
    lat_cp = _fill_nan_nearest_2d(lat_cp)
    lon_cp = _fill_nan_nearest_2d(lon_cp)

    # 经度展开，避免插值穿越 180 度造成跳变
    lon_rad = np.deg2rad(lon_cp)
    lon_rad = np.unwrap(lon_rad, axis=1)
    lon_rad = np.unwrap(lon_rad, axis=0)
    lon_cp_unw = np.rad2deg(lon_rad)

    # 输出网格坐标
    a_out = compute_sample_indices(geo.na, output_step).astype(np.float64)
    r_out = compute_sample_indices(geo.nr, output_step).astype(np.float64)

    from scipy.interpolate import RegularGridInterpolator
    lat_itp = RegularGridInterpolator((a_idx.astype(np.float64), r_idx.astype(np.float64)), lat_cp, bounds_error=False, fill_value=None)
    lon_itp = RegularGridInterpolator((a_idx.astype(np.float64), r_idx.astype(np.float64)), lon_cp_unw, bounds_error=False, fill_value=None)

    # 逐行插值：避免一次性生成巨大的 (N,2) pts / meshgrid 导致内存暴涨
    na_out = a_out.size
    nr_out = r_out.size
    est_bytes = na_out * nr_out * 8  # two float32 grids
    if est_bytes > 2_000_000_000:
        print(
            f"警告：lat/lon 输出网格约占 {est_bytes/1024/1024/1024:.2f} GiB（不含其它数组/开销）。"
            "建议增大 --geocode-step 或仅输出降采样网格。"
        )

    lat_out = np.empty((na_out, nr_out), dtype=np.float32)
    lon_out = np.empty((na_out, nr_out), dtype=np.float32)

    pts = np.empty((nr_out, 2), dtype=np.float64)
    pts[:, 1] = r_out
    for i, a_val in enumerate(a_out):
        pts[:, 0] = a_val
        lat_out[i, :] = lat_itp(pts).astype(np.float32)
        lon_out[i, :] = lon_itp(pts).astype(np.float32)

    lon_out = ((lon_out + 180.0) % 360.0 - 180.0).astype(np.float32)
    return lat_out, lon_out


def sar_to_latlon_grid_high_precision(
    geo,
    R,
    t,
    lat,
    lon,
    h,
    method="yaml",
    output_step=1,
    sparse_step=50,
):
    """
    生成 SAR 像素地理坐标网格。
    说明：
        - 旧实现对整幅影像逐像素 Python 循环，极慢且与几何约束不严谨。
        - yaml：只用四角点双线性插值，速度最快（推荐大图默认）。
        - sparse：按稀疏控制点做 rdr2geo 反解，再用规则网格 2D 插值补全。
    输入:
        geo: Geo2Rdr 实例
        R: 斜距数组 (N,)
        t: 方位时间数组 (N,)
        lat, lon, h: DEM 数据，用于高程修正
    输出:
        lat_grid, lon_grid: 地理坐标网格
    """
    if method == "yaml":
        return sar_to_latlon_grid_from_yaml(geo, step=output_step)
    if method == "sparse":
        return sar_to_latlon_grid_sparse_rdr2geo(
            geo,
            lat,
            lon,
            h,
            sparse_step=sparse_step,
            output_step=output_step,
        )
    raise ValueError(f"Unsupported geocode method: {method}")


def write_latlon_grid_bin(
    geo,
    dem_lat,
    dem_lon,
    dem_h,
    output_prefix,
    method="yaml",
    output_step=1,
    sparse_step=50,
    chunk_rows=256,
    max_iter=20,
    tol=1e-3,
):
    """
    二进制/分块写出 lat/lon 网格（不在内存中保留整幅网格）。

    输出：
    - {output_prefix}_lat_grid_f32le.bin  (C-order, row-major)
    - {output_prefix}_lon_grid_f32le.bin
    - {output_prefix}_latlon_grid_meta.yaml  (包含 shape、dtype、索引、方法和关键参数)
    """
    output_step = int(output_step)
    sparse_step = int(sparse_step)
    chunk_rows = int(chunk_rows)
    if output_step <= 0 or sparse_step <= 0 or chunk_rows <= 0:
        raise ValueError("output_step/sparse_step/chunk_rows must be >= 1")

    a_out = compute_sample_indices(geo.na, output_step)
    r_out = compute_sample_indices(geo.nr, output_step)
    na_out = a_out.size
    nr_out = r_out.size

    lat_path = f"{output_prefix}_lat_grid_f32le.bin"
    lon_path = f"{output_prefix}_lon_grid_f32le.bin"
    meta_path = f"{output_prefix}_latlon_grid_meta.yaml"

    meta = {
        "format": "raw_binary",
        "byte_order": "little_endian",
        "dtype": "float32",
        "order": "C",
        "shape": {"nrows": int(na_out), "ncols": int(nr_out)},
        "source_sar_shape": {"nrows": int(geo.na), "ncols": int(geo.nr)},
        "pixel_indices": {
            "azimuth_lines": a_out.astype(int).tolist(),
            "range_pixels": r_out.astype(int).tolist(),
        },
        "geocode": {
            "method": str(method),
            "output_step": int(output_step),
            "sparse_step": int(sparse_step),
            "rdr2geo_max_iter": int(max_iter),
            "rdr2geo_tol_m": float(tol),
        },
        # 兼容两种键：files.lat/files.lon 以及 files.lat_bin/files.lon_bin
        "files": {
            "lat": os.path.basename(lat_path),
            "lon": os.path.basename(lon_path),
            "lat_bin": os.path.basename(lat_path),
            "lon_bin": os.path.basename(lon_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    # 分块写文件
    with open(lat_path, "wb") as flat, open(lon_path, "wb") as flon:
        if method == "yaml":
            if getattr(geo, "_corner_latlon", None) is None:
                raise ValueError("Geo2Rdr 缺少角点经纬度信息，无法用 yaml 方法写出网格")

            lat_tl, lon_tl, lat_tr, lon_tr, lat_bl, lon_bl, lat_br, lon_br = geo._corner_latlon
            u = (r_out.astype(np.float64) / max(1.0, float(geo.nr - 1))).astype(np.float64)
            lat_top = lat_tl * (1.0 - u) + lat_tr * u
            lat_bot = lat_bl * (1.0 - u) + lat_br * u
            lon_top = lon_tl * (1.0 - u) + lon_tr * u
            lon_bot = lon_bl * (1.0 - u) + lon_br * u

            for i0 in range(0, na_out, chunk_rows):
                i1 = min(i0 + chunk_rows, na_out)
                v = (a_out[i0:i1].astype(np.float64) / max(1.0, float(geo.na - 1))).astype(np.float64)
                lat_blk = (lat_top[None, :] * (1.0 - v)[:, None] + lat_bot[None, :] * v[:, None]).astype(np.float32)
                lon_blk = (lon_top[None, :] * (1.0 - v)[:, None] + lon_bot[None, :] * v[:, None]).astype(np.float32)
                lon_blk = ((lon_blk + 180.0) % 360.0 - 180.0).astype(np.float32)

                np.ascontiguousarray(lat_blk).astype("<f4", copy=False).tofile(flat)
                np.ascontiguousarray(lon_blk).astype("<f4", copy=False).tofile(flon)

        elif method == "sparse":
            # 1) 先解控制点（规模远小于全图，可接受在内存中存）
            a_idx = compute_sample_indices(geo.na, sparse_step)
            r_idx = compute_sample_indices(geo.nr, sparse_step)
            A, Rr = np.meshgrid(a_idx, r_idx, indexing="ij")
            a_samp = A.reshape(-1).astype(np.float64)
            r_samp = Rr.reshape(-1).astype(np.float64)

            rdr2geo = Rdr2Geo(geo, dem=dem_h, dem_lat=dem_lat, dem_lon=dem_lon)
            lat_cp = np.full(a_samp.shape[0], np.nan, dtype=np.float64)
            lon_cp = np.full(a_samp.shape[0], np.nan, dtype=np.float64)

            n_pts = a_samp.size
            chunk_points = 20000
            for start in range(0, n_pts, chunk_points):
                end = min(start + chunk_points, n_pts)
                try:
                    P, _, _ = rdr2geo.solve(r_samp[start:end], a_samp[start:end], max_iter=max_iter, tol=tol)
                    la, lo, _ = xyz_to_llh_batch(np.asarray(P, dtype=np.float64))
                    lat_cp[start:end] = la
                    lon_cp[start:end] = lo
                except Exception:
                    pass

            lat_cp = _fill_nan_nearest_2d(lat_cp.reshape(A.shape))
            lon_cp = _fill_nan_nearest_2d(lon_cp.reshape(A.shape))

            # 经度展开，避免跨 180 度插值跳变
            lon_rad = np.deg2rad(lon_cp)
            lon_rad = np.unwrap(lon_rad, axis=1)
            lon_rad = np.unwrap(lon_rad, axis=0)
            lon_cp_unw = np.rad2deg(lon_rad)

            from scipy.interpolate import RegularGridInterpolator
            lat_itp = RegularGridInterpolator((a_idx.astype(np.float64), r_idx.astype(np.float64)), lat_cp, bounds_error=False, fill_value=None)
            lon_itp = RegularGridInterpolator((a_idx.astype(np.float64), r_idx.astype(np.float64)), lon_cp_unw, bounds_error=False, fill_value=None)

            # 2) 分块插值并写出
            pts = np.empty((nr_out, 2), dtype=np.float64)
            pts[:, 1] = r_out.astype(np.float64)
            for i0 in range(0, na_out, chunk_rows):
                i1 = min(i0 + chunk_rows, na_out)
                lat_blk = np.empty((i1 - i0, nr_out), dtype=np.float32)
                lon_blk = np.empty((i1 - i0, nr_out), dtype=np.float32)

                for ii, a_val in enumerate(a_out[i0:i1].astype(np.float64)):
                    pts[:, 0] = a_val
                    lat_blk[ii, :] = lat_itp(pts).astype(np.float32)
                    lon_blk[ii, :] = lon_itp(pts).astype(np.float32)

                lon_blk = ((lon_blk + 180.0) % 360.0 - 180.0).astype(np.float32)
                np.ascontiguousarray(lat_blk).astype("<f4", copy=False).tofile(flat)
                np.ascontiguousarray(lon_blk).astype("<f4", copy=False).tofile(flon)

        else:
            raise ValueError(f"Unsupported geocode method for binary output: {method}")


def write_latlon_grid_meta_yaml(
    geo,
    output_prefix,
    lat_file,
    lon_file,
    method,
    output_step,
    sparse_step,
):
    """为 lat/lon 网格写 meta.yaml（适用于 tif/npy/bin），便于后处理统一用 meta 读取。"""
    a_out = compute_sample_indices(geo.na, output_step)
    r_out = compute_sample_indices(geo.nr, output_step)
    meta_path = f"{output_prefix}_latlon_grid_meta.yaml"
    meta = {
        "format": "grid",
        "dtype": "float32",
        "byte_order": "little_endian",
        "order": "C",
        "shape": {"nrows": int(a_out.size), "ncols": int(r_out.size)},
        "source_sar_shape": {"nrows": int(geo.na), "ncols": int(geo.nr)},
        "pixel_indices": {
            "azimuth_lines": a_out.astype(int).tolist(),
            "range_pixels": r_out.astype(int).tolist(),
        },
        "geocode": {
            "method": str(method),
            "output_step": int(output_step),
            "sparse_step": int(sparse_step),
        },
        "files": {"lat": os.path.basename(lat_file), "lon": os.path.basename(lon_file)},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    return meta_path


############################################################
# DEM → SLC 模拟
############################################################

def simulate_slc(geo,r,a,R):

    slc=np.zeros((geo.na,geo.nr),dtype=np.complex64)

    phase=4*np.pi*R/geo.wavelength

    signal=np.exp(1j*phase)

    r=np.round(r).astype(int)
    a=np.round(a).astype(int)

    mask=(r>=0)&(r<geo.nr)&(a>=0)&(a<geo.na)

    slc[a[mask],r[mask]]=signal[mask]

    return slc


############################################################
# 并行处理单行SAR像素
############################################################
def process_chunk(args):
    """处理SAR像素块"""
    rdr, start_row, end_row, nr = args
    lat_chunk = np.zeros((end_row - start_row, nr), dtype=np.float32)
    lon_chunk = np.zeros((end_row - start_row, nr), dtype=np.float32)
    
    for a in range(start_row, end_row):
        if (a - start_row) % 50 == 0:
            #print(f"处理块行 {a - start_row}/{end_row - start_row}")
            print('+', end=' ', flush=True)  
        for r in range(nr):
            try:
                xyz, shadow_mask, layover_mask = rdr.solve(r, a)
                # 强制转为一维数组，确保解包正确
                xyz = np.array(xyz).flatten()
                if xyz.shape != (3,):
                    raise ValueError(f"XYZ must be 3-element array, got {xyz.shape}")
                la, lo, _ = xyz_to_llh(xyz[0], xyz[1], xyz[2])
                lat_chunk[a - start_row, r] = la
                lon_chunk[a - start_row, r] = lo
            except Exception as e:
                if (a - start_row) % 100 == 0 and r % 100 == 0:
                    print(f"Error at (r={r}, a={a}): {e}")
                lat_chunk[a - start_row, r] = np.nan
                lon_chunk[a - start_row, r] = np.nan
    
    return start_row, lat_chunk, lon_chunk


@jit(nopython=True, fastmath=True)
def interpolate_orbit(t, precompute_times, precompute_positions, precompute_velocities):
    """Numba兼容的轨道插值"""
    idx = np.searchsorted(precompute_times, t)
    
    if idx <= 0:
        return precompute_positions[0], precompute_velocities[0]
    elif idx >= len(precompute_times):
        return precompute_positions[-1], precompute_velocities[-1]
    
    # 线性插值
    t1 = precompute_times[idx-1]
    t2 = precompute_times[idx]
    p1 = precompute_positions[idx-1]
    p2 = precompute_positions[idx]
    v1 = precompute_velocities[idx-1]
    v2 = precompute_velocities[idx]
    
    frac = (t - t1) / (t2 - t1)
    S = p1 + (p2 - p1) * frac
    V = v1 + (v2 - v1) * frac
    
    return S, V

@jit(nopython=True, fastmath=True)
def least_squares_solve(J, b):
    """Numba兼容的最小二乘求解"""
    # 计算 J^T J
    Jt = J.T
    JtJ = np.dot(J, Jt)
    
    # 计算行列式
    det = JtJ[0,0] * JtJ[1,1] - JtJ[0,1] * JtJ[1,0]
    
    if abs(det) < 1e-10:
        return np.array([0.0, 0.0])
    
    # 计算逆矩阵
    inv_JtJ = np.array([
        [JtJ[1,1], -JtJ[0,1]],
        [-JtJ[1,0], JtJ[0,0]]
    ]) / det
    
    # 计算解
    delta = np.dot(inv_JtJ, np.dot(J, b))
    
    return delta

def process_chunk(args):
    """处理SAR像素块"""
    rdr, start_row, end_row, nr = args
    lat_chunk = np.zeros((end_row - start_row, nr), dtype=np.float32)
    lon_chunk = np.zeros((end_row - start_row, nr), dtype=np.float32)
    
    for a in range(start_row, end_row):
        if (a - start_row) % 50 == 0:
            #print(f"处理块行 {a - start_row}/{end_row - start_row}")
            print('+', end=' ', flush=True)
        for r in range(nr):
            try:
                xyz, shadow_mask, layover_mask = rdr.solve(r, a)
                # 强制转为一维数组，确保解包正确
                xyz = np.array(xyz).flatten()
                if xyz.shape != (3,):
                    raise ValueError(f"XYZ must be 3-element array, got {xyz.shape}")
                la, lo, _ = xyz_to_llh(xyz[0], xyz[1], xyz[2])
                lat_chunk[a - start_row, r] = la
                lon_chunk[a - start_row, r] = lo
            except Exception as e:
                if (a - start_row) % 100 == 0 and r % 100 == 0:
                    print(f"Error at (r={r}, a={a}): {e}")
                lat_chunk[a - start_row, r] = np.nan
                lon_chunk[a - start_row, r] = np.nan
    
    return start_row, lat_chunk, lon_chunk


############################################################
# SAR → LatLon 并行处理
############################################################
def sar_to_latlon_grid_parallel(geo, height=0):
    """并行生成SAR像素地理坐标网格"""
    rdr = Rdr2Geo(geo)
    
    lat = np.zeros((geo.na, geo.nr), dtype=np.float32)
    lon = np.zeros((geo.na, geo.nr), dtype=np.float32)
    
    # 分块处理
    cpu_count = os.cpu_count()
    # 使用90%的CPU核心
    used_cpu_count = max(1, int(cpu_count * 0.9))
    print(f"检测到 {cpu_count} 个CPU核心")
    print(f"使用 {used_cpu_count} 个CPU核心并行处理")
    
    # 计算块大小，确保每个块至少有一定数量的行
    chunk_size = max(50, geo.na // (used_cpu_count * 2))  # 每个CPU处理2个块，每个块至少50行
    tasks = []
    
    for i in range(0, geo.na, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, geo.na)
        tasks.append((
            rdr,
            start_row,
            end_row,
            geo.nr
        ))
    
    print(f"并行处理 {len(tasks)} 个块，共 {geo.na} 行SAR像素")
    print(f"每个块大小: {chunk_size} 行")
    
    # 并行处理
    start_time = time.time()
    with Pool(processes=used_cpu_count) as pool:
        results = pool.map(process_chunk, tasks)
    parallel_time = time.time() - start_time
    print(f"并行处理时间: {parallel_time:.2f}秒")
    
    # 收集结果
    for start_row, lat_chunk, lon_chunk in results:
        end_row = start_row + lat_chunk.shape[0]
        lat[start_row:end_row, :] = lat_chunk
        lon[start_row:end_row, :] = lon_chunk
    
    return lat, lon


############################################################
# SAR → LatLon
############################################################

def sar_to_latlon_grid(geo,height=0):

    rdr=Rdr2Geo(geo)

    lat=np.zeros((geo.na,geo.nr), dtype=np.float32)
    lon=np.zeros((geo.na,geo.nr), dtype=np.float32)

    for a in range(geo.na):
        if a % 100 == 0:
            print(f"处理第 {a}/{geo.na} 行")
        
        for r in range(geo.nr):
            try:
                xyz, shadow_mask, layover_mask = rdr.solve(r, a)
                la,lo,_=xyz_to_llh(*xyz)
                lat[a,r]=la
                lon[a,r]=lo
            except Exception as e:
                lat[a,r] = np.nan
                lon[a,r] = np.nan

    return lat,lon


############################################################
# SAR Geometry Error Correction
############################################################

from scipy.signal import fftconvolve
from scipy.ndimage import affine_transform


def estimate_shift_phase_correlation(img1, img2):

    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)

    cross_power = f1 * np.conj(f2)
    cross_power /= np.abs(cross_power) + 1e-12

    corr = np.fft.ifft2(cross_power)

    corr = np.abs(corr)

    max_idx = np.unravel_index(np.argmax(corr), corr.shape)

    shift_y = max_idx[0]
    shift_x = max_idx[1]

    if shift_y > img1.shape[0]//2:
        shift_y -= img1.shape[0]

    if shift_x > img1.shape[1]//2:
        shift_x -= img1.shape[1]

    return shift_x, shift_y


############################################################
# 仿射变换估计
############################################################

def estimate_affine(sim_sar, real_sar):

    from skimage.feature import ORB
    from skimage.feature import match_descriptors
    from skimage.transform import AffineTransform
    from skimage.measure import ransac

    detector = ORB(n_keypoints=2000)

    detector.detect_and_extract(sim_sar)
    keypoints1 = detector.keypoints
    descriptors1 = detector.descriptors

    detector.detect_and_extract(real_sar)
    keypoints2 = detector.keypoints
    descriptors2 = detector.descriptors

    matches = match_descriptors(descriptors1, descriptors2)

    src = keypoints1[matches[:,0]][:,::-1]
    dst = keypoints2[matches[:,1]][:,::-1]

    model_robust, inliers = ransac(
        (src, dst),
        AffineTransform,
        min_samples=3,
        residual_threshold=2,
        max_trials=100
    )

    return model_robust


############################################################
# 应用仿射校正
############################################################

def apply_affine(image, transform, output_shape):

    # 只使用偏移量，不使用旋转/缩放矩阵
    # matrix = transform.params[:2,:2]
    offset = transform.params[:2,2]

    corrected = affine_transform(
        image,
        np.eye(2),  # 单位矩阵，不做旋转/缩放
        # matrix,
        offset=offset,
        output_shape=output_shape
    )

    return corrected


############################################################
# 自动校正流程
############################################################

def update_geometry(geo, shift_x, shift_y):
    """更新几何模型参数"""
    # range bias
    geo.near_range += shift_x * geo.range_spacing

    # azimuth bias
    geo.t0 += shift_y / geo.prf

    print("Geometry updated:")
    print("near_range:", geo.near_range)
    print("t0:", geo.t0)

def correct_simulated_sar(sim_sar, real_sar):

    # 确保输入是幅度图像（处理复数类型）
    if np.iscomplexobj(sim_sar):
        sim_sar = np.abs(sim_sar)
    if np.iscomplexobj(real_sar):
        real_sar = np.abs(real_sar)

    print("Step1: 估计 range / azimuth shift")

    shift_x, shift_y = estimate_shift_phase_correlation(sim_sar, real_sar)

    print("Estimated shift:", shift_x, shift_y)

    sim_shifted = np.roll(sim_sar, shift_y, axis=0)
    sim_shifted = np.roll(sim_shifted, shift_x, axis=1)

    print("Step2: 估计 affine 几何误差")

    transform = estimate_affine(sim_shifted, real_sar)

    print("Affine matrix:")
    print(transform.params)

    print("Step3: 应用几何校正（只使用了offset）")

    corrected = apply_affine(sim_shifted, transform, sim_sar.shape)

    return corrected, transform, shift_x, shift_y

############################################################
# 保存结果
############################################################
def save_results(sar_dem, slc, lat_grid, lon_grid, output_prefix):
    """保存处理结果"""
    try:
        # 保存SAR DEM
        with rasterio.open(
            f"{output_prefix}_sar_dem.tif", 'w',
            driver='GTiff',
            height=sar_dem.shape[0],
            width=sar_dem.shape[1],
            count=1,
            dtype=sar_dem.dtype,
            crs='EPSG:4326',
            transform=rasterio.transform.from_origin(0, 0, 1, 1)
        ) as dst:
            dst.write(sar_dem, 1)
        print(f"保存SAR DEM到 {output_prefix}_sar_dem.tif")
    
        # 保存SLC（实部和虚部分开保存）
        with rasterio.open(
            f"{output_prefix}_slc_real.tif", 'w',
            driver='GTiff',
            height=slc.shape[0],
            width=slc.shape[1],
            count=1,
            dtype=np.float32,
            crs='EPSG:4326',
            transform=rasterio.transform.from_origin(0, 0, 1, 1)
        ) as dst:
            dst.write(np.real(slc).astype(np.float32), 1)
        
        with rasterio.open(
            f"{output_prefix}_slc_imag.tif", 'w',
            driver='GTiff',
            height=slc.shape[0],
            width=slc.shape[1],
            count=1,
            dtype=np.float32,
            crs='EPSG:4326',
            transform=rasterio.transform.from_origin(0, 0, 1, 1)
        ) as dst:
            dst.write(np.imag(slc).astype(np.float32), 1)
        print(f"保存SLC到 {output_prefix}_slc_real.tif 和 {output_prefix}_slc_imag.tif")
    
        # 保存地理坐标网格（可选）
        if lat_grid is not None and lon_grid is not None:
            with rasterio.open(
                f"{output_prefix}_lat_grid.tif", 'w',
                driver='GTiff',
                height=lat_grid.shape[0],
                width=lat_grid.shape[1],
                count=1,
                dtype=np.float32,
                crs='EPSG:4326',
                transform=rasterio.transform.from_origin(0, 0, 1, 1)
            ) as dst:
                dst.write(lat_grid.astype(np.float32), 1)
            
            with rasterio.open(
                f"{output_prefix}_lon_grid.tif", 'w',
                driver='GTiff',
                height=lon_grid.shape[0],
                width=lon_grid.shape[1],
                count=1,
                dtype=np.float32,
                crs='EPSG:4326',
                transform=rasterio.transform.from_origin(0, 0, 1, 1)
            ) as dst:
                dst.write(lon_grid.astype(np.float32), 1)
            print(f"保存地理坐标网格到 {output_prefix}_lat_grid.tif 和 {output_prefix}_lon_grid.tif")
        else:
            print("跳过保存地理坐标网格（lat_grid/lon_grid 为 None）")
    except Exception as e:
        print(f"保存结果失败: {e}")
        raise


def run_with_correction(yaml_file, dem_file, real_sar, output_prefix="corrected_output"):
    """带几何误差校正的完整处理流程"""
    print("=== 几何误差校正流程 ===")
    start_time = time.time()

    # 第一步：读取配置和DEM
    print("1. 读取配置和DEM")
    yaml_data = yaml.safe_load(open(yaml_file))
    geo = Geo2Rdr(yaml_data)
    lat, lon, h = load_dem(dem_file)
    print(f"DEM尺寸: {h.shape[0]}x{h.shape[1]}, 总点数: {h.size}")
    
    # 第二步：初始模拟
    print("\n2. 初始模拟SAR")
    sar_dem, sim_slc = dem_to_sar_products_streaming(
        geo,
        lat,
        lon,
        h,
        block_size=100000,
        parallel=True,
        yaml_file=yaml_file,
        geo_overrides={"near_range": geo.near_range, "t0": geo.t0},
        use_shared_memory=True,
    )
    sim_amp = np.abs(sim_slc)
    
    # 第三步：估计几何误差
    print("\n3. 估计几何误差")
    corrected, transform, shift_x, shift_y = correct_simulated_sar(sim_amp, real_sar)
    
    # 第四步：更新几何模型
    print("\n4. 更新几何模型")
    update_geometry(geo, shift_x, shift_y)
    
    # 第五步：重新模拟
    print("\n5. 重新模拟SAR")
    sar_dem_corrected, sim_slc_corrected = dem_to_sar_products_streaming(
        geo,
        lat,
        lon,
        h,
        block_size=100000,
        parallel=True,
        yaml_file=yaml_file,
        geo_overrides={"near_range": geo.near_range, "t0": geo.t0},
        use_shared_memory=True,
    )
    
    # 第六步：生成地理坐标网格（默认用 yaml 快速法，避免大图内存爆炸）
    print("\n6. 生成高精度地理坐标网格")
    # 对大图建议改为二进制分块写出：lat/lon grid 会非常大
    write_latlon_grid_bin(
        geo,
        lat,
        lon,
        h,
        output_prefix,
        method="yaml",
        output_step=1,
        sparse_step=50,
        chunk_rows=256,
    )
    lat_grid = lon_grid = None
    
    # 第七步：保存结果
    print("\n7. 保存结果")
    save_results(sar_dem_corrected, sim_slc_corrected, lat_grid, lon_grid, output_prefix)
    
    # 保存校正后的幅度图像
    with rasterio.open(
        f"{output_prefix}_corrected_amp.tif", 'w',
        driver='GTiff',
        height=corrected.shape[0],
        width=corrected.shape[1],
        count=1,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=rasterio.transform.from_origin(0, 0, 1, 1)
    ) as dst:
        dst.write(sim_amp.astype(np.float32), 1)
    print(f"保存校正后的幅度图像到 {output_prefix}_corrected_amp.tif")
    
    print("\n=== 几何误差校正流程完成 ===")
    total_time = time.time() - start_time
    print(f"总处理时间: {total_time:.2f}秒")
    return sar_dem_corrected, sim_slc_corrected, corrected, transform



# 地理校正使用方法
#import rasterio
#with rasterio.open("real_sar.tif") as ds:
#    real_sar = ds.read(1)
#corrected = run_with_correction(
#    "orbit.yaml",
#    "dem.tif",
#    real_sar
#)

############################################################
# 主程序
############################################################

def run(
    yaml_file,
    dem_file,
    output_prefix,
    block_size=100000,
    parallel=True,
    do_geocode=True,
    geocode_method="yaml",
    geocode_step=1,
    sparse_step=50,
    latlon_format="tif",
    latlon_chunk_rows=256,
    dem_crop=False,
    dem_margin_km=10.0,
):

    start_time = time.time()
    
    print("读取yaml")
    try:
        yaml_data=yaml.safe_load(open(yaml_file))
        geo=Geo2Rdr(yaml_data)
    except Exception as e:
        print(f"读取YAML文件失败: {e}")
        raise

    print("读取DEM")
    crop_bounds_ll = None
    if dem_crop:
        crop_bounds_ll = _yaml_corner_bounds_ll(geo, margin_km=dem_margin_km)
        if crop_bounds_ll is None:
            print("DEM裁剪：YAML 缺少角点信息，回退为全幅 DEM 处理")
        else:
            lat_min, lon_min, lat_max, lon_max = crop_bounds_ll
            print(
                "DEM裁剪：使用 YAML 四角点外包矩形 + 缓冲读取 DEM\n"
                f"  bounds(lat/lon, deg): lat[{lat_min:.6f},{lat_max:.6f}] lon[{lon_min:.6f},{lon_max:.6f}]\n"
                f"  margin_km: {dem_margin_km}"
            )
    lat,lon,h = load_dem_windowed(dem_file, crop_bounds_ll=crop_bounds_ll)
    print(f"DEM尺寸: {h.shape[0]}x{h.shape[1]}, 总点数: {h.size}")

    print("DEM → SAR坐标")
    dem2sar_start = time.time()
    sar_dem, slc = dem_to_sar_products_streaming(
        geo,
        lat,
        lon,
        h,
        block_size=block_size,
        parallel=parallel,
        yaml_file=yaml_file,
        geo_overrides={"near_range": geo.near_range, "t0": geo.t0},
        use_shared_memory=True,
    )
    dem2sar_end = time.time()
    print(f"DEM → SAR坐标处理时间: {dem2sar_end - dem2sar_start:.2f}秒")
    
    # 统计有效点
    valid_points = np.sum(~np.isnan(sar_dem))
    print(f"SAR DEM有效点数: {valid_points}/{geo.na*geo.nr}")

    lat_grid = lon_grid = None
    if do_geocode:
        print("SAR → 地理坐标")
        geocode_start = time.time()
        if latlon_format == "bin":
            write_latlon_grid_bin(
                geo,
                lat,
                lon,
                h,
                output_prefix,
                method=geocode_method,
                output_step=geocode_step,
                sparse_step=sparse_step,
                chunk_rows=latlon_chunk_rows,
            )
            # bin 输出内部会写 meta，但为统一 downstream，这里也确保 meta 的 files.lat/files.lon 字段存在
            # （兼容旧键 lat_bin/lon_bin 的读取逻辑）
            lat_grid = lon_grid = None
        elif latlon_format == "tif":
            # geocode_step: 输出网格步长（像素），>1 会显著降低内存/IO
            # sparse_step: 仅对 method=sparse 有效，控制点间隔（像素）
            lat_grid, lon_grid = sar_to_latlon_grid_high_precision(
                geo,
                None,
                None,
                lat,
                lon,
                h,
                method=geocode_method,
                output_step=geocode_step,
                sparse_step=sparse_step,
            )
        elif latlon_format == "none":
            lat_grid = lon_grid = None
        else:
            raise ValueError(f"Unsupported latlon_format: {latlon_format}")
        geocode_end = time.time()
        print(f"SAR → 地理坐标处理时间: {geocode_end - geocode_start:.2f}秒")
    else:
        print("跳过 SAR → 地理坐标（do_geocode=False）")

    print("保存结果")
    save_start = time.time()
    save_results(sar_dem, slc, lat_grid, lon_grid, output_prefix)
    # 若输出为 tif，并且生成了 lat/lon 网格，则额外写一份 meta，方便后处理统一用 meta 读取
    if do_geocode and latlon_format == "tif" and lat_grid is not None and lon_grid is not None:
        write_latlon_grid_meta_yaml(
            geo,
            output_prefix,
            f"{output_prefix}_lat_grid.tif",
            f"{output_prefix}_lon_grid.tif",
            geocode_method,
            geocode_step,
            sparse_step,
        )
    save_end = time.time()
    print(f"结果保存时间: {save_end - save_start:.2f}秒")

    total_time = time.time() - start_time
    print(f"总处理时间: {total_time:.2f}秒")
    print("完成")

    return sar_dem,slc,lat_grid,lon_grid


############################################################

def self_test_rdr2geo(yaml_file, n=200, seed=0, mode="pixel"):
    """
    一致性自检。
    - mode=pixel（默认）：随机抽取图像内像素 (r,a)，做 rdr2geo->geo2rdr 闭环，检查像素误差与位置误差。
    - mode=geo：随机抽取地面点(lat/lon/h=0)，做 geo2rdr->rdr2geo 闭环（若点映射到图像外，误差可能被初值夹紧影响）。
    """
    np.random.seed(seed)
    yaml_data = yaml.safe_load(open(yaml_file))
    geo = Geo2Rdr(yaml_data)
    rdr = Rdr2Geo(geo)

    if mode == "pixel":
        r0 = np.random.uniform(0.0, max(1.0, geo.nr - 1.0), size=n)
        a0 = np.random.uniform(0.0, max(1.0, geo.na - 1.0), size=n)
        P, sh, lo = rdr.solve(r0, a0, max_iter=40, tol=1e-3)
        r1, a1, R1, t1, ld1 = geo.geo2rdr(P)
        pix_err = np.sqrt((r1 - r0) ** 2 + (a1 - a0) ** 2)
        # 位置误差用 “回算像素再正算” 不好直接定义，这里给出以像素为主的误差
        print("Self-test rdr2geo consistency (mode=pixel):")
        print(f"  samples: {n}")
        print(f"  pix_err(px): mean={pix_err.mean():.4f}  p95={np.percentile(pix_err,95):.4f}  max={pix_err.max():.4f}")
        return pix_err

    if mode == "geo":
        if geo._corner_latlon is None:
            raise ValueError("YAML 缺少角点经纬度，无法生成测试点范围")

        lat_tl, lon_tl, lat_tr, lon_tr, lat_bl, lon_bl, lat_br, lon_br = geo._corner_latlon
        lat_min = min(lat_tl, lat_tr, lat_bl, lat_br)
        lat_max = max(lat_tl, lat_tr, lat_bl, lat_br)
        lon_min = min(lon_tl, lon_tr, lon_bl, lon_br)
        lon_max = max(lon_tl, lon_tr, lon_bl, lon_br)

        lat0 = lat_min + (lat_max - lat_min) * np.random.rand(n)
        lon0 = lon_min + (lon_max - lon_min) * np.random.rand(n)
        h0 = np.zeros(n, dtype=np.float64)
        P0 = llh_to_xyz(lat0, lon0, h0)

        r, a, R, t, look_dir = geo.geo2rdr(P0)
        P1, sh, lo = rdr.solve(r, a, max_iter=40, tol=1e-3)
        err = np.linalg.norm(P1 - P0, axis=1)
        print("Self-test rdr2geo consistency (mode=geo):")
        print(f"  samples: {n}")
        print(f"  err(m): mean={err.mean():.3f}  p95={np.percentile(err,95):.3f}  max={err.max():.3f}")
        return err

    raise ValueError("mode must be pixel or geo")


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="DEM 生成模拟 SAR（含 geo2rdr / rdr2geo）与几何校正工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "yaml_file",
        help=(
            "卫星/成像参数 YAML 文件路径。\n"
            "要求至少包含：orbit_data、radar_parameters、metadata、image_parameters。\n"
            "建议同时包含：geolocation_grid 或 corner_coordinates（用于 geocode 初值/快速网格）。"
        ),
    )
    parser.add_argument(
        "dem_file",
        nargs="?",
        default=None,
        help="DEM 文件路径（GeoTIFF 等 rasterio 可读格式）。除非使用 --self-test，否则必填。",
    )
    parser.add_argument(
        "output_prefix",
        nargs="?",
        default=None,
        help=(
            "输出文件前缀。\n"
            "会生成：*_sar_dem.tif、*_slc_real.tif、*_slc_imag.tif，以及可选的 *_lat_grid.tif/*_lon_grid.tif。"
        ),
    )
    parser.add_argument(
        "--real-sar",
        help=(
            "真实 SAR 幅度图（或 SLC 幅度）文件路径，用于几何误差校正。\n"
            "脚本会先模拟，再用相位相关/特征匹配估计 shift+仿射偏移，并更新 near_range/t0 重新模拟。"
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=100000,
        help=(
            "DEM→SAR 处理的分块点数（默认 100000）。\n"
            "更大：更快但占用更多内存；更小：更稳但慢。\n"
            "设置为 0 表示不分块（一次性处理全部 DEM 点，可能吃内存）。"
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "启用多进程并行加速 DEM→SAR 分块处理。\n"
            "适合 DEM 点数很大（例如 3601x7201）。\n"
            "注意：多进程会增加内存占用。"
        ),
    )
    parser.add_argument(
        "--dem-crop",
        action="store_true",
        help=(
            "按 YAML 四角点经纬度对 DEM 先裁剪再处理，以减少数据量（默认关闭）。\n"
            "适用：DEM 远大于 SAR 覆盖范围时（例如全球/大区域 DEM）。\n"
            "注意：角点经纬度通常是低精度，务必配合 --dem-margin-km 留足缓冲。\n"
            "当 YAML 不包含 geolocation_grid/corner_coordinates 时，会自动回退为全幅 DEM 处理。"
        ),
    )
    parser.add_argument(
        "--dem-margin-km",
        type=float,
        default=10.0,
        help=(
            "仅对 --dem-crop 生效：裁剪缓冲距离（公里，默认 10km）。\n"
            "值越大越安全（不漏覆盖），但裁剪后 DEM 更大、速度更慢。\n"
            "建议：5~20km 起步；若后续做几何校正/覆盖不确定，可适当增大。"
        ),
    )
    parser.add_argument(
        "--skip-geocode",
        action="store_true",
        help=(
            "跳过 SAR→地理坐标网格（lat/lon grid）生成。\n"
            "对超大 SAR（例如 15000x15000）强烈建议开启，否则 lat/lon 两张网格本身就可能占用 >1.8GB 内存。"
        ),
    )
    parser.add_argument(
        "--geocode-method",
        default="yaml",
        help=(
            "SAR→地理坐标网格生成方法（默认 yaml）：\n"
            "- yaml：仅用 YAML 四角点做双线性插值，速度最快，和元数据一致；不做严格轨道反解。\n"
            "- sparse：按稀疏控制点做 rdr2geo 严格反解，再做 2D 插值补全；更准确但更慢。\n"
            "建议：大图先用 yaml；需要更准确几何时再用 sparse。"
        ),
    )
    parser.add_argument(
        "--geocode-step",
        type=int,
        default=1,
        help=(
            "输出 lat/lon 网格的降采样步长（像素，>=1，默认 1）。\n"
            "例如 10 表示每隔 10 个像素输出一个点（网格尺寸约缩小到 1/100）。\n"
            "对 15000x15000 建议从 10 或 20 开始。"
        ),
    )
    parser.add_argument(
        "--sparse-step",
        type=int,
        default=50,
        help=(
            "仅对 --geocode-method sparse 生效：控制点间隔（像素，>=1，默认 50）。\n"
            "值越大：控制点越少，速度更快但插值更粗；值越小：更准但更慢。\n"
            "建议范围：30~200（视地形起伏和需求调整）。"
        ),
    )
    parser.add_argument(
        "--latlon-format",
        default="tif",
        help=(
            "lat/lon 网格输出格式（默认 tif）：\n"
            "- tif：输出 GeoTIFF（方便查看，但大图很占内存/磁盘，可能非常慢）。\n"
            "- bin：输出二进制 raw float32 小端，并分块写盘（推荐 15000x15000 级别）。\n"
            "- none：不输出 lat/lon 网格。\n"
            "注意：bin 输出会额外写一个 *_latlon_grid_meta.yaml 记录 shape/dtype/索引。"
        ),
    )
    parser.add_argument(
        "--latlon-chunk-rows",
        type=int,
        default=256,
        help=(
            "仅对 --latlon-format bin 生效：分块写盘的行块大小（输出网格的行数）。\n"
            "更大：更快但占用更多内存；更小：更省内存但可能更慢。\n"
            "建议范围：64~1024。"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "运行 geo2rdr<->rdr2geo 一致性自检并退出。\n"
            "做法：随机生成地面点 -> geo2rdr 得到 (r,a) -> rdr2geo 回算，统计位置误差（米）。\n"
            "用于快速判断 rdr2geo 是否明显跑偏（不需要 DEM）。"
        ),
    )
    parser.add_argument(
        "--self-test-n",
        type=int,
        default=200,
        help="--self-test 的随机样本数（默认 200）。",
    )
    parser.add_argument(
        "--self-test-mode",
        default="pixel",
        help=(
            "自检模式（默认 pixel）：\n"
            "- pixel：随机图像内像素(r,a)，检验 rdr2geo->geo2rdr 的像素闭环误差（推荐）。\n"
            "- geo：随机地面点(lat/lon)，检验 geo2rdr->rdr2geo 的位置误差（可能受域外点影响）。"
        ),
    )
    
    args = parser.parse_args()
    
    try:
        if args.self_test:
            self_test_rdr2geo(args.yaml_file, n=args.self_test_n, mode=args.self_test_mode)
            exit(0)

        if args.dem_file is None or args.output_prefix is None:
            raise ValueError("缺少参数：需要 dem_file 和 output_prefix（除非使用 --self-test）")

        if args.real_sar:
            # 使用几何误差校正流程
            print(f"使用几何误差校正流程，真实SAR图像: {args.real_sar}")
            import rasterio
            with rasterio.open(args.real_sar) as ds:
                real_sar = ds.read(1)
            sar_dem_corrected, sim_slc_corrected, corrected, transform = run_with_correction(
                args.yaml_file, args.dem_file, real_sar, args.output_prefix
            )
        else:
            # 使用常规流程
            sar_dem,slc,lat,lon = run(
                args.yaml_file,
                args.dem_file,
                args.output_prefix,
                args.block_size,
                args.parallel,
                do_geocode=(not args.skip_geocode),
                geocode_method=args.geocode_method,
                geocode_step=args.geocode_step,
                sparse_step=args.sparse_step,
                latlon_format=args.latlon_format,
                latlon_chunk_rows=args.latlon_chunk_rows,
                dem_crop=args.dem_crop,
                dem_margin_km=args.dem_margin_km,
            )
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    #输入
    #常规流程: python dem2sar_full.py orbit.yaml dem.tif output_prefix
    #带几何校正: python dem2sar_full.py orbit.yaml dem.tif output_prefix --real-sar master.tif
    #输出：
    #sar_dem      SAR坐标DEM
    #slc          模拟SAR复数图像
    #lat_grid     SAR像素纬度
    #lon_grid     SAR像素经度
    #corrected    校正后的SAR图像（带几何校正时）
