#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线计算工具
功能：
- 读取master和slave的YAML文件
- 计算每个点的基线向量
- 计算垂直基线长度和水平基线长度
- 生成与master.vrt大小一致的baseline.hdf文件
"""

import numpy as np
import h5py
import yaml
import argparse
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
import os
from osgeo import gdal

# -------------------------------
#  坐标转换函数
# -------------------------------
def llh_to_xyz(lat, lon, h):
    """LLH 到 ECEF 坐标转换"""
    # WGS84
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2-f)
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N*(1-e2) + h) * np.sin(lat)
    return np.stack([X,Y,Z], axis=-1)

# -------------------------------
#  轨道插值
# -------------------------------
class OrbitInterpolator:
    """轨道插值类"""
    def __init__(self, orbit_data, method='HERMITE'):
        """
        初始化轨道插值器
        :param orbit_data: 轨道数据，格式为 [time, x, y, z, vx, vy, vz]
        :param method: 插值方法，'CUBIC' 或 'HERMITE'
        """
        try:
            # 验证轨道数据格式
            if not isinstance(orbit_data, np.ndarray):
                orbit_data = np.array(orbit_data)
            
            if orbit_data.shape[1] != 7:
                raise ValueError("轨道数据格式错误，应为 [time, x, y, z, vx, vy, vz]")
            
            self.times = orbit_data[:, 0]
            self.positions = orbit_data[:, 1:4]
            self.velocities = orbit_data[:, 4:7]
            self.method = method.upper()
            
            # 检查时间是否递增
            if not np.all(np.diff(self.times) > 0):
                raise ValueError("轨道数据时间必须递增")
            
            if self.method == 'CUBIC':
                self.pos_interpolators = [interp1d(self.times, self.positions[:, i], kind='cubic') for i in range(3)]
                self.vel_interpolators = [interp1d(self.times, self.velocities[:, i], kind='cubic') for i in range(3)]
            elif self.method == 'HERMITE':
                # 预处理 Hermite 插值所需的参数
                self._precompute_hermite()
            else:
                raise ValueError(f"不支持的插值方法: {method}")
        except Exception as e:
            raise ValueError(f"轨道插值器初始化失败: {e}")
    
    def _precompute_hermite(self):
        """预处理 Hermite 插值所需的参数"""
        self.n = len(self.times)
        self.h = np.diff(self.times)
        self.alpha = np.zeros(self.n)
        self.beta = np.zeros(self.n)
        self.c = np.zeros((self.n, 3))
        self.d = np.zeros((self.n, 3))
        
        for i in range(3):
            y = self.positions[:, i]
            yp = self.velocities[:, i]
            
            # 计算 alpha 和 beta
            if self.n > 2:
                self.alpha[1:-1] = 3 * (y[2:] - y[1:-1]) / self.h[1:] - 3 * (y[1:-1] - y[:-2]) / self.h[:-1]
            
            # 边界条件：使用速度作为边界导数
            if self.n > 1:
                self.alpha[0] = 3 * (y[1] - y[0]) / self.h[0] - yp[0]
                self.alpha[-1] = yp[-1] - 3 * (y[-1] - y[-2]) / self.h[-1]
            else:
                # 只有一个点时的处理
                self.alpha[0] = yp[0]
            
            # 解三对角方程组
            if self.n > 1:
                self.beta[0] = 2 * self.h[0]
                for j in range(1, self.n-1):
                    if self.beta[j-1] != 0:
                        self.beta[j] = 2 * (self.h[j-1] + self.h[j]) - self.h[j-1]**2 / self.beta[j-1]
                        self.alpha[j] = self.alpha[j] - self.h[j-1] * self.alpha[j-1] / self.beta[j-1]
                    else:
                        self.beta[j] = 1.0
                        self.alpha[j] = 0.0
                
                # 回代求解
                if self.beta[-1] != 0:
                    self.c[-1, i] = self.alpha[-1] / self.beta[-1]
                else:
                    self.c[-1, i] = 0.0
                
                for j in range(self.n-2, -1, -1):
                    if self.beta[j] != 0:
                        self.c[j, i] = (self.alpha[j] - self.h[j] * self.c[j+1, i]) / self.beta[j]
                    else:
                        self.c[j, i] = 0.0
                
                # 计算 d
                for j in range(self.n-1):
                    if self.h[j] != 0:
                        self.d[j, i] = (self.c[j+1, i] - self.c[j, i]) / (3 * self.h[j])
                    else:
                        self.d[j, i] = 0.0
    
    def position(self, t):
        """计算给定时间的卫星位置"""
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 0:
            t = t.reshape(1)

        if self.method == 'CUBIC':
            # scipy 的 interp1d 本身可向量化；这里不做逐点 try/except。
            out = np.empty((t.size, 3), dtype=np.float64)
            out[:, 0] = self.pos_interpolators[0](t)
            out[:, 1] = self.pos_interpolators[1](t)
            out[:, 2] = self.pos_interpolators[2](t)
            return out

        # HERMITE：纯 numpy 向量化评估（无 Python 循环/缓存）
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.h) - 1)
        dt = t - self.times[idx]

        y = self.positions[idx]
        yp = self.velocities[idx]
        c = self.c[idx]
        d = self.d[idx]
        out = y + yp * dt[:, None] + c * (dt[:, None] ** 2) + d * (dt[:, None] ** 3)

        # 极少数情况下会出现 NaN/Inf（例如轨道数据异常）；用端点位置兜底。
        bad = np.isnan(out).any(axis=1) | np.isinf(out).any(axis=1)
        if np.any(bad):
            out[bad] = self.positions[idx[bad]]
        return out

# -------------------------------
#  轨道模拟器
# -------------------------------
class OrbitSimulator:
    def __init__(self, yaml_cfg):
        """
        初始化轨道模拟器
        :param yaml_cfg: YAML 配置
        """
        self.tmin = yaml_cfg.get('tmin', 0)
        self.tmax = yaml_cfg.get('tmax', 1000)
        
        # 检查是否有真实轨道数据
        if 'orbit_data' in yaml_cfg:
            orbit_data = np.array(yaml_cfg['orbit_data'])
            method = yaml_cfg.get('orbit_interpolation', 'HERMITE')
            self.interpolator = OrbitInterpolator(orbit_data, method)
        else:
            # 使用默认线性轨道
            self.interpolator = None
    
    def position(self, t):
        """计算卫星位置"""
        if self.interpolator:
            return self.interpolator.position(t)
        else:
            # 简单线性轨道示例
            t = np.atleast_1d(t)
            return np.stack([t*10, t*0+7000000, t*0+500000], axis=-1)
    
    def velocity(self, t):
        """计算卫星速度"""
        if hasattr(self.interpolator, 'vel_interpolators'):
            t = np.asarray(t, dtype=np.float64)
            if t.ndim == 0:
                t = t.reshape(1)
            out = np.empty((t.size, 3), dtype=np.float64)
            out[:, 0] = self.interpolator.vel_interpolators[0](t)
            out[:, 1] = self.interpolator.vel_interpolators[1](t)
            out[:, 2] = self.interpolator.vel_interpolators[2](t)
            return out
        else:
            # 简单线性速度示例
            t = np.atleast_1d(t)
            return np.stack([t*0+10, t*0, t*0], axis=-1)

# -------------------------------
#  读取YAML配置
# -------------------------------
def read_yaml_config(yaml_file):
    """
    读取YAML配置文件
    :param yaml_file: YAML文件路径
    :return: 配置字典
    """
    with open(yaml_file, 'r') as f:
        ycfg = yaml.safe_load(f)
    if ycfg is None:
        raise ValueError(f"YAML 文件为空或仅包含 null: {yaml_file}")
    if not isinstance(ycfg, dict):
        raise ValueError(f"YAML 顶层必须是映射(dict)，当前为 {type(ycfg).__name__}: {yaml_file}")
    
    # 统一的 ISO8601 时间解析（返回 Unix epoch seconds）
    def _parse_time_iso8601(s):
        import datetime
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        ss = str(s).strip()
        # 兼容 'Z'
        if ss.endswith('Z'):
            ss = ss[:-1] + '+00:00'
        return datetime.datetime.fromisoformat(ss).timestamp()
    
    # 从 YAML 解析参数
    meta = ycfg.get('metadata', {}) or {}
    t0_str = meta.get('first_line_sensing_time', None)
    t1_str = meta.get('last_line_sensing_time', None)
    t0_meta_epoch = _parse_time_iso8601(t0_str) if t0_str else None
    t1_meta_epoch = _parse_time_iso8601(t1_str) if t1_str else None

    # 轨道数据（支持多种 YAML 结构）
    orbit_points = None
    if isinstance(ycfg.get('orbit_data', None), dict) and ('orbit_points' in ycfg['orbit_data']):
        orbit_points = ycfg['orbit_data'].get('orbit_points')
    elif isinstance(ycfg.get('orbit', None), list):
        orbit_points = ycfg.get('orbit')
    elif isinstance(ycfg.get('orbit_points', None), list):
        orbit_points = ycfg.get('orbit_points')

    orbit_rows = []
    orbit_time_is_epoch = False
    if isinstance(orbit_points, list) and orbit_points:
        for p in orbit_points:
            try:
                tt = p.get('time')
                t = _parse_time_iso8601(tt)
                # 粗略判断是否为 epoch（秒级 1e9 量级）
                if t is not None and t > 1e9:
                    orbit_time_is_epoch = True
                pos = p.get('position', {})
                vel = p.get('velocity', {})
                orbit_rows.append([
                    float(t),
                    float(pos.get('x')), float(pos.get('y')), float(pos.get('z')),
                    float(vel.get('vx')), float(vel.get('vy')), float(vel.get('vz')),
                ])
            except Exception:
                continue
    elif isinstance(ycfg.get('orbit_data', None), (list, tuple)):
        # 允许直接给 Nx7 数组
        arr = np.asarray(ycfg.get('orbit_data'), dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 7:
            orbit_rows = arr.tolist()

    orbit_arr = np.asarray(orbit_rows, dtype=np.float64)
    if orbit_arr.size == 0:
        raise ValueError("YAML 中未解析到有效 orbit_points / orbit_data（需要 time/position/velocity）")
    # 按时间排序
    order = np.argsort(orbit_arr[:, 0])
    orbit_arr = orbit_arr[order]

    config = {
        'orbit_data': orbit_arr,
        'tmin': float(orbit_arr[0, 0]),
        'tmax': float(orbit_arr[-1, 0]),
        't0': float(t0_meta_epoch if t0_meta_epoch is not None else orbit_arr[0, 0]),
        't1': float(t1_meta_epoch if t1_meta_epoch is not None else orbit_arr[-1, 0]),
        'metadata': meta,
        'corner_coordinates': ycfg.get('corner_coordinates', {})
    }
    
    # 系统参数
    radar_params = ycfg.get('radar_parameters', {})
    config['prf'] = radar_params.get('prf', ycfg.get('prf', 1500.0))
    config['near_range'] = radar_params.get('near_range', ycfg.get('near_range', 800000.0))
    # 计算range_spacing
    range_spacing = radar_params.get('range_spacing')
    if range_spacing is None:
        range_spacing = radar_params.get('range_pixel_spacing')
    if range_spacing is None:
        range_spacing = ycfg.get('range_spacing')
    if range_spacing is None:
        range_spacing = ycfg.get('range_pixel_spacing')
    if range_spacing is None:
        range_spacing = 6.25
    config['range_spacing'] = range_spacing

    config['wavelength'] = radar_params.get('wavelength', ycfg.get('wavelength', 0.056))
    
    return config


def _resample_to_shape(arr, target_shape, order=1):
    """将 2D 数组重采样到目标形状。"""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape}")
    if tuple(arr.shape) == tuple(target_shape):
        return arr.astype(np.float64, copy=False)
    zf = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zf, order=order).astype(np.float64, copy=False)


def _fill_nan_nearest_2d(arr):
    """用最近邻填充 2D 数组中的 NaN。"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"_fill_nan_nearest_2d expects 2D, got {arr.shape}")
    mask = np.isfinite(arr)
    if np.all(mask):
        return arr
    if not np.any(mask):
        raise ValueError("all values are NaN")
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~mask, return_indices=True)
    return arr[tuple(idx)]


def build_latlon_grid_from_corners(corners, shape):
    """
    根据四角点构建 SAR 网格下的 lat/lon（双线性）。
    需要角点键：top_left/top_right/bottom_left/bottom_right。
    """
    rows, cols = int(shape[0]), int(shape[1])
    required = ("top_left", "top_right", "bottom_left", "bottom_right")
    for k in required:
        if k not in corners or not isinstance(corners[k], dict):
            raise ValueError(f"corner_coordinates 缺少角点: {k}")
        if ("lat" not in corners[k]) or ("lon" not in corners[k]):
            raise ValueError(f"角点 {k} 缺少 lat/lon")

    lat_tl = float(corners["top_left"]["lat"])
    lon_tl = float(corners["top_left"]["lon"])
    lat_tr = float(corners["top_right"]["lat"])
    lon_tr = float(corners["top_right"]["lon"])
    lat_bl = float(corners["bottom_left"]["lat"])
    lon_bl = float(corners["bottom_left"]["lon"])
    lat_br = float(corners["bottom_right"]["lat"])
    lon_br = float(corners["bottom_right"]["lon"])

    u = np.linspace(0.0, 1.0, cols, dtype=np.float64)
    v = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    lat_top = lat_tl * (1.0 - u) + lat_tr * u
    lon_top = lon_tl * (1.0 - u) + lon_tr * u
    lat_bot = lat_bl * (1.0 - u) + lat_br * u
    lon_bot = lon_bl * (1.0 - u) + lon_br * u

    lat_grid = lat_top[None, :] * (1.0 - v)[:, None] + lat_bot[None, :] * v[:, None]
    lon_grid = lon_top[None, :] * (1.0 - v)[:, None] + lon_bot[None, :] * v[:, None]
    return lat_grid.astype(np.float64), lon_grid.astype(np.float64)


def load_ground_grid(master_config, vrt_shape, geosar_hdf=None):
    """
    加载用于 LOS 分解的地面点网格：
    优先 geosar HDF（sar_lat/sar_lon + sar_dem），否则回退到 YAML 角点双线性 + h=0。
    """
    rows, cols = int(vrt_shape[0]), int(vrt_shape[1])
    target_shape = (rows, cols)

    # 1) geosar HDF
    if geosar_hdf and os.path.exists(geosar_hdf):
        with h5py.File(geosar_hdf, 'r') as f:
            lat = None
            lon = None
            h = None

            for k in ("sar_lat", "lat_grid"):
                if k in f:
                    lat = np.asarray(f[k][:], dtype=np.float64)
                    break
            for k in ("sar_lon", "lon_grid"):
                if k in f:
                    lon = np.asarray(f[k][:], dtype=np.float64)
                    break
            for k in ("sar_dem", "sar_dem_raw"):
                if k in f:
                    h = np.asarray(f[k][:], dtype=np.float64)
                    break

        if (lat is not None) and (lon is not None):
            lat = _resample_to_shape(lat, target_shape, order=1)
            lon = _resample_to_shape(lon, target_shape, order=1)
            if h is None:
                h = np.zeros(target_shape, dtype=np.float64)
            else:
                h = _resample_to_shape(h, target_shape, order=1)
            # geosar 反推网格可能有大量空洞；为 LOS 分解需要全覆盖，这里做最近邻补齐。
            lat = _fill_nan_nearest_2d(lat)
            lon = _fill_nan_nearest_2d(lon)
            h = _fill_nan_nearest_2d(h)
            return lat, lon, h, "geosar_hdf"

    # 2) YAML 角点双线性 + 零高程
    corners = master_config.get("corner_coordinates", {}) or {}
    lat, lon = build_latlon_grid_from_corners(corners, target_shape)
    h = np.zeros(target_shape, dtype=np.float64)
    return lat, lon, h, "yaml_corners"


def _safe_unit(v):
    """向量按行归一化，零范数返回 0。"""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.zeros_like(v, dtype=np.float64)
    ok = n[:, 0] > 1e-12
    out[ok] = v[ok] / n[ok]
    return out

# -------------------------------
#  计算基线
# -------------------------------
def calculate_baseline(master_config, slave_config, vrt_shape, lat_grid, lon_grid, h_grid):
    """
    基于 LOS 的严格基线分解：
    - b_parallel_los = B · u_los
    - b_perp_los = sign * ||B - b_parallel_los * u_los||

    其中：
    - B = S_slave - S_master
    - u_los 为地面点->主星的 LOS 单位向量
    - sign 由三重积 sign((u_los × u_az) · B) 给出（u_az 为主星沿轨单位向量）
    """
    master_orbit = OrbitSimulator(master_config)
    slave_orbit = OrbitSimulator(slave_config)

    rows, cols = int(vrt_shape[0]), int(vrt_shape[1])
    N = rows * cols

    # 每行对应一个方位时间，距离向沿行内复用
    row_rel = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    rel = np.repeat(row_rel, cols)

    master_t0, master_t1 = float(master_config['t0']), float(master_config['t1'])
    slave_t0, slave_t1 = float(slave_config['t0']), float(slave_config['t1'])
    master_times = master_t0 + rel * (master_t1 - master_t0)
    slave_times = slave_t0 + rel * (slave_t1 - slave_t0)

    master_pos = master_orbit.position(master_times)
    slave_pos = slave_orbit.position(slave_times)
    master_vel = master_orbit.velocity(master_times)

    baseline_vec = slave_pos - master_pos

    # 地面点 ECEF
    lat_f = np.asarray(lat_grid, dtype=np.float64).reshape(-1)
    lon_f = np.asarray(lon_grid, dtype=np.float64).reshape(-1)
    h_f = np.asarray(h_grid, dtype=np.float64).reshape(-1)
    if not (lat_f.size == lon_f.size == h_f.size == N):
        raise ValueError(f"ground grid size mismatch: {lat_f.size},{lon_f.size},{h_f.size} vs {N}")

    ground_xyz = llh_to_xyz(lat_f, lon_f, h_f)
    valid_geo = np.isfinite(lat_f) & np.isfinite(lon_f) & np.isfinite(h_f)

    # LOS 单位向量（地面 -> 主星）
    los = master_pos - ground_xyz
    u_los = _safe_unit(los)
    u_az = _safe_unit(master_vel)

    b_parallel_los = np.sum(baseline_vec * u_los, axis=1)
    b_perp_vec = baseline_vec - b_parallel_los[:, None] * u_los
    b_perp_mag = np.linalg.norm(b_perp_vec, axis=1)

    # 有符号垂直基线（LOS 分解符号）
    sign_ref = np.sum(np.cross(u_los, u_az) * baseline_vec, axis=1)
    sign = np.sign(sign_ref)
    sign[sign == 0] = 1.0
    b_perp_los = sign * b_perp_mag

    # 无效地理点置 NaN
    bad = ~valid_geo
    b_parallel_los[bad] = np.nan
    b_perp_los[bad] = np.nan
    baseline_vec[bad] = np.nan

    baseline_vec = baseline_vec.reshape(rows, cols, 3).astype(np.float32, copy=False)
    b_perp_los = b_perp_los.reshape(rows, cols).astype(np.float32, copy=False)
    b_parallel_los = b_parallel_los.reshape(rows, cols).astype(np.float32, copy=False)
    return baseline_vec, b_perp_los, b_parallel_los

# -------------------------------
#  读取VRT文件尺寸
# -------------------------------
def get_vrt_shape(vrt_file):
    """
    读取VRT文件的尺寸
    :param vrt_file: VRT文件路径
    :return: (行数, 列数)
    """
    try:
        from osgeo import gdal
        ds = gdal.Open(vrt_file)
        if not ds:
            raise Exception(f"无法打开VRT文件: {vrt_file}")
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        return (rows, cols)
    except ImportError:
        print("GDAL库未安装，使用默认尺寸")
        # 默认返回一个合理的尺寸
        return (3645, 3136)  # 基于master.yaml中的image_parameters
    except Exception as e:
        print(f"读取VRT文件失败: {e}")
        # 默认返回一个合理的尺寸
        return (3645, 3136)  # 基于master.yaml中的image_parameters

# -------------------------------
#  主函数
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='计算SAR影像的基线')
    parser.add_argument('master', help='主影像名称（不带.yaml扩展名）')
    parser.add_argument('slave', help='从影像名称（不带.yaml扩展名）')
    parser.add_argument('--output', default='baseline.hdf', help='输出HDF文件路径')
    parser.add_argument('--vrt', default='master.vrt', help='参考VRT文件路径')
    parser.add_argument('--geosar-hdf', default=None, help='geosar.py 输出 HDF（优先读取 sar_lat/sar_lon/sar_dem）')
    
    args = parser.parse_args()
    
    # 构建文件路径
    master_yaml = f"{args.master}.yaml"
    slave_yaml = f"{args.slave}.yaml"
    vrt_file = args.vrt
    output_file = args.output
    
    # 检查文件是否存在
    if not os.path.exists(master_yaml):
        print(f"主影像YAML文件不存在: {master_yaml}")
        return
    if not os.path.exists(slave_yaml):
        print(f"从影像YAML文件不存在: {slave_yaml}")
        return
    if not os.path.exists(vrt_file):
        print(f"VRT文件不存在: {vrt_file}")
        return
    
    # 读取配置
    print("读取主影像配置...")
    master_config = read_yaml_config(master_yaml)
    print("读取从影像配置...")
    slave_config = read_yaml_config(slave_yaml)
    
    # 获取VRT尺寸
    print("读取VRT文件尺寸...")
    vrt_shape = get_vrt_shape(vrt_file)
    print(f"VRT尺寸: {vrt_shape}")

    # 读取地面网格（用于 LOS 分解）
    print("读取地面点网格（LOS 分解）...")
    lat_grid, lon_grid, h_grid, geo_src = load_ground_grid(master_config, vrt_shape, geosar_hdf=args.geosar_hdf)
    print(f"地面网格来源: {geo_src}")
    print(f"lat范围: [{np.nanmin(lat_grid):.6f}, {np.nanmax(lat_grid):.6f}]")
    print(f"lon范围: [{np.nanmin(lon_grid):.6f}, {np.nanmax(lon_grid):.6f}]")
    
    # 计算基线
    print("计算基线...")
    baseline_vector, b_perp_los, b_parallel_los = calculate_baseline(
        master_config, slave_config, vrt_shape, lat_grid, lon_grid, h_grid
    )
    
    # 保存结果到HDF文件
    print(f"保存结果到 {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('baseline_vector', data=baseline_vector, dtype='float32', compression='lzf')
        # 新字段（推荐）
        f.create_dataset('b_perp_los', data=b_perp_los, dtype='float32', compression='lzf')
        f.create_dataset('b_parallel_los', data=b_parallel_los, dtype='float32', compression='lzf')
        # 兼容旧字段名：vertical/horizontal 对应 LOS 分解结果
        f.create_dataset('vertical_baseline', data=b_perp_los, dtype='float32', compression='lzf')
        f.create_dataset('horizontal_baseline', data=b_parallel_los, dtype='float32', compression='lzf')
        
        # 添加元数据
        f.attrs['master'] = args.master
        f.attrs['slave'] = args.slave
        f.attrs['master_time_range'] = f"{master_config['t0']} to {master_config['t1']}"
        f.attrs['slave_time_range'] = f"{slave_config['t0']} to {slave_config['t1']}"
        f.attrs['decomposition'] = 'los_strict'
        f.attrs['ground_grid_source'] = geo_src
        if args.geosar_hdf:
            f.attrs['geosar_hdf'] = args.geosar_hdf
    
    print("基线计算完成！")
    print(f"输出文件: {output_file}")
    print(f"基线向量形状: {baseline_vector.shape}")
    print(f"b_perp_los 范围: {np.nanmin(b_perp_los)} to {np.nanmax(b_perp_los)} meters")
    print(f"b_parallel_los 范围: {np.nanmin(b_parallel_los)} to {np.nanmax(b_parallel_los)} meters")

if __name__ == '__main__':
    main()
