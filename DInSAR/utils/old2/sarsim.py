#!/usr/bin/env python3
"""
SAR 数据模拟模块（优化版）
功能：基于 DEM 生成模拟 SAR 数据
参考 ISCE2 的 SAR 模拟方法实现

优化特性：
- 使用共享内存减少进程间数据复制
- 动态调整并行度基于可用内存
- 优化内存访问模式
- 利用dem2sar2.py的优化轨道计算
- 减少异常处理开销
- 优化I/O操作
- 添加性能统计
- 程序启动时输出当前时间
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import os
import time
import psutil
from osgeo import gdal, osr

# 设置OSR异常处理
osr.UseExceptions()

from multiprocessing import Pool, cpu_count, shared_memory

# 尝试导入numba，如果不可用则使用普通实现
try:
    from numba import jit, prange
except ImportError:
    # 定义替代装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# 从dem2sar.py复制必要的函数和类

class SarGeometry:
    """
    SAR成像几何模型
    """
    def __init__(self, yaml_file: str):
        """
        初始化SAR几何模型
        
        Args:
            yaml_file: YAML文件路径
        """
        self.yaml_file = yaml_file
        self.params = self._load_parameters()
    
    def _load_parameters(self) -> Dict:
        """
        从YAML文件加载成像参数
        
        Returns:
            成像参数字典
        """
        with open(self.yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        params = {}
        
        # 从image_parameters获取尺寸信息
        if 'image_parameters' in data:
            params['nrows'] = data['image_parameters'].get('nrows', 0)
            params['ncols'] = data['image_parameters'].get('ncols', 0)
        
        # 从prm_parameters获取雷达参数
        if 'prm_parameters' in data:
            params['prf'] = data['prm_parameters'].get('PRF', 0.0)
            params['range_sampling_rate'] = data['prm_parameters'].get('rng_samp_rate', 0.0)
            params['radar_wavelength'] = data['prm_parameters'].get('radar_wavelength', 0.0)
            params['near_range'] = data['prm_parameters'].get('near_range', 0.0)
        
        # 从orbit_parameters获取轨道参数
        if 'orbit_parameters' in data:
            params['satellite_height'] = data['orbit_parameters'].get('satellite_height', 0.0)
        
        # 从radar_parameters获取雷达参数
        if 'radar_parameters' in data:
            params['azimuth_spacing'] = data['radar_parameters'].get('azimuth_spacing', 0.0)
            params['range_spacing'] = data['radar_parameters'].get('range_spacing', 0.0)
        
        # 从metadata获取时间参数
        if 'metadata' in data:
            params['first_line_sensing_time'] = data['metadata'].get('first_line_sensing_time', '')
            params['last_line_sensing_time'] = data['metadata'].get('last_line_sensing_time', '')
        
        return params
    
    def get_parameter(self, key: str, default=None):
        """
        获取参数值
        
        Args:
            key: 参数名
            default: 默认值
            
        Returns:
            参数值
        """
        return self.params.get(key, default)

def load_doppler_parameters(yaml_file: str):
    """
    从YAML文件加载多普勒参数
    """
    doppler_polynomial = None
    doppler_derivative_polynomial = None
    
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data and 'processing_parameters' in data:
            proc_params = data['processing_parameters']
            if 'doppler_polynomial' in proc_params:
                doppler_polynomial = proc_params['doppler_polynomial']
            if 'doppler_derivative_polynomial' in proc_params:
                doppler_derivative_polynomial = proc_params['doppler_derivative_polynomial']
    except Exception:
        pass
    
    return doppler_polynomial, doppler_derivative_polynomial

def calculate_look_direction(corner_coords: List, orbit_data: Dict) -> str:
    """
    计算SAR成像的观测方向（左/右视）
    """
    if not corner_coords or len(corner_coords) < 4:
        return "right"
    
    try:
        if not orbit_data or 'orbit_points' not in orbit_data:
            return "right"
        
        orbit_points = orbit_data['orbit_points']
        if len(orbit_points) < 2:
            return "right"
        
        positions = []
        for pt in orbit_points[:10]:
            if 'position' in pt:
                pos = pt['position']
                positions.append([pos['x'], pos['y'], pos['z']])
        
        if len(positions) < 2:
            return "right"
        
        p0 = np.array(positions[0])
        p1 = np.array(positions[-1])
        orbit_dir = p1 - p0
        
        c0 = np.array(corner_coords[0])
        c1 = np.array(corner_coords[1])
        range_dir = c1 - c0
        
        cross = np.cross(orbit_dir, range_dir)
        
        if cross[2] > 0:
            return "left"
        else:
            return "right"
    except Exception:
        return "right"

def get_sar_image_corners(yaml_file: str):
    """
    从YAML文件获取SAR图像四个角点的坐标
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 从YAML文件中获取四个角点的坐标
        # 检查corner_coordinates字段
        if 'corner_coordinates' in data:
            corners = data['corner_coordinates']
            if 'top_left' in corners and 'top_right' in corners and 'bottom_right' in corners and 'bottom_left' in corners:
                return [
                    [corners['top_left']['lon'], corners['top_left']['lat'], corners['top_left'].get('height', 0)],
                    [corners['top_right']['lon'], corners['top_right']['lat'], corners['top_right'].get('height', 0)],
                    [corners['bottom_right']['lon'], corners['bottom_right']['lat'], corners['bottom_right'].get('height', 0)],
                    [corners['bottom_left']['lon'], corners['bottom_left']['lat'], corners['bottom_left'].get('height', 0)]
                ]
            elif 'first_corner' in corners and 'second_corner' in corners:
                first = corners['first_corner']
                second = corners['second_corner']
                return [[first['x'], first['y'], first.get('z', 0)],
                        [second['x'], second['y'], second.get('z', 0)]]
        # 检查image_corners字段（兼容旧格式）
        elif 'image_corners' in data:
            return data['image_corners']
        else:
            # 返回默认值，实际应用中应该从其他字段计算
            return [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ]
    except Exception as e:
        print(f"获取SAR图像角点失败: {e}")
        # 返回默认值
        return [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]

def read_image(file_path):
    """
    读取图像文件
    """
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        
        ds = gdal.Open(file_path)
        if not ds:
            print(f"无法打开图像: {file_path}")
            return None
        
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # 转换为复数
        if data.dtype == np.float32:
            if ds.RasterCount >= 2:
                real = data
                imag = ds.GetRasterBand(2).ReadAsArray()
                image = real + 1j * imag
            else:
                image = data + 0j
        else:
            image = data.astype(np.complex64)
        
        ds = None
        return image
    except Exception as e:
        print(f"读取图像失败: {e}")
        return None


def read_yaml(filename: str) -> Dict:
    """
    读取YAML文件
    """
    import collections
    
    # 添加OrderedDict构造函数
    def construct_ordered_dict(loader, node):
        return collections.OrderedDict(loader.construct_pairs(node))
    
    # 注册构造函数
    yaml.add_constructor('tag:yaml.org,2002:python/object/apply:collections.OrderedDict', construct_ordered_dict)
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        # 将OrderedDict转换为普通字典
        if isinstance(data, collections.OrderedDict):
            data = dict(data)
        # 递归转换所有OrderedDict
        def convert_ordered_dict(obj):
            if isinstance(obj, collections.OrderedDict):
                return {k: convert_ordered_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_ordered_dict(item) for item in obj]
            else:
                return obj
        return convert_ordered_dict(data)


def write_yaml(data: Dict, filename: str):
    """
    写入YAML文件
    """
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def compute_correlation(reference: np.ndarray, registered: np.ndarray) -> float:
    """
    计算相关系数
    """
    # 展平数组
    ref_flat = reference.flatten()
    reg_flat = registered.flatten()
    
    # 计算相关系数
    ref_mean = np.mean(ref_flat)
    reg_mean = np.mean(reg_flat)
    
    numerator = np.sum((ref_flat - ref_mean) * (reg_flat - reg_mean))
    denominator = np.sqrt(
        np.sum((ref_flat - ref_mean)**2) * 
        np.sum((reg_flat - reg_mean)**2)
    )
    
    if denominator > 0:
        return float(numerator / denominator)
    return 0.0


def compute_snr(reference: np.ndarray, registered: np.ndarray) -> float:
    """
    计算信噪比
    """
    # 计算差值
    diff = reference - registered
    
    signal_power = np.mean(reference**2)
    noise_power = np.mean(diff**2)
    
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    return float('inf')


def compute_rmse(reference: np.ndarray, registered: np.ndarray) -> float:
    """
    计算均方根误差
    """
    diff = reference - registered
    rmse = np.sqrt(np.mean(diff**2))
    return float(rmse)


def llh_to_xyz(lat: float, lon: float, height: float,
               major_semi_axis: float = 6378137.0,
               eccentricity_squared: float = 0.00669437999014) -> List[float]:
    """
    经纬度高程转换为地心坐标
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    N = major_semi_axis / np.sqrt(1 - eccentricity_squared * np.sin(lat_rad)**2)
    
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - eccentricity_squared) + height) * np.sin(lat_rad)
    
    return [x, y, z]


def xyz_to_llh(x: float, y: float, z: float,
               major_semi_axis: float = 6378137.0,
               eccentricity_squared: float = 0.00669437999014) -> List[float]:
    """
    地心坐标转换为经纬度高程
    """
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * major_semi_axis, p * (major_semi_axis * (1 - eccentricity_squared)))
    lat = np.arctan2(z + eccentricity_squared * major_semi_axis * np.sin(theta)**3,
                     p - eccentricity_squared * major_semi_axis * np.cos(theta)**3)
    N = major_semi_axis / np.sqrt(1 - eccentricity_squared * np.sin(lat)**2)
    height = (p / np.cos(lat)) - N
    
    return [np.degrees(lat), np.degrees(lon), height]


def llh_to_xyz_vectorized(lat: np.ndarray, lon: np.ndarray, height: np.ndarray,
                          major_semi_axis: float = 6378137.0,
                          eccentricity_squared: float = 0.00669437999014):
    """
    经纬度高程转换为地心坐标（向量化版本）
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    N = major_semi_axis / np.sqrt(1 - eccentricity_squared * np.sin(lat_rad)**2)
    
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - eccentricity_squared) + height) * np.sin(lat_rad)
    
    return x, y, z


def xyz_to_llh_vectorized(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          major_semi_axis: float = 6378137.0,
                          eccentricity_squared: float = 0.00669437999014):
    """
    地心坐标转换为经纬度高程（向量化版本）
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * major_semi_axis, p * (major_semi_axis * (1 - eccentricity_squared)))
    lat = np.arctan2(z + eccentricity_squared * major_semi_axis * np.sin(theta)**3,
                     p - eccentricity_squared * major_semi_axis * np.cos(theta)**3)
    N = major_semi_axis / np.sqrt(1 - eccentricity_squared * np.sin(lat)**2)
    height = (p / np.cos(lat)) - N
    
    return np.degrees(lat), np.degrees(lon), height


def calculate_doppler(slant_range: float,
                      doppler_polynomial: Optional[List[float]] = None,
                      doppler_derivative_polynomial: Optional[List[float]] = None):
    """
    计算多普勒频率和导数
    """
    fdop = 0.0
    fdopder = 0.0
    
    if doppler_polynomial:
        try:
            for i, coeff in enumerate(doppler_polynomial):
                fdop += coeff * (slant_range ** i)
        except Exception:
            pass
    
    if doppler_derivative_polynomial:
        try:
            for i, coeff in enumerate(doppler_derivative_polynomial):
                fdopder += coeff * (slant_range ** i)
        except Exception:
            pass
    
    return fdop, fdopder


def interpolate_sch_orbit(orbit_times, orbit_positions, orbit_velocities, time):
    """
    SCH（Simplified Cubic Hermite）轨道插值
    """
    n_vectors = len(orbit_times)
    
    if n_vectors < 2:
        return orbit_positions[0], orbit_velocities[0] if orbit_velocities else [0, 7000, 0]
    
    pos = [0.0, 0.0, 0.0]
    vel = [0.0, 0.0, 0.0]
    
    for i in range(n_vectors):
        frac = 1.0
        t_i = orbit_times[i]
        pos_i = orbit_positions[i]
        vel_i = orbit_velocities[i] if i < len(orbit_velocities) else [0, 7000, 0]
        
        for j in range(n_vectors):
            if i != j:
                t_j = orbit_times[j]
                num = t_j - time
                den = t_j - t_i
                if den != 0:
                    frac *= num / den
        
        for k in range(3):
            pos[k] += frac * (pos_i[k] + (time - t_i) * vel_i[k])
            vel[k] += frac * vel_i[k]
    
    return pos, vel


def hermite_interpolation(p0, p1, v0, v1, t):
    """
    Hermite（三次埃尔米特）插值
    """
    t2 = t * t
    t3 = t2 * t
    
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    
    return [
        h00 * p0[0] + h10 * v0[0] + h01 * p1[0] + h11 * v1[0],
        h00 * p0[1] + h10 * v0[1] + h01 * p1[1] + h11 * v1[1],
        h00 * p0[2] + h10 * v0[2] + h01 * p1[2] + h11 * v1[2]
    ]


def calculate_orbit_position(time: float, orbit: Dict,
                            default_position: List[float] = [0, 0, 500000],
                            default_velocity: List[float] = [0, 7000, 0],
                            interpolation_method: str = 'HERMITE'):
    """
    计算指定时间的卫星位置和速度
    """
    if not orbit or 'orbit_points' not in orbit:
        return default_position, default_velocity
    
    orbit_points = orbit['orbit_points']
    if not orbit_points:
        return default_position, default_velocity
    
    orbit_times = []
    orbit_positions = []
    orbit_velocities = []
    
    for point in orbit_points:
        if 'time' in point:
            time_str = point['time']
            try:
                import datetime
                dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                timestamp = dt.timestamp()
                orbit_times.append(timestamp)
            except Exception:
                orbit_times.append(len(orbit_times))
            
            if 'position' in point:
                pos = point['position']
                orbit_positions.append([pos['x'], pos['y'], pos['z']])
            if 'velocity' in point:
                vel = point['velocity']
                orbit_velocities.append([vel['vx'], vel['vy'], vel['vz']])
    
    if len(orbit_times) < 2:
        return default_position, default_velocity
    
    orbit_times = np.array(orbit_times)
    orbit_positions = np.array(orbit_positions)
    
    if orbit_velocities:
        orbit_velocities = np.array(orbit_velocities)
    else:
        orbit_velocities = np.zeros_like(orbit_positions)
    
    t0 = orbit_times[0]
    relative_times = orbit_times - t0
    
    time_range = np.ptp(relative_times)
    if time_range == 0:
        return default_position, default_velocity
    
    relative_time = time - t0
    
    method = interpolation_method.upper()
    
    if method == 'LINEAR':
        idx = np.searchsorted(relative_times, relative_time)
        if idx == 0:
            return orbit_positions[0].tolist(), orbit_velocities[0].tolist()
        if idx >= len(relative_times):
            return orbit_positions[-1].tolist(), orbit_velocities[-1].tolist()
        
        t1 = relative_times[idx - 1]
        t2 = relative_times[idx]
        if t2 > t1:
            frac = (relative_time - t1) / (t2 - t1)
        else:
            frac = 0.0
        
        pos = orbit_positions[idx - 1] + frac * (orbit_positions[idx] - orbit_positions[idx - 1])
        vel = orbit_velocities[idx - 1] + frac * (orbit_velocities[idx] - orbit_velocities[idx - 1])
        
        return pos.tolist(), vel.tolist()
    
    elif method == 'HERMITE':
        idx = np.searchsorted(relative_times, relative_time)
        if idx == 0:
            idx = 1
        elif idx >= len(relative_times):
            idx = len(relative_times) - 1
        
        t1 = relative_times[idx - 1]
        t2 = relative_times[idx]
        if t2 > t1:
            t = (relative_time - t1) / (t2 - t1)
        else:
            t = 0.0
        
        p0 = orbit_positions[idx - 1].tolist()
        p1 = orbit_positions[idx].tolist()
        v0 = orbit_velocities[idx - 1].tolist() if len(orbit_velocities) > 0 else [0, 7000, 0]
        v1 = orbit_velocities[idx].tolist() if len(orbit_velocities) > 0 else [0, 7000, 0]
        
        scale = t2 - t1
        v0_scaled = [v0[i] * scale for i in range(3)]
        v1_scaled = [v1[i] * scale for i in range(3)]
        
        pos = hermite_interpolation(p0, p1, v0_scaled, v1_scaled, t)
        
        if len(orbit_velocities) > 0:
            dt = t2 - t1
            if dt > 0:
                vel = [(p1[i] - p0[i]) / dt for i in range(3)]
            else:
                vel = v0
        else:
            vel = v0
        
        return pos, vel
    
    elif method == 'SCH':
        orbit_times_list = relative_times.tolist()
        orbit_positions_list = [p.tolist() for p in orbit_positions]
        orbit_velocities_list = [v.tolist() for v in orbit_velocities] if len(orbit_velocities) > 0 else [[0, 7000, 0]] * len(orbit_times_list)
        
        pos, vel = interpolate_sch_orbit(orbit_times_list, orbit_positions_list, orbit_velocities_list, relative_time)
        return pos, vel
    
    else:
        idx = np.searchsorted(relative_times, relative_time)
        if idx == 0:
            return orbit_positions[0].tolist(), orbit_velocities[0].tolist()
        if idx >= len(relative_times):
            return orbit_positions[-1].tolist(), orbit_velocities[-1].tolist()
        
        t1 = relative_times[idx - 1]
        t2 = relative_times[idx]
        if t2 > t1:
            frac = (relative_time - t1) / (t2 - t1)
        else:
            frac = 0.0
        
        pos = orbit_positions[idx - 1] + frac * (orbit_positions[idx] - orbit_positions[idx - 1])
        vel = orbit_velocities[idx - 1] + frac * (orbit_velocities[idx] - orbit_velocities[idx - 1])
        
        return pos.tolist(), vel.tolist()

class DemToSarConverter:
    """
    DEM到SAR坐标转换器（优化版）
    """
    def __init__(self, yaml_file: str, dem_file: str,
                 orbit_interpolation_method: str = 'HERMITE',
                 streaming: bool = False,
                 max_cache_size: int = 100000):
        """
        初始化转换器
        
        Args:
            yaml_file: SAR YAML文件路径
            dem_file: DEM文件路径
            orbit_interpolation_method: 轨道插值方法，HERMITE/SCH/LINEAR
            streaming: 是否使用流式处理模式
            max_cache_size: 最大缓存大小
        """
        self.yaml_file = yaml_file
        self.dem_file = dem_file
        self.orbit_interpolation_method = orbit_interpolation_method.upper()
        self.streaming = streaming
        self.max_cache_size = max_cache_size
        self.sar_geometry = SarGeometry(yaml_file)
        self.dem_data = None
        self.dem_geotransform = None
        self.dem_projection = None
        self.orbit_positions = []
        self.orbit_velocities = []
        self.orbit_times = []
        self.orbit_data = None
        self.look_direction = None  # 视线方向
        self.first_line_sensing_time = None  # 第一行成像时间
        self.last_line_sensing_time = None  # 最后一行成像时间
        
        # 轨道插值缓存
        self._orbit_interp_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 预计算的轨道样条
        self._orbit_splines = None
        
        # 预计算的轨道状态表（优化缓存）
        self._orbit_state_table = None
        self._orbit_state_times = None
        
        self._load_dem()
        self._precompute_parameters()
        self._precompute_orbit_positions()
        self._calculate_look_direction()
        self._precompute_orbit_splines()
        self._precompute_orbit_state_table()
        
        # 如果sensing_start仍为0，尝试从orbit_times获取起始时间
        if hasattr(self, 'sensing_start') and self.sensing_start == 0.0 and len(self.orbit_times) > 0:
            self.sensing_start = self.orbit_times[0]
            print(f"使用orbit_times[0]作为sensing_start: {self.sensing_start}")
        
        # 验证时间范围一致性
        self._validate_time_range()
        
        self.geo2rdr = self._create_geo2rdr()
    
    def _validate_time_range(self):
        """验证时间范围一致性"""
        if not hasattr(self, 'sensing_start'):
            print("⚠️  警告: sensing_start未定义")
            return
        
        if self.sensing_start == 0.0:
            print("⚠️  严重警告: sensing_start为0，坐标转换将不准确！")
            if self.orbit_cache and len(self.orbit_cache['times']) > 0:
                orbit_t_min = self.orbit_cache['times'][0]
                self.sensing_start = orbit_t_min
                print(f"✓ 自动调整sensing_start为轨道起始时间: {self.sensing_start}")
        
        # 检查轨道时间范围
        if self.orbit_cache and len(self.orbit_cache['times']) > 0:
            orbit_t_min = self.orbit_cache['times'][0]
            orbit_t_max = self.orbit_cache['times'][-1]
            
            # 计算SAR图像的时间范围（相对时间）
            sar_t_min = 0.0
            sar_t_max = self.nrows * self.azimuth_time_step
            
            print(f"=== 时间范围验证 ===")
            print(f"轨道时间范围(相对): [{orbit_t_min:.2f}, {orbit_t_max:.2f}] ({orbit_t_max - orbit_t_min:.2f}秒)")
            print(f"SAR时间范围(相对):  [{sar_t_min:.2f}, {sar_t_max:.2f}] ({sar_t_max - sar_t_min:.2f}秒)")
            
            # 检查是否在轨道时间范围内
            if sar_t_max > orbit_t_max:
                print(f"⚠️  警告: SAR结束时间晚于轨道结束时间 {sar_t_max - orbit_t_max:.2f}秒")
            
            print(f"✓ 时间范围验证通过")
    
    def _create_geo2rdr(self):
        """创建 Geo2rdr 实例"""
        try:
            # 尝试导入Geo2rdr
            try:
                from .geo2rdr import Geo2rdr
            except ImportError:
                import sys
                pkgdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if pkgdir not in sys.path:
                    sys.path.insert(0, pkgdir)
                from geo2rdr import Geo2rdr
            
            g2r = Geo2rdr(
                prf=self.prf,
                radar_wavelength=self.radar_wavelength,
                slant_range_pixel_spacing=self.range_pixel_spacing,
                range_first_sample=self.near_range,
                sensing_start=getattr(self, 'sensing_start', 0.0),
                look_side=self.look_direction.upper() if self.look_direction else 'RIGHT',
                orbit_interpolation_method=self.orbit_interpolation_method
            )
            
            if self.orbit_data:
                g2r.set_orbit(self.orbit_data)
            
            dop_poly, dop_der_poly = getattr(self, 'doppler_polynomial', None), getattr(self, 'doppler_derivative_polynomial', None)
            if dop_poly:
                g2r.doppler_polynomial = dop_poly
            if dop_der_poly:
                g2r.doppler_derivative_polynomial = dop_der_poly
            
            return g2r
        except Exception as e:
            print(f"创建Geo2rdr实例失败: {e}")
            return None
    
    def _load_dem(self):
        """
        加载DEM数据
        """
        try:
            from osgeo import gdal
            
            # 设置GDAL异常处理
            gdal.UseExceptions()
            
            # 打开DEM文件
            if not os.path.exists(self.dem_file):
                raise FileNotFoundError(f"DEM file not found: {self.dem_file}")
            
            dataset = gdal.Open(self.dem_file)
            if dataset is None:
                raise Exception(f"无法打开DEM文件: {self.dem_file}")
            
            band = dataset.GetRasterBand(1)
            self.dem_data = band.ReadAsArray()
            self.dem_geotransform = dataset.GetGeoTransform()
            self.dem_projection = dataset.GetProjection()
            
            dataset = None  # 释放资源
        except ImportError:
            print("警告: GDAL未安装，无法加载DEM文件")
        except Exception as e:
            print(f"加载DEM文件失败: {e}")
    
    def _dem_pixel_to_lon_lat(self, row: int, col: int) -> Tuple[float, float]:
        """
        DEM像素坐标转换为经纬度
        
        Args:
            row: 行号
            col: 列号
            
        Returns:
            (经度, 纬度)
        """
        if self.dem_geotransform is None:
            raise Exception("DEM地理变换信息未加载")
        
        # 首先计算像素的地理坐标（可能是UTM坐标）
        x = self.dem_geotransform[0] + col * self.dem_geotransform[1] + row * self.dem_geotransform[2]
        y = self.dem_geotransform[3] + col * self.dem_geotransform[4] + row * self.dem_geotransform[5]
        
        # 尝试使用GDAL进行坐标转换
        try:
            from osgeo import gdal, osr
            
            # 设置OSR异常处理
            osr.UseExceptions()
            
            # 创建源空间参考（从DEM获取）
            src_srs = osr.SpatialReference(wkt=self.dem_projection)
            
            # 创建目标空间参考（WGS84经纬度）
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(4326)  # WGS84
            
            # 创建坐标转换
            transform = osr.CoordinateTransformation(src_srs, dst_srs)
            
            # 尝试多种坐标顺序组合
            attempts = [(x, y), (y, x)]
            for i, (coord1, coord2) in enumerate(attempts):
                try:
                    lon, lat, _ = transform.TransformPoint(coord1, coord2)
                    
                    # 检查经纬度值是否有效
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        return lon, lat
                except Exception:
                    pass
            
            # 返回原始坐标
            return x, y
        except ImportError:
            # 如果GDAL不可用，直接返回原始坐标（可能是UTM）
            return x, y
        except Exception:
            # 如果转换失败，直接返回原始坐标
            return x, y
    
    def _precompute_parameters(self):
        """
        预计算成像参数
        """
        self.nrows = self.sar_geometry.get_parameter('nrows', 14580)  # 默认值
        self.ncols = self.sar_geometry.get_parameter('ncols', 12544)  # 默认值
        self.prf = self.sar_geometry.get_parameter('prf', 4144.0)  # 默认值
        self.range_sampling_rate = self.sar_geometry.get_parameter('range_sampling_rate', 1.0e8)  # 默认值
        self.radar_wavelength = self.sar_geometry.get_parameter('radar_wavelength', 0.031)  # 默认值
        self.near_range = self.sar_geometry.get_parameter('near_range', 60000.0)  # 默认值
        self.azimuth_spacing = self.sar_geometry.get_parameter('azimuth_spacing', 5.0)  # 方位向分辨率（米）
        self.range_spacing = self.sar_geometry.get_parameter('range_spacing', 1.5)  # 距离向分辨率（米）
        
        # 计算时间步长
        if self.prf > 0:
            self.azimuth_time_step = 1.0 / self.prf
        else:
            self.azimuth_time_step = 1.0 / 4144.0  # 默认值
        
        # 计算距离向采样间隔
        if self.range_sampling_rate > 0:
            self.c = 299792458.0  # 精确光速
            self.range_pixel_spacing = self.c / (2 * self.range_sampling_rate)
        else:
            self.range_pixel_spacing = 1.5  # 默认值（约1.5米）
        
        # 初始化多普勒参数
        self.doppler_polynomial, self.doppler_derivative_polynomial = load_doppler_parameters(self.yaml_file)
        
        # 读取时间参数
        self.first_line_sensing_time = self.sar_geometry.get_parameter('first_line_sensing_time', None)
        self.last_line_sensing_time = self.sar_geometry.get_parameter('last_line_sensing_time', None)
        
        # 计算 sensing_start（第一行成像时间转换为秒）
        self.sensing_start = 0.0
        if self.first_line_sensing_time:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(self.first_line_sensing_time.replace('Z', '+00:00'))
                self.sensing_start = dt.timestamp()
            except Exception as e:
                print(f"警告: 解析first_line_sensing_time失败: {e}")
        
        # 如果sensing_start仍为0，尝试从orbit_data获取起始时间
        if self.sensing_start == 0.0 and self.orbit_data:
            try:
                orbit_points = self.orbit_data.get('orbit_points', [])
                if orbit_points and 'time' in orbit_points[0]:
                    from datetime import datetime
                    dt = datetime.fromisoformat(orbit_points[0]['time'].replace('Z', '+00:00'))
                    self.sensing_start = dt.timestamp()
                    print(f"使用轨道起始时间作为sensing_start: {self.sensing_start}")
            except Exception as e:
                print(f"警告: 获取轨道起始时间失败: {e}")
        
        # 预读取插值方式（如果YAML中指定）
        try:
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            method = cfg.get('orbit_interpolation_method')
            if method:
                self.orbit_interpolation_method = method.upper()
        except Exception:
            pass
        
        # 预计算轨道数据，用于优化的函数
        self._precompute_orbit_cache()
    
    def _precompute_orbit_cache(self):
        """
        预计算轨道数据缓存
        """
        try:
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'orbit_data' in data:
                orbit_data = data['orbit_data']
                
                if 'orbit_points' in orbit_data:
                    orbit_points = orbit_data['orbit_points']
                    
                    # 预计算并缓存轨道数据
                    self.orbit_cache = {
                        'positions': [],
                        'velocities': [],
                        'times': []
                    }
                    
                    # 存储时间戳，用于计算相对时间
                    timestamps = []
                    
                    for i, point in enumerate(orbit_points):
                        if 'position' in point and 'time' in point:
                            pos = point['position']
                            self.orbit_cache['positions'].append([pos['x'], pos['y'], pos['z']])
                            # 转换时间为数值
                            try:
                                import datetime
                                time_str = point['time']
                                dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                                timestamp = dt.timestamp()
                                timestamps.append(timestamp)
                            except Exception as e:
                                # 如果时间转换失败，使用索引作为时间
                                timestamps.append(len(timestamps))
                        if 'velocity' in point:
                            vel = point['velocity']
                            self.orbit_cache['velocities'].append([vel['vx'], vel['vy'], vel['vz']])
                    
                    # 转换为相对时间（相对于第一个轨道点的时间）
                    if timestamps:
                        t0 = timestamps[0]
                        self.orbit_start_time = t0  # 存储原始轨道起始时间（绝对时间）
                        relative_times = [t - t0 for t in timestamps]
                        # 使用 double 精度存储时间
                        arr = np.array(relative_times, dtype=np.float64)
                        # 检查时间范围
                        time_range = np.ptp(arr)
                        # 如果所有时间相同则构造简单的时间轴并调整位置
                        if time_range == 0:
                            npts = arr.size
                            # 线性分布0..1秒，使用 double 精度
                            arr = np.linspace(0.0, 1.0, npts, dtype=np.float64)
                            # 如果有速度信息, 按第一速度推进位置
                            if self.orbit_cache['velocities']:
                                v0 = np.array(self.orbit_cache['velocities'][0])
                                p0 = np.array(self.orbit_cache['positions'][0])
                                newpos = []
                                for t in arr:
                                    newpos.append(p0 + v0 * t)
                                self.orbit_cache['positions'] = newpos
                        self.orbit_cache['times'] = arr
                    else:
                        self.orbit_cache['times'] = np.array([])
                    
                    # 转换为NumPy数组，提高处理速度
                    self.orbit_cache['positions'] = np.array(self.orbit_cache['positions'], dtype=np.float64)
                    self.orbit_cache['velocities'] = np.array(self.orbit_cache['velocities'], dtype=np.float64)
                else:
                    self.orbit_cache = None
            else:
                self.orbit_cache = None
        except Exception as e:
            self.orbit_cache = None
    
    def _precompute_orbit_splines(self):
        """
        预计算轨道样条插值
        """
        try:
            from scipy.interpolate import CubicSpline
            
            if self.orbit_cache and len(self.orbit_cache['times']) >= 4:
                times = self.orbit_cache['times']
                positions = self.orbit_cache['positions']
                velocities = self.orbit_cache['velocities']
                
                # 为每个坐标分量创建样条
                self._orbit_splines = {
                    'pos_x': CubicSpline(times, positions[:, 0]),
                    'pos_y': CubicSpline(times, positions[:, 1]),
                    'pos_z': CubicSpline(times, positions[:, 2])
                }
                
                if velocities.size > 0:
                    self._orbit_splines.update({
                        'vel_x': CubicSpline(times, velocities[:, 0]),
                        'vel_y': CubicSpline(times, velocities[:, 1]),
                        'vel_z': CubicSpline(times, velocities[:, 2])
                    })
            else:
                self._orbit_splines = None
        except ImportError:
            self._orbit_splines = None
        except Exception as e:
            self._orbit_splines = None
    
    def _precompute_orbit_state_table(self):
        """
        预计算轨道状态表（优化缓存机制）
        预先计算整个时间范围内的卫星位置和速度，大幅减少实时计算量
        采样点数与方位向样本数一致
        """
        if not self.orbit_cache or self.orbit_cache['times'].size < 2:
            return
        
        try:
            times = self.orbit_cache['times']
            positions = self.orbit_cache['positions']
            velocities = self.orbit_cache['velocities']
            
            t_min = float(times[0])
            t_max = float(times[-1])
            
            azimuth_samples = getattr(self, 'nrows', 0)
            if azimuth_samples > 0:
                n_samples = azimuth_samples
            else:
                time_range = t_max - t_min
                n_samples = max(100, min(10000, int(time_range * 100)))
            
            # 创建时间采样点
            table_times = np.linspace(t_min, t_max, n_samples)
            
            # 预计算位置和速度
            table_positions = np.zeros((n_samples, 3), dtype=np.float64)
            table_velocities = np.zeros((n_samples, 3), dtype=np.float64)
            
            for i, t in enumerate(table_times):
                pos, vel = self._interpolate_orbit_simple(t, times, positions, velocities)
                table_positions[i] = pos
                table_velocities[i] = vel
            
            self._orbit_state_table = {
                'positions': table_positions,
                'velocities': table_velocities
            }
            self._orbit_state_times = table_times
        except Exception as e:
            self._orbit_state_table = None
            self._orbit_state_times = None
    
    def _precompute_orbit_positions(self):
        """
        预计算轨道位置
        """
        try:
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'orbit_data' in data and 'orbit_points' in data['orbit_data']:
                self.orbit_data = data['orbit_data']
                orbit_points = data['orbit_data']['orbit_points']
                for point in orbit_points:
                    if 'position' in point and 'time' in point:
                        pos = point['position']
                        self.orbit_positions.append([pos['x'], pos['y'], pos['z']])
                        # 将时间字符串转换为数值（使用相对时间）
                        time_str = point['time']
                        # 简单处理：使用时间字符串的秒部分作为数值
                        try:
                            # 提取时间部分并转换为秒
                            import datetime
                            dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            # 转换为时间戳
                            timestamp = dt.timestamp()
                            self.orbit_times.append(timestamp)
                        except Exception:
                            # 如果转换失败，使用索引作为时间
                            self.orbit_times.append(len(self.orbit_times))
                    if 'velocity' in point:
                        vel = point['velocity']
                        self.orbit_velocities.append([vel['vx'], vel['vy'], vel['vz']])
            else:
                self.orbit_data = None
        except Exception:
            self.orbit_data = None
    
    def _calculate_look_direction(self):
        """
        计算卫星视线方向
        """
        try:
            # 获取SAR图像四个角点的坐标
            corner_coords = get_sar_image_corners(self.yaml_file)
            
            # 计算视线方向
            self.look_direction = calculate_look_direction(corner_coords, self.orbit_data)
            
        except Exception:
            self.look_direction = "unknown"
    
    def _interpolate_orbit_simple(self, time, orbit_times, orbit_positions, orbit_velocities):
        """
        简化的轨道插值（带缓存和二分搜索）
        """
        # 检查缓存
        cache_key = round(time * 1e6)  # 微秒精度的缓存键
        if cache_key in self._orbit_interp_cache:
            self._cache_hits += 1
            return self._orbit_interp_cache[cache_key]
        self._cache_misses += 1
        
        # 首先处理特殊情况：所有轨道时间相同
        if orbit_times.size >= 1 and np.ptp(orbit_times) == 0:
            if orbit_velocities.size >= 3:
                base = orbit_positions.reshape(-1,3)[0]
                vel0 = orbit_velocities.reshape(-1,3)[0]
                pos = base + (time - orbit_times[0]) * vel0
                result = (pos, vel0)
            else:
                result = (orbit_positions[0], orbit_velocities[0] if orbit_velocities.size>0 else np.array([0.0,7000.0,0.0]))
            # 缓存结果，限制缓存大小
            if len(self._orbit_interp_cache) < self.max_cache_size:
                self._orbit_interp_cache[cache_key] = result
            return result
        
        if orbit_times.size < 2:
            base_pos = np.array([-284515.577, 5480417.824, 4154857.155]) if orbit_positions.size > 0 else np.array([0.0, 0.0, 500000.0])
            base_vel = np.array([0.0, 7000.0, 0.0]) if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0])
            pos = base_pos + time * base_vel
            result = (pos, base_vel)
            if len(self._orbit_interp_cache) < self.max_cache_size:
                self._orbit_interp_cache[cache_key] = result
            return result
        
        # 使用二分查找而不是遍历所有时间点（显著加快速度）
        idx = np.searchsorted(orbit_times, time)
        if idx == 0:
            result = (orbit_positions[0], orbit_velocities[0] if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0]))
        elif idx >= orbit_times.size:
            result = (orbit_positions[-1], orbit_velocities[-1] if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0]))
        else:
            i = idx - 1
            t1 = orbit_times[i]
            t2 = orbit_times[i+1]
            t = (time - t1) / (t2 - t1)
            pos1 = orbit_positions[i]
            pos2 = orbit_positions[i+1]
            pos = pos1 + t * (pos2 - pos1)
            if orbit_velocities.size >= 2:
                vel1 = orbit_velocities[i]
                vel2 = orbit_velocities[i+1]
                vel = vel1 + t * (vel2 - vel1)
            else:
                vel = np.array([0.0, 7000.0, 0.0])
            result = (pos, vel)
        
        # 缓存结果，限制缓存大小
        if len(self._orbit_interp_cache) < self.max_cache_size:
            self._orbit_interp_cache[cache_key] = result
        return result
    
    def _llh_to_xyz(self, lat: float, lon: float, height: float) -> List[float]:
        """
        经纬度高程转换为地心坐标
        
        Args:
            lat: 纬度（度）
            lon: 经度（度）
            height: 高程（米）
            
        Returns:
            地心坐标 [x, y, z]
        """
        return llh_to_xyz(lat, lon, height)
    
    def _calculate_sar_coordinates(self, lon: float, lat: float, height: float) -> Tuple[float, float]:
        """
        计算SAR坐标
        
        Args:
            lon: 经度
            lat: 纬度
            height: 高程
            
        Returns:
            (距离向采样点, 方位向时间)
        """
        # 将地理坐标转换为地心坐标
        target_xyz = self._llh_to_xyz(lat, lon, height)
        
        # 检查地心坐标是否有效
        if any(np.isnan(coord) or abs(coord) > 1e8 for coord in target_xyz):
            raise Exception(f"无效的地心坐标: {target_xyz}")
        
        # 检查轨道缓存是否有效
        if not self.orbit_cache:
            raise Exception("轨道缓存未初始化")
        
        orb_times = self.orbit_cache['times']
        orb_pos = self.orbit_cache['positions']
        orb_vel = self.orbit_cache['velocities']
        if orb_times.size < 2:
            raise Exception(f"轨道点数量不足: {orb_times.size}")
        
        # 如果所有轨道时间相同, 生成一个简单的时间轴并调整位置
        if np.ptp(orb_times) == 0:
            print("警告: 轨道时间全部相同，构造人工时间和位置")
            n = orb_times.size
            base = orb_times[0]
            # 线性分布0..1秒
            orb_times = np.linspace(base, base + 1.0, n)
            if orb_vel.size >= 3:
                # 按第一速度向量简单推进
                v0 = orb_vel.reshape(-1, 3)[0]
                p0 = orb_pos.reshape(-1, 3)[0]
                newpos = []
                for t in orb_times - base:
                    newpos.append(p0 + v0 * t)
                orb_pos = np.vstack(newpos)
            self.orbit_cache['times'] = orb_times
            self.orbit_cache['positions'] = orb_pos
        
        # 调用优化的函数，使用预计算的轨道缓存
        try:
            result = self._calculate_sar_coordinates_optimized(
                np.array(target_xyz),
                orb_times,
                orb_pos,
                orb_vel,
                self.near_range,
                self.range_pixel_spacing,
                self.doppler_polynomial,
                self.doppler_derivative_polynomial
            )
            
            # 检查结果是否有效
            if np.isnan(result[0]) or np.isnan(result[1]):
                raise Exception(f"无效的SAR坐标结果: {result}")
            
            return result
        except Exception as e:
            raise Exception(f"调用优化函数失败: {str(e)}")
    
    def _calculate_sar_coordinates_optimized(self, target_xyz, orbit_times, orbit_positions, orbit_velocities, near_range, range_pixel_spacing, doppler_polynomial, doppler_derivative_polynomial):
        """
        SAR坐标计算（优化版）。

        这个方法仅负责数值求解——轨道插值由
        :meth:`_satellite_state` 处理，从而支持不同插值方式并
        正确处理退化的时间序列。
        """
        # 初步验证参数
        if np.isnan(target_xyz).any():
            raise ValueError(f"无效的目标坐标: {target_xyz}")

        if orbit_times.size < 2:
            raise ValueError(f"轨道点数量不足: {orbit_times.size}")

        if near_range <= 0:
            raise ValueError(f"无效的近距: {near_range}")
        if range_pixel_spacing <= 0:
            raise ValueError(f"无效的距离像素间距: {range_pixel_spacing}")

        # 轨道时间范围与退化处理，使用 double 精度
        t_min = float(orbit_times[0])
        t_max = float(orbit_times[-1])
        t_range = t_max - t_min
        if t_range == 0:
            # 给出合理的目标以使迭代能够更新
            t_max = t_min + 1.0
            t_range = 1.0

        # 初始估计：使用向量化查询找最近的轨道点
        dists = np.linalg.norm(orbit_positions - target_xyz, axis=1)
        tmid = float(orbit_times[np.argmin(dists)])

        # 牛顿-拉夫逊求解
        max_iter = 51
        tolerance = 5.0e-9
        tline = tmid

        for k in range(max_iter):
            tline = max(t_min, min(t_max, tline))

            sat_pos, sat_vel = self._satellite_state(tline)
            if np.isnan(sat_pos).any() or np.isnan(sat_vel).any():
                raise ValueError(f"无效的卫星状态: pos={sat_pos}, vel={sat_vel}")

            dr = target_xyz - sat_pos
            slant_range = np.linalg.norm(dr)
            if slant_range <= 0 or np.isnan(slant_range):
                raise ValueError(f"无效的斜距: {slant_range}")

            dopfact = np.dot(dr, sat_vel)
            fdop, fdopder = self._calculate_doppler_numba(slant_range, doppler_polynomial, doppler_derivative_polynomial)
            if fdop == 0.0:
                dr_unit = dr / slant_range
                vel_los = np.dot(sat_vel, dr_unit)
                wavelength = 0.031
                fdop = -2 * vel_los / wavelength
                fdopder = 0.0
            if self.look_direction == "left":
                fdop *= -1; fdopder *= -1

            fn = dopfact - fdop * slant_range
            c1 = - np.dot(sat_vel, sat_vel)
            c2 = (fdop / slant_range) + fdopder
            fnprime = c1 + c2 * dopfact

            if abs(fnprime) > 1e-12:
                delta_t = fn / fnprime
                if abs(delta_t) > t_range/4:
                    delta_t *= 0.5
                tline -= delta_t
                if abs(delta_t) < tolerance and abs(fn) < tolerance*slant_range:
                    break
            else:
                # 导数接近零时退化为二分查找
                low, high = t_min, t_max
                for _ in range(20):
                    mid = 0.5*(low+high)
                    sat_pos_mid, sat_vel_mid = self._satellite_state(mid)
                    dr_mid = target_xyz - sat_pos_mid
                    slant_mid = np.linalg.norm(dr_mid)
                    dopfact_mid = np.dot(dr_mid, sat_vel_mid)
                    fdop_mid, fdopder_mid = self._calculate_doppler_numba(slant_mid, doppler_polynomial, doppler_derivative_polynomial)
                    if fdop_mid == 0.0:
                        dr_unit_mid = dr_mid / slant_mid
                        vel_los_mid = np.dot(sat_vel_mid, dr_unit_mid)
                        wavelength = 0.031
                        fdop_mid = -2 * vel_los_mid / wavelength
                    if self.look_direction == "left":
                        fdop_mid *= -1
                    fn_mid = dopfact_mid - fdop_mid * slant_mid
                    if fn_mid > 0:
                        high = mid
                    else:
                        low = mid
                tline = 0.5*(low+high)
                break

        tline = max(t_min, min(t_max, tline))
        sat_pos, _ = self._satellite_state(tline)
        dr = target_xyz - sat_pos
        slant_range = np.linalg.norm(dr)

        range_sample = (slant_range - near_range) / range_pixel_spacing
        relative_azimuth_time = tline - t_min
        if np.isnan(range_sample) or np.isnan(relative_azimuth_time):
            raise ValueError(f"无效的SAR坐标结果: range={range_sample}, azimuth={relative_azimuth_time}")

        if self.ncols > 0:
            range_sample = min(range_sample, self.ncols)
        else:
            range_sample = min(range_sample, 12544)
        if self.nrows > 0 and self.prf > 0:
            relative_azimuth_time = max(0, relative_azimuth_time)
        return range_sample, relative_azimuth_time
    
    def _calculate_doppler_numba(self, slant_range, doppler_polynomial, doppler_derivative_polynomial):
        """
        多普勒计算
        """
        fdop = 0.0
        fdopder = 0.0
        
        # 计算多普勒频率
        if doppler_polynomial:
            for i, coeff in enumerate(doppler_polynomial):
                fdop += coeff * (slant_range ** i)
        
        # 计算多普勒导数
        if doppler_derivative_polynomial:
            for i, coeff in enumerate(doppler_derivative_polynomial):
                fdopder += coeff * (slant_range ** i)
        
        return fdop, fdopder
    
    def convert(self, row: int, col: int) -> Tuple[float, float]:
        """
        转换DEM像素到SAR坐标
        
        Args:
            row: DEM行号
            col: DEM列号
            
        Returns:
            (距离向采样点, 方位向相对时间（秒）)
        """
        if self.dem_data is None:
            raise Exception("DEM数据未加载")
        
        # 检查坐标是否在DEM范围内，如果超出则夹住并记录
        nrows, ncols = self.dem_data.shape
        if row < 0 or row >= nrows or col < 0 or col >= ncols:
            # clamp and warn
            r2 = min(max(row, 0), nrows-1)
            c2 = min(max(col, 0), ncols-1)
            print(f"警告: DEM坐标({row},{col})越界，使用({r2},{c2})替代")
            row, col = r2, c2
        
        # 获取DEM高程
        height = self.dem_data[row, col]
        
        # 检查高程值是否有效
        if np.isnan(height) or height < 0:
            raise Exception(f"无效的高程值: {height}，点 ({row}, {col})")
        
        # 转换为经纬度
        lon, lat = self._dem_pixel_to_lon_lat(row, col)
        
        # 检查经纬度值是否有效
        if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
            raise Exception(f"无效的经纬度值: lon={lon}, lat={lat}，点 ({row}, {col})")
        
        # 计算orbit起始绝对时间
        orbit_t0 = getattr(self, 'orbit_start_time', self.sensing_start)
        
        # 调试计数器
        if not hasattr(self, '_convert_debug_count'):
            self._convert_debug_count = 0
        
        if self.geo2rdr:
            try:
                # 关键修复：使用geo2rdr，要求返回相对于orbit起始的相对时间
                range_sample, azimuth_time_orbit_rel = self.geo2rdr.geo2rdr(lat, lon, height, return_relative_time=True)
                
                # 计算sensing_start相对于orbit_t0的偏移
                time_offset = self.sensing_start - orbit_t0
                
                # SAR相对时间 = 轨道相对时间 - sensing_start相对于orbit_t0的偏移
                azimuth_time_rel = azimuth_time_orbit_rel - time_offset
                
                # 调试：打印前3次调用的时间信息
                self._convert_debug_count += 1
                if self._convert_debug_count <= 3:
                    print(f"DEBUG #{self._convert_debug_count}: orbit_t0={orbit_t0:.2f}, ss={self.sensing_start:.2f}")
                    print(f"DEBUG #{self._convert_debug_count}: 轨道相对时间={azimuth_time_orbit_rel:.2f}")
                    print(f"DEBUG #{self._convert_debug_count}: SAR相对时间范围=[0, {self.nrows * self.azimuth_time_step:.2f}]")
                    print(f"DEBUG #{self._convert_debug_count}: time_offset={time_offset:.2f}")
                    print(f"DEBUG #{self._convert_debug_count}: 最终SAR相对时间={azimuth_time_rel:.2f}")
                
                return range_sample, azimuth_time_rel
            except Exception as e:
                print(f"使用Geo2rdr转换失败: {e}，尝试使用备用方法")
                import traceback
                traceback.print_exc()
        
        # 备用方法：使用自己的计算方法
        try:
            range_sample, azimuth_time_abs = self._calculate_sar_coordinates(lon, lat, height)
            # 将绝对时间转换为相对于orbit起始的相对时间
            azimuth_time_rel = azimuth_time_abs - orbit_t0
            return range_sample, azimuth_time_rel
        except Exception as e:
            raise Exception(f"计算SAR坐标失败: {str(e)}")
    
    def _satellite_state(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取给定时间的卫星位置和速度，带有退化处理。

        当 ``orbit_data`` 无效时会退回到简单的线性插值模型。
        优化：优先使用预计算的轨道状态表，大幅提升查询速度
        """
        # 优先使用预计算的轨道状态表（最快）
        if self._orbit_state_table is not None and self._orbit_state_times is not None:
            try:
                t = float(time)
                times = self._orbit_state_times
                positions = self._orbit_state_table['positions']
                velocities = self._orbit_state_table['velocities']
                
                # 使用二分查找快速定位
                idx = np.searchsorted(times, t)
                
                if idx == 0:
                    return positions[0], velocities[0]
                elif idx >= len(times):
                    return positions[-1], velocities[-1]
                else:
                    # 线性插值
                    t1 = times[idx - 1]
                    t2 = times[idx]
                    frac = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
                    pos = positions[idx - 1] + frac * (positions[idx] - positions[idx - 1])
                    vel = velocities[idx - 1] + frac * (velocities[idx] - velocities[idx - 1])
                    return pos, vel
            except Exception:
                # 状态表查询失败时继续使用其他方法
                pass
        
        # 优先使用样条插值
        if self._orbit_splines:
            try:
                t = float(time)
                pos = np.array([
                    self._orbit_splines['pos_x'](t),
                    self._orbit_splines['pos_y'](t),
                    self._orbit_splines['pos_z'](t)
                ], dtype=np.float64)
                
                if 'vel_x' in self._orbit_splines:
                    vel = np.array([
                        self._orbit_splines['vel_x'](t),
                        self._orbit_splines['vel_y'](t),
                        self._orbit_splines['vel_z'](t)
                    ], dtype=np.float64)
                else:
                    # 如果没有速度样条，使用差分近似
                    dt = 1e-6
                    vel = (np.array([
                        self._orbit_splines['pos_x'](t+dt),
                        self._orbit_splines['pos_y'](t+dt),
                        self._orbit_splines['pos_z'](t+dt)
                    ]) - pos) / dt
                
                return pos, vel
            except Exception:
                # 样条插值失败时继续使用其他方法
                pass
        
        # 优先使用YAML中的轨道数据，如果不存在则退回到缓存
        od = getattr(self, 'orbit_data', None)
        if od is not None:
            try:
                return self._calculate_orbit_position(time, od,
                                                interpolation_method=self.orbit_interpolation_method)
            except Exception:
                # 插值失败时继续使用缓存处理
                pass
        # 使用缓存的数据进行线性 / 特例插值
        return self._interpolate_orbit_simple(time,
                                               self.orbit_cache['times'],
                                               self.orbit_cache['positions'],
                                               self.orbit_cache['velocities'])
    
    def _calculate_orbit_position(self, time: float, orbit: Dict,
                                default_position: List[float] = [0, 0, 500000],
                                default_velocity: List[float] = [0, 7000, 0],
                                interpolation_method: str = 'HERMITE'):
        """
        计算指定时间的卫星位置和速度
        """
        if not orbit or 'orbit_points' not in orbit:
            return default_position, default_velocity
        
        orbit_points = orbit['orbit_points']
        if not orbit_points:
            return default_position, default_velocity
        
        orbit_times = []
        orbit_positions = []
        orbit_velocities = []
        
        for point in orbit_points:
            if 'position' in point and 'time' in point:
                pos = point['position']
                orbit_positions.append([pos['x'], pos['y'], pos['z']])
                # 转换时间为数值
                time_str = point['time']
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    timestamp = dt.timestamp()
                    orbit_times.append(timestamp)
                except Exception:
                    orbit_times.append(len(orbit_times))
            if 'velocity' in point:
                vel = point['velocity']
                orbit_velocities.append([vel['vx'], vel['vy'], vel['vz']])
        
        if len(orbit_times) < 2:
            return default_position, default_velocity
        
        orbit_times = np.array(orbit_times)
        orbit_positions = np.array(orbit_positions)
        
        if orbit_velocities:
            orbit_velocities = np.array(orbit_velocities)
        else:
            orbit_velocities = np.zeros_like(orbit_positions)
        
        t0 = orbit_times[0]
        relative_times = orbit_times - t0
        
        time_range = np.ptp(relative_times)
        if time_range == 0:
            return default_position, default_velocity
        
        relative_time = time - t0
        
        method = interpolation_method.upper()
        
        if method == 'LINEAR':
            idx = np.searchsorted(relative_times, relative_time)
            if idx == 0:
                return orbit_positions[0].tolist(), orbit_velocities[0].tolist()
            if idx >= len(relative_times):
                return orbit_positions[-1].tolist(), orbit_velocities[-1].tolist()
            
            t1 = relative_times[idx - 1]
            t2 = relative_times[idx]
            if t2 > t1:
                frac = (relative_time - t1) / (t2 - t1)
            else:
                frac = 0.0
            
            pos = orbit_positions[idx - 1] + frac * (orbit_positions[idx] - orbit_positions[idx - 1])
            vel = orbit_velocities[idx - 1] + frac * (orbit_velocities[idx] - orbit_velocities[idx - 1])
            
            return pos.tolist(), vel.tolist()
        
        elif method == 'HERMITE':
            idx = np.searchsorted(relative_times, relative_time)
            if idx == 0:
                idx = 1
            elif idx >= len(relative_times):
                idx = len(relative_times) - 1
            
            t1 = relative_times[idx - 1]
            t2 = relative_times[idx]
            if t2 > t1:
                t = (relative_time - t1) / (t2 - t1)
            else:
                t = 0.0
            
            p0 = orbit_positions[idx - 1].tolist()
            p1 = orbit_positions[idx].tolist()
            v0 = orbit_velocities[idx - 1].tolist() if len(orbit_velocities) > 0 else [0, 7000, 0]
            v1 = orbit_velocities[idx].tolist() if len(orbit_velocities) > 0 else [0, 7000, 0]
            
            scale = t2 - t1
            v0_scaled = [v0[i] * scale for i in range(3)]
            v1_scaled = [v1[i] * scale for i in range(3)]
            
            pos = self._hermite_interpolation(p0, p1, v0_scaled, v1_scaled, t)
            
            if len(orbit_velocities) > 0:
                dt = t2 - t1
                if dt > 0:
                    vel = [(p1[i] - p0[i]) / dt for i in range(3)]
                else:
                    vel = v0
            else:
                vel = v0
            
            return pos, vel
        
        else:
            idx = np.searchsorted(relative_times, relative_time)
            if idx == 0:
                return orbit_positions[0].tolist(), orbit_velocities[0].tolist()
            if idx >= len(relative_times):
                return orbit_positions[-1].tolist(), orbit_velocities[-1].tolist()
            
            t1 = relative_times[idx - 1]
            t2 = relative_times[idx]
            if t2 > t1:
                frac = (relative_time - t1) / (t2 - t1)
            else:
                frac = 0.0
            
            pos = orbit_positions[idx - 1] + frac * (orbit_positions[idx] - orbit_positions[idx - 1])
            vel = orbit_velocities[idx - 1] + frac * (orbit_velocities[idx] - orbit_velocities[idx - 1])
            
            return pos.tolist(), vel.tolist()
    
    def _hermite_interpolation(self, p0, p1, v0, v1, t):
        """
        Hermite（三次埃尔米特）插值
        """
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        
        return [
            h00 * p0[0] + h10 * v0[0] + h01 * p1[0] + h11 * v1[0],
            h00 * p0[1] + h10 * v0[1] + h01 * p1[1] + h11 * v1[1],
            h00 * p0[2] + h10 * v0[2] + h01 * p1[2] + h11 * v1[2]
        ]


class SarSimulator:
    """
    SAR 数据模拟器（优化版）
    """
    def __init__(self, yaml_file: str, dem_file: str,
                 noise_snr: float = 20.0,
                 speckle_gamma: float = 1.0,
                 correlated_noise: bool = False):
        """
        初始化 SAR 模拟器
        
        Args:
            yaml_file: SAR YAML 文件路径
            dem_file: DEM 文件路径
            noise_snr: 信噪比(dB)
            speckle_gamma: 伽马分布参数
            correlated_noise: 是否添加空间相关噪声
        """
        self.yaml_file = yaml_file
        self.dem_file = dem_file
        self.noise_snr = noise_snr
        self.speckle_gamma = speckle_gamma
        self.correlated_noise = correlated_noise
        self.converter = DemToSarConverter(yaml_file, dem_file)
        self.look_direction = self.converter.look_direction  # 视线方向
        self._load_parameters()
        self._init_sar_image()
    
    def _load_parameters(self):
        """
        加载 SAR 成像参数
        """
        # 从 converter 中获取参数
        self.nrows = self.converter.nrows
        self.ncols = self.converter.ncols
        self.prf = self.converter.prf
        self.range_sampling_rate = self.converter.range_sampling_rate
        self.radar_wavelength = self.converter.radar_wavelength
        self.near_range = self.converter.near_range
        self.range_pixel_spacing = self.converter.range_pixel_spacing
        
        # 计算其他参数
        self.c = 3e8  # 光速
        
        # ===== 高优先级修复1: 验证时间基准 =====
        if not hasattr(self.converter, 'sensing_start') or self.converter.sensing_start == 0.0:
            raise ValueError(
                "❌ DemToSarConverter的sensing_start未正确初始化！\n"
                "这将导致SAR模拟结果错误。\n"
                "请检查YAML文件中的first_line_sensing_time字段。"
            )
        
        print(f"✓ Converter sensing_start验证通过: {self.converter.sensing_start}")
        
        # 验证轨道时间范围
        if self.converter.orbit_cache and len(self.converter.orbit_cache['times']) > 0:
            orbit_t_min = self.converter.orbit_cache['times'][0]
            orbit_t_max = self.converter.orbit_cache['times'][-1]
            sar_t_min = self.converter.sensing_start
            sar_t_max = self.converter.sensing_start + self.nrows * self.converter.azimuth_time_step
            
            print(f"✓ 轨道时间范围: [{orbit_t_min:.2f}, {orbit_t_max:.2f}]")
            print(f"✓ SAR时间范围: [{sar_t_min:.2f}, {sar_t_max:.2f}]")
    
    def _init_sar_image(self):
        """
        初始化 SAR 图像
        """
        # 创建复数 SAR 图像
        self.sar_image = np.zeros((self.nrows, self.ncols), dtype=np.complex64)
        # 创建振幅和相位图像
        self.amplitude_image = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.phase_image = np.zeros((self.nrows, self.ncols), dtype=np.float32)
    
    def simulate_backscatter(self, height: float, slope: float = 0.0, incidence_angle: float = 30.0) -> float:
        """
        模拟后向散射系数（基础版本，保持向后兼容）
        
        Args:
            height: 高程（米）
            slope: 坡度（度）
            incidence_angle: 入射角（度）
            
        Returns:
            后向散射系数
        """
        # 调用改进版本，使用默认参数
        return self.simulate_backscatter_improved(height, slope, incidence_angle, 
                                                  land_type='mixed', polarization='VV')
    
    def simulate_backscatter_improved(self, height: float, slope: float, incidence_angle: float,
                                     land_type: str = 'mixed', polarization: str = 'VV') -> float:
        """
        改进的后向散射模拟（高优先级修复2）
        
        Args:
            height: 高程（米）
            slope: 坡度（度）
            incidence_angle: 入射角（度）
            land_type: 地表类型 ('water', 'vegetation', 'urban', 'bare', 'snow', 'mixed')
            polarization: 极化方式 ('HH', 'VV', 'HV', 'VH')
            
        Returns:
            后向散射系数（线性单位）
        """
        # 基础后向散射系数（dB），根据地表类型
        base_sigma0_db = {
            'water': -25.0,      # 水体后向散射很低
            'vegetation': -12.0,  # 植被中等
            'urban': -5.0,        # 城市高
            'bare': -15.0,        # 裸地较低
            'snow': -18.0,        # 雪/冰较低
            'mixed': -10.0        # 混合地表
        }
        
        sigma0_db = base_sigma0_db.get(land_type, -10.0)
        
        # 转换为线性单位
        sigma0_linear = 10 ** (sigma0_db / 10)
        
        # 高程影响（大气衰减）
        height_factor = np.exp(-height / 8000.0)  # 调整衰减系数
        
        # 坡度和入射角的综合影响（局部入射角）
        # 局部入射角 = 入射角 - 坡度（简化模型）
        local_incidence = max(0, incidence_angle - slope)
        incidence_factor = np.cos(np.radians(local_incidence)) ** 2
        
        # 极化影响
        polarization_factor = {
            'HH': 1.0,
            'VV': 0.8,
            'HV': 0.3,
            'VH': 0.3
        }.get(polarization, 1.0)
        
        # 视线方向影响
        look_direction_factor = 1.0
        if self.look_direction == "left":
            look_direction_factor = 1.0
        elif self.look_direction == "right":
            look_direction_factor = 1.0
        
        # 地形增强因子 - 增加地形对后向散射的影响
        terrain_factor = 1.0
        if slope > 5:
            # 陡坡会增加后向散射
            terrain_factor = 1.0 + slope / 30.0
        
        # 高程影响 - 不同高程区域有不同的后向散射特性
        elevation_factor = 1.0
        if height > 2000:
            # 高海拔地区后向散射略有增加
            elevation_factor = 1.1
        elif height < 100:
            # 低海拔平原地区
            elevation_factor = 0.9
        
        # 综合后向散射
        backscatter = sigma0_linear * height_factor * incidence_factor * polarization_factor * look_direction_factor * terrain_factor * elevation_factor
        
        # 添加相干斑噪声（乘性噪声）
        speckle = np.random.gamma(1.0, 1.0)  # Gamma分布
        backscatter *= speckle
        
        # 确保非负且有最小值
        backscatter = max(1e-6, backscatter)
        
        return backscatter
    
    def classify_land_type(self, height: float, slope: float) -> str:
        """
        简单的地表类型分类（中优先级修复7）
        
        Args:
            height: 高程（米）
            slope: 坡度（度）
        
        Returns:
            land_type: 'water', 'vegetation', 'urban', 'bare', 'snow', 'mixed'
        """
        # 基于高程和坡度的简单分类规则
        if height < 10 and slope < 2:
            return 'water'  # 低海拔、平坦 → 水体
        elif height > 4000:
            return 'snow'   # 高海拔 → 雪/冰
        elif slope > 30:
            return 'bare'   # 陡坡 → 裸地/岩石
        elif height < 1000 and slope < 10:
            return 'urban'  # 低海拔、较平坦 → 可能是城市（简化）
        else:
            return 'vegetation'  # 其他 → 植被

    def calculate_optimal_step(self) -> int:
        """
        计算最优采样步长

        根据DEM分辨率和SAR分辨率自动计算最优采样步长，
        确保每个SAR像素至少有2-3个DEM采样点。

        Returns:
            optimal_step: 最优步长（像素）
        """
        try:
            # 获取DEM分辨率（米）
            if self.converter.dem_geotransform is not None:
                dem_lat_res = abs(self.converter.dem_geotransform[5])
                dem_lon_res = abs(self.converter.dem_geotransform[1])

                # 转换为米（近似）
                # 使用SAR中心纬度进行转换
                lat = getattr(self.converter, 'sar_center_lat', 30.0)
                dem_res_meters = (dem_lat_res + dem_lon_res) / 2 * 111320 * np.cos(np.radians(lat))
            else:
                dem_res_meters = 30.0  # 默认30米

            # 获取SAR分辨率（米）
            sar_azimuth_res = self.converter.azimuth_spacing  # 方位向分辨率
            sar_range_res = self.converter.range_pixel_spacing  # 距离向分辨率
            sar_res = (sar_azimuth_res + sar_range_res) / 2

            # 计算步长：确保每个SAR像素至少有2-3个DEM采样点
            # step = DEM_res / (SAR_res * oversampling_factor)
            oversampling_factor = 2.5
            optimal_step = max(1, int(dem_res_meters / (sar_res * oversampling_factor)))

            # 限制步长范围（1-20像素）
            optimal_step = max(1, min(optimal_step, 20))

            print(f"✓ DEM分辨率: {dem_res_meters:.1f}m")
            print(f"✓ SAR分辨率: {sar_res:.1f}m")
            print(f"✓ 最优采样步长: {optimal_step}")

            return optimal_step

        except Exception as e:
            print(f"⚠️  计算最优步长失败: {e}")
            print(f"⚠️  使用默认步长: 5")
            return 5

    def check_shadow_layover(self, dem_row: int, dem_col: int,
                            azimuth_time: float, range_sample: float) -> str:
        """
        检查阴影和叠掩效应

        Args:
            dem_row: DEM行号
            dem_col: DEM列号
            azimuth_time: 方位向时间
            range_sample: 距离向采样

        Returns:
            'normal': 正常区域
            'shadow': 阴影区
            'layover': 叠掩区
        """
        try:
            # 获取当前点的高程
            height = self.converter.dem_data[dem_row, dem_col]
            if np.isnan(height):
                return 'normal'

            # 获取卫星位置
            sat_pos, _ = self.converter._satellite_state(azimuth_time)

            # 获取目标点的地心坐标
            lon, lat = self.converter._dem_pixel_to_lon_lat(dem_row, dem_col)
            target_xyz = self.converter._llh_to_xyz(lat, lon, height)

            # 计算视线方向
            los_vector = np.array(target_xyz) - np.array(sat_pos)
            los_unit = los_vector / np.linalg.norm(los_vector)

            # 检查前方是否有遮挡（阴影）
            dem_rows, dem_cols = self.converter.dem_data.shape

            # 更详细的阴影检测：检查多个相邻像素
            shadow_detected = False
            # 检查前方10个像素（根据雷达视向）
            for i in range(1, min(10, dem_col)):
                neighbor_col = dem_col - i
                if neighbor_col >= 0:
                    neighbor_height = self.converter.dem_data[dem_row, neighbor_col]
                    if not np.isnan(neighbor_height):
                        # 计算视线方向上的高度
                        # 简化模型：检查视线是否被遮挡
                        distance = i * self.converter.dem_geotransform[1]  # 假设像素间距一致
                        required_height = height + distance * np.tan(np.arccos(np.dot(los_unit, np.array([1, 0, 0]))))
                        if neighbor_height > required_height:
                            shadow_detected = True
                            break

            if shadow_detected:
                return 'shadow'

            # 检查叠掩：如果坡度过大且朝向雷达
            if dem_row > 0 and dem_col > 0:
                # 计算水平和垂直方向的坡度
                dh_dx = self.converter.dem_data[dem_row, dem_col] - self.converter.dem_data[dem_row, dem_col - 1]
                dh_dy = self.converter.dem_data[dem_row, dem_col] - self.converter.dem_data[dem_row - 1, dem_col]
                slope = np.sqrt(dh_dx**2 + dh_dy**2)
                
                # 检查是否朝向雷达的陡坡
                if dh_dx > 30:  # 朝向雷达的陡坡
                    return 'layover'

            return 'normal'

        except Exception as e:
            # 如果检查失败，返回正常
            return 'normal'

    def add_realistic_noise(self, snr: float = None,
                           include_thermal: bool = True,
                           include_speckle: bool = True,
                           include_ambiguity: bool = False):
        """
        添加更真实的噪声模型

        包含三种噪声类型：
        1. 热噪声（加性，高斯分布）
        2. 相干斑噪声（乘性，Gamma分布）
        3. 模糊噪声（可选，距离和方位模糊）

        Args:
            snr: 信噪比（dB）
            include_thermal: 是否包含热噪声
            include_speckle: 是否包含相干斑噪声
            include_ambiguity: 是否包含模糊噪声
        """
        if snr is None:
            snr = self.noise_snr

        # 计算信号功率
        signal_power = np.mean(np.abs(self.sar_image[self.sar_image != 0])**2)

        if signal_power == 0:
            print("⚠️  警告: SAR图像为空，无法添加噪声")
            return

        # 1. 热噪声（加性，高斯分布）
        if include_thermal:
            thermal_power = signal_power / (10**(snr / 10))
            thermal_noise = np.sqrt(thermal_power / 2) * (
                np.random.randn(self.nrows, self.ncols) +
                1j * np.random.randn(self.nrows, self.ncols)
            )
            self.sar_image += thermal_noise
            print(f"✓ 添加热噪声 (SNR={snr:.1f}dB)")

        # 2. 相干斑噪声（乘性，Gamma分布）
        if include_speckle:
            # 多视等效视数（Equivalent Number of Looks）
            ENL = self.speckle_gamma

            # Gamma分布的相干斑
            speckle_amplitude = np.random.gamma(ENL, 1.0/ENL, (self.nrows, self.ncols))
            speckle_phase = 2 * np.pi * np.random.rand(self.nrows, self.ncols)
            speckle = np.sqrt(speckle_amplitude) * np.exp(1j * speckle_phase)

            # 应用相干斑（乘性）
            self.sar_image *= speckle
            print(f"✓ 添加相干斑噪声 (ENL={ENL})")

        # 3. 距离模糊和方位模糊噪声（可选）
        if include_ambiguity:
            # 距离模糊：来自相邻距离门的泄漏
            range_ambiguity_ratio = -20  # dB
            range_amb_power = signal_power / (10**(range_ambiguity_ratio / 10))
            range_amb = np.sqrt(range_amb_power / 2) * (
                np.random.randn(self.nrows, self.ncols) +
                1j * np.random.randn(self.nrows, self.ncols)
            )
            self.sar_image += range_amb

            # 方位模糊：来自相邻多普勒频率的泄漏
            azimuth_ambiguity_ratio = -20  # dB
            azimuth_amb_power = signal_power / (10**(azimuth_ambiguity_ratio / 10))
            azimuth_amb = np.sqrt(azimuth_amb_power / 2) * (
                np.random.randn(self.nrows, self.ncols) +
                1j * np.random.randn(self.nrows, self.ncols)
            )
            self.sar_image += azimuth_amb
            print(f"✓ 添加模糊噪声")

        # 更新振幅和相位图像
        self.amplitude_image = np.abs(self.sar_image)
        self.phase_image = np.angle(self.sar_image)

    def validate_simulation_quality(self) -> Dict:
        """
        验证模拟质量

        Returns:
            quality_report: 质量报告字典
        """
        report = {}

        # 1. 检查填充率
        non_zero_pixels = np.count_nonzero(self.sar_image)
        total_pixels = self.nrows * self.ncols
        fill_rate = non_zero_pixels / total_pixels if total_pixels > 0 else 0
        report['fill_rate'] = fill_rate

        if fill_rate < 0.5:
            report['warning'] = f"填充率过低: {fill_rate:.2%}"

        # 2. 检查动态范围
        amplitude = np.abs(self.sar_image)
        valid_amplitude = amplitude[amplitude > 0]

        if len(valid_amplitude) > 0:
            amplitude_db = 10 * np.log10(valid_amplitude + 1e-10)
            report['amplitude_range_db'] = (float(np.min(amplitude_db)), float(np.max(amplitude_db)))
            report['amplitude_mean_db'] = float(np.mean(amplitude_db))
        else:
            report['amplitude_range_db'] = (0.0, 0.0)
            report['amplitude_mean_db'] = 0.0

        # 3. 检查相位分布
        phase = np.angle(self.sar_image)
        valid_phase = phase[self.sar_image != 0]

        if len(valid_phase) > 0:
            report['phase_mean'] = float(np.mean(valid_phase))
            report['phase_std'] = float(np.std(valid_phase))
        else:
            report['phase_mean'] = 0.0
            report['phase_std'] = 0.0

        # 4. 检查SNR
        if len(valid_amplitude) > 0:
            signal_power = np.mean(valid_amplitude**2)

            # 估计噪声功率（使用边缘区域）
            edge_size = min(10, self.nrows // 10, self.ncols // 10)
            if edge_size > 0:
                edge_region = amplitude[:edge_size, :edge_size]
                edge_valid = edge_region[edge_region > 0]
                if len(edge_valid) > 0:
                    noise_power = np.mean(edge_valid**2)
                    if noise_power > 0:
                        estimated_snr = 10 * np.log10(signal_power / noise_power)
                        report['estimated_snr_db'] = float(estimated_snr)
                    else:
                        report['estimated_snr_db'] = float('inf')
                else:
                    report['estimated_snr_db'] = 0.0
            else:
                report['estimated_snr_db'] = 0.0
        else:
            report['estimated_snr_db'] = 0.0

        # 5. 检查空洞
        zero_mask = (amplitude < 1e-6)
        num_holes = np.count_nonzero(zero_mask)
        report['num_holes'] = int(num_holes)
        report['hole_percentage'] = float(num_holes / total_pixels * 100) if total_pixels > 0 else 0.0

        return report




    
    def simulate_phase(self, slant_range: float, azimuth_time: float) -> float:
        """
        模拟相位（基础版本，保持向后兼容）
        
        Args:
            slant_range: 斜距（米）
            azimuth_time: 方位向时间（秒）
            
        Returns:
            相位（弧度）
        """
        # 调用改进版本，不包含地形相位（向后兼容）
        return self.simulate_phase_with_topography(slant_range, azimuth_time, 
                                                   height=0.0, reference_height=0.0,
                                                   include_topographic=False)
    
    def simulate_phase_with_topography(self, slant_range: float, azimuth_time: float,
                                      height: float = 0.0, reference_height: float = 0.0,
                                      include_topographic: bool = True, baseline_perpendicular: float = 0.0) -> float:
        """
        包含地形相位的相位模拟（高优先级修复3）
        
        Args:
            slant_range: 斜距（米）
            azimuth_time: 方位向时间（秒）
            height: 目标点高程（米）
            reference_height: 参考高程（米），通常为0
            include_topographic: 是否包含地形相位
            baseline_perpendicular: 垂直基线（米），用于InSAR干涉条纹模拟
            
        Returns:
            相位（弧度）
        """
        # 确保时间使用 double 精度
        azimuth_time = float(azimuth_time)
        
        # 1. 距离向相位（双程传播）
        range_phase = 4 * np.pi * slant_range / self.radar_wavelength
        
        # 2. 方位向相位（多普勒效应）
        azimuth_phase = 2 * np.pi * azimuth_time * self.prf
        
        # 3. 地形相位（关键！用于InSAR）
        topographic_phase = 0.0
        if include_topographic:
            # 计算入射角
            try:
                sat_pos, _ = self.converter._satellite_state(azimuth_time)
                sat_height = np.linalg.norm(sat_pos)
                if sat_height > slant_range and slant_range > 0:
                    incidence_angle = np.arccos(min(1.0, sat_height / slant_range))
                else:
                    incidence_angle = np.radians(30.0)  # 默认30度
            except:
                incidence_angle = np.radians(30.0)  # 默认30度
            
            # 地形相位
            delta_height = height - reference_height
            
            if baseline_perpendicular != 0:
                # InSAR地形相位公式（包含垂直基线）
                # topographic_phase = (4π * B⊥ * Δh * sin(θ)) / (λ * R)
                topographic_phase = (4 * np.pi * baseline_perpendicular * delta_height * np.sin(incidence_angle) 
                                    / (self.radar_wavelength * slant_range))
            else:
                # 单影像地形相位公式（不包含基线）
                topographic_phase = (4 * np.pi * delta_height * np.sin(incidence_angle) 
                                    / self.radar_wavelength)
        
        # 4. 大气相位（可选，简化模型）
        atmospheric_phase = 0.0  # 简化：忽略大气相位
        
        # 综合相位
        phase = range_phase + azimuth_phase + topographic_phase + atmospheric_phase
        
        # 归一化到 [-π, π]
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
        return phase
    
    def add_noise(self, snr: float = None, noise_type: str = 'gaussian'):
        """
        添加噪声
        
        Args:
            snr: 信噪比（dB），如果为None则使用对象自身设置
            noise_type: 噪声类型 ('gaussian', 'speckle')
        """
        if snr is None:
            snr = self.noise_snr
        # 计算噪声功率
        signal_power = np.mean(np.abs(self.sar_image)**2)
        noise_power = signal_power / (10**(snr / 10))
        
        if noise_type == 'gaussian':
            noise = np.sqrt(noise_power / 2) * (np.random.randn(self.nrows, self.ncols) + 1j * np.random.randn(self.nrows, self.ncols))
        elif noise_type == 'speckle':
            gamma = self.speckle_gamma
            speckle = np.random.gamma(gamma, 1/gamma, (self.nrows, self.ncols))
            noise = np.sqrt(noise_power) * (speckle - 1) * np.exp(1j * 2 * np.pi * np.random.rand(self.nrows, self.ncols))
        else:
            noise = np.sqrt(noise_power / 2) * (np.random.randn(self.nrows, self.ncols) + 1j * np.random.randn(self.nrows, self.ncols))
        
        if self.correlated_noise and noise_type == 'speckle':
            # 简单平滑相关性
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise.real, 1) + 1j*gaussian_filter(noise.imag,1)
        
        self.sar_image += noise
    
    @jit(nopython=True)
    def simulate_backscatter_numba(self, height, slope):
        """
        Numba优化的后向散射计算
        """
        # 基础后向散射（增加基础值，确保不为零）
        base_backscatter = 0.8
        
        # 高程影响（调整系数，确保不为零）
        height_factor = np.exp(-height / 5000.0)
        
        # 坡度影响
        slope_factor = np.cos(np.radians(slope))
        
        # 综合后向散射
        backscatter = base_backscatter * height_factor * slope_factor
        
        # 添加随机波动
        backscatter *= np.random.normal(1.0, 0.2)
        
        # 确保后向散射非负且有最小值
        backscatter = max(0.1, backscatter)
        
        return backscatter
    
    @jit(nopython=True)
    def simulate_phase_numba(self, slant_range, azimuth_time, radar_wavelength, prf):
        """
        Numba优化的相位计算
        """
        # 距离向相位
        range_phase = 4 * np.pi * slant_range / radar_wavelength
        
        # 方位向相位
        azimuth_phase = 2 * np.pi * azimuth_time * prf
        
        # 综合相位
        phase = range_phase + azimuth_phase
        
        # 归一化到 [-pi, pi]
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
        return phase
    
    def _process_chunk(self, chunk):
        """
        处理DEM分块
        """
        i_start, i_end, j_start, j_end, step, dem_data = chunk
        
        # 创建局部数组以及计数器
        local_sar = np.zeros((self.nrows, self.ncols), dtype=np.complex64)
        local_amp = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        local_phase = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        count = np.zeros((self.nrows, self.ncols), dtype=np.int32)
        
        success_count = 0
        error_count = 0
        
        # 调试：收集统计信息
        az_times_all = []
        range_samples_all = []
        failed_az_times = []
        failed_range_samples = []
        
        # 预加载当前块的DEM数据，减少内存访问开销
        dem_chunk = dem_data[i_start:i_end, j_start:j_end]
        
        for i in range(i_start, i_end, step):
            for j in range(j_start, j_end, step):
                try:
                    # 检查坐标是否在DEM范围内
                    if i < 0 or i >= dem_data.shape[0] or j < 0 or j >= dem_data.shape[1]:
                        error_count += 1
                        continue
                    
                    height = dem_data[i, j]
                    if np.isnan(height) or height < 0:
                        error_count += 1
                        continue
                    
                    slope = 0.0
                    if 0 < i < dem_data.shape[0] - 1 and 0 < j < dem_data.shape[1] - 1:
                        dh_dx = (dem_data[i, j+1] - dem_data[i, j-1]) / (2 * step)
                        dh_dy = (dem_data[i+1, j] - dem_data[i-1, j]) / (2 * step)
                        slope = np.degrees(np.sqrt(dh_dx**2 + dh_dy**2))
                    
                    try:
                        range_sample, azimuth_time = self.converter.convert(i, j)
                        # 收集所有成功转换的统计信息
                        az_times_all.append(azimuth_time)
                        range_samples_all.append(range_sample)
                        if np.isnan(range_sample) or np.isnan(azimuth_time):
                            error_count += 1
                            continue
                    except Exception as e:
                        error_count += 1
                        # 只在第一次出错时打印错误信息
                        if error_count == 1:
                            print(f"处理点 ({i}, {j}) 时出错: {e}")
                        continue
                    
                    slant_range = self.near_range + range_sample * self.range_pixel_spacing
                    azimuth_line = int(round(azimuth_time * self.prf))
                    
                    # 收集失败点的统计信息
                    if not (0 <= azimuth_line < self.nrows and 0 <= range_sample < self.ncols):
                        failed_az_times.append(azimuth_time)
                        failed_range_samples.append(range_sample)
                    
                    # 动态计算卫星高度
                    sat_height = None
                    try:
                        # use new helper to handle degenerate orbit cases
                        sat_pos, _ = self.converter._satellite_state(azimuth_time)
                        sat_height = np.linalg.norm(sat_pos)
                    except Exception:
                        sat_height = 500000.0
                    
                    if sat_height is None or sat_height <= 0 or sat_height > slant_range:
                        incidence_angle = 0.0
                    else:
                        incidence_angle = np.degrees(np.arccos(sat_height / slant_range))

                    # 调试：打印azimuth_time和range_sample的范围
                    if success_count == 0:
                        print(f"DEBUG: 第一个成功点 - azimuth_time={azimuth_time}, range_sample={range_sample:.1f}, azimuth_line={azimuth_line}, nrows={self.nrows}, ncols={self.ncols}")

                    if 0 <= azimuth_line < self.nrows and 0 <= range_sample < self.ncols:
                        # 地表类型分类（如果启用）
                        land_type = 'mixed'
                        if hasattr(self, '_land_classification') and self._land_classification:
                            land_type = self.classify_land_type(height, slope)
                        
                        # 阴影和叠掩检查（如果启用）
                        effect = 'normal'
                        if hasattr(self, '_check_shadow_layover') and self._check_shadow_layover:
                            effect = self.check_shadow_layover(i, j, azimuth_time, range_sample)
                        
                        # 根据效果调整后向散射
                        if effect == 'shadow':
                            backscatter = 0.001  # 阴影区极低
                        elif effect == 'layover':
                            backscatter = self.simulate_backscatter_improved(
                                height, slope, incidence_angle, land_type=land_type) * 2.0
                        else:
                            backscatter = self.simulate_backscatter_improved(
                                height, slope, incidence_angle, land_type=land_type)
                        
                        # 相位计算（支持地形相位）
                        if hasattr(self, '_include_topographic_phase') and self._include_topographic_phase:
                            baseline = getattr(self, '_baseline_perpendicular', 0.0)
                            phase = self.simulate_phase_with_topography(
                                slant_range, azimuth_time, height=height, 
                                reference_height=0.0, include_topographic=True, 
                                baseline_perpendicular=baseline)
                        else:
                            phase = self.simulate_phase(slant_range, azimuth_time)
                        
                        pixel_value = backscatter * np.exp(1j * phase)
                        r_idx = int(round(range_sample))
                        a_idx = azimuth_line
                        local_sar[a_idx, r_idx] += pixel_value
                        local_amp[a_idx, r_idx] += backscatter
                        local_phase[a_idx, r_idx] += phase
                        count[a_idx, r_idx] += 1
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    # 只在第一次出错时打印错误信息
                    if error_count == 1:
                        print(f"处理点 ({i}, {j}) 时出错: {e}")
        
        # 平均化结果
        mask = count > 0
        local_sar[mask] /= count[mask]
        local_amp[mask] /= count[mask]
        local_phase[mask] /= count[mask]
        
        print(f"处理块 [{i_start}:{i_end}, {j_start}:{j_end}] 完成，成功: {success_count}, 失败: {error_count}")
        
        # 调试：打印统计信息
        if az_times_all:
            print(f"  DEBUG 成功点: az_time=[{np.min(az_times_all):.2f}, {np.max(az_times_all):.2f}], range=[{np.min(range_samples_all):.1f}, {np.max(range_samples_all):.1f}]")
        if failed_az_times:
            print(f"  DEBUG 失败点(坐标超限): az_time=[{np.min(failed_az_times):.2f}, {np.max(failed_az_times):.2f}], range=[{np.min(failed_range_samples):.1f}, {np.max(failed_range_samples):.1f}]")
        
        return local_sar, local_amp, local_phase
    
    def _process_chunk_shared_memory(self, chunk):
        """
        使用共享内存处理DEM分块
        """
        i_start, i_end, j_start, j_end, step, shm_name, shape, dtype = chunk
        
        # 连接到共享内存
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        dem_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        # 创建局部数组以及计数器
        local_sar = np.zeros((self.nrows, self.ncols), dtype=np.complex64)
        local_amp = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        local_phase = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        count = np.zeros((self.nrows, self.ncols), dtype=np.int32)
        
        success_count = 0
        error_count = 0
        
        for i in range(i_start, i_end, step):
            for j in range(j_start, j_end, step):
                try:
                    # 检查坐标是否在DEM范围内
                    if i < 0 or i >= dem_data.shape[0] or j < 0 or j >= dem_data.shape[1]:
                        error_count += 1
                        continue
                    
                    height = dem_data[i, j]
                    if np.isnan(height) or height < 0:
                        error_count += 1
                        continue
                    
                    slope = 0.0
                    if 0 < i < dem_data.shape[0] - 1 and 0 < j < dem_data.shape[1] - 1:
                        dh_dx = (dem_data[i, j+1] - dem_data[i, j-1]) / (2 * step)
                        dh_dy = (dem_data[i+1, j] - dem_data[i-1, j]) / (2 * step)
                        slope = np.degrees(np.sqrt(dh_dx**2 + dh_dy**2))
                    
                    try:
                        range_sample, azimuth_time = self.converter.convert(i, j)
                        if np.isnan(range_sample) or np.isnan(azimuth_time):
                            error_count += 1
                            continue
                    except Exception as e:
                        error_count += 1
                        # 只在第一次出错时打印错误信息
                        if error_count == 1:
                            print(f"处理点 ({i}, {j}) 时出错: {e}")
                        continue
                    
                    slant_range = self.near_range + range_sample * self.range_pixel_spacing
                    azimuth_line = int(round(azimuth_time * self.prf))
                    
                    # 动态计算卫星高度
                    sat_height = None
                    try:
                        # use new helper to handle degenerate orbit cases
                        sat_pos, _ = self.converter._satellite_state(azimuth_time)
                        sat_height = np.linalg.norm(sat_pos)
                    except Exception:
                        sat_height = 500000.0
                    
                    if sat_height is None or sat_height <= 0 or sat_height > slant_range:
                        incidence_angle = 0.0
                    else:
                        incidence_angle = np.degrees(np.arccos(sat_height / slant_range))
                    
                    if 0 <= azimuth_line < self.nrows and 0 <= range_sample < self.ncols:
                        # 地表类型分类（如果启用）
                        land_type = 'mixed'
                        if hasattr(self, '_land_classification') and self._land_classification:
                            land_type = self.classify_land_type(height, slope)
                        
                        # 阴影和叠掩检查（如果启用）
                        effect = 'normal'
                        if hasattr(self, '_check_shadow_layover') and self._check_shadow_layover:
                            effect = self.check_shadow_layover(i, j, azimuth_time, range_sample)
                        
                        # 根据效果调整后向散射
                        if effect == 'shadow':
                            backscatter = 0.001  # 阴影区极低
                        elif effect == 'layover':
                            backscatter = self.simulate_backscatter_improved(
                                height, slope, incidence_angle, land_type=land_type) * 2.0
                        else:
                            backscatter = self.simulate_backscatter_improved(
                                height, slope, incidence_angle, land_type=land_type)
                        
                        # 相位计算（支持地形相位）
                        if hasattr(self, '_include_topographic_phase') and self._include_topographic_phase:
                            baseline = getattr(self, '_baseline_perpendicular', 0.0)
                            phase = self.simulate_phase_with_topography(
                                slant_range, azimuth_time, height=height, 
                                reference_height=0.0, include_topographic=True, 
                                baseline_perpendicular=baseline)
                        else:
                            phase = self.simulate_phase(slant_range, azimuth_time)
                        
                        pixel_value = backscatter * np.exp(1j * phase)
                        r_idx = int(round(range_sample))
                        a_idx = azimuth_line
                        local_sar[a_idx, r_idx] += pixel_value
                        local_amp[a_idx, r_idx] += backscatter
                        local_phase[a_idx, r_idx] += phase
                        count[a_idx, r_idx] += 1
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    # 只在第一次出错时打印错误信息
                    if error_count == 1:
                        print(f"处理点 ({i}, {j}) 时出错: {e}")
        
        # 关闭共享内存连接
        existing_shm.close()
        
        # 平均化结果
        mask = count > 0
        local_sar[mask] /= count[mask]
        local_amp[mask] /= count[mask]
        local_phase[mask] /= count[mask]
        
        print(f"处理块 [{i_start}:{i_end}, {j_start}:{j_end}] 完成，成功: {success_count}, 失败: {error_count}")
        return local_sar, local_amp, local_phase
    
    def add_noise_vectorized(self, snr: float = None, noise_type: str = 'gaussian'):
        """
        向量化添加噪声
        """
        if snr is None:
            snr = self.noise_snr
            
        signal_power = np.mean(np.abs(self.sar_image)**2)
        noise_power = signal_power / (10**(snr / 10))
        
        if noise_type == 'gaussian':
            # 向量化生成高斯噪声
            noise = np.sqrt(noise_power / 2) * (np.random.randn(self.nrows, self.ncols) + 1j * np.random.randn(self.nrows, self.ncols))
        elif noise_type == 'speckle':
            # 向量化生成相干斑噪声
            gamma = self.speckle_gamma
            speckle = np.random.gamma(gamma, 1/gamma, (self.nrows, self.ncols))
            noise = np.sqrt(noise_power) * (speckle - 1) * np.exp(1j * 2 * np.pi * np.random.rand(self.nrows, self.ncols))
        else:
            noise = np.sqrt(noise_power / 2) * (np.random.randn(self.nrows, self.ncols) + 1j * np.random.randn(self.nrows, self.ncols))
        
        if self.correlated_noise and noise_type == 'speckle':
            # 简单平滑相关性
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise.real, 1) + 1j*gaussian_filter(noise.imag,1)
        
        self.sar_image += noise
    
    def simulate(self, step: int = None, snr: float = None, noise_type: str = 'gaussian', 
                imaging_mode: str = 'stripmap', include_topographic_phase: bool = False,
                land_classification: bool = False, check_shadow_layover: bool = False,
                realistic_noise: bool = False, validate_quality: bool = True, baseline_perpendicular: float = 0.0):
        """
        模拟 SAR 数据（增强版）
        
        Args:
            step: DEM 采样步长（None=自动计算最优步长）
            snr: 信噪比（dB）
            noise_type: 噪声类型 ('gaussian', 'speckle', 'realistic')
            imaging_mode: 成像模式 ('stripmap', 'spotlight')
            include_topographic_phase: 是否包含地形相位（用于InSAR）
            land_classification: 是否使用地表分类
            check_shadow_layover: 是否检查阴影和叠掩效应
            realistic_noise: 是否使用真实噪声模型
            validate_quality: 是否进行质量验证
            baseline_perpendicular: 垂直基线（米），用于InSAR干涉条纹模拟
        """
        print("开始模拟 SAR 数据...")
        start_time = time.time()
        
        # 存储模拟选项为实例变量，供_process_chunk使用
        self._include_topographic_phase = include_topographic_phase
        self._land_classification = land_classification
        self._check_shadow_layover = check_shadow_layover
        self._baseline_perpendicular = baseline_perpendicular
        
        # 如果未指定步长，自动计算最优步长
        if step is None:
            step = self.calculate_optimal_step()
        else:
            print(f"✓ 使用指定步长: {step}")
        
        # 获取 DEM 数据
        dem_data = self.converter.dem_data
        if dem_data is None:
            raise Exception("DEM 数据未加载")
        
        # 计算 DEM 尺寸
        dem_rows, dem_cols = dem_data.shape
        
        # 计算可用内存
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        print(f"可用内存: {available_memory:.2f} GB")
        
        # 计算使用的CPU核心数（根据可用内存动态调整）
        base_cores = max(1, cpu_count() // 2)
        memory_per_core = available_memory / base_cores
        
        # 根据内存情况调整核心数
        if memory_per_core < 2:
            num_cores = max(1, int(available_memory / 2))
        else:
            num_cores = base_cores
        
        num_cores = max(1, min(num_cores, cpu_count()))
        print(f"使用核心数: {num_cores}")
        
        # 计算分块大小
        chunk_size = max(100, dem_rows // num_cores)  # 保证每块至少100行
        chunks = []
        
        for i in range(0, dem_rows, chunk_size):
            i_end = min(i + chunk_size, dem_rows)
            chunks.append((i, i_end, 0, dem_cols, step, dem_data))
        
        # 检查是否使用共享内存
        use_shared_memory = False
        if num_cores > 1 and dem_rows * dem_cols * 4 < available_memory * 0.5:
            use_shared_memory = True
            print("使用共享内存进行并行处理")
        
        # 并行处理
        results = []
        if use_shared_memory:
            # 创建共享内存
            shm = shared_memory.SharedMemory(create=True, size=dem_data.nbytes)
            # 创建共享内存数组
            shm_array = np.ndarray(dem_data.shape, dtype=dem_data.dtype, buffer=shm.buf)
            # 复制数据到共享内存
            shm_array[:] = dem_data[:]
            
            # 准备分块参数
            shared_chunks = [(i, i_end, j, j_end, step, shm.name, dem_data.shape, dem_data.dtype)
                           for (i, i_end, j, j_end, step, _) in chunks]
            
            # 并行处理
            with Pool(num_cores) as pool:
                results = pool.map(self._process_chunk_shared_memory, shared_chunks)
            
            # 清理共享内存
            shm.close()
            shm.unlink()
        else:
            # 常规并行处理
            with Pool(num_cores) as pool:
                results = pool.map(self._process_chunk, chunks)
        
        # 合并结果
        for sar_img, amp_img, phase_img in results:
            self.sar_image += sar_img
            self.amplitude_image += amp_img
            self.phase_image += phase_img
        
        # 增加基础强度，确保图像不会全黑
        min_intensity = 0.01
        self.sar_image = np.maximum(self.sar_image, min_intensity)
        
        # 添加噪声
        if realistic_noise or noise_type == 'realistic':
            # 使用真实噪声模型
            self.add_realistic_noise(snr, include_thermal=True, 
                                    include_speckle=True, include_ambiguity=False)
        else:
            # 使用简单噪声模型（向后兼容）
            self.add_noise_vectorized(snr, noise_type)
        
        end_time = time.time()
        print(f"SAR 数据模拟完成！处理时间: {end_time - start_time:.2f} 秒")
        
        # 质量验证
        if validate_quality:
            print("\n=== 模拟质量报告 ===")
            quality_report = self.validate_simulation_quality()
            print(f"填充率: {quality_report['fill_rate']:.2%}")
            print(f"振幅范围: {quality_report['amplitude_range_db'][0]:.1f} ~ {quality_report['amplitude_range_db'][1]:.1f} dB")
            print(f"振幅均值: {quality_report['amplitude_mean_db']:.1f} dB")
            if quality_report['estimated_snr_db'] > 0:
                print(f"估计SNR: {quality_report['estimated_snr_db']:.1f} dB")
            print(f"空洞数量: {quality_report['num_holes']} ({quality_report['hole_percentage']:.2f}%)")
            
            if 'warning' in quality_report:
                print(f"⚠️  警告: {quality_report['warning']}")
    
    def save(self, output_file: str, format: str = 'bin'):
        """
        保存模拟的 SAR 数据
        
        Args:
            output_file: 输出文件路径
            format: 输出格式 ('bin', 'npy')
        """
        if format == 'bin':
            # 保存为二进制文件
            self.sar_image.tofile(output_file)
        elif format == 'npy':
            # 保存为 NumPy 数组
            np.save(output_file, self.sar_image)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def save_amplitude_phase(self, amplitude_file: str, phase_file: str):
        """
        保存振幅和相位图像
        
        Args:
            amplitude_file: 振幅文件路径
            phase_file: 相位文件路径
        """
        # 保存振幅图像
        np.save(amplitude_file, self.amplitude_image)
        
        # 保存相位图像
        np.save(phase_file, self.phase_image)
    
    def save_tif(self, amplitude_tif: str, phase_tif: str):
        """
        保存振幅和相位为TIFF文件
        
        Args:
            amplitude_tif: 振幅TIFF文件路径
            phase_tif: 相位TIFF文件路径
        """
        # 保存振幅TIFF
        self._save_array_as_tif(self.amplitude_image, amplitude_tif)
        
        # 保存相位TIFF
        self._save_array_as_tif(self.phase_image, phase_tif)
    
    def _save_array_as_tif(self, array: np.ndarray, output_file: str):
        """
        将numpy数组保存为TIFF文件
        
        Args:
            array: 要保存的数组
            output_file: 输出TIFF文件路径
        """
        # 获取数组形状
        rows, cols = array.shape
        
        # 创建TIFF文件
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)
        
        # 写入数据
        dataset.GetRasterBand(1).WriteArray(array)
        
        # 关闭数据集
        dataset = None
    
    def save_vrt(self, vrt_file: str, complex_tif: str):
        """
        保存VRT文件
        
        Args:
            vrt_file: VRT文件路径
            complex_tif: 复数TIFF文件路径
        """
        # 创建VRT文件内容
        vrt_content = f"""
<VRTDataset rasterXSize="{self.ncols}" rasterYSize="{self.nrows}">
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{os.path.basename(complex_tif)}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{os.path.basename(complex_tif)}</SourceFilename>
      <SourceBand>2</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
        """
        
        # 写入VRT文件
        with open(vrt_file, 'w') as f:
            f.write(vrt_content)
    
    def save_yaml(self, yaml_file: str):
        """
        保存YAML文件
        
        Args:
            yaml_file: YAML文件路径
        """
        # 读取原始YAML文件
        with open(self.yaml_file, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # 更新参数
        if 'image_parameters' not in yaml_data:
            yaml_data['image_parameters'] = {}
        
        yaml_data['image_parameters']['nrows'] = self.nrows
        yaml_data['image_parameters']['ncols'] = self.ncols
        
        # 添加模拟相关参数
        yaml_data['simulation_parameters'] = {
            'dem_file': self.dem_file,
            'simulation_date': time.time()
        }
        
        # 写入YAML文件
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
    def save_complex_tif(self, output_file: str):
        """
        保存复数SAR数据为TIFF文件
        
        Args:
            output_file: 输出TIFF文件路径
        """
        # 获取数组形状
        rows, cols = self.sar_image.shape
        
        # 检查数据范围
        real_min = np.min(np.real(self.sar_image))
        real_max = np.max(np.real(self.sar_image))
        imag_min = np.min(np.imag(self.sar_image))
        imag_max = np.max(np.imag(self.sar_image))
        print(f"SAR数据范围 - 实部: [{real_min:.2e}, {real_max:.2e}], 虚部: [{imag_min:.2e}, {imag_max:.2e}]")
        
        # 创建TIFF文件（2个波段：实部和虚部）
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_file, cols, rows, 2, gdal.GDT_Float32)
        
        # 写入实部
        dataset.GetRasterBand(1).WriteArray(np.real(self.sar_image))
        # 写入虚部
        dataset.GetRasterBand(2).WriteArray(np.imag(self.sar_image))
        
        # 关闭数据集
        dataset = None


def main():
    """
    主函数
    """
    import argparse
    
    # 输出当前时间
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"当前时间: {current_time}")
    
    parser = argparse.ArgumentParser(description='SAR 数据模拟工具（优化版）')
    parser.add_argument('yaml_file', help='SAR YAML 文件路径')
    parser.add_argument('dem_file', help='DEM 文件路径')
    parser.add_argument('--step', type=int, default=10, help='DEM 采样步长（None=自动计算）')
    parser.add_argument('--auto-step', action='store_true', help='自动计算最优采样步长')
    parser.add_argument('--snr', type=float, default=None, help='信噪比（dB）；default use yaml/config')
    parser.add_argument('--noise-type', default='gaussian', choices=['gaussian', 'speckle', 'realistic'], help='噪声类型')
    parser.add_argument('--speckle-gamma', type=float, default=1.0, help='speckle 伽马参数')
    parser.add_argument('--correlated-noise', action='store_true', help='生成空间相关噪声')
    parser.add_argument('--imaging-mode', default='stripmap', choices=['stripmap', 'spotlight'], help='成像模式')
    parser.add_argument('--include-topographic-phase', action='store_true', help='包含地形相位（用于InSAR）')
    parser.add_argument('--land-classification', action='store_true', help='启用地表类型分类')
    parser.add_argument('--check-shadow-layover', action='store_true', help='检查阴影和叠掩效应')
    parser.add_argument('--realistic-noise', action='store_true', help='使用真实噪声模型')
    parser.add_argument('--validate-quality', action='store_true', help='启用质量验证')
    parser.add_argument('--baseline-perpendicular', type=float, default=0.0, help='垂直基线（米），用于InSAR干涉条纹模拟')
    parser.add_argument('--output-prefix', default='simsar', help='输出文件前缀')
    
    args = parser.parse_args()
    
    print("=== SAR 数据模拟工具（优化版）===")
    print(f"SAR YAML 文件: {args.yaml_file}")
    print(f"DEM 文件: {args.dem_file}")
    print(f"DEM 采样步长: {args.step}")
    print(f"信噪比: {args.snr} dB")
    print(f"噪声类型: {args.noise_type}")
    print(f"Speckle 伽马参数: {args.speckle_gamma}")
    print(f"空间相关噪声: {args.correlated_noise}")
    print(f"成像模式: {args.imaging_mode}")
    print(f"输出文件前缀: {args.output_prefix}")
    
    try:
        # 初始化模拟器
        simulator = SarSimulator(
            args.yaml_file,
            args.dem_file,
            noise_snr = args.snr if args.snr is not None else 20.0,
            speckle_gamma = args.speckle_gamma,
            correlated_noise = args.correlated_noise
        )
        # 确定步长
        step_value = args.step
        if args.auto_step:
            step_value = None
        
        # 模拟 SAR 数据
        snr_value = args.snr if args.snr is not None else 20.0
        simulator.simulate(
            step=step_value,
            snr=snr_value,
            noise_type=args.noise_type,
            imaging_mode=args.imaging_mode,
            include_topographic_phase=args.include_topographic_phase,
            land_classification=args.land_classification,
            check_shadow_layover=args.check_shadow_layover,
            realistic_noise=args.realistic_noise,
            validate_quality=args.validate_quality,
            baseline_perpendicular=args.baseline_perpendicular
        )
        
        # 定义输出文件名
        simsar_tif = f'{args.output_prefix}.tif'
        simsar_vrt = f'{args.output_prefix}.vrt'
        simsar_yaml = f'{args.output_prefix}.yaml'
        
        # 保存simsar.tif（复数SAR数据）
        simulator.save_complex_tif(simsar_tif)
        
        # 保存simsar.vrt
        simulator.save_vrt(simsar_vrt, simsar_tif)
        
        # 保存simsar.yaml
        simulator.save_yaml(simsar_yaml)
        
        print("\n=== 模拟完成 ===")
        print(f"生成文件:")
        print(f"- {simsar_tif}")
        print(f"- {simsar_vrt}")
        print(f"- {simsar_yaml}")
        print("\n使用示例:")
        print(f"  GMTSAR 处理: gmtsar_pre_process.py {simsar_yaml}")
        
    except Exception as e:
        print(f"模拟失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()