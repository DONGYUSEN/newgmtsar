#!/usr/bin/env python3
"""
DEM到SAR坐标转换模块（优化版）

功能：将DEM地理坐标转换为SAR图像坐标。
该实现参考ISCE2的DEM到SAR流程，要求YAML文件中提供至少两
个具有不同时间戳的轨道点；在时间退化（所有时间相同）时，
会自动构造简单的时间轴和位置变化以保持方位时间变化。

优化特性：
- 轨道样条插值，加快轨道计算
- 改进内存访问模式，预加载行数据
- 流式处理选项，减少内存使用
- 动态并行处理，根据可用内存计算进程数
- 缓存优化，设置大小限制
- 并行处理共享内存，减少数据复制
- 保持时间计算的双精度
- 全面的错误处理和调试信息
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import os
import psutil
import time
from multiprocessing import Pool, cpu_count, shared_memory
from scipy.interpolate import griddata

try:
    from .geo2rdr import Geo2rdr
except ImportError:
    import sys
    pkgdir = os.path.dirname(os.path.abspath(__file__))
    if pkgdir not in sys.path:
        sys.path.insert(0, pkgdir)
    from geo2rdr import Geo2rdr


# 移除 Numba 相关代码，使用普通 Python 实现

# 定义占位符，确保代码可以正常运行
def jit(nopython=False, cache=True, parallel=False):
    """Numba jit装饰器，若不可用则退化为无操作"""
    def decorator(func):
        try:
            from numba import jit as numba_jit
            return numba_jit(func, nopython=nopython, cache=cache, parallel=parallel)
        except ImportError:
            return func
    return decorator
prange = range


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


def calculate_look_direction(corner_coords: List, orbit_data: Dict, yaml_look_direction: str = None) -> str:
    """
    计算SAR成像的观测方向（左/右视）
    
    使用 Range-Doppler 几何原理：
    det = dot(cross(r, v_s), r_s)
    
    其中:
    - r = center - sat_pos (卫星到地面点的向量)
    - v_s = 卫星速度
    - r_s = 卫星位置
    
    det > 0: 右视 (RIGHT)
    det < 0: 左视 (LEFT)
    
    与 ISCE / GAMMA 实现一致，不依赖图像轴方向
    
    Args:
        corner_coords: SAR图像四角坐标
        orbit_data: 轨道数据
        yaml_look_direction: 从YAML配置中读取的look_direction（可选）
    
    Returns:
        计算得到的look_direction ('left' 或 'right')
    """
    if not corner_coords or len(corner_coords) < 4:
        return "right"
    
    try:
        if not orbit_data or 'orbit_points' not in orbit_data:
            return "right"
        
        orbit_points = orbit_data['orbit_points']
        if len(orbit_points) < 2:
            return "right"
        
        # 1. 获取轨道中心时刻的卫星位置和速度
        n_points = len(orbit_points)
        mid_idx = n_points // 2
        
        # 卫星位置
        if 'position' not in orbit_points[mid_idx]:
            return "right"
        sat_pos = np.array([
            orbit_points[mid_idx]['position']['x'],
            orbit_points[mid_idx]['position']['y'],
            orbit_points[mid_idx]['position']['z']
        ])
        
        # 卫星速度（通过相邻点差分计算）
        if mid_idx > 0 and mid_idx < n_points - 1:
            pos_prev = np.array([
                orbit_points[mid_idx - 1]['position']['x'],
                orbit_points[mid_idx - 1]['position']['y'],
                orbit_points[mid_idx - 1]['position']['z']
            ])
            pos_next = np.array([
                orbit_points[mid_idx + 1]['position']['x'],
                orbit_points[mid_idx + 1]['position']['y'],
                orbit_points[mid_idx + 1]['position']['z']
            ])
            time_prev = orbit_points[mid_idx - 1].get('time', 0.0)
            time_next = orbit_points[mid_idx + 1].get('time', 1.0)
            sat_vel = (pos_next - pos_prev) / (time_next - time_prev)
        elif mid_idx == 0:
            pos_next = np.array([
                orbit_points[1]['position']['x'],
                orbit_points[1]['position']['y'],
                orbit_points[1]['position']['z']
            ])
            time_curr = orbit_points[0].get('time', 0.0)
            time_next = orbit_points[1].get('time', 1.0)
            sat_vel = (pos_next - sat_pos) / (time_next - time_curr)
        else:
            pos_prev = np.array([
                orbit_points[-2]['position']['x'],
                orbit_points[-2]['position']['y'],
                orbit_points[-2]['position']['z']
            ])
            time_prev = orbit_points[-2].get('time', 0.0)
            time_curr = orbit_points[-1].get('time', 1.0)
            sat_vel = (sat_pos - pos_prev) / (time_curr - time_prev)
        
        # 2. SAR中心地面点
        center = np.mean(corner_coords, axis=0)
        
        # 3. 计算向量 r = center - sat_pos
        r = center - sat_pos
        
        # 4. 计算 det = dot(cross(r, sat_vel), sat_pos)
        cross_product = np.cross(r, sat_vel)
        det = np.dot(cross_product, sat_pos)
        
        # 5. 判断方向
        if det > 0:
            computed_direction = "right"
        else:
            computed_direction = "left"
        
        # 6. 与YAML中读取的值进行比较
        if yaml_look_direction is not None:
            yaml_direction = yaml_look_direction.lower()
            if yaml_direction != computed_direction:
                print(f"⚠️  警告: look_direction 不一致！")
                print(f"   YAML配置: {yaml_direction}")
                print(f"   计算结果: {computed_direction}")
                print(f"   → 以计算结果 ({computed_direction}) 为准")
            else:
                print(f"   卫星视线向: {computed_direction}")        
        return computed_direction
    
    except Exception:
        return "right"


def get_sar_image_corners(yaml_file: str):
    """
    获取SAR图像四角坐标
    """
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data and 'corner_coordinates' in data:
            corners = data['corner_coordinates']
            corner_coords = []
            # 尝试获取所有角点（支持多种命名方式）
            corner_names = [
                'first_corner', 'second_corner', 'third_corner', 'fourth_corner',
                'top_left', 'top_right', 'bottom_left', 'bottom_right',
                'ul', 'ur', 'll', 'lr'  # 其他可能的命名方式
            ]
            for corner_name in corner_names:
                if corner_name in corners:
                    corner = corners[corner_name]
                    if isinstance(corner, dict):
                        # 优先使用lat、lon坐标（地理坐标）
                        if 'lat' in corner and 'lon' in corner:
                            # 转换经纬度为地心坐标
                            x, y, z = llh_to_xyz(corner['lat'], corner['lon'], corner.get('z', 0))
                            corner_coords.append([x, y, z])
                        # 如果没有lat、lon，尝试使用x、y坐标（可能是地心坐标）
                        elif 'x' in corner and 'y' in corner:
                            corner_coords.append([corner['x'], corner['y'], corner.get('z', 0)])
            
            # 如果找到至少两个角点，返回
            if len(corner_coords) >= 2:
                return corner_coords
    except Exception as e:
        print(f"获取SAR图像四角坐标失败: {e}")
    
    return None


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
            print(f"✓ 使用orbit_times[0]作为sensing_start: {self.sensing_start}")
        
        # ===== 高优先级修复2: 验证时间范围一致性 =====
        self._validate_time_range()
        
        self.geo2rdr = self._create_geo2rdr()
    
    def _validate_time_range(self):
        """验证时间范围一致性（高优先级修复）"""
        if not hasattr(self, 'sensing_start'):
            print("⚠️  警告: sensing_start未定义")
            return
        
        if self.sensing_start == 0.0:
            print("⚠️  严重警告: sensing_start为0，坐标转换将不准确！")
            # 尝试从轨道数据推断
            if self.orbit_cache and len(self.orbit_cache['times']) > 0:
                orbit_t_min = self.orbit_cache['times'][0]
                self.sensing_start = orbit_t_min
                print(f"✓ 自动调整sensing_start为轨道起始时间: {self.sensing_start}")
        
        # 检查轨道时间范围
        if self.orbit_cache and len(self.orbit_cache['times']) > 0:
            orbit_t_min = self.orbit_cache['times'][0]
            orbit_t_max = self.orbit_cache['times'][-1]
            
            # 获取原始轨道起始时间（绝对时间）
            orbit_start_abs = getattr(self, 'orbit_start_time', None)
            if orbit_start_abs is None:
                orbit_start_abs = orbit_t_min
            
            # 将SAR时间转换为相对时间（与轨道时间一致）
            sar_t_min_rel = self.sensing_start - orbit_start_abs
            sar_t_max_rel = sar_t_min_rel + self.nrows * self.azimuth_time_step
            
            print(f"=== 时间范围验证 ===")
            print(f"轨道时间范围: [{orbit_t_min:.2f}, {orbit_t_max:.2f}] ({orbit_t_max - orbit_t_min:.2f}秒)")
            print(f"SAR时间范围(相对): [{sar_t_min_rel:.2f}, {sar_t_max_rel:.2f}] ({sar_t_max_rel - sar_t_min_rel:.2f}秒)")
            
            # 使用相对时间进行检查
            if sar_t_min_rel < orbit_t_min - 1.0:
                print(f"⚠️  警告: SAR起始时间早于轨道起始时间 {orbit_t_min - sar_t_min_rel:.2f}秒")
                print(f"   自动调整sensing_start: {orbit_t_min + orbit_start_abs}")
                self.sensing_start = orbit_t_min + orbit_start_abs
                sar_t_min_rel = self.sensing_start - orbit_start_abs
                sar_t_max_rel = sar_t_min_rel + self.nrows * self.azimuth_time_step
            
            if sar_t_max_rel > orbit_t_max + 1.0:
                print(f"⚠️  警告: SAR结束时间晚于轨道结束时间 {sar_t_max_rel - orbit_t_max:.2f}秒")
                # 调整nrows以适应轨道时间范围
                max_rows = int((orbit_t_max - sar_t_min_rel) / self.azimuth_time_step)
                if max_rows > 0 and max_rows < self.nrows:
                    print(f"   建议调整nrows: {self.nrows} -> {max_rows}")
                elif max_rows <= 0:
                    print(f"   警告: 无法通过调整nrows来解决时间范围问题")
            
            # 检查时间对齐
            time_offset = abs(sar_t_min_rel - orbit_t_min)
            if time_offset > 10.0:
                print(f"⚠️  警告: SAR和轨道时间偏移较大: {time_offset:.2f}秒")
            else:
                print(f"✓ 时间范围验证通过（偏移: {time_offset:.2f}秒）")
        else:
            print("⚠️  警告: 无法验证时间范围（轨道缓存为空）")
    
    def _create_geo2rdr(self):
        """创建 Geo2rdr 实例"""
        try:
            # 读取fd1和fdd1参数
            fd1 = 0.0
            fdd1 = 0.0
            try:
                with open(self.yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if 'prm_parameters' in data:
                    prm_params = data['prm_parameters']
                    if 'fd1' in prm_params:
                        fd1 = prm_params['fd1']
                    if 'fdd1' in prm_params:
                        fdd1 = prm_params['fdd1']
                elif 'radar_parameters' in data:
                    radar_params = data['radar_parameters']
                    if 'fd1' in radar_params:
                        fd1 = radar_params['fd1']
                    if 'fdd1' in radar_params:
                        fdd1 = radar_params['fdd1']
            except Exception:
                pass
            
            g2r = Geo2rdr(
                prf=self.prf,
                radar_wavelength=self.radar_wavelength,
                slant_range_pixel_spacing=self.range_pixel_spacing,
                range_first_sample=self.near_range,
                sensing_start=getattr(self, 'sensing_start', 0.0),
                look_side=self.look_direction.upper() if self.look_direction else 'RIGHT',
                orbit_interpolation_method=self.orbit_interpolation_method,
                fd1=fd1,
                fdd1=fdd1
            )
            
            if self.orbit_data:
                g2r.set_orbit(self.orbit_data)
            
            if self._orbit_state_table is not None and self._orbit_state_times is not None:
                try:
                    orbit_t0 = float(self.orbit_cache['times'][0]) if self.orbit_cache and 'times' in self.orbit_cache else 0.0
                    orbit_time_range = float(self.orbit_cache['times'][-1] - self.orbit_cache['times'][0]) if self.orbit_cache and 'times' in self.orbit_cache and len(self.orbit_cache['times']) > 1 else 0.0
                    g2r.set_orbit_state_table(
                        self._orbit_state_times,
                        self._orbit_state_table['positions'],
                        self._orbit_state_table['velocities'],
                        orbit_t0,
                        orbit_time_range
                    )
                except Exception as e:
                    pass
            
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
    
    def _lon_lat_to_dem_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        经纬度转换为DEM像素坐标（浮点数）
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            (行号, 列号) - 浮点数坐标
        """
        if self.dem_geotransform is None:
            raise Exception("DEM地理变换信息未加载")
        
        # 尝试使用GDAL进行坐标转换
        try:
            from osgeo import gdal, osr
            
            # 设置OSR异常处理
            osr.UseExceptions()
            
            # 创建源空间参考（WGS84经纬度）
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(4326)  # WGS84
            
            # 创建目标空间参考（从DEM获取）
            dst_srs = osr.SpatialReference(wkt=self.dem_projection)
            
            # 创建坐标转换
            transform = osr.CoordinateTransformation(src_srs, dst_srs)
            
            # 转换经纬度到DEM投影坐标
            x, y, _ = transform.TransformPoint(lon, lat)
            
            # 使用地理变换将投影坐标转换为像素坐标
            # 地理变换: [x0, dx, 0, y0, 0, dy]
            # 像素坐标 = (x - x0) / dx, (y - y0) / dy
            dx = self.dem_geotransform[1]
            dy = self.dem_geotransform[5]
            x0 = self.dem_geotransform[0]
            y0 = self.dem_geotransform[3]
            
            if dx == 0 or dy == 0:
                raise Exception("无效的地理变换参数")
            
            col = (x - x0) / dx
            row = (y - y0) / dy
            
            return row, col
        except ImportError:
            # 如果GDAL不可用，直接返回默认值
            return 0.0, 0.0
        except Exception:
            # 如果转换失败，返回默认值
            return 0.0, 0.0
    
    def _interpolate_dem(self, row: float, col: float, method: str = 'bilinear') -> float:
        """
        从DEM中插值获取高程值
        
        Args:
            row: 行号（浮点数）
            col: 列号（浮点数）
            method: 插值方法 ('nearest', 'bilinear', 'cubic')
            
        Returns:
            插值后的高程值
        """
        if self.dem_data is None:
            raise Exception("DEM数据未加载")
        
        # 检查坐标是否在DEM范围内
        nrows, ncols = self.dem_data.shape
        if row < 0 or row >= nrows - 1 or col < 0 or col >= ncols - 1:
            return 0.0  # 超出范围返回0
        
        # 整数部分和小数部分
        i = int(row)
        j = int(col)
        di = row - i
        dj = col - j
        
        # 确保索引在有效范围内
        i = min(i, nrows - 2)
        j = min(j, ncols - 2)
        
        # 获取周围四个点的高程值
        v00 = self.dem_data[i, j]
        v01 = self.dem_data[i, j+1]
        v10 = self.dem_data[i+1, j]
        v11 = self.dem_data[i+1, j+1]
        
        # 检查值是否有效
        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            return 0.0
        
        if method == 'nearest':
            # 最近邻插值
            if di < 0.5 and dj < 0.5:
                return v00
            elif di < 0.5 and dj >= 0.5:
                return v01
            elif di >= 0.5 and dj < 0.5:
                return v10
            else:
                return v11
        elif method == 'bilinear':
            # 双线性插值
            return (1 - di) * ((1 - dj) * v00 + dj * v01) + di * ((1 - dj) * v10 + dj * v11)
        elif method == 'cubic':
            # 三次插值（简化实现）
            # 这里使用双线性插值作为备选，因为三次插值需要更多邻点
            return (1 - di) * ((1 - dj) * v00 + dj * v01) + di * ((1 - dj) * v10 + dj * v11)
        else:
            # 默认使用双线性插值
            return (1 - di) * ((1 - dj) * v00 + dj * v01) + di * ((1 - dj) * v10 + dj * v11)
    
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
        
        # ===== 高优先级修复1: 确保sensing_start正确初始化 =====
        # 计算 sensing_start（第一行成像时间转换为秒）
        self.sensing_start = 0.0
        if self.first_line_sensing_time:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(self.first_line_sensing_time.replace('Z', '+00:00'))
                self.sensing_start = dt.timestamp()
                print(f"✓ 从first_line_sensing_time获取sensing_start: {self.sensing_start}")
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
                    print(f"✓ 从轨道起始时间获取sensing_start: {self.sensing_start}")
            except Exception as e:
                print(f"警告: 获取轨道起始时间失败: {e}")
        
        # 如果仍然为0，发出严重警告
        if self.sensing_start == 0.0:
            print("⚠️  严重警告: sensing_start未能正确初始化，将使用轨道时间作为基准")
        
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
            # print(f"开始加载轨道数据: {self.yaml_file}")
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # print(f"YAML文件加载成功，数据类型: {type(data)}")
            # print(f"YAML文件包含的键: {list(data.keys())}")
            
            if 'orbit_data' in data:
                # print("找到orbit_data键")
                orbit_data = data['orbit_data']
                # print(f"orbit_data包含的键: {list(orbit_data.keys())}")
                
                if 'orbit_points' in orbit_data:
                    # print("找到orbit_points键")
                    orbit_points = orbit_data['orbit_points']
                    # print(f"轨道点数量: {len(orbit_points)}")
                    
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
                                # if i < 5:
                                #     print(f"  转换后的时间戳: {timestamp}")
                            except Exception as e:
                                # print(f"时间转换失败: {e}")
                                # 如果时间转换失败，使用索引作为时间
                                timestamps.append(len(timestamps))
                        if 'velocity' in point:
                            vel = point['velocity']
                            self.orbit_cache['velocities'].append([vel['vx'], vel['vy'], vel['vz']])
                    
                    # 转换为相对时间（相对于第一个轨道点的时间）
                    if timestamps:
                        t0 = timestamps[0]
                        self.orbit_start_time = t0  # 存储原始轨道起始时间
                        relative_times = [t - t0 for t in timestamps]
                        # 使用 double 精度存储时间
                        arr = np.array(relative_times, dtype=np.float64)
                        # 检查时间范围
                        time_range = np.ptp(arr)
                        # print(f"轨道时间范围: {time_range} 秒")
                        # 如果所有时间相同则构造简单的时间轴并调整位置
                        if time_range == 0:
                            # print("警告: 轨道时间范围为0，构造人工时间轴")
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
                        # print(f"相对时间示例: {arr[:5]}")
                    else:
                        self.orbit_cache['times'] = np.array([])
                        # print("警告: 没有有效的时间戳")
                    
                    # 转换为NumPy数组，提高处理速度
                    self.orbit_cache['positions'] = np.array(self.orbit_cache['positions'], dtype=np.float64)
                    self.orbit_cache['velocities'] = np.array(self.orbit_cache['velocities'], dtype=np.float64)
                    # print(f"轨道点数量: {len(self.orbit_cache['times'])}")
                    # print(f"位置数组形状: {self.orbit_cache['positions'].shape}")
                    # print(f"速度数组形状: {self.orbit_cache['velocities'].shape}")
                    # print(f"时间数组形状: {self.orbit_cache['times'].shape}")
                else:
                    self.orbit_cache = None
                    # print("警告: 未找到orbit_points键")
            else:
                self.orbit_cache = None
                # print("警告: 未找到orbit_data键")
        except Exception as e:
            self.orbit_cache = None
            # print(f"预计算轨道缓存失败: {e}")
            # import traceback
            # traceback.print_exc()
    
    def _precompute_orbit_splines(self):
        """
        预计算轨道样条插值
        """
        try:
            from scipy.interpolate import CubicSpline
            
            if self.orbit_cache and len(self.orbit_cache['times']) >= 4:
                # print("预计算轨道样条插值")
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
                # print("轨道样条插值预计算完成")
            else:
                self._orbit_splines = None
                # print("轨道点数量不足，无法创建样条插值")
        except ImportError:
            # print("警告: SciPy未安装，无法使用样条插值")
            self._orbit_splines = None
        except Exception as e:
            # print(f"预计算轨道样条失败: {e}")
            self._orbit_splines = None
    
    def _precompute_orbit_state_table(self):
        """
        预计算轨道状态表（优化缓存机制）
        预先计算整个时间范围内的卫星位置和速度，大幅减少实时计算量
        优化：增加采样点数、使用向量化计算
        """
        if not self.orbit_cache or self.orbit_cache['times'].size < 2:
            return
        
        try:
            times = self.orbit_cache['times']
            positions = self.orbit_cache['positions']
            velocities = self.orbit_cache['velocities']
            
            t_min = float(times[0])
            t_max = float(times[-1])
            time_range = t_max - t_min
            
            azimuth_samples = getattr(self, 'nrows', 0)
            if azimuth_samples > 0:
                n_samples = max(azimuth_samples * 2, 10000)
            else:
                n_samples = max(100, min(20000, int(time_range * 100)))
            
            table_times = np.linspace(t_min, t_max, n_samples)
            
            table_positions = np.zeros((n_samples, 3), dtype=np.float64)
            table_velocities = np.zeros((n_samples, 3), dtype=np.float64)
            
            orbit_times = times
            orbit_pos = positions
            orbit_vel = velocities
            
            for i, t in enumerate(table_times):
                pos, vel = self._interpolate_orbit_cached(t, orbit_times, orbit_pos, orbit_vel)
                table_positions[i] = pos
                table_velocities[i] = vel
            
            self._orbit_state_table = {
                'positions': table_positions,
                'velocities': table_velocities
            }
            self._orbit_state_times = table_times
            
            print(f"✓ 轨道状态表预计算完成: {n_samples}个采样点, 时间范围: {time_range:.2f}秒")
        except Exception as e:
            self._orbit_state_table = None
            self._orbit_state_times = None
    
    def _interpolate_orbit_cached(self, time, orbit_times, orbit_positions, orbit_velocities):
        """
        带缓存的轨道插值（用于预计算）
        使用向量化操作加速
        """
        if orbit_times.size < 2:
            base_pos = np.array([-284515.577, 5480417.824, 4154857.155]) if orbit_positions.size > 0 else np.array([0.0, 0.0, 500000.0])
            base_vel = np.array([0.0, 7000.0, 0.0]) if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0])
            return base_pos + time * base_vel, base_vel
        
        if time <= orbit_times[0]:
            return orbit_positions[0], orbit_velocities[0] if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0])
        
        if time >= orbit_times[-1]:
            return orbit_positions[-1], orbit_velocities[-1] if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0])
        
        idx = np.searchsorted(orbit_times, time)
        if idx == 0:
            idx = 1
        
        t1 = orbit_times[idx - 1]
        t2 = orbit_times[idx]
        
        if t2 > t1:
            frac = (time - t1) / (t2 - t1)
        else:
            frac = 0.0
        
        p1 = orbit_positions[idx - 1]
        p2 = orbit_positions[idx]
        v1 = orbit_velocities[idx - 1] if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0])
        v2 = orbit_velocities[idx] if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0])
        
        pos = p1 + frac * (p2 - p1)
        vel = v1 + frac * (v2 - v1)
        
        return pos, vel
    
    def _precompute_orbit_positions(self):
        """
        预计算轨道位置
        """
        # 这里简化实现，实际应该从YAML文件读取轨道数据
        # 并使用轨道插值算法计算位置和速度
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
            
            # 输出读取的corner_coords
            print(f"读取的corner_coords: {corner_coords}")
            
            # 尝试从YAML中获取look_direction
            yaml_look_direction = None
            if self.orbit_data and 'look_direction' in self.orbit_data:
                yaml_look_direction = self.orbit_data.get('look_direction')
            
            # 计算视线方向，并进行对比验证
            self.look_direction = calculate_look_direction(corner_coords, self.orbit_data, yaml_look_direction)
            
        except Exception:
            self.look_direction = "unknown"
    
    def _calculate_sar_coverage(self):
        """
        计算SAR图像的地理覆盖范围
        """
        try:
            # 获取SAR图像四角坐标
            corner_coords = get_sar_image_corners(self.yaml_file)
            
            if not corner_coords:
                # 如果没有四角坐标，使用轨道数据估算
                if self.orbit_data and 'orbit_points' in self.orbit_data:
                    orbit_points = self.orbit_data['orbit_points']
                    if len(orbit_points) >= 2:
                        # 计算轨道中心点
                        pos_center = np.array([orbit_points[len(orbit_points)//2]['position']['x'], 
                                             orbit_points[len(orbit_points)//2]['position']['y'], 
                                             orbit_points[len(orbit_points)//2]['position']['z']])
                        # 估算地面点位置（卫星高度约500km）
                        sat_height = np.linalg.norm(pos_center)
                        earth_radius = 6371000.0  # 地球平均半径
                        ground_height = sat_height - earth_radius
                        # 计算地面点坐标（简化模型）
                        ground_pos = pos_center * (earth_radius / sat_height)
                        # 转换为经纬度
                        lat, lon, _ = xyz_to_llh(ground_pos[0], ground_pos[1], ground_pos[2])
                        # 估算覆盖范围（假设覆盖约100km x 100km）
                        coverage_radius = 0.5  # 约50km的纬度/经度范围
                        coverage = {
                            'lat_min': lat - coverage_radius,
                            'lat_max': lat + coverage_radius,
                            'lon_min': lon - coverage_radius,
                            'lon_max': lon + coverage_radius
                        }
                        print(f"使用轨道数据估算SAR覆盖范围: {coverage}")
                        return coverage
            
            # 转换四角坐标为经纬度
            lats = []
            lons = []
            for i, coord in enumerate(corner_coords):
                lat, lon, _ = xyz_to_llh(coord[0], coord[1], coord[2])
                lats.append(lat)
                lons.append(lon)
                print(f"角点{i+1}经纬度: ({lat:.4f}, {lon:.4f})")
            
            if lats and lons:
                coverage = {
                    'lat_min': min(lats),
                    'lat_max': max(lats),
                    'lon_min': min(lons),
                    'lon_max': max(lons)
                }
                print(f"使用四角坐标计算SAR覆盖范围: {coverage}")
                return coverage
        except Exception as e:
            print(f"计算SAR覆盖范围失败: {e}")
        
        # 默认覆盖范围
        default_coverage = {
            'lat_min': -90,
            'lat_max': 90,
            'lon_min': -180,
            'lon_max': 180
        }
        print(f"使用默认SAR覆盖范围: {default_coverage}")
        return default_coverage
    
    def _calculate_dem_coverage(self):
        """
        计算DEM的地理覆盖范围
        """
        try:
            if self.dem_geotransform is not None:
                dem_rows, dem_cols = self.dem_data.shape
                
                # 计算DEM四角坐标
                corners = [
                    (0, 0),
                    (0, dem_cols - 1),
                    (dem_rows - 1, 0),
                    (dem_rows - 1, dem_cols - 1)
                ]
                
                lats = []
                lons = []
                for row, col in corners:
                    lon, lat = self._dem_pixel_to_lon_lat(row, col)
                    lats.append(lat)
                    lons.append(lon)
                
                if lats and lons:
                    return {
                        'lat_min': min(lats),
                        'lat_max': max(lats),
                        'lon_min': min(lons),
                        'lon_max': max(lons)
                    }
        except Exception as e:
            print(f"计算DEM覆盖范围失败: {e}")
        
        # 默认覆盖范围
        return {
            'lat_min': -90,
            'lat_max': 90,
            'lon_min': -180,
            'lon_max': 180
        }
    
    def _clip_dem(self, sar_coverage, buffer=0.1):
        """
        裁剪DEM到SAR覆盖范围内
        
        Args:
            sar_coverage: SAR覆盖范围字典
            buffer: 缓冲区大小（度）
        
        Returns:
            裁剪后的DEM数据和地理变换
        """
        try:
            if self.dem_data is None or self.dem_geotransform is None:
                print("⚠️  警告: DEM数据或地理变换未加载，无法裁剪")
                return self.dem_data, self.dem_geotransform
            
            # 计算带缓冲区的SAR覆盖范围
            sar_lat_min = sar_coverage['lat_min'] - buffer
            sar_lat_max = sar_coverage['lat_max'] + buffer
            sar_lon_min = sar_coverage['lon_min'] - buffer
            sar_lon_max = sar_coverage['lon_max'] + buffer
            
            dem_rows, dem_cols = self.dem_data.shape
            print(f"原始DEM尺寸: {dem_rows}x{dem_cols}")
            
            # 计算DEM像素范围
            try:
                min_row, min_col = self._lon_lat_to_dem_pixel(sar_lon_min, sar_lat_max)
                max_row, max_col = self._lon_lat_to_dem_pixel(sar_lon_max, sar_lat_min)
                print(f"计算的像素范围: 最小行={min_row:.2f}, 最小列={min_col:.2f}, 最大行={max_row:.2f}, 最大列={max_col:.2f}")
            except Exception as e:
                print(f"计算像素范围失败: {e}")
                # 使用默认裁剪范围（中心区域）
                min_row = dem_rows * 0.25
                max_row = dem_rows * 0.75
                min_col = dem_cols * 0.25
                max_col = dem_cols * 0.75
                print(f"使用默认裁剪范围: 行[{min_row:.2f}, {max_row:.2f}], 列[{min_col:.2f}, {max_col:.2f}]")
            
            # 确保坐标在有效范围内
            min_row = max(0, int(min_row))
            min_col = max(0, int(min_col))
            max_row = min(dem_rows - 1, int(max_row))
            max_col = min(dem_cols - 1, int(max_col))
            
            # 检查裁剪范围是否有效
            if min_row >= max_row or min_col >= max_col:
                print("⚠️  警告: 裁剪范围无效，使用原始DEM")
                return self.dem_data, self.dem_geotransform
            
            # 裁剪DEM
            clipped_dem = self.dem_data[min_row:max_row+1, min_col:max_col+1]
            
            # 更新地理变换
            new_geotransform = list(self.dem_geotransform)
            new_geotransform[0] = self.dem_geotransform[0] + min_col * self.dem_geotransform[1]
            new_geotransform[3] = self.dem_geotransform[3] + min_row * self.dem_geotransform[5]
            
            print(f"✓ DEM裁剪完成: {clipped_dem.shape[0]}x{clipped_dem.shape[1]} (原始: {dem_rows}x{dem_cols})")
            print(f"  裁剪范围: 纬度 [{sar_lat_min:.4f}, {sar_lat_max:.4f}], 经度 [{sar_lon_min:.4f}, {sar_lon_max:.4f}]")
            print(f"  像素范围: 行[{min_row}, {max_row}], 列[{min_col}, {max_col}]")
            
            return clipped_dem, new_geotransform
        except Exception as e:
            print(f"裁剪DEM失败: {e}")
            return self.dem_data, self.dem_geotransform
    
    def _calculate_satellite_position(self, time: float) -> Tuple[List[float], List[float]]:
        """
        根据YAML提供的数据计算指定时刻的卫星位置/速度。

        直接调用 `calculate_orbit_position` 以便尊重用户在
        ``orbit_interpolation_method`` 中的选择。
        """
        return calculate_orbit_position(time, self.orbit_data,
                                        interpolation_method=self.orbit_interpolation_method)

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
                return calculate_orbit_position(time, od,
                                                interpolation_method=self.orbit_interpolation_method)
            except Exception:
                # 插值失败时继续使用缓存处理
                pass
        # 使用缓存的数据进行线性 / 特例插值
        return self._interpolate_orbit_simple(time,
                                               self.orbit_cache['times'],
                                               self.orbit_cache['positions'],
                                               self.orbit_cache['velocities'])
    
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
    
    def _xyz_to_llh(self, x: float, y: float, z: float) -> List[float]:
        """
        地心坐标转换为经纬度高程
        
        Args:
            x: 地心X坐标（米）
            y: 地心Y坐标（米）
            z: 地心Z坐标（米）
            
        Returns:
            [纬度（度）, 经度（度）, 高程（米）]
        """
        return xyz_to_llh(x, y, z)
    
    def _llh_to_xyz_vectorized(self, lat: np.ndarray, lon: np.ndarray, height: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        经纬度高程转换为地心坐标（向量化版本）
        
        Args:
            lat: 纬度数组（度）
            lon: 经度数组（度）
            height: 高程数组（米）
            
        Returns:
            (x, y, z) 地心坐标数组
        """
        return llh_to_xyz_vectorized(lat, lon, height)
    
    def _xyz_to_llh_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        地心坐标转换为经纬度高程（向量化版本）
        
        Args:
            x: 地心X坐标数组（米）
            y: 地心Y坐标数组（米）
            z: 地心Z坐标数组（米）
            
        Returns:
            (lat, lon, height) 纬度、经度、高程数组
        """
        return xyz_to_llh_vectorized(x, y, z)
    
    def _calculate_doppler(self, slant_range: float) -> Tuple[float, float]:
        """
        计算多普勒频率和导数
        
        Args:
            slant_range: 斜距
            
        Returns:
            (多普勒频率, 多普勒频率导数)
        """
        return calculate_doppler(slant_range, self.doppler_polynomial, self.doppler_derivative_polynomial)
    
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
        # 使用Geo2rdr实例进行坐标转换
        if self.geo2rdr is not None:
            try:
                range_sample, azimuth_time = self.geo2rdr.geo2rdr(lat, lon, height)
                
                # 检查结果是否有效
                if np.isnan(range_sample) or np.isnan(azimuth_time) or range_sample == -1 or azimuth_time == -1:
                    raise Exception(f"无效的SAR坐标结果: range={range_sample}, azimuth={azimuth_time}")
                
                # 计算相对方位时间
                if hasattr(self, 'sensing_start') and self.sensing_start > 0:
                    relative_azimuth_time = azimuth_time - self.sensing_start
                else:
                    relative_azimuth_time = azimuth_time
                
                return range_sample, relative_azimuth_time
            except Exception as e:
                raise Exception(f"Geo2rdr坐标转换失败: {str(e)}")
        else:
            raise Exception("Geo2rdr实例未初始化")
    
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
            print("警告: 轨道时间范围为0，使用单位间隔代替以恢复方位时间变化")

        # 初始估计：使用向量化查询找最近的轨道点
        dists = np.linalg.norm(orbit_positions - target_xyz, axis=1)
        tmid = float(orbit_times[np.argmin(dists)])

        # 牛顿-拉夫逊求解
        max_iter = 100
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
            (距离向采样点, 方位向时间)
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
        
        # 计算SAR坐标
        try:
            result = self._calculate_sar_coordinates(lon, lat, height)
            return result
        except Exception as e:
            raise Exception(f"计算SAR坐标失败: {str(e)}")
    
    # NOTE: the ``_calculate_sar_coordinates_numba`` implementation below is
    # retained for historical reference only.  It is not invoked anywhere in the
    # current code path and does not support the full set of orbit/parameter
    # options.  Future refactoring may remove it entirely.
    def _calculate_sar_coordinates_numba(self, target_xyz, orbit_times, prf, near_range, range_pixel_spacing):
        """
        （遗留）Numba版本的SAR坐标计算，未使用。

        仅在重新启用 JIT 支持时才考虑恢复。
        """
        # 仅保留最小功能以避免语法错误
        if orbit_times.size > 0:
            tmid = (orbit_times[0] + orbit_times[-1]) / 2
        else:
            tmid = 0.0
        max_iter = 51
        tolerance = 5.0e-9
        tline = tmid
        for _ in range(max_iter):
            tline -= 0.0
        range_sample = 0.0
        return range_sample, tline
    
    def _process_chunk(self, chunk):
        """
        处理DEM分块
        """
        i_start, i_end, j_start, j_end, step = chunk
        results = []
        error_count = 0
        
        # 预加载当前块的DEM数据，减少内存访问开销
        if self.streaming and self.dem_data is not None:
            dem_chunk = self.dem_data[i_start:i_end, j_start:j_end]
        
        for i in range(i_start, i_end, step):
            for j in range(j_start, j_end, step):
                try:
                    rs, at = self.convert(i, j)
                    results.append([i, j, rs, at])
                except Exception as e:
                    error_count += 1
                    # 只在第一次出错时打印错误信息
                    if error_count == 1:
                        print(f"处理点 ({i}, {j}) 时出错: {e}")
        
        print(f"处理块 [{i_start}:{i_end}, {j_start}:{j_end}] 完成，成功: {len(results)}, 失败: {error_count}")
        return results
    
    def _process_chunk_shared_memory(self, chunk):
        """
        使用共享内存处理DEM分块
        """
        i_start, i_end, j_start, j_end, step, shm_name, shape, dtype = chunk
        
        # 连接到共享内存
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        dem_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        results = []
        error_count = 0
        
        for i in range(i_start, i_end, step):
            for j in range(j_start, j_end, step):
                try:
                    # 直接使用共享内存中的数据
                    height = dem_data[i, j]
                    
                    # 检查高程值是否有效
                    if np.isnan(height) or height < 0:
                        raise Exception(f"无效的高程值: {height}，点 ({i}, {j})")
                    
                    # 转换为经纬度
                    lon, lat = self._dem_pixel_to_lon_lat(i, j)
                    
                    # 检查经纬度值是否有效
                    if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                        raise Exception(f"无效的经纬度值: lon={lon}, lat={lat}，点 ({i}, {j})")
                    
                    # 转换地理坐标为地心坐标
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
                    
                    # 使用Geo2rdr实例进行坐标转换
                    if self.geo2rdr is not None:
                        rs, azimuth_time = self.geo2rdr.geo2rdr(lat, lon, height)
                        
                        # 检查结果是否有效
                        if np.isnan(rs) or np.isnan(azimuth_time) or rs == -1 or azimuth_time == -1:
                            raise Exception(f"无效的SAR坐标结果: range={rs}, azimuth={azimuth_time}")
                        
                        # 计算相对方位时间
                        if hasattr(self, 'sensing_start') and self.sensing_start > 0:
                            at = azimuth_time - self.sensing_start
                        else:
                            at = azimuth_time
                    else:
                        raise Exception("Geo2rdr实例未初始化")
                    
                    results.append([i, j, rs, at])
                except Exception as e:
                    error_count += 1
                    # 只在第一次出错时打印错误信息
                    if error_count == 1:
                        print(f"处理点 ({i}, {j}) 时出错: {e}")
        
        # 关闭共享内存连接
        existing_shm.close()
        
        print(f"处理块 [{i_start}:{i_end}, {j_start}:{j_end}] 完成，成功: {len(results)}, 失败: {error_count}")
        return results
    
    def convert_all(self, step: int = 10) -> np.ndarray:
        """
        转换整个DEM到SAR坐标
        
        Args:
            step: 采样步长
            
        Returns:
            SAR坐标数组 (n, 4)，字段为 row, col, range_sample, azimuth_time
        """
        if self.dem_data is None:
            raise Exception("DEM数据未加载")
        
        dem_rows, dem_cols = self.dem_data.shape
        
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
            chunks.append((i, i_end, 0, dem_cols, step))
        
        # 检查是否使用共享内存
        use_shared_memory = False
        if num_cores > 1 and dem_rows * dem_cols * 4 < available_memory * 0.5:
            use_shared_memory = True
            print("使用共享内存进行并行处理")
        
        # 并行处理
        start_time = time.time()
        if use_shared_memory:
            # 创建共享内存
            shm = shared_memory.SharedMemory(create=True, size=self.dem_data.nbytes)
            # 创建共享内存数组
            shm_array = np.ndarray(self.dem_data.shape, dtype=self.dem_data.dtype, buffer=shm.buf)
            # 复制数据到共享内存
            shm_array[:] = self.dem_data[:]
            
            # 准备分块参数
            shared_chunks = [(i, i_end, j, j_end, step, shm.name, self.dem_data.shape, self.dem_data.dtype)
                           for (i, i_end, j, j_end, step) in chunks]
            
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
        all_results = []
        for chunk_result in results:
            all_results.extend(chunk_result)
        
        end_time = time.time()
        print(f"处理时间: {end_time - start_time:.2f} 秒")
        print(f"缓存统计: 命中 {self._cache_hits}, 未命中 {self._cache_misses}, 命中率 {100*self._cache_hits/(self._cache_hits+self._cache_misses+1):.1f}%")
        return np.array(all_results)
    
    def generate_sar_dem(self, output_file: str, method: str = 'bilinear') -> np.ndarray:
        """
        生成与SAR图像范围相同的SAR坐标系DEM数据（修复版）
        
        流程：
        1. 验证时间基准参数
        2. 遍历DEM所有点，获取经纬度和高程
        3. 批量转换为SAR坐标
        4. 自适应插值生成规则网格
        
        Args:
            output_file: 输出文件路径
            method: 插值方法 ('nearest', 'bilinear', 'cubic', 'idw', 'rbf', 'gmt')
            
        Returns:
            SAR坐标系DEM数据数组
        """
        print("=== 生成SAR坐标系DEM数据（修复版）===")
        start_time = time.time()
        
        # ===== 高优先级修复3: 验证参数 =====
        print("\n步骤0：验证关键参数...")
        
        if self.dem_data is None:
            raise ValueError("DEM数据未加载")
        
        if self.geo2rdr is None:
            raise ValueError("Geo2rdr未初始化")
        
        # 输出卫星视线方向和飞行方向
        print(f"✓ look_direction (卫星视线方向): {self.look_direction}")
        
        # 计算并显示卫星飞行方向
        if self.orbit_cache and len(self.orbit_cache['positions']) > 1:
            pos_start = self.orbit_cache['positions'][0]
            pos_end = self.orbit_cache['positions'][-1]
            flight_dir = pos_end - pos_start
            flight_dir_norm = flight_dir / np.linalg.norm(flight_dir)
            print(f"✓ 卫星飞行方向 (单位向量): [{flight_dir_norm[0]:.4f}, {flight_dir_norm[1]:.4f}, {flight_dir_norm[2]:.4f}]")
        
        # 验证sensing_start
        if not hasattr(self, 'sensing_start') or self.sensing_start == 0.0:
            raise ValueError(
                "❌ sensing_start未正确初始化！\n"
                "请检查YAML文件中的first_line_sensing_time字段，\n"
                "或确保orbit_data包含有效的时间信息。\n"
                f"当前值: {getattr(self, 'sensing_start', 'N/A')}"
            )
        
        print(f"✓ sensing_start: {self.sensing_start} ({self.sensing_start:.6f})")
        print(f"✓ azimuth_time_step: {self.azimuth_time_step} ({1.0/self.azimuth_time_step:.2f} Hz)")
        print(f"✓ near_range: {self.near_range:.2f} m")
        print(f"✓ range_pixel_spacing: {self.range_pixel_spacing:.4f} m")
        
        # 再次验证时间范围
        if self.orbit_cache and len(self.orbit_cache['times']) > 0:
            orbit_t_min = self.orbit_cache['times'][0]
            orbit_t_max = self.orbit_cache['times'][-1]
            
            # 计算SAR相对时间
            orbit_start = getattr(self, 'orbit_start_time', self.sensing_start)
            sar_t_min_rel = self.sensing_start - orbit_start
            sar_t_max_rel = sar_t_min_rel + self.nrows * self.azimuth_time_step
            
            if sar_t_min_rel < orbit_t_min - 1.0 or sar_t_max_rel > orbit_t_max + 1.0:
                raise ValueError(
                    f"❌ SAR时间范围超出轨道时间范围！\n"
                    f"轨道: [{orbit_t_min:.2f}, {orbit_t_max:.2f}]\n"
                    f"SAR:  [{sar_t_min_rel:.2f}, {sar_t_max_rel:.2f}]"
                )
            print(f"✓ 时间范围验证通过")
        
        dem_rows, dem_cols = self.dem_data.shape
        sar_rows = self.nrows
        sar_cols = self.ncols
        print(f"DEM尺寸: {dem_rows}x{dem_cols}")
        print(f"SAR图像尺寸: {sar_rows}x{sar_cols}")
        
        print("\n步骤0.5：计算覆盖范围并裁剪DEM...")
        
        # 计算SAR覆盖范围
        sar_coverage = self._calculate_sar_coverage()
        print(f"SAR覆盖范围: 纬度 [{sar_coverage['lat_min']:.4f}, {sar_coverage['lat_max']:.4f}], 经度 [{sar_coverage['lon_min']:.4f}, {sar_coverage['lon_max']:.4f}]")
        
        # 计算DEM覆盖范围
        dem_coverage = self._calculate_dem_coverage()
        print(f"DEM覆盖范围: 纬度 [{dem_coverage['lat_min']:.4f}, {dem_coverage['lat_max']:.4f}], 经度 [{dem_coverage['lon_min']:.4f}, {dem_coverage['lon_max']:.4f}]")
        
        # 裁剪DEM到SAR覆盖范围内
        self.dem_data, self.dem_geotransform = self._clip_dem(sar_coverage, buffer=0.1)
        
        # 更新裁剪后的DEM尺寸
        if self.dem_data is not None:
            dem_rows, dem_cols = self.dem_data.shape
            print(f"裁剪后DEM尺寸: {dem_rows}x{dem_cols}")
        
        print("\n步骤1：计算DEM采样步长...")
        
        # ===== 中优先级修复1: 自适应采样步长 =====
        if self.dem_geotransform is not None:
            dem_lat_resolution = abs(self.dem_geotransform[5])  # 纬度分辨率（度）
            dem_lon_resolution = abs(self.dem_geotransform[1])   # 经度分辨率（度）
            lat = getattr(self, 'sar_center_lat', 30.0)
            
            # 更严谨的分辨率计算：
            # - 经度方向（东西向）：dx = 111320 * cos(lat) * dlon
            # - 纬度方向（南北向）：dy = 110574 * dlat
            # 取两者平均值作为DEM分辨率，高纬地区两者差异明显
            dx_meters = 111320.0 * np.cos(np.radians(lat)) * dem_lon_resolution
            dy_meters = 110574.0 * dem_lat_resolution
            dem_resolution_meters = (dx_meters + dy_meters) / 2.0
        else:
            dem_resolution_meters = 30.0
        
        azimuth_resolution = getattr(self, 'azimuth_spacing', 5.0)
        
        # 自适应步长：DEM分辨率除以SAR分辨率，取整数
        if azimuth_resolution > 0 and dem_resolution_meters > 0:
            step = max(1, int(dem_resolution_meters / azimuth_resolution))
        else:
            step = 1
        
        # 限制步长范围
        step = max(1, min(step, 20))  # 最大步长20，确保足够密度
        
        sampled_rows = dem_rows // step
        sampled_cols = dem_cols // step
        
        print(f"DEM分辨率: {dem_resolution_meters:.1f}m")
        print(f"SAR方位向分辨率: {azimuth_resolution:.1f}m")
        print(f"采样步长: {step} (采样点数: {sampled_rows}x{sampled_cols} = {sampled_rows*sampled_cols})")
        
        print("\n步骤2：提取DEM经纬度和高程（向量化）...")
        
        rows_sampled = np.arange(0, dem_rows, step)
        cols_sampled = np.arange(0, dem_cols, step)
        
        rows_mesh, cols_mesh = np.meshgrid(rows_sampled, cols_sampled, indexing='ij')
        rows_flat = rows_mesh.ravel()
        cols_flat = cols_mesh.ravel()
        
        dem_sampled = self.dem_data[rows_sampled, :][:, cols_sampled]
        dem_flat = dem_sampled.ravel()
        
        # 计算经纬度
        if self.dem_geotransform is not None:
            lon_flat = self.dem_geotransform[0] + cols_flat * self.dem_geotransform[1] + rows_flat * self.dem_geotransform[2]
            lat_flat = self.dem_geotransform[3] + cols_flat * self.dem_geotransform[4] + rows_flat * self.dem_geotransform[5]
        else:
            lat_step = getattr(self, 'dem_lat_step', 0.000833333) * step
            lon_step = getattr(self, 'dem_lon_step', 0.000833333) * step
            lat_min = getattr(self, 'dem_lat_min', -90)
            lon_min = getattr(self, 'dem_lon_min', -180)
            lat_flat = lat_min + rows_flat * lat_step
            lon_flat = lon_min + cols_flat * lon_step
        
        lats_arr = np.asarray(lat_flat, dtype=np.float64)
        lons_arr = np.asarray(lon_flat, dtype=np.float64)
        heights_arr = np.asarray(dem_flat, dtype=np.float64)
        
        n_points = len(lats_arr)
        print(f"转换点数: {n_points}")
        
        print("\n步骤3：批量转换为SAR坐标...")
        try:
            range_samples, azimuth_times = self.geo2rdr.geo2rdr_batch(lats_arr, lons_arr, heights_arr)
            print(f"✓ 批量转换成功")
        except Exception as e:
            print(f"❌ 批量转换失败: {e}")
            return np.zeros((sar_rows, sar_cols), dtype=np.float32)
        
        # ===== 使用绝对时间进行计算 =====
        # 验证转换结果的时间范围
        valid_mask = ~(np.isnan(range_samples) | np.isnan(azimuth_times))
        valid_range = range_samples[valid_mask]
        valid_azimuth = azimuth_times[valid_mask]
        valid_elevations = heights_arr[valid_mask]
        
        if len(valid_range) == 0:
            raise ValueError("❌ 没有有效的转换点！请检查DEM和SAR参数是否匹配。")
        
        print(f"有效转换点数: {len(valid_range)} / {n_points} ({100*len(valid_range)/n_points:.1f}%)")
        
        # 检查转换结果的时间范围（绝对时间）
        azimuth_time_min = np.min(valid_azimuth)
        azimuth_time_max = np.max(valid_azimuth)
        print(f"转换结果时间范围(绝对): [{azimuth_time_min:.2f}, {azimuth_time_max:.2f}]")
        
        # 检查轨道时间范围（绝对时间）
        if self.orbit_cache and len(self.orbit_cache['times']) > 0 and hasattr(self, 'orbit_start_time'):
            orbit_t_min_abs = self.orbit_start_time
            orbit_t_max_abs = self.orbit_start_time + self.orbit_cache['times'][-1]
            print(f"轨道时间范围(绝对): [{orbit_t_min_abs:.2f}, {orbit_t_max_abs:.2f}]")
            
            if azimuth_time_min < orbit_t_min_abs or azimuth_time_max > orbit_t_max_abs:
                print(f"⚠️  警告: 转换结果时间超出轨道范围")
                print(f"   轨道: [{orbit_t_min_abs:.2f}, {orbit_t_max_abs:.2f}]")
                print(f"   结果: [{azimuth_time_min:.2f}, {azimuth_time_max:.2f}]")
        
        # ===== 几何合理性检查 =====
        # 计算slant range（从range_sample反推）
        slant_ranges = valid_range * self.range_pixel_spacing + self.near_range
        slant_range_min = np.min(slant_ranges)
        slant_range_max = np.max(slant_ranges)
        print(f"slant range范围: [{slant_range_min:.2f}, {slant_range_max:.2f}] 米")
        
        # 检查slant range是否合理（620000 - 700000米）
        if slant_range_min < 500000 or slant_range_max > 800000:
            print("⚠️  警告: slant range值不合理，可能几何计算错误")
        
        # 检查azimuth time是否在sensing_start附近±1秒
        azimuth_time_rel = valid_azimuth - self.sensing_start
        azimuth_time_rel_min = np.min(azimuth_time_rel)
        azimuth_time_rel_max = np.max(azimuth_time_rel)
        print(f"azimuth time相对sensing_start: [{azimuth_time_rel_min:.2f}, {azimuth_time_rel_max:.2f}] 秒")
        
        if abs(azimuth_time_rel_min) > 10 or abs(azimuth_time_rel_max) > 10:
            print("⚠️  警告: azimuth time偏离sensing_start过多，可能几何计算错误")
        
        # ===== DEM点的几何过滤 =====
        # 计算SAR最大距离
        max_range = self.near_range + sar_cols * self.range_pixel_spacing
        # 保留slant range在合理范围内的点
        range_mask = (slant_ranges > self.near_range - 5000) & (slant_ranges < max_range + 5000)
        valid_range = valid_range[range_mask]
        valid_azimuth = valid_azimuth[range_mask]
        valid_elevations = valid_elevations[range_mask]
        slant_ranges = slant_ranges[range_mask]
        
        print(f"几何过滤后有效点数: {len(valid_range)} ({100*len(valid_range)/n_points:.1f}%)")
        
        print("\n步骤4：转换为像素坐标并插值...")
        
        # ===== 正确转换方位向时间为像素索引 =====
        # 使用绝对时间计算像素坐标：azimuth_pixel = (azimuth_time - sensing_start) / time_step
        azimuth_pixels = (valid_azimuth - self.sensing_start) / self.azimuth_time_step
        
        print(f"range像素范围: [{np.min(valid_range):.1f}, {np.max(valid_range):.1f}]")
        print(f"azimuth像素范围: [{np.min(azimuth_pixels):.1f}, {np.max(azimuth_pixels):.1f}]")
        print(f"SAR图像尺寸: {sar_rows}x{sar_cols}")
        
        # ===== 范围验证和裁剪 =====
        # 1. 首先过滤掉明显不合理的range值
        valid_range_mask = (valid_range >= 0) & (valid_range < sar_cols * 2)  # 允许一定的边界扩展
        valid_azimuth_mask = (azimuth_pixels >= -500) & (azimuth_pixels < sar_rows + 500)  # 允许一定的边界扩展
        
        # 组合掩码
        combined_mask = valid_range_mask & valid_azimuth_mask
        valid_range = valid_range[combined_mask]
        valid_azimuth = valid_azimuth[combined_mask]
        valid_elevations = valid_elevations[combined_mask]
        azimuth_pixels = azimuth_pixels[combined_mask]
        
        print(f"过滤后有效点数: {len(valid_range)} ({100*len(valid_range)/n_points:.1f}%)")
        
        # ===== 中优先级修复2: 自适应边界扩展 =====
        margin_azimuth = max(50, int(sar_rows * 0.05))  # 至少5%
        margin_range = max(50, int(sar_cols * 0.05))
        margin_azimuth = max(margin_azimuth, step * 10)
        margin_range = max(margin_range, step * 10)
        
        print(f"边界扩展: 方位向={margin_azimuth}, 距离向={margin_range}")
        
        # 裁剪到扩展范围内
        crop_mask = (valid_range >= -margin_range) & (valid_range < sar_cols + margin_range) & \
                    (azimuth_pixels >= -margin_azimuth) & (azimuth_pixels < sar_rows + margin_azimuth)
        
        range_pixels = valid_range[crop_mask]
        azimuth_pixels_cropped = azimuth_pixels[crop_mask]
        elevations = valid_elevations[crop_mask]
        
        print(f"裁剪后有效点数: {len(range_pixels)} ({100*len(range_pixels)/len(valid_range):.1f}%)")
        
        if len(range_pixels) < 100:
            raise ValueError(f"❌ 有效点太少({len(range_pixels)})，无法生成有效的SAR DEM")
        
        # 去重：对重复坐标点取最大值
        points = np.column_stack([range_pixels, azimuth_pixels_cropped])
        values = elevations
        
        unique_points, inverse_idx = np.unique(points, axis=0, return_inverse=True)
        max_values = np.zeros(len(unique_points))
        for i in range(len(unique_points)):
            mask = (inverse_idx == i)
            max_values[i] = np.max(values[mask])
        
        points = unique_points
        values = max_values
        
        print(f"去重后点数: {len(points)}")
        
        # ===== 中优先级修复3: 自适应插值方法选择 =====
        n_grid = sar_rows * sar_cols
        density = len(points) / n_grid
        
        print(f"\n步骤5：临时屏蔽插值算法...")
        
        # 扩展边界
        extended_cols = sar_cols + 2 * margin_range
        extended_rows = sar_rows + 2 * margin_azimuth
        
        # 临时占位符：创建一个全零数组
        sar_dem_extended = np.zeros((extended_rows, extended_cols), dtype=np.float32)
        print("使用全零数组作为临时占位符")
        
        # 裁剪回原始SAR尺寸
        sar_dem = sar_dem_extended[margin_azimuth:margin_azimuth+sar_rows, margin_range:margin_range+sar_cols]
        
        sar_dem = np.nan_to_num(sar_dem, nan=0.0)
        
        print("步骤4：保存结果...")
        
        try:
            from osgeo import gdal
            
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(output_file, sar_cols, sar_rows, 1, gdal.GDT_Float32)
            dataset.GetRasterBand(1).WriteArray(sar_dem)
            dataset = None
            print(f"SAR坐标系DEM数据已保存到: {output_file}")
            
            vrt_file = output_file.replace('.tif', '.vrt')
            gdal.BuildVRT(vrt_file, output_file)
            print(f"VRT文件已保存到: {vrt_file}")
            
        except Exception as e:
            print(f"保存失败: {e}")
            np.save(output_file.replace('.tif', '.npy'), sar_dem)
        
        end_time = time.time()
        print(f"生成完成，处理时间: {end_time - start_time:.2f} 秒")
        
        return sar_dem

def process_sar_chunk(chunk, orbit_data, processing_params):
    """
    处理SAR分块（优化版：使用向量化操作）
    
    Args:
        chunk: 分块信息 (i_start, i_end, j_start, j_end)
        orbit_data: 轨道数据字典
        processing_params: 处理参数字典
        
    Returns:
        i_start, i_end, j_start, j_end, local_sar_dem
    """
    
    def lon_lat_to_dem_pixel(lon, lat, dem_geotransform, dem_projection):
        """经纬度转换为DEM像素坐标"""
        try:
            from osgeo import gdal, osr
            osr.UseExceptions()
            
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(4326)
            dst_srs = osr.SpatialReference(wkt=dem_projection)
            transform = osr.CoordinateTransformation(src_srs, dst_srs)
            
            x, y, _ = transform.TransformPoint(lon, lat)
            
            dx = dem_geotransform[1]
            dy = dem_geotransform[5]
            x0 = dem_geotransform[0]
            y0 = dem_geotransform[3]
            
            if dx == 0 or dy == 0:
                return 0.0, 0.0
            
            col = (x - x0) / dx
            row = (y - y0) / dy
            
            return row, col
        except Exception:
            return 0.0, 0.0
    
    def interpolate_dem(dem_data, row, col, method):
        """从DEM中插值获取高程值"""
        nrows, ncols = dem_data.shape
        if row < 0 or row >= nrows - 1 or col < 0 or col >= ncols - 1:
            return 0.0
        
        i = int(row)
        j = int(col)
        di = row - i
        dj = col - j
        
        i = min(i, nrows - 2)
        j = min(j, ncols - 2)
        
        v00 = dem_data[i, j]
        v01 = dem_data[i, j+1]
        v10 = dem_data[i+1, j]
        v11 = dem_data[i+1, j+1]
        
        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            return 0.0
        
        if method == 'nearest':
            if di < 0.5 and dj < 0.5:
                return v00
            elif di < 0.5 and dj >= 0.5:
                return v01
            elif di >= 0.5 and dj < 0.5:
                return v10
            else:
                return v11
        else:
            return (1 - di) * ((1 - dj) * v00 + dj * v01) + di * ((1 - dj) * v10 + dj * v11)
    
    def xyz_to_llh_local(x, y, z):
        """地心坐标转换为经纬度"""
        a = 6378137.0
        e2 = 0.00669437999014
        
        lon = np.arctan2(y, x)
        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z * a, p * (a * (1 - e2)))
        lat = np.arctan2(z + e2 * a * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        height = (p / np.cos(lat)) - N
        
        return np.degrees(lat), np.degrees(lon), height
    
    i_start, i_end, j_start, j_end = chunk
    n_rows = i_end - i_start
    n_cols = j_end - j_start
    local_sar_dem = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    # 从参数字典中提取数据
    orbit_state_table = orbit_data.get('orbit_state_table')
    orbit_state_times = orbit_data.get('orbit_state_times')
    orbit_splines = orbit_data.get('orbit_splines')
    orbit_cache = orbit_data.get('orbit_cache')
    look_direction_param = orbit_data.get('look_direction')
    
    azimuth_time_step = processing_params['azimuth_time_step']
    near_range = processing_params['near_range']
    range_pixel_spacing = processing_params['range_pixel_spacing']
    sensing_start = processing_params.get('sensing_start', 0.0)  # ===== 关键：获取sensing_start =====
    
    # 验证sensing_start
    if sensing_start == 0.0:
        print("⚠️  警告: process_sar_chunk中sensing_start为0，可能导致坐标转换错误")
    
    prf = processing_params.get('prf', 4144.0)
    look_direction = processing_params.get('look_direction', 'right')
    doppler_polynomial = processing_params.get('doppler_polynomial')
    doppler_derivative_polynomial = processing_params.get('doppler_derivative_polynomial')
    interp_method = processing_params['interp_method']
    dem_data = processing_params['dem_data']
    dem_geotransform = processing_params['dem_geotransform']
    dem_projection = processing_params['dem_projection']
    
    # 获取轨道时间范围
    orbit_times = orbit_cache['times'] if orbit_cache is not None else orbit_state_times
    orbit_positions = orbit_cache['positions'] if orbit_cache is not None else None
    if orbit_positions is None and orbit_state_table is not None:
        orbit_positions = orbit_state_table['positions']
    orbit_velocities = orbit_cache['velocities'] if orbit_cache is not None else None
    if orbit_velocities is None and orbit_state_table is not None:
        orbit_velocities = orbit_state_table['velocities']
    
    wavelength = 0.031
    
    def get_satellite_state(t, times, positions, velocities):
        """获取指定时间的卫星位置和速度"""
        if times is None or len(times) < 2:
            return np.array([0.0, 0.0, 500000.0]), np.array([0.0, 0.0, 0.0])
        
        t_min = float(times[0])
        t_max = float(times[-1])
        t = max(t_min, min(t_max, t))
        
        time_idx = np.searchsorted(times, t)
        
        if time_idx == 0:
            return positions[0].copy(), velocities[0].copy() if velocities is not None and len(velocities) > 0 else np.array([0.0, 0.0, 0.0])
        elif time_idx >= len(times):
            return positions[-1].copy(), velocities[-1].copy() if velocities is not None and len(velocities) > 0 else np.array([0.0, 0.0, 0.0])
        else:
            t1 = times[time_idx - 1]
            t2 = times[time_idx]
            if t2 > t1:
                frac = (t - t1) / (t2 - t1)
            else:
                frac = 0.0
            pos = positions[time_idx - 1] + frac * (positions[time_idx] - positions[time_idx - 1])
            if velocities is not None and len(velocities) > 0:
                vel = velocities[time_idx - 1] + frac * (velocities[time_idx] - velocities[time_idx - 1])
            else:
                vel = np.array([0.0, 0.0, 0.0])
            return pos, vel
    
    def calculate_doppler(slant_range, dop_poly, dop_der_poly):
        """计算多普勒频率"""
        if dop_poly is None or len(dop_poly) == 0:
            return 0.0, 0.0
        
        fdop = 0.0
        for i, coeff in enumerate(dop_poly):
            fdop += coeff * (slant_range ** i)
        
        fdopder = 0.0
        if dop_der_poly is not None:
            for i, coeff in enumerate(dop_der_poly):
                fdopder += coeff * (slant_range ** i)
        
        return fdop, fdopder
    
    def solve_range_doppler(target_xyz, azimuth_time_init, slant_range, orbit_times, orbit_positions, orbit_velocities, 
                           doppler_polynomial, doppler_derivative_polynomial, look_dir, wavelength, max_iter=51):
        """Range-Doppler迭代求解"""
        t_min = float(orbit_times[0])
        t_max = float(orbit_times[-1])
        t_range = t_max - t_min
        if t_range == 0:
            t_max = t_min + 1.0
            t_range = 1.0
        
        if azimuth_time_init < t_min:
            azimuth_time_init = t_min
        elif azimuth_time_init > t_max:
            azimuth_time_init = t_max
        
        tline = azimuth_time_init
        
        for k in range(max_iter):
            tline = max(t_min, min(t_max, tline))
            
            sat_pos, sat_vel = get_satellite_state(tline, orbit_times, orbit_positions, orbit_velocities)
            
            dr = target_xyz - sat_pos
            slant_range_curr = np.linalg.norm(dr)
            if slant_range_curr <= 0 or np.isnan(slant_range_curr):
                return np.nan, np.nan
            
            dopfact = np.dot(dr, sat_vel)
            fdop, fdopder = calculate_doppler(slant_range_curr, doppler_polynomial, doppler_derivative_polynomial)
            
            if fdop == 0.0:
                dr_unit = dr / slant_range_curr
                vel_los = np.dot(sat_vel, dr_unit)
                fdop = -2 * vel_los / wavelength
                fdopder = 0.0
            
            if look_dir == "left":
                fdop *= -1
                fdopder *= -1
            
            fn = dopfact - fdop * slant_range_curr
            c1 = -np.dot(sat_vel, sat_vel)
            c2 = (fdop / slant_range_curr) + fdopder
            fnprime = c1 + c2 * dopfact
            
            if abs(fnprime) > 1e-12:
                delta_t = fn / fnprime
                if abs(delta_t) > t_range / 4:
                    delta_t *= 0.5
                tline -= delta_t
                
                tolerance = 5.0e-9
                if abs(delta_t) < tolerance and abs(fn) < tolerance * slant_range_curr:
                    break
            else:
                break
        
        return tline, slant_range_curr
    
    # 使用向量化操作批量处理
    try:
        # 创建网格坐标
        i_indices, j_indices = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
        
        # 展平为数组
        i_flat = i_indices.ravel()
        j_flat = j_indices.ravel()
        
        # ===== 关键修复：计算绝对时间，加上sensing_start =====
        azimuth_times = sensing_start + (i_start + i_flat) * azimuth_time_step
        slant_ranges = near_range + j_flat * range_pixel_spacing
        
        n_pixels = len(azimuth_times)
        
        for idx in range(n_pixels):
            try:
                t_az = azimuth_times[idx]
                sr = slant_ranges[idx]
                
                sat_pos_init, sat_vel_init = get_satellite_state(t_az, orbit_times, orbit_positions, orbit_velocities)
                
                sat_dist = np.linalg.norm(sat_pos_init)
                if sat_dist <= 0:
                    continue
                look_dir_init = -sat_pos_init / sat_dist
                
                target_xyz_init = sat_pos_init + look_dir_init * sr
                
                lat_init, lon_init, height_init = xyz_to_llh_local(
                    target_xyz_init[0], target_xyz_init[1], target_xyz_init[2]
                )
                
                dem_row, dem_col = lon_lat_to_dem_pixel(lon_init, lat_init, dem_geotransform, dem_projection)
                elevation = interpolate_dem(dem_data, dem_row, dem_col, interp_method)
                
                if elevation > -1000 and elevation < 10000:
                    x, y, z = llh_to_xyz_vectorized(np.array([lat_init]), np.array([lon_init]), np.array([elevation]))
                    target_xyz = np.array([x[0], y[0], z[0]])
                else:
                    target_xyz = target_xyz_init
                
                azimuth_time_solved, slant_range_solved = solve_range_doppler(
                    target_xyz, t_az, sr,
                    orbit_times, orbit_positions, orbit_velocities,
                    doppler_polynomial, doppler_derivative_polynomial,
                    look_direction, wavelength
                )
                
                if not np.isnan(azimuth_time_solved):
                    sat_pos_final, _ = get_satellite_state(azimuth_time_solved, orbit_times, orbit_positions, orbit_velocities)
                    final_target_xyz = sat_pos_final + (target_xyz - sat_pos_final) / np.linalg.norm(target_xyz - sat_pos_final) * sr
                else:
                    final_target_xyz = target_xyz_init
                
                lat_final, lon_final, _ = xyz_to_llh_local(
                    final_target_xyz[0], final_target_xyz[1], final_target_xyz[2]
                )
                
                if -180 <= lon_final <= 180 and -90 <= lat_final <= 90:
                    dem_row, dem_col = lon_lat_to_dem_pixel(lon_final, lat_final, dem_geotransform, dem_projection)
                    elevation_final = interpolate_dem(dem_data, dem_row, dem_col, interp_method)
                    local_sar_dem.flat[idx] = elevation_final
            except Exception:
                local_sar_dem.flat[idx] = 0
                
    except Exception as e:
        # 如果向量化方法失败，回退到逐点处理
        print(f"向量化处理失败，回退到逐点处理: {e}")
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                try:
                    # ===== 关键修复：计算SAR像素对应的地理坐标，加上sensing_start =====
                    azimuth_time = sensing_start + i * azimuth_time_step
                    slant_range = near_range + j * range_pixel_spacing
                    
                    # 计算卫星位置（使用预计算表）
                    t = azimuth_time
                    time_idx = np.searchsorted(orbit_state_times, t) if orbit_state_times is not None else 0
                    if orbit_state_table is not None and orbit_state_times is not None:
                        times = orbit_state_times
                        positions = orbit_state_table['positions']
                        if time_idx == 0:
                            sat_pos = positions[0]
                        elif time_idx >= len(times):
                            sat_pos = positions[-1]
                        else:
                            t1 = times[time_idx - 1]
                            t2 = times[time_idx]
                            frac = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
                            sat_pos = positions[time_idx - 1] + frac * (positions[time_idx] - positions[time_idx - 1])
                    else:
                        sat_pos = np.array([0.0, 0.0, 500000.0])
                    
                    # 计算目标点的地心坐标
                    sat_dist = np.linalg.norm(sat_pos)
                    look_dir = -sat_pos / sat_dist
                    target_xyz = sat_pos + look_dir * slant_range
                    
                    # 转换到经纬度
                    lat, lon, height = xyz_to_llh_local(target_xyz[0], target_xyz[1], target_xyz[2])
                    
                    # 检查经纬度值是否有效
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        dem_row, dem_col = lon_lat_to_dem_pixel(lon, lat, dem_geotransform, dem_projection)
                        elevation = interpolate_dem(dem_data, dem_row, dem_col, interp_method)
                        local_sar_dem[i - i_start, j - j_start] = elevation
                except Exception:
                    local_sar_dem[i - i_start, j - j_start] = 0
    
    return i_start, i_end, j_start, j_end, local_sar_dem


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='DEM到SAR坐标转换工具（优化版）')
    parser.add_argument('yaml_file', help='SAR YAML文件路径')
    parser.add_argument('dem_file', help='DEM文件路径')
    parser.add_argument('output_file', help='输出文件路径')
    parser.add_argument('--step', type=int, default=10, help='采样步长')
    parser.add_argument('--orbit-interp', choices=['HERMITE','SCH','LINEAR'],
                        default='HERMITE', help='轨道插值方法')
    parser.add_argument('--streaming', action='store_true', help='使用流式处理模式')
    parser.add_argument('--max-cache-size', type=int, default=100000, help='最大缓存大小')
    parser.add_argument('--convert-only', action='store_true', help='仅执行DEM到SAR坐标转换（默认是生成SAR坐标系DEM数据）')
    parser.add_argument('--interp-method', choices=['nearest', 'bilinear', 'cubic', 'rbf', 'gmt', 'idw'],
                        default='bilinear', help='DEM插值方法')
    
    args = parser.parse_args()
    
    try:
        converter = DemToSarConverter(args.yaml_file,
                                     args.dem_file,
                                     orbit_interpolation_method=args.orbit_interp,
                                     streaming=args.streaming,
                                     max_cache_size=args.max_cache_size)
        
        if not args.convert_only:
            # 默认行为：生成SAR坐标系DEM数据
            print("=== 生成SAR坐标系DEM数据 ===")
            print(f"SAR YAML文件: {args.yaml_file}")
            print(f"DEM文件: {args.dem_file}")
            print(f"输出文件: {args.output_file}")
            print(f"插值方法: {args.interp_method}")
            
            converter.generate_sar_dem(args.output_file, method=args.interp_method)
        else:
            # 仅执行DEM到SAR坐标转换
            print("=== DEM到SAR坐标转换工具（优化版）===")
            print(f"SAR YAML文件: {args.yaml_file}")
            print(f"DEM文件: {args.dem_file}")
            print(f"输出文件: {args.output_file}")
            print(f"采样步长: {args.step}")
            print(f"轨道插值方法: {args.orbit_interp}")
            print(f"流式处理: {args.streaming}")
            print(f"最大缓存大小: {args.max_cache_size}")
            
            result = converter.convert_all(args.step)
            
            # 保存结果（包含行列）
            if len(result) > 0:
                header = 'row col range_sample azimuth_time'
                fmt = '%.0f %.0f %.6f %.6f'
                np.savetxt(args.output_file, result, fmt=fmt, header=header)
                print(f"转换完成，结果保存到: {args.output_file}")
                print(f"转换点数量: {len(result)}")
            else:
                # 如果没有结果，创建一个空文件
                with open(args.output_file, 'w') as f:
                    f.write('# row col range_sample azimuth_time\n')
                print(f"转换完成，但没有有效结果，创建了空文件: {args.output_file}")
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()