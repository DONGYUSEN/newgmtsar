#!/usr/bin/env python3
"""
geo2rdr模块

功能：将地理坐标（经纬度+高程）转换为雷达坐标。
实现参考ISCE2的 geo2rdr 算法。YAML 中的轨道数据应包含至少
两个具有不同时间戳的轨道点；若轨道时间退化，代码会构造
简单的线性运动以保持方位时间可变。
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from multiprocessing import Pool, cpu_count
try:
    import numba
except ImportError:
    numba = None

# 尝试导入numba，如果不可用则使用普通实现
try:
    from numba import jit, prange
except ImportError:
    print("Numba not available, running without JIT optimization")
    # 定义替代装饰器
    def jit(*args, **kwargs):
        def decorator(func):
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


def calculate_doppler(slant_range: float,
                      doppler_polynomial: Optional[List[float]] = None,
                      doppler_derivative_polynomial: Optional[List[float]] = None) -> Tuple[float, float]:
    """
    计算多普勒频率和导数
    """
    fdop = 0.0
    fdopder = 0.0
    
    if doppler_polynomial:
        try:
            fdop = 0.0
            for i, coeff in enumerate(doppler_polynomial):
                fdop += coeff * (slant_range ** i)
        except Exception:
            pass
    
    if doppler_derivative_polynomial:
        try:
            fdopder = 0.0
            for i, coeff in enumerate(doppler_derivative_polynomial):
                fdopder += coeff * (slant_range ** i)
        except Exception:
            pass
    
    # 确保多普勒调频率始终为负
    if fdopder > 0:
        fdopder = -fdopder
    
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
    p0, p1: 端点位置
    v0, v1: 端点速度
    t: 插值参数 [0, 1]
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
                            interpolation_method: str = 'HERMITE') -> Tuple[List[float], List[float]]:
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
                vel = [
                    (p1[i] - p0[i]) / dt for i in range(3)
                ]
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


class Geo2rdr:
    """
    geo2rdr转换器
    """
    def __init__(self,
                 prf: float = 0.0,
                 radar_wavelength: float = 0.0,
                 slant_range_pixel_spacing: float = 0.0,
                 range_first_sample: float = 0.0,
                 sensing_start: float = 0.0,
                 look_side: str = 'LEFT',
                 bistatic_delay_correction: bool = False,
                 orbit_interpolation_method: str = 'HERMITE',
                 fd1: float = 0.0,
                 fdd1: float = 0.0):
        """
        初始化geo2rdr

        Args:
            prf: 脉冲重复频率
            radar_wavelength: 雷达波长
            slant_range_pixel_spacing: 斜距像素间距
            range_first_sample: 距离向第一个采样点
            sensing_start: sensing开始时间
            look_side: 观测方向（LEFT/RIGHT）
            bistatic_delay_correction: 是否启用双站延迟校正
            orbit_interpolation_method: 插值方法('HERMITE','SCH','LINEAR')
            fd1: 多普勒中心频率一阶系数
            fdd1: 多普勒中心频率二阶系数
        """
        # 地球椭球参数
        self.major_semi_axis = 6378137.0  # 长半轴
        self.eccentricity_squared = 0.00669437999014  # 第一偏心率平方
        
        # 雷达参数
        self.slant_range_pixel_spacing = slant_range_pixel_spacing
        self.range_first_sample = range_first_sample
        self.prf = prf
        self.radar_wavelength = radar_wavelength
        self.sensing_start = sensing_start
        
        # 图像参数
        self.length = 0
        self.width = 0
        
        # DEM参数
        self.dem_width = 0
        self.dem_length = 0
        
        # Looks参数
        self.number_range_looks = 1
        self.number_azimuth_looks = 1
        
        # 轨道参数
        self.orbit = None
        
        # 其他参数
        self.look_side = look_side.upper()
        self.bistatic_delay_correction_flag = bistatic_delay_correction
        self.orbit_interpolation_method = orbit_interpolation_method.upper()
        
        # 多普勒参数
        self.doppler_polynomial = None
        self.doppler_derivative_polynomial = None
        self.fd1 = fd1
        self.fdd1 = fdd1
    
    def set_ellipsoid(self, major_semi_axis: float, eccentricity_squared: float):
        """
        设置椭球参数
        
        Args:
            major_semi_axis: 长半轴
            eccentricity_squared: 第一偏心率平方
        """
        self.major_semi_axis = major_semi_axis
        self.eccentricity_squared = eccentricity_squared
    
    def set_range_parameters(self, slant_range_pixel_spacing: float, range_first_sample: float):
        """
        设置距离向参数
        
        Args:
            slant_range_pixel_spacing: 斜距像素间距
            range_first_sample: 距离向第一个采样点
        """
        self.slant_range_pixel_spacing = slant_range_pixel_spacing
        self.range_first_sample = range_first_sample
    
    def set_radar_parameters(self, prf: float, radar_wavelength: float, sensing_start: float):
        """
        设置雷达参数
        
        Args:
            prf: 脉冲重复频率
            radar_wavelength: 雷达波长
            sensing_start:  sensing开始时间
        """
        self.prf = prf
        self.radar_wavelength = radar_wavelength
        self.sensing_start = sensing_start
    
    def set_image_size(self, length: int, width: int):
        """
        设置图像尺寸
        
        Args:
            length: 长度（方位向）
            width: 宽度（距离向）
        """
        self.length = length
        self.width = width
    
    def set_dem_size(self, dem_width: int, dem_length: int):
        """
        设置DEM尺寸
        
        Args:
            dem_width: DEM宽度
            dem_length: DEM长度
        """
        self.dem_width = dem_width
        self.dem_length = dem_length
    
    def set_looks(self, number_range_looks: int, number_azimuth_looks: int):
        """
        设置looks数
        
        Args:
            number_range_looks: 距离向looks数
            number_azimuth_looks: 方位向looks数
        """
        self.number_range_looks = number_range_looks
        self.number_azimuth_looks = number_azimuth_looks
    
    def set_orbit(self, orbit: Dict):
        """
        设置轨道信息
        
        Args:
            orbit: 轨道信息（包含orbit_points列表）
        """
        self.orbit = orbit
        
        self._precompute_orbit_data()
    
    def set_orbit_state_table(self, table_times: np.ndarray, table_positions: np.ndarray, table_velocities: np.ndarray, t0: float = 0.0, time_range: float = 0.0):
        """
        设置预计算的轨道状态表（用于加速批量处理）
        
        Args:
            table_times: 时间数组（相对时间）
            table_positions: 位置数组 (n, 3)
            table_velocities: 速度数组 (n, 3)
            t0: 起始时间
            time_range: 时间范围
        """
        self._orbit_state_table = {
            'positions': table_positions,
            'velocities': table_velocities
        }
        self._orbit_state_times = table_times
        self._orbit_t0 = t0
        self._orbit_time_range = time_range
    
    def _precompute_orbit_data(self):
        """预计算轨道数据（优化：避免重复解析）"""
        if not self.orbit or 'orbit_points' not in self.orbit:
            self._orbit_times = None
            self._orbit_positions = None
            self._orbit_velocities = None
            self._orbit_t0 = 0.0
            return
        
        orbit_points = self.orbit['orbit_points']
        if not orbit_points:
            self._orbit_times = None
            self._orbit_positions = None
            self._orbit_velocities = None
            return
        
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
            self._orbit_times = None
            self._orbit_positions = None
            self._orbit_velocities = None
            return
        
        self._orbit_times = np.array(orbit_times, dtype=np.float64)
        self._orbit_positions = np.array(orbit_positions, dtype=np.float64)
        
        if orbit_velocities:
            self._orbit_velocities = np.array(orbit_velocities, dtype=np.float64)
        else:
            self._orbit_velocities = np.zeros_like(self._orbit_positions)
        
        self._orbit_t0 = float(self._orbit_times[0])
        self._orbit_relative_times = self._orbit_times - self._orbit_t0
        
        self._orbit_time_range = float(np.ptp(self._orbit_relative_times))
        
        self._precompute_orbit_state_table()
    
    def _precompute_orbit_state_table(self, n_samples: int = 10000):
        """预计算轨道状态表（大幅加速查询）"""
        if self._orbit_times is None or len(self._orbit_times) < 2:
            self._orbit_state_table = None
            self._orbit_state_times = None
            return
        
        if self._orbit_time_range <= 0:
            self._orbit_state_table = None
            self._orbit_state_times = None
            return
        
        n_samples = min(n_samples, max(100, int(self._orbit_time_range * 100)))
        
        table_times = np.linspace(0, self._orbit_time_range, n_samples)
        
        table_positions = np.zeros((n_samples, 3), dtype=np.float64)
        table_velocities = np.zeros((n_samples, 3), dtype=np.float64)
        
        for i, t in enumerate(table_times):
            pos, vel = self._interpolate_orbit_fast(t)
            table_positions[i] = pos
            table_velocities[i] = vel
        
        self._orbit_state_table = {
            'positions': table_positions,
            'velocities': table_velocities
        }
        self._orbit_state_times = table_times
    
    def _interpolate_orbit_hermite(self, time):
        """使用HERMITE插值进行轨道插值"""
        if self._orbit_times is None or len(self._orbit_times) < 2:
            return [0, 0, 500000], [0, 7000, 0]
        
        # 找到时间所在的区间
        idx = np.searchsorted(self._orbit_times, time)
        if idx == 0:
            idx = 1
        elif idx >= len(self._orbit_times):
            idx = len(self._orbit_times) - 1
        
        # 获取相邻的轨道点
        t1 = self._orbit_times[idx - 1]
        t2 = self._orbit_times[idx]
        p1 = self._orbit_positions[idx - 1]
        p2 = self._orbit_positions[idx]
        v1 = self._orbit_velocities[idx - 1]
        v2 = self._orbit_velocities[idx]
        
        # 计算插值参数
        if t2 > t1:
            t = (time - t1) / (t2 - t1)
        else:
            t = 0.0
        
        # HERMITE插值
        t2_f = t * t
        t3_f = t2_f * t
        
        h00 = 2 * t3_f - 3 * t2_f + 1
        h10 = t3_f - 2 * t2_f + t
        h01 = -2 * t3_f + 3 * t2_f
        h11 = t3_f - t2_f
        
        scale = t2 - t1
        v1_scaled = v1 * scale
        v2_scaled = v2 * scale
        
        pos = h00 * p1 + h10 * v1_scaled + h01 * p2 + h11 * v2_scaled
        vel = (p2 - p1) / (t2 - t1) if t2 > t1 else v1
        
        return pos.tolist(), vel.tolist()
    
    def _interpolate_orbit_fast(self, relative_time: float):
        """快速轨道插值（使用预计算的numpy数组）"""
        if self._orbit_times is None or len(self._orbit_times) < 2:
            return [0, 0, 500000], [0, 7000, 0]
        
        rt = relative_time
        
        if rt <= 0:
            return self._orbit_positions[0].tolist(), self._orbit_velocities[0].tolist()
        if rt >= self._orbit_time_range:
            return self._orbit_positions[-1].tolist(), self._orbit_velocities[-1].tolist()
        
        idx = np.searchsorted(self._orbit_relative_times, rt)
        
        if idx == 0:
            idx = 1
        elif idx >= len(self._orbit_relative_times):
            idx = len(self._orbit_relative_times) - 1
        
        t1 = self._orbit_relative_times[idx - 1]
        t2 = self._orbit_relative_times[idx]
        
        if t2 > t1:
            frac = (rt - t1) / (t2 - t1)
        else:
            frac = 0.0
        
        method = getattr(self, 'orbit_interpolation_method', 'HERMITE').upper()
        
        if method == 'LINEAR':
            pos = self._orbit_positions[idx - 1] + frac * (self._orbit_positions[idx] - self._orbit_positions[idx - 1])
            vel = self._orbit_velocities[idx - 1] + frac * (self._orbit_velocities[idx] - self._orbit_velocities[idx - 1])
        elif method == 'HERMITE':
            # 使用优化的HERMITE插值
            return self._interpolate_orbit_hermite(self._orbit_t0 + relative_time)
        else:
            pos = self._orbit_positions[idx - 1] + frac * (self._orbit_positions[idx] - self._orbit_positions[idx - 1])
            vel = self._orbit_velocities[idx - 1] + frac * (self._orbit_velocities[idx] - self._orbit_velocities[idx - 1])
        
        return pos.tolist(), vel.tolist()
    
    def set_look_side(self, look_side: str):
        """
        设置观测方向
        
        Args:
            look_side: 观测方向（LEFT/RIGHT）
        """
        self.look_side = look_side
    
    def set_bistatic_delay_correction(self, flag: bool):
        """
        设置双站延迟校正
        
        Args:
            flag: 是否启用双站延迟校正
        """
        self.bistatic_delay_correction_flag = flag
    
    def set_orbit_interpolation_method(self, method: str):
        """
        设置轨道插值方法
        
        Args:
            method: 轨道插值方法（HERMITE/SCH/LEGENDRE）
        """
        self.orbit_interpolation_method = method
    
    def set_doppler_polynomial(self, doppler_polynomial: List[float]):
        """
        设置多普勒多项式
        
        Args:
            doppler_polynomial: 多普勒多项式系数列表
        """
        self.doppler_polynomial = doppler_polynomial
    
    def set_doppler_derivative_polynomial(self, doppler_derivative_polynomial: List[float]):
        """
        设置多普勒导数多项式
        
        Args:
            doppler_derivative_polynomial: 多普勒导数多项式系数列表
        """
        self.doppler_derivative_polynomial = doppler_derivative_polynomial
    
    def set_fd1(self, fd1: float):
        """
        设置多普勒中心频率一阶系数
        
        Args:
            fd1: 多普勒中心频率一阶系数
        """
        self.fd1 = fd1
    
    def set_fdd1(self, fdd1: float):
        """
        设置多普勒中心频率二阶系数
        
        Args:
            fdd1: 多普勒中心频率二阶系数
        """
        self.fdd1 = fdd1
    
    def _calculate_doppler(self, slant_range: float) -> Tuple[float, float]:
        """
        计算多普勒频率和导数
        
        Args:
            slant_range: 斜距
            
        Returns:
            (多普勒频率, 多普勒频率导数)
        """
        return calculate_doppler(slant_range, self.doppler_polynomial, self.doppler_derivative_polynomial)

    def _interpolate_simple(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        在轨道数据不完整或退化时使用的简单线性模型。

        如果 ``orbit`` 字典包含速度则使用它。否则假定固定
       高度 500 km、速度 7 km/s 的直线运动。
        """
        if self.orbit and 'orbit_points' in self.orbit and len(self.orbit['orbit_points'])>0:
            first = self.orbit['orbit_points'][0]
            pos = np.array([first['position']['x'], first['position']['y'], first['position']['z']]) if 'position' in first else np.array([0.,0.,5e5])
            vel = np.array([first['velocity']['vx'], first['velocity']['vy'], first['velocity']['vz']]) if 'velocity' in first else np.array([0.,7e3,0])
            # linear extrapolate
            # use time offset relative to first timestamp if available
            try:
                import datetime
                base = datetime.datetime.fromisoformat(first['time'].replace('Z','+00:00')).timestamp()
                dt = time - base
            except Exception:
                dt = time
            return pos + vel * dt, vel
        # fallback constant values
        return np.array([0.,0.,5e5]), np.array([0.,7e3,0])    
    def llh_to_xyz(self, lat: float, lon: float, height: float) -> List[float]:
        """
        经纬度高程转换为地心坐标
        
        Args:
            lat: 纬度（度）
            lon: 经度（度）
            height: 高程（米）
            
        Returns:
            地心坐标 [x, y, z]
        """
        return llh_to_xyz(lat, lon, height, self.major_semi_axis, self.eccentricity_squared)
    
    def calculate_orbit_position(self, time: float) -> Tuple[List[float], List[float]]:
        """
        计算指定时间的轨道位置和速度（优化版：使用预计算数据）
        """
        if self._orbit_state_table is not None and self._orbit_state_times is not None:
            return self._get_orbit_from_table(time)
        
        if self._orbit_times is None:
            return self._interpolate_simple(time)
        
        try:
            t0 = self._orbit_t0
            relative_time = time - t0
            
            if self._orbit_time_range > 0:
                return self._interpolate_orbit_fast(relative_time)
            else:
                return self._interpolate_simple(time)
        except Exception:
            return self._interpolate_simple(time)
    
    def get_satellite_position(self, time: float) -> List[float]:
        """
        获取指定时间的卫星位置
        
        Args:
            time: 时间
            
        Returns:
            卫星位置 [x, y, z]
        """
        pos, _ = self.calculate_orbit_position(time)
        return pos
    
    def _get_orbit_from_table(self, time: float):
        """从预计算的轨道状态表快速获取卫星位置"""
        t0 = self._orbit_t0
        relative_time = time - t0
        
        if relative_time <= 0:
            return self._orbit_state_table['positions'][0].tolist(), self._orbit_state_table['velocities'][0].tolist()
        
        if relative_time >= self._orbit_time_range:
            return self._orbit_state_table['positions'][-1].tolist(), self._orbit_state_table['velocities'][-1].tolist()
        
        times = self._orbit_state_times
        positions = self._orbit_state_table['positions']
        velocities = self._orbit_state_table['velocities']
        
        idx = np.searchsorted(times, relative_time)
        
        if idx == 0:
            return positions[0].tolist(), velocities[0].tolist()
        if idx >= len(times):
            return positions[-1].tolist(), velocities[-1].tolist()
        
        t1 = times[idx - 1]
        t2 = times[idx]
        
        if t2 > t1:
            frac = (relative_time - t1) / (t2 - t1)
        else:
            frac = 0.0
        
        pos = positions[idx - 1] + frac * (positions[idx] - positions[idx - 1])
        vel = velocities[idx - 1] + frac * (velocities[idx] - velocities[idx - 1])
        
        return pos.tolist(), vel.tolist()
    
    def _find_nearest_orbit_point(self, target_xyz):
        """
        找到距离目标点最近的轨道点作为初始猜测值
        
        Args:
            target_xyz: 目标点的地心坐标
            
        Returns:
            最近轨道点的时间
        """
        if not hasattr(self, '_orbit_positions') or self._orbit_positions is None:
            # 计算sensing_end
            imaging_duration = 0.0
            if self.prf > 0 and hasattr(self, 'length') and self.length > 0:
                imaging_duration = (self.length - 1) / self.prf
            sensing_end = self.sensing_start + imaging_duration
            return (self.sensing_start + sensing_end) / 2
        
        # 计算所有轨道点到目标点的距离
        distances = np.linalg.norm(self._orbit_positions - target_xyz, axis=1)
        nearest_idx = np.argmin(distances)
        
        # 线性修正以提高初值精度（用户建议的改进）
        nearest_time = self._orbit_times[nearest_idx]
        sat_pos = self._orbit_positions[nearest_idx]
        sat_vel = self._orbit_velocities[nearest_idx]
        
        dr = target_xyz - sat_pos
        
        # 一次线性修正：t0 = nearest_time - (dr·v)/(v·v)
        # 物理意义：计算卫星需要多少时间才能使视线方向与速度方向垂直
        velocity_magnitude_squared = np.dot(sat_vel, sat_vel)
        if velocity_magnitude_squared > 1e-10:
            correction = -np.dot(dr, sat_vel) / velocity_magnitude_squared
            t0 = nearest_time + correction
        else:
            t0 = nearest_time
        
        # 确保 t0 在合理范围内
        t0 = max(self._orbit_times[0], min(self._orbit_times[-1], t0))
        
        return t0
    
    def geo2rdr(self, lat: float, lon: float, height: float, return_relative_time: bool = False) -> Tuple[float, float]:
        """
        地理坐标转换为雷达坐标（优化版）
        
        Args:
            lat: 纬度
            lon: 经度
            height: 高度（米）
            return_relative_time: 是否返回相对于orbit起始的相对时间，默认False返回绝对时间
            
        Returns:
            (range_sample, azimuth_time): 
                - range_sample: 距离向采样点
                - azimuth_time: 方位向时间（绝对时间或相对时间，取决于return_relative_time）
        """
        try:
            target_xyz = np.array(self.llh_to_xyz(lat, lon, height), dtype=np.float64)
            
            tmin = 0.0
            tmax = 0.0
            
            # 优先使用基于sensing_start的搜索窗口（更精确）
            if self.sensing_start > 0:
                imaging_duration = 0.0
                if self.prf > 0 and hasattr(self, 'length') and self.length > 0:
                    imaging_duration = (self.length - 1) / self.prf
                
                sensing_end = self.sensing_start + imaging_duration
                
                # 动态计算搜索窗口大小
                if self._orbit_time_range > 0:
                    orbit_start = self._orbit_t0
                    orbit_end = self._orbit_t0 + self._orbit_time_range
                    
                    # 计算搜索窗口，确保覆盖轨道和成像时间范围
                    tmin = min(self.sensing_start, orbit_start) - 30.0
                    tmax = max(sensing_end, orbit_end) + 30.0
                else:
                    # 使用固定边距
                    search_margin = 30.0
                    tmin = self.sensing_start - search_margin
                    tmax = sensing_end + search_margin
            elif self._orbit_time_range > 0:
                # 如果没有sensing_start，使用整个轨道时间范围
                tmin = self._orbit_t0
                tmax = self._orbit_t0 + self._orbit_time_range
                # 添加额外的时间边距
                tmin -= 10.0
                tmax += 10.0
            elif self.orbit and 'orbit_points' in self.orbit and len(self.orbit['orbit_points']) > 0:
                try:
                    import datetime
                    tmin = float(datetime.datetime.fromisoformat(self.orbit['orbit_points'][0]['time'].replace('Z','+00:00')).timestamp())
                    tmax = float(datetime.datetime.fromisoformat(self.orbit['orbit_points'][-1]['time'].replace('Z','+00:00')).timestamp())
                except Exception:
                    pass
            
            # 确保时间范围有效
            if tmax == tmin:
                if not getattr(self, '_time_warned', False):
                    print("警告: orbit 时间退化，使用单位间隔")
                    self._time_warned = True
                tmax = tmin + 10.0  # 使用更大的默认范围
            
            # 优化初始猜测值选择
            tmid = self._find_nearest_orbit_point(target_xyz)
            
            max_iter = 8  # 限制最大迭代次数
            tolerance = 1e-9  # 提高精度
            tline = tmid
            
            look_left = self.look_side == 'LEFT'
            damping_factor = 0.5  # 调整阻尼因子，提高收敛稳定性
            
            # Zero-Doppler 检查（步骤3.1）
            # 计算 dr = target_xyz - sat_pos(tmid)，检查 dr · sat_vel ≈ 0
            sat_pos_init, sat_vel_init = self.calculate_orbit_position(tline)
            sat_pos_init = np.array(sat_pos_init, dtype=np.float64)
            sat_vel_init = np.array(sat_vel_init, dtype=np.float64)
            dr_init = target_xyz - sat_pos_init
            doppler_check = np.dot(dr_init, sat_vel_init)
            
            zero_doppler_threshold = 1e-6  # 阈值（m²/s）
            
            # 若 |dr·v_s| < ε，则 tmid 已接近零多普勒时间，直接作为 azimuth_time
            use_zero_doppler_result = abs(doppler_check) < zero_doppler_threshold
            
            if use_zero_doppler_result:
                # Zero-Doppler 条件满足，直接使用 tmid，跳过迭代
                tline = tmid
            else:
                # 不满足 Zero-Doppler 条件，进入 Newton-Raphson 迭代
                
                # 记录最佳解
                best_tline = tline
                best_fn = float('inf')
                
                for k in range(max_iter):
                    tline = max(tmin, min(tmax, tline))
                    
                    try:
                        sat_pos_list, sat_vel_list = self.calculate_orbit_position(tline)
                        sat_pos = np.array(sat_pos_list, dtype=np.float64)
                        sat_vel = np.array(sat_vel_list, dtype=np.float64)
                        
                        dr = target_xyz - sat_pos
                        slant_range = np.linalg.norm(dr)
                        
                        if slant_range <= 1e-10 or np.isnan(slant_range):
                            tline = self._fallback_bisection_search(target_xyz, tmin, tmax, look_left)
                            break
                        
                        dopfact = np.dot(dr, sat_vel)
                        fdop, fdopder = self._calculate_doppler(slant_range)
                        
                        if look_left:
                            fdop = -fdop
                            fdopder = -fdopder
                            if fdopder > 0:
                                fdopder = -fdopder
                
                        fn = dopfact - fdop * slant_range
                        
                        # 更新最佳解
                        if abs(fn) < abs(best_fn):
                            best_fn = fn
                            best_tline = tline
                        
                        # 计算卫星加速度（通过差分）
                        dt_acc = 1e-3
                        sat_pos_next, sat_vel_next = self.calculate_orbit_position(tline + dt_acc)
                        sat_vel_next = np.array(sat_vel_next, dtype=np.float64)
                        sat_acc = (sat_vel_next - sat_vel) / dt_acc
                        
                        # 完整的fnprime公式：f'(t) = -||v_s||² + dr·a_s + (f_d/R + f_d') * (dr·v_s)
                        # 其中 dr = target_xyz - sat_pos, v_s = sat_vel, a_s = sat_acc
                        dr = target_xyz - sat_pos
                        dr_dot_a = np.dot(dr, sat_acc)
                        
                        c1 = -np.dot(sat_vel, sat_vel) + dr_dot_a
                        c2 = (fdop / slant_range) + fdopder
                        fnprime = c1 + c2 * dopfact
                    except Exception as e:
                        tline = self._fallback_bisection_search(target_xyz, tmin, tmax, look_left)
                        break
                    
                    if abs(fnprime) > 1e-10:
                        # 添加阻尼因子，提高收敛稳定性
                        delta_t = (fn / fnprime) * damping_factor
                        
                        # 限制每次时间修正量，防止发散
                        max_step = 1.0  # 限制每次修正不超过1秒
                        if abs(delta_t) > max_step:
                            delta_t = np.sign(delta_t) * max_step
                        
                        tline -= delta_t
                        
                        # 收敛检查
                        if abs(delta_t) < tolerance and abs(fn) < tolerance * slant_range:
                            break
                    else:
                        # 导数为零，尝试使用二分法
                        tline = self._fallback_bisection_search(target_xyz, tmin, tmax, look_left)
                        break
            
            # 如果没有收敛，使用最佳解或二分法
            if not use_zero_doppler_result and k == max_iter - 1:
                # 尝试使用二分法作为最终fallback
                tline = self._fallback_bisection_search(target_xyz, tmin, tmax, look_left)
            
            # ===== 限制Newton求解时间在轨道时间范围内 =====
            # 确保求解时间在轨道时间范围内
            if hasattr(self, '_orbit_t0') and hasattr(self, '_orbit_time_range'):
                orbit_time_start = self._orbit_t0
                orbit_time_end = self._orbit_t0 + self._orbit_time_range
                tline = max(orbit_time_start, min(orbit_time_end, tline))
            
            if self.bistatic_delay_correction_flag:
                # 计算当前的斜距用于双站延迟校正
                sat_pos_temp, _ = self.calculate_orbit_position(tline)
                sat_pos_temp = np.array(sat_pos_temp, dtype=np.float64)
                dr_temp = target_xyz - sat_pos_temp
                slant_range_temp = np.linalg.norm(dr_temp)
                c = 3e8
                tline += 2 * slant_range_temp / c
                # 再次限制时间范围
                if hasattr(self, '_orbit_t0') and hasattr(self, '_orbit_time_range'):
                    orbit_time_start = self._orbit_t0
                    orbit_time_end = self._orbit_t0 + self._orbit_time_range
                    tline = max(orbit_time_start, min(orbit_time_end, tline))
            
            # 计算最终的卫星位置和斜距
            sat_pos, _ = self.calculate_orbit_position(tline)
            sat_pos = np.array(sat_pos, dtype=np.float64)
            dr = target_xyz - sat_pos
            slant_range = np.linalg.norm(dr)
            
            # 侧视方向处理（参考GMTSAR的SAT_llt2rat.c）
            # 计算轨道切线方向（使用相邻时间点）
            dt_small = 1e-6
            sat_pos_prev, _ = self.calculate_orbit_position(tline - dt_small)
            sat_pos_prev = np.array(sat_pos_prev, dtype=np.float64)
            
            vec_v = sat_pos - sat_pos_prev           # 轨道切线方向
            vec_r = target_xyz - sat_pos             # 从卫星到目标的向量
            
            # 计算轨道法向：cross(sat_pos, vec_v)
            orbit_normal = np.cross(sat_pos, vec_v)
            
            # 判断目标在轨道哪一侧
            det = np.dot(vec_r, orbit_normal)
            
            # 结合look方向：det_sign = sign(det) * lookdir
            lookdir = -1 if self.look_side == 'LEFT' else 1
            det_sign = np.sign(det) * lookdir if det != 0 else lookdir
            
            # 斜距始终保持正数
            slant_range = np.linalg.norm(vec_r)
            
            # 参考GMTSAR的多普勒校正实现
            if hasattr(self, 'fd1') and self.fd1 != 0.0:
                try:
                    # 计算多普勒中心频率
                    if hasattr(self, 'fdd1'):
                        mid_range = self.range_first_sample + self.slant_range_pixel_spacing * getattr(self, 'width', 1000) / 2.0
                        dopc = self.fd1 + self.fdd1 * mid_range
                    else:
                        dopc = self.fd1
                    
                    # 计算卫星速度
                    _, sat_vel = self.calculate_orbit_position(tline)
                    sat_vel = np.array(sat_vel, dtype=np.float64)
                    vel_mag = np.linalg.norm(sat_vel)
                    
                    # 多普勒校正（使用det_sign）
                    rdd = (vel_mag ** 2) / slant_range
                    if rdd > 1e-10 and hasattr(self, 'radar_wavelength') and self.radar_wavelength > 0:
                        daa = -0.5 * (self.radar_wavelength * dopc * det_sign) / rdd
                        drr = 0.5 * rdd * daa * daa / self.slant_range_pixel_spacing
                        daa = self.prf * daa
                        
                        # 应用校正
                        tline += daa / self.prf  # 转换为时间
                        slant_range += drr * self.slant_range_pixel_spacing
                except Exception:
                    pass
            
            if self.slant_range_pixel_spacing > 0:
                range_sample = (slant_range - self.range_first_sample) / self.slant_range_pixel_spacing
            else:
                range_sample = 0.0
            
            # 如果需要返回相对时间，则转换为相对于orbit起始的相对时间
            if return_relative_time and hasattr(self, '_orbit_t0'):
                tline = tline - self._orbit_t0
            
            return range_sample, tline
        except Exception as e:
            # 发生错误时返回无效值
            return -1, -1
    
    def rdr2geo(self, slant_range: float, azimuth_time: float, dem_extent=None) -> Tuple[float, float, float]:
        """
        雷达坐标转换为地理坐标
        
        Args:
            slant_range: 斜距（米）
            azimuth_time: 方位时间（秒）
            dem_extent: DEM范围 (min_lon, max_lon, min_lat, max_lat)
            
        Returns:
            (lat, lon, height): 纬度、经度、高程
        """
        # 计算卫星位置
        sat_pos, sat_vel = self.calculate_orbit_position(azimuth_time)
        sat_pos = np.array(sat_pos, dtype=np.float64)
        
        # 计算视线方向 - 基于卫星轨道几何
        look_dir = None
        if len(sat_vel) > 0:
            sat_vel_norm = np.linalg.norm(sat_vel)
            if sat_vel_norm > 0:
                # 卫星前进方向（ECEF坐标系中的速度方向）
                velocity_unit = np.array(sat_vel) / sat_vel_norm
                
                # 径向方向（从地心指向卫星）
                radial_dir = np.array(sat_pos) / np.linalg.norm(sat_pos)
                
                # 右向 = 径向方向 × 前进方向
                right_dir = np.cross(radial_dir, velocity_unit)
                right_dir = right_dir / np.linalg.norm(right_dir)
                
                # 根据look_side选择视线方向
                if self.look_side == 'LEFT':
                    look_dir = -right_dir
                else:
                    look_dir = right_dir
        
        # 如果无法计算，使用默认方向
        if look_dir is None:
            # 默认指向地球中心方向
            earth_center_dir = -sat_pos / np.linalg.norm(sat_pos)
            look_dir = earth_center_dir
        
        # 根据斜距计算地面点
        target_xyz = sat_pos + look_dir * slant_range
        
        # 转换为地理坐标
        target_lat, target_lon, target_h = xyz_to_llh(target_xyz[0], target_xyz[1], target_xyz[2])
        
        return target_lat, target_lon, target_h
    
    def _geo2rdr_scalar_loop(self, lats: np.ndarray, lons: np.ndarray, heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        标量循环版本的批量地理坐标转换（使用稳健的标量geo2rdr）
        """
        n = len(lats)
        range_samples = np.zeros(n, dtype=np.float64)
        azimuth_times = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            try:
                r, a = self.geo2rdr(lats[i], lons[i], heights[i])
                range_samples[i] = r
                azimuth_times[i] = a
            except Exception:
                range_samples[i] = -1
                azimuth_times[i] = -1
        
        return range_samples, azimuth_times
    
    def geo2rdr_batch(self, lats: np.ndarray, lons: np.ndarray, heights: np.ndarray,
                      n_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量地理坐标转换为雷达坐标（向量化版本）
        
        Args:
            lats: 纬度数组
            lons: 经度数组  
            heights: 高程数组
            n_workers: 并行worker数量，默认使用CPU核心数
            
        Returns:
            (距离向采样点数组, 方位向时间数组)
        """
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)
        heights = np.asarray(heights, dtype=np.float64)
        
        n_points = len(lats)
        
        if n_workers is None:
            n_workers = max(1, cpu_count())
        
        # 使用标量版本的geo2rdr()循环，避免向量化版本在某些点上不收敛导致空行
        if n_points < 1000 or n_workers <= 1:
            return self._geo2rdr_scalar_loop(lats, lons, heights)
        
        chunk_size = max(1000, n_points // n_workers)
        
        orbit_data = getattr(self, 'orbit', None)
        orbit_interp_method = getattr(self, 'orbit_interpolation_method', 'HERMITE')
        dop_poly = getattr(self, 'doppler_polynomial', None)
        dop_der_poly = getattr(self, 'doppler_derivative_polynomial', None)
        length = getattr(self, 'length', 0)
        width = getattr(self, 'width', 0)
        fd1 = getattr(self, 'fd1', 0.0)
        fdd1 = getattr(self, 'fdd1', 0.0)
        
        args_list = []
        for i in range(0, n_points, chunk_size):
            end = min(i + chunk_size, n_points)
            args = (
                lats[i:end], lons[i:end], heights[i:end],
                orbit_data, orbit_interp_method, dop_poly, dop_der_poly,
                self.prf, self.radar_wavelength, self.slant_range_pixel_spacing,
                self.range_first_sample, self.look_side, self.sensing_start,
                length, width, fd1, fdd1
            )
            args_list.append(args)
        
        results = []
        with Pool(n_workers) as pool:
            results = pool.map(Geo2rdr._geo2rdr_chunk_worker_static, args_list)
        
        range_samples = np.concatenate([r[0] for r in results])
        azimuth_times = np.concatenate([r[1] for r in results])
        
        return range_samples, azimuth_times
    
    @staticmethod
    def _geo2rdr_chunk_worker_static(args):
        """多进程工作函数（静态方法）"""
        lats, lons, heights, orbit_data, orbit_interp_method, dop_poly, dop_der_poly, prf, wavelength, range_spacing, range_first, look_side, sensing_start, length, width, fd1, fdd1 = args
        
        # print(f"    [GEO2RDR WORKER] look_side={look_side}, sensing_start={sensing_start:.3f}, length={length}, width={width}, fd1={fd1}, fdd1={fdd1}")
        
        g2r = Geo2rdr(
            prf=prf,
            radar_wavelength=wavelength,
            slant_range_pixel_spacing=range_spacing,
            range_first_sample=range_first,
            sensing_start=sensing_start,
            look_side=look_side,
            orbit_interpolation_method=orbit_interp_method,
            fd1=fd1,
            fdd1=fdd1
        )
        
        g2r.length = length
        g2r.width = width
        
        if orbit_data:
            g2r.set_orbit(orbit_data)
        
        if dop_poly:
            g2r.doppler_polynomial = dop_poly
        if dop_der_poly:
            g2r.doppler_derivative_polynomial = dop_der_poly
        
        n = len(lats)
        range_samples = np.zeros(n, dtype=np.float64)
        azimuth_times = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            try:
                r, a = g2r.geo2rdr(lats[i], lons[i], heights[i])
                range_samples[i] = r
                azimuth_times[i] = a
            except Exception:
                range_samples[i] = -1
                azimuth_times[i] = -1
        
        return range_samples, azimuth_times
    
    def _geo2rdr_chunk_worker(self, lats, lons, heights):
        """批量处理的工作函数（实例方法）"""
        n = len(lats)
        range_samples = np.zeros(n, dtype=np.float64)
        azimuth_times = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            try:
                r, a = self.geo2rdr(lats[i], lons[i], heights[i])
                range_samples[i] = r
                azimuth_times[i] = a
            except Exception:
                range_samples[i] = -1
                azimuth_times[i] = -1
        
        return range_samples, azimuth_times
    
    def _geo2rdr_vectorized(self, lats: np.ndarray, lons: np.ndarray, heights: np.ndarray):
        """
        向量化批量处理（不使用多进程）
        """
        n = len(lats)
        range_samples = np.zeros(n, dtype=np.float64)
        azimuth_times = np.zeros(n, dtype=np.float64)
        
        target_xyzs = self._llh_to_xyz_vectorized(lats, lons, heights)
        
        tmin = 0.0
        tmax = 0.0
        
        search_margin = 5.0
        
        if self.sensing_start > 0:
            # 计算成像时间
            imaging_duration = 0.0
            if self.prf > 0 and hasattr(self, 'length') and self.length > 0:
                imaging_duration = (self.length - 1) / self.prf
            
            # 计算成像结束时间
            sensing_end = self.sensing_start + imaging_duration
            
            search_margin = 120.0  # 增大搜索窗口边距到120秒，以覆盖轨道时间与成像时间的偏移
            tmin = self.sensing_start - search_margin
            tmax = sensing_end + search_margin
            
            if self._orbit_time_range > 0:
                orbit_start = self._orbit_t0
                orbit_end = self._orbit_t0 + self._orbit_time_range
                if orbit_start <= self.sensing_start <= orbit_end:
                    tmin = max(tmin, orbit_start)
                    tmax = min(tmax, orbit_end)
        elif self._orbit_time_range > 0:
            tmin = self._orbit_t0
            tmax = self._orbit_t0 + self._orbit_time_range
        
        if tmax == tmin:
            tmax = tmin + 1.0
        
        if not getattr(self, '_geo2rdr_debug_printed', False):
            print(f"    [GEO2RDR DEBUG] vectorized搜索窗口: tmin={tmin:.3f}, tmax={tmax:.3f}, sensing_start={self.sensing_start:.3f}, look_side={self.look_side}")
            self._geo2rdr_debug_printed = True
        
        look_left = self.look_side == 'LEFT'
        max_iter = 15
        tolerance = 1e-4
        
        for idx in range(n):
            target_xyz = target_xyzs[:, idx]
            tmid = (tmin + tmax) / 2
            tline = tmid
            
            for _ in range(max_iter):
                tline = max(tmin, min(tmax, tline))
                
                sat_pos_list, _ = self.calculate_orbit_position(tline)
                sat_pos = np.array(sat_pos_list, dtype=np.float64)
                
                dr = target_xyz - sat_pos
                slant_range = np.linalg.norm(dr)
                
                if slant_range <= 0:
                    break
                
                sat_vel_list = self._get_velocity_fast(tline)
                sat_vel = np.array(sat_vel_list, dtype=np.float64)
                
                dopfact = np.dot(dr, sat_vel)
                fdop, _ = self._calculate_doppler(slant_range)
                
                if look_left:
                    fdop = -fdop
                
                fn = dopfact - fdop * slant_range
                
                if abs(fn) < tolerance * slant_range:
                    break
                
                c1 = -np.dot(sat_vel, sat_vel)
                if abs(c1) > 1e-10:
                    delta_t = fn / c1
                    tline -= delta_t
            
            range_samples[idx] = (slant_range - self.range_first_sample) / self.slant_range_pixel_spacing if self.slant_range_pixel_spacing > 0 else 0
            azimuth_times[idx] = tline
        
        # 调试：检查返回值的异常情况
        nan_count = np.sum(np.isnan(azimuth_times))
        if nan_count > 0:
            print(f"    [GEO2RDR DEBUG] 发现 {nan_count} 个 NaN azimuth_times!")
        
        # 检查是否有超出合理范围的azimuth时间
        if self.sensing_start > 0:
            az_pixels = (azimuth_times - self.sensing_start) * self.prf if self.prf > 0 else azimuth_times
            valid_az = (az_pixels > -100000) & (az_pixels < 100000)
            invalid_count = np.sum(~valid_az)
            if invalid_count > 0:
                print(f"    [GEO2RDR DEBUG] 发现 {invalid_count} 个超出范围的azimuth像素值!")
        
        return range_samples, azimuth_times
    
    def _llh_to_xyz_vectorized(self, lats, lons, heights):
        """向量化 LLH → XYZ"""
        lats = np.radians(lats)
        lons = np.radians(lons)
        
        a = self.major_semi_axis
        e2 = self.eccentricity_squared
        
        N = a / np.sqrt(1 - e2 * np.sin(lats)**2)
        
        x = (N + heights) * np.cos(lats) * np.cos(lons)
        y = (N + heights) * np.cos(lats) * np.sin(lons)
        z = (N * (1 - e2) + heights) * np.sin(lats)
        
        return np.vstack([x, y, z])
    
    def _get_velocity_fast(self, time):
        """快速获取速度"""
        if self._orbit_state_table is not None and self._orbit_state_times is not None:
            return self._get_orbit_from_table(time)[1]
        
        if self._orbit_times is not None:
            _, vel = self._interpolate_orbit_fast(time - self._orbit_t0)
            return vel
        
        return [0, 7000, 0]
    
    def _fallback_bisection_search(self, target_xyz, tmin, tmax, look_left):
        """
        二分法搜索作为牛顿-拉夫逊迭代的fallback
        
        Args:
            target_xyz: 目标点的地心坐标
            tmin: 搜索时间下界
            tmax: 搜索时间上界
            look_left: 是否为左视
            
        Returns:
            找到的方位时间
        """
        max_bisection_iter = 30
        tolerance = 1e-9
        
        for _ in range(max_bisection_iter):
            tmid = (tmin + tmax) / 2
            
            # 计算卫星位置和速度
            sat_pos_list, sat_vel_list = self.calculate_orbit_position(tmid)
            sat_pos = np.array(sat_pos_list, dtype=np.float64)
            sat_vel = np.array(sat_vel_list, dtype=np.float64)
            
            # 计算目标点到卫星的向量和斜距
            dr = target_xyz - sat_pos
            slant_range = np.linalg.norm(dr)
            if slant_range <= 0 or np.isnan(slant_range):
                tmin = tmid
                continue
            
            # 计算多普勒因子和多普勒频率
            dopfact = np.dot(dr, sat_vel)
            fdop, _ = self._calculate_doppler(slant_range)
            # print(f"    [GEO2RDR DEBUG] 二分法: tmid={tmid:.6f}, slant_range={slant_range:.2f}, 原始fdop={fdop:.2f}")
            
            # 根据观测方向调整多普勒频率
            if look_left:
                fdop = -fdop
                # print(f"    [GEO2RDR DEBUG] 二分法左视调整后: fdop={fdop:.2f}")
            
            # 计算残差
            fn = dopfact - fdop * slant_range
            # print(f"    [GEO2RDR DEBUG] 二分法: dopfact={dopfact:.2f}, fn={fn:.2f}")
            
            # 根据残差符号调整搜索区间
            if fn > 0:
                tmax = tmid
            else:
                tmin = tmid
            
            # 检查收敛
            if abs(tmax - tmin) < tolerance:
                break
        
        return (tmin + tmax) / 2
    
    @jit(nopython=True)
    def _geo2rdr_numba(self, lat, lon, height, major_semi_axis, eccentricity_squared, slant_range_pixel_spacing, range_first_sample):
        """
        Numba优化的geo2rdr计算
        """
        # 将地理坐标转换为地心坐标
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        N = major_semi_axis / np.sqrt(1 - eccentricity_squared * np.sin(lat_rad)**2)
        
        target_xyz = np.array([
            (N + height) * np.cos(lat_rad) * np.cos(lon_rad),
            (N + height) * np.cos(lat_rad) * np.sin(lon_rad),
            (N * (1 - eccentricity_squared) + height) * np.sin(lat_rad)
        ])
        
        # 初始时间估计
        tmid = 0.0
        
        # 牛顿-拉夫逊迭代
        max_iter = 51
        tolerance = 5.0e-9
        tline = tmid
        
        for k in range(max_iter):
            # 计算当前时间的卫星位置和速度（简化）
            sat_pos = np.array([0.0, 0.0, 500000.0])
            sat_vel = np.array([0.0, 7000.0, 0.0])
            
            # 计算视线向量
            dr = target_xyz - sat_pos
            slant_range = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            
            # 计算多普勒因子
            dopfact = dr[0] * sat_vel[0] + dr[1] * sat_vel[1] + dr[2] * sat_vel[2]
            
            # 计算多普勒频率和导数（简化）
            fdop = 0.0
            fdopder = 0.0
            
            # 计算函数值和导数
            fn = dopfact - fdop * slant_range
            
    
    def _process_row(self, row_data):
        """
        处理一行数据
        """
        row_idx, lat_row, lon_row, dem_row = row_data
        range_sample = np.zeros_like(lat_row)
        azimuth_time = np.zeros_like(lat_row)
        
        for j in range(len(lat_row)):
            range_sample[j], azimuth_time[j] = self.geo2rdr(lat_row[j], lon_row[j], dem_row[j])
        
        return row_idx, range_sample, azimuth_time
    
    def process(self, lat_image: np.ndarray, lon_image: np.ndarray, dem_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理整个图像
        
        Args:
            lat_image: 纬度图像
            lon_image: 经度图像
            dem_image: DEM图像
            
        Returns:
            (距离向图像, 方位向时间图像)
        """
        if lat_image.shape != lon_image.shape or lat_image.shape != dem_image.shape:
            raise Exception("输入图像尺寸不匹配")
        
        rows, cols = lat_image.shape
        range_image = np.zeros((rows, cols), dtype=np.float64)
        azimuth_image = np.zeros((rows, cols), dtype=np.float64)
        
        # 计算使用的CPU核心数（80%）
        num_cores = max(1, int(cpu_count() * 0.8))
        
        # 准备并行任务
        tasks = []
        for i in range(rows):
            tasks.append((i, lat_image[i, :], lon_image[i, :], dem_image[i, :]))
        
        # 并行处理
        with Pool(num_cores) as pool:
            results = pool.map(self._process_row, tasks)
        
        # 填充结果
        for row_idx, range_row, azimuth_row in results:
            range_image[row_idx, :] = range_row
            azimuth_image[row_idx, :] = azimuth_row
        
        return range_image, azimuth_image


def read_orbit_from_yaml(yaml_file: str) -> Dict:
    """
    从YAML文件读取轨道信息
    
    Args:
        yaml_file: YAML文件路径
        
    Returns:
        轨道信息
    """
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if 'orbit_data' in data:
        return data['orbit_data']
    else:
        return {}


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='geo2rdr工具')
    parser.add_argument('lat_file', help='纬度文件路径')
    parser.add_argument('lon_file', help='经度文件路径')
    parser.add_argument('dem_file', help='DEM文件路径')
    parser.add_argument('output_range', help='输出距离向文件路径')
    parser.add_argument('output_azimuth', help='输出方位向文件路径')
    parser.add_argument('--yaml-file', help='YAML文件路径，用于读取轨道信息')
    
    args = parser.parse_args()
    
    print("=== geo2rdr工具 ===")
    print(f"纬度文件: {args.lat_file}")
    print(f"经度文件: {args.lon_file}")
    print(f"DEM文件: {args.dem_file}")
    print(f"输出距离向文件: {args.output_range}")
    print(f"输出方位向文件: {args.output_azimuth}")
    
    try:
        # 读取输入文件
        lat_image = np.loadtxt(args.lat_file)
        lon_image = np.loadtxt(args.lon_file)
        dem_image = np.loadtxt(args.dem_file)
        
        # 初始化geo2rdr
        g2r = Geo2rdr()
        
        # 如果提供YAML文件，从中读取轨道和雷达参数
        if args.yaml_file:
            orbit = read_orbit_from_yaml(args.yaml_file)
            g2r.set_orbit(orbit)
            try:
                with open(args.yaml_file, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                # radar parameters
                imgp = cfg.get('image_parameters', {})
                prm = cfg.get('prm_parameters', {})
                orbitp = cfg.get('orbit_parameters', {})
                if 'prf' in imgp:
                    g2r.prf = imgp['prf']
                if 'radar_wavelength' in prm:
                    g2r.radar_wavelength = prm['radar_wavelength']
                if 'range_sampling_rate' in prm:
                    # compute spacing
                    c = 3e8
                    g2r.slant_range_pixel_spacing = c/(2*prm['range_sampling_rate'])
                if 'near_range' in prm:
                    g2r.range_first_sample = prm['near_range']
                if 'sensing_start' in imgp:
                    g2r.sensing_start = imgp['sensing_start']
                if 'look_side' in cfg:
                    g2r.look_side = cfg['look_side'].upper()
                if 'orbit_interpolation_method' in cfg:
                    g2r.orbit_interpolation_method = cfg['orbit_interpolation_method'].upper()
                # 读取多普勒参数
                if 'fd1' in prm:
                    g2r.fd1 = prm['fd1']
                if 'fdd1' in prm:
                    g2r.fdd1 = prm['fdd1']
            except Exception:
                pass
        
        # 处理数据
        range_image, azimuth_image = g2r.process(lat_image, lon_image, dem_image)
        
        # 保存结果
        np.savetxt(args.output_range, range_image)
        np.savetxt(args.output_azimuth, azimuth_image)
        
        print("处理完成！")
        print(f"距离向结果保存到: {args.output_range}")
        print(f"方位向结果保存到: {args.output_azimuth}")
    except Exception as e:
        print(f"处理失败: {e}")


if __name__ == '__main__':
    main()
