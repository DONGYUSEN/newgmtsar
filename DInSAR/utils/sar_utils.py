#!/usr/bin/env python3
"""
SAR 工具函数模块
功能：提供 SAR 处理中常用的工具函数
"""

import numpy as np
import yaml
from typing import Tuple, List, Dict, Optional

# 尝试导入Numba，用于性能优化
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = lambda *args, **kwargs: lambda f: f
    prange = range


@njit(fastmath=True, cache=True)
def llh_to_xyz(lat: float, lon: float, height: float, 
               major_semi_axis: float = 6378137.0, 
               eccentricity_squared: float = 0.00669437999014) -> List[float]:
    """
    经纬度高程转换为地心坐标
    
    Args:
        lat: 纬度（度）
        lon: 经度（度）
        height: 高程（米）
        major_semi_axis: 地球长半轴（米）
        eccentricity_squared: 地球第一偏心率平方
        
    Returns:
        地心坐标 [x, y, z]
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    N = major_semi_axis / np.sqrt(1 - eccentricity_squared * np.sin(lat_rad)**2)
    
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - eccentricity_squared) + height) * np.sin(lat_rad)
    
    return [x, y, z]


@njit(fastmath=True, cache=True)
def xyz_to_llh(x: float, y: float, z: float, 
               major_semi_axis: float = 6378137.0, 
               eccentricity_squared: float = 0.00669437999014) -> List[float]:
    """
    地心坐标转换为经纬度高程
    
    Args:
        x: 地心X坐标（米）
        y: 地心Y坐标（米）
        z: 地心Z坐标（米）
        major_semi_axis: 地球长半轴（米）
        eccentricity_squared: 地球第一偏心率平方
        
    Returns:
        [纬度（度）, 经度（度）, 高程（米）]
    """
    a = major_semi_axis
    e2 = eccentricity_squared
    
    # 计算经度
    lon = np.arctan2(y, x)
    
    # 计算纬度（使用迭代法）
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * (a * (1 - e2)))
    lat = np.arctan2(z + e2 * a * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    
    # 计算N和高程
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    height = (p / np.cos(lat)) - N
    
    # 转换为度
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    
    return [lat_deg, lon_deg, height]


@njit(parallel=True, fastmath=True, cache=True)
def llh_to_xyz_vectorized(lat: np.ndarray, lon: np.ndarray, height: np.ndarray,
                          major_semi_axis: float = 6378137.0, 
                          eccentricity_squared: float = 0.00669437999014) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    经纬度高程转换为地心坐标（向量化版本）
    
    Args:
        lat: 纬度数组（度）
        lon: 经度数组（度）
        height: 高程数组（米）
        major_semi_axis: 地球长半轴（米）
        eccentricity_squared: 地球第一偏心率平方
        
    Returns:
        (x, y, z) 地心坐标数组
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


@njit(parallel=True, fastmath=True, cache=True)
def xyz_to_llh_vectorized(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         major_semi_axis: float = 6378137.0, 
                         eccentricity_squared: float = 0.00669437999014) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    地心坐标转换为经纬度高程（向量化版本）
    
    Args:
        x: 地心X坐标数组（米）
        y: 地心Y坐标数组（米）
        z: 地心Z坐标数组（米）
        major_semi_axis: 地球长半轴（米）
        eccentricity_squared: 地球第一偏心率平方
        
    Returns:
        (lat, lon, height) 纬度、经度、高程数组
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    a = major_semi_axis
    e2 = eccentricity_squared
    
    # 计算经度
    lon = np.arctan2(y, x)
    
    # 计算纬度（使用迭代法）
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * (a * (1 - e2)))
    lat = np.arctan2(z + e2 * a * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    
    # 计算N和高程
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    height = (p / np.cos(lat)) - N
    
    # 转换为度
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    
    return lat_deg, lon_deg, height


def orbit_hermite(positions: List[List[float]], velocities: List[List[float]], times: List[float], time: float) -> Tuple[List[float], List[float]]:
    """
    4点Hermite多项式轨道插值
    参考ISCE2的orbitHermite实现
    
    Args:
        positions: 4个轨道点的位置
        velocities: 4个轨道点的速度
        times: 4个轨道点的时间
        time: 要插值的时间
        
    Returns:
        (插值后的位置, 插值后的速度)
    """
    n1 = 4
    n2 = 3
    
    # 计算基函数
    h = [0.0] * n1
    hdot = [0.0] * n1
    f0 = [0.0] * n1
    f1 = [0.0] * n1
    g0 = [0.0] * n1
    g1 = [0.0] * n1
    
    for i in range(n1):
        # 计算f0和f1
        f1[i] = time - times[i]
        sum_val = 0.0
        for j in range(n1):
            if i != j:
                sum_val += 1.0 / (times[i] - times[j])
        f0[i] = 1.0 - 2.0 * f1[i] * sum_val
        
        # 计算g0和g1
        g0[i] = f1[i]
        g1[i] = f1[i] * f1[i]
        
        # 计算h
        product = 1.0
        for k in range(n1):
            if k != i:
                product *= (time - times[k]) / (times[i] - times[k])
        h[i] = product
        
        # 计算hdot
        sum_val = 0.0
        for j in range(n1):
            if j != i:
                product = 1.0
                for k in range(n1):
                    if k != i and k != j:
                        product *= (time - times[k]) / (times[i] - times[k])
                sum_val += product / (times[i] - times[j])
        hdot[i] = sum_val
    
    # 计算插值结果
    pos = [0.0] * n2
    vel = [0.0] * n2
    
    for k in range(n2):
        sum_pos = 0.0
        sum_vel = 0.0
        for i in range(n1):
            sum_pos += (positions[i][k] * f0[i] + velocities[i][k] * f1[i]) * h[i] * h[i]
            sum_vel += (positions[i][k] * (2 * h[i] * hdot[i] * f0[i] + h[i] * h[i] * (-2 * sum_val)) + 
                       velocities[i][k] * (2 * h[i] * hdot[i] * f1[i] + h[i] * h[i]))
        pos[k] = sum_pos
        vel[k] = sum_vel
    
    return pos, vel


def interpolate_wgs84_orbit(orbit_times: List[float], orbit_positions: List[List[float]], 
                           orbit_velocities: List[List[float]], time: float) -> Tuple[List[float], List[float]]:
    """
    WGS84轨道插值（使用4点Hermite多项式）
    参考ISCE2的interpolateWGS84Orbit实现
    
    Args:
        orbit_times: 轨道点时间
        orbit_positions: 轨道点位置
        orbit_velocities: 轨道点速度
        time: 要插值的时间
        
    Returns:
        (插值后的位置, 插值后的速度)
    """
    n_vectors = len(orbit_times)
    
    if n_vectors < 4:
        # 向量不足4个，使用线性插值
        return linear_interpolation(orbit_times, orbit_positions, orbit_velocities, time)
    
    # 找到时间所在的区间
    i = 0
    while i < n_vectors and orbit_times[i] < time:
        i += 1
    
    # 选择4个点进行插值
    i -= 2
    if i < 0:
        i = 0
    if i > n_vectors - 4:
        i = n_vectors - 4
    
    # 提取4个点的时间、位置和速度
    times = orbit_times[i:i+4]
    positions = orbit_positions[i:i+4]
    velocities = orbit_velocities[i:i+4]
    
    # 使用Hermite插值
    return orbit_hermite(positions, velocities, times, time)


def interpolate_sch_orbit(orbit_times: List[float], orbit_positions: List[List[float]], 
                         orbit_velocities: List[List[float]], time: float) -> Tuple[List[float], List[float]]:
    """
    SCH（Simplified Cubic Hermite）轨道插值
    参考ISCE2的interpolateSCHOrbit实现
    
    Args:
        orbit_times: 轨道点时间
        orbit_positions: 轨道点位置
        orbit_velocities: 轨道点速度
        time: 要插值的时间
        
    Returns:
        (插值后的位置, 插值后的速度)
    """
    n_vectors = len(orbit_times)
    
    if n_vectors < 2:
        # 向量不足2个，返回第一个点
        return orbit_positions[0], orbit_velocities[0] if orbit_velocities else [0, 7000, 0]
    
    # 初始化输出
    pos = [0.0, 0.0, 0.0]
    vel = [0.0, 0.0, 0.0]
    
    # SCH插值
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
                frac *= num / den
        
        # 累加结果
        for k in range(3):
            pos[k] += frac * pos_i[k]
            vel[k] += frac * vel_i[k]
    
    return pos, vel


def linear_interpolation(orbit_times: List[float], orbit_positions: List[List[float]], 
                        orbit_velocities: List[List[float]], time: float) -> Tuple[List[float], List[float]]:
    """
    线性轨道插值
    
    Args:
        orbit_times: 轨道点时间
        orbit_positions: 轨道点位置
        orbit_velocities: 轨道点速度
        time: 要插值的时间
        
    Returns:
        (插值后的位置, 插值后的速度)
    """
    n_vectors = len(orbit_times)
    
    if n_vectors < 2:
        # 只有一个轨道点，直接返回
        return orbit_positions[0], orbit_velocities[0] if orbit_velocities else [0, 7000, 0]
    
    # 找到时间所在的区间
    for i in range(n_vectors - 1):
        if orbit_times[i] <= time <= orbit_times[i+1]:
            # 线性插值
            t1 = orbit_times[i]
            t2 = orbit_times[i+1]
            t = (time - t1) / (t2 - t1)
            
            # 位置插值
            pos1 = orbit_positions[i]
            pos2 = orbit_positions[i+1]
            pos = [pos1[j] + t * (pos2[j] - pos1[j]) for j in range(3)]
            
            # 速度插值
            if len(orbit_velocities) >= 2:
                vel1 = orbit_velocities[i]
                vel2 = orbit_velocities[i+1]
                vel = [vel1[j] + t * (vel2[j] - vel1[j]) for j in range(3)]
            else:
                vel = [0, 7000, 0]
            
            return pos, vel
    
    # 时间超出范围，返回最近的轨道点
    if time < orbit_times[0]:
        return orbit_positions[0], orbit_velocities[0] if orbit_velocities else [0, 7000, 0]
    else:
        return orbit_positions[-1], orbit_velocities[-1] if orbit_velocities else [0, 7000, 0]


def calculate_orbit_position(time: float, orbit: Dict, 
                            default_position: List[float] = [0, 0, 500000], 
                            default_velocity: List[float] = [0, 7000, 0],
                            interpolation_method: str = 'HERMITE') -> Tuple[List[float], List[float]]:
    """
    计算指定时间的卫星位置和速度
    参考ISCE2的轨道插值实现
    
    Args:
        time: 时间
        orbit: 轨道信息，包含轨道点
        default_position: 默认位置
        default_velocity: 默认速度
        interpolation_method: 插值方法 ('HERMITE', 'SCH', 'LINEAR')
        
    Returns:
        (位置, 速度)
    """
    # 检查轨道数据
    if not orbit or 'orbit_points' not in orbit:
        return default_position, default_velocity
    
    orbit_points = orbit['orbit_points']
    if not orbit_points:
        return default_position, default_velocity
    
    # 提取轨道点的时间、位置和速度
    orbit_times = []
    orbit_positions = []
    orbit_velocities = []
    
    for point in orbit_points:
        if 'time' in point:
            # 将时间字符串转换为数值
            time_str = point['time']
            try:
                import datetime
                dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                timestamp = dt.timestamp()
                orbit_times.append(timestamp)
            except Exception:
                # 如果转换失败，使用索引作为时间
                orbit_times.append(len(orbit_times))
            
            if 'position' in point:
                pos = point['position']
                orbit_positions.append([pos['x'], pos['y'], pos['z']])
            if 'velocity' in point:
                vel = point['velocity']
                orbit_velocities.append([vel['vx'], vel['vy'], vel['vz']])
    
    # 根据插值方法选择插值函数
    if interpolation_method.upper() == 'HERMITE':
        return interpolate_wgs84_orbit(orbit_times, orbit_positions, orbit_velocities, time)
    elif interpolation_method.upper() == 'SCH':
        return interpolate_sch_orbit(orbit_times, orbit_positions, orbit_velocities, time)
    else:
        return linear_interpolation(orbit_times, orbit_positions, orbit_velocities, time)


def calculate_doppler(slant_range: float, 
                      doppler_polynomial: Optional[List[float]] = None, 
                      doppler_derivative_polynomial: Optional[List[float]] = None) -> Tuple[float, float]:
    """
    计算多普勒频率和导数
    
    Args:
        slant_range: 斜距
        doppler_polynomial: 多普勒多项式系数列表
        doppler_derivative_polynomial: 多普勒导数多项式系数列表
        
    Returns:
        (多普勒频率, 多普勒频率导数)
    """
    # 默认返回0
    fdop = 0.0
    fdopder = 0.0
    
    # 如果有多普勒多项式，使用多项式计算
    if doppler_polynomial:
        try:
            # 简单的多项式计算（实际应该根据多项式阶数计算）
            if isinstance(doppler_polynomial, list):
                # 假设多项式系数按次数从低到高排列
                fdop = 0.0
                for i, coeff in enumerate(doppler_polynomial):
                    fdop += coeff * (slant_range ** i)
        except Exception as e:
            print(f"计算多普勒频率失败: {e}")
    
    if doppler_derivative_polynomial:
        try:
            # 简单的多项式计算（实际应该根据多项式阶数计算）
            if isinstance(doppler_derivative_polynomial, list):
                # 假设多项式系数按次数从低到高排列
                fdopder = 0.0
                for i, coeff in enumerate(doppler_derivative_polynomial):
                    fdopder += coeff * (slant_range ** i)
        except Exception as e:
            print(f"计算多普勒频率导数失败: {e}")
    
    return fdop, fdopder


def load_doppler_parameters(yaml_file: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    从YAML文件加载多普勒参数
    
    Args:
        yaml_file: YAML文件路径
        
    Returns:
        (多普勒多项式, 多普勒导数多项式)
    """
    doppler_polynomial = None
    doppler_derivative_polynomial = None
    
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'doppler_parameters' in data:
            doppler_polynomial = data['doppler_parameters'].get('doppler_polynomial', None)
            doppler_derivative_polynomial = data['doppler_parameters'].get('doppler_derivative_polynomial', None)
    except Exception as e:
        print(f"读取多普勒参数失败: {e}")
    
    return doppler_polynomial, doppler_derivative_polynomial


def read_image(file_path):
    """
    读取图像文件
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        图像数据（复数数组）
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
    
    Args:
        filename: YAML文件路径
        
    Returns:
        YAML文件内容（转换为普通字典）
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
    
    Args:
        data: 要写入的数据
        filename: 输出文件路径
    """
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def generate_offset_field(image_shape: Tuple[int, int], offset_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成完整的多项式偏移场
    
    Args:
        image_shape: 图像形状 (height, width)
        offset_data: 包含偏移参数的字典
        
    Returns:
        (azimuth_offset_field, range_offset_field)
    """
    h, w = image_shape
    
    # 获取参数
    params = offset_data.get('parameters', {})
    normalization = offset_data.get('normalization', {})
    
    # 从normalization或直接从offset_data中获取归一化参数
    rows_mean = normalization.get('rows_mean', offset_data.get('rows_mean', 0.0))
    rows_std = normalization.get('rows_std', offset_data.get('rows_std', 1.0))
    cols_mean = normalization.get('cols_mean', offset_data.get('cols_mean', 0.0))
    cols_std = normalization.get('cols_std', offset_data.get('cols_std', 1.0))
    
    # 获取多项式参数
    a0 = params.get('a0', 0.0)
    a1 = params.get('a1', 0.0)
    a2 = params.get('a2', 0.0)
    a3 = params.get('a3', 0.0)
    a4 = params.get('a4', 0.0)
    a5 = params.get('a5', 0.0)
    
    b0 = params.get('b0', 0.0)
    b1 = params.get('b1', 0.0)
    b2 = params.get('b2', 0.0)
    b3 = params.get('b3', 0.0)
    b4 = params.get('b4', 0.0)
    b5 = params.get('b5', 0.0)
    
    # 创建网格
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # 归一化坐标
    rows_norm = (rows - rows_mean) / (rows_std + 1e-10)
    cols_norm = (cols - cols_mean) / (cols_std + 1e-10)
    
    # 计算方位向偏移场
    az_field = (a0 + 
               a1 * rows_norm +
               a2 * cols_norm +
               a3 * rows_norm * cols_norm +
               a4 * rows_norm**2 +
               a5 * cols_norm**2)
    
    # 计算距离向偏移场
    rg_field = (b0 + 
               b1 * rows_norm +
               b2 * cols_norm +
               b3 * rows_norm * cols_norm +
               b4 * rows_norm**2 +
               b5 * cols_norm**2)
    
    return az_field.astype(np.float32), rg_field.astype(np.float32)


@njit(fastmath=True, cache=True)
def compute_correlation(reference: np.ndarray, registered: np.ndarray) -> float:
    """
    计算相关系数
    
    Args:
        reference: 参考图像
        registered: 配准后图像
        
    Returns:
        相关系数
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


@njit(fastmath=True, cache=True)
def compute_snr(reference: np.ndarray, registered: np.ndarray) -> float:
    """
    计算信噪比
    
    Args:
        reference: 参考图像
        registered: 配准后图像
        
    Returns:
        信噪比 (dB)
    """
    # 计算差值
    diff = reference - registered
    
    signal_power = np.mean(reference**2)
    noise_power = np.mean(diff**2)
    
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    return float('inf')


@njit(fastmath=True, cache=True)
def compute_rmse(reference: np.ndarray, registered: np.ndarray) -> float:
    """
    计算均方根误差
    
    Args:
        reference: 参考图像
        registered: 配准后图像
        
    Returns:
        RMSE
    """
    diff = reference - registered
    rmse = np.sqrt(np.mean(diff**2))
    return float(rmse)


@njit(parallel=True, fastmath=True, cache=True)
def compute_local_correlation(reference: np.ndarray, registered: np.ndarray, window_size: int = 32) -> np.ndarray:
    """
    计算局部相关系数
    
    Args:
        reference: 参考图像
        registered: 配准后图像
        window_size: 窗口大小
        
    Returns:
        局部相关系数图
    """
    h, w = reference.shape
    corr_map = np.zeros((h, w))
    
    half = window_size // 2
    
    for i in prange(half, h - half):
        for j in range(half, w - half):
            ref_window = reference[i-half:i+half, j-half:j+half]
            reg_window = registered[i-half:i+half, j-half:j+half]
            
            # 计算相关系数
            ref_flat = ref_window.flatten()
            reg_flat = reg_window.flatten()
            
            ref_mean = np.mean(ref_flat)
            reg_mean = np.mean(reg_flat)
            
            numerator = np.sum((ref_flat - ref_mean) * (reg_flat - reg_mean))
            denominator = np.sqrt(
                np.sum((ref_flat - ref_mean)**2) * 
                np.sum((reg_flat - reg_mean)**2)
            )
            
            if denominator > 0:
                corr_map[i, j] = numerator / denominator
            else:
                corr_map[i, j] = 0.0
    
    return corr_map


def _i0(x: float) -> float:
    """
    零阶修正贝塞尔函数
    
    Args:
        x: 输入值
        
    Returns:
        i0(x)的值
    """
    # 简化实现，使用指数函数近似
    if x < 0:
        x = -x
    return np.cosh(x)


def _kaiser_window(n: int, beta: float) -> np.ndarray:
    """
    生成kaiser窗口
    
    Args:
        n: 窗口大小
        beta: kaiser窗口参数
        
    Returns:
        kaiser窗口
    """
    # 手动实现kaiser窗口，避免依赖scipy
    n -= 1
    if n == 0:
        return np.array([1.0])
    
    even = (n % 2 == 0)
    m = n // 2
    
    a = np.linspace(-1, 1, n+1)
    w = np.zeros(n+1)
    
    if even:
        for i in range(m+1):
            w[m+i] = w[m-i] = _i0(beta * np.sqrt(1 - (2*i/n)**2))
    else:
        for i in range(m+1):
            w[m+i] = w[m-i] = _i0(beta * np.sqrt(1 - (2*i/(n))**2))
    
    return w / w[0]


def calculate_look_direction(corner_coords: List[Dict], orbit_data: Dict) -> str:
    """
    计算卫星视线方向
    
    Args:
        corner_coords: SAR图像四个角点的坐标，格式为[{"lat": lat, "lon": lon}, ...]
        orbit_data: 轨道数据
        
    Returns:
        视线方向 ("left" 或 "right")
    """
    # 计算图像中心点
    lats = [corner["lat"] for corner in corner_coords]
    lons = [corner["lon"] for corner in corner_coords]
    center_lat = (max(lats) + min(lats)) / 2
    center_lon = (max(lons) + min(lons)) / 2
    
    # 计算轨道中心点的位置
    if not orbit_data or "orbit_points" not in orbit_data:
        return "unknown"
    
    orbit_points = orbit_data["orbit_points"]
    if not orbit_points:
        return "unknown"
    
    # 提取轨道点的位置
    orbit_positions = []
    for point in orbit_points:
        if "position" in point:
            pos = point["position"]
            orbit_positions.append([pos["x"], pos["y"], pos["z"]])
    
    if not orbit_positions:
        return "unknown"
    
    # 计算轨道中心点
    orbit_center = np.mean(orbit_positions, axis=0)
    
    # 将图像中心点转换为地心坐标
    center_xyz = llh_to_xyz(center_lat, center_lon, 0)
    
    # 计算视线向量（从卫星到目标）
    look_vector = np.array(center_xyz) - np.array(orbit_center)
    
    # 计算卫星速度向量（使用第一个和最后一个轨道点）
    if len(orbit_positions) >= 2:
        sat_vel = np.array(orbit_positions[-1]) - np.array(orbit_positions[0])
    else:
        return "unknown"
    
    # 计算轨道法向量（右手定则：速度向量叉乘位置向量）
    orbit_normal = np.cross(sat_vel, orbit_center)
    
    # 计算视线向量与轨道法向量的点积
    dot_product = np.dot(look_vector, orbit_normal)
    
    # 根据点积符号判断视线方向
    if dot_product > 0:
        return "right"
    else:
        return "left"


def get_sar_image_corners(yaml_file: str) -> List[Dict]:
    """
    从YAML文件获取SAR图像四个角点的坐标
    
    Args:
        yaml_file: YAML文件路径
        
    Returns:
        四个角点的坐标列表，格式为[{"lat": lat, "lon": lon}, ...]
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 从YAML文件中获取四个角点的坐标
        # 检查corner_coordinates字段
        if 'corner_coordinates' in data:
            corners = data['corner_coordinates']
            return [
                {"lat": corners['top_left']['lat'], "lon": corners['top_left']['lon']},
                {"lat": corners['top_right']['lat'], "lon": corners['top_right']['lon']},
                {"lat": corners['bottom_right']['lat'], "lon": corners['bottom_right']['lon']},
                {"lat": corners['bottom_left']['lat'], "lon": corners['bottom_left']['lon']}
            ]
        # 检查image_corners字段（兼容旧格式）
        elif 'image_corners' in data:
            return data['image_corners']
        else:
            # 返回默认值，实际应用中应该从其他字段计算
            return [
                {"lat": 0, "lon": 0},
                {"lat": 0, "lon": 1},
                {"lat": 1, "lon": 1},
                {"lat": 1, "lon": 0}
            ]
    except Exception as e:
        print(f"获取SAR图像角点失败: {e}")
        # 返回默认值
        return [
            {"lat": 0, "lon": 0},
            {"lat": 0, "lon": 1},
            {"lat": 1, "lon": 1},
            {"lat": 1, "lon": 0}
        ]
