#!/usr/bin/env python3
"""
DEM到SAR坐标转换模块

功能：将DEM地理坐标转换为SAR图像坐标。
该实现参考ISCE2的DEM到SAR流程，要求YAML文件中提供至少两
个具有不同时间戳的轨道点；在时间退化（所有时间相同）时，
会自动构造简单的时间轴和位置变化以保持方位时间变化。
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, List
# Relative imports only work when package is installed or run via -m.
# provide fallback so script can be executed directly from workspace.
try:
    from .sar_utils import llh_to_xyz, calculate_orbit_position, calculate_doppler, load_doppler_parameters, calculate_look_direction, get_sar_image_corners
except ImportError:
    # adjust sys.path and retry absolute import
    import sys, os
    pkgdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkgdir not in sys.path:
        sys.path.insert(0, pkgdir)
    from sar_utils import llh_to_xyz, calculate_orbit_position, calculate_doppler, load_doppler_parameters, calculate_look_direction, get_sar_image_corners
from multiprocessing import Pool, cpu_count


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
    DEM到SAR坐标转换器
    """
    def __init__(self, yaml_file: str, dem_file: str,
                 orbit_interpolation_method: str = 'HERMITE'):
        """
        初始化转换器
        
        Args:
            yaml_file: SAR YAML文件路径
            dem_file: DEM文件路径
            orbit_interpolation_method: 轨道插值方法，HERMITE/SCH/LINEAR
        """
        self.yaml_file = yaml_file
        self.dem_file = dem_file
        self.orbit_interpolation_method = orbit_interpolation_method.upper()
        self.sar_geometry = SarGeometry(yaml_file)
        self.dem_data = None
        self.dem_geotransform = None
        self.dem_projection = None
        self.orbit_positions = []
        self.orbit_velocities = []
        self.orbit_times = []
        self.orbit_data = None
        self.look_direction = None  # 视线方向
        
        # 轨道插值缓存
        self._orbit_interp_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._load_dem()
        self._precompute_parameters()
        self._precompute_orbit_positions()
        self._calculate_look_direction()
    
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
        
        # 计算时间步长
        if self.prf > 0:
            self.azimuth_time_step = 1.0 / self.prf
        else:
            self.azimuth_time_step = 1.0 / 4144.0  # 默认值
        
        # 计算距离向采样间隔
        if self.range_sampling_rate > 0:
            self.c = 3e8  # 光速
            self.range_pixel_spacing = self.c / (2 * self.range_sampling_rate)
        else:
            self.range_pixel_spacing = 1.5  # 默认值（约1.5米）
        
        # 初始化多普勒参数
        self.doppler_polynomial, self.doppler_derivative_polynomial = load_doppler_parameters(self.yaml_file)
        
        # 预读取插值方式（如果YAML中指定）
        try:
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            method = cfg.get('orbit_interpolation_method')
            if method:
                self.orbit_interpolation_method = method.upper()
        except Exception:
            pass
        
        # 预计算轨道数据，用于Numba优化的函数
        self._precompute_orbit_cache()
    
    def _precompute_orbit_cache(self):
        """
        预计算轨道数据缓存
        """
        try:
            print(f"开始加载轨道数据: {self.yaml_file}")
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            print(f"YAML文件加载成功，数据类型: {type(data)}")
            print(f"YAML文件包含的键: {list(data.keys())}")
            
            if 'orbit_data' in data:
                print("找到orbit_data键")
                orbit_data = data['orbit_data']
                print(f"orbit_data包含的键: {list(orbit_data.keys())}")
                
                if 'orbit_points' in orbit_data:
                    print("找到orbit_points键")
                    orbit_points = orbit_data['orbit_points']
                    print(f"轨道点数量: {len(orbit_points)}")
                    
                    # 预计算并缓存轨道数据
                    self.orbit_cache = {
                        'positions': [],
                        'velocities': [],
                        'times': []
                    }
                    
                    # 存储时间戳，用于计算相对时间
                    timestamps = []
                    
                    for i, point in enumerate(orbit_points):
                        #if i < 5:  # 只打印前5个轨道点的信息
                            #print(f"处理轨道点 {i}: {point.keys()}")
                            #if 'position' in point:
                            #    print(f"  位置: {point['position']}")
                            #if 'time' in point:
                            #    print(f"  时间: {point['time']}")
                            #if 'velocity' in point:
                            #    print(f"  速度: {point['velocity']}")
                        
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
                                if i < 5:
                                    print(f"  转换后的时间戳: {timestamp}")
                            except Exception as e:
                                print(f"时间转换失败: {e}")
                                # 如果时间转换失败，使用索引作为时间
                                timestamps.append(len(timestamps))
                        if 'velocity' in point:
                            vel = point['velocity']
                            self.orbit_cache['velocities'].append([vel['vx'], vel['vy'], vel['vz']])
                    
                    # 转换为相对时间（相对于第一个轨道点的时间）
                    if timestamps:
                        t0 = timestamps[0]
                        relative_times = [t - t0 for t in timestamps]
                        # 使用 double 精度存储时间
                        arr = np.array(relative_times, dtype=np.float64)
                        # 检查时间范围
                        time_range = np.ptp(arr)
                        print(f"轨道时间范围: {time_range} 秒")
                        # 如果所有时间相同则构造简单的时间轴并调整位置
                        if time_range == 0:
                            print("警告: 轨道时间范围为0，构造人工时间轴")
                            npts = arr.size
                            # 线性分布0..1秒，使用 double 精度
                            arr = np.linspace(0.0, 1.0, npts, dtype=np.float64)
                            # 如果有速度信息, 按第一速度推进位置
                            if self.orbit_cache['velocities'].size >= 3:
                                v0 = self.orbit_cache['velocities'].reshape(-1,3)[0]
                                p0 = self.orbit_cache['positions'].reshape(-1,3)[0]
                                newpos = []
                                for t in arr:
                                    newpos.append(p0 + v0 * t)
                                self.orbit_cache['positions'] = np.vstack(newpos)
                        self.orbit_cache['times'] = arr
                        print(f"相对时间示例: {arr[:5]}")
                    else:
                        self.orbit_cache['times'] = np.array([])
                        print("警告: 没有有效的时间戳")
                    
                    # 转换为NumPy数组，提高处理速度
                    self.orbit_cache['positions'] = np.array(self.orbit_cache['positions'])
                    self.orbit_cache['velocities'] = np.array(self.orbit_cache['velocities'])
                    print(f"轨道点数量: {len(self.orbit_cache['times'])}")
                    print(f"位置数组形状: {self.orbit_cache['positions'].shape}")
                    print(f"速度数组形状: {self.orbit_cache['velocities'].shape}")
                    print(f"时间数组形状: {self.orbit_cache['times'].shape}")
                else:
                    self.orbit_cache = None
                    print("警告: 未找到orbit_points键")
            else:
                self.orbit_cache = None
                print("警告: 未找到orbit_data键")
        except Exception as e:
            self.orbit_cache = None
            print(f"预计算轨道缓存失败: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            # 计算视线方向
            self.look_direction = calculate_look_direction(corner_coords, self.orbit_data)
            
        except Exception:
            self.look_direction = "unknown"
    
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
        """
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
            print("警告: 轨道时间范围为0，使用单位间隔代替以恢复方位时间变化")

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
            if len(self._orbit_interp_cache) < 10000:  # 防止缓存过大
                self._orbit_interp_cache[cache_key] = result
            return result
        
        if orbit_times.size < 2:
            base_pos = np.array([-284515.577, 5480417.824, 4154857.155]) if orbit_positions.size > 0 else np.array([0.0, 0.0, 500000.0])
            base_vel = np.array([0.0, 7000.0, 0.0]) if orbit_velocities.size > 0 else np.array([0.0, 7000.0, 0.0])
            pos = base_pos + time * base_vel
            result = (pos, base_vel)
            if len(self._orbit_interp_cache) < 10000:
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
        
        # 缓存结果
        if len(self._orbit_interp_cache) < 10000:
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
        
        # 计算使用的CPU核心数（使用更多核心以加快速度）
        num_cores = max(4, cpu_count())  # 至少4个核心
        
        # 计算分块大小
        chunk_size = max(100, dem_rows // num_cores)  # 保证每块至少100行
        chunks = []
        
        for i in range(0, dem_rows, chunk_size):
            i_end = min(i + chunk_size, dem_rows)
            chunks.append((i, i_end, 0, dem_cols, step))
        
        # 并行处理
        with Pool(num_cores) as pool:
            results = pool.map(self._process_chunk, chunks)
        
        # 合并结果
        all_results = []
        for chunk_result in results:
            all_results.extend(chunk_result)
        
        print(f"\n缓存统计: 命中 {self._cache_hits}, 未命中 {self._cache_misses}, 命中率 {100*self._cache_hits/(self._cache_hits+self._cache_misses+1):.1f}%")
        return np.array(all_results)


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='DEM到SAR坐标转换工具')
    parser.add_argument('yaml_file', help='SAR YAML文件路径')
    parser.add_argument('dem_file', help='DEM文件路径')
    parser.add_argument('output_file', help='输出文件路径')
    parser.add_argument('--step', type=int, default=10, help='采样步长')
    parser.add_argument('--orbit-interp', choices=['HERMITE','SCH','LINEAR'],
                        default='HERMITE', help='轨道插值方法')
    
    args = parser.parse_args()
    
    print("=== DEM到SAR坐标转换工具 ===")
    print(f"SAR YAML文件: {args.yaml_file}")
    print(f"DEM文件: {args.dem_file}")
    print(f"输出文件: {args.output_file}")
    print(f"采样步长: {args.step}")
    
    try:
        converter = DemToSarConverter(args.yaml_file,
                                     args.dem_file,
                                     orbit_interpolation_method=args.orbit_interp)
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
        print(f"转换失败: {e}")


if __name__ == '__main__':
    main()
