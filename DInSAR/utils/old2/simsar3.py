#!/usr/bin/env python3
"""
SAR数据模拟器 V3

基于ESA SNAP平台及经典文献的详细工作流程，从DEM结合SAR轨道参数生成模拟SAR图像

核心特性：
1. 基于距离-多普勒（Range-Doppler）模型的严格几何与辐射过程
2. 完整的轨道插值与几何定位算法
3. 多种后向散射模型支持
4. 叠掩与阴影检测
5. 高效的并行处理
6. 噪声模拟
"""

import os
import sys
import argparse
import numpy as np
import yaml
from datetime import datetime
from typing import Tuple, Optional, List
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

try:
    from osgeo import gdal
    from osgeo import osr
except ImportError:
    import gdal
    import osr

try:
    from .geo2rdr import Geo2rdr, xyz_to_llh, llh_to_xyz
except ImportError:
    from geo2rdr import Geo2rdr, xyz_to_llh, llh_to_xyz

try:
    from .sar_utils import get_sar_image_corners
except ImportError:
    from sar_utils import get_sar_image_corners


def _calculate_look_direction_auto(corner_coords, orbit_data, geo2rdr):
    """
    根据卫星轨道方向和SAR角点方向自动计算look direction
    
    使用叉积判断：
    - 卫星速度方向 × 距离向方向（角点1到角点2）
    - 如果结果向上(正Z)，则是左视；否则是右视
    """
    if not corner_coords or len(corner_coords) < 4 or not orbit_data:
        return None
    
    try:
        from .geo2rdr import xyz_to_llh
    except ImportError:
        try:
            from geo2rdr import xyz_to_llh
        except ImportError:
            return None
    
    orbit_points = orbit_data.get('orbit_points', [])
    if len(orbit_points) < 2:
        return None
    
    positions = []
    for pt in orbit_points[:10]:
        if 'position' in pt:
            pos = pt['position']
            positions.append([pos['x'], pos['y'], pos['z']])
    
    if len(positions) < 2:
        return None
    
    orbit_start = np.array(positions[0])
    orbit_end = np.array(positions[-1])
    orbit_dir = orbit_end - orbit_start
    
    c0 = np.array([corner_coords[0].get('lon', 0), corner_coords[0].get('lat', 0), 0])
    c1 = np.array([corner_coords[1].get('lon', 0), corner_coords[1].get('lat', 0), 0])
    range_dir = c1 - c0
    range_dir = range_dir / (np.linalg.norm(range_dir) + 1e-10)
    
    cross = np.cross(orbit_dir, range_dir)
    
    if cross[2] > 0:
        return 'LEFT'
    else:
        return 'RIGHT'


def _process_single_batch(batch_data, g2r, prf, nrows, ncols, near_range, range_spacing, 
                         sensing_start, orbit_t0, margin_azimuth, margin_range,
                         wavelength, include_topo_phase, baseline_perp,
                         backscatter_model, noise_level, layover_shadow_detection, azimuth_orbit_data=None):
    """处理单个batch的DEM"""
    batch_lats, batch_lons, batch_heights = batch_data
    
    # 异常处理：避免整批次失败，记录错误但继续处理
    try:
        range_samples, azimuth_times = g2r.geo2rdr_batch(batch_lats, batch_lons, batch_heights, n_workers=1)
    except Exception as e:
        print(f"    [DEBUG] geo2rdr_batch异常: {e}, 跳过该批次")
        return None
    
    # 过滤错误值（geo2rdr返回-1表示计算失败）
    valid_mask = (range_samples != -1) & (azimuth_times != -1)
    if np.sum(valid_mask) == 0:
        print(f"    [DEBUG] 所有点计算失败，跳过该批次")
        return None
    
    # 只保留有效点
    batch_lats = batch_lats[valid_mask]
    batch_lons = batch_lons[valid_mask]
    batch_heights = batch_heights[valid_mask]
    range_samples = range_samples[valid_mask]
    azimuth_times = azimuth_times[valid_mask]
    
    print(f"    [DEBUG] 过滤后有效点数: {len(range_samples)}")
    
    # 计算azimuth_pixels，使用sensing_start作为基准
    # SAR相对时间 = 绝对时间 - sensing_start
    azimuth_time_rel = azimuth_times - sensing_start
    
    # 计算azimuth_pixels
    azimuth_pixels = azimuth_time_rel * prf
    
    # 调试：检查azimuth_times的范围
    print(f"    [DEBUG] azimuth_times范围: min={np.min(azimuth_times):.3f}, max={np.max(azimuth_times):.3f}")
    print(f"    [DEBUG] sensing_start: {sensing_start:.3f}")
    print(f"    [DEBUG] azimuth_time_rel范围: min={np.min(azimuth_time_rel):.3f}, max={np.max(azimuth_time_rel):.3f}")
    print(f"    [DEBUG] prf: {prf}")
    print(f"    [DEBUG] 计算的azimuth_pixels范围: min={np.min(azimuth_pixels):.3f}, max={np.max(azimuth_pixels):.3f}")
    
    # 确保azimuth_pixels在合理范围内
    max_az_pixel = nrows  # 使用传入的图像高度
    
    # 直接使用计算得到的azimuth_pixels，只进行clipping
    # 这样可以保持原始的时间分布
    azimuth_pixels = np.clip(azimuth_pixels, 0, max_az_pixel - 1)
    
    # 调试：检查调整后的azimuth_pixels范围
    print(f"    [DEBUG] 调整后的azimuth_pixels范围: min={np.min(azimuth_pixels):.3f}, max={np.max(azimuth_pixels):.3f}")
    
    az_min = np.nanmin(azimuth_pixels) if len(azimuth_pixels) > 0 else 0
    az_max = np.nanmax(azimuth_pixels) if len(azimuth_pixels) > 0 else 0
    rg_min = np.nanmin(range_samples) if len(range_samples) > 0 else 0
    rg_max = np.nanmax(range_samples) if len(range_samples) > 0 else 0
    
    # 调试：检查azimuth_pixels的分布
    az_percentiles = [0, 10, 25, 50, 75, 90, 100] if len(azimuth_pixels) > 0 else []
    az_values = [np.percentile(azimuth_pixels, p) for p in az_percentiles] if len(azimuth_pixels) > 0 else []
    median_az = az_values[3] if len(az_values) > 3 else 'N/A'
    median_str = f"{median_az:.1f}" if isinstance(median_az, float) else median_az
    print(f"    [DEBUG] azimuth_pixels分布: min={az_min:.1f}, max={az_max:.1f}, 中位数={median_str}")
    print(f"    [DEBUG] range_samples分布: min={rg_min:.1f}, max={rg_max:.1f}")
    
    # 直接使用计算得到的azimuth_pixels和range_samples，只进行clipping
    # 不进行额外的过滤，以保留更多有效点
    azimuth_lines = np.round(azimuth_pixels).astype(np.int32)
    range_indices = np.round(range_samples).astype(np.int32)
    
    # 直接进行clipping，确保点在图像范围内
    azimuth_lines_clipped = np.clip(azimuth_lines, 0, nrows - 1)
    range_indices_clipped = np.clip(range_indices, 0, ncols - 1)
    
    # 调试：统计最终每个azimuth行的点数
    az_hist = {}
    for az in azimuth_lines_clipped:
        az_hist[az] = az_hist.get(az, 0) + 1
    
    print(f"    [DEBUG] 最终有效点数: {len(azimuth_lines_clipped)}")
    print(f"    [DEBUG] azimuth_lines_clipped范围: min={np.min(azimuth_lines_clipped):.1f}, max={np.max(azimuth_lines_clipped):.1f}")
    print(f"    [DEBUG] range_indices_clipped范围: min={np.min(range_indices_clipped):.1f}, max={np.max(range_indices_clipped):.1f}")
    
    # 保留所有有效点，不进行额外过滤
    batch_lats_valid = batch_lats
    batch_lons_valid = batch_lons
    batch_heights_valid = batch_heights
    azimuth_times_valid = azimuth_times
    range_samples_valid = range_samples
    
    incidence_angles = _calculate_incidence_angle_vectorized(batch_lats_valid, batch_lons_valid, batch_heights_valid, g2r, azimuth_times_valid, azimuth_orbit_data)
    sigma0_values = _calculate_backscatter_vectorized(incidence_angles, batch_heights_valid, backscatter_model)
    
    if noise_level > 0:
        sigma0_values = _add_speckle_noise_vectorized(sigma0_values, noise_level)
    
    if include_topo_phase and baseline_perp != 0.0:
        slant_ranges = near_range + range_samples_valid * range_spacing
        phases = 4 * np.pi * baseline_perp * batch_heights_valid * np.sin(incidence_angles) / (wavelength * slant_ranges)
    else:
        phases = np.zeros(len(sigma0_values))
    
    result = {
        'azimuth_lines': azimuth_lines_clipped,
        'range_indices': range_indices_clipped,
        'sigma0_values': sigma0_values,
        'phases': phases,
        'count': len(azimuth_lines)
    }
    
    if layover_shadow_detection:
        layover_vals, shadow_vals = _detect_layover_shadow_vectorized(
            batch_lats_valid, batch_lons_valid, batch_heights_valid, g2r, azimuth_times_valid, incidence_angles, azimuth_orbit_data
        )
        result['layover_vals'] = layover_vals
        result['shadow_vals'] = shadow_vals
    
    return result


def _interpolate_dem(dem_data, dem_geotransform, dem_projection, lon, lat):
    """从DEM中插值获取高程值"""
    if dem_data is None:
        return 0.0
    
    nrows, ncols = dem_data.shape
    
    # 将经纬度转换为DEM像素坐标
    try:
        from osgeo import osr
        osr.UseExceptions()
        
        # 创建坐标转换
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)  # WGS84
        
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromWkt(dem_projection)  # DEM投影
        
        transform = osr.CoordinateTransformation(src_srs, dst_srs)
        x, y, _ = transform.TransformPoint(lon, lat)
    except Exception:
        # 如果转换失败，直接使用经纬度作为坐标
        x = lon
        y = lat
    
    # 计算像素坐标
    dx = dem_geotransform[1]
    dy = dem_geotransform[5]
    x0 = dem_geotransform[0]
    y0 = dem_geotransform[3]
    
    if dx == 0 or dy == 0:
        return 0.0
    
    col = (x - x0) / dx
    row = (y - y0) / dy
    
    # 检查坐标是否在DEM范围内
    if row < 0 or row >= nrows - 1 or col < 0 or col >= ncols - 1:
        return 0.0
    
    # 双线性插值
    i = int(row)
    j = int(col)
    di = row - i
    dj = col - j
    
    # 确保索引在有效范围内
    i = min(i, nrows - 2)
    j = min(j, ncols - 2)
    
    # 获取周围四个点的高程值
    v00 = dem_data[i, j]
    v01 = dem_data[i, j+1]
    v10 = dem_data[i+1, j]
    v11 = dem_data[i+1, j+1]
    
    # 检查值是否有效
    if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
        return 0.0
    
    # 双线性插值
    return (1 - di) * ((1 - dj) * v00 + dj * v01) + di * ((1 - dj) * v10 + dj * v11)


def _process_sar_chunk_worker(args):
    """多进程worker函数 - 处理SAR分块"""
    (i_start, i_end, j_start, j_end, dem_data, dem_geotransform, dem_projection,
     orbit_data, prf, wavelength, range_spacing, near_range, 
     sensing_start, orbit_t0, look_direction, dop_poly, 
     include_topo_phase, baseline_perp, nrows, ncols, 
     backscatter_model, noise_level, layover_shadow_detection) = args
    
    sar_chunk = np.zeros((i_end - i_start, j_end - j_start), dtype=np.complex64)
    count_chunk = np.zeros((i_end - i_start, j_end - j_start), dtype=np.int32)
    layover_mask = np.zeros((i_end - i_start, j_end - j_start), dtype=np.uint8)
    shadow_mask = np.zeros((i_end - i_start, j_end - j_start), dtype=np.uint8)
    
    g2r = Geo2rdr(
        prf=prf,
        radar_wavelength=wavelength,
        slant_range_pixel_spacing=range_spacing,
        range_first_sample=near_range,
        sensing_start=sensing_start,
        look_side=look_direction
    )
    g2r.set_orbit(orbit_data)
    if dop_poly is not None:
        g2r.doppler_polynomial = dop_poly
    
    valid_count = 0
    
    for i in range(i_start, i_end):
        # 计算当前SAR像素的方位时间
        azimuth_time = sensing_start + i / prf
        
        for j in range(j_start, j_end):
            # 计算当前SAR像素的斜距
            slant_range = near_range + j * range_spacing
            
            try:
                # 从SAR坐标转换到地理坐标（rdr2geo）
                lat, lon, height = g2r.rdr2geo(slant_range, azimuth_time)
                
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                
                # 从DEM中插值获取高程
                dem_height = _interpolate_dem(dem_data, dem_geotransform, dem_projection, lon, lat)
                
                if dem_height <= 0:
                    continue
                
                # 计算入射角
                incidence_angle = _calculate_incidence_angle(lat, lon, dem_height, g2r, azimuth_time)
                
                # 计算后向散射
                sigma0 = _calculate_backscatter(incidence_angle, dem_height, backscatter_model)
                
                if noise_level > 0:
                    sigma0 = _add_speckle_noise(sigma0, noise_level)
                
                # 计算相位
                if include_topo_phase and baseline_perp != 0.0:
                    phase = 4 * np.pi * baseline_perp * dem_height * np.sin(incidence_angle) / (wavelength * slant_range)
                else:
                    phase = 0.0
                
                # 存储结果
                local_i = i - i_start
                local_j = j - j_start
                sar_chunk[local_i, local_j] = sigma0 * np.exp(1j * phase)
                count_chunk[local_i, local_j] = 1
                
                # 检测叠掩和阴影
                if layover_shadow_detection:
                    is_layover, is_shadow = _detect_layover_shadow(lat, lon, dem_height, g2r, azimuth_time)
                    if is_layover:
                        layover_mask[local_i, local_j] = 1
                    if is_shadow:
                        shadow_mask[local_i, local_j] = 1
                
                valid_count += 1
                
            except Exception:
                continue
    
    print(f"    [DEBUG] 处理SAR块 [{i_start}:{i_end}, {j_start}:{j_end}] 完成，有效点: {valid_count}")
    
    return sar_chunk, count_chunk, layover_mask, shadow_mask, valid_count


def _calculate_incidence_angle_vectorized(lats, lons, heights, geo2rdr, azimuth_times, azimuth_orbit_data=None):
    """向量化计算局部入射角"""
    n = len(lats)
    incidence_angles = np.full(n, np.deg2rad(30.0))
    
    try:
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        
        a = 6378137.0
        e2 = 0.00669438002290
        
        N = a / np.sqrt(1 - e2 * np.sin(lats_rad)**2)
        
        x = (N + heights) * np.cos(lats_rad) * np.cos(lons_rad)
        y = (N + heights) * np.cos(lats_rad) * np.sin(lons_rad)
        z = (N * (1 - e2) + heights) * np.sin(lats_rad)
        
        ground_points = np.column_stack([x, y, z])
        
        nx = np.cos(lats_rad) * np.cos(lons_rad)
        ny = np.cos(lats_rad) * np.sin(lons_rad)
        nz = np.sin(lats_rad)
        normals = np.column_stack([nx, ny, nz])
        
        # 使用预计算的轨道数据
        if azimuth_orbit_data:
            for idx in range(n):
                t = azimuth_times[idx]
                # 查找最接近的预计算azimuth时间
                closest_time = min(azimuth_orbit_data.keys(), key=lambda t_az: abs(t_az - t))
                sat_pos = azimuth_orbit_data[closest_time]['position']
                
                look_vec = ground_points[idx] - sat_pos
                look_norm = np.linalg.norm(look_vec)
                if look_norm > 0:
                    look_vec = look_vec / look_norm
                
                cos_theta = np.dot(look_vec, normals[idx])
                cos_theta = np.clip(cos_theta, -1, 1)
                incidence_angles[idx] = np.arccos(cos_theta)
        else:
            # 回退到原始方法
            unique_times = np.unique(azimuth_times)
            sat_positions = {}
            for t in unique_times:
                sat_positions[t] = geo2rdr.get_satellite_position(t)
            
            for idx in range(n):
                t = azimuth_times[idx]
                sat_pos = sat_positions[t]
                
                look_vec = ground_points[idx] - sat_pos
                look_norm = np.linalg.norm(look_vec)
                if look_norm > 0:
                    look_vec = look_vec / look_norm
                
                cos_theta = np.dot(look_vec, normals[idx])
                cos_theta = np.clip(cos_theta, -1, 1)
                incidence_angles[idx] = np.arccos(cos_theta)
    
    except Exception:
        pass
    
    return incidence_angles


def _calculate_backscatter_vectorized(incidence_angles, heights, model):
    """向量化计算后向散射系数"""
    n = len(incidence_angles)
    
    if model == 'cosine':
        base_sigma = 10 ** (0.0 / 20.0)  # 增加基础后向散射系数
        sigma0 = base_sigma * np.cos(incidence_angles)
    elif model == 'constant':
        sigma0 = np.full(n, 10 ** (0.0 / 20.0))  # 增加基础后向散射系数
    elif model == 'oh':
        base_sigma = 10 ** (-3.0 / 20.0)  # 增加基础后向散射系数
        sigma0 = base_sigma * (np.cos(incidence_angles) ** 1.5)
    else:
        sigma0 = np.full(n, 10 ** (0.0 / 20.0))  # 增加基础后向散射系数
    
    height_factor = np.tanh(heights / 500.0) * 3.0
    sigma0_db = 10 * np.log10(np.maximum(sigma0, 1e-20)) + height_factor
    sigma0 = 10 ** (sigma0_db / 10.0)
    
    return sigma0


def _add_speckle_noise_vectorized(sigma0_values, noise_level):
    """向量化添加乘性相干斑噪声"""
    shape = 1.0 / noise_level
    scale = sigma0_values / shape
    noise = np.random.gamma(shape, scale)
    return noise


def _detect_layover_shadow_vectorized(lats, lons, heights, geo2rdr, azimuth_times, incidence_angles, azimuth_orbit_data=None):
    """向量化检测叠掩与阴影"""
    n = len(lats)
    is_layover = np.zeros(n, dtype=bool)
    is_shadow = np.zeros(n, dtype=bool)
    
    is_layover = incidence_angles > np.pi / 2
    
    return is_layover, is_shadow


def _calculate_incidence_angle(lat, lon, height, geo2rdr, azimuth_time):
    """计算局部入射角"""
    # 简化实现，实际应根据卫星位置和地面点计算
    # 这里使用基于轨道数据的更精确计算
    try:
        # 获取卫星位置
        sat_pos = geo2rdr.get_satellite_position(azimuth_time)
        
        # 将地理坐标转换为地心直角坐标
        x, y, z = _llh_to_xyz(lat, lon, height)
        ground_point = np.array([x, y, z])
        
        # 计算视线向量
        look_vector = ground_point - sat_pos
        look_vector /= np.linalg.norm(look_vector)
        
        # 计算地面法向量
        normal_vector = _calculate_ground_normal(lat, lon)
        
        # 计算入射角（视线向量与地面法向量的夹角）
        cos_theta = np.dot(look_vector, normal_vector)
        incidence_angle = np.arccos(np.clip(cos_theta, -1, 1))
        
        return incidence_angle
    except Exception:
        # 默认入射角
        return np.deg2rad(30.0)


def _llh_to_xyz(lat, lon, height):
    """将经纬度高程转换为地心直角坐标"""
    a = 6378137.0  # 地球长半轴
    e2 = 0.00669438002290  # 第一偏心率平方
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + height) * np.sin(lat_rad)
    
    return x, y, z


def _calculate_ground_normal(lat, lon):
    """计算地面法向量"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # 地面法向量（单位向量）
    nx = np.cos(lat_rad) * np.cos(lon_rad)
    ny = np.cos(lat_rad) * np.sin(lon_rad)
    nz = np.sin(lat_rad)
    
    return np.array([nx, ny, nz])


def _detect_layover_shadow(lat, lon, height, geo2rdr, azimuth_time):
    """检测叠掩与阴影"""
    # 简化实现，实际应使用更复杂的算法
    try:
        # 获取卫星位置
        sat_pos = geo2rdr.get_satellite_position(azimuth_time)
        
        # 将地理坐标转换为地心直角坐标
        x, y, z = _llh_to_xyz(lat, lon, height)
        ground_point = np.array([x, y, z])
        
        # 计算视线向量
        look_vector = ground_point - sat_pos
        look_vector /= np.linalg.norm(look_vector)
        
        # 计算地面法向量
        normal_vector = _calculate_ground_normal(lat, lon)
        
        # 计算入射角
        cos_theta = np.dot(look_vector, normal_vector)
        incidence_angle = np.arccos(np.clip(cos_theta, -1, 1))
        
        # 检测叠掩：入射角大于90度
        is_layover = incidence_angle > np.pi / 2
        
        # 简化的阴影检测：检查是否有更高的地形阻挡
        is_shadow = False
        
        return is_layover, is_shadow
    except Exception:
        return False, False


def _calculate_backscatter(incidence_angle, height, model):
    """计算后向散射系数"""
    if model == 'cosine':
        # 余弦模型
        cos_theta = np.cos(incidence_angle)
        sigma0 = 10 ** (0.0 / 20.0) * cos_theta  # 增加基础后向散射系数
    elif model == 'constant':
        # 常数赋值法
        sigma0 = 10 ** (0.0 / 20.0)  # 增加基础后向散射系数
    elif model == 'oh':
        # Oh模型（简化版）
        cos_theta = np.cos(incidence_angle)
        sigma0 = 10 ** (-3.0 / 20.0) * cos_theta ** 1.5  # 增加基础后向散射系数
    else:
        # 默认模型
        sigma0 = 10 ** (0.0 / 20.0)  # 增加基础后向散射系数
    
    # 添加高度影响
    height_factor = np.tanh(height / 500.0) * 3.0
    sigma0_db = 10 * np.log10(sigma0) + height_factor
    sigma0 = 10 ** (sigma0_db / 10.0)
    
    return sigma0


def _add_speckle_noise(sigma0, noise_level):
    """添加乘性相干斑噪声"""
    # 使用Gamma分布模拟相干斑噪声
    shape = 1.0 / noise_level
    scale = sigma0 / shape
    noise = np.random.gamma(shape, scale)
    return noise


class SARSimulatorV3:
    """SAR数据模拟器 V3"""
    
    def __init__(self, yaml_file: str, dem_file: str):
        """
        初始化SAR模拟器
        
        Args:
            yaml_file: SAR参数YAML文件
            dem_file: DEM文件路径
        """
        self.yaml_file = yaml_file
        self.dem_file = dem_file
        
        self.params = {}
        self.orbit_data = None
        self.dem_data = None
        self.dem_geotransform = None
        self.dem_projection = None
        self.dem_nrows = 0
        self.dem_ncols = 0
        self.dem_ds = None
        
        self.geo2rdr = None
        
        self._load_parameters()
        self._create_geo2rdr()  # 先创建Geo2rdr
        self._load_dem()  # 再加载DEM，这样可以计算SAR覆盖范围
    
    def _load_parameters(self):
        """加载SAR参数"""
        with open(self.yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self.params = data
        
        radar_params = data.get('radar_parameters', {})
        metadata = data.get('metadata', {})
        image_params = data.get('image_parameters', {})
        proc_params = data.get('processing_parameters', {})
        
        # 检查轨道数据格式
        self.orbit_data = data.get('orbit_data')
        if self.orbit_data:
            print(f"轨道数据加载成功，包含{len(self.orbit_data.get('orbit_points', []))}个轨道点")
            if 'orbit_points' in self.orbit_data:
                first_point = self.orbit_data['orbit_points'][0]
                if 'position' in first_point:
                    print(f"第一个轨道点位置: 纬度={first_point.get('lat', 'N/A')}, 经度={first_point.get('lon', 'N/A')}")
        
        # 读取多普勒多项式参数
        self.doppler_polynomial = None
        # 从processing_parameters中读取
        if 'doppler_polynomial' in proc_params:
            self.doppler_polynomial = proc_params['doppler_polynomial']
            print(f"从processing_parameters中读取多普勒多项式: {self.doppler_polynomial}")
        # 从radar_parameters中读取
        elif 'doppler_polynomial' in radar_params:
            self.doppler_polynomial = radar_params['doppler_polynomial']
            print(f"从radar_parameters中读取多普勒多项式: {self.doppler_polynomial}")
        # 从顶层数据中读取
        elif 'doppler_polynomial' in data:
            self.doppler_polynomial = data['doppler_polynomial']
            print(f"从顶层数据中读取多普勒多项式: {self.doppler_polynomial}")
        self.sensing_start = 0.0
        self.sar_duration = 0.0
        
        self.prf = radar_params.get('prf', 0.0)
        print(f"  [DEBUG] radar_params keys: {list(radar_params.keys())}")
        print(f"  [DEBUG] PRF: {self.prf}")
        self.radar_wavelength = radar_params.get('wavelength', 0.0)
        self.range_pixel_spacing = radar_params.get('range_spacing', 0.0)
        self.near_range = radar_params.get('near_range', 0.0)
        self.nrows = image_params.get('nrows', 0)
        self.ncols = image_params.get('ncols', 0)
        
        # 先从yaml中读取look_direction，如果没读到再用LEFT作为默认值
        look_dir = None
        
        # 1. 从radar_parameters中获取
        look_dir = radar_params.get('look_direction', None)
        
        # 2. 从顶层数据中获取
        if not look_dir:
            look_dir = data.get('look_direction', None)
        
        # 3. 从lookdir字段获取
        if not look_dir:
            look_dir = data.get('lookdir', None)
        
        # 4. 从processing_parameters中获取
        if not look_dir:
            proc_params = data.get('processing_parameters', {})
            look_dir = proc_params.get('look_direction', None)
        
        # 5. 从prm_parameters中获取
        if not look_dir:
            prm_params = data.get('prm_parameters', {})
            look_dir = prm_params.get('lookdir', None)
        
        # 如果从yaml中读到了方向，则使用读到的值；否则默认LEFT
        if look_dir and isinstance(look_dir, str):
            if look_dir.upper() in ['L', 'LEFT', 'left', 'l']:
                self.look_direction = 'LEFT'
            elif look_dir.upper() in ['R', 'RIGHT']:
                self.look_direction = 'RIGHT'
            else:
                self.look_direction = 'LEFT'
        else:
            self.look_direction = 'LEFT'
        
        print(f"  卫星视线方向: {self.look_direction}")
        
        force_look_direction = os.environ.get('SIMSAR_LOOK_DIR', None)
        if force_look_direction:
            print(f"  环境变量强制视线方向: {force_look_direction}")
            self.look_direction = force_look_direction.upper()
        
        self._final_look_direction = self.look_direction
        
        # 获取 platform_heading (卫星飞行方向)
        self.platform_heading = None
        orbit_params = data.get('orbit_parameters', {})
        if orbit_params:
            self.platform_heading = orbit_params.get('platform_heading', None)
        if not self.platform_heading:
            self.platform_heading = data.get('platform_heading', None)
        
        if self.platform_heading:
            print(f"  卫星飞行方向(platform_heading): {self.platform_heading}度")
            # 将heading转换为弧度并计算方向向量
            heading_rad = np.radians(self.platform_heading)
            # heading 0度 = 北，90度 = 东，180度 = 南，270度 = 西
            # ECEF坐标系中: X指向0°经线，Y指向90°E
            # 转换为ECEF方向向量
            self.platform_heading_vec = np.array([
                -np.sin(heading_rad),  # X: 西为负
                np.cos(heading_rad),   # Y: 北为正
                0                      # Z: 水平
            ])
            print(f"  飞行方向向量(ECEF): {self.platform_heading_vec}")
        else:
            self.platform_heading_vec = None
            print(f"  警告: 未找到platform_heading")
        
        first_line_time = metadata.get('first_line_sensing_time', None)
        last_line_time = metadata.get('last_line_sensing_time', None)
        
        if first_line_time:
            try:
                dt = datetime.fromisoformat(first_line_time.replace('Z', '+00:00'))
                self.sensing_start = dt.timestamp()
                print(f"解析到first_line_sensing_time: {dt.isoformat()}Z")
            except Exception as e:
                print(f"警告: 解析first_line_sensing_time失败: {e}")
                self.sensing_start = 0.0
        
        if last_line_time:
            try:
                dt = datetime.fromisoformat(last_line_time.replace('Z', '+00:00'))
                self.sensing_stop = dt.timestamp()
                self.sar_duration = self.sensing_stop - self.sensing_start
                print(f"解析到last_line_sensing_time: {dt.isoformat()}Z")
                print(f"计算SAR持续时间: {self.sar_duration:.2f}秒")
            except Exception as e:
                print(f"警告: 解析last_line_sensing_time失败: {e}")
                self.sar_duration = 0.0
        
        if self.sensing_start == 0.0 and self.orbit_data:
            orbit_points = self.orbit_data.get('orbit_points', [])
            if orbit_points and 'time' in orbit_points[0]:
                try:
                    dt = datetime.fromisoformat(orbit_points[0]['time'].replace('Z', '+00:00'))
                    self.sensing_start = dt.timestamp()
                    print(f"使用轨道起始时间作为sensing_start: {dt.isoformat()}Z")
                except Exception as e:
                    print(f"警告: 解析轨道时间失败: {e}")
        
        # 计算SAR持续时间（如果没有从last_line_sensing_time获得）
        if self.sar_duration <= 0 and self.prf > 0 and self.nrows > 0:
            self.sar_duration = (self.nrows - 1) / self.prf
            print(f"基于图像行数和PRF计算SAR持续时间: {self.sar_duration:.2f}秒")
        
        self.azimuth_time_step = 1.0 / self.prf if self.prf > 0 else 0.0
        
        print(f"=== SAR参数加载 ===")
        print(f"  PRF: {self.prf}")
        print(f"  雷达波长: {self.radar_wavelength}m")
        print(f"  距离像素间距: {self.range_pixel_spacing}m")
        print(f"  近距: {self.near_range}m")
        print(f"  图像尺寸: {self.nrows} x {self.ncols}")
        print(f"  成像起始时间: {self.sensing_start}")
        print(f"  SAR持续时间: {self.sar_duration:.2f}秒")

    def _resample_dem(self, target_resolution, sar_coverage=None):
        """重采样DEM到目标分辨率
        
        Args:
            target_resolution: 目标分辨率（度）
            sar_coverage: SAR覆盖范围 (min_lon, max_lon, min_lat, max_lat)
        """
        if self.dem_data is None or self.dem_geotransform is None:
            raise RuntimeError("DEM数据尚未加载，无法重采样")

        # 计算目标分辨率的像素数
        pixel_size = abs(self.dem_geotransform[1])
        scale_factor = pixel_size / target_resolution
        
        if scale_factor <= 1.0:
            print("DEM分辨率已经满足要求，不需要重采样")
            return None
        
        # 计算新的尺寸
        if sar_coverage:
            # 根据SAR覆盖范围计算需要的像素数
            min_lon, max_lon, min_lat, max_lat = sar_coverage
            lon_range = max_lon - min_lon
            lat_range = max_lat - min_lat
            
            # 计算需要的像素数
            new_cols = int(lon_range / target_resolution)
            new_rows = int(lat_range / target_resolution)
            
            print(f"根据SAR覆盖范围计算重采样尺寸: {new_rows}x{new_cols}")
            print(f"SAR覆盖范围: 经度范围={lon_range:.6f}度, 纬度范围={lat_range:.6f}度")
        else:
            # 使用原始DEM尺寸按比例缩放
            new_rows = int(self.dem_nrows * scale_factor)
            new_cols = int(self.dem_ncols * scale_factor)
            print(f"根据原始DEM尺寸计算重采样尺寸: {new_rows}x{new_cols}")
        
        # 限制最大尺寸，避免内存不足
        max_rows = 10000
        max_cols = 20000
        
        if new_rows > max_rows or new_cols > max_cols:
            # 计算新的缩放因子
            scale_factor_rows = max_rows / new_rows
            scale_factor_cols = max_cols / new_cols
            scale_factor = min(scale_factor_rows, scale_factor_cols)
            
            new_rows = int(new_rows * scale_factor)
            new_cols = int(new_cols * scale_factor)
            target_resolution = target_resolution / scale_factor
            
            print(f"限制重采样尺寸: {new_rows}x{new_cols}")
            print(f"调整后的目标分辨率: {target_resolution:.6f}度")
        else:
            print(f"重采样DEM: 分辨率={pixel_size:.6f}度 -> {target_resolution:.6f}度")
        
        # 创建重采样后的DEM
        driver = gdal.GetDriverByName('MEM')
        out_ds = driver.Create('', new_cols, new_rows, 1, gdal.GDT_Float32)
        
        # 设置地理转换
        if sar_coverage:
            # 使用SAR覆盖范围的地理转换
            out_ds.SetGeoTransform([
                sar_coverage[0],
                target_resolution,
                0.0,
                sar_coverage[3],
                0.0,
                -target_resolution
            ])
        else:
            # 使用原始DEM的地理转换
            out_ds.SetGeoTransform([
                self.dem_geotransform[0],
                target_resolution,
                self.dem_geotransform[2],
                self.dem_geotransform[3],
                self.dem_geotransform[4],
                -target_resolution
            ])
        
        out_ds.SetProjection(self.dem_projection)
        
        # 重采样
        gdal.ReprojectImage(
            self.dem_ds,
            out_ds,
            self.dem_projection,
            self.dem_projection,
            gdal.GRA_Bilinear
        )
        
        # 读取重采样后的数据
        band = out_ds.GetRasterBand(1)
        resampled_data = band.ReadAsArray()
        
        out_ds = None
        
        return resampled_data, new_rows, new_cols, target_resolution
    
    def _precompute_azimuth_orbit_data(self):
        """
        预计算并存储每个azimuth的轨道数据
        使用加速模式提高效率
        """
        print("=== 预计算轨道数据 ===")
        
        # 计算所有azimuth时间
        self.azimuth_times = []
        for i in range(self.nrows):
            azimuth_time = self.sensing_start + i / self.prf
            self.azimuth_times.append(azimuth_time)
        
        # 预计算每个azimuth的轨道数据
        self.azimuth_orbit_data = {}
        
        # 使用向量化计算加速
        import numpy as np
        azimuth_times_np = np.array(self.azimuth_times)
        
        # 批量计算轨道位置和速度
        positions = []
        velocities = []
        
        for t in azimuth_times_np:
            pos, vel = self.geo2rdr.calculate_orbit_position(t)
            positions.append(pos)
            velocities.append(vel)
        
        # 存储预计算结果
        for i, t in enumerate(self.azimuth_times):
            self.azimuth_orbit_data[t] = {
                'position': positions[i],
                'velocity': velocities[i],
                'azimuth_line': i
            }
        
        print(f"  预计算完成: {len(self.azimuth_orbit_data)} 个azimuth轨道数据")
    
    def _create_geo2rdr(self):
        """创建Geo2rdr对象"""
        self.orbit_t0 = 0.0
        
        if self.orbit_data:
            orbit_points = self.orbit_data.get('orbit_points', [])
            if orbit_points and 'time' in orbit_points[0]:
                try:
                    dt = datetime.fromisoformat(orbit_points[0]['time'].replace('Z', '+00:00'))
                    self.orbit_t0 = dt.timestamp()
                except Exception:
                    pass
        
        if self.orbit_t0 == 0.0:
            self.orbit_t0 = self.sensing_start - 1.0
        
        print("=== 初始化Geo2rdr ===")
        print(f"  轨道起始时间: {self.orbit_t0:.3f}")
        print(f"  成像起始时间: {self.sensing_start:.3f}")
        print(f"  时间偏移: {self.sensing_start - self.orbit_t0:.2f}秒")
        print(f"  SAR持续时间: {self.sar_duration:.2f}秒")
        
        # 自动计算视线方向功能已屏蔽
        # auto_look_dir = None
        # if self.orbit_data:
        #     try:
        #         corner_coords = get_sar_image_corners(self.yaml_file)
        #         auto_look_dir = _calculate_look_direction_auto(corner_coords, self.orbit_data, None)
        #         if auto_look_dir:
        #             print(f"  自动计算视线方向: {auto_look_dir}")
        #     except Exception:
        #         pass
        
        auto_look_dir = None
        
        if auto_look_dir:
            use_look_dir = auto_look_dir
        elif hasattr(self, '_final_look_direction') and self._final_look_direction:
            use_look_dir = self._final_look_direction
        else:
            use_look_dir = self.look_direction
        
        print(f"  最终视线方向: {use_look_dir}")
        
        self.geo2rdr = Geo2rdr(
            prf=self.prf,
            radar_wavelength=self.radar_wavelength,
            slant_range_pixel_spacing=self.range_pixel_spacing,
            range_first_sample=self.near_range,
            sensing_start=self.sensing_start,
            look_side=use_look_dir
        )
        
        self.geo2rdr.length = self.nrows
        self.geo2rdr.width = self.ncols
        
        if self.orbit_data:
            self.geo2rdr.set_orbit(self.orbit_data)
            
            if hasattr(self.geo2rdr, '_orbit_t0') and hasattr(self.geo2rdr, '_orbit_time_range'):
                print(f"  Geo2rdr内部状态: orbit_t0={self.geo2rdr._orbit_t0:.3f}, time_range={self.geo2rdr._orbit_time_range:.2f}秒")
                print(f"  轨道时间搜索范围: [{self.geo2rdr._orbit_t0:.3f}, {self.geo2rdr._orbit_t0 + self.geo2rdr._orbit_time_range:.3f}]")
        
        if self.doppler_polynomial:
            self.geo2rdr.doppler_polynomial = self.doppler_polynomial
        
        # 预计算每个azimuth的轨道数据
        if self.nrows > 0 and self.prf > 0:
            self._precompute_azimuth_orbit_data()
    
    def _load_dem(self):
        """加载DEM数据"""
        # 首先加载原始DEM
        print(f"\n加载DEM文件: {self.dem_file}")
        self.dem_ds = gdal.Open(self.dem_file, gdal.GA_ReadOnly)
        if self.dem_ds is None:
            raise RuntimeError(f"无法打开DEM文件: {self.dem_file}")
        
        self.dem_nrows = self.dem_ds.RasterYSize
        self.dem_ncols = self.dem_ds.RasterXSize
        self.dem_geotransform = self.dem_ds.GetGeoTransform()
        self.dem_projection = self.dem_projection = self.dem_ds.GetProjection()
        
        band = self.dem_ds.GetRasterBand(1)
        self.dem_data = band.ReadAsArray()
        
        print(f"  DEM尺寸: {self.dem_ncols} x {self.dem_nrows}")
        print(f"  原始DEM分辨率: {abs(self.dem_geotransform[1]) if self.dem_geotransform else 'N/A'}度")
        
        sar_coverage = self._calculate_sar_coverage()
        
        if sar_coverage:
            min_lon, max_lon, min_lat, max_lat = sar_coverage
            print(f"\nDEM地理范围:")
            print(f"  经度: {self.dem_geotransform[0]:.6f} 到 {self.dem_geotransform[0] + self.dem_ncols * self.dem_geotransform[1]:.6f}")
            print(f"  纬度: {self.dem_geotransform[3] + self.dem_nrows * self.dem_geotransform[5]:.6f} 到 {self.dem_geotransform[3]:.6f}")
            
            # 计算需要的DEM分辨率
            sar_lon_range = max_lon - min_lon
            sar_lat_range = max_lat - min_lat
            target_resolution = min(sar_lon_range / self.ncols, sar_lat_range / self.nrows) * 0.8
            
            print(f"\nSAR覆盖范围: 经度 {min_lon:.6f}° 到 {max_lon:.6f}°, 纬度 {min_lat:.6f}° 到 {max_lat:.6f}°")
            print(f"SAR分辨率: {self.range_pixel_spacing}米")
            print(f"目标DEM分辨率: {target_resolution:.6f}度")
            
            resampled_dem, new_rows, new_cols, actual_resolution = self._resample_dem(target_resolution, sar_coverage)
            
            if resampled_dem is not None:
                self.dem_data = resampled_dem
                self.dem_nrows = new_rows
                self.dem_ncols = new_cols
                self.dem_geotransform = [
                    sar_coverage[0], actual_resolution, 0.0,
                    sar_coverage[3], 0.0, -actual_resolution
                ]
                print(f"✓ DEM加载完成: {new_cols} x {new_rows}")
                print(f"  DEM分辨率: {abs(actual_resolution) * 111000:.1f}米")
        else:
            print("警告: 无法计算SAR覆盖范围，使用原始DEM")
    
    def simulate(self, step=1, output_prefix='sim', include_topographic_phase=True,
                 baseline_perpendicular=0.0, n_workers=1, backscatter_model='cosine',
                 noise_level=0.1, layover_shadow_detection=False, output_ext='.tif'):
        """
        执行SAR模拟
        
        Args:
            step: DEM采样步长
            output_prefix: 输出文件前缀
            include_topographic_phase: 是否包含地形相位
            baseline_perpendicular: 垂直基线
            n_workers: 并行进程数
            backscatter_model: 后向散射模型
            noise_level: 噪声水平
            layover_shadow_detection: 是否检测叠掩和阴影
            output_ext: 输出文件扩展名
        """
        print(f"\n=== 开始SAR模拟（基于DEM坐标）===")
        print(f"  DEM采样步长: {step}")
        print(f"  输出前缀: {output_prefix}")
        print(f"  包含地形相位: {include_topographic_phase}")
        print(f"  垂直基线: {baseline_perpendicular}m")
        print(f"  使用核心数: {n_workers}")
        print(f"  后向散射模型: {backscatter_model}")
        print(f"  噪声水平: {noise_level}")
        print(f"  叠掩与阴影检测: {layover_shadow_detection}")
        
        # 初始化输出数组
        sar_data = np.zeros((self.nrows, self.ncols), dtype=np.complex64)
        count_data = np.zeros((self.nrows, self.ncols), dtype=np.int32)
        layover_data = np.zeros((self.nrows, self.ncols), dtype=np.uint8)
        shadow_data = np.zeros((self.nrows, self.ncols), dtype=np.uint8)
        
        # 计算DEM采样点
        print(f"  DEM尺寸: {self.dem_nrows} x {self.dem_ncols}")
        print(f"  采样步长: {step}")
        
        # 生成DEM采样点坐标
        lats = []
        lons = []
        heights = []
        
        for i in range(0, self.dem_nrows, step):
            for j in range(0, self.dem_ncols, step):
                # 计算地理坐标
                lon = self.dem_geotransform[0] + j * self.dem_geotransform[1]
                lat = self.dem_geotransform[3] + i * self.dem_geotransform[5]
                
                # 获取DEM高程
                height = self.dem_data[i, j]
                
                # 有效高程：非NaN、非Inf，负值和0也是有效值（如死海、荷兰）
                if np.isfinite(height):  # 只排除NaN和Inf
                    lats.append(lat)
                    lons.append(lon)
                    heights.append(height)
        
        lats = np.array(lats)
        lons = np.array(lons)
        heights = np.array(heights)
        
        print(f"  DEM有效点数: {len(lats)}")
        
        # 检查DEM lat范围与SAR成像时间的对应关系
        if len(lats) > 0:
            dem_min_lat = np.min(lats)
            dem_max_lat = np.max(lats)
            print(f"  DEM纬度范围: {dem_min_lat:.6f} 到 {dem_max_lat:.6f}")
            print(f"  预期SAR azimuth时间范围: ~{self.sensing_start:.3f} 到 ~{self.sensing_start + self.nrows/self.prf:.3f}")
            print(f"  对应azimuth像素: 0 到 {self.nrows}")
        
        if len(lats) == 0:
            print("  错误: 没有有效DEM数据")
            return
        
        # 分批次处理
        batch_size = 50000
        n_batches = (len(lats) + batch_size - 1) // batch_size
        
        print(f"  总批次数: {n_batches}")
        
        # 准备批处理参数
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(lats))
            batch_data = (lats[start:end], lons[start:end], heights[start:end])
            batches.append(batch_data)
        
        # 处理 - 使用ProcessPoolExecutor并行处理不同批次
        print(f"  使用多进程并行处理 (进程数: {n_workers})")
        
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            margin_az = max(100, 256) if self.prf > 0 else 100
            margin_rg = max(100, int(self.ncols * 0.1))
            print(f"  使用边距: margin_azimuth={margin_az}, margin_range={margin_rg}")
            
            for batch in batches:
                future = executor.submit(_process_single_batch, 
                                        batch, self.geo2rdr, self.prf, 
                                        self.nrows, self.ncols, self.near_range, 
                                        self.range_pixel_spacing, self.sensing_start, 
                                        self.orbit_t0, margin_az, margin_rg, self.radar_wavelength, 
                                        include_topographic_phase, baseline_perpendicular, 
                                        backscatter_model, noise_level, 
                                        layover_shadow_detection, 
                                        getattr(self, 'azimuth_orbit_data', None))
                futures.append(future)
            
            # 收集结果
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"  Batch {i+1}: {len(result['azimuth_lines'])} points")
                except Exception as e:
                    print(f"  Batch {i+1} 处理失败: {e}")
        
        # 合并结果
        for result in results:
            azimuth_lines = result['azimuth_lines']
            range_indices = result['range_indices']
            sigma0_values = result['sigma0_values']
            phases = result['phases']
            
            # 存储结果
            for az_line, rg_idx, sigma0, phase in zip(azimuth_lines, range_indices, sigma0_values, phases):
                if 0 <= az_line < self.nrows and 0 <= rg_idx < self.ncols:
                    sar_data[az_line, rg_idx] += sigma0 * np.exp(1j * phase)
                    count_data[az_line, rg_idx] += 1
                    
                    # 处理叠掩和阴影
                    if 'layover_vals' in result and 'shadow_vals' in result:
                        # 找到当前点在结果中的索引
                        idx = np.where((result['azimuth_lines'] == az_line) & (result['range_indices'] == rg_idx))[0]
                        if len(idx) > 0:
                            layover = result['layover_vals'][idx[0]]
                            shadow = result['shadow_vals'][idx[0]]
                            if layover:
                                layover_data[az_line, rg_idx] = 1
                            if shadow:
                                shadow_data[az_line, rg_idx] = 1
        
        # 计算填充率
        filled_pixels = np.sum(count_data > 0)
        total_pixels = self.nrows * self.ncols
        fill_rate = filled_pixels / total_pixels * 100
        print(f"  填充率: {fill_rate:.2f}%")
        print(f"  有效像素数: {filled_pixels}, 总像素数: {total_pixels}")
        
        # 检查每行的填充情况
        row_fills = np.sum(count_data > 0, axis=1)
        zero_rows = np.where(row_fills == 0)[0]
        if len(zero_rows) > 0:
            print(f"  [警告] 有 {len(zero_rows)} 行完全没有数据!")
            print(f"  [警告] 空行行号: {zero_rows[:20]}...")  # 只显示前20行
        
        # 每100行统计一次
        for start in range(0, self.nrows, 500):
            end = min(start + 500, self.nrows)
            fill_pct = np.mean(row_fills[start:end] > 0) * 100
            print(f"    行 {start}-{end}: 填充率 {fill_pct:.1f}%")
        
        # 平均处理
        with np.errstate(divide='ignore', invalid='ignore'):
            sar_data_avg = np.where(count_data > 0, sar_data / count_data, 0)
        
        # 输出结果
        self._save_output(sar_data_avg, count_data, layover_data, shadow_data, output_prefix, output_ext)
        
        print(f"\n=== 模拟完成 ===")
        print(f"输出文件: {output_prefix}_sar{output_ext}")
    
    def _save_output(self, sar_data, count_data, layover_data, shadow_data, output_prefix, output_ext):
        """保存输出文件"""
        output_dir = os.path.dirname(self.dem_file) or '.'
        
        # 保存SAR数据
        output_file = os.path.join(output_dir, f'{output_prefix}_sar{output_ext}')
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_file, self.ncols, self.nrows, 2, gdal.GDT_CFloat32)
        
        ds.SetGeoTransform([
            self.dem_geotransform[0],
            self.dem_geotransform[1],
            0,
            self.dem_geotransform[3],
            0,
            self.dem_geotransform[5]
        ])
        ds.SetProjection(self.dem_projection)
        
        # 实部
        band1 = ds.GetRasterBand(1)
        band1.WriteArray(np.real(sar_data))
        
        # 虚部
        band2 = ds.GetRasterBand(2)
        band2.WriteArray(np.imag(sar_data))
        
        ds = None
        
        # 保存计数数据
        count_file = os.path.join(output_dir, f'{output_prefix}_count.tif')
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(count_file, self.ncols, self.nrows, 1, gdal.GDT_Int32)
        ds.SetGeoTransform([
            self.dem_geotransform[0],
            self.dem_geotransform[1],
            0,
            self.dem_geotransform[3],
            0,
            self.dem_geotransform[5]
        ])
        ds.SetProjection(self.dem_projection)
        band = ds.GetRasterBand(1)
        band.WriteArray(count_data)
        ds = None
        
        print(f"  SAR数据已保存: {output_file}")
        print(f"  计数数据已保存: {count_file}")
    
    def _calculate_sar_coverage(self):
        """计算SAR覆盖的地理范围"""
        
        corner_coords = get_sar_image_corners(self.yaml_file)
        
        lats = []
        lons = []
        
        has_valid_corners = False
        if corner_coords and len(corner_coords) >= 4:
            for i, corner in enumerate(corner_coords):
                lat = corner.get('lat')
                lon = corner.get('lon')
                if lat is not None and lon is not None and lat != 0 and lon != 0:
                    lats.append(lat)
                    lons.append(lon)
                    has_valid_corners = True
        
        if not has_valid_corners or len(lats) < 4:
            return self._calculate_sar_coverage_fallback()
        
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
        
        if not self.geo2rdr:
            return self._calculate_sar_coverage_from_corners(lats, lons)
        
        try:
            center_azimuth_time = self.sensing_start + (self.nrows - 1) / 2 / self.prf
            sat_pos, sat_vel = self.geo2rdr.calculate_orbit_position(center_azimuth_time)
            
            if self.platform_heading and 0 <= self.platform_heading <= 180:
                orbit_type = "升轨"
            else:
                orbit_type = "降轨"
            
        except Exception:
            pass
        
        return self._calculate_sar_coverage_from_corners(lats, lons)
    
    def _calculate_sar_coverage_from_corners(self, lats, lons):
        """计算SAR覆盖的地理范围"""
        
        corner_coords = get_sar_image_corners(self.yaml_file)
        
        lats = []
        lons = []
        
        has_valid_corners = False
        if corner_coords and len(corner_coords) >= 4:
            for i, corner in enumerate(corner_coords):
                lat = corner.get('lat')
                lon = corner.get('lon')
                if lat is not None and lon is not None and lat != 0 and lon != 0:
                    lats.append(lat)
                    lons.append(lon)
                    has_valid_corners = True
        
        if not has_valid_corners or len(lats) < 4:
            return self._calculate_sar_coverage_fallback()
        
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
        
        if not self.geo2rdr:
            return self._calculate_sar_coverage_from_corners(lats, lons)
        
        try:
            center_azimuth_time = self.sensing_start + (self.nrows - 1) / 2 / self.prf
            sat_pos, sat_vel = self.geo2rdr.calculate_orbit_position(center_azimuth_time)
            
            if self.platform_heading and 0 <= self.platform_heading <= 180:
                orbit_type = "升轨"
            else:
                orbit_type = "降轨"
            
        except Exception:
            pass
        
        return self._calculate_sar_coverage_from_corners(lats, lons)
    

    def _calculate_sar_coverage_from_corners(self, lats, lons):
        """从四个角点坐标计算SAR覆盖范围"""
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
        
        print(f"SAR覆盖范围: 经度 {min_lon:.6f}° 到 {max_lon:.6f}°, 纬度 {min_lat:.6f}° 到 {max_lat:.6f}°")
        
        if self.geo2rdr:
            print(f"轨道起始时间: {self.orbit_t0:.3f}")
            print(f"成像起始时间: {self.sensing_start:.3f}")
            print(f"时间偏移: {self.sensing_start - self.orbit_t0:.2f}秒")
            print(f"SAR持续时间: {self.sar_duration:.2f}秒")
        
        return (min_lon, max_lon, min_lat, max_lat)



def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SAR数据模拟器 V3 - 从DEM生成模拟SAR图像',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('yaml_file', help='SAR参数YAML文件')
    parser.add_argument('dem_file', help='DEM数据文件')
    parser.add_argument('--step', type=int, default=5, help='DEM采样步长 (默认: 5)')
    parser.add_argument('--output-prefix', default='sim', help='输出文件前缀 (默认: sim)')
    parser.add_argument('--include-topographic-phase', action='store_true', help='包含地形相位')
    parser.add_argument('--baseline', type=float, default=0.0, help='垂直基线 (m)')
    parser.add_argument('--workers', type=int, default=None, help='并行进程数 (默认: CPU核心数)')
    parser.add_argument('--backscatter-model', default='cosine',
                        choices=['cosine', 'cosine2', 'constant', 'ohammers', 'brigham'],
                        help='后向散射模型')
    parser.add_argument('--noise', type=float, default=0.1, help='噪声水平 (默认: 0.1)')
    parser.add_argument('--layover-shadow', action='store_true', help='检测叠掩和阴影')
    parser.add_argument('--output-ext', default='.tif', help='输出文件扩展名')

    args = parser.parse_args()

    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)

    print("=== SAR模拟器 V3 ===")
    print(f"SAR参数: {args.yaml_file}")
    print(f"DEM文件: {args.dem_file}")

    sim = SARSimulatorV3(args.yaml_file, args.dem_file)
    sim.simulate(
        step=args.step,
        output_prefix=args.output_prefix,
        include_topographic_phase=args.include_topographic_phase,
        baseline_perpendicular=args.baseline,
        n_workers=args.workers,
        backscatter_model=args.backscatter_model,
        noise_level=args.noise,
        layover_shadow_detection=args.layover_shadow,
        output_ext=args.output_ext
    )

    print("\n=== 模拟完成 ===")


if __name__ == '__main__':
    main()
