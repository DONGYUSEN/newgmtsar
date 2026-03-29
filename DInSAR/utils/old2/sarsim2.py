#!/usr/bin/env python3
"""
SAR数据模拟器 V2

参考ISCE2的实现，从DEM模拟SAR SLC数据

关键修正：
1. 使用轨道相对时间进行geo2rdr坐标转换
2. 正确处理时间系统
3. 支持多核并行处理
4. 高效的向量化操作
"""

import os
import sys
import argparse
import numpy as np
import yaml
from datetime import datetime
from typing import Tuple, Optional
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

try:
    from osgeo import gdal
except ImportError:
    import gdal

try:
    from .geo2rdr import Geo2rdr
except ImportError:
    from geo2rdr import Geo2rdr


def _process_chunk_worker(args):
    """多进程worker函数"""
    (i_start, i_end, j_start, j_end, step, dem_data, geotransform, 
     orbit_data, prf, wavelength, range_spacing, near_range, 
     sensing_start, orbit_t0, look_direction, dop_poly, 
     include_topo_phase, baseline_perp, nrows, ncols) = args
    
    sar_chunk = np.zeros((nrows, ncols), dtype=np.complex64)
    count_chunk = np.zeros((nrows, ncols), dtype=np.int32)
    
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
    
    time_offset = sensing_start - orbit_t0
    
    valid_count = 0
    
    for i in range(i_start, i_end, step):
        lat = geotransform[3] + i * geotransform[5]
        
        for j in range(j_start, j_end, step):
            lon = geotransform[0] + j * geotransform[1]
            height = dem_data[i - i_start, j]  # 调整索引，因为我们传递的是子数组
            
            if np.isnan(height) or height < 0:
                continue
            
            try:
                range_sample, azimuth_time_orbit = g2r.geo2rdr(lat, lon, height, return_relative_time=True)
                
                azimuth_time_rel = azimuth_time_orbit - time_offset
                
                if azimuth_time_rel < 0:
                    continue
                
                azimuth_line = int(round(azimuth_time_rel * prf))
                range_idx = int(round(range_sample))
                
                if 0 <= azimuth_line < nrows and 0 <= range_idx < ncols:
                    base_sigma = -15.0
                    height_factor = np.tanh(height / 500.0) * 3.0
                    sigma_db = base_sigma + height_factor + np.random.randn() * 0.5
                    sigma_linear = 10 ** (sigma_db / 20.0)
                    
                    phase = 0.0
                    if include_topo_phase and baseline_perp != 0.0:
                        slant_range = near_range + range_sample * range_spacing
                        incidence_angle = np.deg2rad(23.0)
                        phase = 4 * np.pi * baseline_perp * height * np.sin(incidence_angle) / (wavelength * slant_range)
                    
                    sar_chunk[azimuth_line, range_idx] += sigma_linear * np.exp(1j * phase)
                    count_chunk[azimuth_line, range_idx] += 1
                    valid_count += 1
                    
            except Exception:
                continue
    
    return sar_chunk, count_chunk, valid_count


class SARSimulatorV2:
    """SAR数据模拟器 V2"""
    
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
        
        # 检查轨道数据格式
        self.orbit_data = data.get('orbit_data')
        if self.orbit_data:
            print(f"轨道数据加载成功，包含{len(self.orbit_data.get('orbit_points', []))}个轨道点")
            if 'orbit_points' in self.orbit_data:
                first_point = self.orbit_data['orbit_points'][0]
                if 'position' in first_point:
                    print(f"第一个轨道点位置: 纬度={first_point.get('lat', 'N/A')}, 经度={first_point.get('lon', 'N/A')}")
        
        self.doppler_polynomial = None
        self.sensing_start = 0.0
        self.sar_duration = 0.0
        
        self.prf = radar_params.get('prf', 0.0)
        self.radar_wavelength = radar_params.get('wavelength', 0.0)
        self.range_pixel_spacing = radar_params.get('range_spacing', 0.0)
        self.near_range = radar_params.get('near_range', 0.0)
        self.nrows = image_params.get('nrows', 0)
        self.ncols = image_params.get('ncols', 0)
        
        self.look_direction = 'RIGHT'
        
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
        # 此处应有表达式
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
        """重采样DEM到目标分辨率
        
        Args:
            target_resolution: 目标分辨率（度）
            sar_coverage: SAR覆盖范围 (min_lon, max_lon, min_lat, max_lat)
        """
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

    def _calculate_sar_coverage(self):
        """计算SAR覆盖的地理范围"""
        if not self.geo2rdr:
            print("警告: Geo2rdr未初始化，无法计算SAR覆盖范围")
            return None
        
        # 计算DEM范围
        dem_min_lon = self.dem_geotransform[0]
        dem_max_lon = self.dem_geotransform[0] + self.dem_geotransform[1] * self.dem_ncols
        dem_max_lat = self.dem_geotransform[3]
        dem_min_lat = self.dem_geotransform[3] + self.dem_geotransform[5] * self.dem_nrows
        
        # 确保范围正确（处理负的像素大小）
        if dem_min_lon > dem_max_lon:
            dem_min_lon, dem_max_lon = dem_max_lon, dem_min_lon
        if dem_min_lat > dem_max_lat:
            dem_min_lat, dem_max_lat = dem_max_lat, dem_min_lat
        
        dem_extent = (dem_min_lon, dem_max_lon, dem_min_lat, dem_max_lat)
        print(f"传递DEM范围给rdr2geo: {dem_extent}")
        
        # 计算SAR图像四个角点的地理坐标
        # 左上、右上、左下、右下
        corners = [
            (0, 0),  # 左上角
            (0, self.ncols - 1),  # 右上角
            (self.nrows - 1, 0),  # 左下角
            (self.nrows - 1, self.ncols - 1)  # 右下角
        ]
        
        lats = []
        lons = []
        
        for line, sample in corners:
            try:
                # 计算斜距和方位时间
                slant_range = self.near_range + sample * self.range_pixel_spacing
                # 使用绝对时间（时间戳）而不是相对时间
                azimuth_time = self.sensing_start + line / self.prf
                
                print(f"角点: 线={line}, 采样={sample}")
                print(f"斜距: {slant_range:.2f}米, 方位时间: {azimuth_time}")
                
                # 使用geo2rdr的逆过程（rdr2geo）计算地理坐标
                # 注意：这里假设Geo2rdr类有rdr2geo方法
                # 如果没有，需要实现这个功能
                if hasattr(self.geo2rdr, 'rdr2geo'):
                    lat, lon, height = self.geo2rdr.rdr2geo(slant_range, azimuth_time, dem_extent)
                    print(f"计算得到的地理坐标: 纬度={lat:.6f}, 经度={lon:.6f}, 高度={height:.2f}米")
                    lats.append(lat)
                    lons.append(lon)
            except Exception as e:
                print(f"计算角点时出错: {e}")
                continue
        
        if not lats or not lons:
            print("警告: 无法计算SAR覆盖范围")
            return None
        
        # 计算覆盖范围的边界
        min_lon = min(lons)
        max_lon = max(lons)
        min_lat = min(lats)
        max_lat = max(lats)
        
        # 添加缓冲区，确保覆盖完整
        buffer_deg = 0.01  # 约1km缓冲区
        min_lon -= buffer_deg
        max_lon += buffer_deg
        min_lat -= buffer_deg
        max_lat += buffer_deg
        
        print(f"SAR覆盖范围:")
        print(f"  经度: {min_lon:.6f} 到 {max_lon:.6f}")
        print(f"  纬度: {min_lat:.6f} 到 {max_lat:.6f}")
        
        return (min_lon, max_lon, min_lat, max_lat)

    def _load_dem(self):
        """加载DEM数据"""
        self.dem_ds = gdal.Open(self.dem_file, gdal.GA_ReadOnly)
        if self.dem_ds is None:
            raise ValueError(f"无法打开DEM文件: {self.dem_file}")
        
        self.dem_geotransform = self.dem_ds.GetGeoTransform()
        self.dem_projection = self.dem_ds.GetProjection()
        self.dem_nrows = self.dem_ds.RasterYSize
        self.dem_ncols = self.dem_ds.RasterXSize
        
        # 计算DEM范围
        dem_min_lon = self.dem_geotransform[0]
        dem_max_lon = self.dem_geotransform[0] + self.dem_geotransform[1] * self.dem_ncols
        dem_max_lat = self.dem_geotransform[3]
        dem_min_lat = self.dem_geotransform[3] + self.dem_geotransform[5] * self.dem_nrows
        
        # 确保范围正确（处理负的像素大小）
        if dem_min_lon > dem_max_lon:
            dem_min_lon, dem_max_lon = dem_max_lon, dem_min_lon
        if dem_min_lat > dem_max_lat:
            dem_min_lat, dem_max_lat = dem_max_lat, dem_min_lat
        
        # 计算DEM分辨率（米）
        pixel_size_deg = abs(self.dem_geotransform[1])
        pixel_size_m = pixel_size_deg * 111000  # 1度≈111km
        print(f"原始DEM分辨率: {pixel_size_m:.2f}米")
        print(f"DEM范围:")
        print(f"  经度: {dem_min_lon:.6f} 到 {dem_max_lon:.6f}")
        print(f"  纬度: {dem_min_lat:.6f} 到 {dem_max_lat:.6f}")
        print(f"DEM地理转换参数: {self.dem_geotransform}")
        
        # 计算SAR覆盖范围
        sar_coverage = self._calculate_sar_coverage()
        
        # 裁剪DEM到SAR覆盖范围
        if sar_coverage:
            min_lon, max_lon, min_lat, max_lat = sar_coverage
            
            # 检查SAR覆盖范围是否与DEM范围有重叠
            overlap_min_lon = max(min_lon, dem_min_lon)
            overlap_max_lon = min(max_lon, dem_max_lon)
            overlap_min_lat = max(min_lat, dem_min_lat)
            overlap_max_lat = min(max_lat, dem_max_lat)
            
            if overlap_min_lon < overlap_max_lon and overlap_min_lat < overlap_max_lat:
                # 计算裁剪窗口
                x_min = int((overlap_min_lon - self.dem_geotransform[0]) / self.dem_geotransform[1])
                y_max = int((overlap_max_lat - self.dem_geotransform[3]) / self.dem_geotransform[5])
                x_max = int((overlap_max_lon - self.dem_geotransform[0]) / self.dem_geotransform[1])
                y_min = int((overlap_min_lat - self.dem_geotransform[3]) / self.dem_geotransform[5])
                
                # 确保窗口在DEM范围内
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(self.dem_ncols - 1, x_max)
                y_max = min(self.dem_nrows - 1, y_max)
                
                # 计算裁剪后的大小
                clip_width = x_max - x_min + 1
                clip_height = y_max - y_min + 1
                
                if clip_width > 0 and clip_height > 0:
                    print(f"裁剪DEM到SAR覆盖范围: {clip_height}x{clip_width}")
                    
                    # 读取裁剪后的DEM数据
                    band = self.dem_ds.GetRasterBand(1)
                    self.dem_data = band.ReadAsArray(x_min, y_min, clip_width, clip_height).astype(np.float64)
                    
                    # 更新地理转换
                    self.dem_geotransform = [
                        self.dem_geotransform[0] + x_min * self.dem_geotransform[1],
                        self.dem_geotransform[1],
                        self.dem_geotransform[2],
                        self.dem_geotransform[3] + y_min * self.dem_geotransform[5],
                        self.dem_geotransform[4],
                        self.dem_geotransform[5]
                    ]
                    self.dem_nrows = clip_height
                    self.dem_ncols = clip_width
                else:
                    print("警告: SAR覆盖范围与DEM范围无重叠")
                    band = self.dem_ds.GetRasterBand(1)
                    self.dem_data = band.ReadAsArray().astype(np.float64)
            else:
                print("警告: SAR覆盖范围与DEM范围无重叠")
                band = self.dem_ds.GetRasterBand(1)
                self.dem_data = band.ReadAsArray().astype(np.float64)
        else:
            # 如果无法计算SAR覆盖范围，加载整个DEM
            band = self.dem_ds.GetRasterBand(1)
            self.dem_data = band.ReadAsArray().astype(np.float64)
        
        # 计算SAR覆盖范围
        sar_coverage = self._calculate_sar_coverage()
        
        # 重采样DEM到SAR分辨率
        if self.range_pixel_spacing > 0:
            target_resolution_m = self.range_pixel_spacing
            target_resolution_deg = target_resolution_m / 111000
            
            print(f"SAR分辨率: {target_resolution_m:.2f}米")
            print(f"目标DEM分辨率: {target_resolution_deg:.6f}度")
            
            # 传递SAR覆盖范围给重采样方法
            resampled_result = self._resample_dem(target_resolution_deg, sar_coverage)
            if resampled_result is not None:
                resampled_data, new_rows, new_cols, new_res = resampled_result
                self.dem_data = resampled_data
                self.dem_nrows = new_rows
                self.dem_ncols = new_cols
                
                # 更新地理转换
                if sar_coverage:
                    # 使用SAR覆盖范围的地理转换
                    self.dem_geotransform = [
                        sar_coverage[0],
                        new_res,
                        0.0,
                        sar_coverage[3],
                        0.0,
                        -new_res
                    ]
                else:
                    # 使用原始DEM的地理转换
                    self.dem_geotransform = [
                        self.dem_geotransform[0],
                        new_res,
                        self.dem_geotransform[2],
                        self.dem_geotransform[3],
                        self.dem_geotransform[4],
                        -new_res
                    ]
        
        self.dem_no_data = self.dem_ds.GetRasterBand(1).GetNoDataValue()
        if self.dem_no_data is not None:
            self.dem_data[self.dem_data == self.dem_no_data] = np.nan
        
        self.dem_ds = None
        
        final_res_m = abs(self.dem_geotransform[1]) * 111000
        print(f"✓ DEM加载完成: {self.dem_nrows} x {self.dem_ncols}")
        print(f"  DEM分辨率: {final_res_m:.2f}米")
    
    def _create_geo2rdr(self):
        """创建Geo2rdr实例"""
        print("=== 初始化Geo2rdr ===")
        
        self.geo2rdr = Geo2rdr(
            prf=self.prf,
            radar_wavelength=self.radar_wavelength,
            slant_range_pixel_spacing=self.range_pixel_spacing,
            range_first_sample=self.near_range,
            sensing_start=self.sensing_start,
            look_side=self.look_direction.upper()
        )
        
        if self.orbit_data:
            self.geo2rdr.set_orbit(self.orbit_data)
        
        if self.doppler_polynomial is not None:
            self.geo2rdr.doppler_polynomial = self.doppler_polynomial
        
        self.orbit_t0 = getattr(self.geo2rdr, '_orbit_t0', 0.0)
        print(f"  轨道起始时间: {self.orbit_t0}")
        print(f"  成像起始时间: {self.sensing_start}")
        print(f"  时间偏移: {self.sensing_start - self.orbit_t0:.2f}秒")
        print(f"  SAR持续时间: {self.sar_duration:.2f}秒")
    
    def simulate(self, step: int = None, output_prefix: str = "simsar",
                include_topographic_phase: bool = False,
                baseline_perpendicular: float = 0.0,
                n_workers: int = None):
        """
        运行SAR模拟
        
        Args:
            step: DEM采样步长
            output_prefix: 输出文件前缀
            include_topographic_phase: 是否包含地形相位
            baseline_perpendicular: 垂直基线（用于InSAR干涉）
            n_workers: 并行worker数量
        """
        if step is None:
            step = self._calculate_optimal_step()
        
        if n_workers is None:
            n_workers = min(cpu_count(), 8)  # 增加默认worker数量
        
        print(f"\n=== 开始SAR模拟 ===")
        print(f"  DEM采样步长: {step}")
        print(f"  输出前缀: {output_prefix}")
        print(f"  包含地形相位: {include_topographic_phase}")
        print(f"  垂直基线: {baseline_perpendicular}m")
        print(f"  使用核心数: {n_workers}")
        
        sar_data = np.zeros((self.nrows, self.ncols), dtype=np.complex64)
        count_data = np.zeros((self.nrows, self.ncols), dtype=np.int32)
        
        # 优化块大小计算，确保每个块至少有一定数量的行
        min_chunk_size = 500
        max_chunk_size = 2000
        chunk_size = max(min_chunk_size, min(max_chunk_size, self.dem_nrows // n_workers))
        
        # 生成更合理的块
        chunks = []
        for i_start in range(0, self.dem_nrows, chunk_size):
            i_end = min(i_start + chunk_size, self.dem_nrows)
            # 只传递必要的数据，避免复制大型数组
            chunks.append((
                i_start, i_end, 0, self.dem_ncols, step,
                self.dem_data[i_start:i_end, :], self.dem_geotransform,
                self.orbit_data, self.prf, self.radar_wavelength,
                self.range_pixel_spacing, self.near_range,
                self.sensing_start, self.orbit_t0,
                self.look_direction, self.doppler_polynomial,
                include_topographic_phase, baseline_perpendicular,
                self.nrows, self.ncols
            ))
        
        print(f"  处理块数: {len(chunks)}")
        print(f"  每个块大小: ~{chunk_size}行")
        
        if n_workers > 1 and len(chunks) > 1:
            # 使用imap代替map，减少内存使用
            with Pool(n_workers) as pool:
                results = list(pool.imap(_process_chunk_worker, chunks, chunksize=1))
            
            for sar_chunk, count_chunk, valid_count in results:
                sar_data += sar_chunk
                count_data += count_chunk
        else:
            for chunk in chunks:
                sar_chunk, count_chunk, valid_count = _process_chunk_worker(chunk)
                sar_data += sar_chunk
                count_data += count_chunk
        
        valid_count = np.sum(count_data > 0)
        
        mask = count_data > 0
        amp_data = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        amp_data[mask] = np.abs(sar_data[mask]) / count_data[mask]
        
        fill_rate = 0.0
        if self.nrows > 0 and self.ncols > 0:
            fill_rate = valid_count / self.nrows / self.ncols * 100
        
        print(f"\n=== 模拟完成 ===")
        print(f"  有效像素: {valid_count}")
        print(f"  填充率: {fill_rate:.2f}%")
        
        self._save_output(sar_data, amp_data, output_prefix)
        
        return sar_data, amp_data
    
    def _calculate_optimal_step(self) -> int:
        """计算最优DEM采样步长"""
        dem_resolution = abs(self.dem_geotransform[1]) * 111000  # 转换为米
        
        if self.range_pixel_spacing > 0:
            sar_range_resolution = self.range_pixel_spacing
        else:
            sar_range_resolution = dem_resolution
        
        step = max(1, int(dem_resolution / sar_range_resolution))
        
        return min(step, 10)
    
    def _save_output(self, sar_data: np.ndarray, amp_data: np.ndarray, output_prefix: str):
        """保存输出文件"""
        output_dir = os.path.dirname(output_prefix) or '.'
        
        tif_file = f"{output_prefix}.tif"
        vrt_file = f"{output_prefix}.vrt"
        yaml_file = f"{output_prefix}.yaml"
        
        print(f"\n=== 保存输出 ===")
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(tif_file, self.ncols, self.nrows, 2, gdal.GDT_CFloat32)
        
        if out_ds is None:
            print(f"错误: 无法创建输出文件 {tif_file}")
            return
        
        out_ds.SetGeoTransform([
            0, self.range_pixel_spacing, 0,
            0, 0, self.azimuth_time_step * self.prf
        ])
        
        out_band1 = out_ds.GetRasterBand(1)
        out_band1.WriteArray(sar_data)
        out_band1.SetDescription('Complex')
        
        out_band2 = out_ds.GetRasterBand(2)
        out_band2.WriteArray(amp_data)
        out_band2.SetDescription('Amplitude')
        
        out_ds = None
        
        print(f"  {tif_file}")
        
        self._create_vrt(vrt_file, tif_file)
        print(f"  {vrt_file}")
        
        self._create_yaml(yaml_file, output_prefix)
        print(f"  {yaml_file}")
    
    def _create_vrt(self, vrt_file: str, tif_file: str):
        """创建VRT文件"""
        vrt_content = f'''<VRTDataset rasterXSize="{self.ncols}" rasterYSize="{self.nrows}">
  <VRTRasterBand dataType="CFloat32" band="1">
    <SimpleSource>
      <SourceFilename>{tif_file}</SourceFilename>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>'''
        
        with open(vrt_file, 'w') as f:
            f.write(vrt_content)
    
    def _create_yaml(self, yaml_file: str, output_prefix: str):
        """创建YAML文件"""
        yaml_content = f'''# SAR模拟参数
output_prefix: {output_prefix}
generated_by: sarsim2.py

# 雷达参数
prf: {self.prf}
radar_wavelength: {self.radar_wavelength}
range_pixel_spacing: {self.range_pixel_spacing}
near_range: {self.near_range}

# 图像参数
number_of_lines: {self.nrows}
number_of_samples: {self.ncols}

# 时间参数
first_line_sensing_time: "{datetime.fromtimestamp(self.sensing_start).isoformat()}Z"

# 轨道参数
look_direction: {self.look_direction}
'''
        
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(description='SAR数据模拟器 V2')
    parser.add_argument('yaml_file', help='SAR参数YAML文件')
    parser.add_argument('dem_file', help='DEM文件')
    parser.add_argument('--step', type=int, default=None, help='DEM采样步长')
    parser.add_argument('--output-prefix', default='simsar', help='输出文件前缀')
    parser.add_argument('--include-topographic-phase', action='store_true', help='包含地形相位')
    parser.add_argument('--baseline-perpendicular', type=float, default=0.0, help='垂直基线（米）')
    parser.add_argument('--workers', type=int, default=None, help='并行worker数量')
    
    args = parser.parse_args()
    
    print("=== SAR模拟器 V2 ===")
    print(f"SAR参数: {args.yaml_file}")
    print(f"DEM文件: {args.dem_file}")
    
    sim = SARSimulatorV2(args.yaml_file, args.dem_file)
    
    sim.simulate(
        step=args.step,
        output_prefix=args.output_prefix,
        include_topographic_phase=args.include_topographic_phase,
        baseline_perpendicular=args.baseline_perpendicular,
        n_workers=args.workers
    )


if __name__ == '__main__':
    main()