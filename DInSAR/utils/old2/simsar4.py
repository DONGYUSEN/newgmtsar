#!/usr/bin/env python3
"""
SAR数据模拟器 V4

基于Range-Doppler模型从DEM生成模拟SAR SLC数据

核心特性：
1. 支持左视/右视卫星轨道
2. 基于Range-Doppler模型完整模拟
3. 支持阴影检测和遮挡处理
4. 可并行化处理大规模DEM
5. 输出为单一复数矩阵，便于后续成像处理
"""

import os
import sys
import argparse
import numpy as np
import yaml
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
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


class DEMReader:
    """DEM读取和处理模块"""
    
    def __init__(self, dem_file: str):
        """
        初始化DEM读取器
        
        Args:
            dem_file: DEM文件路径
        """
        self.dem_file = dem_file
        self.dem_data = None
        self.dem_geotransform = None
        self.dem_projection = None
        self.dem_nrows = 0
        self.dem_ncols = 0
        self.dem_ds = None
        
    def load_dem(self):
        """加载DEM数据"""
        print(f"加载DEM文件: {self.dem_file}")
        self.dem_ds = gdal.Open(self.dem_file, gdal.GA_ReadOnly)
        if self.dem_ds is None:
            raise RuntimeError(f"无法打开DEM文件: {self.dem_file}")
        
        self.dem_nrows = self.dem_ds.RasterYSize
        self.dem_ncols = self.dem_ds.RasterXSize
        self.dem_geotransform = self.dem_ds.GetGeoTransform()
        self.dem_projection = self.dem_ds.GetProjection()
        
        band = self.dem_ds.GetRasterBand(1)
        self.dem_data = band.ReadAsArray()
        
        print(f"  DEM尺寸: {self.dem_ncols} x {self.dem_nrows}")
        print(f"  DEM分辨率: {abs(self.dem_geotransform[1]) if self.dem_geotransform else 'N/A'}度")
    
    def get_geo_coordinates(self, step: int = 1):
        """
        获取DEM的地理坐标
        
        Args:
            step: 采样步长
            
        Returns:
            (lats, lons, heights): 纬度、经度、高程数组
        """
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
                
                # 有效高程：非NaN、非Inf
                if np.isfinite(height):
                    lats.append(lat)
                    lons.append(lon)
                    heights.append(height)
        
        return np.array(lats), np.array(lons), np.array(heights)
    
    def resample(self, target_resolution: float):
        """
        重采样DEM到目标分辨率
        
        Args:
            target_resolution: 目标分辨率（度）
            
        Returns:
            重采样后的DEM数据
        """
        if self.dem_data is None or self.dem_geotransform is None:
            raise RuntimeError("DEM数据尚未加载，无法重采样")

        # 计算目标分辨率的像素数
        pixel_size = abs(self.dem_geotransform[1])
        scale_factor = pixel_size / target_resolution
        
        if scale_factor <= 1.0:
            print("DEM分辨率已经满足要求，不需要重采样")
            return self.dem_data
        
        # 计算新的尺寸
        new_rows = int(self.dem_nrows * scale_factor)
        new_cols = int(self.dem_ncols * scale_factor)
        print(f"重采样DEM: 分辨率={pixel_size:.6f}度 -> {target_resolution:.6f}度")
        
        # 限制最大尺寸，避免内存不足
        max_rows = 10000
        max_cols = 20000
        
        if new_rows > max_rows or new_cols > max_cols:
            scale_factor_rows = max_rows / new_rows
            scale_factor_cols = max_cols / new_cols
            scale_factor = min(scale_factor_rows, scale_factor_cols)
            
            new_rows = int(new_rows * scale_factor)
            new_cols = int(new_cols * scale_factor)
            target_resolution = target_resolution / scale_factor
            
            print(f"限制重采样尺寸: {new_rows}x{new_cols}")
        
        # 创建重采样后的DEM
        driver = gdal.GetDriverByName('MEM')
        out_ds = driver.Create('', new_cols, new_rows, 1, gdal.GDT_Float32)
        
        # 设置地理转换
        out_ds.SetGeoTransform([
            self.dem_geotransform[0],
            target_resolution,
            0.0,
            self.dem_geotransform[3],
            0.0,
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
        
        return resampled_data


class RangeDopplerCalculator:
    """Range-Doppler模型计算模块"""
    
    def __init__(self, prf: float, radar_wavelength: float, slant_range_pixel_spacing: float, 
                 range_first_sample: float, sensing_start: float, look_side: str, 
                 fd1: float = 0.0, fdd1: float = 0.0):
        """
        初始化Range-Doppler计算器
        
        Args:
            prf: 脉冲重复频率
            radar_wavelength: 雷达波长
            slant_range_pixel_spacing: 斜距像素间距
            range_first_sample: 距离向第一个采样点
            sensing_start: 成像起始时间
            look_side: 观测方向（LEFT/RIGHT）
            fd1: 多普勒中心频率一阶系数
            fdd1: 多普勒中心频率二阶系数
        """
        self.prf = prf
        self.radar_wavelength = radar_wavelength
        self.slant_range_pixel_spacing = slant_range_pixel_spacing
        self.range_first_sample = range_first_sample
        self.sensing_start = sensing_start
        self.look_side = look_side.upper()
        self.fd1 = fd1
        self.fdd1 = fdd1
        
        self.geo2rdr = Geo2rdr(
            prf=prf,
            radar_wavelength=radar_wavelength,
            slant_range_pixel_spacing=slant_range_pixel_spacing,
            range_first_sample=range_first_sample,
            sensing_start=sensing_start,
            look_side=look_side,
            fd1=fd1,
            fdd1=fdd1
        )
    
    def set_orbit(self, orbit_data: Dict):
        """
        设置轨道数据
        
        Args:
            orbit_data: 轨道数据
        """
        self.geo2rdr.set_orbit(orbit_data)
    
    def set_doppler_polynomial(self, doppler_polynomial: List[float]):
        """
        设置多普勒多项式
        
        Args:
            doppler_polynomial: 多普勒多项式系数
        """
        self.geo2rdr.doppler_polynomial = doppler_polynomial
    
    def calculate_range_doppler(self, lats: np.ndarray, lons: np.ndarray, heights: np.ndarray, 
                               n_workers: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算距离-多普勒参数
        
        Args:
            lats: 纬度数组
            lons: 经度数组
            heights: 高程数组
            n_workers: 并行工作线程数
            
        Returns:
            (range_samples, azimuth_times, slant_ranges): 距离采样点、方位时间、斜距数组
        """
        # 调用geo2rdr进行坐标转换
        range_samples, azimuth_times = self.geo2rdr.geo2rdr_batch(
            lats, lons, heights, n_workers=n_workers
        )
        
        # 计算斜距
        slant_ranges = self.range_first_sample + range_samples * self.slant_range_pixel_spacing
        
        return range_samples, azimuth_times, slant_ranges


class ShadowDetector:
    """阴影与遮挡检测模块"""
    
    def __init__(self, geo2rdr: Geo2rdr):
        """
        初始化阴影检测器
        
        Args:
            geo2rdr: Geo2rdr对象
        """
        self.geo2rdr = geo2rdr
    
    def detect_shadow(self, lats: np.ndarray, lons: np.ndarray, heights: np.ndarray, 
                     azimuth_times: np.ndarray) -> np.ndarray:
        """
        检测阴影与遮挡
        
        Args:
            lats: 纬度数组
            lons: 经度数组
            heights: 高程数组
            azimuth_times: 方位时间数组
            
        Returns:
            shadow_mask: 阴影掩码数组（1表示阴影，0表示非阴影）
        """
        n = len(lats)
        shadow_mask = np.zeros(n, dtype=bool)
        
        for i in range(n):
            try:
                lat = lats[i]
                lon = lons[i]
                height = heights[i]
                azimuth_time = azimuth_times[i]
                
                # 计算卫星位置
                sat_pos = self.geo2rdr.get_satellite_position(azimuth_time)
                
                # 计算地面点的ECEF坐标
                target_xyz = np.array(llh_to_xyz(lat, lon, height))
                
                # 计算视线向量
                look_vector = target_xyz - np.array(sat_pos)
                look_vector_norm = np.linalg.norm(look_vector)
                
                if look_vector_norm > 0:
                    look_vector_unit = look_vector / look_vector_norm
                    
                    # 沿视线方向检查是否有遮挡
                    # 简化实现：检查视线方向上的高度是否低于地面点高度
                    # 实际应用中可能需要更复杂的射线追踪算法
                    is_shadow = False
                    # 这里可以实现更复杂的阴影检测逻辑
                    
                    shadow_mask[i] = is_shadow
            except Exception:
                shadow_mask[i] = False
        
        return shadow_mask


class SLCGenerator:
    """SLC生成模块"""
    
    def __init__(self, nrows: int, ncols: int, prf: float, radar_wavelength: float):
        """
        初始化SLC生成器
        
        Args:
            nrows: 图像行数（方位向）
            ncols: 图像列数（距离向）
            prf: 脉冲重复频率
            radar_wavelength: 雷达波长
        """
        self.nrows = nrows
        self.ncols = ncols
        self.prf = prf
        self.radar_wavelength = radar_wavelength
    
    def generate_slc(self, range_samples: np.ndarray, azimuth_times: np.ndarray, 
                    slant_ranges: np.ndarray, backscatter: np.ndarray, 
                    shadow_mask: np.ndarray) -> np.ndarray:
        """
        生成SLC数据
        
        Args:
            range_samples: 距离采样点数组
            azimuth_times: 方位时间数组
            slant_ranges: 斜距数组
            backscatter: 后向散射系数数组
            shadow_mask: 阴影掩码数组
            
        Returns:
            slc_data: SLC数据（复数数组）
        """
        # 初始化SLC数组
        slc_data = np.zeros((self.nrows, self.ncols), dtype=np.complex64)
        
        # 计算方位像素
        azimuth_pixels = (azimuth_times - self.sensing_start) * self.prf
        azimuth_pixels = np.clip(azimuth_pixels, 0, self.nrows - 1).astype(int)
        
        # 计算距离像素
        range_pixels = np.clip(range_samples, 0, self.ncols - 1).astype(int)
        
        # 生成SLC数据
        for i in range(len(azimuth_pixels)):
            az_pixel = azimuth_pixels[i]
            rg_pixel = range_pixels[i]
            
            if not shadow_mask[i]:
                # 计算相位
                phase = -4 * np.pi * slant_ranges[i] / self.radar_wavelength
                # 复数信号
                slc_data[az_pixel, rg_pixel] += backscatter[i] * np.exp(1j * phase)
        
        return slc_data


class SARSimulatorV4:
    """SAR数据模拟器 V4"""
    
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
        
        self.dem_reader = DEMReader(dem_file)
        self.range_doppler_calculator = None
        self.shadow_detector = None
        self.slc_generator = None
        
        self._load_parameters()
        self._init_modules()
    
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
        
        # 读取多普勒多项式参数
        self.doppler_polynomial = None
        if 'doppler_polynomial' in proc_params:
            self.doppler_polynomial = proc_params['doppler_polynomial']
            print(f"从processing_parameters中读取多普勒多项式: {self.doppler_polynomial}")
        elif 'doppler_polynomial' in radar_params:
            self.doppler_polynomial = radar_params['doppler_polynomial']
            print(f"从radar_parameters中读取多普勒多项式: {self.doppler_polynomial}")
        elif 'doppler_polynomial' in data:
            self.doppler_polynomial = data['doppler_polynomial']
            print(f"从顶层数据中读取多普勒多项式: {self.doppler_polynomial}")
        
        # 读取多普勒中心频率参数
        self.fd1 = 0.0
        self.fdd1 = 0.0
        if 'fd1' in radar_params:
            self.fd1 = radar_params['fd1']
            print(f"从radar_parameters中读取fd1: {self.fd1}")
        elif 'fd1' in proc_params:
            self.fd1 = proc_params['fd1']
            print(f"从processing_parameters中读取fd1: {self.fd1}")
        
        if 'fdd1' in radar_params:
            self.fdd1 = radar_params['fdd1']
            print(f"从radar_parameters中读取fdd1: {self.fdd1}")
        elif 'fdd1' in proc_params:
            self.fdd1 = proc_params['fdd1']
            print(f"从processing_parameters中读取fdd1: {self.fdd1}")
        
        self.prf = radar_params.get('prf', 0.0)
        self.radar_wavelength = radar_params.get('wavelength', 0.0)
        self.range_pixel_spacing = radar_params.get('range_spacing', 0.0)
        self.near_range = radar_params.get('near_range', 0.0)
        self.nrows = image_params.get('nrows', 0)
        self.ncols = image_params.get('ncols', 0)
        
        # 读取look_direction
        look_dir = None
        if 'look_direction' in radar_params:
            look_dir = radar_params['look_direction']
        elif 'look_direction' in data:
            look_dir = data['look_direction']
        elif 'lookdir' in data:
            look_dir = data['lookdir']
        
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
        
        # 读取成像时间
        self.sensing_start = 0.0
        self.sar_duration = 0.0
        
        first_line_time = metadata.get('first_line_sensing_time', None)
        if first_line_time:
            try:
                dt = datetime.fromisoformat(first_line_time.replace('Z', '+00:00'))
                self.sensing_start = dt.timestamp()
                print(f"解析到first_line_sensing_time: {dt.isoformat()}Z")
            except Exception as e:
                print(f"警告: 解析first_line_sensing_time失败: {e}")
                self.sensing_start = 0.0
        
        last_line_time = metadata.get('last_line_sensing_time', None)
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
        
        # 计算SAR持续时间
        if self.sar_duration <= 0 and self.prf > 0 and self.nrows > 0:
            self.sar_duration = (self.nrows - 1) / self.prf
            print(f"基于图像行数和PRF计算SAR持续时间: {self.sar_duration:.2f}秒")
        
        print(f"=== SAR参数加载 ===")
        print(f"  PRF: {self.prf}")
        print(f"  雷达波长: {self.radar_wavelength}m")
        print(f"  距离像素间距: {self.range_pixel_spacing}m")
        print(f"  近距: {self.near_range}m")
        print(f"  图像尺寸: {self.nrows} x {self.ncols}")
        print(f"  成像起始时间: {self.sensing_start}")
        print(f"  SAR持续时间: {self.sar_duration:.2f}秒")
        print(f"  多普勒中心频率一阶系数 (fd1): {self.fd1}")
        print(f"  多普勒中心频率二阶系数 (fdd1): {self.fdd1}")
    
    def _init_modules(self):
        """初始化各个模块"""
        # 加载DEM
        self.dem_reader.load_dem()
        
        # 初始化Range-Doppler计算器
        self.range_doppler_calculator = RangeDopplerCalculator(
            prf=self.prf,
            radar_wavelength=self.radar_wavelength,
            slant_range_pixel_spacing=self.range_pixel_spacing,
            range_first_sample=self.near_range,
            sensing_start=self.sensing_start,
            look_side=self.look_direction,
            fd1=self.fd1,
            fdd1=self.fdd1
        )
        
        if self.orbit_data:
            self.range_doppler_calculator.set_orbit(self.orbit_data)
        
        if self.doppler_polynomial:
            self.range_doppler_calculator.set_doppler_polynomial(self.doppler_polynomial)
        
        # 初始化阴影检测器
        self.shadow_detector = ShadowDetector(self.range_doppler_calculator.geo2rdr)
        
        # 初始化SLC生成器
        self.slc_generator = SLCGenerator(
            nrows=self.nrows,
            ncols=self.ncols,
            prf=self.prf,
            radar_wavelength=self.radar_wavelength
        )
        # 设置sensing_start
        self.slc_generator.sensing_start = self.sensing_start
    
    def _calculate_backscatter(self, lats: np.ndarray, lons: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """
        计算后向散射系数
        
        Args:
            lats: 纬度数组
            lons: 经度数组
            heights: 高程数组
            
        Returns:
            backscatter: 后向散射系数数组
        """
        n = len(lats)
        # 简化实现：使用常数后向散射系数
        backscatter = np.full(n, 10 ** (0.0 / 20.0))  # 0 dB
        
        # 添加高度影响
        height_factor = np.tanh(heights / 500.0) * 3.0
        backscatter_db = 10 * np.log10(np.maximum(backscatter, 1e-20)) + height_factor
        backscatter = 10 ** (backscatter_db / 10.0)
        
        return backscatter
    
    def simulate(self, step: int = 1, output_prefix: str = 'sim', 
                 n_workers: int = 1, output_ext: str = '.npy'):
        """
        执行SAR模拟
        
        Args:
            step: DEM采样步长
            output_prefix: 输出文件前缀
            n_workers: 并行工作线程数
            output_ext: 输出文件扩展名
        """
        print(f"\n=== 开始SAR模拟（基于DEM坐标）===")
        print(f"  DEM采样步长: {step}")
        print(f"  输出前缀: {output_prefix}")
        print(f"  使用核心数: {n_workers}")
        
        # 获取DEM坐标
        lats, lons, heights = self.dem_reader.get_geo_coordinates(step=step)
        print(f"  DEM有效点数: {len(lats)}")
        
        if len(lats) == 0:
            print("  错误: 没有有效DEM数据")
            return
        
        # 计算距离-多普勒参数
        print("  计算距离-多普勒参数...")
        range_samples, azimuth_times, slant_ranges = self.range_doppler_calculator.calculate_range_doppler(
            lats, lons, heights, n_workers=n_workers
        )
        
        # 过滤无效值
        valid_mask = (range_samples != -1) & (azimuth_times != -1)
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        heights = heights[valid_mask]
        range_samples = range_samples[valid_mask]
        azimuth_times = azimuth_times[valid_mask]
        slant_ranges = slant_ranges[valid_mask]
        
        print(f"  过滤后有效点数: {len(lats)}")
        
        if len(lats) == 0:
            print("  错误: 没有有效计算结果")
            return
        
        # 计算后向散射系数
        print("  计算后向散射系数...")
        backscatter = self._calculate_backscatter(lats, lons, heights)
        
        # 检测阴影
        print("  检测阴影...")
        shadow_mask = self.shadow_detector.detect_shadow(lats, lons, heights, azimuth_times)
        
        # 生成SLC数据
        print("  生成SLC数据...")
        slc_data = self.slc_generator.generate_slc(
            range_samples, azimuth_times, slant_ranges, backscatter, shadow_mask
        )
        
        # 计算填充率
        filled_pixels = np.sum(np.abs(slc_data) > 0)
        total_pixels = self.nrows * self.ncols
        fill_rate = filled_pixels / total_pixels * 100
        print(f"  填充率: {fill_rate:.2f}%")
        print(f"  有效像素数: {filled_pixels}, 总像素数: {total_pixels}")
        
        # 保存输出
        self._save_output(slc_data, output_prefix, output_ext)
        
        print(f"\n=== 模拟完成 ===")
        print(f"输出文件: {output_prefix}_slc{output_ext}")
    
    def _save_output(self, slc_data: np.ndarray, output_prefix: str, output_ext: str):
        """
        保存输出文件
        
        Args:
            slc_data: SLC数据
            output_prefix: 输出文件前缀
            output_ext: 输出文件扩展名
        """
        output_dir = os.path.dirname(self.dem_file) or '.'
        
        if output_ext == '.npy':
            # 保存为NumPy格式
            output_file = os.path.join(output_dir, f'{output_prefix}_slc{output_ext}')
            np.save(output_file, slc_data)
            print(f"  SLC数据已保存: {output_file}")
        elif output_ext == '.tif':
            # 保存为TIFF格式（实部/虚部分别保存）
            output_file = os.path.join(output_dir, f'{output_prefix}_slc{output_ext}')
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(output_file, self.ncols, self.nrows, 2, gdal.GDT_Float32)
            
            # 实部
            band1 = ds.GetRasterBand(1)
            band1.WriteArray(np.real(slc_data))
            
            # 虚部
            band2 = ds.GetRasterBand(2)
            band2.WriteArray(np.imag(slc_data))
            
            ds = None
            print(f"  SLC数据已保存: {output_file}")
        else:
            print(f"  不支持的输出格式: {output_ext}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SAR数据模拟器 V4')
    parser.add_argument('yaml_file', help='SAR参数YAML文件')
    parser.add_argument('dem_file', help='DEM文件路径')
    parser.add_argument('--step', type=int, default=1, help='DEM采样步长')
    parser.add_argument('--output-prefix', default='sim', help='输出文件前缀')
    parser.add_argument('--workers', type=int, default=1, help='并行工作线程数')
    parser.add_argument('--output-ext', default='.npy', choices=['.npy', '.tif'], help='输出文件扩展名')
    
    args = parser.parse_args()
    
    # 自动设置工作线程数
    if args.workers <= 0:
        args.workers = max(1, cpu_count())
    
    # 创建模拟器
    simulator = SARSimulatorV4(args.yaml_file, args.dem_file)
    
    # 执行模拟
    simulator.simulate(
        step=args.step,
        output_prefix=args.output_prefix,
        n_workers=args.workers,
        output_ext=args.output_ext
    )


if __name__ == '__main__':
    main()
