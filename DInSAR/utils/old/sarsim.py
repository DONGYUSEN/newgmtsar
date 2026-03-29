#!/usr/bin/env python3
"""
SAR 数据模拟模块
功能：基于 DEM 生成模拟 SAR 数据
参考 ISCE2 的 SAR 模拟方法实现
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, List
try:
    from .dem2sar import DemToSarConverter
    from .sar_utils import calculate_look_direction, get_sar_image_corners
except ImportError:
    import sys, os
    pkgdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkgdir not in sys.path:
        sys.path.insert(0, pkgdir)
    from dem2sar import DemToSarConverter
    from sar_utils import calculate_look_direction, get_sar_image_corners
import os
from osgeo import gdal, osr

# 设置OSR异常处理
osr.UseExceptions()

from multiprocessing import Pool, cpu_count
import numba

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


class SarSimulator:
    """
    SAR 数据模拟器
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
    
    def _init_sar_image(self):
        """
        初始化 SAR 图像
        """
        # 创建复数 SAR 图像
        self.sar_image = np.zeros((self.nrows, self.ncols), dtype=np.complex64)
        # 创建振幅和相位图像
        self.amplitude_image = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.phase_image = np.zeros((self.nrows, self.ncols), dtype=np.float32)
    
    def simulate_backscatter(self, height: float, slope: float = 0.0, land_cover: str = 'vegetation', incidence_angle: float = 30.0) -> float:
        """
        模拟后向散射系数
        
        Args:
            height: 高程（米）
            slope: 坡度（度）
            land_cover: 土地覆盖类型 ('vegetation', 'urban', 'water', 'bare')
            incidence_angle: 入射角（度）
            
        Returns:
            后向散射系数
        """
        # 基础后向散射（增加基础值，确保不为零）
        base_backscatter = 0.8
        
        # 土地覆盖类型影响
        land_cover_factors = {
            'vegetation': 0.7,
            'urban': 1.0,
            'water': 0.3,
            'bare': 0.5
        }
        land_factor = land_cover_factors.get(land_cover, 0.7)
        
        # 高程影响（调整系数，确保不为零）
        height_factor = np.exp(-height / 5000.0)  # 高程越高，后向散射越小，但不会太小
        
        # 坡度影响
        slope_factor = np.cos(np.radians(slope))  # 坡度越大，后向散射越小
        
        # 入射角影响（考虑视线方向）
        # 入射角对后向散射的影响：通常入射角增大，后向散射减小
        incidence_factor = np.cos(np.radians(incidence_angle))
        
        # 视线方向影响
        look_direction_factor = 1.0
        if self.look_direction == "left":
            # 左视情况下的调整因子
            look_direction_factor = 1.0
        elif self.look_direction == "right":
            # 右视情况下的调整因子
            look_direction_factor = 1.0
        
        # 综合后向散射
        backscatter = base_backscatter * land_factor * height_factor * slope_factor * incidence_factor * look_direction_factor
        
        # 添加随机波动
        backscatter *= np.random.normal(1.0, 0.2)
        
        # 确保后向散射非负且有最小值
        backscatter = max(0.1, backscatter)
        
        return backscatter
    
    def simulate_phase(self, slant_range: float, azimuth_time: float) -> float:
        """
        模拟相位
        
        Args:
            slant_range: 斜距（米）
            azimuth_time: 方位向时间（秒）
            
        Returns:
            相位（弧度）
        """
        # 确保时间使用 double 精度
        azimuth_time = float(azimuth_time)
        
        # 距离向相位
        range_phase = 4 * np.pi * slant_range / self.radar_wavelength
        
        # 方位向相位（简化模型）
        azimuth_phase = 2 * np.pi * azimuth_time * self.prf
        
        # 综合相位
        phase = range_phase + azimuth_phase
        
        # 归一化到 [-pi, pi]
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
    def simulate_backscatter_numba(self, height, slope, land_cover_code):
        """
        Numba优化的后向散射计算
        """
        # 基础后向散射（增加基础值，确保不为零）
        base_backscatter = 0.8
        
        # 土地覆盖类型影响
        land_cover_factors = {
            0: 0.7,  # vegetation
            1: 1.0,  # urban
            2: 0.3,  # water
            3: 0.5   # bare
        }
        land_factor = land_cover_factors.get(land_cover_code, 0.7)
        
        # 高程影响（调整系数，确保不为零）
        height_factor = np.exp(-height / 5000.0)
        
        # 坡度影响
        slope_factor = np.cos(np.radians(slope))
        
        # 综合后向散射
        backscatter = base_backscatter * land_factor * height_factor * slope_factor
        
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
        
        for i in range(i_start, i_end, step):
            for j in range(j_start, j_end, step):
                try:
                    height = dem_data[i, j]
                    if np.isnan(height) or height < 0:
                        error_count += 1
                        continue
                    
                    slope = 0.0
                    if 0 < i < dem_data.shape[0] - 1 and 0 < j < dem_data.shape[1] - 1:
                        dh_dx = (dem_data[i, j+1] - dem_data[i, j-1]) / (2 * step)
                        dh_dy = (dem_data[i+1, j] - dem_data[i-1, j]) / (2 * step)
                        slope = np.degrees(np.sqrt(dh_dx**2 + dh_dy**2))
                    
                    land_cover_code = 0
                    if height < 10:
                        land_cover_code = 2
                    elif height > 1000 or slope > 30:
                        land_cover_code = 3
                    
                    try:
                        range_sample, azimuth_time = self.converter.convert(i, j)
                        if np.isnan(range_sample) or np.isnan(azimuth_time):
                            error_count += 1
                            continue
                    except Exception:
                        error_count += 1
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
                        backscatter = self.simulate_backscatter(height, slope, 'vegetation', incidence_angle)
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
                except Exception:
                    error_count += 1
        
        # 平均化结果
        mask = count > 0
        local_sar[mask] /= count[mask]
        local_amp[mask] /= count[mask]
        local_phase[mask] /= count[mask]
        
        return local_sar, local_amp, local_phase
    
    def add_noise_vectorized(self, snr: float = 20.0, noise_type: str = 'gaussian'):
        """
        向量化添加噪声
        """
        signal_power = np.mean(np.abs(self.sar_image)**2)
        noise_power = signal_power / (10**(snr / 10))
        
        if noise_type == 'gaussian':
            # 向量化生成高斯噪声
            noise = np.sqrt(noise_power / 2) * (np.random.randn(self.nrows, self.ncols) + 1j * np.random.randn(self.nrows, self.ncols))
        elif noise_type == 'speckle':
            # 向量化生成相干斑噪声
            gamma = 1.0
            speckle = np.random.gamma(gamma, 1/gamma, (self.nrows, self.ncols))
            noise = np.sqrt(noise_power) * (speckle - 1) * np.exp(1j * 2 * np.pi * np.random.rand(self.nrows, self.ncols))
        else:
            noise = np.sqrt(noise_power / 2) * (np.random.randn(self.nrows, self.ncols) + 1j * np.random.randn(self.nrows, self.ncols))
        
        self.sar_image += noise
    
    def simulate(self, step: int = 5, snr: float = None, noise_type: str = 'gaussian', imaging_mode: str = 'stripmap'):
        """
        模拟 SAR 数据
        
        Args:
            step: DEM 采样步长
            snr: 信噪比（dB）
            noise_type: 噪声类型 ('gaussian', 'speckle')
            imaging_mode: 成像模式 ('stripmap', 'spotlight')
        """
        print("开始模拟 SAR 数据...")
        
        # 获取 DEM 数据
        dem_data = self.converter.dem_data
        if dem_data is None:
            raise Exception("DEM 数据未加载")
        
        # 计算 DEM 尺寸
        dem_rows, dem_cols = dem_data.shape
        
        # 计算使用的CPU核心数（80%）
        num_cores = max(1, int(cpu_count() * 0.8))
        
        # 计算分块大小
        chunk_size = dem_rows // num_cores
        chunks = []
        
        for i in range(0, dem_rows, chunk_size):
            i_end = min(i + chunk_size, dem_rows)
            chunks.append((i, i_end, 0, dem_cols, step, dem_data))
        
        # 并行处理
        results = []
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
        
        # 向量化添加噪声
        self.add_noise_vectorized(snr, noise_type)
        
        print("SAR 数据模拟完成！")
    
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
            'simulation_date': Path(__file__).stat().st_mtime
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
    
    parser = argparse.ArgumentParser(description='SAR 数据模拟工具')
    parser.add_argument('yaml_file', help='SAR YAML 文件路径')
    parser.add_argument('dem_file', help='DEM 文件路径')
    parser.add_argument('--step', type=int, default=10, help='DEM 采样步长')
    parser.add_argument('--snr', type=float, default=None, help='信噪比（dB）；default use yaml/config')
    parser.add_argument('--noise-type', default='gaussian', choices=['gaussian', 'speckle'], help='噪声类型')
    parser.add_argument('--speckle-gamma', type=float, default=1.0, help='speckle 伽马参数')
    parser.add_argument('--correlated-noise', action='store_true', help='生成空间相关噪声')
    parser.add_argument('--imaging-mode', default='stripmap', choices=['stripmap', 'spotlight'], help='成像模式')
    
    args = parser.parse_args()
    
    print("=== SAR 数据模拟工具 ===")
    print(f"SAR YAML 文件: {args.yaml_file}")
    print(f"DEM 文件: {args.dem_file}")
    print(f"DEM 采样步长: {args.step}")
    print(f"信噪比: {args.snr} dB")
    print(f"噪声类型: {args.noise_type}")
    print(f"成像模式: {args.imaging_mode}")
    
    try:
        # 初始化模拟器
        simulator = SarSimulator(
            args.yaml_file,
            args.dem_file,
            noise_snr = args.snr if args.snr is not None else 20.0,
            speckle_gamma = args.speckle_gamma,
            correlated_noise = args.correlated_noise
        )
        # 模拟 SAR 数据
        simulator.simulate(step=args.step, snr=args.snr, noise_type=args.noise_type, imaging_mode=args.imaging_mode)
        
        # 定义输出文件名
        simsar_tif = 'simsar.tif'
        simsar_vrt = 'simsar.vrt'
        simsar_yaml = 'simsar.yaml'
        
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
        
    except Exception as e:
        print(f"模拟失败: {e}")


if __name__ == '__main__':
    main()
