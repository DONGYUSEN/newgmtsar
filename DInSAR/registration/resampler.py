#!/usr/bin/env python3
"""
图像重采样模块
功能：使用sinc方法对复数图像进行重采样
参考：/home/ysdong/Software/GMTSAR/bin/resamp_omp
优化：参考ISCE2重采样方法，提高效率
"""

import numpy as np
from typing import Tuple, Optional, Dict
import math
from concurrent.futures import ProcessPoolExecutor
import gc
import os
import sys
from pathlib import Path
import yaml
try:
    import numba
except ImportError:
    class _NumbaFallback:
        @staticmethod
        def jit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def prange(*args):
            return range(*args)

        @staticmethod
        def set_num_threads(*args, **kwargs):
            return None

    numba = _NumbaFallback()

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from sar_utils import read_yaml, write_yaml, generate_offset_field, _i0, _kaiser_window

# ISCE2风格的预计算权重表
_kernel_cache = {}

# 常量
PI = math.pi

class ImageResampler:
    """图像重采样器"""
    
    def __init__(self, sinc_window_size: int = 9, beta: float = 2.5, resolution: int = 1000):
        """初始化重采样器
        
        Args:
            sinc_window_size: sinc窗口大小，默认9（与GMTSAR一致）
            beta: kaiser窗口参数，默认2.5（与ISCE2一致）
            resolution: 权重表分辨率，默认1000
        """
        self.sinc_window_size = sinc_window_size
        self.ns = sinc_window_size
        self.ns2 = sinc_window_size // 2
        self.beta = beta
        self.resolution = resolution
        # 预计算核
        self.kernel, self.hnm = self._precompute_kernel()
    
    def _precompute_kernel(self):
        """预计算sinc和kaiser组合核
        
        Returns:
            预计算的核和hnm值
        """
        cache_key = (self.ns, self.resolution, self.beta)
        if cache_key in _kernel_cache:
            return _kernel_cache[cache_key]
        
        hnm = self.ns * self.resolution // 2
        
        # 预计算sinc函数
        sincc = np.zeros(2 * hnm + 1, dtype=np.float32)
        for i in range(-hnm, hnm + 1):
            x = i / self.resolution
            arg = abs(PI * x)
            if arg > 1e-8:
                sincc[i + hnm] = math.sin(arg) / arg
            else:
                sincc[i + hnm] = 1.0
        
        # 预计算kaiser窗口
        kaiserc = _kaiser_window(2 * hnm + 1, self.beta)
        
        # 组合核
        kernel = sincc * kaiserc.astype(np.float32)
        
        _kernel_cache[cache_key] = (kernel, hnm)
        return kernel, hnm
    
    def _normalize_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """归一化核函数
        
        Args:
            kernel: 输入核
            
        Returns:
            归一化后的核
        """
        kernel_sum = np.sum(kernel)
        if kernel_sum > 1e-8:
            return kernel / kernel_sum
        return kernel
    
    def resample(self, slave_image: np.ndarray, 
                offset_field: Tuple[np.ndarray, np.ndarray],
                num_workers: int = 8,
                block_size: int = 2000  # 增加分块大小
                ) -> np.ndarray:
        """对辅图像进行重采样（默认使用并行处理）
        
        Args:
            slave_image: 辅图像（复数）
            offset_field: 偏移场 (az_offsets, range_offsets)
            num_workers: 进程数，默认8
            block_size: 分块大小，默认2000行
            
        Returns:
            重采样后的图像
        """
        # 默认使用单进程 + Numba 多线程，避免多进程拷贝大数组开销
        return self.resample_numba(slave_image, offset_field, num_workers=num_workers)

    def resample_numba(self, slave_image: np.ndarray,
                       offset_field: Tuple[np.ndarray, np.ndarray],
                       num_workers: int = 8) -> np.ndarray:
        if slave_image.ndim != 2:
            raise ValueError("输入图像必须是2D复数数组")
        if not np.iscomplexobj(slave_image):
            raise ValueError("输入图像必须是复数数组")

        az_offsets, rg_offsets = offset_field
        if az_offsets.shape != rg_offsets.shape:
            raise ValueError("方位/距离偏移场形状不一致")

        src_height, src_width = slave_image.shape
        out_height, out_width = az_offsets.shape
        if num_workers and num_workers > 0:
            try:
                numba.set_num_threads(int(num_workers))
            except Exception:
                pass
        _, _, resampled = _process_block_numba(
            0, out_height, slave_image, az_offsets, rg_offsets,
            src_height, src_width, out_width, self.ns, self.ns2, self.kernel, self.hnm, self.resolution
        )
        return resampled
    
    def resample_with_offset(self, slave_image: np.ndarray, 
                           initial_offset: Tuple[float, float]
                           ) -> np.ndarray:
        """使用初始偏移对辅图像进行重采样
        
        Args:
            slave_image: 辅图像（复数）
            initial_offset: 初始偏移 (azimuth_offset, range_offset)
            
        Returns:
            重采样后的图像
        """
        height, width = slave_image.shape
        az_offset, rg_offset = initial_offset
        
        # 创建偏移场
        az_offsets = np.full((height, width), az_offset, dtype=np.float32)
        rg_offsets = np.full((height, width), rg_offset, dtype=np.float32)
        
        return self.resample(slave_image, (az_offsets, rg_offsets))
    
    def resample_parallel(self, slave_image: np.ndarray, 
                         offset_field: Tuple[np.ndarray, np.ndarray],
                         num_workers: int = 8,
                         block_size: int = 2000
                         ) -> np.ndarray:
        """并行对辅图像进行重采样
        
        Args:
            slave_image: 辅图像（复数）
            offset_field: 偏移场 (az_offsets, range_offsets)
            num_workers: 进程数，默认8
            block_size: 分块大小，默认2000行
            
        Returns:
            重采样后的图像
        """
        if slave_image.ndim != 2:
            raise ValueError("输入图像必须是2D复数数组")
        
        if not np.iscomplexobj(slave_image):
            raise ValueError("输入图像必须是复数数组")
        
        az_offsets, rg_offsets = offset_field
        if az_offsets.shape != rg_offsets.shape:
            raise ValueError("方位/距离偏移场形状不一致")
        
        src_height, src_width = slave_image.shape
        out_height, out_width = az_offsets.shape
        resampled = np.zeros((out_height, out_width), dtype=np.complex64)
        
        print(f"  开始并行重采样，输入图像大小: {src_height}x{src_width}")
        print(f"  输出网格大小: {out_height}x{out_width}")
        print(f"  Sinc窗口大小: {self.sinc_window_size}x{self.sinc_window_size}")
        print(f"  进程数: {num_workers}")
        print(f"  分块大小: {block_size} 行")
        
        # 计算分块数量
        num_blocks = (out_height + block_size - 1) // block_size
        blocks = []
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, out_height)
            blocks.append((start, end))
        
        print(f"  总块数: {num_blocks}")
        
        # 准备参数列表
        args_list = []
        for block in blocks:
            start, end = block
            args = (
                start, end, slave_image, az_offsets, rg_offsets, 
                src_height, src_width, out_width, self.ns, self.ns2, self.kernel, self.hnm, self.resolution
            )
            args_list.append(args)
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self._process_block, args_list))
        
        # 合并结果
        for start, end, block_result in results:
            resampled[start:end, :] = block_result
        
        print(f"  并行重采样完成")
        return resampled

    def _process_block(self, args):
        """进程池处理函数
        
        Args:
            args: 处理参数
            
        Returns:
            start, end, block_result
        """
        start, end, slave_image, az_offsets, rg_offsets, height, width, ns, ns2, kernel, hnm, resolution = args
        
        # 调用Numba加速的函数
        return _process_block_numba(start, end, slave_image, az_offsets, rg_offsets, height, width, ns, ns2, kernel, hnm, resolution)


@numba.jit(nopython=True, parallel=True, cache=True)
def _process_block_numba(start, end, slave_image, az_offsets, rg_offsets, src_height, src_width, out_width, ns, ns2, kernel, hnm, resolution):
    """Numba加速的进程池处理函数
    
    Args:
        start: 块起始行
        end: 块结束行
        slave_image: 辅图像数据
        az_offsets: 方位向偏移场
        rg_offsets: 距离向偏移场
        src_height: 输入图像高度
        src_width: 输入图像宽度
        out_width: 输出图像宽度
        ns: 窗口大小
        ns2: 窗口大小的一半
        kernel: 预计算的核
        hnm: 核的半长度
        resolution: 分辨率
        
    Returns:
        start, end, block_result
    """
    block_height = end - start
    block_result = np.zeros((block_height, out_width), dtype=np.complex64)
    
    for i in numba.prange(block_height):
        global_row = start + i
        
        for j in range(out_width):
            src_y = global_row + az_offsets[global_row, j]
            src_x = j + rg_offsets[global_row, j]
            
            if src_x < ns2 or src_x >= src_width - ns2 or \
               src_y < ns2 or src_y >= src_height - ns2:
                block_result[i, j] = 0j
                continue
            
            # 计算整数部分和小数部分
            j0 = int(math.floor(src_x))
            i0 = int(math.floor(src_y))
            dx = src_x - j0
            dy = src_y - i0
            
            # 计算权重索引
            rgfn = int(round(dx * resolution))
            azfn = int(round(dy * resolution))
            
            # 提取range方向核
            rg_kernel = np.zeros(ns, dtype=np.float32)
            for k in range(-ns2, ns2 + 1):
                tmp = k * resolution - rgfn
                if tmp > hnm:
                    tmp = hnm
                elif tmp < -hnm:
                    tmp = -hnm
                rg_kernel[k + ns2] = kernel[tmp + hnm]
            
            # 提取azimuth方向核
            az_kernel = np.zeros(ns, dtype=np.float32)
            for k in range(-ns2, ns2 + 1):
                tmp = k * resolution - azfn
                if tmp > hnm:
                    tmp = hnm
                elif tmp < -hnm:
                    tmp = -hnm
                az_kernel[k + ns2] = kernel[tmp + hnm]
            
            # 归一化核
            rg_sum = np.sum(rg_kernel)
            if rg_sum > 1e-8:
                rg_kernel = rg_kernel / rg_sum
            
            az_sum = np.sum(az_kernel)
            if az_sum > 1e-8:
                az_kernel = az_kernel / az_sum
            
            # 两阶段插值
            # 1. Range方向插值
            range_interp = np.zeros(ns, dtype=np.complex64)
            for k1 in range(ns):
                y_idx = i0 - ns2 + k1
                if y_idx < 0 or y_idx >= src_height:
                    continue
                
                window = slave_image[y_idx, j0 - ns2:j0 + ns2 + 1]
                sum_val = 0j
                for k2 in range(ns):
                    sum_val += window[k2] * rg_kernel[k2]
                range_interp[k1] = sum_val
            
            # 2. Azimuth方向插值
            az_interp = 0j
            for k in range(ns):
                az_interp += range_interp[k] * az_kernel[k]
            block_result[i, j] = az_interp
    
    return start, end, block_result


def read_offset_estimate(filename: str) -> Dict:
    """读取offset_estimate.txt文件
    
    Args:
        filename: offset_estimate.txt文件路径
        
    Returns:
        包含偏移参数的字典
    """
    offset_data = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 解析文件内容
    in_fitted_params = False
    in_parameters = False
    in_normalization = False
    points = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line == 'FITTING_FORMULA':
            in_fitted_params = False
            in_parameters = False
            in_normalization = False
        elif line == 'fitted_params:':
            in_fitted_params = True
            offset_data['fitted_params'] = {}
        elif line == 'PARAMETERS':
            in_parameters = True
            offset_data['parameters'] = {}
        elif line == 'NORMALIZATION':
            in_normalization = True
            offset_data['normalization'] = {}
        elif line == 'REGISTRATION_QUALITY':
            in_fitted_params = False
            in_parameters = False
            in_normalization = False
            offset_data['registration_quality'] = {}
        
        elif in_fitted_params and ':' in line:
            key, value = line.split(':', 1)
            try:
                offset_data['fitted_params'][key.strip()] = float(value.strip())
            except ValueError:
                # 非数值字段忽略
                pass
        elif in_parameters and '=' in line:
            key, value = line.split('=', 1)
            try:
                offset_data['parameters'][key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif in_normalization and '=' in line:
            key, value = line.split('=', 1)
            try:
                offset_data['normalization'][key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif line.startswith('azimuth_rms:'):
            key, value = line.split(':', 1)
            try:
                offset_data[key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif line.startswith('range_rms:'):
            key, value = line.split(':', 1)
            try:
                offset_data[key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif line.startswith('polynomial_order:'):
            key, value = line.split(':', 1)
            try:
                offset_data[key.strip()] = int(float(value.strip()))
            except ValueError:
                pass
        elif line.startswith('rows_mean:'):
            key, value = line.split(':', 1)
            try:
                offset_data[key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif line.startswith('rows_std:'):
            key, value = line.split(':', 1)
            try:
                offset_data[key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif line.startswith('cols_mean:'):
            key, value = line.split(':', 1)
            try:
                offset_data[key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif line.startswith('cols_std:'):
            key, value = line.split(':', 1)
            try:
                offset_data[key.strip()] = float(value.strip())
            except ValueError:
                pass
        else:
            # 兼容老格式点行：x dx y dy corr
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x = float(parts[0])
                    dx = float(parts[1])
                    y = float(parts[2])
                    dy = float(parts[3])
                    corr = float(parts[4]) if len(parts) >= 5 else 0.0
                    points.append((x, y, dx, dy, corr))
                except ValueError:
                    pass

    # 若没有参数区，回退到常值偏移模型
    if 'parameters' not in offset_data:
        offset_data['parameters'] = {}
    params = offset_data['parameters']
    if 'a0' not in params or 'b0' not in params:
        if points:
            dx_arr = np.array([p[2] for p in points], dtype=np.float64)
            dy_arr = np.array([p[3] for p in points], dtype=np.float64)
            rows = np.array([p[1] for p in points], dtype=np.float64)
            cols = np.array([p[0] for p in points], dtype=np.float64)
            params['a0'] = float(np.mean(dy_arr))
            params['b0'] = float(np.mean(dx_arr))
            for key in ('a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5'):
                params.setdefault(key, 0.0)
            offset_data.setdefault('rows_mean', float(np.mean(rows)))
            offset_data.setdefault('rows_std', float(np.std(rows) if np.std(rows) > 1e-8 else 1.0))
            offset_data.setdefault('cols_mean', float(np.mean(cols)))
            offset_data.setdefault('cols_std', float(np.std(cols) if np.std(cols) > 1e-8 else 1.0))
        else:
            for key in ('a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'):
                params.setdefault(key, 0.0)
            offset_data.setdefault('rows_mean', 0.0)
            offset_data.setdefault('rows_std', 1.0)
            offset_data.setdefault('cols_mean', 0.0)
            offset_data.setdefault('cols_std', 1.0)
    
    return offset_data


def update_yaml_with_offsets(master_data: Dict, slave_data: Dict, offset_data: Dict) -> Dict:
    """根据偏移数据和master参数更新YAML文件
    
    Args:
        master_data: 主图像YAML数据
        slave_data: 辅图像YAML数据
        offset_data: 偏移数据
        
    Returns:
        更新后的YAML数据
    """
    # 创建更新后的YAML数据
    updated_yaml = slave_data.copy()
    
    # 1. 更新图像尺寸参数，使其与master一致
    if 'image_parameters' in master_data and 'image_parameters' in updated_yaml:
        updated_yaml['image_parameters']['nrows'] = master_data['image_parameters'].get('nrows', updated_yaml['image_parameters'].get('nrows', 0))
        updated_yaml['image_parameters']['ncols'] = master_data['image_parameters'].get('ncols', updated_yaml['image_parameters'].get('ncols', 0))
    
    # 2. 更新雷达参数
    if 'radar_parameters' in master_data and 'radar_parameters' in updated_yaml:
        if 'range_sampling_rate' in master_data['radar_parameters']:
            updated_yaml['radar_parameters']['range_sampling_rate'] = master_data['radar_parameters']['range_sampling_rate']
        if 'prf' in master_data['radar_parameters']:
            updated_yaml['radar_parameters']['prf'] = master_data['radar_parameters']['prf']
    
    # 3. 更新metadata部分
    if 'metadata' in updated_yaml:
        # 添加偏移相关信息
        updated_yaml['metadata']['registration'] = {
            'azimuth_rms': offset_data.get('azimuth_rms', 0.0),
            'range_rms': offset_data.get('range_rms', 0.0),
            'polynomial_order': offset_data.get('polynomial_order', 2)
        }
    
    # 4. 更新orbit_parameters部分
    if 'orbit_parameters' in updated_yaml:
        # 计算并添加偏移相关参数
        parameters = offset_data.get('parameters', {})
        updated_yaml['orbit_parameters']['registration_offset'] = {
            'azimuth_a0': parameters.get('a0', 0.0),
            'range_b0': parameters.get('b0', 0.0)
        }
    
    # 5. 更新处理参数
    if 'processing_parameters' in updated_yaml:
        updated_yaml['processing_parameters']['registration'] = {
            'offset_estimate_file': 'offset_estimate.txt',
            'fitted_parameters': offset_data.get('fitted_params', {})
        }
    
    # 6. 更新PRM相关参数（如果存在）
    if 'prm_parameters' in master_data and 'prm_parameters' in updated_yaml:
        prm_master = master_data['prm_parameters']
        prm_updated = updated_yaml['prm_parameters']
        
        # 更新PRM参数，与resamp.c保持一致
        prm_updated['num_rng_bins'] = prm_master.get('num_rng_bins', prm_updated.get('num_rng_bins', 0))
        prm_updated['fs'] = prm_master.get('fs', prm_updated.get('fs', 0))
        prm_updated['bytes_per_line'] = prm_master.get('bytes_per_line', prm_updated.get('bytes_per_line', 0))
        prm_updated['good_bytes_per_line'] = prm_master.get('good_bytes_per_line', prm_updated.get('good_bytes_per_line', 0))
        prm_updated['PRF'] = prm_master.get('PRF', prm_updated.get('PRF', 0))
        prm_updated['num_valid_az'] = prm_master.get('num_valid_az', prm_updated.get('num_valid_az', 0))
        prm_updated['num_lines'] = prm_master.get('num_lines', prm_updated.get('num_lines', 0))
        prm_updated['num_patches'] = prm_master.get('num_patches', prm_updated.get('num_patches', 0))
        prm_updated['nrows'] = prm_master.get('nrows', prm_updated.get('nrows', 0))
    
    return updated_yaml


def resample_with_yaml(
    master_yaml: str,
    slave_yaml: str,
    output_yaml: str,
    offset_file: str,
    sinc_window_size: int = 9,
    num_workers: int = 8,
    block_size: int = 2000,
    min_valid_ratio: float = 0.05,
    min_nonzero_ratio: float = 0.01,
):
    """根据YAML文件和偏移数据进行重采样
    
    Args:
        master_yaml: 主图像YAML文件路径
        slave_yaml: 辅图像YAML文件路径
        output_yaml: 输出YAML文件路径
        offset_file: offset_estimate.txt文件路径
        min_valid_ratio: 偏移场可采样覆盖率下限（低于该值判失败）
        min_nonzero_ratio: 输出非零占比下限（低于该值判失败）
    """
    # 读取YAML文件
    master_data = read_yaml(master_yaml)
    slave_data = read_yaml(slave_yaml)
    
    # 读取偏移数据
    offset_data = read_offset_estimate(offset_file)
    
    # 更新YAML数据，将slave参数更新为master参数
    updated_yaml = update_yaml_with_offsets(master_data, slave_data, offset_data)
    if 'processing_parameters' in updated_yaml:
        reg = updated_yaml['processing_parameters'].setdefault('registration', {})
        reg['offset_estimate_file'] = os.path.basename(offset_file)
        reg['sinc_window_size'] = int(sinc_window_size)
        reg['num_workers'] = int(num_workers)
        reg['block_size'] = int(block_size)
    
    # 写入新的YAML文件
    output_dir = os.path.dirname(os.path.abspath(output_yaml))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_yaml(updated_yaml, output_yaml)
    
    # 实际执行重采样操作
    try:
        # 导入GDAL
        from osgeo import gdal
        
        # 获取图像文件路径
        slave_vrt = None
        if 'metadata' in slave_data and 'data_file' in slave_data['metadata']:
            slave_vrt = os.path.join(os.path.dirname(slave_yaml), slave_data['metadata']['data_file'])
            # 如果是SLC文件，尝试找到对应的VRT文件
            if slave_vrt.endswith('.SLC'):
                slave_vrt = slave_vrt.replace('.SLC', '.vrt')
            elif slave_vrt.endswith('.tiff'):
                slave_vrt = slave_vrt.replace('.tiff', '.vrt')
            elif slave_vrt.endswith('.tif'):
                slave_vrt = slave_vrt.replace('.tif', '.vrt')
        
        # 如果没有找到VRT文件，查找目录中的VRT文件
        if not slave_vrt or not os.path.exists(slave_vrt):
            slave_dir = os.path.dirname(slave_yaml) or '.'
            vrt_files = [f for f in os.listdir(slave_dir) if f.endswith('.vrt')]
            if vrt_files:
                # 优先匹配 slave_yaml 同名前缀，避免误用 master.vrt
                slave_stem = os.path.splitext(os.path.basename(slave_yaml))[0]
                preferred = [f for f in vrt_files if os.path.splitext(f)[0] == slave_stem or f.startswith(f"{slave_stem}.")]
                picked = preferred[0] if preferred else sorted(vrt_files)[0]
                slave_vrt = os.path.join(slave_dir, picked)
        
        if not slave_vrt or not os.path.exists(slave_vrt):
            print("警告：未找到辅图像VRT文件，跳过重采样")
            return False
        
        print(f"  开始重采样，使用辅图像: {slave_vrt}")
        
        # 打开辅图像
        slave_ds = gdal.Open(slave_vrt)
        if not slave_ds:
            print(f"无法打开辅图像: {slave_vrt}")
            return False
        
        # 读取辅图像数据
        slave_band = slave_ds.GetRasterBand(1)
        slave_data_array = slave_band.ReadAsArray()
        
        # 转换为复数
        if slave_data_array.dtype == np.float32:
            # 假设是单精度浮点数据，需要组合两个波段
            if slave_ds.RasterCount >= 2:
                slave_real = slave_data_array
                slave_imag = slave_ds.GetRasterBand(2).ReadAsArray()
                slave_image = slave_real + 1j * slave_imag
            else:
                # 如果只有一个波段，假设是幅度数据
                slave_image = slave_data_array + 0j
        else:
            # 直接使用数据
            slave_image = slave_data_array.astype(np.complex64)
        
        # 关闭数据集
        slave_ds = None
        
        print(f"  辅图像大小: {slave_image.shape}, 类型: {slave_image.dtype}")

        # 输出尺寸应严格对齐 master 网格
        master_img = (master_data.get('image_parameters', {}) or {})
        target_height = int(master_img.get('nrows', slave_image.shape[0]))
        target_width = int(master_img.get('ncols', slave_image.shape[1]))
        if target_height <= 0 or target_width <= 0:
            raise ValueError(f"无效目标尺寸: {target_height}x{target_width}")
        print(f"  目标输出大小(与master一致): {target_height}x{target_width}")
        
        # 使用完整的多项式偏移场计算
        az_offsets, range_offsets = generate_offset_field((target_height, target_width), offset_data)
        print(f"  生成多项式偏移场，形状: {az_offsets.shape}")
        print(
            "  偏移场统计: "
            f"az[min/med/max]=({float(np.nanmin(az_offsets)):.3f}, {float(np.nanmedian(az_offsets)):.3f}, {float(np.nanmax(az_offsets)):.3f}), "
            f"rg[min/med/max]=({float(np.nanmin(range_offsets)):.3f}, {float(np.nanmedian(range_offsets)):.3f}, {float(np.nanmax(range_offsets)):.3f})"
        )

        # 预估重采样窗口覆盖率：若偏移场导致几乎全越界，直接判失败并让上层回退。
        half = int(max(1, sinc_window_size // 2))
        rows = np.arange(target_height, dtype=np.float32)[:, None]
        cols = np.arange(target_width, dtype=np.float32)[None, :]
        src_y = rows + az_offsets
        src_x = cols + range_offsets
        valid = (
            np.isfinite(src_y)
            & np.isfinite(src_x)
            & (src_y >= half)
            & (src_y < (slave_image.shape[0] - half))
            & (src_x >= half)
            & (src_x < (slave_image.shape[1] - half))
        )
        valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)
        print(f"  偏移场有效覆盖率: {valid_ratio:.4f}")
        if valid_ratio < float(min_valid_ratio):
            print(
                "  重采样失败: 偏移场有效覆盖率过低，疑似偏移异常（例如 ESD 失稳导致的超大位移）。"
            )
            return False
        
        # 创建重采样器
        resampler = ImageResampler(sinc_window_size=sinc_window_size)
        
        # 执行重采样
        resampled_image = resampler.resample(
            slave_image,
            (az_offsets, range_offsets),
            num_workers=num_workers,
            block_size=block_size,
        )

        # 输出有效性检查：防止“结果文件存在但几乎全 0”的静默错误。
        out_mag = np.abs(resampled_image)
        nz_ratio = float(np.count_nonzero(out_mag > 0.0)) / float(out_mag.size)
        print(f"  重采样输出非零占比: {nz_ratio:.4f}")
        if nz_ratio < float(min_nonzero_ratio):
            print("  重采样失败: 输出几乎全零，保留上一步结果。")
            return False
        
        # 保存为TIFF文件
        output_tiff = output_yaml.replace('.yaml', '.tiff')
        # 创建输出数据集
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_tiff, target_width, target_height, 2, gdal.GDT_Float32)
        if out_ds:
            # 写入实部和虚部
            out_ds.GetRasterBand(1).WriteArray(np.real(resampled_image))
            out_ds.GetRasterBand(2).WriteArray(np.imag(resampled_image))
            out_ds.FlushCache()
            out_ds = None
            print(f"  重采样结果已保存为TIFF: {output_tiff}")
        
        # 生成VRT文件
        output_vrt = output_yaml.replace('.yaml', '.vrt')
        # 直接写入VRT XML内容
        vrt_content = f'''
<VRTDataset rasterXSize="{target_width}" rasterYSize="{target_height}">
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{os.path.basename(output_tiff)}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{os.path.basename(output_tiff)}</SourceFilename>
      <SourceBand>2</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
'''
        with open(output_vrt, 'w') as f:
            f.write(vrt_content)
        print(f"  重采样结果已保存为VRT: {output_vrt}")
        
    except Exception as e:
        print(f"重采样执行失败: {e}")
        return False
    
    print(f"已生成更新后的YAML文件: {output_yaml}")
    print("参数更新完成，确保输出文件大小与主图像一致")
    return True


if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图像重采样模块')
    parser.add_argument('master_yaml', help='主图像YAML文件路径')
    parser.add_argument('slave_yaml', help='辅图像YAML文件路径')
    parser.add_argument('output_yaml', help='输出YAML文件路径')
    parser.add_argument('offset_file', help='offset_estimate.txt文件路径')
    parser.add_argument('--sinc-window-size', type=int, default=9, help='sinc窗口大小')
    parser.add_argument('--num-workers', type=int, default=8, help='并行处理线程数')
    parser.add_argument('--block-size', type=int, default=2000, help='并行处理分块行数')
    
    args = parser.parse_args()
    
    print("=== 图像重采样模块 ===")
    print(f"主图像YAML: {args.master_yaml}")
    print(f"辅图像YAML: {args.slave_yaml}")
    print(f"输出YAML: {args.output_yaml}")
    print(f"偏移估计文件: {args.offset_file}")
    print(f"Sinc窗口大小: {args.sinc_window_size}")
    print(f"并行线程数: {args.num_workers}")
    
    # 执行重采样
    print("\n执行重采样...")
    resample_with_yaml(
        args.master_yaml,
        args.slave_yaml,
        args.output_yaml,
        args.offset_file,
        sinc_window_size=args.sinc_window_size,
        num_workers=args.num_workers,
        block_size=args.block_size,
    )
    
    print("\n重采样完成")
