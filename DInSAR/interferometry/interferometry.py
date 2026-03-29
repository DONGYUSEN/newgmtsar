#!/usr/bin/env python3
"""
干涉处理模块
功能：生成干涉图，进行滤波处理
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from typing import Tuple, Optional, Dict, Any
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# 尝试导入Numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, using non-accelerated version")


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    """相位归一化到 [-pi, pi]。"""
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def _resample_to_shape(arr: np.ndarray, target_shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    """将 2D 数组重采样到目标形状。"""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape}")
    if arr.shape == target_shape:
        return arr.astype(np.float32, copy=False)
    zoom_factor = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return ndimage.zoom(arr, zoom_factor, order=order).astype(np.float32, copy=False)


def _ensure_grid(value: Any, target_shape: Tuple[int, int], name: str, order: int = 1) -> Optional[np.ndarray]:
    """
    将输入转换为目标形状的 2D 网格：
    - 标量 -> 全图常量
    - 1D (len==W 或 len==H) -> 广播到 2D
    - 2D -> 必要时重采样
    """
    if value is None:
        return None

    arr = np.asarray(value)
    h, w = target_shape

    if arr.ndim == 0:
        return np.full(target_shape, float(arr), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == w:
            return np.tile(arr.reshape(1, w), (h, 1)).astype(np.float32, copy=False)
        if arr.size == h:
            return np.tile(arr.reshape(h, 1), (1, w)).astype(np.float32, copy=False)
        raise ValueError(f"{name} 1D长度={arr.size} 与目标形状={target_shape} 不匹配")

    if arr.ndim == 2:
        return _resample_to_shape(arr, target_shape, order=order)

    raise ValueError(f"{name} 维度不支持: shape={arr.shape}")


def _prepare_phase_model_grids(
    target_shape: Tuple[int, int],
    dem: Optional[np.ndarray],
    range_spacing: float,
    incidence_angle: float,
    geometry: Optional[Dict[str, Any]],
) -> Dict[str, Optional[np.ndarray]]:
    """
    准备平地/地形相位模型所需网格。
    geometry 支持键：
    - incidence_angle (2D/1D/scalar, degree)
    - slant_range (2D/1D/scalar, meter)
    - near_range (scalar, meter)
    - range_spacing (scalar, meter)
    - b_perp (2D/1D/scalar, meter)
    - b_parallel (2D/1D/scalar, meter)
    - height (2D/1D/scalar, meter)
    """
    h, w = target_shape
    geometry = geometry or {}

    inc = _ensure_grid(geometry.get("incidence_angle", None), target_shape, "incidence_angle", order=1)
    if inc is None:
        inc = np.full(target_shape, float(incidence_angle), dtype=np.float32)
    inc = np.clip(inc, 1e-3, 89.999).astype(np.float32, copy=False)

    slant = _ensure_grid(geometry.get("slant_range", None), target_shape, "slant_range", order=1)
    if slant is None:
        near_range = geometry.get("near_range", None)
        rg_spacing = geometry.get("range_spacing", range_spacing)
        if (near_range is not None) and (rg_spacing is not None):
            rg = float(near_range) + np.arange(w, dtype=np.float64) * float(rg_spacing)
            slant = np.tile(rg.reshape(1, w), (h, 1)).astype(np.float32, copy=False)
    if slant is not None:
        slant = np.maximum(slant.astype(np.float32, copy=False), 1e-3)

    b_perp = _ensure_grid(geometry.get("b_perp", None), target_shape, "b_perp", order=1)
    b_parallel = _ensure_grid(geometry.get("b_parallel", None), target_shape, "b_parallel", order=1)

    height = _ensure_grid(geometry.get("height", None), target_shape, "height", order=1)
    if height is None and dem is not None:
        height = _ensure_grid(dem, target_shape, "dem", order=1)

    return {
        "incidence_angle": inc,
        "slant_range": slant,
        "b_perp": b_perp,
        "b_parallel": b_parallel,
        "height": height,
    }


class InterferogramGenerator:
    """干涉图生成类"""
    
    def __init__(self):
        """初始化干涉图生成器"""
        pass
    
    def generate(self, master: np.ndarray, 
                slave: np.ndarray,
                subtract_flat: bool = False,
                dem: Optional[np.ndarray] = None,
                geometry: Optional[Dict[str, Any]] = None,
                wavelength: float = 0.0555,
                range_spacing: float = 5.0,
                azimuth_spacing: float = 10.0,
                incidence_angle: float = 23.0
                ) -> Dict[str, np.ndarray]:
        """生成干涉图
        
        Args:
            master: 主图像 (复数)
            slave: 从图像 (复数)
            subtract_flat: 是否去除平地相位
            dem: DEM数据
            geometry: 几何网格/参数（来自 geosar/baseline）
            wavelength: 雷达波长 (m)
            range_spacing: 距离向像素间距 (m)
            azimuth_spacing: 方位向像素间距 (m)
            incidence_angle: 入射角 (度)
            
        Returns:
            干涉图字典
        """
        # 生成干涉图
        interferogram = master * np.conj(slave)
        
        # 提取相位
        phase = np.angle(interferogram)
        
        result = {
            'interferogram': interferogram,
            'phase': phase,
            'amplitude': np.abs(interferogram)
        }
        
        # 去除平地相位
        if subtract_flat:
            flat_phase = self._calculate_flat_phase(
                dem=dem,
                geometry=geometry,
                wavelength=wavelength,
                range_spacing=range_spacing,
                azimuth_spacing=azimuth_spacing,
                incidence_angle=incidence_angle,
                target_shape=phase.shape,
            )
            result['phase_flat'] = _wrap_phase(phase - flat_phase)
            result['flat_phase'] = flat_phase
            result['interferogram_flat'] = np.abs(interferogram) * np.exp(1j * result['phase_flat'])
        
        return result
    
    def _calculate_flat_phase(self, dem: Optional[np.ndarray],
                             wavelength: float,
                             range_spacing: float,
                             azimuth_spacing: float,
                             incidence_angle: float,
                             target_shape: tuple,
                             geometry: Optional[Dict[str, Any]] = None
                             ) -> np.ndarray:
        """计算平地相位
        
        Args:
            dem: DEM数据
            wavelength: 雷达波长
            range_spacing: 距离向像素间距
            azimuth_spacing: 方位向像素间距
            incidence_angle: 入射角
            target_shape: 目标形状（与SAR图像匹配）
            geometry: 几何网格/参数（推荐提供 baseline 的 b_parallel）
            
        Returns:
            平地相位
        """
        grids = _prepare_phase_model_grids(
            target_shape=target_shape,
            dem=dem,
            range_spacing=range_spacing,
            incidence_angle=incidence_angle,
            geometry=geometry,
        )

        b_parallel = grids["b_parallel"]
        if b_parallel is None:
            # 无基线并行分量时，不构造经验平地相位，避免引入伪相位
            print("警告: 未提供 b_parallel，平地相位将置零。建议提供 baseline.hdf。")
            return np.zeros(target_shape, dtype=np.float32)

        # 符号约定：当前 baseline(b_parallel_los) 定义下，平地项应取正号。
        flat_phase = (4.0 * np.pi / float(wavelength)) * b_parallel
        return _wrap_phase(flat_phase).astype(np.float32, copy=False)


# Numba加速的相干性计算函数
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def numba_coherence_calculator(master, slave, window_size):
        """Numba加速的相干性计算"""
        h, w = master.shape
        half_window = window_size // 2
        coherence = np.zeros((h, w), dtype=np.float32)
        
        # 计算窗口大小
        win_area = window_size * window_size
        
        # 计算相关性
        correlation = master * np.conj(slave)
        
        # 计算功率
        master_power = np.abs(master) ** 2
        slave_power = np.abs(slave) ** 2
        
        # 对每个像素计算相干性
        for i in prange(half_window, h - half_window):
            for j in prange(half_window, w - half_window):
                # 提取窗口
                corr_window = correlation[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                master_win = master_power[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                slave_win = slave_power[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                
                # 计算均值
                corr_mean = np.mean(corr_window)
                master_mean = np.mean(master_win)
                slave_mean = np.mean(slave_win)
                
                # 计算相干性
                numerator = np.abs(corr_mean)
                denominator = np.sqrt(master_mean * slave_mean)
                
                if denominator > 0:
                    coherence[i, j] = numerator / denominator
                else:
                    coherence[i, j] = 0
        
        # 填充边界
        for i in prange(half_window):
            coherence[i, :] = coherence[half_window, :]
            coherence[h-1-i, :] = coherence[h-1-half_window, :]
            coherence[:, i] = coherence[:, half_window]
            coherence[:, w-1-i] = coherence[:, w-1-half_window]
        
        # 裁剪到 [0, 1]
        for i in prange(h):
            for j in prange(w):
                if coherence[i, j] > 1:
                    coherence[i, j] = 1
                elif coherence[i, j] < 0:
                    coherence[i, j] = 0
        
        return coherence

class CoherenceCalculator:
    """相干性计算类"""
    
    def __init__(self, window_size: int = 5):
        """初始化相干性计算器
        
        Args:
            window_size: 窗口大小
        """
        window_size = int(window_size)
        if window_size < 3 or (window_size % 2 == 0):
            raise ValueError(f"window_size 必须是 >=3 的奇数，当前: {window_size}")
        self.window_size = window_size
    
    def calculate(self, master: np.ndarray, 
                 slave: np.ndarray) -> np.ndarray:
        """计算相干性
        
        Args:
            master: 主图像 (复数)
            slave: 从图像 (复数)
            
        Returns:
            相干性图
        """
        h, w = master.shape
        
        # 使用滑动窗口计算相干性
        coherence = np.zeros((h, w), dtype=np.float32)
        
        half_window = self.window_size // 2
        
        for i in range(half_window, h - half_window):
            for j in range(half_window, w - half_window):
                # 提取窗口
                master_window = master[i-half_window:i+half_window+1, 
                                       j-half_window:j+half_window+1]
                slave_window = slave[i-half_window:i+half_window+1, 
                                    j-half_window:j+half_window+1]
                
                # 计算相干性
                numerator = np.abs(np.mean(master_window * np.conj(slave_window)))
                denominator = np.sqrt(np.mean(np.abs(master_window)**2) * 
                                     np.mean(np.abs(slave_window)**2))
                
                if denominator > 0:
                    coherence[i, j] = numerator / denominator
                else:
                    coherence[i, j] = 0
        
        # 填充边界
        coherence[:half_window, :] = coherence[half_window, :]
        coherence[-half_window:, :] = coherence[-half_window-1, :]
        coherence[:, :half_window] = coherence[:, half_window:half_window+1]
        coherence[:, -half_window:] = coherence[:, -half_window-1:-half_window]
        
        return coherence
    
    def calculate_efficient(self, master: np.ndarray,
                           slave: np.ndarray) -> np.ndarray:
        """高效计算相干性
        
        Args:
            master: 主图像 (复数)
            slave: 从图像 (复数)
            
        Returns:
            相干性图
        """
        h, w = master.shape
        block_size = 1024  # 分块大小
        
        # 如果图像较小，直接处理
        if h <= block_size and w <= block_size:
            # 尝试使用Numba加速版本
            if NUMBA_AVAILABLE:
                try:
                    return numba_coherence_calculator(master, slave, self.window_size)
                except Exception as e:
                    print(f"Numba加速失败: {e}, 回退到向量化版本")
            
            # 使用向量化版本
            # 计算相关性
            correlation = master * np.conj(slave)
            
            # 使用高斯滤波近似窗口平均
            kernel_size = self.window_size
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            
            # 计算分子
            correlation_smooth = ndimage.convolve(correlation.real, kernel) + \
                               1j * ndimage.convolve(correlation.imag, kernel)
            
            # 计算分母
            master_power = np.abs(master) ** 2
            slave_power = np.abs(slave) ** 2
            
            master_smooth = ndimage.convolve(master_power, kernel)
            slave_smooth = ndimage.convolve(slave_power, kernel)
            
            # 计算相干性
            denominator = np.sqrt(master_smooth * slave_smooth)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                coherence = np.abs(correlation_smooth) / denominator
                coherence = np.where(denominator > 0, coherence, 0)
            
            # 裁剪到 [0, 1]
            coherence = np.clip(coherence, 0, 1)
            
            return coherence.astype(np.float32)
        else:
            # 使用分块处理
            print(f"检测到大图像 ({h}x{w})，使用分块处理计算相干性")
            coherence = np.zeros((h, w), dtype=np.float32)
            half_window = self.window_size // 2
            
            # 生成所有块的坐标
            blocks = []
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    # 核心块边界
                    dest_i_start = i
                    dest_j_start = j
                    dest_i_end = min(i + block_size, h)
                    dest_j_end = min(j + block_size, w)

                    # 含 halo 的输入块边界
                    i_start = max(0, dest_i_start - half_window)
                    j_start = max(0, dest_j_start - half_window)
                    i_end = min(h, dest_i_end + half_window)
                    j_end = min(w, dest_j_end + half_window)

                    blocks.append((dest_i_start, dest_i_end, dest_j_start, dest_j_end, i_start, i_end, j_start, j_end))
            
            # 定义处理单个块的函数
            def process_block(block_info):
                dest_i_start, dest_i_end, dest_j_start, dest_j_end, i_start, i_end, j_start, j_end = block_info
                
                master_block = master[i_start:i_end, j_start:j_end]
                slave_block = slave[i_start:i_end, j_start:j_end]
                
                # 处理块
                if NUMBA_AVAILABLE:
                    try:
                        block_result = numba_coherence_calculator(master_block, slave_block, self.window_size)
                    except Exception as e:
                        print(f"Numba加速失败: {e}, 回退到向量化版本")
                        # 使用向量化版本
                        correlation = master_block * np.conj(slave_block)
                        kernel_size = self.window_size
                        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
                        correlation_smooth = ndimage.convolve(correlation.real, kernel) + \
                                           1j * ndimage.convolve(correlation.imag, kernel)
                        master_power = np.abs(master_block) ** 2
                        slave_power = np.abs(slave_block) ** 2
                        master_smooth = ndimage.convolve(master_power, kernel)
                        slave_smooth = ndimage.convolve(slave_power, kernel)
                        denominator = np.sqrt(master_smooth * slave_smooth)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            block_result = np.abs(correlation_smooth) / denominator
                            block_result = np.where(denominator > 0, block_result, 0)
                        block_result = np.clip(block_result, 0, 1)
                else:
                    # 使用向量化版本
                    correlation = master_block * np.conj(slave_block)
                    kernel_size = self.window_size
                    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
                    correlation_smooth = ndimage.convolve(correlation.real, kernel) + \
                                       1j * ndimage.convolve(correlation.imag, kernel)
                    master_power = np.abs(master_block) ** 2
                    slave_power = np.abs(slave_block) ** 2
                    master_smooth = ndimage.convolve(master_power, kernel)
                    slave_smooth = ndimage.convolve(slave_power, kernel)
                    denominator = np.sqrt(master_smooth * slave_smooth)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        block_result = np.abs(correlation_smooth) / denominator
                        block_result = np.where(denominator > 0, block_result, 0)
                    block_result = np.clip(block_result, 0, 1)
                
                # 核心块在含 halo 结果中的对应索引
                src_i_start = dest_i_start - i_start
                src_j_start = dest_j_start - j_start
                src_i_end = src_i_start + (dest_i_end - dest_i_start)
                src_j_end = src_j_start + (dest_j_end - dest_j_start)
                
                # 提取结果区域
                result_block = block_result[src_i_start:src_i_end, src_j_start:src_j_end].astype(np.float32)
                
                return (dest_i_start, dest_i_end, dest_j_start, dest_j_end, result_block)
            
            # 使用多线程并行处理
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            max_workers = min(8, len(blocks))  # 最多使用8个线程
            print(f"使用 {max_workers} 个线程并行计算相干性")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有块的处理任务
                future_to_block = {executor.submit(process_block, block): block for block in blocks}
                
                # 处理完成的任务
                for future in as_completed(future_to_block):
                    try:
                        dest_i_start, dest_i_end, dest_j_start, dest_j_end, result_block = future.result()
                        # 将结果复制回完整图像
                        coherence[dest_i_start:dest_i_end, dest_j_start:dest_j_end] = result_block
                    except Exception as e:
                        print(f"处理块失败: {e}")
            
            return coherence


class InterferogramFilter:
    """干涉图滤波类"""
    
    def __init__(self, method: str = 'goldstein'):
        """初始化干涉图滤波器
        
        Args:
            method: 滤波方法
        """
        self.method = method
    
    def filter(self, phase: np.ndarray,
              coherence: Optional[np.ndarray] = None,
              window_size: int = 5,
              sigma: float = 1.0,
              alpha: float = 0.5  # Goldstein滤波参数
              ) -> np.ndarray:
        """滤波干涉图
        
        Args:
            phase: 相位数据
            coherence: 相干性数据
            window_size: 窗口大小
            sigma: 高斯核sigma
            alpha: Goldstein滤波参数（0-1，值越大滤波越强）
            
        Returns:
            滤波后的相位
        """
        window_size = int(window_size)
        if window_size < 3 or (window_size % 2 == 0):
            raise ValueError(f"window_size 必须是 >=3 的奇数，当前: {window_size}")

        if self.method == 'gaussian':
            return self._gaussian_filter(phase, sigma)
        elif self.method == 'adaptive':
            return self._adaptive_filter(phase, coherence, window_size)
        elif self.method == 'median':
            return self._median_filter(phase, window_size)
        elif self.method == 'goldstein':
            return self._goldstein_filter(phase, coherence, alpha)
        else:
            raise ValueError(f"不支持的滤波方法: {self.method}")
    
    def _gaussian_filter(self, phase: np.ndarray, sigma: float) -> np.ndarray:
        """高斯滤波
        
        Args:
            phase: 相位数据
            sigma: 高斯核sigma
            
        Returns:
            滤波后的相位
        """
        # 对相位需要特殊处理
        # 使用余弦和正弦分量分别滤波
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        
        cos_filtered = gaussian_filter(cos_phase, sigma)
        sin_filtered = gaussian_filter(sin_phase, sigma)
        
        # 重建相位
        filtered_phase = np.arctan2(sin_filtered, cos_filtered)
        
        return filtered_phase.astype(np.float32)
    
    def _adaptive_filter(self, phase: np.ndarray,
                        coherence: Optional[np.ndarray],
                        window_size: int
                        ) -> np.ndarray:
        """自适应滤波
        
        Args:
            phase: 相位数据
            coherence: 相干性数据
            window_size: 窗口大小
            
        Returns:
            滤波后的相位
        """
        if coherence is None:
            return self._gaussian_filter(phase, window_size / 3)
        
        # 根据相干性调整滤波强度
        # 低相干区域使用更强的滤波
        h, w = phase.shape
        
        half_window = window_size // 2
        filtered = np.zeros_like(phase)
        
        for i in range(half_window, h - half_window):
            for j in range(half_window, w - half_window):
                # 获取相干性
                coh = coherence[i, j]
                
                # 根据相干性确定滤波窗口
                if coh > 0.8:
                    window = 3
                elif coh > 0.5:
                    window = 5
                elif coh > 0.3:
                    window = 7
                else:
                    window = 9
                
                half_w = window // 2
                phase_window = phase[i-half_w:i+half_w+1, j-half_w:j+half_w+1]
                
                # 相位应在复平面做平均，避免 -pi/pi 跳变引入伪影
                filtered[i, j] = np.angle(np.mean(np.exp(1j * phase_window)))
        
        # 填充边界
        filtered[:half_window, :] = filtered[half_window, :]
        filtered[-half_window:, :] = filtered[-half_window-1, :]
        filtered[:, :half_window] = filtered[:, half_window:half_window+1]
        filtered[:, -half_window:] = filtered[:, -half_window-1:-half_window]
        
        return filtered
    
    def _goldstein_filter(self, phase: np.ndarray,
                         coherence: Optional[np.ndarray],
                         alpha: float = 1.0
                         ) -> np.ndarray:
        """Goldstein滤波（参考ISCE2实现）
        
        Args:
            phase: 相位数据
            coherence: 相干性数据
            alpha: 滤波参数（0-1，值越大滤波越强）
            
        Returns:
            滤波后的相位
        """
        # 如果没有相干性数据，使用默认值
        if coherence is None:
            coherence = np.ones_like(phase)
        
        # 检查图像大小，对于大型图像使用分块处理
        h, w = phase.shape
        block_size = 2048  # 增大分块大小提高效率
        
        # 如果图像较小，直接处理
        if h <= block_size and w <= block_size:
            # 尝试使用Numba加速版本
            if NUMBA_AVAILABLE:
                try:
                    return numba_goldstein_filter(phase, coherence, alpha)
                except Exception as e:
                    print(f"Numba加速失败: {e}, 回退到向量化版本")
            
            # 使用向量化版本
            try:
                from scipy.signal import convolve2d
                
                # 转换为余弦和正弦分量
                cos_phase = np.cos(phase)
                sin_phase = np.sin(phase)
                
                # 计算权重（基于相干性）
                weights = np.power(coherence, alpha)
                
                # 创建窗口
                window_size = 5
                window = np.ones((window_size, window_size)) / (window_size * window_size)
                
                # 计算权重和
                weights_sum = convolve2d(weights, window, mode='same', boundary='symm')
                
                # 对实部和虚部分别进行卷积
                cos_filtered = convolve2d(cos_phase * weights, window, mode='same', boundary='symm')
                sin_filtered = convolve2d(sin_phase * weights, window, mode='same', boundary='symm')
                
                # 除以权重和
                with np.errstate(divide='ignore', invalid='ignore'):
                    cos_filtered /= weights_sum
                    sin_filtered /= weights_sum
                    cos_filtered = np.where(weights_sum > 0, cos_filtered, 0)
                    sin_filtered = np.where(weights_sum > 0, sin_filtered, 0)
                
                # 重建相位
                filtered_phase = np.arctan2(sin_filtered, cos_filtered)
                
                return filtered_phase.astype(np.float32)
            except Exception:
                # 回退到原始实现
                # 转换为复数表示
                interferogram = np.exp(1j * phase)
                
                # 计算滤波窗口大小（使用相干性自适应窗口）
                window_size = 5
                half_window = window_size // 2
                
                filtered = np.zeros_like(interferogram)
                
                # 填充边界
                padded = np.pad(interferogram, ((half_window, half_window), (half_window, half_window)), mode='edge')
                padded_coh = np.pad(coherence, ((half_window, half_window), (half_window, half_window)), mode='edge')
                
                # 对每个像素进行滤波
                for i in range(h):
                    for j in range(w):
                        # 提取窗口
                        window = padded[i:i+window_size, j:j+window_size]
                        coh_window = padded_coh[i:i+window_size, j:j+window_size]
                        
                        # 计算权重（基于相干性）
                        weights = np.power(coh_window, alpha)
                        weights /= np.sum(weights)
                        
                        # 应用加权平均
                        filtered[i, j] = np.sum(window * weights)
                
                # 提取滤波后的相位
                filtered_phase = np.angle(filtered)
                
                return filtered_phase.astype(np.float32)
        else:
            # 使用分块处理
            print(f"检测到大图像 ({h}x{w})，使用分块处理")
            filtered_phase = np.zeros_like(phase, dtype=np.float32)
            half_window = 2  # 5x5窗口的半窗口大小
            
            # 生成所有块的坐标
            blocks = []
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    # 核心块边界
                    dest_i_start = i
                    dest_j_start = j
                    dest_i_end = min(i + block_size, h)
                    dest_j_end = min(j + block_size, w)

                    # 含 halo 的输入块边界
                    i_start = max(0, dest_i_start - half_window)
                    j_start = max(0, dest_j_start - half_window)
                    i_end = min(h, dest_i_end + half_window)
                    j_end = min(w, dest_j_end + half_window)

                    blocks.append((dest_i_start, dest_i_end, dest_j_start, dest_j_end, i_start, i_end, j_start, j_end))
            
            # 定义处理单个块的函数
            def process_block(block_info):
                dest_i_start, dest_i_end, dest_j_start, dest_j_end, i_start, i_end, j_start, j_end = block_info
                
                phase_block = phase[i_start:i_end, j_start:j_end]
                coherence_block = coherence[i_start:i_end, j_start:j_end]
                
                # 处理块
                if NUMBA_AVAILABLE:
                    try:
                        block_result = numba_goldstein_filter(phase_block, coherence_block, alpha)
                    except Exception as e:
                        print(f"Numba加速失败: {e}, 回退到向量化版本")
                        # 使用向量化版本
                        from scipy.signal import convolve2d
                        cos_phase = np.cos(phase_block)
                        sin_phase = np.sin(phase_block)
                        weights = np.power(coherence_block, alpha)
                        window_size = 5
                        window = np.ones((window_size, window_size)) / (window_size * window_size)
                        weights_sum = convolve2d(weights, window, mode='same', boundary='symm')
                        cos_filtered = convolve2d(cos_phase * weights, window, mode='same', boundary='symm')
                        sin_filtered = convolve2d(sin_phase * weights, window, mode='same', boundary='symm')
                        with np.errstate(divide='ignore', invalid='ignore'):
                            cos_filtered /= weights_sum
                            sin_filtered /= weights_sum
                            cos_filtered = np.where(weights_sum > 0, cos_filtered, 0)
                            sin_filtered = np.where(weights_sum > 0, sin_filtered, 0)
                        block_result = np.arctan2(sin_filtered, cos_filtered)
                else:
                    # 使用向量化版本
                    from scipy.signal import convolve2d
                    cos_phase = np.cos(phase_block)
                    sin_phase = np.sin(phase_block)
                    weights = np.power(coherence_block, alpha)
                    window_size = 5
                    window = np.ones((window_size, window_size)) / (window_size * window_size)
                    weights_sum = convolve2d(weights, window, mode='same', boundary='symm')
                    cos_filtered = convolve2d(cos_phase * weights, window, mode='same', boundary='symm')
                    sin_filtered = convolve2d(sin_phase * weights, window, mode='same', boundary='symm')
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cos_filtered /= weights_sum
                        sin_filtered /= weights_sum
                        cos_filtered = np.where(weights_sum > 0, cos_filtered, 0)
                        sin_filtered = np.where(weights_sum > 0, sin_filtered, 0)
                    block_result = np.arctan2(sin_filtered, cos_filtered)
                
                # 核心块在含 halo 结果中的对应索引
                src_i_start = dest_i_start - i_start
                src_j_start = dest_j_start - j_start
                src_i_end = src_i_start + (dest_i_end - dest_i_start)
                src_j_end = src_j_start + (dest_j_end - dest_j_start)
                
                # 提取结果区域
                result_block = block_result[src_i_start:src_i_end, src_j_start:src_j_end].astype(np.float32)
                
                return (dest_i_start, dest_i_end, dest_j_start, dest_j_end, result_block)
            
            # 使用多线程并行处理
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            max_workers = min(16, len(blocks))  # 增加线程数提高并行度
            print(f"使用 {max_workers} 个线程并行处理")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有块的处理任务
                future_to_block = {executor.submit(process_block, block): block for block in blocks}
                
                # 处理完成的任务
                for future in as_completed(future_to_block):
                    try:
                        dest_i_start, dest_i_end, dest_j_start, dest_j_end, result_block = future.result()
                        # 将结果复制回完整图像
                        filtered_phase[dest_i_start:dest_i_end, dest_j_start:dest_j_end] = result_block
                    except Exception as e:
                        print(f"处理块失败: {e}")
            
            return filtered_phase

    def _median_filter(self, phase: np.ndarray, 
                      window_size: int) -> np.ndarray:
        """中值滤波
        
        Args:
            phase: 相位数据
            window_size: 窗口大小
            
        Returns:
            滤波后的相位
        """
        # 对相位需要特殊处理
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        
        cos_filtered = ndimage.median_filter(cos_phase, window_size)
        sin_filtered = ndimage.median_filter(sin_phase, window_size)
        
        filtered_phase = np.arctan2(sin_filtered, cos_filtered)
        
        return filtered_phase.astype(np.float32)

# Numba加速的Goldstein滤波函数
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True)
    def numba_goldstein_filter(phase, coherence, alpha):
        """Numba加速的Goldstein滤波"""
        h, w = phase.shape
        window_size = 5
        half_window = window_size // 2
        
        # 预计算复数表示
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        
        # 填充边界
        padded_cos = np.zeros((h + 2 * half_window, w + 2 * half_window), dtype=np.float32)
        padded_sin = np.zeros((h + 2 * half_window, w + 2 * half_window), dtype=np.float32)
        padded_coh = np.zeros((h + 2 * half_window, w + 2 * half_window), dtype=np.float32)
        
        # 复制数据到填充区域
        padded_cos[half_window:h+half_window, half_window:w+half_window] = cos_phase
        padded_sin[half_window:h+half_window, half_window:w+half_window] = sin_phase
        padded_coh[half_window:h+half_window, half_window:w+half_window] = coherence
        
        # 填充边界
        for i in prange(half_window):
            # 顶部和底部
            padded_cos[i, :] = padded_cos[half_window, :]
            padded_cos[h+half_window+i, :] = padded_cos[h+half_window-1, :]
            padded_sin[i, :] = padded_sin[half_window, :]
            padded_sin[h+half_window+i, :] = padded_sin[h+half_window-1, :]
            padded_coh[i, :] = padded_coh[half_window, :]
            padded_coh[h+half_window+i, :] = padded_coh[h+half_window-1, :]
            # 左侧和右侧
            padded_cos[:, i] = padded_cos[:, half_window]
            padded_cos[:, w+half_window+i] = padded_cos[:, w+half_window-1]
            padded_sin[:, i] = padded_sin[:, half_window]
            padded_sin[:, w+half_window+i] = padded_sin[:, w+half_window-1]
            padded_coh[:, i] = padded_coh[:, half_window]
            padded_coh[:, w+half_window+i] = padded_coh[:, w+half_window-1]
        
        # 滤波
        filtered_phase = np.zeros((h, w), dtype=np.float32)
        
        # 预计算窗口索引
        window_indices = np.arange(window_size)
        
        for i in prange(h):
            for j in prange(w):
                # 计算窗口范围
                i_start = i
                i_end = i + window_size
                j_start = j
                j_end = j + window_size
                
                # 提取窗口
                cos_win = padded_cos[i_start:i_end, j_start:j_end]
                sin_win = padded_sin[i_start:i_end, j_start:j_end]
                coh_win = padded_coh[i_start:i_end, j_start:j_end]
                
                # 计算权重（基于相干性）
                weights = np.power(coh_win, alpha)
                weight_sum = np.sum(weights)
                
                if weight_sum > 0:
                    # 归一化权重
                    weights /= weight_sum
                    
                    # 计算加权平均
                    weighted_cos = np.sum(cos_win * weights)
                    weighted_sin = np.sum(sin_win * weights)
                    
                    # 计算相位
                    if weighted_cos != 0 or weighted_sin != 0:
                        filtered_phase[i, j] = np.arctan2(weighted_sin, weighted_cos)
                    else:
                        filtered_phase[i, j] = 0
                else:
                    filtered_phase[i, j] = 0
        
        return filtered_phase


class TopographicPhaseRemoval:
    """地形相位移除类"""
    
    def __init__(self, wavelength: float = 0.0555):
        """初始化地形相位移除器
        
        Args:
            wavelength: 雷达波长 (m)
        """
        self.wavelength = wavelength
    
    def remove(self, interferogram: np.ndarray,
              dem: Optional[np.ndarray],
              geometry: Optional[Dict[str, Any]] = None,
              range_spacing: float = 5.0,
              azimuth_spacing: float = 10.0,
              incidence_angle: float = 23.0,
              satellite_height: float = 693000.0
              ) -> np.ndarray:
        """移除地形相位
        
        Args:
            interferogram: 干涉图 (复数)
            dem: DEM数据 (m)
            geometry: 几何网格/参数（来自 geosar/baseline）
            range_spacing: 距离向像素间距 (m)
            azimuth_spacing: 方位向像素间距 (m)
            incidence_angle: 入射角 (度)
            satellite_height: 卫星高度 (m)
            
        Returns:
            移除了地形相位的干涉图
        """
        # 计算地形相位，传递目标形状参数
        topographic_phase = self._calculate_topographic_phase(
            dem, range_spacing, azimuth_spacing,
            incidence_angle, satellite_height, interferogram.shape, geometry=geometry
        )
        
        # 移除地形相位
        interferogram_flat = interferogram * np.exp(-1j * topographic_phase)
        
        return interferogram_flat.astype(np.complex64)
    
    def _calculate_topographic_phase(self, dem: Optional[np.ndarray],
                                    range_spacing: float,
                                    azimuth_spacing: float,
                                    incidence_angle: float,
                                    satellite_height: float,
                                    target_shape: tuple
                                    ,
                                    geometry: Optional[Dict[str, Any]] = None
                                    ) -> np.ndarray:
        """计算地形相位
        
        Args:
            dem: DEM数据
            range_spacing: 距离向像素间距
            azimuth_spacing: 方位向像素间距
            incidence_angle: 入射角
            satellite_height: 卫星高度
            target_shape: 目标形状（与SAR图像匹配）
            geometry: 几何网格/参数（推荐提供 geosar+baseline）
            
        Returns:
            地形相位
        """
        grids = _prepare_phase_model_grids(
            target_shape=target_shape,
            dem=dem,
            range_spacing=range_spacing,
            incidence_angle=incidence_angle,
            geometry=geometry,
        )

        b_perp = grids["b_perp"]
        inc = grids["incidence_angle"]
        slant = grids["slant_range"]
        h_grid = grids["height"]

        if b_perp is None:
            raise ValueError("缺少 b_perp（baseline 垂直基线）。请提供 --baseline-hdf。")
        if slant is None:
            raise ValueError("缺少 slant_range。请提供 geosar 几何 near_range/range_spacing 或 slant_range 网格。")
        if h_grid is None:
            raise ValueError("缺少 DEM 高程网格。请提供 --dem 或 geosar HDF 中的 sar_dem。")

        # 近似 DInSAR 地形相位：
        # phi_topo = (4*pi/lambda) * (B_perp * h) / (R * sin(theta))
        inc_rad = np.deg2rad(inc.astype(np.float64, copy=False))
        den = slant.astype(np.float64, copy=False) * np.sin(inc_rad)
        den = np.where(np.abs(den) < 1e-6, np.nan, den)

        phase = (4.0 * np.pi / float(self.wavelength)) * (
            b_perp.astype(np.float64, copy=False) * h_grid.astype(np.float64, copy=False) / den
        )
        phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

        phase = _wrap_phase(phase)
        print(f"  地形相位范围: [{phase.min():.2f}, {phase.max():.2f}]")
        print(f"  地形相位标准差: {phase.std():.6f}")
        return phase.astype(np.float32, copy=False)


def load_geosar_geometry_hdf(hdf_file: str, target_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    从 geosar 输出 HDF 读取几何信息。
    支持：
    - incidence_angle / incidence_angle_raw
    - sar_dem / sar_dem_raw （作为 height）
    - updated_near_range / original_near_range
    """
    import h5py

    geometry: Dict[str, Any] = {}
    with h5py.File(hdf_file, "r") as f:
        if "incidence_angle" in f:
            geometry["incidence_angle"] = np.asarray(f["incidence_angle"][:], dtype=np.float32)
        elif "incidence_angle_raw" in f:
            geometry["incidence_angle"] = np.asarray(f["incidence_angle_raw"][:], dtype=np.float32)

        if "sar_dem" in f:
            geometry["height"] = np.asarray(f["sar_dem"][:], dtype=np.float32)
        elif "sar_dem_raw" in f:
            geometry["height"] = np.asarray(f["sar_dem_raw"][:], dtype=np.float32)

        if "updated_near_range" in f:
            geometry["near_range"] = float(np.asarray(f["updated_near_range"][()]).item())
        elif "original_near_range" in f:
            geometry["near_range"] = float(np.asarray(f["original_near_range"][()]).item())

    if target_shape is not None:
        for k in ("incidence_angle", "height"):
            if k in geometry:
                geometry[k] = _ensure_grid(geometry[k], target_shape, k, order=1)

    return geometry


def load_baseline_hdf(hdf_file: str, target_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    从 baseline.py 输出 HDF 读取基线信息。
    约定：
    - 优先读取新字段：b_perp_los / b_parallel_los
    - 兼容旧字段：vertical_baseline / horizontal_baseline
    """
    import h5py

    geometry: Dict[str, Any] = {}
    with h5py.File(hdf_file, "r") as f:
        if "b_perp_los" in f:
            geometry["b_perp"] = np.asarray(f["b_perp_los"][:], dtype=np.float32)
        elif "vertical_baseline" in f:
            geometry["b_perp"] = np.asarray(f["vertical_baseline"][:], dtype=np.float32)

        if "b_parallel_los" in f:
            geometry["b_parallel"] = np.asarray(f["b_parallel_los"][:], dtype=np.float32)
        elif "horizontal_baseline" in f:
            geometry["b_parallel"] = np.asarray(f["horizontal_baseline"][:], dtype=np.float32)

    if target_shape is not None:
        for k in ("b_perp", "b_parallel"):
            if k in geometry:
                geometry[k] = _ensure_grid(geometry[k], target_shape, k, order=1)

    return geometry


def _read_range_spacing_from_yaml(yaml_file: str) -> Optional[float]:
    """从 YAML 中读取 range_spacing（含兼容字段）。"""
    with open(yaml_file, "r", encoding="utf-8") as f:
        ycfg = yaml.safe_load(f) or {}
    radar = ycfg.get("radar_parameters", {}) or {}
    val = radar.get("range_spacing", None)
    if val is None:
        val = radar.get("range_pixel_spacing", None)
    if val is None:
        val = ycfg.get("range_spacing", None)
    if val is None:
        val = ycfg.get("range_pixel_spacing", None)
    if val is None:
        return None
    return float(val)


def resolve_range_spacing(master_path: str, explicit_range_spacing: Optional[float], master_yaml: Optional[str]) -> Tuple[float, str]:
    """
    解析实际使用的 range_spacing。
    优先级：
    1) --range-spacing 显式传入
    2) --master-yaml
    3) 依据 master 路径自动推断 *.yaml / master.yaml
    4) 回退默认 5.0
    """
    if explicit_range_spacing is not None:
        return float(explicit_range_spacing), "cli"

    candidates = []
    if master_yaml:
        candidates.append(master_yaml)

    m = Path(master_path)
    if m.suffix.lower() in (".yaml", ".yml"):
        candidates.append(str(m))
    else:
        candidates.append(str(m.with_suffix(".yaml")))
        candidates.append(str(m.with_suffix(".yml")))
        candidates.append(str(m.parent / "master.yaml"))
        candidates.append(str(m.parent / "master.yml"))

    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)

    for c in uniq:
        try:
            if os.path.exists(c):
                v = _read_range_spacing_from_yaml(c)
                if v is not None:
                    return float(v), f"yaml:{c}"
        except Exception:
            continue

    return 5.0, "default"


def load_slc_data(slc_file: str) -> np.ndarray:
    """加载SLC数据
    
    Args:
        slc_file: SLC文件路径
        
    Returns:
        SLC复数数组
    """
    data = np.fromfile(slc_file, dtype=np.complex64)
    
    # 尝试确定数据形状
    size = int(np.sqrt(len(data)))
    if size * size == len(data):
        data = data.reshape(size, size)
    else:
        # 尝试其他形状
        for h in range(int(np.sqrt(len(data))), 0, -1):
            if len(data) % h == 0:
                w = len(data) // h
                data = data.reshape(h, w)
                break
    
    return data


def save_data(data: np.ndarray, output_file: str):
    """保存数据
    
    Args:
        data: 数据数组
        output_file: 输出文件路径
    """
    if data.dtype == np.complex64:
        data.tofile(output_file)
    else:
        data.astype(np.float32).tofile(output_file)


def main():
    """主函数 - 命令行工具"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='干涉处理模块命令行工具')
    parser.add_argument('master', help='主图像文件路径（VRT或TIFF格式）')
    parser.add_argument('slave', help='辅图像文件路径（VRT或TIFF格式，重采样后）')
    parser.add_argument('--output-dir', default='.', help='输出目录')
    parser.add_argument('--window-size', type=int, default=5, help='相干性计算窗口大小')
    parser.add_argument('--filter-method', choices=['gaussian', 'adaptive', 'median', 'goldstein'], default='goldstein', help='滤波方法')
    parser.add_argument('--sigma', type=float, default=1.0, help='高斯滤波sigma值')
    parser.add_argument('--alpha', type=float, default=0.5, help='Goldstein滤波参数（0-1，值越大滤波越强）')
    parser.add_argument('--remove-topographic', action='store_true', help='去除地形效应')
    parser.add_argument('--remove-flat', action='store_true', help='去除平地效应')
    parser.add_argument('--dem', help='DEM文件路径，用于去除地形效应')
    parser.add_argument('--master-yaml', default=None, help='主影像 YAML 路径（用于自动读取 range_spacing）')
    parser.add_argument('--geosar-hdf', help='geosar.py 输出的 HDF 文件（读取 incidence_angle/sar_dem/near_range）')
    parser.add_argument('--baseline-hdf', help='utils/baseline.py 输出的 baseline.hdf（读取垂直/水平基线）')
    parser.add_argument('--wavelength', type=float, default=0.0555, help='雷达波长 (m)')
    parser.add_argument('--range-spacing', type=float, default=None, help='距离向像素间距 (m)，不传则自动从 master.yaml 读取')
    parser.add_argument('--near-range', type=float, default=None, help='近距斜距 near_range (m)，可覆盖 geosar_hdf 内值')
    parser.add_argument('--azimuth-spacing', type=float, default=10.0, help='方位向像素间距 (m)')
    parser.add_argument('--incidence-angle', type=float, default=23.0, help='入射角 (度)')
    parser.add_argument('--satellite-height', type=float, default=693000.0, help='卫星高度 (m)')
    parser.add_argument('--unwrap', action='store_true', help='调用 snaphu 对滤波相位进行解缠，并输出 LOS 形变')
    parser.add_argument('--snaphu-bin', default='snaphu', help='snaphu 可执行文件路径/名称')
    parser.add_argument('--snaphu-cost-mode', choices=['topo', 'defo', 'smooth'], default='defo', help='snaphu 代价模式')
    parser.add_argument('--snaphu-init-method', choices=['mst', 'mcf'], default='mst', help='snaphu 初始化方法')
    parser.add_argument('--snaphu-corr-thresh', type=float, default=None, help='snaphu 相干性阈值（低于阈值置零）')
    parser.add_argument('--snaphu-tile-rows', type=int, default=None, help='snaphu tile 行数（启用分块解缠）')
    parser.add_argument('--snaphu-tile-cols', type=int, default=None, help='snaphu tile 列数（启用分块解缠）')
    parser.add_argument('--snaphu-tile-row-overlap', type=int, default=512, help='snaphu tile 行重叠')
    parser.add_argument('--snaphu-tile-col-overlap', type=int, default=512, help='snaphu tile 列重叠')
    parser.add_argument('--snaphu-nproc', type=int, default=None, help='snaphu tile 模式并行进程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    import os
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧版遗留的相位叠加强度 PNG，避免与纯相位结果混淆
    removed_overlay = 0
    for pat in ("*_overlay_master.png", "*_overlay_amplitude.png"):
        for p in sorted(output_dir.glob(pat)):
            if not p.is_file():
                continue
            try:
                p.unlink()
                removed_overlay += 1
            except Exception as e:
                print(f"警告: 清理旧 overlay PNG 失败: {p} ({e})")
    if removed_overlay > 0:
        print(f"已清理旧 overlay PNG: {removed_overlay} 个")
    
    print("=== 干涉处理流程 ===")
    print(f"主图像: {args.master}")
    print(f"辅图像: {args.slave}")
    print(f"输出目录: {output_dir}")
    print(f"滤波方法: {args.filter_method}")
    if args.filter_method == 'goldstein':
        print(f"Goldstein参数: {args.alpha}")

    range_spacing, range_spacing_src = resolve_range_spacing(args.master, args.range_spacing, args.master_yaml)
    print(f"range_spacing: {range_spacing} m (source={range_spacing_src})")
    print()
    
    # 读取图像
    def read_image(file_path):
        """读取图像文件"""
        try:
            from osgeo import gdal
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
    
    # 保存PNG文件
    def save_png(data, output_file):
        """保存数据为PNG文件"""
        try:
            # 确保数据类型正确
            if data.dtype == np.complex64:
                # 保存幅度
                data = np.abs(data)
                print(f"  复数数据转换为幅度: 形状={data.shape}, 类型={data.dtype}")
            
            # 打印数据统计信息
            min_val = data.min()
            max_val = data.max()
            mean_val = data.mean()
            print(f"  数据统计: 最小值={min_val:.6f}, 最大值={max_val:.6f}, 平均值={mean_val:.6f}")
            
            # 归一化到 0-255 范围
            if max_val > min_val:
                # 检查是否是幅度图像（通常文件名包含amplitude）
                if 'amplitude' in output_file.lower():
                    print("  处理幅度图像: 应用log变换和95%拉伸")
                    
                    # 避免log(0)，添加一个小的epsilon
                    epsilon = 1e-10
                    data_log = np.log(data + epsilon)
                    
                    # 计算95%分位数（对于大型图像，使用更高效的方法）
                    h, w = data_log.shape
                    if h * w > 1000000:  # 超过100万像素
                        # 随机采样100万个点进行分位数计算
                        sample_size = min(1000000, h * w)
                        sample_indices = np.random.choice(h * w, sample_size, replace=False)
                        data_sample = data_log.reshape(-1)[sample_indices]
                        p5 = np.percentile(data_sample, 5)
                        p95 = np.percentile(data_sample, 95)
                        print(f"  使用随机采样计算分位数（{sample_size}个点）")
                    else:
                        # 对于小图像，直接计算
                        p5 = np.percentile(data_log, 5)
                        p95 = np.percentile(data_log, 95)
                    
                    print(f"  log变换后: 最小值={data_log.min():.6f}, 最大值={data_log.max():.6f}")
                    print(f"  95%拉伸范围: {p5:.6f} - {p95:.6f}")
                    
                    # 应用95%拉伸
                    if p95 > p5:
                        data_stretched = (data_log - p5) / (p95 - p5)
                        # 裁剪到[0, 1]范围
                        data_stretched = np.clip(data_stretched, 0, 1)
                        data_normalized = (data_stretched * 255).astype(np.uint8)
                    else:
                        # 如果范围太小，使用线性拉伸
                        data_normalized = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    # 普通图像使用线性拉伸
                    data_normalized = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                
                print(f"  归一化后: 最小值={data_normalized.min()}, 最大值={data_normalized.max()}")
            else:
                data_normalized = np.zeros_like(data, dtype=np.uint8)
                print(f"  数据范围相同，生成全黑图像")
            
            # 创建PNG文件
            import matplotlib.pyplot as plt
            height, width = data.shape
            
            # 对于单通道图像
            if len(data_normalized.shape) == 2:
                plt.imsave(output_file, data_normalized, cmap='gray')
            else:
                plt.imsave(output_file, data_normalized)
            
            print(f"已保存PNG文件: {output_file}")
        except Exception as e:
            print(f"保存PNG失败: {e}")
    
    def save_phase_png(phase, output_file, coherence=None, amplitude=None):
        """保存纯相位彩色 PNG（使用 gamma 色调，不叠加强度）。"""
        try:
            # 计算相位值范围
            min_phase = phase.min()
            max_phase = phase.max()
            mean_phase = phase.mean()
            std_phase = phase.std()
            print(f"  相位范围: [{min_phase:.2f}, {max_phase:.2f}]")
            print(f"  相位平均值: {mean_phase:.2f}")
            print(f"  相位标准差: {std_phase:.6f}")
            
            # 调整归一化方法，确保相位值范围合理
            if max_phase > min_phase:
                # 检查相位值范围是否过小
                phase_range = max_phase - min_phase
                if phase_range < 0.1:
                    # 相位值几乎是常数，使用固定范围进行归一化
                    print("  警告: 相位值范围过小，使用固定范围进行归一化")
                    phase_normalized = (phase - mean_phase + np.pi) / (2 * np.pi)
                    phase_normalized = np.clip(phase_normalized, 0, 1)
                else:
                    # 正常归一化
                    phase_normalized = (phase - min_phase) / (max_phase - min_phase)
            else:
                # 所有相位值相同，使用中间值
                print("  警告: 所有相位值相同，使用默认颜色")
                phase_normalized = np.ones_like(phase) * 0.5
            
            # 创建RGB彩色映射（使用gamma色调）
            height, width = phase.shape
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 使用gamma色调（参考gamma软件的相位色调）
            # 相位循环映射到gamma色调：红->绿->蓝->红
            p = phase_normalized.reshape(-1)
            r = np.zeros_like(p, dtype=np.uint8)
            g = np.zeros_like(p, dtype=np.uint8)
            b = np.zeros_like(p, dtype=np.uint8)
            
            # gamma色调映射
            # 红色到绿色
            mask = p < 1/3
            r[mask] = 255 - (p[mask] * 3 * 255).astype(np.uint8)
            g[mask] = (p[mask] * 3 * 255).astype(np.uint8)
            b[mask] = 0
            
            # 绿色到蓝色
            mask = (p >= 1/3) & (p < 2/3)
            r[mask] = 0
            g[mask] = 255 - ((p[mask] - 1/3) * 3 * 255).astype(np.uint8)
            b[mask] = ((p[mask] - 1/3) * 3 * 255).astype(np.uint8)
            
            # 蓝色到红色
            mask = p >= 2/3
            r[mask] = ((p[mask] - 2/3) * 3 * 255).astype(np.uint8)
            g[mask] = 0
            b[mask] = 255 - ((p[mask] - 2/3) * 3 * 255).astype(np.uint8)
            
            # 重塑回原始形状
            rgb[:, :, 0] = r.reshape(height, width)
            rgb[:, :, 1] = g.reshape(height, width)
            rgb[:, :, 2] = b.reshape(height, width)
            
            # 如果提供了相干性，将相干性小于0.3的位置设置为空值
            if coherence is not None:
                print("  设置低相干性区域为空值")
                mask = (coherence < 0.3)
                rgb[mask] = 0
            
            # 保持纯相位显示：即使传入 amplitude 也不做叠加
            if amplitude is not None:
                print("  忽略 amplitude 输入：当前仅输出纯相位图")

            # 若叠加后仍需低相干掩膜，这里再次应用，避免被覆盖
            if coherence is not None:
                mask = (coherence < 0.3)
                rgb[mask] = 0
            
            # 创建PNG文件
            import matplotlib.pyplot as plt
            plt.imsave(output_file, rgb)
            print(f"已保存彩色相位PNG文件: {output_file}")
        except Exception as e:
            print(f"保存彩色相位PNG失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 1. 读取图像
    print("1. 读取图像...")
    master = read_image(args.master)
    slave = read_image(args.slave)
    
    if master is None or slave is None:
        print("错误：无法读取图像文件")
        return 1

    if master.shape != slave.shape:
        print(f"错误：主辅图尺寸不一致: master={master.shape}, slave={slave.shape}")
        return 1
    
    print(f"  主图像大小: {master.shape}, 类型: {master.dtype}")
    print(f"  辅图像大小: {slave.shape}, 类型: {slave.dtype}")
    
    # 读取DEM文件（如果提供）
    dem = None
    if args.dem:
        print("\n读取DEM文件...")
        try:
            from osgeo import gdal
            ds = gdal.Open(args.dem)
            if ds:
                band = ds.GetRasterBand(1)
                dem = band.ReadAsArray()
                print(f"  DEM文件读取成功，形状: {dem.shape}")
                ds = None
            else:
                print(f"  无法打开DEM文件: {args.dem}")
        except Exception as e:
            print(f"  读取DEM文件失败: {e}")

    # 读取 geosar/baseline 几何参数
    geometry: Dict[str, Any] = {}
    if args.geosar_hdf:
        print("\n读取 geosar 几何信息...")
        try:
            geo_dict = load_geosar_geometry_hdf(args.geosar_hdf, target_shape=master.shape)
            geometry.update(geo_dict)
            print(f"  geosar 几何读取成功: keys={list(geo_dict.keys())}")
        except Exception as e:
            print(f"  读取 geosar HDF 失败: {e}")

    if args.baseline_hdf:
        print("\n读取 baseline 信息...")
        try:
            base_dict = load_baseline_hdf(args.baseline_hdf, target_shape=master.shape)
            geometry.update(base_dict)
            print(f"  baseline 读取成功: keys={list(base_dict.keys())}")
        except Exception as e:
            print(f"  读取 baseline HDF 失败: {e}")

    # 命令行参数优先覆盖
    geometry["range_spacing"] = float(range_spacing)
    if args.near_range is not None:
        geometry["near_range"] = float(args.near_range)

    # 若未显式提供 DEM，但 geosar HDF 提供了 SAR 网格高程，可直接用于地形相位
    if (dem is None) and ("height" in geometry):
        dem = np.asarray(geometry["height"], dtype=np.float32)
        print("  使用 geosar HDF 中的 sar_dem 作为地形高程网格。")
    
    # 2. 生成干涉图
    print("\n2. 生成干涉图...")
    generator = InterferogramGenerator()
    result = generator.generate(
        master, slave,
        subtract_flat=args.remove_flat,
        dem=dem,
        geometry=geometry,
        wavelength=args.wavelength,
        range_spacing=range_spacing,
        azimuth_spacing=args.azimuth_spacing,
        incidence_angle=args.incidence_angle
    )
    
    print(f"  干涉图生成完成")
    print(f"  相位范围: [{result['phase'].min():.2f}, {result['phase'].max():.2f}]")
    
    # 3. 计算相干性
    print("\n3. 计算相干性...")
    calculator = CoherenceCalculator(window_size=args.window_size)
    coherence = calculator.calculate_efficient(master, slave)
    
    print(f"  相干性计算完成")
    print(f"  相干性范围: [{coherence.min():.4f}, {coherence.max():.4f}]")
    print(f"  平均相干性: {coherence.mean():.4f}")
    
    # 4. 去除地形效应（如果请求）
    if args.remove_topographic and ((dem is not None) or ("height" in geometry)):
        print("\n4. 去除地形效应...")
        try:
            # 输出地形相位移除前的相位统计
            print(f"  去除前相位范围: [{result['phase'].min():.2f}, {result['phase'].max():.2f}]")
            print(f"  去除前相位平均值: {result['phase'].mean():.2f}")
            print(f"  去除前相位标准差: {result['phase'].std():.6f}")
            
            topo_remover = TopographicPhaseRemoval(wavelength=args.wavelength)
            interferogram_flat = topo_remover.remove(
                result['interferogram'],
                dem,
                geometry=geometry,
                range_spacing=range_spacing,
                azimuth_spacing=args.azimuth_spacing,
                incidence_angle=args.incidence_angle,
                satellite_height=args.satellite_height
            )
            # 更新相位
            result['phase_flat'] = np.angle(interferogram_flat)
            result['interferogram_flat'] = interferogram_flat
            
            # 输出地形相位移除后的相位统计
            print(f"  地形效应去除完成")
            print(f"  去除地形效应后相位范围: [{result['phase_flat'].min():.2f}, {result['phase_flat'].max():.2f}]")
            print(f"  去除地形效应后相位平均值: {result['phase_flat'].mean():.2f}")
            print(f"  去除地形效应后相位标准差: {result['phase_flat'].std():.6f}")
            
            # 检查相位值是否接近常数
            if result['phase_flat'].std() < 0.01:
                print("  警告: 去除地形效应后相位值几乎是常数，可能导致图像显示异常")
        except Exception as e:
            print(f"  去除地形效应失败: {e}")
            import traceback
            traceback.print_exc()
    elif args.remove_topographic:
        print("\n4. 去除地形效应...")
        print("  跳过：缺少高程网格（--dem 或 geosar HDF 中的 sar_dem）。")
    
    # 5. 滤波干涉图
    print("\n5. 滤波干涉图...")
    filter_obj = InterferogramFilter(method=args.filter_method)
    
    # 选择要滤波的相位
    phase_to_filter = result.get('phase_flat', result['phase'])
    filtered_phase = filter_obj.filter(phase_to_filter, coherence, sigma=args.sigma, alpha=args.alpha)
    
    print(f"  滤波完成")
    print(f"  滤波后相位范围: [{filtered_phase.min():.2f}, {filtered_phase.max():.2f}]")
    
    # 6. 输出结果
    print("\n6. 输出结果...")
    
    # 构建输出文件名
    master_name = Path(args.master).stem
    slave_name = Path(args.slave).stem
    
    # 保存干涉数据
    interferogram_file = output_dir / f"{master_name}_{slave_name}_interferogram.bin"
    save_data(result['interferogram'], str(interferogram_file))
    print(f"  干涉数据已保存: {interferogram_file}")
    
    # 保存去除地形效应后的干涉数据（如果有）
    if 'interferogram_flat' in result:
        interferogram_flat_file = output_dir / f"{master_name}_{slave_name}_interferogram_flat.bin"
        save_data(result['interferogram_flat'], str(interferogram_flat_file))
        print(f"  去除地形效应后的干涉数据已保存: {interferogram_flat_file}")
    
    # 保存相干数据
    coherence_file = output_dir / f"{master_name}_{slave_name}_coherence.bin"
    save_data(coherence, str(coherence_file))
    print(f"  相干数据已保存: {coherence_file}")
    
    # 保存滤波后的相位
    filtered_phase_file = output_dir / f"{master_name}_{slave_name}_filtered_phase.bin"
    save_data(filtered_phase, str(filtered_phase_file))
    print(f"  滤波后相位已保存: {filtered_phase_file}")
    
    # 生成VRT文件
    def create_vrt(output_file, data_shape, data_type):
        """创建VRT文件"""
        try:
            vrt_file = output_file.replace('.bin', '.vrt')
            width, height = data_shape[1], data_shape[0]
            
            # 确定数据类型
            gdal_type = 'Float32'
            if data_type == np.complex64:
                bands = 2
                pixel_size = 8  # 复数占8字节
                line_offset = width * 8
            else:
                bands = 1
                pixel_size = 4  # 实数占4字节
                line_offset = width * 4
            
            # 生成VRT内容
            vrt_content = f'''
<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
'''
            
            for i in range(bands):
                band_num = i + 1
                vrt_content += f'''
  <VRTRasterBand dataType="{gdal_type}" band="{band_num}" subClass="VRTRawRasterBand">
    <SourceFilename relativeToVRT="1">{os.path.basename(output_file)}</SourceFilename>
    <ImageOffset>{i * 4}</ImageOffset>
    <PixelOffset>{pixel_size}</PixelOffset>
    <LineOffset>{line_offset}</LineOffset>
    <ByteOrder>LSB</ByteOrder>
  </VRTRasterBand>
'''
            
            vrt_content += '''
</VRTDataset>
'''
            
            with open(vrt_file, 'w') as f:
                f.write(vrt_content)
            print(f"  已创建VRT文件: {vrt_file}")
        except Exception as e:
            print(f"创建VRT文件失败: {e}")
    
    # 为二进制文件创建VRT文件
    create_vrt(str(interferogram_file), result['interferogram'].shape, result['interferogram'].dtype)
    if 'interferogram_flat' in result:
        create_vrt(str(interferogram_flat_file), result['interferogram_flat'].shape, result['interferogram_flat'].dtype)
    create_vrt(str(coherence_file), coherence.shape, coherence.dtype)
    create_vrt(str(filtered_phase_file), filtered_phase.shape, filtered_phase.dtype)
    
    # 生成PNG图像
    # 干涉条纹图（相位）
    phase_png = output_dir / f"{master_name}_{slave_name}_phase.png"
    save_phase_png(result['phase'], str(phase_png), coherence, None)

    # 去除地形效应后的干涉条纹图（如果有）
    if 'phase_flat' in result:
        phase_flat_png = output_dir / f"{master_name}_{slave_name}_phase_flat.png"
        save_phase_png(result['phase_flat'], str(phase_flat_png), coherence, None)

    # 滤波后的干涉条纹图
    filtered_phase_png = output_dir / f"{master_name}_{slave_name}_filtered_phase.png"
    save_phase_png(filtered_phase, str(filtered_phase_png), coherence, None)
    
    # 相干性图
    coherence_png = output_dir / f"{master_name}_{slave_name}_coherence.png"
    save_png(coherence, str(coherence_png))
    
    # 幅度图
    amplitude_png = output_dir / f"{master_name}_{slave_name}_amplitude.png"
    save_png(result['amplitude'], str(amplitude_png))

    # 7. snaphu 解缠和 LOS 形变（可选）
    if args.unwrap:
        print("\n7. snaphu 解缠与 LOS 形变...")
        try:
            try:
                from interferometry.snaphu import run_unwrap_and_los
            except Exception:
                from snaphu import run_unwrap_and_los

            unwrap_result = run_unwrap_and_los(
                wrapped_phase=filtered_phase,
                coherence=coherence,
                output_dir=output_dir,
                prefix=f"{master_name}_{slave_name}",
                wavelength=args.wavelength,
                snaphu_bin=args.snaphu_bin,
                cost_mode=args.snaphu_cost_mode,
                init_method=args.snaphu_init_method,
                corr_thresh=args.snaphu_corr_thresh,
                tile_rows=args.snaphu_tile_rows,
                tile_cols=args.snaphu_tile_cols,
                tile_row_overlap=args.snaphu_tile_row_overlap,
                tile_col_overlap=args.snaphu_tile_col_overlap,
                nproc=args.snaphu_nproc,
            )

            unwrapped_phase = unwrap_result["unwrapped_phase"]
            los_m = unwrap_result["los_deformation_m"]
            los_mm = unwrap_result["los_deformation_mm"]

            print("  snaphu 解缠完成")
            print(f"  unwrapped phase: {unwrap_result['unwrapped_phase_file']}")
            print(f"  LOS deformation (m): {unwrap_result['los_deformation_m_file']}")
            print(f"  LOS deformation (mm): {unwrap_result['los_deformation_mm_file']}")
            if "snaphu_log_file" in unwrap_result:
                print(f"  snaphu log: {unwrap_result['snaphu_log_file']}")
            if "snaphu_version" in unwrap_result:
                print(f"  snaphu version: {unwrap_result['snaphu_version']}")
            if "snaphu_warnings" in unwrap_result:
                for w in unwrap_result["snaphu_warnings"]:
                    print(f"  snaphu warning: {w}")
            print(f"  解缠相位范围: [{unwrapped_phase.min():.2f}, {unwrapped_phase.max():.2f}]")
            print(f"  LOS(m) 范围: [{los_m.min():.6f}, {los_m.max():.6f}]")

            # 生成可视化
            unwrapped_phase_png = output_dir / f"{master_name}_{slave_name}_unwrapped_phase.png"
            save_phase_png(unwrapped_phase, str(unwrapped_phase_png), coherence, None)

            los_m_png = output_dir / f"{master_name}_{slave_name}_los_deformation_m.png"
            save_png(los_m, str(los_m_png))
        except Exception as e:
            print(f"  snaphu 解缠失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n=== 干涉处理完成 ===")
    print(f"所有结果已输出到: {output_dir}")
    
    return 0


if __name__ == '__main__':
    main()
