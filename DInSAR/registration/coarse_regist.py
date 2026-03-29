#!/usr/bin/env python3
"""
粗配准模块 - 基于FFT
窗口2048，获得总体偏移
"""

import numpy as np
import os
import sys
from pathlib import Path
from scipy import fftpack

# 添加路径以导入 utils 模块
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from sar_utils import read_image

# 修复GDAL警告
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    pass


def fft_coarse_registration(master_data, slave_data, window_size=256, search_range=64, window_pos=None):
    """FFT粗配准 - 与xcorr3保持一致
    
    参数:
        master_data: 主图像
        slave_data: 辅图像
        window_size: 数据窗口大小 (默认256，对应搜索窗口64)
        search_range: 搜索范围 (默认64)
        window_pos: 窗口位置 (row, col)，如果为None则使用中心位置
        
    返回:
        (azimuth_offset, range_offset, correlation)
    """
    h, w = master_data.shape
    
    # 窗口位置
    if window_pos is None:
        r0 = h // 2
        c0 = w // 2
    else:
        r0, c0 = window_pos
    
    # 相关窗口大小 = 2 * 搜索范围
    nx_corr = 2 * search_range
    ny_corr = 2 * search_range
    
    # 确保数据窗口大小至少是搜索范围的4倍
    if window_size < 4 * search_range:
        print(f"警告: 窗口大小 ({window_size}) 对于搜索范围 ({search_range}) 太小。设置为 4 * 搜索范围。")
        nx_win = 4 * search_range
        ny_win = 4 * search_range
    else:
        nx_win = window_size
        ny_win = window_size
    
    # 提取窗口
    astretcha = 0.0  # 方位向拉伸因子，默认为0
    x_offset = 0.0   # 初始x偏移
    y_offset = 0.0   # 初始y偏移
    
    # 主图像窗口位置
    r_start_master = max(0, r0 - ny_win // 2)
    r_end_master = min(h, r0 + ny_win // 2)
    c_start_master = max(0, c0 - nx_win // 2)
    c_end_master = min(w, c0 + nx_win // 2)
    
    # 辅图像窗口位置
    slave_loc_y = (1 + astretcha) * r0 + y_offset
    slave_loc_x = (1 + astretcha) * c0 + x_offset
    
    r_start_slave = max(0, int(slave_loc_y - ny_win // 2))
    r_end_slave = min(h, int(slave_loc_y + ny_win // 2))
    c_start_slave = max(0, int(slave_loc_x - nx_win // 2))
    c_end_slave = min(w, int(slave_loc_x + nx_win // 2))
    
    master_win = master_data[r_start_master:r_end_master, c_start_master:c_end_master]
    slave_win = slave_data[r_start_slave:r_end_slave, c_start_slave:c_end_slave]
    
    if master_win.shape[0] < ny_win or master_win.shape[1] < nx_win:
        print(f"警告: 窗口太小 {master_win.shape}")
        return 0.0, 0.0, 0.0
    
    # 计算幅度
    master_amp = np.abs(master_win).astype(np.float64)
    slave_amp = np.abs(slave_win).astype(np.float64)
    
    # 均值减法
    master_amp -= master_amp.mean()
    slave_amp -= slave_amp.mean()
    
    # 应用掩码（向量化）
    slave_amp[:search_range, :] = 0
    slave_amp[ny_win - search_range:, :] = 0
    slave_amp[:, :search_range] = 0
    slave_amp[:, nx_win - search_range:] = 0
    
    # FFT相关
    master_fft = fftpack.fft2(master_amp)
    slave_fft = fftpack.fft2(slave_amp)
    
    # 计算交叉功率谱
    cross_power = master_fft * np.conj(slave_fft)
    
    # 应用相位因子（只对前 nx_win/2+1 个频率分量，向量化）
    cols = nx_win // 2 + 1
    sign = (1.0 - 2.0 * (np.arange(ny_win * cols) % 2)).reshape(ny_win, cols)
    cross_power[:, :cols] *= sign
    
    # 逆FFT得到相关结果
    correlation = np.abs(np.real(fftpack.ifft2(cross_power))) / (nx_win * ny_win)
    
    # 搜索区域
    corr_search = correlation[search_range:search_range+ny_corr, search_range:search_range+nx_corr]
    
    if corr_search.size == 0:
        print("警告: 搜索范围太大")
        search_range = min(search_range, ny_win // 4)
        corr_search = correlation[search_range:search_range+2*search_range, search_range:search_range+2*search_range]
    
    # 找最大值
    max_idx = np.unravel_index(np.argmax(corr_search), corr_search.shape)
    cmax = float(corr_search[max_idx])
    
    # 计算偏移量
    offset_az = max_idx[0] - search_range  # 方位向偏移
    offset_rg = max_idx[1] - search_range  # 距离向偏移
    
    # 时间域相关计算
    def time_corr(c1, c2, xoff, yoff):
        y1_start = search_range + max(0, yoff)
        y1_end = search_range + ny_corr + min(0, yoff)
        x1_start = search_range + max(0, xoff)
        x1_end = search_range + nx_corr + min(0, xoff)

        if y1_end <= y1_start or x1_end <= x1_start:
            return 0.0

        y2_start = search_range + max(0, -yoff)
        y2_end = y2_start + (y1_end - y1_start)
        x2_start = search_range + max(0, -xoff)
        x2_end = x2_start + (x1_end - x1_start)

        a = c1[y1_start:y1_end, x1_start:x1_end]
        b = c2[y2_start:y2_end, x2_start:x2_end]
        if a.size == 0 or b.size == 0:
            return 0.0

        num = float(np.sum(a * b))
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        return 100.0 * np.abs(num / denom) if denom > 0 else 0.0
    
    # 计算时间域相关系数
    max_corr = time_corr(master_amp, slave_amp, int(offset_rg), int(offset_az))
    
    # 亚像素拟合
    interp_factor = 16  # 插值因子
    n2x = 8  # 相关窗口大小
    n2y = 8
    range_interp = 1  # 距离向插值因子
    
    if interp_factor > 1:
        # 对相关结果进行缩放
        if cmax > 0:
            corr_search = corr_search * (max_corr / cmax)
        
        # 确保峰值位置在有效范围内
        if offset_az + search_range < n2y/2:
            offset_az = n2y/2 - search_range
        elif offset_az + search_range >= ny_corr - n2y/2:
            offset_az = ny_corr - n2y/2 - search_range - 1
        
        if offset_rg + search_range < n2x/2:
            offset_rg = n2x/2 - search_range
        elif offset_rg + search_range >= nx_corr - n2x/2:
            offset_rg = nx_corr - n2x/2 - search_range - 1
        
        # 提取相关窗口
        y_start = int(offset_az + search_range - n2y/2)
        y_end = y_start + n2y
        x_start = int(offset_rg + search_range - n2x/2)
        x_end = x_start + n2x
        
        if 0 <= y_start < corr_search.shape[0] and 0 <= x_start < corr_search.shape[1]:
            corr2 = corr_search[y_start:y_end, x_start:x_end]
            
            # 应用幂次变换
            corr2 = np.power(corr2, 0.25)
            
            # 使用FFT插值进行亚像素估计
            if corr2.shape == (8, 8):
                # 扩展数组到interp_factor倍大小
                ny_hi = n2y * interp_factor
                nx_hi = n2x * interp_factor
                
                # FFT插值
                corr_fft = fftpack.fft2(corr2)
                fft_padded = np.zeros((ny_hi, nx_hi), dtype=np.complex128)
                fft_padded[:n2y//2, :n2x//2] = corr_fft[:n2y//2, :n2x//2]
                fft_padded[:n2y//2, -n2x//2:] = corr_fft[:n2y//2, -n2x//2:]
                fft_padded[-n2y//2:, :n2x//2] = corr_fft[-n2y//2:, :n2x//2]
                fft_padded[-n2y//2:, -n2x//2:] = corr_fft[-n2y//2:, -n2x//2:]
                
                hi_corr = np.abs(fftpack.ifft2(fft_padded))
                
                # 找到高分辨率相关图中的峰值
                max_idx = np.unravel_index(np.argmax(hi_corr), hi_corr.shape)
                ypeak2 = max_idx[0] - ny_hi // 2
                xpeak2 = max_idx[1] - nx_hi // 2
                
                # 计算亚像素偏移
                offset_az += ypeak2 / interp_factor
                offset_rg += xpeak2 / interp_factor
    
    # 最终偏移量计算
    final_offset_az = y_offset - offset_az + r0 * astretcha
    final_offset_rg = x_offset - (offset_rg / range_interp)
    
    return float(final_offset_az), float(final_offset_rg), max_corr


class CoarseRegistration:
    """粗配准类"""
    
    def __init__(self, window_size=256, search_range=64, correlation_threshold=30):
        self.window_size = window_size
        self.search_range = search_range
        self.correlation_threshold = correlation_threshold
    
    def register(self, master, slave, initial_guess=(0.0, 0.0), output_file=None):
        h, w = master.shape
        quarter_h = h // 4
        quarter_w = w // 4
        
        # 2*2窗口位置
        window_positions = [
            (quarter_h, quarter_w),      # 左上
            (quarter_h, 3 * quarter_w),  # 右上
            (3 * quarter_h, quarter_w),  # 左下
            (3 * quarter_h, 3 * quarter_w)  # 右下
        ]
        
        offsets = []
        correlations = []
        
        # 使用线程池并行处理窗口
        from concurrent.futures import ThreadPoolExecutor
        
        def process_window(window_pos):
            row, col = window_pos
            offset_az, offset_rg, corr = fft_coarse_registration(
                master, slave, 
                window_size=self.window_size, 
                search_range=self.search_range,
                window_pos=(row, col)
            )
            
            total_az = offset_az + initial_guess[0]
            total_rg = offset_rg + initial_guess[1]
            
            return (total_az, total_rg), corr
        
        # 使用4个线程并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_window, window_positions))
        
        # 收集结果
        for (offset, corr) in results:
            offsets.append(offset)
            correlations.append(corr)
        
        # 计算平均值
        if not offsets:
            return {
                'method': 'fft_coarse_2x2',
                'azimuth_offset': initial_guess[0],
                'range_offset': initial_guess[1],
                'window_offsets': [],
                'correlations': [],
                'average_correlation': 0.0,
                'window_size': self.window_size,
                'window_positions': window_positions
            }
        
        # 线性分析和异常值检测
        # 收集数据点
        points = []
        for (row, col), (az, rg), corr in zip(window_positions, offsets, correlations):
            # 只考虑相关系数大于阈值的点
            if corr >= self.correlation_threshold:
                points.append((col, row, rg, az, corr))  # (x, y, dx, dy, corr)
        
        # 如果有效点太少，使用所有点
        if len(points) < 3:
            filtered_offsets = offsets
            filtered_correlations = correlations
            filtered_positions = window_positions
        else:
            # 提取坐标和偏移量
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            dx = np.array([p[2] for p in points])
            dy = np.array([p[3] for p in points])
            corr = np.array([p[4] for p in points])
            
            # 对距离向偏移进行线性拟合
            A = np.vstack([x, y, np.ones(len(x))]).T
            coeffs_dx, _, _, _ = np.linalg.lstsq(A, dx, rcond=None)
            
            # 对方位向偏移进行线性拟合
            coeffs_dy, _, _, _ = np.linalg.lstsq(A, dy, rcond=None)
            
            # 计算预测值和残差
            dx_pred = coeffs_dx[0] * x + coeffs_dx[1] * y + coeffs_dx[2]
            dy_pred = coeffs_dy[0] * x + coeffs_dy[1] * y + coeffs_dy[2]
            
            # 计算残差的标准差
            dx_residuals = np.abs(dx - dx_pred)
            dy_residuals = np.abs(dy - dy_pred)
            
            # 使用3倍标准差作为异常值阈值
            dx_threshold = 3 * np.std(dx_residuals)
            dy_threshold = 3 * np.std(dy_residuals)
            
            # 过滤异常值
            filtered_points = []
            for i, (px, py, pdx, pdy, pcorr) in enumerate(points):
                if dx_residuals[i] < dx_threshold and dy_residuals[i] < dy_threshold:
                    filtered_points.append((px, py, pdx, pdy, pcorr))
            
            # 如果过滤后点太少，使用所有点
            if len(filtered_points) < 3:
                filtered_offsets = offsets
                filtered_correlations = correlations
                filtered_positions = window_positions
            else:
                # 提取过滤后的数据
                filtered_offsets = [(p[3], p[2]) for p in filtered_points]  # (az, rg)
                filtered_correlations = [p[4] for p in filtered_points]
                filtered_positions = [(p[1], p[0]) for p in filtered_points]  # (row, col)
        
        # 计算过滤后的平均值
        avg_az = np.mean([o[0] for o in filtered_offsets])
        avg_rg = np.mean([o[1] for o in filtered_offsets])
        avg_corr = np.mean(filtered_correlations)
        
        # 输出结果到 offset_estimate.txt（覆盖模式）
        if output_file is None:
            output_dir = os.getcwd()
            output_file = os.path.join(output_dir, 'offset_estimate.txt')
        else:
            output_dir = os.path.dirname(os.path.abspath(output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(f'WINDOW_SIZE={self.window_size} SEARCH_RANGE={self.search_range} METHOD=coarse\n')
            # 按照x dx y dy 相关系数的形式输出
            for (row, col), (az, rg), corr in zip(filtered_positions, filtered_offsets, filtered_correlations):
                # x: 列坐标（距离向），dx: 距离向偏移
                # y: 行坐标（方位向），dy: 方位向偏移
                x = int(col)  # 转为整型
                dx = rg
                y = int(row)  # 转为整型
                dy = az
                f.write(f'{x} {dx:.3f} {y} {dy:.3f} {corr:.1f}\n')
        
        return {
            'method': 'fft_coarse_2x2',
            'azimuth_offset': avg_az,
            'range_offset': avg_rg,
            'initial_offset': (avg_az, avg_rg),
            'window_offsets': filtered_offsets,
            'correlations': filtered_correlations,
            'average_correlation': avg_corr,
            'window_size': self.window_size,
            'search_range': self.search_range,
            'window_positions': filtered_positions
        }


def coarse_register(master, slave, window_size=256, search_range=64, correlation_threshold=30, output_file=None, auto=True):
    """粗配准函数

    Args:
        auto: 是否自动搜索最佳 search_range / window_size。默认 True。
    """
    if auto:
        return auto_coarse_register(master, slave, correlation_threshold=correlation_threshold, output_file=output_file)
    registrar = CoarseRegistration(window_size, search_range, correlation_threshold)
    return registrar.register(master, slave, output_file=output_file)


def auto_coarse_register(master, slave, correlation_threshold=30, output_file=None):
    """自动粗配准 - 测试不同search_range并选择最佳结果"""
    print("=== 自动粗配准 - 测试不同搜索范围 ===")
    
    # 对于大图像，使用更小的搜索范围以提高效率
    height, width = master.shape
    is_large_image = height > 5000 or width > 5000
    
    if is_large_image:
        # 对于大图像，只测试较小的搜索范围
        search_ranges = [64, 128, 256]
        print(f"检测到大图像 ({height}x{width})，使用简化的搜索范围")
    else:
        # 对于小图像，测试完整的搜索范围
        search_ranges = [64, 128, 256, 512]
    
    # 定义处理单个搜索范围的函数
    def process_search_range(search_range):
        # 以search_range为基准，window_size设置为4倍search_range
        window_size = 4 * search_range
        
        # 执行粗配准，静默模式
        registrar = CoarseRegistration(
            window_size=window_size,
            search_range=search_range,
            correlation_threshold=correlation_threshold
        )
        
        # 暂时重定向stdout，减少输出
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            result = registrar.register(master, slave)
        finally:
            sys.stdout = old_stdout
        
        # 计算偏移量一致性（标准差）
        avg_corr = float(result.get('average_correlation', 0.0))
        if len(result['window_offsets']) >= 3:
            az_offsets = [o[0] for o in result['window_offsets']]
            rg_offsets = [o[1] for o in result['window_offsets']]
            
            az_std = np.std(az_offsets)
            rg_std = np.std(rg_offsets)
            total_consistency = az_std + rg_std  # 总一致性指标
            
            return (result, search_range, total_consistency, avg_corr)
        else:
            return (None, search_range, float('inf'), avg_corr)
    
    # 串行处理以减少内存使用
    best_result = None
    best_consistency = float('inf')
    best_search_range = None
    fallback_result = None
    fallback_search_range = None
    fallback_avg_corr = -float('inf')
    
    for search_range in search_ranges:
        print(f"  测试搜索范围: {search_range}")
        result, search_range, consistency, avg_corr = process_search_range(search_range)
        if result and avg_corr > fallback_avg_corr:
            fallback_result = result
            fallback_search_range = search_range
            fallback_avg_corr = avg_corr

        # 先保证相关性可用，再比较一致性
        if result and avg_corr >= correlation_threshold and consistency < best_consistency:
            best_consistency = consistency
            best_result = result
            best_search_range = search_range
            print(f"  更新最佳结果: search_range={search_range}, 一致性={consistency:.3f}, 平均相关={avg_corr:.3f}")

    # 如果没有达到阈值的候选，回退到平均相关性最高的候选
    if best_result is None and fallback_result is not None:
        best_result = fallback_result
        best_search_range = fallback_search_range
        print(f"  未找到满足相关阈值({correlation_threshold})的候选，回退为相关性最高解: search_range={best_search_range}, 平均相关={fallback_avg_corr:.3f}")
    
    # 输出最佳结果
    if best_result:
        print(f"\n=== 最佳参数组合 ===")
        best_window_size = 4 * best_search_range
        print(f"  window_size={best_window_size}, search_range={best_search_range}")
        print(f"  方位向偏移: {best_result['azimuth_offset']:.4f}")
        print(f"  距离向偏移: {best_result['range_offset']:.4f}")
        print(f"  平均相关系数: {best_result['average_correlation']:.4f}")
        
        # 重新输出最佳结果到offset_estimate.txt，第一行写入最佳参数
        if output_file is None:
            output_dir = os.getcwd()
            output_file = os.path.join(output_dir, 'offset_estimate.txt')
        else:
            output_dir = os.path.dirname(os.path.abspath(output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            # 第一行写入最佳参数
            f.write(f"WINDOW_SIZE={best_window_size} SEARCH_RANGE={best_search_range} METHOD=coarse_auto\n")
            # 输出过滤后的数据
            for (row, col), (az, rg), corr in zip(
                best_result['window_positions'], 
                best_result['window_offsets'], 
                best_result['correlations']
            ):
                x = col
                dx = rg
                y = row
                dy = az
                f.write(f'{x:.3f} {dx:.3f} {y:.3f} {dy:.3f} {corr:.2f}\n')
        print(f"  最佳结果已写入 {output_file}")

        # 回填最佳参数与初始偏移，供后续流程直接使用
        best_result['method'] = 'fft_coarse_2x2_auto'
        best_result['window_size'] = best_window_size
        best_result['search_range'] = best_search_range
        best_result['best_window_size'] = best_window_size
        best_result['best_search_range'] = best_search_range
        best_result['initial_offset'] = (best_result['azimuth_offset'], best_result['range_offset'])
        return best_result
    else:
        print("\n❌ 未找到有效结果")
        return None

def write_offset_result(result, output_file):
    """写入偏移结果到文件
    
    Args:
        result: 粗配准结果字典
        output_file: 输出文件路径
    """
    if not result:
        print("没有有效结果可写入")
        return
    
    try:
        with open(output_file, 'w') as f:
            f.write(f"METHOD={result['method']}\n")
            f.write(f"AZIMUTH_OFFSET={result['azimuth_offset']:.4f}\n")
            f.write(f"RANGE_OFFSET={result['range_offset']:.4f}\n")
            f.write(f"AVERAGE_CORRELATION={result['average_correlation']:.4f}\n")
            f.write(f"WINDOW_SIZE={result['window_size']}\n")
            f.write("\nWINDOW_OFFSETS:\n")
            for i, (az, rg) in enumerate(result['window_offsets']):
                f.write(f"{i}: azimuth={az:.4f}, range={rg:.4f}\n")
        print(f"结果已写入: {output_file}")
    except Exception as e:
        print(f"写入结果失败: {e}")

if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='粗配准模块 - 基于FFT')
    parser.add_argument('master', help='主图像文件路径（VRT或TIFF格式）')
    parser.add_argument('slave', help='辅图像文件路径（VRT或TIFF格式）')
    parser.add_argument('--output', '-o', default='coarse_offset.txt', help='输出结果文件路径')
    parser.add_argument('--correlation-threshold', type=float, default=30, help='相关系数阈值')
    parser.add_argument('--auto', dest='auto', action='store_true', default=True, help='使用自动粗配准模式（默认）')
    parser.add_argument('--manual', dest='auto', action='store_false', help='关闭自动模式，使用 --window-size/--search-range')
    parser.add_argument('--window-size', type=int, default=256, help='窗口大小')
    parser.add_argument('--search-range', type=int, default=64, help='搜索范围')
    
    args = parser.parse_args()
    
    print("=== 粗配准模块 ===")
    print(f"主图像: {args.master}")
    print(f"辅图像: {args.slave}")
    
    # 读取图像
    master = read_image(args.master)
    slave = read_image(args.slave)
    
    if master is None or slave is None:
        print("无法读取图像文件")
        exit(1)
    
    print(f"主图像大小: {master.shape}, 类型: {master.dtype}")
    print(f"辅图像大小: {slave.shape}, 类型: {slave.dtype}")
    
    # 执行粗配准
    if args.auto:
        print("执行自动粗配准...")
        result = auto_coarse_register(master, slave, args.correlation_threshold)
    else:
        print(f"执行粗配准，窗口大小={args.window_size}, 搜索范围={args.search_range}")
        result = coarse_register(master, slave, args.window_size, args.search_range, args.correlation_threshold, auto=False)
    
    # 输出结果
    if result:
        print(f"\n=== 粗配准结果 ===")
        print(f"方法: {result['method']}")
        print(f"方位向偏移: {result['azimuth_offset']:.4f}")
        print(f"距离向偏移: {result['range_offset']:.4f}")
        print(f"平均相关系数: {result['average_correlation']:.4f}")
        print(f"窗口大小: {result['window_size']}")
        if 'search_range' in result:
            print(f"搜索范围: {result['search_range']}")
        print(f"有效窗口数量: {len(result['window_offsets'])}")
        if 'initial_offset' in result:
            print(f"初始偏移: az={result['initial_offset'][0]:.4f}, rg={result['initial_offset'][1]:.4f}")
        
        # 检查输出路径
        output_path = args.output
        if os.path.isdir(output_path):
            # 如果是目录，使用默认文件名
            output_file = os.path.join(output_path, 'coarse_offset.txt')
            print(f"输出路径是目录，使用默认文件名: {output_file}")
        else:
            output_file = output_path
        
        # 写入结果
        write_offset_result(result, output_file)
    else:
        print("粗配准失败")
        exit(1)
