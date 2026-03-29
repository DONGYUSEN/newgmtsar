#!/usr/bin/env python3
"""
精配准模块 - 优化版本
功能：在Coarse Registration基础上，使用多尺度窗口在全图均匀分布配准
使用多进程并行计算提高效率
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
from scipy import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from sar_utils import read_image


def _read_coarse_offset(input_file='offset_estimate.txt'):
    """从 offset_estimate.txt 读取粗配准结果和参数"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # 读取第一行的参数
        window_size = 1024
        search_range = 256
        if lines:
            first_line = lines[0].strip()
            if 'WINDOW_SIZE' in first_line and 'SEARCH_RANGE' in first_line:
                # 解析第一行参数
                parts = first_line.split()
                for part in parts:
                    if part.startswith('WINDOW_SIZE='):
                        window_size = int(part.split('=')[1])
                    elif part.startswith('SEARCH_RANGE='):
                        search_range = int(part.split('=')[1])
        
        # 读取粗配准结果（兼容有/无头行的格式）
        az_offset = 0.0  # dy
        rg_offset = 0.0  # dx
        point_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 解析x dx y dy correlation格式的行
            try:
                parts = list(map(float, line.split()))
                if len(parts) == 5:
                    # 取所有点的平均值
                    az_offset += parts[3]  # dy (方位向偏移)
                    rg_offset += parts[1]  # dx (距离向偏移)
                    point_count += 1
            except:
                pass
        
        # 计算平均值
        if point_count > 0:
            az_offset /= point_count
            rg_offset /= point_count
        
        print(f"  读取粗配准结果: dx={rg_offset}, dy={az_offset}, file={input_file}")
        return (az_offset, rg_offset, window_size, search_range)
    except Exception as e:
        print(f"  读取粗配准结果失败: {e}")
        return (0.0, 0.0, 1024, 256)


def _register_single_point(args):
    """单点配准函数（用于并行计算）"""
    row, col, master, slave, window_size, initial_offset = args
    
    half = window_size // 2
    h, w = master.shape
    
    # 使用初始偏移调整窗口位置
    r0 = int(row + initial_offset[0])
    c0 = int(col + initial_offset[1])
    
    r_start = max(0, r0 - half)
    r_end = min(h, r0 + half)
    c_start = max(0, c0 - half)
    c_end = min(w, c0 + half)
    
    master_window = master[r_start:r_end, c_start:c_end]
    slave_window = slave[r_start:r_end, c_start:c_end]
    
    if master_window.shape[0] < 10 or master_window.shape[1] < 10:
        return None
    
    master_win = _adjust_window(master_window, window_size)
    slave_win = _adjust_window(slave_window, window_size)
    
    # 计算局部偏移
    offset_az, offset_rg, corr = _cross_correlation(master_win, slave_win)
    
    # 注意：根据coarse_registration.py的实现，需要取负值
    offset_az = -offset_az
    offset_rg = -offset_rg
    
    # 将局部偏移与初始偏移相加，得到最终偏移
    real_az = offset_az + initial_offset[0]
    real_rg = offset_rg + initial_offset[1]
    
    return {
        'row': float(row),
        'col': float(col),
        'azimuth': real_az,
        'range': real_rg,
        'azimuth_local': offset_az,
        'range_local': offset_rg,
        'correlation': corr,
        'window_size': window_size
    }


def _adjust_window(window, target_size):
    h, w = window.shape
    if h == target_size and w == target_size:
        return window
    result = np.zeros((target_size, target_size), dtype=window.dtype)
    copy_h = min(h, target_size)
    copy_w = min(w, target_size)
    dst_h = (target_size - copy_h) // 2
    dst_w = (target_size - copy_w) // 2
    result[dst_h:dst_h+copy_h, dst_w:dst_w+copy_w] = window[:copy_h, :copy_w]
    return result


def _cross_correlation(master, slave):
    """使用与coarse_register.py相同的FFT相关算法"""
    # 确保输入是实数类型
    ref = np.abs(master) if np.iscomplexobj(master) else master
    sec = np.abs(slave) if np.iscomplexobj(slave) else slave
    
    # 转换为float64提高精度
    ref_amp = ref.astype(np.float64)
    sec_amp = sec.astype(np.float64)
    
    # 均值减法
    ref_amp -= ref_amp.mean()
    sec_amp -= sec_amp.mean()
    
    # 应用掩码（与 coarse_regist.py 保持一致，向量化）
    h, w = ref_amp.shape
    search_range = min(h, w) // 4  # 与coarse_registration.py中的逻辑保持一致
    sec_amp[:search_range, :] = 0
    sec_amp[h - search_range:, :] = 0
    sec_amp[:, :search_range] = 0
    sec_amp[:, w - search_range:] = 0
    
    # FFT相关
    from scipy import fftpack
    ref_fft = fftpack.fft2(ref_amp)
    sec_fft = fftpack.fft2(sec_amp)
    
    # 计算交叉功率谱
    cross_power = ref_fft * np.conj(sec_fft)
    
    # 应用相位因子（向量化）
    cols = w // 2 + 1
    sign = (1.0 - 2.0 * (np.arange(h * cols) % 2)).reshape(h, cols)
    cross_power[:, :cols] *= sign
    
    # 逆FFT得到相关结果
    correlation = np.abs(np.real(fftpack.ifft2(cross_power))) / (h * w)
    
    # 搜索区域
    corr_search = correlation[search_range:search_range+2*search_range, search_range:search_range+2*search_range]
    
    if corr_search.size == 0:
        search_range = min(search_range, h // 4)
        corr_search = correlation[search_range:search_range+2*search_range, search_range:search_range+2*search_range]
    
    # 找到最大值位置
    max_idx = np.unravel_index(np.argmax(corr_search), corr_search.shape)
    cmax = float(corr_search[max_idx])
    
    # 计算偏移量
    offset_az = max_idx[0] - search_range  # 方位向偏移
    offset_rg = max_idx[1] - search_range  # 距离向偏移
    
    # 时间域相关计算
    def time_corr(c1, c2, xoff, yoff):
        ny_corr = 2 * search_range
        nx_corr = 2 * search_range
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
    max_corr = time_corr(ref_amp, sec_amp, int(offset_rg), int(offset_az))
    
    # 亚像素拟合
    interp_factor = 16  # 插值因子
    n2x = 8  # 相关窗口大小
    n2y = 8
    
    if interp_factor > 1:
        # 对相关结果进行缩放
        if cmax > 0:
            corr_search = corr_search * (max_corr / cmax)
        
        # 确保峰值位置在有效范围内
        if offset_az + search_range < n2y/2:
            offset_az = n2y/2 - search_range
        elif offset_az + search_range >= 2*search_range - n2y/2:
            offset_az = 2*search_range - n2y/2 - search_range - 1
        
        if offset_rg + search_range < n2x/2:
            offset_rg = n2x/2 - search_range
        elif offset_rg + search_range >= 2*search_range - n2x/2:
            offset_rg = 2*search_range - n2x/2 - search_range - 1
        
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
    
    return (offset_az, offset_rg, max_corr)


def _subpixel_fit(correlation, max_idx):
    h, w = correlation.shape
    row, col = max_idx
    if row > 0 and row < h - 1 and col > 0 and col < w - 1:
        c = correlation[row, col]
        cy1 = correlation[row-1, col]
        cy2 = correlation[row+1, col]
        cy = (cy2 - cy1) / 2
        cyy = cy2 - 2*c + cy1
        cx1 = correlation[row, col-1]
        cx2 = correlation[row, col+1]
        cx = (cx2 - cx1) / 2
        cxx = cx2 - 2*c + cx1
        az_sub = -cy / cyy if abs(cyy) > 1e-10 else 0
        rg_sub = -cx / cxx if abs(cxx) > 1e-10 else 0
        return (az_sub, rg_sub)
    return (0.0, 0.0)


def _normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        return (data - mean) / std
    return data - mean


class FineRegistration:
    def __init__(self, window_sizes=None, grid_spacing=None, 
                 initial_offset=None, num_workers=None, coarse_offset_file='offset_estimate.txt',
                 correlation_threshold=30.0):
        # 从文件读取粗配准结果和参数（作为默认值）
        coarse_result = _read_coarse_offset(coarse_offset_file)
        coarse_initial = (coarse_result[0], coarse_result[1])
        self.coarse_window_size = coarse_result[2]
        self.search_range = coarse_result[3]
        print(f"  从文件读取粗配准结果: {coarse_initial}")
        print(f"  粗配准参数: window_size={self.coarse_window_size}, search_range={self.search_range}")
        
        # 如果没有提供window_sizes，使用默认值
        if window_sizes is None:
            window_sizes = [64, 128, 256, 512]
        self.window_sizes = sorted(window_sizes)
        
        # 优先使用用户输入参数
        if grid_spacing is None:
            self.grid_spacing = self.search_range
        else:
            self.grid_spacing = int(grid_spacing)
        if initial_offset is None:
            self.initial_offset = coarse_initial
        else:
            self.initial_offset = (float(initial_offset[0]), float(initial_offset[1]))
        self.initial_offset = (round(self.initial_offset[0]), round(self.initial_offset[1]))
        print(f"  网格大小: {self.grid_spacing}")
        print(f"  初始偏移（取整）: {self.initial_offset}")
        
        # 设置线程数
        if num_workers is None:
            num_workers = 8  # 使用8个线程
        self.num_workers = num_workers
        self.correlation_threshold = float(correlation_threshold)
    
    def register(self, master, slave, output_file=None):
        h, w = master.shape
        grid_points = self._create_grid(h, w)
        all_offsets = []
        
        # 使用候选窗口里最接近 256 的值
        best_window_size = min(self.window_sizes, key=lambda v: abs(v - 256))
        print(f"  使用窗口大小: {best_window_size}x{best_window_size}, 线程数: {self.num_workers}")
        offsets = self._register_parallel(master, slave, grid_points, best_window_size)
        all_offsets = offsets
        
        combined_offsets = self._combine_offsets(all_offsets)
        
        # 过滤低相关点
        filtered_offsets = [offset for offset in combined_offsets if offset['correlation'] >= self.correlation_threshold]
        print(f"  过滤后配准点数: {len(filtered_offsets)}")
        
        # 输出结果到offset_estimate.txt（覆盖模式）
        if output_file is None:
            output_file = os.path.join(os.getcwd(), 'offset_estimate.txt')
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            # 第一行写入精配准参数
            f.write(f'WINDOW_SIZE={best_window_size} SEARCH_RANGE={self.search_range} METHOD=fine\n')
            # 按照x dx y dy corr的格式写入所有点，靠右对齐
            for offset in filtered_offsets:
                x = int(offset['col'])  # 列坐标（距离向），转为整型
                dx = offset['range_local']  # 距离向局部偏移
                y = int(offset['row'])  # 行坐标（方位向），转为整型
                dy = offset['azimuth_local']  # 方位向局部偏移
                corr = offset['correlation']  # 相关系数
                # 格式化输出，靠右对齐
                f.write(f'{x:8d} {dx:8.3f} {y:8d} {dy:8.3f} {corr:8.1f}\n')
        print(f"    结果已写入 {output_file}")
        
        return {
            'method': 'fine_registration',
            'window_size': best_window_size,
            'grid_spacing': self.grid_spacing,
            'initial_offset': self.initial_offset,
            'offsets': filtered_offsets,
            'num_points': len(filtered_offsets),
            'num_workers': self.num_workers
        }
    
    def _create_grid(self, h, w):
        """创建网格点，使用粗配准的search_range作为网格大小
        当配准点数不足时，允许窗口有一定的重叠
        """
        # 直接使用粗配准的search_range作为网格大小
        grid_spacing = self.grid_spacing
        
        # 创建网格点
        points = []
        row = grid_spacing // 2
        while row < h - grid_spacing:
            col = grid_spacing // 2
            while col < w - grid_spacing:
                points.append((row, col))
                col += grid_spacing
            row += grid_spacing
        
        # 打印网格点信息
        rows = (h - grid_spacing) // grid_spacing + 1
        cols = (w - grid_spacing) // grid_spacing + 1
        print(f"  网格点分布: {rows}行 × {cols}列 = {len(points)}个点")
        
        # 检查是否有足够的点，如果不足则增加重叠
        if len(points) < 900:
            print(f"  ⚠️  配准点数不足（{len(points)} < 900），增加窗口重叠以获取更多点")
            # 计算需要的重叠率
            overlap_factor = 0.5  # 50% 重叠
            new_spacing = int(grid_spacing * (1 - overlap_factor))
            
            # 重新创建网格点，使用更小的间距（增加重叠）
            points = []
            row = grid_spacing // 2
            while row < h - grid_spacing:
                col = grid_spacing // 2
                while col < w - grid_spacing:
                    points.append((row, col))
                    col += new_spacing
                row += new_spacing
            
            # 重新计算网格点信息
            new_rows = (h - grid_spacing) // new_spacing + 1
            new_cols = (w - grid_spacing) // new_spacing + 1
            print(f"  增加重叠后: {new_rows}行 × {new_cols}列 = {len(points)}个点")
        else:
            print(f"  ✅ 配准点数充足（{len(points)} ≥ 900）")
        
        return points
    
    def _register_parallel(self, master, slave, grid_points, window_size):
        args_list = [
            (row, col, master, slave, window_size, self.initial_offset)
            for row, col in grid_points
        ]
        offsets = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_register_single_point, args): args for args in args_list}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        offsets.append(result)
                except Exception as exc:
                    row, col, _, _, _, _ = futures[future]
                    print(f"  配准点失败 row={row}, col={col}: {exc}")
        return offsets
    
    def _combine_offsets(self, all_offsets):
        """合并偏移量，确保保留足够的点
        当配准点数不足时，保留所有点而不合并重叠区域
        """
        # 检查总点数
        total_points = len(all_offsets)
        print(f"  原始配准点数: {total_points}")
        
        # 直接返回所有点，不进行合并
        # 这样可以确保我们有足够的配准点
        print(f"  保留所有配准点: {total_points}")
        combined = all_offsets
        
        return combined


def fine_register(master, slave, window_sizes=None, grid_spacing=None,
                initial_offset=(0.0, 0.0), num_workers=None, output_dir=None,
                coarse_offset_file='offset_estimate.txt', correlation_threshold=30.0,
                output_file=None):
    registrar = FineRegistration(
        window_sizes=window_sizes,
        grid_spacing=grid_spacing,
        initial_offset=initial_offset,
        num_workers=num_workers,
        coarse_offset_file=coarse_offset_file,
        correlation_threshold=correlation_threshold
    )
    if output_file is None and output_dir is not None:
        output_file = os.path.join(output_dir, 'offset_estimate.txt')
    return registrar.register(master, slave, output_file=output_file)


def write_fine_result(result, output_file):
    """写入精配准结果到文件
    
    Args:
        result: 精配准结果字典
        output_file: 输出文件路径
    """
    if not result:
        print("没有有效结果可写入")
        return
    
    try:
        with open(output_file, 'w') as f:
            f.write(f"METHOD={result['method']}\n")
            f.write(f"WINDOW_SIZE={result['window_size']}\n")
            f.write(f"GRID_SPACING={result['grid_spacing']}\n")
            f.write(f"INITIAL_OFFSET={result['initial_offset'][0]:.4f}, {result['initial_offset'][1]:.4f}\n")
            f.write(f"NUM_POINTS={result['num_points']}\n")
            f.write(f"NUM_WORKERS={result['num_workers']}\n")
            f.write("\nOFFSETS:\n")
            for i, offset in enumerate(result['offsets']):
                f.write(f"{i}: row={offset['row']:.1f}, col={offset['col']:.1f}, azimuth={offset['azimuth']:.4f}, range={offset['range']:.4f}, correlation={offset['correlation']:.2f}\n")
        print(f"结果已写入: {output_file}")
    except Exception as e:
        print(f"写入结果失败: {e}")

if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='精配准模块 - 优化版本')
    parser.add_argument('master', help='主图像文件路径（VRT或TIFF格式）')
    parser.add_argument('slave', help='辅图像文件路径（VRT或TIFF格式）')
    parser.add_argument('--output', '-o', default='fine_offset.txt', help='输出结果文件路径')
    parser.add_argument('--window-sizes', type=int, nargs='+', default=[64, 128, 256, 512], help='窗口大小列表')
    parser.add_argument('--grid-spacing', type=int, help='网格间距')
    parser.add_argument('--initial-offset', type=float, nargs=2, default=(0.0, 0.0), help='初始偏移 (azimuth, range)')
    parser.add_argument('--num-workers', type=int, help='线程数')
    
    args = parser.parse_args()
    
    print("=== 精配准模块 ===")
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
    
    # 执行精配准
    print("执行精配准...")
    
    # 确定输出目录
    output_path = args.output
    if os.path.isdir(output_path):
        # 如果是目录，使用默认文件名
        output_file = os.path.join(output_path, 'fine_offset.txt')
        output_dir = output_path
        print(f"输出路径是目录，使用默认文件名: {output_file}")
    else:
        output_file = output_path
        output_dir = os.path.dirname(output_file)
        if not output_dir:
            output_dir = os.getcwd()
    
    result = fine_register(master, slave, 
                         window_sizes=args.window_sizes,
                         grid_spacing=args.grid_spacing,
                         initial_offset=tuple(args.initial_offset),
                         num_workers=args.num_workers,
                         output_dir=output_dir)
    
    # 输出结果
    if result:
        print(f"\n=== 精配准结果 ===")
        print(f"方法: {result['method']}")
        print(f"窗口大小: {result['window_size']}")
        print(f"网格间距: {result['grid_spacing']}")
        print(f"初始偏移: {result['initial_offset']}")
        print(f"配准点数: {result['num_points']}")
        print(f"线程数: {result['num_workers']}")
        
        # 写入结果
        write_fine_result(result, output_file)
    else:
        print("精配准失败")
        exit(1)
