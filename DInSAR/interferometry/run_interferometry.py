#!/usr/bin/env python3
"""
干涉处理执行脚本
功能：执行完整的干涉处理流程，包括生成干涉图、计算相干性、滤波和输出结果
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from interferometry import (
    InterferogramGenerator,
    CoherenceCalculator,
    InterferogramFilter,
    load_slc_data,
    save_data
)
from osgeo import gdal

def read_image(file_path):
    """读取图像文件
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        图像数据（复数数组）
    """
    try:
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

def save_tiff(data, output_file):
    """保存数据为TIFF文件
    
    Args:
        data: 数据数组
        output_file: 输出文件路径
    """
    try:
        # 确保数据类型正确
        if data.dtype == np.complex64:
            # 保存幅度
            data = np.abs(data)
        
        # 归一化到 0-255 范围
        min_val = data.min()
        max_val = data.max()
        if max_val > min_val:
            data_normalized = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            data_normalized = np.zeros_like(data, dtype=np.uint8)
        
        # 创建TIFF文件
        height, width = data.shape
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_file, width, height, 1, gdal.GDT_Byte)
        if out_ds:
            out_ds.GetRasterBand(1).WriteArray(data_normalized)
            out_ds.FlushCache()
            out_ds = None
            print(f"已保存TIFF文件: {output_file}")
        else:
            print(f"无法创建TIFF文件: {output_file}")
    except Exception as e:
        print(f"保存TIFF失败: {e}")

def save_phase_tiff(phase, output_file):
    """保存相位数据为彩色TIFF文件
    
    Args:
        phase: 相位数据 (-pi 到 pi)
        output_file: 输出文件路径
    """
    try:
        # 将相位转换为 0-255 范围
        phase_normalized = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        
        # 创建TIFF文件
        height, width = phase.shape
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_file, width, height, 1, gdal.GDT_Byte)
        if out_ds:
            out_ds.GetRasterBand(1).WriteArray(phase_normalized)
            out_ds.FlushCache()
            out_ds = None
            print(f"已保存相位TIFF文件: {output_file}")
        else:
            print(f"无法创建相位TIFF文件: {output_file}")
    except Exception as e:
        print(f"保存相位TIFF失败: {e}")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='干涉处理执行脚本')
    parser.add_argument('master', help='主图像文件路径（VRT或TIFF格式）')
    parser.add_argument('slave', help='辅图像文件路径（VRT或TIFF格式，重采样后）')
    parser.add_argument('--output-dir', default='.', help='输出目录')
    parser.add_argument('--window-size', type=int, default=5, help='相干性计算窗口大小')
    parser.add_argument('--filter-method', choices=['gaussian', 'adaptive', 'median', 'goldstein'], default='adaptive', help='滤波方法')
    parser.add_argument('--sigma', type=float, default=1.0, help='高斯滤波sigma值')
    parser.add_argument('--alpha', type=float, default=1.0, help='Goldstein滤波参数（0-1，值越大滤波越强）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== 干涉处理流程 ===")
    print(f"主图像: {args.master}")
    print(f"辅图像: {args.slave}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 1. 读取图像
    print("1. 读取图像...")
    master = read_image(args.master)
    slave = read_image(args.slave)
    
    if master is None or slave is None:
        print("错误：无法读取图像文件")
        return 1
    
    print(f"  主图像大小: {master.shape}, 类型: {master.dtype}")
    print(f"  辅图像大小: {slave.shape}, 类型: {slave.dtype}")
    
    # 2. 生成干涉图
    print("\n2. 生成干涉图...")
    generator = InterferogramGenerator()
    result = generator.generate(master, slave)
    
    print(f"  干涉图生成完成")
    print(f"  相位范围: [{result['phase'].min():.2f}, {result['phase'].max():.2f}]")
    
    # 3. 计算相干性
    print("\n3. 计算相干性...")
    calculator = CoherenceCalculator(window_size=args.window_size)
    coherence = calculator.calculate_efficient(master, slave)
    
    print(f"  相干性计算完成")
    print(f"  相干性范围: [{coherence.min():.4f}, {coherence.max():.4f}]")
    print(f"  平均相干性: {coherence.mean():.4f}")
    
    # 4. 滤波干涉图
    print("\n4. 滤波干涉图...")
    filter_obj = InterferogramFilter(method=args.filter_method)
    filtered_phase = filter_obj.filter(result['phase'], coherence, sigma=args.sigma, alpha=args.alpha)
    
    print(f"  滤波完成")
    print(f"  滤波后相位范围: [{filtered_phase.min():.2f}, {filtered_phase.max():.2f}]")
    
    # 5. 输出结果
    print("\n5. 输出结果...")
    
    # 构建输出文件名
    master_name = Path(args.master).stem
    slave_name = Path(args.slave).stem
    
    # 保存干涉数据
    interferogram_file = output_dir / f"{master_name}_{slave_name}_interferogram.bin"
    save_data(result['interferogram'], str(interferogram_file))
    print(f"  干涉数据已保存: {interferogram_file}")
    
    # 保存相干数据
    coherence_file = output_dir / f"{master_name}_{slave_name}_coherence.bin"
    save_data(coherence, str(coherence_file))
    print(f"  相干数据已保存: {coherence_file}")
    
    # 保存滤波后的相位
    filtered_phase_file = output_dir / f"{master_name}_{slave_name}_filtered_phase.bin"
    save_data(filtered_phase, str(filtered_phase_file))
    print(f"  滤波后相位已保存: {filtered_phase_file}")
    
    # 生成TIFF图像
    # 干涉条纹图（相位）
    phase_tiff = output_dir / f"{master_name}_{slave_name}_phase.tif"
    save_phase_tiff(result['phase'], str(phase_tiff))
    
    # 滤波后的干涉条纹图
    filtered_phase_tiff = output_dir / f"{master_name}_{slave_name}_filtered_phase.tif"
    save_phase_tiff(filtered_phase, str(filtered_phase_tiff))
    
    # 相干性图
    coherence_tiff = output_dir / f"{master_name}_{slave_name}_coherence.tif"
    save_tiff(coherence, str(coherence_tiff))
    
    # 幅度图
    amplitude_tiff = output_dir / f"{master_name}_{slave_name}_amplitude.tif"
    save_tiff(result['amplitude'], str(amplitude_tiff))
    
    print("\n=== 干涉处理完成 ===")
    print(f"所有结果已输出到: {output_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
