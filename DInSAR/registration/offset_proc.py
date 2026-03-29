#!/usr/bin/env python3
"""
偏移处理模块
功能：整合偏移估计和配准质量评估功能
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# 添加工具目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from sar_utils import read_image, compute_correlation, compute_snr, compute_rmse, compute_local_correlation

# 从offset_estimation.py导入函数
from offset_estimation import (
    _read_fine_offsets,
    IterativeOffsetEstimator,
    estimate_offsets,
    read_offsets_from_file,
    write_offset_estimate_result
)

# 从registration_quality.py导入函数
from registration_quality import (
    _read_offset_estimation,
    RegistrationQualityAssessment,
    assess_registration_quality,
    write_quality_result
)


if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='偏移处理模块 - 整合偏移估计和配准质量评估')
    parser.add_argument('--mode', '-m', choices=['estimate', 'quality', 'both'], default='both',
                        help='运行模式: estimate(仅偏移估计), quality(仅质量评估), both(两者都运行)')
    parser.add_argument('--input', '-i', help='输入偏移数据文件路径（可选，默认从offset_estimate.txt读取）')
    parser.add_argument('--output', '-o', default='offset_estimate.txt', help='输出偏移估计结果文件路径')
    parser.add_argument('--quality-output', default='registration_quality.txt', help='质量评估结果文件路径')
    parser.add_argument('--polynomial-order', type=int, default=2, choices=[1, 2], help='多项式阶数（1=线性，2=二次）')
    parser.add_argument('--max-iterations', type=int, default=10, help='最大迭代次数')
    parser.add_argument('--initial-outlier-threshold', type=float, default=2.0, help='初始离群点阈值（标准差倍数）')
    parser.add_argument('--use-segmentation', action='store_true', help='使用分段式拟合')
    parser.add_argument('--num-segments', type=int, default=5, help='分段数量')
    parser.add_argument('--from-file', action='store_true', help='从offset_estimate.txt文件读取偏移估计结果进行质量评估')
    parser.add_argument('--reference', '-r', help='参考图像文件路径（用于直接质量评估）')
    parser.add_argument('--registered', '-g', help='配准后图像文件路径（用于直接质量评估）')
    
    args = parser.parse_args()
    
    print("=== 偏移处理模块 ===")
    input_file = args.input if args.input else 'offset_estimate.txt'
    output_file = args.output
    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, 'offset_estimate.txt')
    
    if args.mode in ['estimate', 'both']:
        print("\n--- 执行偏移估计 ---")
        # 读取偏移数据
        if args.input:
            print(f"从文件读取偏移数据: {args.input}")
            offsets = read_offsets_from_file(args.input)
        else:
            print("从默认文件offset_estimate.txt读取偏移数据")
            offsets = _read_fine_offsets(input_file)
        
        print(f"读取到 {len(offsets)} 个偏移点")
        
        if not offsets:
            print("没有找到偏移数据")
            exit(1)
        
        # 执行偏移估计
        print("执行迭代偏移估计...")
        result = estimate_offsets(
            offsets=offsets,
            polynomial_order=args.polynomial_order,
            max_iterations=args.max_iterations,
            initial_outlier_threshold=args.initial_outlier_threshold,
            use_segmentation=args.use_segmentation,
            num_segments=args.num_segments,
            input_file=input_file,
            output_file=output_file,
        )
        
        # 输出结果
        if result:
            print(f"\n=== 偏移估计结果 ===")
            print(f"多项式阶数: {result['polynomial_order']}")
            print(f"初始点数: {result['initial_points']}")
            print(f"最终点数: {result['final_points']}")
            print(f"删除的离群点数: {result['outliers_removed']}")
            print(f"方位向RMS残差: {result['final_residuals']['azimuth_rms']:.4f}")
            print(f"距离向RMS残差: {result['final_residuals']['range_rms']:.4f}")
            
            if 'fitted_params' in result and 'fit_type' in result['fitted_params']:
                print(f"拟合类型: {result['fitted_params']['fit_type']}")
            
            # estimate_offsets 已经写入 output_file，这里不重复写盘
        else:
            print("偏移估计失败")
            exit(1)
    
    if args.mode in ['quality', 'both']:
        print("\n--- 执行配准质量评估 ---")
        # 执行质量评估
        if args.from_file:
            print("从文件读取偏移估计结果进行评估")
            quality_input = output_file if args.mode in ['estimate', 'both'] else input_file
            quality_result = assess_registration_quality(from_file=True, offset_file=quality_input)
        elif args.reference and args.registered:
            print(f"直接评估图像配准质量")
            print(f"参考图像: {args.reference}")
            print(f"配准后图像: {args.registered}")
            
            # 读取图像
            reference = read_image(args.reference)
            registered = read_image(args.registered)
            
            if reference is None or registered is None:
                print("无法读取图像文件")
                exit(1)
            
            print(f"参考图像大小: {reference.shape}, 类型: {reference.dtype}")
            print(f"配准后图像大小: {registered.shape}, 类型: {registered.dtype}")
            
            # 执行评估
            quality_result = assess_registration_quality(reference, registered)
        else:
            print("错误：请指定 --from-file 或同时指定 --reference 和 --registered")
            exit(1)
        
        # 输出结果
        if quality_result:
            print(f"\n=== 配准质量评估结果 ===")
            print(f"质量等级: {quality_result.get('quality', 'poor')}")
            
            if 'azimuth_rms' in quality_result:
                print(f"方位向RMS: {quality_result['azimuth_rms']:.4f}")
            if 'range_rms' in quality_result:
                print(f"距离向RMS: {quality_result['range_rms']:.4f}")
            if 'correlation' in quality_result:
                print(f"相关系数: {quality_result['correlation']:.4f}")
            if 'snr_db' in quality_result:
                print(f"信噪比: {quality_result['snr_db']:.2f} dB")
            if 'rmse' in quality_result:
                print(f"RMSE: {quality_result['rmse']:.4f}")
            
            if 'points' in quality_result:
                print("\n点统计:")
                for key, value in quality_result['points'].items():
                    print(f"  {key}: {value:.4f}")
            
            if 'details' in quality_result:
                print("\n详细统计:")
                for key, value in quality_result['details'].items():
                    print(f"  {key}: {value:.4f}")
            
            # 写入质量评估结果
            quality_output_file = args.quality_output
            if os.path.isdir(quality_output_file):
                quality_output_file = os.path.join(quality_output_file, 'registration_quality.txt')
            write_quality_result(quality_result, quality_output_file)
        else:
            print("质量评估失败")
            exit(1)
    
    print("\n=== 偏移处理模块执行完成 ===")
