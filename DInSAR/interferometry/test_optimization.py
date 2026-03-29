#!/usr/bin/env python3
"""
测试优化后的干涉处理性能
"""

import time
import numpy as np
from interferometry import InterferogramGenerator, CoherenceCalculator, InterferogramFilter

def test_performance():
    """测试性能"""
    print("=== 测试优化后的性能 ===")
    
    # 创建测试数据
    h, w = 1024, 1024
    print(f"创建测试数据: {h}x{w}")
    
    # 生成随机复数数据
    master = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    slave = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    
    # 1. 测试干涉图生成
    print("\n1. 测试干涉图生成...")
    generator = InterferogramGenerator()
    start_time = time.time()
    result = generator.generate(master, slave)
    end_time = time.time()
    print(f"   用时: {end_time - start_time:.4f}秒")
    
    # 2. 测试相干性计算
    print("\n2. 测试相干性计算...")
    calculator = CoherenceCalculator(window_size=5)
    start_time = time.time()
    coherence = calculator.calculate_efficient(master, slave)
    end_time = time.time()
    print(f"   用时: {end_time - start_time:.4f}秒")
    print(f"   相干性范围: [{coherence.min():.4f}, {coherence.max():.4f}]")
    
    # 3. 测试Goldstein滤波
    print("\n3. 测试Goldstein滤波...")
    filter_obj = InterferogramFilter(method='goldstein')
    start_time = time.time()
    filtered_phase = filter_obj.filter(result['phase'], coherence, alpha=0.5)
    end_time = time.time()
    print(f"   用时: {end_time - start_time:.4f}秒")
    print(f"   滤波后相位范围: [{filtered_phase.min():.2f}, {filtered_phase.max():.2f}]")
    
    # 4. 测试大型图像（分块处理）
    print("\n4. 测试大型图像（分块处理）...")
    h_large, w_large = 4096, 4096
    print(f"创建大型测试数据: {h_large}x{w_large}")
    
    # 生成随机复数数据
    master_large = np.random.randn(h_large, w_large) + 1j * np.random.randn(h_large, w_large)
    slave_large = np.random.randn(h_large, w_large) + 1j * np.random.randn(h_large, w_large)
    
    # 测试相干性计算
    print("   测试相干性计算...")
    start_time = time.time()
    coherence_large = calculator.calculate_efficient(master_large, slave_large)
    end_time = time.time()
    print(f"   用时: {end_time - start_time:.4f}秒")
    print(f"   相干性范围: [{coherence_large.min():.4f}, {coherence_large.max():.4f}]")
    
    # 测试Goldstein滤波
    print("   测试Goldstein滤波...")
    start_time = time.time()
    result_large = generator.generate(master_large, slave_large)
    filtered_phase_large = filter_obj.filter(result_large['phase'], coherence_large, alpha=0.5)
    end_time = time.time()
    print(f"   用时: {end_time - start_time:.4f}秒")
    print(f"   滤波后相位范围: [{filtered_phase_large.min():.2f}, {filtered_phase_large.max():.2f}]")
    
    print("\n=== 测试完成 ===")

if __name__ == '__main__':
    test_performance()
