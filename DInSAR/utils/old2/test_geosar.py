#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 geosar.py 读取 master.yaml 的功能
"""

import numpy as np
import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geosar import Geo2Rdr

# 测试读取 master.yaml
def test_read_master_yaml():
    """测试读取 master.yaml 文件"""
    print("测试读取 master.yaml 文件...")
    
    # 路径设置
    yaml_file = "~/Temp/yaxia/output/master.yaml"
    # 扩展 ~ 为绝对路径
    yaml_file = os.path.expanduser(yaml_file)
    
    # 检查文件是否存在
    if not os.path.exists(yaml_file):
        print(f"错误: 文件 {yaml_file} 不存在")
        return False
    
    # 创建一个简单的 DEM 网格用于测试
    # 这里使用一个小的 DEM 网格，实际应用中应该使用真实的 DEM 文件
    dem_lat = np.array([[29.6, 29.6], [29.7, 29.7]])
    dem_lon = np.array([[94.8, 94.9], [94.8, 94.9]])
    dem_h = np.array([[0, 0], [0, 0]])
    
    try:
        # 初始化 Geo2Rdr
        geo = Geo2Rdr(yaml_file, dem_lat, dem_lon, dem_h)
        print("成功初始化 Geo2Rdr")
        
        # 打印读取的参数
        print(f"t0: {geo.t0}")
        if hasattr(geo, 't0_str'):
            print(f"t0_str: {geo.t0_str}")
        print(f"prf: {geo.prf}")
        print(f"near_range: {geo.near_range}")
        print(f"range_spacing: {geo.range_spacing}")
        print(f"look_dir: {geo.look_dir}")
        print(f"ascending: {geo.ascending}")
        print(f"wavelength: {geo.wavelength}")
        print(f"azimuth_pixel_spacing: {geo.azimuth_pixel_spacing}")
        
        # 测试轨道插值
        if hasattr(geo.orbit, 'interpolator') and geo.orbit.interpolator:
            print("轨道插值器初始化成功")
            # 测试位置计算
            t = 0.0
            pos = geo.orbit.position(t)
            print(f"t={t} 时的卫星位置: {pos}")
        else:
            print("警告: 轨道插值器未初始化")
        
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_read_master_yaml()
    if success:
        print("\n测试通过!")
    else:
        print("\n测试失败!")
