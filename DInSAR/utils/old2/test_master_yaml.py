#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试读取 master.yaml 文件的功能
"""

import yaml
import os
import datetime

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
    
    try:
        # 读取 YAML 文件
        with open(yaml_file, 'r') as f:
            ycfg = yaml.safe_load(f)
        
        print("成功读取 master.yaml 文件")
        
        # 验证文件结构
        print("\n文件结构验证:")
        
        # 检查必要的键
        required_keys = ['corner_coordinates', 'geolocation_grid', 'image_parameters', 'metadata', 'orbit_data']
        for key in required_keys:
            if key in ycfg:
                print(f"✓ 存在 {key}")
            else:
                print(f"✗ 缺少 {key}")
        
        # 检查轨道数据
        if 'orbit_data' in ycfg and 'orbit_points' in ycfg['orbit_data']:
            orbit_points = ycfg['orbit_data']['orbit_points']
            print(f"\n轨道数据包含 {len(orbit_points)} 个点")
            
            # 测试时间转换
            if orbit_points:
                first_point = orbit_points[0]
                time_str = first_point['time']
                print(f"第一个轨道点的时间: {time_str}")
                
                # 转换时间字符串为 datetime 对象
                dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                print(f"转换后的时间: {dt}")
        
        # 检查元数据
        if 'metadata' in ycfg:
            metadata = ycfg['metadata']
            print("\n元数据:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
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
