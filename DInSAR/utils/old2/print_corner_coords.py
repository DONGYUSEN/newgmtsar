#!/usr/bin/env python3
"""
输出master.yaml文件中的corner_coords
"""

import yaml
from dem2sar import get_sar_image_corners, xyz_to_llh

# 读取master.yaml文件
yaml_file = "master.yaml"

print(f"读取文件: {yaml_file}")

# 测试get_sar_image_corners函数
print("\n获取corner_coords:")
corner_coords = get_sar_image_corners(yaml_file)
print(f"获取到的角点数量: {len(corner_coords) if corner_coords else 0}")

if corner_coords:
    print("\n角点坐标:")
    for i, coord in enumerate(corner_coords):
        # 转换为经纬度
        lat, lon, height = xyz_to_llh(coord[0], coord[1], coord[2])
        print(f"角点{i+1}: 地心坐标 = ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
        print(f"       经纬度 = ({lat:.6f