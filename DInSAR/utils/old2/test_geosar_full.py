#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 geosar.py 的完整功能
"""

import numpy as np
import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geosar import Geo2Rdr, Rdr2Geo, geo_to_sar, utm_to_sar

# 测试完整功能
def test_full_functionality():
    """测试 geosar.py 的完整功能"""
    print("测试 geosar.py 的完整功能...")
    
    # 路径设置
    yaml_file = "~/Temp/yaxia/output/master.yaml"
    # 扩展 ~ 为绝对路径
    yaml_file = os.path.expanduser(yaml_file)
    
    # 检查文件是否存在
    if not os.path.exists(yaml_file):
        print(f"错误: 文件 {yaml_file} 不存在")
        return False
    
    # 创建一个简单的 DEM 网格用于测试
    # 基于 master.yaml 中的 SAR 覆盖范围
    # 从之前的测试中，我们知道覆盖范围大约在 29.5-29.9 纬度，94.7-95.1 经度
    lat_min = 29.5
    lat_max = 29.9
    lon_min = 94.7
    lon_max = 95.1
    
    # 创建 DEM 网格
    lat_steps = 100
    lon_steps = 100
    dem_lat = np.linspace(lat_max, lat_min, lat_steps)
    dem_lon = np.linspace(lon_min, lon_max, lon_steps)
    DEM_lat, DEM_lon = np.meshgrid(dem_lat, dem_lon, indexing='ij')
    # 创建简单的高度数据（模拟地形）
    dem_h = np.zeros_like(DEM_lat)
    # 添加一些地形特征
    dem_h[20:80, 20:80] = 1000  # 中间有一个高地
    
    try:
        # 初始化 Geo2Rdr
        print("初始化 Geo2Rdr...")
        geo = Geo2Rdr(yaml_file, DEM_lat, DEM_lon, dem_h)
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
        if hasattr(geo, 'sar_bbox') and geo.sar_bbox:
            print(f"sar_bbox: {geo.sar_bbox}")
        
        # 测试左右视计算
        print("\n测试左右视计算...")
        look_dir = geo._calculate_look_dir()
        print(f"计算的 look_dir: {look_dir} (1: 右视, -1: 左视)")
        
        # 测试轨道插值
        print("\n测试轨道插值...")
        if hasattr(geo.orbit, 'interpolator') and geo.orbit.interpolator:
            print("轨道插值器初始化成功")
            # 测试位置计算
            t = 0.0
            pos = geo.orbit.position(t)
            print(f"t={t} 时的卫星位置: {pos}")
            # 测试速度计算
            vel = geo.orbit.velocity(t)
            print(f"t={t} 时的卫星速度: {vel}")
        else:
            print("警告: 轨道插值器未初始化")
        
        # 测试单个点的 DEM->SAR 映射
        print("\n测试单个点的 DEM->SAR 映射...")
        lat = 29.7
        lon = 94.9
        h = 500
        range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover = geo.geo2rdr_single(lat, lon, h)
        print(f"地理坐标 ({lat}, {lon}, {h}) 对应的 SAR 坐标: range={range_pixel:.2f}, azimuth={azimuth_pixel:.2f}")
        print(f"斜距: {slant_range:.2f} m, 方位时间: {azimuth_time:.6f} s")
        print(f"Shadow: {shadow}, Layover: {layover}")
        
        # 测试批量 DEM->SAR 映射
        print("\n测试批量 DEM->SAR 映射...")
        # 只测试一小部分点，避免计算时间过长
        small_dem_lat = DEM_lat[:10, :10]
        small_dem_lon = DEM_lon[:10, :10]
        small_dem_h = dem_h[:10, :10]
        # 创建临时 Geo2Rdr 实例
        small_geo = Geo2Rdr(yaml_file, small_dem_lat, small_dem_lon, small_dem_h)
        range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow_mask, layover_mask = small_geo.geo2rdr_batch(chunk_size=100, num_workers=2)
        print(f"批量映射完成，处理了 {len(range_pixel)} 个点")
        print(f"range_pixel 范围: {np.min(range_pixel):.2f} - {np.max(range_pixel):.2f}")
        print(f"azimuth_pixel 范围: {np.min(azimuth_pixel):.2f} - {np.max(azimuth_pixel):.2f}")
        print(f"阴影点数量: {np.sum(shadow_mask)}")
        print(f"叠掩点数量: {np.sum(layover_mask)}")
        
        # 测试 SAR->DEM 地理编码
        print("\n测试 SAR->DEM 地理编码...")
        rdr2geo = Rdr2Geo(geo, DEM_lat, DEM_lon, dem_h)
        # 测试单个 SAR 坐标
        sar_range = range_pixel[0]  # 取第一个点
        sar_azimuth = azimuth_pixel[0]
        lat_out, lon_out, h_out = rdr2geo.rdr2dem(sar_range, sar_azimuth)
        print(f"SAR 坐标 ({sar_range:.2f}, {sar_azimuth:.2f}) 对应的地理坐标: {lat_out:.6f}, {lon_out:.6f}, {h_out:.2f}")
        
        # 测试地理坐标到 SAR 坐标的转换
        print("\n测试地理坐标到 SAR 坐标的转换...")
        range_pixel2, azimuth_pixel2 = geo_to_sar(lat, lon, h, geo)
        print(f"地理坐标 ({lat}, {lon}, {h}) 对应的 SAR 坐标: range={range_pixel2:.2f}, azimuth={azimuth_pixel2:.2f}")
        
        # 测试 UTM 坐标到 SAR 坐标的转换
        print("\n测试 UTM 坐标到 SAR 坐标的转换...")
        # 注意：这里使用的 UTM 坐标是示例值，可能与实际位置不匹配
        easting = 500000
        northing = 3370000
        zone = 46
        hemisphere = 'N'
        try:
            range_pixel3, azimuth_pixel3 = utm_to_sar(easting, northing, zone, hemisphere, geo)
            print(f"UTM 坐标 ({easting}, {northing}, {zone}{hemisphere}) 对应的 SAR 坐标: range={range_pixel3:.2f}, azimuth={azimuth_pixel3:.2f}")
        except Exception as e:
            print(f"UTM 转换测试失败: {e}")
        
        print("\n所有测试完成!")
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_functionality()
    if success:
        print("\n测试通过!")
    else:
        print("\n测试失败!")
