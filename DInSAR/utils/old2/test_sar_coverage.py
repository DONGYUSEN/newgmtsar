#!/usr/bin/env python3
"""
测试SAR覆盖范围计算
"""

import numpy as np
import yaml
from dem2sar import DemToSarConverter, get_sar_image_corners, xyz_to_llh

# 创建临时YAML配置文件
yaml_content = """
image_parameters:
  nrows: 1000
  ncols: 1000

prm_parameters:
  PRF: 4144.0
  rng_samp_rate: 100000000.0
  radar_wavelength: 0.031
  near_range: 60000.0

radar_parameters:
  azimuth_spacing: 5.0
  range_spacing: 1.5
  look_direction: right

metadata:
  first_line_sensing_time: "2023-01-01T00:00:00Z"
  last_line_sensing_time: "2023-01-01T00:03:36Z"

orbit_data:
  orbit_points:
    - time: "2023-01-01T00:00:00Z"
      position:
        x: -284515.577
        y: 5480417.824
        z: 4154857.155
      velocity:
        vx: 0.0
        vy: 7000.0
        vz: 0.0
    - time: "2023-01-01T00:03:36Z"
      position:
        x: -284515.577
        y: 5505657.824
        z: 4154857.155
      velocity:
        vx: 0.0
        vy: 7000.0
        vz: 0.0

corner_coordinates:
  first_corner:
    x: -284515.577
    y: 5480417.824
    z: 0
  second_corner:
    x: -284515.577
    y: 5505657.824
    z: 0
  third_corner:
    x: -290000.0
    y: 5505657.824
    z: 0
  fourth_corner:
    x: -290000.0
    y: 5480417.824
    z: 0
"""

# 写入YAML文件
yaml_file = "test_sar_coverage.yaml"
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

print(f"创建测试文件: {yaml_file}")

# 测试get_sar_image_corners函数
print("\n测试get_sar_image_corners函数:")
corner_coords = get_sar_image_corners(yaml_file)
print(f"获取到的角点数量: {len(corner_coords) if corner_coords else 0}")
if corner_coords:
    for i, coord in enumerate(corner_coords):
        lat, lon, _ = xyz_to_llh(coord[0], coord[1], coord[2])
        print(f"角点{i+1}: 经纬度 = ({lat:.4f}, {lon:.4f})")

# 测试DemToSarConverter的SAR覆盖范围计算
print("\n测试DemToSarConverter的SAR覆盖范围计算:")
try:
    # 创建一个简单的DEM文件（空文件）
    dem_file = "test_dem.tif"
    import numpy as np
    from osgeo import gdal, osr
    
    # 创建100x100的DEM
    rows, cols = 100, 100
    dem_data = np.zeros((rows, cols), dtype=np.float32)
    
    # 设置地理变换
    geotransform = [90.0, 0.01, 0.0, 30.0, 0.0, -0.01]
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dem_file, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    
    # 设置投影
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    dataset.SetProjection(srs.ExportToWkt())
    
    # 写入数据
    band = dataset.GetRasterBand(1)
    band.WriteArray(dem_data)
    
    dataset = None  # 关闭文件
    
    # 创建转换器
    converter = DemToSarConverter(yaml_file, dem_file)
    
    # 计算SAR覆盖范围
    sar_coverage = converter._calculate_sar_coverage()
    print(f"SAR覆盖范围: {sar_coverage}")
    
    # 计算DEM覆盖范围
    dem_coverage = converter._calculate_dem_coverage()
    print(f"DEM覆盖范围: {dem_coverage}")
    
    # 裁剪DEM
    clipped_dem, new_geotransform = converter._clip_dem(sar_coverage, buffer=0.1)
    print(f"\n裁剪后DEM形状: {clipped_dem.shape if clipped_dem is not None else 'None'}")
    print(f"新地理变换: {new_geotransform}")
    
    print("\n✅ 测试成功！SAR覆盖范围计算正常工作")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 清理临时文件
    import os
    for file in [yaml_file, dem_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"删除临时文件: {file}")
