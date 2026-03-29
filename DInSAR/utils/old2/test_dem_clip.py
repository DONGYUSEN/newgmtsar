#!/usr/bin/env python3
"""
测试DEM范围判断和裁剪功能
"""

import numpy as np
import yaml
from pathlib import Path
from dem2sar import DemToSarConverter

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
"""

# 写入YAML文件
yaml_file = "test_config.yaml"
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

# 创建临时DEM文件
import numpy as np
from osgeo import gdal, osr

# 创建1000x1000的DEM
rows, cols = 1000, 1000
dem_data = np.zeros((rows, cols), dtype=np.float32)

# 设置地理变换（左上角坐标，像素大小）
geotransform = [100.0, 0.0001, 0.0, 40.0, 0.0, -0.0001]

# 创建输出文件
driver = gdal.GetDriverByName('GTiff')
dem_file = "test_dem.tif"
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

print(f"创建测试文件: {yaml_file}, {dem_file}")

# 测试DEM范围判断和裁剪功能
try:
    converter = DemToSarConverter(yaml_file, dem_file)
    
    # 计算SAR覆盖范围
    sar_coverage = converter._calculate_sar_coverage()
    print(f"\nSAR覆盖范围: {sar_coverage}")
    
    # 计算DEM覆盖范围
    dem_coverage = converter._calculate_dem_coverage()
    print(f"DEM覆盖范围: {dem_coverage}")
    
    # 裁剪DEM
    clipped_dem, new_geotransform = converter._clip_dem(sar_coverage, buffer=0.1)
    print(f"\n裁剪后DEM形状: {clipped_dem.shape if clipped_dem is not None else 'None'}")
    print(f"新地理变换: {new_geotransform}")
    
    # 测试生成SAR DEM
    output_file = "test_output.tif"
    print(f"\n测试生成SAR DEM...")
    result = converter.generate_sar_dem(output_file, method='bilinear')
    print(f"生成完成，结果形状: {result.shape}")
    
    print("\n✅ 测试成功！DEM范围判断和裁剪功能正常工作")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 清理临时文件
    import os
    for file in [yaml_file, dem_file, output_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"删除临时文件: {file}")
