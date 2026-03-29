#!/usr/bin/env python3
"""
测试输出corner_coords
"""

import numpy as np
import yaml
from dem2sar import DemToSarConverter

# 创建模拟的master.yaml文件
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
  bottom_left: 
    lat: 29.61637752961 
    lon: 94.76472301434 
    x: 0.0 
    y: 14579.0 
  bottom_right: 
    lat: 29.56228388868 
    lon: 95.02012170766 
    x: 12543.0 
    y: 14579.0 
  top_left: 
    lat: 29.83848514853 
    lon: 94.82614434952 
    x: 0.0 
    y: 0.0 
  top_right: 
    lat: 29.78736169099 
    lon: 95.06766367008 
    x: 12543.0 
    y: 0.0
"""

# 写入YAML文件
yaml_file = "test_corner_coords.yaml"
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

print(f"创建测试文件: {yaml_file}")

# 创建一个简单的DEM文件
import numpy as np
from osgeo import gdal, osr

dem_file = "test_dem.tif"
rows, cols = 100, 100
dem_data = np.zeros((rows, cols), dtype=np.float32)
geotransform = [90.0, 0.01, 0.0, 30.0, 0.0, -0.01]

driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(dem_file, cols, rows, 1, gdal.GDT_Float32)
dataset.SetGeoTransform(geotransform)

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)  # WGS84
dataset.SetProjection(srs.ExportToWkt())

band = dataset.GetRasterBand(1)
band.WriteArray(dem_data)
dataset = None  # 关闭文件

print(f"创建DEM文件: {dem_file}")

# 测试DemToSarConverter
try:
    print("\n创建DemToSarConverter实例...")
    converter = DemToSarConverter(yaml_file, dem_file)
    print(f"\nlook_direction: {converter.look_direction}")
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

print("\n✅ 测试完成！")
