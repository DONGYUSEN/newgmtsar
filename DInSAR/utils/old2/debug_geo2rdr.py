#!/usr/bin/env python3
"""调试脚本：检查geo2rdr返回的时间范围"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DInSAR.utils.dem2sar import DemToSarConverter

def main():
    yaml_file = "20231110.yaml"
    dem_file = "dem_latlon.vrt"

    print("=== 调试geo2rdr时间范围 ===")

    # 创建converter
    converter = DemToSarConverter(yaml_file, dem_file)

    # 检查关键参数
    print(f"\n=== 关键参数 ===")
    print(f"sensing_start: {converter.sensing_start}")
    print(f"orbit_start_time: {getattr(converter, 'orbit_start_time', 'N/A')}")
    print(f"nrows: {converter.nrows}")
    print(f"ncols: {converter.ncols}")
    print(f"prf: {converter.prf}")
    print(f"azimuth_time_step: {converter.azimuth_time_step}")

    # 计算SAR持续时间
    sar_duration = converter.nrows * converter.azimuth_time_step
    print(f"\nSAR持续时间: {sar_duration:.2f}秒")

    # 检查轨道时间范围
    if converter.orbit_cache:
        orb_times = converter.orbit_cache['times']
        print(f"\n=== 轨道时间范围(相对) ===")
        print(f"最小: {orb_times[0]:.2f}")
        print(f"最大: {orb_times[-1]:.2f}")
        print(f"范围: {orb_times[-1] - orb_times[0]:.2f}秒")

    # 测试几个DEM点
    print(f"\n=== 测试geo2rdr转换 ===")
    dem_data = converter.dem_data
    dem_rows, dem_cols = dem_data.shape

    # 测试几个代表性的点
    test_points = [
        (0, 0),
        (dem_rows // 4, dem_cols // 4),
        (dem_rows // 2, dem_cols // 2),
        (3 * dem_rows // 4, 3 * dem_cols // 4),
        (dem_rows - 1, dem_cols - 1),
    ]

    az_times = []
    rng_samples = []

    for row, col in test_points:
        try:
            rng_sample, az_time = converter.convert(row, col)
            print(f"点({row}, {col}): range={rng_sample:.1f}, az_time={az_time:.2f}")
            az_times.append(az_time)
            rng_samples.append(rng_sample)
        except Exception as e:
            print(f"点({row}, {col}): 失败 - {e}")

    if az_times:
        print(f"\n=== 转换结果统计 ===")
        print(f"az_time范围: [{min(az_times):.2f}, {max(az_times):.2f}]")
        print(f"有效范围: [0, {sar_duration:.2f}]")

        # 检查有多少点在有效范围内
        valid_count = sum(1 for t in az_times if 0 <= t <= sar_duration)
        print(f"有效点比例: {valid_count}/{len(az_times)}")

if __name__ == '__main__':
    main()
