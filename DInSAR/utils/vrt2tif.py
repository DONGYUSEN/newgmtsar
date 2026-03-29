#!/usr/bin/env python3
"""
多格式栅格数据转换为8位灰度TIF
功能：
1) 支持复数/整型/浮点等数据类型
2) 统一做95%线性拉伸到0-255
3) 若源数据包含坐标信息，输出时保留
输入：文件名（可含后缀）
输出：对应的文件名_intensity.tif
"""

import os
import numpy as np
from osgeo import gdal

# 设置GDAL异常处理
gdal.UseExceptions()

SUPPORTED_FORMATS = [".vrt", ".tif", ".tiff", ".img", ".hgt"]


def _resolve_input_file(input_name):
    """根据输入名称解析实际存在的文件路径与基础文件名。"""
    base_name, ext = os.path.splitext(input_name)
    ext = ext.lower()

    if ext in SUPPORTED_FORMATS:
        if not os.path.exists(input_name):
            raise FileNotFoundError(f"输入文件不存在: {input_name}")
        return input_name, os.path.basename(base_name)

    for fmt in SUPPORTED_FORMATS:
        test_file = f"{input_name}{fmt}"
        if os.path.exists(test_file):
            return test_file, os.path.basename(input_name)

    tried = ", ".join(SUPPORTED_FORMATS)
    raise FileNotFoundError(f"输入文件不存在，尝试后缀: {tried}")


def _has_real_imag_desc(band):
    """通过波段描述判断是否为实部/虚部语义。"""
    desc = (band.GetDescription() or "").strip().lower()
    if not desc:
        return False, False
    is_real = any(k in desc for k in ["real", "实部"])
    is_imag = any(k in desc for k in ["imag", "虚部", "imaginary"])
    return is_real, is_imag


def _read_source_array(ds):
    """
    读取主数据数组。
    优先级：
    1) 单波段复数 -> 直接读取
    2) 双波段且看起来是实部+虚部 -> 合成为复数
    3) 其他情况 -> 读取第一波段
    """
    band1 = ds.GetRasterBand(1)
    data1 = band1.ReadAsArray()
    if data1 is None:
        raise RuntimeError("无法读取第1波段数据")

    if np.iscomplexobj(data1):
        return data1, "complex(single_band)"

    if ds.RasterCount >= 2:
        band2 = ds.GetRasterBand(2)
        data2 = band2.ReadAsArray()
        if data2 is None:
            raise RuntimeError("无法读取第2波段数据")

        b1_real, b1_imag = _has_real_imag_desc(band1)
        b2_real, b2_imag = _has_real_imag_desc(band2)
        looks_like_real_imag = (b1_real and b2_imag) or (b1_imag and b2_real)

        # 为兼容常见SAR实/虚双波段文件，双波段默认按复数实虚部合成
        if ds.RasterCount == 2 or looks_like_real_imag:
            return data1.astype(np.float64) + 1j * data2.astype(np.float64), "complex(real_imag_2bands)"

    return data1, "real"


def _stretch_to_uint8(data, valid_mask=None, stretch_percent=95.0):
    """
    将输入数据线性拉伸到0-255（uint8）。
    - 复数先取幅值
    - 百分位拉伸区间按 stretch_percent 居中截取
    """
    if np.iscomplexobj(data):
        work = np.abs(data).astype(np.float64)
    else:
        work = data.astype(np.float64)

    finite_mask = np.isfinite(work)
    if valid_mask is None:
        mask = finite_mask
    else:
        mask = finite_mask & valid_mask

    out = np.zeros(work.shape, dtype=np.uint8)
    valid_values = work[mask]
    if valid_values.size == 0:
        return out, (0.0, 1.0)

    tail = (100.0 - float(stretch_percent)) / 2.0
    low_q = max(0.0, tail)
    high_q = min(100.0, 100.0 - tail)
    p_low, p_high = np.percentile(valid_values, [low_q, high_q])

    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        # 退化场景回退到min/max
        p_low = float(np.min(valid_values))
        p_high = float(np.max(valid_values))
        if p_high <= p_low:
            out[mask] = 255
            return out, (p_low, p_high)

    scaled = (work - p_low) / (p_high - p_low)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[mask] = (scaled[mask] * 255.0).astype(np.uint8)
    return out, (float(p_low), float(p_high))


def convert_to_tif(input_name, output_dir=None):
    """
    将多格式栅格数据转换为带坐标信息的8位灰度TIF文件
    
    Args:
        input_name: 输入文件名（可含后缀）
        output_dir: 输出目录（默认与输入文件同目录）
    """
    try:
        input_file, base_name = _resolve_input_file(input_name)
    except FileNotFoundError as exc:
        print(f"错误：{exc}")
        return False
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
        if output_dir == "":
            output_dir = "."
    
    # 构建输出文件路径
    output_tif = os.path.join(output_dir, f"{base_name}_intensity.tif")
    
    print(f"处理文件: {input_file}")
    print(f"输出文件: {output_tif}")
    
    try:
        ds = gdal.Open(input_file)
        if not ds:
            print(f"错误：无法打开文件: {input_file}")
            return False

        # 读取数据并判断数据模式
        source_data, data_mode = _read_source_array(ds)
        print(f"数据模式: {data_mode}")
        print(f"数据形状: {source_data.shape}, 类型: {source_data.dtype}")

        band1 = ds.GetRasterBand(1)
        nodata = band1.GetNoDataValue()
        valid_mask = None
        if nodata is not None and np.isfinite(nodata):
            if np.iscomplexobj(source_data):
                valid_mask = source_data != complex(nodata)
            else:
                valid_mask = source_data != nodata

        stretched, (p_low, p_high) = _stretch_to_uint8(
            source_data, valid_mask=valid_mask, stretch_percent=95.0
        )
        print(f"95%拉伸区间: {p_low:.6g} - {p_high:.6g}")
        print(f"输出范围: 最小值={stretched.min()}, 最大值={stretched.max()}")

        # 获取地理信息（若存在）
        geo_transform = ds.GetGeoTransform(can_return_null=True)
        projection = ds.GetProjection()
        gcp_count = ds.GetGCPCount()
        gcps = ds.GetGCPs() if gcp_count > 0 else []
        gcp_projection = ds.GetGCPProjection() if gcp_count > 0 else ""
        metadata = ds.GetMetadata() or {}

        height, width = stretched.shape
        driver = gdal.GetDriverByName('GTiff')
        if not driver:
            print("错误：找不到GTiff驱动")
            return False

        out_ds = driver.Create(
            output_tif,
            width,
            height,
            1,
            gdal.GDT_Byte,
            options=["COMPRESS=LZW", "TILED=YES"],
        )
        if not out_ds:
            print("错误：无法创建输出文件")
            return False

        # 复制地理信息
        if geo_transform is not None:
            out_ds.SetGeoTransform(geo_transform)
        if projection:
            out_ds.SetProjection(projection)
        if gcp_count > 0 and gcps:
            out_ds.SetGCPs(gcps, gcp_projection)
        if metadata:
            out_ds.SetMetadata(metadata)

        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(stretched)
        out_band.SetNoDataValue(0)
        out_band.FlushCache()
        out_ds.FlushCache()

        out_ds = None
        ds = None
        print(f"成功生成8位TIFF: {output_tif}")
        return True
        
    except Exception as e:
        print(f"处理失败: {e}")
        return False


def vrt_to_jpg(input_name, output_dir=None):
    """
    兼容旧接口（历史函数名保留）。
    实际输出仍为TIFF。
    """
    return convert_to_tif(input_name, output_dir)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多格式栅格数据转换为带坐标信息的8位tiff文件')
    parser.add_argument('input_name', help='输入文件名（可含后缀）')
    parser.add_argument('--output-dir', '-o', help='输出目录')
    
    args = parser.parse_args()
    
    print("=== 多格式栅格数据转换为带坐标信息的8位TIFF ===")
    print(f"输入文件: {args.input_name}")
    
    success = convert_to_tif(args.input_name, args.output_dir)
    
    if success:
        print("\n✅ 转换成功！")
    else:
        print("\n❌ 转换失败！")
        exit(1)

if __name__ == '__main__':
    main()
