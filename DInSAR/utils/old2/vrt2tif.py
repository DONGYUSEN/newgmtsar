#!/usr/bin/env python3
"""
多格式SAR数据转换为强度图JPG
功能：将VRT、TIFF等格式的复数数据转换为强度图，应用对数处理和95%区间拉伸
输入：文件名（可含后缀）
输出：对应的文件名_intensity.jpg
"""

import os
import numpy as np
from osgeo import gdal

# 设置GDAL异常处理
gdal.UseExceptions()

def vrt_to_jpg(input_name, output_dir=None):
    """
    将多格式SAR数据转换为带坐标信息的JPG文件
    
    Args:
        input_name: 输入文件名（可含后缀）
        output_dir: 输出目录（默认与输入文件同目录）
    """
    # 检测文件类型
    base_name, ext = os.path.splitext(input_name)
    ext = ext.lower()
    
    # 支持的文件格式
    supported_formats = ['.vrt', '.tif', '.tiff', '.img', '.hgt']
    
    # 确定输入文件路径
    if ext in supported_formats:
        # 如果已经包含后缀，直接使用
        input_file = input_name
        # 提取基础名称（不含后缀）
        base_name = os.path.basename(base_name)
    else:
        # 尝试添加支持的后缀
        input_file = None
        for fmt in supported_formats:
            test_file = f"{input_name}{fmt}"
            if os.path.exists(test_file):
                input_file = test_file
                base_name = os.path.basename(input_name)
                break
        
        # 如果没有找到文件
        if input_file is None:
            print(f"错误：输入文件不存在，尝试了以下格式: {supported_formats}")
            return False
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
        if output_dir == '':
            output_dir = '.'
    
    # 构建输出文件路径
    output_tif = os.path.join(output_dir, f"{base_name}_intensity.tif")
    
    print(f"处理文件: {input_file}")
    print(f"输出文件: {output_tif}")
    
    try:
        # 打开文件以获取地理信息和数据
        ds = gdal.Open(input_file)
        if not ds:
            print(f"错误：无法打开文件: {input_file}")
            return False
        
        # 获取地理变换和投影信息
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # 读取数据
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # 根据波段数处理数据
        if ds.RasterCount >= 2:
            # 多波段数据，假设是复数数据
            print(f"数据形状: {data.shape}, 类型: {data.dtype}")
            
            # 组合两个波段为复数
            real = data
            imag = ds.GetRasterBand(2).ReadAsArray()
            complex_data = real + 1j * imag
            
            # 计算强度（模的平方）
            intensity = np.abs(complex_data) # ** 2
            print(f"强度数据范围: 最小值={np.min(intensity):.4e}, 最大值={np.max(intensity):.4e}")
            
            # 应用对数处理（加一个小值避免log(0)）
            log_intensity = np.log1p(intensity)
            print(f"对数处理后范围: 最小值={np.min(log_intensity):.6e}, 最大值={np.max(log_intensity):.6e}")
            
            # 计算95%区间
            p5, p95 = np.percentile(log_intensity, [2, 98])
            print(f"95%区间: {p5:.4f} - {p95:.4f}")
            
            # 拉伸到0-255
            stretched = np.clip((log_intensity - p5) / (p95 - p5), 0, 1) * 255
            stretched = stretched.astype(np.uint8)
            print(f"拉伸后范围: 最小值={np.min(stretched)}, 最大值={np.max(stretched)}")
        else:
            # 单波段数据，处理可能的复数数据
            print(f"数据形状: {data.shape}, 类型: {data.dtype}")
            
            # 检查数据是否为复数类型
            if np.iscomplexobj(data):
                # 计算强度（模）
                intensity = np.abs(data)
                print(f"复数数据强度范围: 最小值={np.min(intensity):.2e}, 最大值={np.max(intensity):.2e}")
                
                # 应用对数处理（加一个小值避免log(0)）
                log_intensity = np.log1p(intensity)
                print(f"对数处理后范围: 最小值={np.min(log_intensity):.2f}, 最大值={np.max(log_intensity):.2f}")
                
                # 计算98%区间
                p5, p95 = np.percentile(log_intensity, [2, 98])
                print(f"95%区间: {p5:.2f} - {p95:.2f}")
                
                # 拉伸到0-255
                stretched = np.clip((log_intensity - p5) / (p95 - p5), 0, 1) * 255
                stretched = stretched.astype(np.uint8)
                print(f"拉伸后范围: 最小值={np.min(stretched)}, 最大值={np.max(stretched)}")
            else:
                # 实数数据，直接拉伸98%到0-255
                print(f"原始数据范围: 最小值={np.min(data):.2f}, 最大值={np.max(data):.2f}")
                
                # 计算98%区间
                # 处理大部分数据为0的情况
                non_zero_data = data[data > 0]
                if len(non_zero_data) > 0:
                    p1, p99 = np.percentile(non_zero_data, [1, 99])
                    print(f"非零数据98%区间: {p1:.2f} - {p99:.2f}")
                else:
                    # 如果所有数据都是0，使用固定范围
                    p1, p99 = 0, 1
                    print("所有数据都是0，使用固定范围: 0 - 1")
                
                # 拉伸到0-255
                if p99 > p1:
                    stretched = np.clip((data - p1) / (p99 - p1), 0, 1) * 255
                else:
                    # 避免除以零
                    stretched = np.zeros_like(data, dtype=np.uint8)
                    if np.max(data) > 0:
                        stretched = (data / np.max(data) * 255).astype(np.uint8)
                
                stretched = stretched.astype(np.uint8)
                print(f"拉伸后范围: 最小值={np.min(stretched)}, 最大值={np.max(stretched)}")
        
        # 关闭数据集
        ds = None
        
        # 创建输出文件
        height, width = stretched.shape
        
        # 尝试使用JPEG格式
        driver = gdal.GetDriverByName('GTiff')
        if driver:
            out_ds = driver.Create(output_tif, width, height, 1, gdal.GDT_Byte)
            if out_ds:
                # 复制地理变换和投影信息
                if geo_transform:
                    out_ds.SetGeoTransform(geo_transform)
                if projection:
                    out_ds.SetProjection(projection)
                
                # 写入数据
                out_band = out_ds.GetRasterBand(1)
                out_band.WriteArray(stretched)
                
                # 刷新缓存并关闭
                out_band.FlushCache()
                out_ds.FlushCache()
                out_ds = None
                
                print(f"成功生成带坐标信息的tiff文件: {output_tif}")
                return True
        
        # 如果JPEG失败，尝试使用TIFF格式
        print("JPEG格式创建失败，尝试使用TIFF格式...")
        # output_tif = output_jpg.replace('.png', '.tiff')
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_tif, width, height, 1, gdal.GDT_Byte)
        if out_ds:
            # 复制地理变换和投影信息
            if geo_transform:
                out_ds.SetGeoTransform(geo_transform)
            if projection:
                out_ds.SetProjection(projection)
            
            # 写入数据
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(stretched)
            
            # 刷新缓存并关闭
            out_band.FlushCache()
            out_ds.FlushCache()
            out_ds = None
            
            print(f"成功生成带坐标信息的TIFF文件: {output_tif}")
            return True
        
        # 如果所有格式都失败
        print(f"错误：无法创建输出文件")
        return False
        
    except Exception as e:
        print(f"处理失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多格式SAR数据转换为带坐标信息的tiff文件')
    parser.add_argument('input_name', help='输入文件名（可含后缀）')
    parser.add_argument('--output-dir', '-o', help='输出目录')
    
    args = parser.parse_args()
    
    print("=== 多格式SAR数据转换为带坐标信息的JPG文件 ===")
    print(f"输入文件: {args.input_name}")
    
    success = vrt_to_jpg(args.input_name, args.output_dir)
    
    if success:
        print("\n✅ 转换成功！")
    else:
        print("\n❌ 转换失败！")
        exit(1)

if __name__ == '__main__':
    main()
