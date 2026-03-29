#!/usr/bin/env python3
"""
多视处理模块
功能：对SAR图像进行多视处理，降低分辨率，提高信噪比
参考ISCE2-2.6.4中的multilook算法实现
"""

import numpy as np
import yaml
import os
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
from osgeo import gdal, osr


def multilook(data: np.ndarray, nalks: int, nrlks: int, mean: bool = True, 
             data_type: str = 'complex64', boundary: str = 'crop') -> np.ndarray:
    """
    对数据进行多视处理（增强版）
    参考ISCE2-2.6.4的multilook算法实现
    
    Args:
        data: 输入数据数组，形状为 (长度, 宽度) 或 (长度, 宽度, 通道数)
        nalks: 方位向视数
        nrlks: 距离向视数
        mean: 是否取平均值，默认为True
        data_type: 数据类型，默认为'complex64'
        boundary: 边界处理方式
            - 'crop': 裁剪不能整除的部分（默认）
            - 'pad': 填充到可整除
            
    Returns:
        多视处理后的数据数组
    """
    # 验证数据类型
    expected_dtype = np.dtype(data_type)
    if data.dtype != expected_dtype:
        print(f"⚠️  警告: 数据类型不匹配")
        print(f"   期望: {expected_dtype}, 实际: {data.dtype}")
        # 不强制转换，保持原始类型
    
    # 获取输入数据形状
    if len(data.shape) == 3:
        # 多通道数据
        length, width, channels = data.shape
    else:
        # 单通道数据
        length, width = data.shape
        channels = 1
        data = data[..., np.newaxis]  # 添加通道维度
    
    # 计算多视后的尺寸
    length2 = length // nalks
    width2 = width // nrlks
    
    # 边界处理
    if boundary == 'crop':
        # 裁剪到可整除的尺寸
        if length % nalks != 0 or width % nrlks != 0:
            print(f"⚠️  警告: 输入数据尺寸 ({length}x{width}) 不能被视数 ({nalks}x{nrlks}) 整除")
            print(f"   将裁剪到: ({length2*nalks}x{width2*nrlks})")
        
        data_processed = data[:length2*nalks, :width2*nrlks, :]
        
    elif boundary == 'pad':
        # 填充到可整除的尺寸
        pad_length = (nalks - length % nalks) % nalks
        pad_width = (nrlks - width % nrlks) % nrlks
        
        if pad_length > 0 or pad_width > 0:
            print(f"⚠️  填充边界: 长度+{pad_length}, 宽度+{pad_width}")
            # 使用边缘值填充
            data_processed = np.pad(
                data,
                ((0, pad_length), (0, pad_width), (0, 0)),
                mode='edge'
            )
            length2 = (length + pad_length) // nalks
            width2 = (width + pad_width) // nrlks
        else:
            data_processed = data
    else:
        raise ValueError(f"未知的边界处理方式: {boundary}")
    
    # 使用NumPy向量化操作进行多视处理（更高效）
    # 方位向多视（垂直方向）
    # 将数据重塑为 (length2, nalks, width2*nrlks, channels)，然后在nalks维度求和
    data_azimuth = data_processed.reshape(length2, nalks, width2*nrlks, channels).sum(axis=1)
    
    # 距离向多视（水平方向）
    # 将数据重塑为 (length2, width2, nrlks, channels)，然后在nrlks维度求和
    result = data_azimuth.reshape(length2, width2, nrlks, channels).sum(axis=2)
    
    # 取平均值
    if mean:
        # 根据数据类型进行适当的除法
        if np.issubdtype(data.dtype, np.complexfloating):
            result = result / (nalks * nrlks)
        elif np.issubdtype(data.dtype, np.integer):
            # 整数类型使用整数除法
            result = result // (nalks * nrlks)
        else:
            result = result / (nalks * nrlks)
    
    # 移除通道维度（如果是单通道）
    if channels == 1:
        result = result.squeeze(axis=-1)
    
    return result


def multilook_v1(data: np.ndarray, nalks: int, nrlks: int, mean: bool = True, data_type: str = 'complex64') -> np.ndarray:
    """
    对数据进行多视处理（原地修改版本）
    
    Args:
        data: 输入数据数组，形状为 (长度, 宽度)
        nalks: 方位向视数
        nrlks: 距离向视数
        mean: 是否取平均值，默认为True
        data_type: 数据类型，默认为'complex64'
        
    Returns:
        多视处理后的数据数组
    """
    # 获取输入数据形状
    length, width = data.shape
    
    # 计算多视后的尺寸
    length2 = length // nalks
    width2 = width // nrlks
    
    # 确保输入数据尺寸能被视数整除
    if length % nalks != 0 or width % nrlks != 0:
        print(f"警告: 输入数据尺寸 ({length}x{width}) 不能被视数 ({nalks}x{nrlks}) 整除")
        print(f"将使用裁剪后的数据: ({length2*nalks}x{width2*nrlks})")
    
    # 方位向多视（垂直方向）
    for i in range(1, nalks):
        data[0:length2*nalks:nalks, :] += data[i:length2*nalks:nalks, :]
    
    # 距离向多视（水平方向）
    for i in range(1, nrlks):
        data[0:length2*nalks:nalks, 0:width2*nrlks:nrlks] += data[0:length2*nalks:nalks, i:width2*nrlks:nrlks]
    
    # 取平均值
    result = data[0:length2*nalks:nalks, 0:width2*nrlks:nrlks]
    if mean:
        # 根据数据类型进行适当的除法
        if np.issubdtype(data.dtype, np.complexfloating):
            result = result / (nalks * nrlks)
        elif np.issubdtype(data.dtype, np.integer):
            # 整数类型使用整数除法
            result = result // (nalks * nrlks)
        else:
            result = result / (nalks * nrlks)
    
    return result


def multilook_file(input_file: str, output_file: str, nalks: int, nrlks: int, data_type: str = 'complex64', input_yaml: Optional[str] = None, preserve_phase: bool = False) -> bool:
    """
    对文件进行多视处理
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        data_type: 数据类型，默认为'complex64'
        input_yaml: 输入YAML文件路径，用于读取图像尺寸信息
        preserve_phase: 是否使用相位保持多视处理（适用于SLC和干涉数据）
        
    Returns:
        处理是否成功
    """
    try:
        # 检查输入文件是否为VRT格式
        if input_file.lower().endswith('.vrt'):
            print(f"读取VRT文件: {input_file}")
            # 使用GDAL读取VRT文件
            ds = gdal.Open(input_file)
            if ds is None:
                raise Exception(f"无法打开VRT文件: {input_file}")
            
            band = ds.GetRasterBand(1)
            data = band.ReadAsArray()
            print(f"从VRT文件读取数据形状: {data.shape}")
            
            # 获取地理信息
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            ds = None  # 释放资源
        else:
            # 读取二进制文件
            print(f"读取文件: {input_file}")
            data = np.fromfile(input_file, dtype=data_type)
            
            # 确定数据形状
            if input_yaml:
                # 从YAML文件中读取尺寸信息
                yaml_data = read_yaml(input_yaml)
                if 'image_parameters' in yaml_data:
                    nrows = yaml_data['image_parameters'].get('nrows')
                    ncols = yaml_data['image_parameters'].get('ncols')
                    if nrows and ncols:
                        data = data.reshape(nrows, ncols)
                        print(f"从YAML文件读取数据形状: ({nrows}, {ncols})")
            
            # 如果没有从YAML文件中获取到尺寸信息，尝试自动确定
            if len(data.shape) == 1:
                # 尝试确定数据形状
                size = int(np.sqrt(len(data)))
                if size * size == len(data):
                    data = data.reshape(size, size)
                else:
                    # 尝试其他形状
                    for h in range(int(np.sqrt(len(data))), 0, -1):
                        if len(data) % h == 0:
                            w = len(data) // h
                            data = data.reshape(h, w)
                            break
        
        print(f"输入数据形状: {data.shape}")
        
        # 执行多视处理
        if preserve_phase and np.issubdtype(data.dtype, np.complexfloating):
            # 使用相位保持多视处理
            result = multilook_preserve_phase(data, nalks, nrlks, boundary='crop')
        else:
            # 使用常规多视处理
            result = multilook(data, nalks, nrlks, data_type=data_type, boundary='crop')
        print(f"多视后数据形状: {result.shape}")
        
        # 保存结果为TIFF文件
        save_as_tiff(output_file + '.tiff', result, data_type)
        print(f"保存结果到: {output_file}.tiff")
        
        # 生成VRT文件
        vrt_file = output_file + '.vrt'
        create_vrt_file(vrt_file, output_file + '.tiff', result.shape, data_type)
        print(f"生成VRT文件: {vrt_file}")
        
        return True
    except Exception as e:
        print(f"处理失败: {e}")
        return False

def multilook_chunked(input_file: str, output_file: str,
                     nalks: int, nrlks: int,
                     chunk_size: int = 1000,
                     data_type: str = 'complex64',
                     input_yaml: Optional[str] = None,
                     preserve_phase: bool = False) -> bool:
    """
    分块多视处理（内存优化版）

    适用于大型SAR图像，避免内存溢出。

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        chunk_size: 分块大小（行数），默认1000
        data_type: 数据类型
        input_yaml: 输入YAML文件路径
        preserve_phase: 是否使用相位保持多视处理（适用于SLC和干涉数据）

    Returns:
        处理是否成功
    """
    try:
        print(f"开始分块多视处理...")
        print(f"分块大小: {chunk_size} 行")

        # 检查输入文件是否为GDAL支持的格式（VRT、TIFF等）
        is_gdal_format = input_file.lower().endswith(('.vrt', '.tif', '.tiff', '.tiff'))
        
        if is_gdal_format:
            # 使用GDAL打开文件
            ds = gdal.Open(input_file)
            if ds is None:
                raise Exception(f"无法打开文件: {input_file}")

            band = ds.GetRasterBand(1)
            nrows = ds.RasterYSize
            ncols = ds.RasterXSize

            print(f"输入图像尺寸: {nrows} x {ncols}")

            # 复制地理参考信息
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
        else:
            # 处理二进制文件
            print(f"处理二进制文件: {input_file}")
            
            # 从YAML文件中读取尺寸信息
            if not input_yaml:
                raise Exception("处理二进制文件时需要提供input_yaml参数")
            
            yaml_data = read_yaml(input_yaml)
            if 'image_parameters' not in yaml_data:
                raise Exception("YAML文件中缺少image_parameters部分")
            
            nrows = yaml_data['image_parameters'].get('nrows')
            ncols = yaml_data['image_parameters'].get('ncols')
            
            if not nrows or not ncols:
                raise Exception("YAML文件中缺少nrows或ncols信息")
            
            print(f"从YAML文件读取图像尺寸: {nrows} x {ncols}")
            
            # 读取整个文件
            data = np.fromfile(input_file, dtype=data_type)
            data = data.reshape(nrows, ncols)
            
            geotransform = None
            projection = None

        # 计算输出尺寸
        out_nrows = nrows // nalks
        out_ncols = ncols // nrlks

        print(f"输出图像尺寸: {out_nrows} x {out_ncols}")

        # 确定GDAL数据类型
        gdal_type = gdal.GDT_CFloat32
        if data_type == 'complex64':
            gdal_type = gdal.GDT_CFloat32
        elif data_type == 'complex128':
            gdal_type = gdal.GDT_CFloat64
        elif data_type == 'float32':
            gdal_type = gdal.GDT_Float32
        elif data_type == 'float64':
            gdal_type = gdal.GDT_Float64
        elif data_type == 'int32':
            gdal_type = gdal.GDT_Int32

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_file, out_ncols, out_nrows, 1, gdal_type,
                              options=['COMPRESS=LZW', 'TILED=YES'])
        if out_ds is None:
            raise Exception(f"无法创建输出文件: {output_file}")

        out_band = out_ds.GetRasterBand(1)

        # 设置地理参考信息
        if geotransform is not None:
            # 调整地理变换参数
            new_geotransform = list(geotransform)
            new_geotransform[1] *= nrlks  # 像素宽度
            new_geotransform[5] *= nalks  # 像素高度（通常为负）
            out_ds.SetGeoTransform(tuple(new_geotransform))

        if projection:
            out_ds.SetProjection(projection)

        # 分块处理
        processed_rows = 0
        for i in range(0, nrows, chunk_size * nalks):
            # 读取分块
            chunk_rows = min(chunk_size * nalks, nrows - i)

            # 确保分块行数是nalks的倍数
            chunk_rows = (chunk_rows // nalks) * nalks
            if chunk_rows == 0:
                break

            if is_gdal_format:
                chunk_data = band.ReadAsArray(0, i, ncols, chunk_rows)
            else:
                # 从内存中读取分块
                chunk_data = data[i:i+chunk_rows, :]

            if chunk_data is None:
                print(f"⚠️  警告: 无法读取分块 [{i}:{i+chunk_rows}]")
                continue

            # 多视处理
            if preserve_phase and np.issubdtype(chunk_data.dtype, np.complexfloating):
                # 使用相位保持多视处理
                chunk_result = multilook_preserve_phase(chunk_data, nalks, nrlks, boundary='crop')
            else:
                # 使用常规多视处理
                chunk_result = multilook(chunk_data, nalks, nrlks, mean=True, data_type=data_type, boundary='crop')

            # 写入结果
            out_row = i // nalks
            out_band.WriteArray(chunk_result, 0, out_row)

            processed_rows += chunk_result.shape[0]
            progress = processed_rows / out_nrows * 100
            print(f"进度: {progress:.1f}% ({processed_rows}/{out_nrows} 行)")

        # 关闭文件
        if is_gdal_format:
            ds = None
        out_ds = None

        print(f"✓ 多视处理完成")
        print(f"✓ 输出文件: {output_file}")

        # 生成VRT文件
        vrt_file = output_file.replace('.tif', '.vrt').replace('.tiff', '.vrt')
        if not vrt_file.endswith('.vrt'):
            vrt_file = output_file + '.vrt'

        create_vrt_file(vrt_file, output_file, (out_nrows, out_ncols), data_type)
        print(f"✓ 生成VRT文件: {vrt_file}")

        return True

    except Exception as e:
        print(f"❌ 分块多视处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False



def calculate_output_size(input_size: Tuple[int, int], nalks: int, nrlks: int) -> Tuple[int, int]:
    """
    计算多视后的输出尺寸
    
    Args:
        input_size: 输入尺寸 (长度, 宽度)
        nalks: 方位向视数
        nrlks: 距离向视数
        
    Returns:
        输出尺寸 (长度, 宽度)
    """
    length, width = input_size
    return (length // nalks, width // nrlks)


def multilook_phase(phase_data: np.ndarray, nalks: int, nrlks: int, boundary: str = 'crop') -> np.ndarray:
    """
    对相位数据进行多视处理
    
    相位数据需要特殊处理，因为相位是周期性的（-π到π）
    这里使用矢量平均法处理相位，保留相位的方向信息
    
    Args:
        phase_data: 相位数据数组，形状为 (长度, 宽度) 或 (长度, 宽度, 通道数)
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 边界处理方式
            - 'crop': 裁剪不能整除的部分（默认）
            - 'pad': 填充到可整除
            
    Returns:
        多视处理后的相位数据数组
    """
    # 获取输入数据形状
    if len(phase_data.shape) == 3:
        # 多通道数据
        length, width, channels = phase_data.shape
    else:
        # 单通道数据
        length, width = phase_data.shape
        channels = 1
        phase_data = phase_data[..., np.newaxis]  # 添加通道维度
    
    # 计算多视后的尺寸
    length2 = length // nalks
    width2 = width // nrlks
    
    # 边界处理
    if boundary == 'crop':
        # 裁剪到可整除的尺寸
        if length % nalks != 0 or width % nrlks != 0:
            print(f"⚠️  警告: 输入数据尺寸 ({length}x{width}) 不能被视数 ({nalks}x{nrlks}) 整除")
            print(f"   将裁剪到: ({length2*nalks}x{width2*nrlks})")
        
        phase_processed = phase_data[:length2*nalks, :width2*nrlks, :]
        
    elif boundary == 'pad':
        # 填充到可整除的尺寸
        pad_length = (nalks - length % nalks) % nalks
        pad_width = (nrlks - width % nrlks) % nrlks
        
        if pad_length > 0 or pad_width > 0:
            print(f"⚠️  填充边界: 长度+{pad_length}, 宽度+{pad_width}")
            # 使用边缘值填充
            phase_processed = np.pad(
                phase_data,
                ((0, pad_length), (0, pad_width), (0, 0)),
                mode='edge'
            )
            length2 = (length + pad_length) // nalks
            width2 = (width + pad_width) // nrlks
        else:
            phase_processed = phase_data
    else:
        raise ValueError(f"未知的边界处理方式: {boundary}")
    
    # 将相位转换为复数表示（单位圆上的点）
    complex_repr = np.exp(1j * phase_processed)
    
    # 方位向多视（垂直方向）
    # 将数据重塑为 (length2, nalks, width2*nrlks, channels)，然后在nalks维度求和
    azimuth_sum = complex_repr.reshape(length2, nalks, width2*nrlks, channels).sum(axis=1)
    
    # 距离向多视（水平方向）
    # 将数据重塑为 (length2, width2, nrlks, channels)，然后在nrlks维度求和
    total_sum = azimuth_sum.reshape(length2, width2, nrlks, channels).sum(axis=2)
    
    # 计算平均相位（取复数的角度）
    mean_phase = np.angle(total_sum)
    
    # 移除通道维度（如果是单通道）
    if channels == 1:
        mean_phase = mean_phase.squeeze(axis=-1)
    
    return mean_phase


def multilook_preserve_phase(data: np.ndarray, nalks: int, nrlks: int, boundary: str = 'crop') -> np.ndarray:
    """
    相位保持多视处理
    
    对于SLC数据和干涉数据，分别处理振幅和相位，
    以更好地保留相位信息
    
    Args:
        data: 复数数据数组，形状为 (长度, 宽度) 或 (长度, 宽度, 通道数)
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 边界处理方式
            - 'crop': 裁剪不能整除的部分（默认）
            - 'pad': 填充到可整除
            
    Returns:
        多视处理后的复数数据数组
    """
    # 分离振幅和相位
    amplitude = np.abs(data)
    phase = np.angle(data)
    
    # 对振幅进行多视处理
    amp_ml = multilook(amplitude, nalks, nrlks, mean=True, data_type='float32', boundary=boundary)
    
    # 对相位进行多视处理
    phase_ml = multilook_phase(phase, nalks, nrlks, boundary=boundary)
    
    # 重新组合振幅和相位
    result = amp_ml * np.exp(1j * phase_ml)
    
    return result

def multilook_with_stats(data: np.ndarray, nalks: int, nrlks: int,
                        mean: bool = True, data_type: str = 'complex64',
                        preserve_phase: bool = False, boundary: str = 'crop') -> Tuple[np.ndarray, Dict]:
    """
    多视处理并返回统计信息

    Args:
        data: 输入数据
        nalks: 方位向视数
        nrlks: 距离向视数
        mean: 是否取平均
        data_type: 数据类型
        preserve_phase: 是否使用相位保持多视处理（适用于SLC和干涉数据）
        boundary: 边界处理方式

    Returns:
        (result, stats): 多视结果和统计信息字典
    """
    # 计算原始数据统计
    amplitude_orig = np.abs(data)
    mean_amp_orig = np.mean(amplitude_orig)
    std_amp_orig = np.std(amplitude_orig)
    snr_orig = (mean_amp_orig / std_amp_orig)**2 if std_amp_orig > 0 else float('inf')

    original_stats = {
        'mean_amplitude': float(mean_amp_orig),
        'std_amplitude': float(std_amp_orig),
        'enl_estimate': float(snr_orig),
        'shape': data.shape
    }

    # 执行多视
    if preserve_phase and np.issubdtype(data.dtype, np.complexfloating):
        # 使用相位保持多视处理
        result = multilook_preserve_phase(data, nalks, nrlks, boundary=boundary)
    else:
        # 使用常规多视处理
        result = multilook(data, nalks, nrlks, mean, data_type, boundary=boundary)

    # 计算多视后统计
    amplitude_ml = np.abs(result)
    mean_amp_ml = np.mean(amplitude_ml)
    std_amp_ml = np.std(amplitude_ml)
    snr_ml = (mean_amp_ml / std_amp_ml)**2 if std_amp_ml > 0 else float('inf')

    multilook_stats = {
        'mean_amplitude': float(mean_amp_ml),
        'std_amplitude': float(std_amp_ml),
        'enl_estimate': float(snr_ml),
        'shape': result.shape
    }

    # 计算改进
    theoretical_enl_gain = nalks * nrlks
    actual_enl_gain = snr_ml / snr_orig if snr_orig > 0 and snr_orig != float('inf') else 1.0

    # 确保对数计算的安全性
    enl_gain_db = 10 * np.log10(actual_enl_gain) if actual_enl_gain > 0 else 0.0
    theoretical_enl_gain_db = 10 * np.log10(theoretical_enl_gain) if theoretical_enl_gain > 0 else 0.0

    stats = {
        'original': original_stats,
        'multilook': multilook_stats,
        'improvement': {
            'enl_gain': float(actual_enl_gain),
            'enl_gain_db': float(enl_gain_db),
            'theoretical_enl_gain': int(theoretical_enl_gain),
            'theoretical_enl_gain_db': float(theoretical_enl_gain_db),
            'noise_reduction_percent': float((std_amp_orig - std_amp_ml) / std_amp_orig * 100) if std_amp_orig > 0 else 0.0,
            'efficiency': float(actual_enl_gain / theoretical_enl_gain * 100) if theoretical_enl_gain > 0 else 0.0
        },
        'parameters': {
            'nalks': nalks,
            'nrlks': nrlks,
            'total_looks': nalks * nrlks,
            'preserve_phase': preserve_phase
        }
    }

    return result, stats



def read_yaml(filename: str) -> Dict:
    """
    读取YAML文件
    
    Args:
        filename: YAML文件路径
        
    Returns:
        YAML文件内容
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(data: Dict, filename: str):
    """
    写入YAML文件
    
    Args:
        data: 要写入的数据
        filename: 输出文件路径
    """
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def save_as_tiff(output_file: str, data: np.ndarray, data_type: str = 'complex64'):
    """
    将数据保存为TIFF文件
    
    Args:
        output_file: 输出TIFF文件路径
        data: 输入数据数组
        data_type: 数据类型，默认为'complex64'
    """
    # 确定GDAL数据类型
    gdal_type = gdal.GDT_Float32
    if data_type == 'complex64':
        gdal_type = gdal.GDT_CFloat32
    elif data_type == 'int32':
        gdal_type = gdal.GDT_Int32
    elif data_type == 'float64':
        gdal_type = gdal.GDT_Float64
    elif data_type == 'complex128':
        gdal_type = gdal.GDT_CFloat64
    
    # 获取数据形状
    if len(data.shape) == 2:
        height, width = data.shape
        bands = 1
    else:
        height, width, bands = data.shape
    
    # 创建TIFF文件
    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        raise Exception("无法获取GTiff驱动")
    
    ds = driver.Create(output_file, width, height, bands, gdal_type)
    if ds is None:
        raise Exception(f"无法创建TIFF文件: {output_file}")
    
    # 写入数据
    if bands == 1:
        band = ds.GetRasterBand(1)
        band.WriteArray(data)
    else:
        for i in range(bands):
            band = ds.GetRasterBand(i + 1)
            band.WriteArray(data[:, :, i])
    
    # 关闭数据集
    ds = None


def create_vrt_file(vrt_file: str, data_file: str, shape: Tuple[int, int], data_type: str = 'complex64'):
    """
    创建VRT文件
    
    Args:
        vrt_file: VRT文件路径
        data_file: 数据文件路径
        shape: 数据形状 (高度, 宽度)
        data_type: 数据类型，默认为'complex64'
    """
    # 确定GDAL数据类型
    gdal_type = gdal.GDT_Float32
    if data_type == 'complex64':
        gdal_type = gdal.GDT_CFloat32
    elif data_type == 'int32':
        gdal_type = gdal.GDT_Int32
    elif data_type == 'float64':
        gdal_type = gdal.GDT_Float64
    elif data_type == 'complex128':
        gdal_type = gdal.GDT_CFloat64
    
    height, width = shape
    
    # 创建VRT数据集
    vrt_xml = f"""
<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <VRTRasterBand dataType="{gdal.GetDataTypeName(gdal_type)}" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{os.path.basename(data_file)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" />
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    
    # 写入VRT文件
    with open(vrt_file, 'w') as f:
        f.write(vrt_xml)


def update_yaml_for_multilook(input_yaml: str, output_yaml: str, nalks: int, nrlks: int) -> bool:
    """
    更新YAML文件参数以反映多视处理后的变化
    参考ISCE2的参数修正方法
    
    Args:
        input_yaml: 输入YAML文件路径
        output_yaml: 输出YAML文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        
    Returns:
        处理是否成功
    """
    try:
        # 读取输入YAML文件
        data = read_yaml(input_yaml)
        
        # 保存原始尺寸用于后续计算
        original_nrows = data.get('image_parameters', {}).get('nrows', 0)
        original_ncols = data.get('image_parameters', {}).get('ncols', 0)
        
        # 更新图像参数
        if 'image_parameters' in data:
            if 'nrows' in data['image_parameters']:
                data['image_parameters']['nrows'] = data['image_parameters']['nrows'] // nalks
            if 'ncols' in data['image_parameters']:
                data['image_parameters']['ncols'] = data['image_parameters']['ncols'] // nrlks
            # 更新像素尺寸（如果存在）
            if 'pixel_size' in data['image_parameters']:
                data['image_parameters']['pixel_size'] = data['image_parameters']['pixel_size'] * nrlks
            if 'azimuth_pixel_size' in data['image_parameters']:
                data['image_parameters']['azimuth_pixel_size'] = data['image_parameters']['azimuth_pixel_size'] * nalks
            if 'range_pixel_size' in data['image_parameters']:
                data['image_parameters']['range_pixel_size'] = data['image_parameters']['range_pixel_size'] * nrlks
            # 更新分辨率（如果存在）
            if 'resolution' in data['image_parameters']:
                data['image_parameters']['resolution'] = data['image_parameters']['resolution'] * max(nalks, nrlks)
        
        # 更新PRM参数（如果存在）
        if 'prm_parameters' in data:
            if 'num_lines' in data['prm_parameters']:
                data['prm_parameters']['num_lines'] = data['prm_parameters']['num_lines'] // nalks
            if 'num_rng_bins' in data['prm_parameters']:
                data['prm_parameters']['num_rng_bins'] = data['prm_parameters']['num_rng_bins'] // nrlks
            if 'nrows' in data['prm_parameters']:
                data['prm_parameters']['nrows'] = data['prm_parameters']['nrows'] // nalks
            if 'bytes_per_line' in data['prm_parameters']:
                data['prm_parameters']['bytes_per_line'] = data['prm_parameters']['bytes_per_line'] // nrlks
            if 'good_bytes_per_line' in data['prm_parameters']:
                data['prm_parameters']['good_bytes_per_line'] = data['prm_parameters']['good_bytes_per_line'] // nrlks
        
        # 更新雷达参数（如果存在）
        if 'radar_parameters' in data:
            if 'prf' in data['radar_parameters']:
                data['radar_parameters']['prf'] = data['radar_parameters']['prf'] / nalks
            if 'range_sampling_rate' in data['radar_parameters']:
                data['radar_parameters']['range_sampling_rate'] = data['radar_parameters']['range_sampling_rate'] / nrlks
            if 'azimuth_resolution' in data['radar_parameters']:
                data['radar_parameters']['azimuth_resolution'] = data['radar_parameters']['azimuth_resolution'] * nalks
            if 'range_resolution' in data['radar_parameters']:
                data['radar_parameters']['range_resolution'] = data['radar_parameters']['range_resolution'] * nrlks
            # 更新方位向和距离向间距（如果存在）
            if 'azimuth_spacing' in data['radar_parameters']:
                data['radar_parameters']['azimuth_spacing'] = data['radar_parameters']['azimuth_spacing'] * nalks
            if 'range_spacing' in data['radar_parameters']:
                data['radar_parameters']['range_spacing'] = data['radar_parameters']['range_spacing'] * nrlks
        
        # ===== 修复：轨道参数不应该改变 =====
        # 多视处理只是降采样，不改变成像时间范围和轨道
        # sensing_start和sensing_end保持不变
        # 轨道点也保持不变，因为：
        # 1. 轨道是卫星的真实轨迹，与图像分辨率无关
        # 2. 后续处理仍需要完整的轨道信息进行插值
        
        if 'orbit_parameters' in data:
            print("✓ 保持轨道参数不变（sensing_start, sensing_end, orbit_points）")
        
        # 更新角坐标（如果存在）
        if 'corner_coordinates' in data:
            # 角坐标不需要更新，因为多视处理不会改变图像的地理范围
            pass
        
        # 更新元数据（如果存在）
        if 'metadata' in data:
            # 更新数据文件名称（如果存在）
            if 'data_file' in data['metadata']:
                data_file = data['metadata']['data_file']
                if '.' in data_file:
                    name, ext = data_file.rsplit('.', 1)
                    data['metadata']['data_file'] = f"{name}_ml{nalks}x{nrlks}.{ext}"
            # 更新处理历史
            if 'processing_history' in data['metadata']:
                data['metadata']['processing_history'].append(f"Multilook processing: {nalks}x{nrlks}")
            else:
                data['metadata']['processing_history'] = [f"Multilook processing: {nalks}x{nrlks}"]
        
        # 添加多视处理信息
        if 'processing_parameters' not in data:
            data['processing_parameters'] = {}
        data['processing_parameters']['multilook'] = {
            'nalks': nalks,
            'nrlks': nrlks,
            'original_size': {
                'nrows': original_nrows,
                'ncols': original_ncols
            },
            'output_size': {
                'nrows': data['image_parameters'].get('nrows', 0),
                'ncols': data['image_parameters'].get('ncols', 0)
            },
            'pixel_size_adjustment': {
                'azimuth': nalks,
                'range': nrlks
            },
            'resolution_adjustment': {
                'azimuth': nalks,
                'range': nrlks
            }
        }
        
        # 写入更新后的YAML文件
        write_yaml(data, output_yaml)
        print(f"已生成更新后的YAML文件: {output_yaml}")
        return True
    except Exception as e:
        print(f"更新YAML文件失败: {e}")
        return False


def main():
    """
    主函数 - 命令行工具
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='多视处理工具')
    parser.add_argument('input_file', help='输入文件路径')
    parser.add_argument('output_file', help='输出文件路径')
    parser.add_argument('--nalks', type=int, default=1, help='方位向视数')
    parser.add_argument('--nrlks', type=int, default=1, help='距离向视数')
    parser.add_argument('--data-type', default='complex64', help='数据类型')
    parser.add_argument('--input-yaml', help='输入YAML文件路径')
    parser.add_argument('--output-yaml', help='输出YAML文件路径')
    parser.add_argument('--preserve-phase', action='store_true', help='使用相位保持多视处理（适用于SLC和干涉数据）')
    
    args = parser.parse_args()
    
    print("=== 多视处理工具 ===")
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"方位向视数: {args.nalks}")
    print(f"距离向视数: {args.nrlks}")
    print(f"数据类型: {args.data_type}")
    print(f"相位保持: {args.preserve_phase}")
    
    if args.input_yaml:
        print(f"输入YAML文件: {args.input_yaml}")
    if args.output_yaml:
        print(f"输出YAML文件: {args.output_yaml}")
    
    # 执行多视处理
    success = multilook_file(
        args.input_file,
        args.output_file,
        args.nalks,
        args.nrlks,
        args.data_type,
        args.input_yaml,
        args.preserve_phase
    )
    
    # 如果提供了YAML文件，生成更新后的YAML文件
    if success and args.input_yaml and args.output_yaml:
        yaml_success = update_yaml_for_multilook(
            args.input_yaml,
            args.output_yaml,
            args.nalks,
            args.nrlks
        )
        if not yaml_success:
            print("YAML文件更新失败！")
    
    if success:
        print("多视处理完成！")
    else:
        print("多视处理失败！")


if __name__ == '__main__':
    main()
