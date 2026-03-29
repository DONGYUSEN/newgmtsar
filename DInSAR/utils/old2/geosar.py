#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科研级 DEM → SAR 模拟 + 映射表输出 + SAR → DEM geocoding
功能：
- DEM -> SAR SLC 模拟
- 输出映射表 (DEM坐标 <-> SAR像素)
- SAR -> DEM geocoding
- 支持 Shadow/Layover、Zero-Doppler
- 支持 YAML 参数读取
- 支持轨道插值 (CUBIC/HERMITE)
- 支持 DEM 读取与裁剪
- 支持 SAR 数据到地理坐标映射
- 支持从 lat/lon 或 UTM 到 SAR 坐标系的转换
"""

import numpy as np
import h5py
import yaml
import argparse
from scipy.interpolate import RegularGridInterpolator, interp1d
from osgeo import gdal, osr
import concurrent.futures
import time
import os

# 导入Numba
from numba import njit, prange
from functools import lru_cache

# -------------------------------
#  全局轨道数据缓存
# -------------------------------
global_orbit_cache = {}

# -------------------------------
#  多进程处理函数
# -------------------------------
def process_chunk_vectorized(args):
    """处理一个数据块（用于多进程）"""
    try:
        chunk, xyz_array, t0, near_range, range_spacing, prf, orbit_config, wavelength = args
        start, end = chunk
        xyz_chunk = xyz_array[start:end]
        
        # 确保 xyz_chunk 是连续内存
        xyz_chunk = np.ascontiguousarray(xyz_chunk)
        
        # 限制块大小，防止内存溢出
        max_chunk_size = 5000
        if len(xyz_chunk) > max_chunk_size:
            # 对大块数据进行二次分块处理
            results = []
            for i in range(0, len(xyz_chunk), max_chunk_size):
                sub_end = min(i + max_chunk_size, len(xyz_chunk))
                sub_chunk = (start + i, start + sub_end)
                sub_args = (sub_chunk, xyz_array, t0, near_range, range_spacing, prf, orbit_config, wavelength)
                sub_result = process_chunk_vectorized(sub_args)
                results.append(sub_result)
            
            # 合并结果
            range_pix = np.concatenate([r[2] for r in results])
            az_pix = np.concatenate([r[3] for r in results])
            slant = np.concatenate([r[4] for r in results])
            az_time = np.concatenate([r[5] for r in results])
            shadow = np.concatenate([r[6] for r in results])
            layover = np.concatenate([r[7] for r in results])
            return start, end, range_pix, az_pix, slant, az_time, shadow, layover
        
        # 使用全局轨道缓存
        orbit_key = tuple(orbit_config['orbit_data'].flatten())
        if orbit_key not in global_orbit_cache:
            from geosar import OrbitSimulator
            global_orbit_cache[orbit_key] = OrbitSimulator(orbit_config)
        orbit = global_orbit_cache[orbit_key]
        
        # 估计初始时间
        t_initial = np.full(end - start, t0)
        
        # 求解 Zero-Doppler 时间（向量化）
        N = len(xyz_chunk)
        t_zd = np.copy(t_initial)
        
        # 向量化牛顿迭代，减少迭代次数
        for i in range(10):  # 减少迭代次数
            # 批量计算卫星位置和速度
            S = orbit.position(t_zd)
            V = orbit.velocity(t_zd)
            
            # 计算视线方向
            r = xyz_chunk - S
            r_norm = np.linalg.norm(r, axis=1)
            
            # 处理零范数情况
            zero_norm_mask = r_norm < 1e-10
            if zero_norm_mask.any():
                # 对零范数点使用初始时间
                t_zd[zero_norm_mask] = t_initial[zero_norm_mask]
                # 跳过这些点的计算
                valid_mask = ~zero_norm_mask
                if not np.any(valid_mask):
                    break
                # 只处理有效点
                r_valid = r[valid_mask]
                r_norm_valid = r_norm[valid_mask]
                r_unit_valid = r_valid / r_norm_valid[:, None]
                V_valid = V[valid_mask]
                t_zd_valid = t_zd[valid_mask]
                
                # 计算多普勒频率
                doppler_valid = -2 * np.sum(V_valid * r_unit_valid, axis=1) / wavelength
                
                # 计算多普勒频率的导数（解析解）
                try:
                    # 计算加速度（使用中心差分）
                    dt = 1e-6
                    t_minus = t_zd_valid - dt
                    t_plus = t_zd_valid + dt
                    V_minus = orbit.velocity(t_minus)
                    V_plus = orbit.velocity(t_plus)
                    A_valid = (V_plus - V_minus) / (2 * dt)
                    
                    # 计算第一项：加速度与视线方向的点积
                    term1 = np.sum(A_valid * r_unit_valid, axis=1)
                    
                    # 计算第二项：速度与视线方向变化率的点积
                    V_dot_r = np.sum(V_valid * r_unit_valid, axis=1)
                    term2 = np.sum(V_valid * (-V_valid + V_dot_r[:, None] * r_unit_valid), axis=1) / r_norm_valid
                    
                    # 计算多普勒频率导数
                    doppler_dot_valid = -2 / wavelength * (term1 + term2)
                except:
                    # 回退到数值导数
                    dt = 1e-6
                    t_plus = t_zd_valid + dt
                    S_plus = orbit.position(t_plus)
                    V_plus = orbit.velocity(t_plus)
                    r_plus = r_valid - (S_plus - S[valid_mask])
                    r_norm_plus = np.linalg.norm(r_plus, axis=1)
                    
                    # 处理零范数情况
                    r_norm_plus[r_norm_plus < 1e-10] = 1.0
                    r_unit_plus = r_plus / r_norm_plus[:, None]
                    doppler_plus = -2 * np.sum(V_plus * r_unit_plus, axis=1) / wavelength
                    doppler_dot_valid = (doppler_plus - doppler_valid) / dt
                
                # 处理零导数情况
                mask = np.abs(doppler_dot_valid) < 1e-10
                if mask.any():
                    # 对零导数的点使用简单处理
                    t_zd_valid[mask] = t_initial[valid_mask][mask]
                    doppler_dot_valid[mask] = 1.0  # 避免除零
                
                # 牛顿迭代
                t_new_valid = t_zd_valid - doppler_valid / doppler_dot_valid
                
                # 检查时间是否在有效范围内
                out_of_bounds = (t_new_valid < orbit.tmin) | (t_new_valid > orbit.tmax)
                if out_of_bounds.any():
                    t_zd_valid[out_of_bounds] = t_initial[valid_mask][out_of_bounds]
                    t_zd_valid[~out_of_bounds] = t_new_valid[~out_of_bounds]
                else:
                    t_zd_valid = t_new_valid
                
                # 更新结果
                t_zd[valid_mask] = t_zd_valid
                
                # 检查收敛
                if np.max(np.abs(t_new_valid - t_zd_valid)) < 1e-9:
                    break
            else:
                # 所有点都有效
                r_unit = r / r_norm[:, None]
                
                # 计算多普勒频率
                doppler = -2 * np.sum(V * r_unit, axis=1) / wavelength
                
                # 计算多普勒频率的导数（解析解）
                try:
                    # 计算加速度（使用中心差分）
                    dt = 1e-6
                    t_minus = t_zd - dt
                    t_plus = t_zd + dt
                    V_minus = orbit.velocity(t_minus)
                    V_plus = orbit.velocity(t_plus)
                    A = (V_plus - V_minus) / (2 * dt)
                    
                    # 计算第一项：加速度与视线方向的点积
                    term1 = np.sum(A * r_unit, axis=1)
                    
                    # 计算第二项：速度与视线方向变化率的点积
                    V_dot_r = np.sum(V * r_unit, axis=1)
                    term2 = np.sum(V * (-V + V_dot_r[:, None] * r_unit), axis=1) / r_norm
                    
                    # 计算多普勒频率导数
                    doppler_dot = -2 / wavelength * (term1 + term2)
                except:
                    # 回退到数值导数
                    dt = 1e-6
                    t_plus = t_zd + dt
                    S_plus = orbit.position(t_plus)
                    V_plus = orbit.velocity(t_plus)
                    r_plus = xyz_chunk - S_plus
                    r_norm_plus = np.linalg.norm(r_plus, axis=1)
                    
                    # 处理零范数情况
                    r_norm_plus[r_norm_plus < 1e-10] = 1.0
                    r_unit_plus = r_plus / r_norm_plus[:, None]
                    doppler_plus = -2 * np.sum(V_plus * r_unit_plus, axis=1) / wavelength
                    doppler_dot = (doppler_plus - doppler) / dt
                
                # 处理零导数情况
                mask = np.abs(doppler_dot) < 1e-10
                if mask.any():
                    # 对零导数的点使用简单处理
                    t_zd[mask] = t_initial[mask]
                    doppler_dot[mask] = 1.0  # 避免除零
                
                # 牛顿迭代
                t_new = t_zd - doppler / doppler_dot
                
                # 检查时间是否在有效范围内
                out_of_bounds = (t_new < orbit.tmin) | (t_new > orbit.tmax)
                if out_of_bounds.any():
                    t_zd[out_of_bounds] = t_initial[out_of_bounds]
                    t_zd[~out_of_bounds] = t_new[~out_of_bounds]
                else:
                    t_zd = t_new
                
                # 检查收敛
                if np.max(np.abs(t_new - t_zd)) < 1e-9:
                    break
        
        # 计算卫星位置和速度（向量化）
        S = orbit.position(t_zd)
        V = orbit.velocity(t_zd)
        
        # 计算斜距（向量化）
        r = xyz_chunk - S
        slant = np.linalg.norm(r, axis=1)
        
        # 计算距离向像素（向量化）
        range_pix = (slant - near_range) / range_spacing
        
        # 计算方位向时间和像素（向量化）
        az_time = t_zd - t0
        az_pix = az_time * prf
        
        # 计算 LOS 单位向量（缓存）
        r_unit = r / slant[:, None]
        
        # 简化的 Shadow/Layover 检测（向量化）
        V_r = np.sum(V * r_unit, axis=1)
        layover = V_r > 0
        
        # 计算仰角（缓存）
        radial_unit = xyz_chunk / np.linalg.norm(xyz_chunk, axis=1)[:, None]
        cos_theta = np.sum(r_unit * radial_unit, axis=1)
        theta = np.arccos(cos_theta)
        shadow = theta > np.pi / 2
        
        return start, end, range_pix, az_pix, slant, az_time, shadow, layover
    except Exception as e:
        # 捕获所有异常，确保进程不会崩溃
        import traceback
        print(f"处理数据块时出错: {e}")
        print(traceback.format_exc())
        # 返回空结果，避免进程池崩溃
        start, end = args[0]
        N = end - start
        return start, end, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)

def process_chunk_numba(args):
    """处理一个数据块（用于Numba多进程）"""
    try:
        chunk, xyz_array, t0, near_range, range_spacing, prf, orbit_config, wavelength = args
        start, end = chunk
        xyz_chunk = xyz_array[start:end]
        
        # 确保 xyz_chunk 是连续内存
        xyz_chunk = np.ascontiguousarray(xyz_chunk)
        
        # 限制块大小，防止内存溢出
        max_chunk_size = 5000
        if len(xyz_chunk) > max_chunk_size:
            # 对大块数据进行二次分块处理
            results = []
            for i in range(0, len(xyz_chunk), max_chunk_size):
                sub_end = min(i + max_chunk_size, len(xyz_chunk))
                sub_chunk = (start + i, start + sub_end)
                sub_args = (sub_chunk, xyz_array, t0, near_range, range_spacing, prf, orbit_config, wavelength)
                sub_result = process_chunk_numba(sub_args)
                results.append(sub_result)
            
            # 合并结果
            range_pix = np.concatenate([r[2] for r in results])
            az_pix = np.concatenate([r[3] for r in results])
            slant = np.concatenate([r[4] for r in results])
            az_time = np.concatenate([r[5] for r in results])
            shadow = np.concatenate([r[6] for r in results])
            layover = np.concatenate([r[7] for r in results])
            return start, end, range_pix, az_pix, slant, az_time, shadow, layover
        
        # 使用全局轨道缓存
        orbit_key = tuple(orbit_config['orbit_data'].flatten())
        if orbit_key not in global_orbit_cache:
            from geosar import OrbitSimulator
            global_orbit_cache[orbit_key] = OrbitSimulator(orbit_config)
        orbit = global_orbit_cache[orbit_key]
        
        # 估计初始时间
        t_initial = np.full(end - start, t0)
        
        # 求解 Zero-Doppler 时间（向量化）
        N = len(xyz_chunk)
        t_zd = np.copy(t_initial)
        
        # 向量化牛顿迭代，减少迭代次数
        for i in range(10):  # 减少迭代次数
            # 批量计算卫星位置和速度
            S = orbit.position(t_zd)
            V = orbit.velocity(t_zd)
            
            # 计算视线方向
            r = xyz_chunk - S
            r_norm = np.linalg.norm(r, axis=1)
            
            # 处理零范数情况
            zero_norm_mask = r_norm < 1e-10
            if zero_norm_mask.any():
                # 对零范数点使用初始时间
                t_zd[zero_norm_mask] = t_initial[zero_norm_mask]
                # 跳过这些点的计算
                valid_mask = ~zero_norm_mask
                if not np.any(valid_mask):
                    break
                # 只处理有效点
                r_valid = r[valid_mask]
                r_norm_valid = r_norm[valid_mask]
                r_unit_valid = r_valid / r_norm_valid[:, None]
                V_valid = V[valid_mask]
                t_zd_valid = t_zd[valid_mask]
                
                # 计算多普勒频率
                doppler_valid = -2 * np.sum(V_valid * r_unit_valid, axis=1) / wavelength
                
                # 计算多普勒频率的导数（解析解）
                try:
                    # 计算加速度（使用中心差分）
                    dt = 1e-6
                    t_minus = t_zd_valid - dt
                    t_plus = t_zd_valid + dt
                    V_minus = orbit.velocity(t_minus)
                    V_plus = orbit.velocity(t_plus)
                    A_valid = (V_plus - V_minus) / (2 * dt)
                    
                    # 计算第一项：加速度与视线方向的点积
                    term1 = np.sum(A_valid * r_unit_valid, axis=1)
                    
                    # 计算第二项：速度与视线方向变化率的点积
                    V_dot_r = np.sum(V_valid * r_unit_valid, axis=1)
                    term2 = np.sum(V_valid * (-V_valid + V_dot_r[:, None] * r_unit_valid), axis=1) / r_norm_valid
                    
                    # 计算多普勒频率导数
                    doppler_dot_valid = -2 / wavelength * (term1 + term2)
                except:
                    # 回退到数值导数
                    dt = 1e-6
                    t_plus = t_zd_valid + dt
                    S_plus = orbit.position(t_plus)
                    V_plus = orbit.velocity(t_plus)
                    r_plus = r_valid - (S_plus - S[valid_mask])
                    r_norm_plus = np.linalg.norm(r_plus, axis=1)
                    
                    # 处理零范数情况
                    r_norm_plus[r_norm_plus < 1e-10] = 1.0
                    r_unit_plus = r_plus / r_norm_plus[:, None]
                    doppler_plus = -2 * np.sum(V_plus * r_unit_plus, axis=1) / wavelength
                    doppler_dot_valid = (doppler_plus - doppler_valid) / dt
                
                # 处理零导数情况
                mask = np.abs(doppler_dot_valid) < 1e-10
                if mask.any():
                    # 对零导数的点使用简单处理
                    t_zd_valid[mask] = t_initial[valid_mask][mask]
                    doppler_dot_valid[mask] = 1.0  # 避免除零
                
                # 牛顿迭代
                t_new_valid = t_zd_valid - doppler_valid / doppler_dot_valid
                
                # 检查时间是否在有效范围内
                out_of_bounds = (t_new_valid < orbit.tmin) | (t_new_valid > orbit.tmax)
                if out_of_bounds.any():
                    t_zd_valid[out_of_bounds] = t_initial[valid_mask][out_of_bounds]
                    t_zd_valid[~out_of_bounds] = t_new_valid[~out_of_bounds]
                else:
                    t_zd_valid = t_new_valid
                
                # 更新结果
                t_zd[valid_mask] = t_zd_valid
                
                # 检查收敛
                if np.max(np.abs(t_new_valid - t_zd_valid)) < 1e-9:
                    break
            else:
                # 所有点都有效
                r_unit = r / r_norm[:, None]
                
                # 计算多普勒频率
                doppler = -2 * np.sum(V * r_unit, axis=1) / wavelength
                
                # 计算多普勒频率的导数（解析解）
                try:
                    # 计算加速度（使用中心差分）
                    dt = 1e-6
                    t_minus = t_zd - dt
                    t_plus = t_zd + dt
                    V_minus = orbit.velocity(t_minus)
                    V_plus = orbit.velocity(t_plus)
                    A = (V_plus - V_minus) / (2 * dt)
                    
                    # 计算第一项：加速度与视线方向的点积
                    term1 = np.sum(A * r_unit, axis=1)
                    
                    # 计算第二项：速度与视线方向变化率的点积
                    V_dot_r = np.sum(V * r_unit, axis=1)
                    term2 = np.sum(V * (-V + V_dot_r[:, None] * r_unit), axis=1) / r_norm
                    
                    # 计算多普勒频率导数
                    doppler_dot = -2 / wavelength * (term1 + term2)
                except:
                    # 回退到数值导数
                    dt = 1e-6
                    t_plus = t_zd + dt
                    S_plus = orbit.position(t_plus)
                    V_plus = orbit.velocity(t_plus)
                    r_plus = xyz_chunk - S_plus
                    r_norm_plus = np.linalg.norm(r_plus, axis=1)
                    
                    # 处理零范数情况
                    r_norm_plus[r_norm_plus < 1e-10] = 1.0
                    r_unit_plus = r_plus / r_norm_plus[:, None]
                    doppler_plus = -2 * np.sum(V_plus * r_unit_plus, axis=1) / wavelength
                    doppler_dot = (doppler_plus - doppler) / dt
                
                # 处理零导数情况
                mask = np.abs(doppler_dot) < 1e-10
                if mask.any():
                    # 对零导数的点使用简单处理
                    t_zd[mask] = t_initial[mask]
                    doppler_dot[mask] = 1.0  # 避免除零
                
                # 牛顿迭代
                t_new = t_zd - doppler / doppler_dot
                
                # 检查时间是否在有效范围内
                out_of_bounds = (t_new < orbit.tmin) | (t_new > orbit.tmax)
                if out_of_bounds.any():
                    t_zd[out_of_bounds] = t_initial[out_of_bounds]
                    t_zd[~out_of_bounds] = t_new[~out_of_bounds]
                else:
                    t_zd = t_new
                
                # 检查收敛
                if np.max(np.abs(t_new - t_zd)) < 1e-9:
                    break
        
        # 计算卫星位置和速度（向量化）
        S = orbit.position(t_zd)
        V = orbit.velocity(t_zd)
        
        # 使用 Numba 优化的计算
        from numba import njit, prange
        
        @njit(parallel=True, fastmath=True, cache=True)
        def geo2rdr_core_numba(xyz_chunk, S, V, near_range, range_spacing, wavelength):
            n = len(xyz_chunk)
            range_pix = np.empty(n)
            slant = np.empty(n)
            V_r = np.empty(n)
            theta = np.empty(n)
            
            for i in prange(n):
                # 计算斜距
                r = xyz_chunk[i] - S[i]
                slant[i] = np.linalg.norm(r)
                
                # 计算距离向像素
                range_pix[i] = (slant[i] - near_range) / range_spacing
                
                # 计算 LOS 单位向量
                r_unit = r / slant[i]
                
                # 计算卫星速度在视线方向的分量
                V_r[i] = np.dot(V[i], r_unit)
                
                # 计算仰角
                radial_unit = xyz_chunk[i] / np.linalg.norm(xyz_chunk[i])
                cos_theta = np.dot(r_unit, radial_unit)
                theta[i] = np.arccos(cos_theta)
            
            return range_pix, slant, V_r, theta
        
        range_pix, slant, V_r, theta = geo2rdr_core_numba(
            xyz_chunk, S, V, near_range, range_spacing, wavelength
        )
        
        # 计算方位向时间和像素（向量化）
        az_time = t_zd - t0
        az_pix = az_time * prf
        
        # 简化的 Shadow/Layover 检测
        layover = V_r > 0
        shadow = theta > np.pi / 2
        
        return start, end, range_pix, az_pix, slant, az_time, shadow, layover
    except Exception as e:
        # 捕获所有异常，确保进程不会崩溃
        import traceback
        print(f"处理数据块时出错: {e}")
        print(traceback.format_exc())
        # 返回空结果，避免进程池崩溃
        start, end = args[0]
        N = end - start
        return start, end, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)

# -------------------------------
#  坐标转换函数
# -------------------------------
from numba import njit

def llh_to_xyz(lat, lon, h):
    """LLH 到 ECEF 坐标转换"""
    # WGS84
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2-f)
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N*(1-e2) + h) * np.sin(lat)
    return np.stack([X,Y,Z], axis=-1)

@njit(fastmath=True, cache=True)
def llh_to_xyz_numba(lat, lon, h):
    """Numba 优化的 LLH 到 ECEF 坐标转换"""
    # WGS84
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2 - f)
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + h) * np.sin(lat)
    return np.array([X, Y, Z])

@njit(parallel=True, fastmath=True, cache=True)
def llh_to_xyz_vectorized_numba(lat_array, lon_array, h_array):
    """Numba 优化的向量化 LLH 到 ECEF 坐标转换"""
    n = len(lat_array)
    result = np.empty((n, 3))
    for i in prange(n):
        result[i] = llh_to_xyz_numba(lat_array[i], lon_array[i], h_array[i])
    return result

def xyz_to_llh(XYZ):
    """ECEF 到 LLH 坐标转换"""
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2-f)
    X = XYZ[:,0]
    Y = XYZ[:,1]
    Z = XYZ[:,2]
    lon = np.arctan2(Y,X)
    p = np.sqrt(X**2+Y**2)
    lat = np.arctan2(Z, p*(1-e2))
    for _ in range(5):
        N = a/np.sqrt(1-e2*np.sin(lat)**2)
        lat = np.arctan2(Z + e2*N*np.sin(lat), p)
    h = p/np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h

@njit(fastmath=True, cache=True)
def xyz_to_llh_numba(XYZ):
    """Numba 优化的 ECEF 到 LLH 坐标转换"""
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2 - f)
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]
    lon = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)
    lat = np.arctan2(Z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(Z + e2 * N * np.sin(lat), p)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h

@njit(parallel=True, fastmath=True, cache=True)
def xyz_to_llh_vectorized_numba(XYZ_array):
    """Numba 优化的向量化 ECEF 到 LLH 坐标转换"""
    n = len(XYZ_array)
    lat_array = np.empty(n)
    lon_array = np.empty(n)
    h_array = np.empty(n)
    for i in prange(n):
        lat, lon, h = xyz_to_llh_numba(XYZ_array[i])
        lat_array[i] = lat
        lon_array[i] = lon
        h_array[i] = h
    return lat_array, lon_array, h_array

def utm_to_llh(easting, northing, zone, hemisphere):
    """
    UTM 到 LLH 坐标转换
    """
    try:
        from osgeo import osr
        # 创建 UTM 坐标系
        utm_proj = osr.SpatialReference()
        utm_proj.SetUTM(zone, hemisphere == 'N')
        # 创建 WGS84 坐标系
        wgs84_proj = osr.SpatialReference()
        wgs84_proj.SetWellKnownGeogCS('WGS84')
        # 创建坐标转换
        transform = osr.CoordinateTransformation(utm_proj, wgs84_proj)
        # 执行转换
        lon, lat, _ = transform.TransformPoint(easting, northing, 0)
        return lat, lon, 0
    except Exception as e:
        raise ValueError(f"UTM 坐标转换失败: {e}")

# -------------------------------
#  轨道插值
# -------------------------------
class OrbitInterpolator:
    """轨道插值类"""
    def __init__(self, orbit_data, method='HERMITE'):
        """
        初始化轨道插值器
        :param orbit_data: 轨道数据，格式为 [time, x, y, z, vx, vy, vz]
        :param method: 插值方法，'CUBIC' 或 'HERMITE'
        """
        try:
            # 验证轨道数据格式
            if not isinstance(orbit_data, np.ndarray):
                orbit_data = np.array(orbit_data)
            
            if orbit_data.shape[1] != 7:
                raise ValueError("轨道数据格式错误，应为 [time, x, y, z, vx, vy, vz]")
            
            self.times = orbit_data[:, 0]
            self.positions = orbit_data[:, 1:4]
            self.velocities = orbit_data[:, 4:7]
            self.method = method.upper()
            
            # 检查时间是否递增
            if not np.all(np.diff(self.times) > 0):
                raise ValueError("轨道数据时间必须递增")
            
            if self.method == 'CUBIC':
                self.pos_interpolators = [interp1d(self.times, self.positions[:, i], kind='cubic') for i in range(3)]
                self.vel_interpolators = [interp1d(self.times, self.velocities[:, i], kind='cubic') for i in range(3)]
            elif self.method == 'HERMITE':
                # 预处理 Hermite 插值所需的参数
                self._precompute_hermite()
            else:
                raise ValueError(f"不支持的插值方法: {method}")
            
            # 初始化缓存
            self._position_cache = {}
            self._velocity_cache = {}
        except Exception as e:
            raise ValueError(f"轨道插值器初始化失败: {e}")
    
    def _precompute_hermite(self):
        """预处理 Hermite 插值所需的参数"""
        self.n = len(self.times)
        self.h = np.diff(self.times)
        self.alpha = np.zeros(self.n)
        self.beta = np.zeros(self.n)
        self.c = np.zeros((self.n, 3))
        self.d = np.zeros((self.n, 3))
        
        for i in range(3):
            y = self.positions[:, i]
            yp = self.velocities[:, i]
            
            # 计算 alpha 和 beta
            if self.n > 2:
                self.alpha[1:-1] = 3 * (y[2:] - y[1:-1]) / self.h[1:] - 3 * (y[1:-1] - y[:-2]) / self.h[:-1]
            
            # 边界条件：使用速度作为边界导数
            if self.n > 1:
                self.alpha[0] = 3 * (y[1] - y[0]) / self.h[0] - yp[0]
                self.alpha[-1] = yp[-1] - 3 * (y[-1] - y[-2]) / self.h[-1]
            else:
                # 只有一个点时的处理
                self.alpha[0] = yp[0]
            
            # 解三对角方程组
            if self.n > 1:
                self.beta[0] = 2 * self.h[0]
                for j in range(1, self.n-1):
                    if self.beta[j-1] != 0:
                        self.beta[j] = 2 * (self.h[j-1] + self.h[j]) - self.h[j-1]**2 / self.beta[j-1]
                        self.alpha[j] = self.alpha[j] - self.h[j-1] * self.alpha[j-1] / self.beta[j-1]
                    else:
                        self.beta[j] = 1.0
                        self.alpha[j] = 0.0
                
                # 回代求解
                if self.beta[-1] != 0:
                    self.c[-1, i] = self.alpha[-1] / self.beta[-1]
                else:
                    self.c[-1, i] = 0.0
                
                for j in range(self.n-2, -1, -1):
                    if self.beta[j] != 0:
                        self.c[j, i] = (self.alpha[j] - self.h[j] * self.c[j+1, i]) / self.beta[j]
                    else:
                        self.c[j, i] = 0.0
                
                # 计算 d
                for j in range(self.n-1):
                    if self.h[j] != 0:
                        self.d[j, i] = (self.c[j+1, i] - self.c[j, i]) / (3 * self.h[j])
                    else:
                        self.d[j, i] = 0.0
    
    def position(self, t):
        """计算给定时间的卫星位置"""
        t = np.atleast_1d(t)
        n = len(t)
        pos = np.zeros((n, 3))
        
        # 检查缓存中是否有所有时间点
        cached_indices = []
        uncached_indices = []
        uncached_times = []
        
        for i, ti in enumerate(t):
            if ti in self._position_cache:
                cached_indices.append(i)
                pos[i] = self._position_cache[ti]
            else:
                uncached_indices.append(i)
                uncached_times.append(ti)
        
        # 处理未缓存的时间点
        if uncached_times:
            uncached_times = np.array(uncached_times)
            uncached_pos = np.zeros((len(uncached_times), 3))
            
            if self.method == 'CUBIC':
                # 向量化计算
                try:
                    uncached_pos[:, 0] = self.pos_interpolators[0](uncached_times)
                    uncached_pos[:, 1] = self.pos_interpolators[1](uncached_times)
                    uncached_pos[:, 2] = self.pos_interpolators[2](uncached_times)
                except:
                    # 回退到循环处理
                    for i, ti in enumerate(uncached_times):
                        idx = np.searchsorted(self.times, ti) - 1
                        idx = max(0, min(idx, len(self.h)-1))
                        try:
                            uncached_pos[i, 0] = self.pos_interpolators[0](ti)
                            uncached_pos[i, 1] = self.pos_interpolators[1](ti)
                            uncached_pos[i, 2] = self.pos_interpolators[2](ti)
                        except:
                            uncached_pos[i] = self.positions[idx]
            elif self.method == 'HERMITE':
                # 向量化计算
                idx = np.searchsorted(self.times, uncached_times) - 1
                idx = np.clip(idx, 0, len(self.h)-1)
                dt = uncached_times - self.times[idx]
                
                y = self.positions[idx]
                yp = self.velocities[idx]
                c = self.c[idx]
                d = self.d[idx]
                
                try:
                    uncached_pos = y + yp * dt[:, None] + c * dt[:, None]**2 + d * dt[:, None]**3
                    # 处理无效值
                    mask = np.isnan(uncached_pos).any(axis=1) | np.isinf(uncached_pos).any(axis=1)
                    if mask.any():
                        uncached_pos[mask] = self.positions[idx[mask]]
                except:
                    # 回退到循环处理
                    for i, ti in enumerate(uncached_times):
                        idx_i = np.searchsorted(self.times, ti) - 1
                        idx_i = max(0, min(idx_i, len(self.h)-1))
                        dt_i = ti - self.times[idx_i]
                        y_i = self.positions[idx_i]
                        yp_i = self.velocities[idx_i]
                        c_i = self.c[idx_i]
                        d_i = self.d[idx_i]
                        try:
                            pos_i = y_i + yp_i*dt_i + c_i*dt_i**2 + d_i*dt_i**3
                            if np.any(np.isnan(pos_i)) or np.any(np.isinf(pos_i)):
                                pos_i = y_i
                            uncached_pos[i] = pos_i
                        except:
                            uncached_pos[i] = y_i
            
            # 存储到缓存并填充结果
            for i, (idx, ti, p) in enumerate(zip(uncached_indices, uncached_times, uncached_pos)):
                self._position_cache[ti] = p
                pos[idx] = p
        
        return pos
    
    def velocity(self, t):
        """计算给定时间的卫星速度"""
        t = np.atleast_1d(t)
        n = len(t)
        vel = np.zeros((n, 3))
        
        # 检查缓存中是否有所有时间点
        cached_indices = []
        uncached_indices = []
        uncached_times = []
        
        for i, ti in enumerate(t):
            if ti in self._velocity_cache:
                cached_indices.append(i)
                vel[i] = self._velocity_cache[ti]
            else:
                uncached_indices.append(i)
                uncached_times.append(ti)
        
        # 处理未缓存的时间点
        if uncached_times:
            uncached_times = np.array(uncached_times)
            uncached_vel = np.zeros((len(uncached_times), 3))
            
            if self.method == 'CUBIC':
                # 向量化计算
                try:
                    uncached_vel[:, 0] = self.vel_interpolators[0](uncached_times)
                    uncached_vel[:, 1] = self.vel_interpolators[1](uncached_times)
                    uncached_vel[:, 2] = self.vel_interpolators[2](uncached_times)
                except:
                    # 回退到循环处理
                    for i, ti in enumerate(uncached_times):
                        idx = np.searchsorted(self.times, ti) - 1
                        idx = max(0, min(idx, len(self.h)-1))
                        try:
                            uncached_vel[i, 0] = self.vel_interpolators[0](ti)
                            uncached_vel[i, 1] = self.vel_interpolators[1](ti)
                            uncached_vel[i, 2] = self.vel_interpolators[2](ti)
                        except:
                            uncached_vel[i] = self.velocities[idx]
            elif self.method == 'HERMITE':
                # 向量化计算
                idx = np.searchsorted(self.times, uncached_times) - 1
                idx = np.clip(idx, 0, len(self.h)-1)
                dt = uncached_times - self.times[idx]
                
                yp = self.velocities[idx]
                c = self.c[idx]
                d = self.d[idx]
                
                try:
                    uncached_vel = yp + 2 * c * dt[:, None] + 3 * d * dt[:, None]**2
                    # 处理无效值
                    mask = np.isnan(uncached_vel).any(axis=1) | np.isinf(uncached_vel).any(axis=1)
                    if mask.any():
                        uncached_vel[mask] = self.velocities[idx[mask]]
                except:
                    # 回退到循环处理
                    for i, ti in enumerate(uncached_times):
                        idx_i = np.searchsorted(self.times, ti) - 1
                        idx_i = max(0, min(idx_i, len(self.h)-1))
                        dt_i = ti - self.times[idx_i]
                        yp_i = self.velocities[idx_i]
                        c_i = self.c[idx_i]
                        d_i = self.d[idx_i]
                        try:
                            vel_i = yp_i + 2*c_i*dt_i + 3*d_i*dt_i**2
                            if np.any(np.isnan(vel_i)) or np.any(np.isinf(vel_i)):
                                vel_i = yp_i
                            uncached_vel[i] = vel_i
                        except:
                            uncached_vel[i] = yp_i
            
            # 存储到缓存并填充结果
            for i, (idx, ti, v) in enumerate(zip(uncached_indices, uncached_times, uncached_vel)):
                self._velocity_cache[ti] = v
                vel[idx] = v
        
        return vel

# -------------------------------
#  轨道模拟器 (支持真实轨道数据)
# -------------------------------
class OrbitSimulator:
    def __init__(self, yaml_cfg):
        """
        初始化轨道模拟器
        :param yaml_cfg: YAML 配置
        """
        self.tmin = yaml_cfg.get('tmin', 0)
        self.tmax = yaml_cfg.get('tmax', 1000)
        
        # 检查是否有真实轨道数据
        if 'orbit_data' in yaml_cfg:
            orbit_data = np.array(yaml_cfg['orbit_data'])
            method = yaml_cfg.get('orbit_interpolation', 'HERMITE')
            self.interpolator = OrbitInterpolator(orbit_data, method)
        else:
            # 使用默认线性轨道
            self.interpolator = None
    
    def position(self, t):
        """计算卫星位置"""
        if self.interpolator:
            return self.interpolator.position(t)
        else:
            # 简单线性轨道示例
            t = np.atleast_1d(t)
            return np.stack([t*10, t*0+7000000, t*0+500000], axis=-1)
    
    def velocity(self, t):
        """计算卫星速度"""
        if self.interpolator:
            return self.interpolator.velocity(t)
        else:
            # 简单线性轨道速度
            t = np.atleast_1d(t)
            return np.stack([np.ones_like(t)*10, np.zeros_like(t), np.zeros_like(t)], axis=-1)

# -------------------------------
#  DEM 读取与处理
# -------------------------------
def read_dem(dem_file, bbox=None):
    """
    读取 DEM 文件并可选裁剪
    :param dem_file: DEM 文件路径
    :param bbox: 边界框 [min_lon, min_lat, max_lon, max_lat]
    :return: dem_h, dem_lat, dem_lon
    """
    try:
        from osgeo import gdal, osr
        
        ds = gdal.Open(dem_file)
        if not ds:
            raise Exception(f"无法打开 DEM 文件: {dem_file}")
        
        dem_band = ds.GetRasterBand(1)
        dem_h = dem_band.ReadAsArray().astype(np.float32)
        gt = ds.GetGeoTransform()
        rows, cols = dem_h.shape
        
        # 生成原始坐标网格
        lats = np.linspace(gt[3], gt[3]+gt[5]*rows, rows)
        lons = np.linspace(gt[0], gt[0]+gt[1]*cols, cols)
        DEM_lat, DEM_lon = np.meshgrid(lats, lons, indexing='ij')
        
        # 如果指定了边界框，进行裁剪
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            
            # 找到边界框在 DEM 中的索引
            lon_mask = (DEM_lon >= min_lon) & (DEM_lon <= max_lon)
            lat_mask = (DEM_lat >= min_lat) & (DEM_lat <= max_lat)
            mask = lon_mask & lat_mask
            
            # 找到裁剪后的行列范围
            rows_idx = np.where(np.any(mask, axis=1))[0]
            cols_idx = np.where(np.any(mask, axis=0))[0]
            
            if len(rows_idx) > 0 and len(cols_idx) > 0:
                row_start, row_end = rows_idx[0], rows_idx[-1]+1
                col_start, col_end = cols_idx[0], cols_idx[-1]+1
                
                # 裁剪 DEM 和坐标
                dem_h = dem_h[row_start:row_end, col_start:col_end]
                DEM_lat = DEM_lat[row_start:row_end, col_start:col_end]
                DEM_lon = DEM_lon[row_start:row_end, col_start:col_end]
        
        return dem_h, DEM_lat, DEM_lon
    except ImportError:
        # 如果没有 GDAL，生成一个示例 DEM
        print("警告：GDAL 库未安装，生成示例 DEM")
        # 生成一个 100x100 的示例 DEM
        rows, cols = 100, 100
        dem_h = np.zeros((rows, cols), dtype=np.float32)
        # 生成坐标网格
        lats = np.linspace(30.0, 31.0, rows)
        lons = np.linspace(94.0, 95.0, cols)
        DEM_lat, DEM_lon = np.meshgrid(lats, lons, indexing='ij')
        # 添加一些地形起伏
        for i in range(rows):
            for j in range(cols):
                dem_h[i, j] = 1000 + 500 * np.sin(i/20) * np.cos(j/20)
        return dem_h, DEM_lat, DEM_lon

# -------------------------------
#  DEM -> SAR 映射计算
# -------------------------------
class Geo2Rdr:
    def __init__(self, yaml_file, dem_lat, dem_lon, dem_h):
        """
        初始化 Geo2Rdr 类
        :param yaml_file: YAML 配置文件
        :param dem_lat: DEM 纬度网格
        :param dem_lon: DEM 经度网格
        :param dem_h: DEM 高度数据
        """
        self.dem_lat = dem_lat
        self.dem_lon = dem_lon
        self.dem_h = dem_h
        
        # 读取 YAML 参数
        with open(yaml_file, 'r') as f:
            ycfg = yaml.safe_load(f)
        
        # 从 master.yaml 解析参数
        # 时间参数
        if 'metadata' in ycfg and 'first_line_sensing_time' in ycfg['metadata']:
            # 存储原始时间字符串
            self.t0_str = ycfg['metadata']['first_line_sensing_time']
            # 对于计算，我们使用相对于轨道数据起始时间的秒数
            self.t0 = 0.0  # 轨道数据已经相对于第一个时间点转换为秒
        else:
            self.t0 = ycfg.get('t0', 0)
        
        # 轨道数据
        orbit_data = []
        if 'orbit_data' in ycfg:
            if 'orbit_points' in ycfg['orbit_data']:
                orbit_points = ycfg['orbit_data']['orbit_points']
                if orbit_points:
                    import datetime
                    # 获取第一个时间点作为参考
                    t0 = datetime.datetime.fromisoformat(orbit_points[0]['time'].replace('Z', '+00:00'))
                    
                    for point in orbit_points:
                        time_str = point['time']
                        # 将时间字符串转换为秒
                        dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        # 转换为相对于第一个时间点的秒数
                        t = (dt - t0).total_seconds()
                        
                        pos = point['position']
                        vel = point['velocity']
                        orbit_data.append([t, pos['x'], pos['y'], pos['z'], vel['vx'], vel['vy'], vel['vz']])
        
        # 构建配置字典
        self.orbit_config = {
            'orbit_data': np.array(orbit_data),
            'tmin': 0.0,
            'tmax': orbit_data[-1][0] if orbit_data else 1000.0
        }
        
        # 系统参数（从 radar_parameters 部分读取）
        radar_params = ycfg.get('radar_parameters', {})
        self.prf = radar_params.get('prf', ycfg.get('prf', 1500.0))  # 先从 radar_parameters 读取，再从根级别读取
        self.near_range = radar_params.get('near_range', ycfg.get('near_range', 800000.0))  # 先从 radar_parameters 读取，再从根级别读取
        self.range_spacing = radar_params.get('range_pixel_spacing', ycfg.get('range_pixel_spacing', 6.25))  # 先从 radar_parameters 读取，再从根级别读取
        self.wavelength = radar_params.get('wavelength', ycfg.get('wavelength', 0.056))  # 先从 radar_parameters 读取，再从根级别读取
        self.look_dir = radar_params.get('look_dir', ycfg.get('look_dir', None))  # 先从 radar_parameters 读取，再从根级别读取
        # 升/降轨
        self.ascending = ycfg.get('orbit_ascending', True)
        # 轨道数据
        self.orbit = OrbitSimulator(self.orbit_config)
        # SAR 系统参数
        self.azimuth_pixel_spacing = ycfg.get('azimuth_pixel_spacing', 10.0)
        
        # 从 master.yaml 中提取 SAR 覆盖范围
        self.sar_bbox = None
        if 'corner_coordinates' in ycfg:
            corners = ycfg['corner_coordinates']
            # 提取所有角点的经纬度
            lats = []
            lons = []
            for corner_name, corner in corners.items():
                if 'lat' in corner and 'lon' in corner:
                    lats.append(corner['lat'])
                    lons.append(corner['lon'])
            
            if lats and lons:
                # 计算覆盖范围
                min_lat = min(lats)
                max_lat = max(lats)
                min_lon = min(lons)
                max_lon = max(lons)
                # 外扩 0.1 度
                self.sar_bbox = [min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1]
        
        # 从 YAML 配置文件中读取 SAR 图像尺寸（真实SAR的参数）
        self.azimuth_lines = None
        self.range_samples = None
        self.sar_data_format = None
        self.sar_byte_order = None
        self.sar_bands = None
        
        if 'image_parameters' in ycfg:
            # 读取 nrows 和 ncols（SAR图像大小）
            if 'nrows' in ycfg['image_parameters']:
                self.azimuth_lines = ycfg['image_parameters']['nrows']
            if 'ncols' in ycfg['image_parameters']:
                self.range_samples = ycfg['image_parameters']['ncols']
            # 读取数据格式信息
            if 'data_format' in ycfg['image_parameters']:
                self.sar_data_format = ycfg['image_parameters']['data_format']
            if 'byte_order' in ycfg['image_parameters']:
                self.sar_byte_order = ycfg['image_parameters']['byte_order']
            if 'bands' in ycfg['image_parameters']:
                self.sar_bands = ycfg['image_parameters']['bands']
        
        print(f"从 YAML 文件读取的真实 SAR 参数:")
        print(f"  方位向行数 (azimuth_lines): {self.azimuth_lines}")
        print(f"  距离向列数 (range_samples): {self.range_samples}")
        print(f"  数据格式 (data_format): {self.sar_data_format}")
        print(f"  字节序 (byte_order): {self.sar_byte_order}")
        print(f"  波段 (bands): {self.sar_bands}")
        
        # 初始 R 和 t0 值
        self.original_near_range = self.near_range
        self.original_t0 = self.t0
        
        # 预计算 ECEF 坐标
        self._precompute_ecef()
        
    def _precompute_ecef(self):
        """
        预计算 DEM → ECEF 坐标
        """
        print("预计算 DEM → ECEF 坐标...")
        
        # 检查SAR覆盖范围是否可用
        if self.sar_bbox is not None:
            print(f"SAR覆盖范围: {self.sar_bbox}")
            # 对DEM进行裁剪，只保留SAR覆盖范围内的数据
            min_lon, min_lat, max_lon, max_lat = self.sar_bbox
            
            # 创建掩码，只保留在SAR覆盖范围内的DEM点
            if self.dem_lat.ndim == 2:
                mask = (self.dem_lon >= min_lon) & (self.dem_lon <= max_lon) & \
                       (self.dem_lat >= min_lat) & (self.dem_lat <= max_lat)
            else:
                mask = (self.dem_lon >= min_lon) & (self.dem_lon <= max_lon) & \
                       (self.dem_lat >= min_lat) & (self.dem_lat <= max_lat)
            
            # 应用掩码
            self.lat_flat = self.dem_lat.flatten()[mask.flatten()]
            self.lon_flat = self.dem_lon.flatten()[mask.flatten()]
            self.h_flat = self.dem_h.flatten()[mask.flatten()]
            self.N = self.lat_flat.size
            
            print(f"DEM裁剪前点数: {self.dem_lat.flatten().size}")
            print(f"DEM裁剪后点数: {self.N}")
        else:
            # 展平整个DEM数据
            self.lat_flat = self.dem_lat.flatten()
            self.lon_flat = self.dem_lon.flatten()
            self.h_flat = self.dem_h.flatten()
            self.N = self.lat_flat.size
        
        # 一次性预计算 ECEF 坐标
        self.xyz_array = llh_to_xyz(self.lat_flat, self.lon_flat, self.h_flat)
    
    def _calculate_look_dir(self):
        """科研级稳健计算 SAR 左右视"""
        # 检查输入数据
        if self.dem_lat is None or self.dem_lon is None:
            return 1  # 默认右视
        
        # 计算 DEM 中心坐标
        lat_c = np.mean(self.dem_lat)
        lon_c = np.mean(self.dem_lon)
        
        # 计算轨道中点时间
        t_mid = (self.orbit.tmin + self.orbit.tmax) / 2
        
        # 获取卫星位置和速度
        try:
            S = self.orbit.position(t_mid)
            V = self.orbit.velocity(t_mid)
            
            # 确保形状正确
            S = S.reshape(1, 3)
            V = V.reshape(1, 3)
        except Exception as e:
            print(f"轨道计算错误: {e}")
            return 1  # 出错时返回默认值
        
        # 计算 DEM 中心的 ECEF 坐标
        try:
            dem_xyz = llh_to_xyz(lat_c, lon_c, 0)
            dem_xyz = dem_xyz.reshape(1, 3)
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return 1  # 出错时返回默认值
        
        # 计算视线方向
        dr = dem_xyz - S
        
        # 使用三重积计算左右视 (V × dr) · S
        # 这是 ISCE/GAMMA 内部使用的方法，更稳健
        cross_vec = np.cross(V, dr)
        side = np.sum(cross_vec * S, axis=1)
        
        # 确定左右视
        look_dir = -1 if side[0] > 0 else 1
        
        # 考虑升/降轨
        if not self.ascending:
            look_dir *= -1
        
        self.look_dir = look_dir
        return look_dir
    
    def _zero_doppler_time(self, xyz, t_initial, max_iter=20, tol=1e-9):
        """
        求解 Zero-Doppler 时间
        :param xyz: DEM 点的 ECEF 坐标
        :param t_initial: 初始时间估计
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :return: Zero-Doppler 时间
        """
        t = t_initial
        
        # 减少迭代次数，提高收敛速度
        for i in range(max_iter):
            # 计算卫星位置和速度
            S = self.orbit.position(t).reshape(3,)
            V = self.orbit.velocity(t).reshape(3,)
            
            # 计算视线方向
            r = xyz - S
            r_norm = np.linalg.norm(r)
            r_unit = r / r_norm
            
            # 计算多普勒频率
            doppler = -2 * np.dot(V, r_unit) / self.wavelength
            
            # 计算多普勒频率的导数（解析解，避免数值导数）
            # 解析导数公式：d(doppler)/dt = -2/λ * [ (A · r_unit) + (V · (-V + (V · r_unit) * r_unit) / r_norm) ]
            # 其中 A 是卫星加速度
            try:
                # 计算加速度（使用中心差分，更准确）
                dt = 1e-6
                V_minus = self.orbit.velocity(t - dt).reshape(3,)
                V_plus = self.orbit.velocity(t + dt).reshape(3,)
                A = (V_plus - V_minus) / (2 * dt)
                
                # 计算第一项：加速度与视线方向的点积
                term1 = np.dot(A, r_unit)
                
                # 计算第二项：速度与视线方向变化率的点积
                V_dot_r = np.dot(V, r_unit)
                term2 = np.dot(V, (-V + V_dot_r * r_unit)) / r_norm
                
                # 计算多普勒频率导数
                doppler_dot = -2 / self.wavelength * (term1 + term2)
            except:
                # 回退到数值导数
                dt = 1e-6
                S_plus = self.orbit.position(t + dt).reshape(3,)
                V_plus = self.orbit.velocity(t + dt).reshape(3,)
                r_plus = xyz - S_plus
                r_norm_plus = np.linalg.norm(r_plus)
                r_unit_plus = r_plus / r_norm_plus
                doppler_plus = -2 * np.dot(V_plus, r_unit_plus) / self.wavelength
                doppler_dot = (doppler_plus - doppler) / dt
            
            # 牛顿迭代
            if abs(doppler_dot) < 1e-10:
                # 使用二分法作为后备
                return self._zero_doppler_time_bisection(xyz, t_initial - 0.1, t_initial + 0.1, max_iter, tol)
            
            t_new = t - doppler / doppler_dot
            
            # 检查时间是否在有效范围内
            if t_new < self.orbit.tmin or t_new > self.orbit.tmax:
                # 使用二分法作为后备
                return self._zero_doppler_time_bisection(xyz, self.orbit.tmin, self.orbit.tmax, max_iter, tol)
            
            # 检查收敛
            if abs(t_new - t) < tol:
                return t_new
            
            t = t_new
        
        # 如果牛顿法不收敛，使用二分法
        return self._zero_doppler_time_bisection(xyz, self.orbit.tmin, self.orbit.tmax, max_iter, tol)
    
    def _zero_doppler_time_numba(self, xyz, t_initial, max_iter=20, tol=1e-9):
        """
        Numba 优化的 Zero-Doppler 时间求解
        :param xyz: DEM 点的 ECEF 坐标
        :param t_initial: 初始时间估计
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :return: Zero-Doppler 时间
        """
        # 获取轨道数据
        if hasattr(self.orbit, 'interpolator') and self.orbit.interpolator:
            times = self.orbit.interpolator.times
            positions = self.orbit.interpolator.positions
            velocities = self.orbit.interpolator.velocities
            method = self.orbit.interpolator.method
        else:
            # 使用默认轨道数据
            times = np.array([0, 1000])
            positions = np.array([[0, 7000000, 500000], [10000, 7000000, 500000]])
            velocities = np.array([[10, 0, 0], [10, 0, 0]])
            method = 'LINEAR'
        
        # 使用 Numba 优化的核心计算
        return zero_doppler_time_core_numba(
            xyz, t_initial, max_iter, tol, self.wavelength, 
            self.orbit.tmin, self.orbit.tmax, times, positions, velocities, method
        )

    def _zero_doppler_time_vectorized(self, xyz_array, t_initial_array, max_iter=20, tol=1e-9):
        """
        向量化求解 Zero-Doppler 时间
        :param xyz_array: DEM 点的 ECEF 坐标数组 (N, 3)
        :param t_initial_array: 初始时间估计数组 (N,)
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :return: Zero-Doppler 时间数组 (N,)
        """
        N = len(xyz_array)
        t_zd = np.copy(t_initial_array)
        
        # 确保 xyz_array 是连续内存
        xyz_array = np.ascontiguousarray(xyz_array)
        
        # 向量化牛顿迭代
        for i in range(max_iter):
            # 批量计算卫星位置和速度
            S = self.orbit.position(t_zd)
            V = self.orbit.velocity(t_zd)
            
            # 计算视线方向
            r = xyz_array - S
            r_norm = np.linalg.norm(r, axis=1)
            r_unit = r / r_norm[:, None]
            
            # 计算多普勒频率
            doppler = -2 * np.sum(V * r_unit, axis=1) / self.wavelength
            
            # 计算多普勒频率的导数（解析解，避免数值导数）
            try:
                # 计算加速度（使用中心差分）
                dt = 1e-6
                t_minus = t_zd - dt
                t_plus = t_zd + dt
                V_minus = self.orbit.velocity(t_minus)
                V_plus = self.orbit.velocity(t_plus)
                A = (V_plus - V_minus) / (2 * dt)
                
                # 计算第一项：加速度与视线方向的点积
                term1 = np.sum(A * r_unit, axis=1)
                
                # 计算第二项：速度与视线方向变化率的点积
                V_dot_r = np.sum(V * r_unit, axis=1)
                term2 = np.sum(V * (-V + V_dot_r[:, None] * r_unit), axis=1) / r_norm
                
                # 计算多普勒频率导数
                doppler_dot = -2 / self.wavelength * (term1 + term2)
            except:
                # 回退到数值导数
                dt = 1e-6
                t_plus = t_zd + dt
                S_plus = self.orbit.position(t_plus)
                V_plus = self.orbit.velocity(t_plus)
                r_plus = xyz_array - S_plus
                r_norm_plus = np.linalg.norm(r_plus, axis=1)
                r_unit_plus = r_plus / r_norm_plus[:, None]
                doppler_plus = -2 * np.sum(V_plus * r_unit_plus, axis=1) / self.wavelength
                doppler_dot = (doppler_plus - doppler) / dt
            
            # 处理零导数情况
            mask = np.abs(doppler_dot) < 1e-10
            if mask.any():
                # 对零导数的点使用批量二分法
                bad_indices = np.where(mask)[0]
                for j in bad_indices:
                    t_zd[j] = self._zero_doppler_time_bisection(
                        xyz_array[j], 
                        self.orbit.tmin, 
                        self.orbit.tmax, 
                        max_iter, 
                        tol
                    )
            
            # 牛顿迭代
            t_new = t_zd - doppler / doppler_dot
            
            # 检查时间是否在有效范围内
            out_of_bounds = (t_new < self.orbit.tmin) | (t_new > self.orbit.tmax)
            if out_of_bounds.any():
                # 对越界的点使用批量二分法
                bad_indices = np.where(out_of_bounds)[0]
                for j in bad_indices:
                    t_zd[j] = self._zero_doppler_time_bisection(
                        xyz_array[j], 
                        self.orbit.tmin, 
                        self.orbit.tmax, 
                        max_iter, 
                        tol
                    )
                # 对未越界的点更新时间
                t_zd[~out_of_bounds] = t_new[~out_of_bounds]
            else:
                t_zd = t_new
            
            # 检查收敛
            if np.max(np.abs(t_new - t_zd)) < tol:
                break
        
        return t_zd
    
    def _zero_doppler_time_vectorized_numba(self, xyz_array, t_initial_array, max_iter=20, tol=1e-9):
        """
        Numba 优化的向量化求解 Zero-Doppler 时间
        :param xyz_array: DEM 点的 ECEF 坐标数组 (N, 3)
        :param t_initial_array: 初始时间估计数组 (N,)
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :return: Zero-Doppler 时间数组 (N,)
        """
        N = len(xyz_array)
        t_zd = np.copy(t_initial_array)
        
        # 确保 xyz_array 是连续内存
        xyz_array = np.ascontiguousarray(xyz_array)
        
        # 向量化牛顿迭代
        for i in range(max_iter):
            # 批量计算卫星位置和速度
            S = self.orbit.position(t_zd)
            V = self.orbit.velocity(t_zd)
            
            # 计算加速度（使用中心差分）
            dt = 1e-6
            t_minus = t_zd - dt
            t_plus = t_zd + dt
            V_minus = self.orbit.velocity(t_minus)
            V_plus = self.orbit.velocity(t_plus)
            A = (V_plus - V_minus) / (2 * dt)
            
            # 计算视线方向
            r = xyz_array - S
            r_norm = np.linalg.norm(r, axis=1)
            r_unit = r / r_norm[:, None]
            
            # 使用 Numba 优化的核心计算（使用解析导数）
            doppler, doppler_dot = zero_doppler_vectorized_core_analytical_numba(
                xyz_array, S, V, A, r_unit, r_norm, self.wavelength
            )
            
            # 处理零导数情况
            mask = np.abs(doppler_dot) < 1e-10
            if mask.any():
                # 对零导数的点使用批量二分法
                bad_indices = np.where(mask)[0]
                for j in bad_indices:
                    t_zd[j] = self._zero_doppler_time_bisection_numba(
                        xyz_array[j], 
                        self.orbit.tmin, 
                        self.orbit.tmax, 
                        max_iter, 
                        tol
                    )
            
            # 牛顿迭代
            t_new = t_zd - doppler / doppler_dot
            
            # 检查时间是否在有效范围内
            out_of_bounds = (t_new < self.orbit.tmin) | (t_new > self.orbit.tmax)
            if out_of_bounds.any():
                # 对越界的点使用批量二分法
                bad_indices = np.where(out_of_bounds)[0]
                for j in bad_indices:
                    t_zd[j] = self._zero_doppler_time_bisection_numba(
                        xyz_array[j], 
                        self.orbit.tmin, 
                        self.orbit.tmax, 
                        max_iter, 
                        tol
                    )
                # 对未越界的点更新时间
                t_zd[~out_of_bounds] = t_new[~out_of_bounds]
            else:
                t_zd = t_new
            
            # 检查收敛
            if np.max(np.abs(t_new - t_zd)) < tol:
                break
        
        return t_zd
    
    def _zero_doppler_time_bisection(self, xyz, t_min, t_max, max_iter=50, tol=1e-9):
        """
        使用二分法求解 Zero-Doppler 时间
        :param xyz: DEM 点的 ECEF 坐标
        :param t_min: 时间范围最小值
        :param t_max: 时间范围最大值
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :return: Zero-Doppler 时间
        """
        # 计算边界点的多普勒频率
        S_min = self.orbit.position(t_min).reshape(3,)
        V_min = self.orbit.velocity(t_min).reshape(3,)
        r_min = xyz - S_min
        r_unit_min = r_min / np.linalg.norm(r_min)
        doppler_min = -2 * np.dot(V_min, r_unit_min) / self.wavelength
        
        S_max = self.orbit.position(t_max).reshape(3,)
        V_max = self.orbit.velocity(t_max).reshape(3,)
        r_max = xyz - S_max
        r_unit_max = r_max / np.linalg.norm(r_max)
        doppler_max = -2 * np.dot(V_max, r_unit_max) / self.wavelength
        
        # 检查是否存在零点
        if doppler_min * doppler_max > 0:
            # 如果没有零点，返回中间值
            return (t_min + t_max) / 2
        
        # 二分法迭代
        for i in range(max_iter):
            t_mid = (t_min + t_max) / 2
            
            S_mid = self.orbit.position(t_mid).reshape(3,)
            V_mid = self.orbit.velocity(t_mid).reshape(3,)
            r_mid = xyz - S_mid
            r_unit_mid = r_mid / np.linalg.norm(r_mid)
            doppler_mid = -2 * np.dot(V_mid, r_unit_mid) / self.wavelength
            
            if abs(doppler_mid) < tol:
                return t_mid
            
            if doppler_min * doppler_mid < 0:
                t_max = t_mid
                doppler_max = doppler_mid
            else:
                t_min = t_mid
                doppler_min = doppler_mid
        
        return (t_min + t_max) / 2
    
    def _zero_doppler_time_bisection_numba(self, xyz, t_min, t_max, max_iter=50, tol=1e-9):
        """
        Numba 优化的二分法求解 Zero-Doppler 时间
        :param xyz: DEM 点的 ECEF 坐标
        :param t_min: 时间范围最小值
        :param t_max: 时间范围最大值
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :return: Zero-Doppler 时间
        """
        # 使用 Numba 优化的核心计算
        return zero_doppler_time_bisection_core_numba(
            xyz, t_min, t_max, max_iter, tol, self.wavelength, self.orbit
        )
    
    def shadow_layover_batch(self, slant_range, dem_h, sat_height):
        """
        工程级 Shadow/Layover 检测
        使用逐像素投影+单调性约束方法
        
        :param slant_range: 斜距数据 (azimuth, range)
        :param dem_h: DEM 高度数据 (azimuth, range)
        :param sat_height: 卫星高度
        :return: shadow_mask, layover_mask
        """
        # 初始化掩码
        shadow_mask = np.zeros_like(dem_h, dtype=bool)
        layover_mask = np.zeros_like(dem_h, dtype=bool)
        
        # 计算 Layover：使用梯度法（更稳定）
        eps = 1e-6  # 阈值
        dR = np.diff(slant_range, axis=1)
        layover_mask[:, 1:] = dR < -eps
        
        # 计算仰角
        theta = compute_theta_numba(dem_h, sat_height, slant_range)
        
        # 计算 Shadow：完全向量化
        shadow_mask = compute_shadow_numba(theta)
        
        return shadow_mask, layover_mask
    
    def shadow_layover_batch_numba(self, slant_range, dem_h, sat_height):
        """
        Numba 优化的工程级 Shadow/Layover 检测
        :param slant_range: 斜距数据 (azimuth, range)
        :param dem_h: DEM 高度数据 (azimuth, range)
        :param sat_height: 卫星高度
        :return: shadow_mask, layover_mask
        """
        # 使用 Numba 优化的核心计算
        return shadow_layover_batch_core_numba(slant_range, dem_h, sat_height)
    
    def get_dem_index(self, lat, lon):
        """
        O(1) DEM 索引映射
        :param lat: 纬度
        :param lon: 经度
        :return: (lat_idx, lon_idx) 或 None
        """
        if self.dem_lat is None or self.dem_lon is None:
            return None
        
        # 计算分辨率
        if self.dem_lat.ndim == 2:
            lat0 = self.dem_lat[0, 0]
            lon0 = self.dem_lon[0, 0]
            if self.dem_lat.shape[0] > 1:
                lat_res = self.dem_lat[1, 0] - self.dem_lat[0, 0]
            else:
                lat_res = 0.01
            if self.dem_lon.shape[1] > 1:
                lon_res = self.dem_lon[0, 1] - self.dem_lon[0, 0]
            else:
                lon_res = 0.01
        else:
            lat0 = self.dem_lat[0]
            lon0 = self.dem_lon[0]
            if len(self.dem_lat) > 1:
                lat_res = self.dem_lat[1] - self.dem_lat[0]
            else:
                lat_res = 0.01
            if len(self.dem_lon) > 1:
                lon_res = self.dem_lon[1] - self.dem_lon[0]
            else:
                lon_res = 0.01
        
        # 避免除零错误
        if abs(lat_res) < 1e-10 or abs(lon_res) < 1e-10:
            return None
        
        # 计算索引
        lat_idx = int((lat - lat0) / lat_res)
        lon_idx = int((lon - lon0) / lon_res)
        
        # 检查索引是否在有效范围内
        if (0 <= lat_idx < self.dem_h.shape[0] and 
            0 <= lon_idx < self.dem_h.shape[1]):
            return (lat_idx, lon_idx)
        else:
            return None
    
    def compute_offset(self, sim_range_pixel, sim_azimuth_pixel, real_range_pixel, real_azimuth_pixel):
        """
        计算模拟SAR与真实SAR之间的偏移量（使用互相关方法）
        :param sim_range_pixel: 模拟SAR的距离向像素坐标（np.array）
        :param sim_azimuth_pixel: 模拟SAR的方位向像素坐标（np.array）
        :param real_range_pixel: 真实SAR的距离向像素坐标（np.array）
        :param real_azimuth_pixel: 真实SAR的方位向像素坐标（np.array）
        :return: (range_offset, azimuth_offset) 距离向和方位向的偏移量
        """
        # 步骤1：转换为numpy数组并校验输入维度一致
        sim_r = np.asarray(sim_range_pixel, dtype=np.float64)
        sim_a = np.asarray(sim_azimuth_pixel, dtype=np.float64)
        real_r = np.asarray(real_range_pixel, dtype=np.float64)
        real_a = np.asarray(real_azimuth_pixel, dtype=np.float64)

        if not (sim_r.shape == sim_a.shape == real_r.shape == real_a.shape):
            raise ValueError("所有输入坐标的维度必须一致！")
        
        # 步骤2：过滤无效值（空值、无穷值）
        valid_mask = np.isfinite(sim_r) & np.isfinite(sim_a) & \
                     np.isfinite(real_r) & np.isfinite(real_a)
        valid_num = np.sum(valid_mask)
        
        if valid_num < 3:  # 至少需要3个有效点计算偏移
            import warnings
            warnings.warn(f"有效匹配点仅{valid_num}个，不足以计算可靠偏移量")
            return 0.0, 0.0
        
        # 步骤3：使用互相关方法计算偏移量
        # 构建模拟SAR和真实SAR的二维直方图
        # 使用SAR图像尺寸
        sar_az_size = self.azimuth_lines or 3645
        sar_range_size = self.range_samples or 3136
        
        # 初始化直方图
        sim_hist = np.zeros((sar_az_size, sar_range_size), dtype=np.float32)
        real_hist = np.zeros((sar_az_size, sar_range_size), dtype=np.float32)
        
        # 填充直方图 - 使用NumPy向量化操作替代循环
        if valid_mask.ndim == 2:
            # 对于二维数组
            valid_sim_r = sim_r[valid_mask]
            valid_sim_a = sim_a[valid_mask]
            valid_real_r = real_r[valid_mask]
            valid_real_a = real_a[valid_mask]
        else:
            # 对于一维数组
            valid_sim_r = sim_r[valid_mask]
            valid_sim_a = sim_a[valid_mask]
            valid_real_r = real_r[valid_mask]
            valid_real_a = real_a[valid_mask]
        
        # 四舍五入并转换为整数
        sim_r_rounded = np.round(valid_sim_r).astype(int)
        sim_a_rounded = np.round(valid_sim_a).astype(int)
        real_r_rounded = np.round(valid_real_r).astype(int)
        real_a_rounded = np.round(valid_real_a).astype(int)
        
        # 过滤有效坐标
        sim_mask = (sim_r_rounded >= 0) & (sim_r_rounded < sar_range_size) & \
                   (sim_a_rounded >= 0) & (sim_a_rounded < sar_az_size)
        real_mask = (real_r_rounded >= 0) & (real_r_rounded < sar_range_size) & \
                    (real_a_rounded >= 0) & (real_a_rounded < sar_az_size)
        
        # 使用bincount填充直方图
        if np.any(sim_mask):
            sim_indices = sim_a_rounded[sim_mask] * sar_range_size + sim_r_rounded[sim_mask]
            sim_counts = np.bincount(sim_indices, minlength=sar_az_size * sar_range_size)
            sim_hist.flat[:len(sim_counts)] = sim_counts
        
        if np.any(real_mask):
            real_indices = real_a_rounded[real_mask] * sar_range_size + real_r_rounded[real_mask]
            real_counts = np.bincount(real_indices, minlength=sar_az_size * sar_range_size)
            real_hist.flat[:len(real_counts)] = real_counts
        
        # 计算互相关 - 使用FFT-based方法加速
        from scipy.signal import fftconvolve
        # 对直方图进行归一化
        sim_hist_norm = sim_hist / (np.sum(sim_hist) + 1e-10)
        real_hist_norm = real_hist / (np.sum(real_hist) + 1e-10)
        # 使用FFT卷积计算互相关
        corr = fftconvolve(sim_hist_norm, real_hist_norm[::-1, ::-1], mode='same')
        
        # 找到互相关的最大值位置
        max_idx = np.unravel_index(np.argmax(corr), corr.shape)
        shift_y = max_idx[0] - sar_az_size // 2
        shift_x = max_idx[1] - sar_range_size // 2
        
        # 步骤4：返回偏移量
        return shift_x, shift_y
    
    def update_geometry(self, range_offset, azimuth_offset):
        """
        根据偏移量更新几何参数
        :param range_offset: 距离向偏移（像素）
        :param azimuth_offset: 方位向偏移（像素）
        """
        # 更新距离向参数（near_range）
        self.near_range -= range_offset * self.range_spacing
        
        # 更新方位向参数（t0）
        self.t0 += azimuth_offset / self.prf
        
        print("Geometry updated:")
        print(f"Original near_range: {self.original_near_range}")
        print(f"Updated near_range: {self.near_range}")
        print(f"Original t0: {self.original_t0}")
        print(f"Updated t0: {self.t0}")
        print(f"Range offset: {range_offset} pixels")
        print(f"Azimuth offset: {azimuth_offset} pixels")
    
    def generate_sar_dem(self, range_pixel, azimuth_pixel, dem_h):
        """
        生成模拟SAR DEM
        :param range_pixel: 距离向像素坐标
        :param azimuth_pixel: 方位向像素坐标
        :param dem_h: DEM高度数据
        :return: sar_dem: 模拟SAR DEM
        """
        # 使用 Geo2Rdr 类中从 YAML 配置文件读取的 SAR 图像尺寸
        sar_az_size = self.azimuth_lines
        sar_range_size = self.range_samples
        
        # 如果从 YAML 配置文件中读取失败，使用计算值
        if sar_az_size is None or sar_range_size is None:
            sar_az_size = int(np.max(azimuth_pixel)) + 1
            sar_range_size = int(np.max(range_pixel)) + 1
        
        print(f"生成模拟SAR DEM，尺寸: {sar_az_size} x {sar_range_size}")
        
        # 创建SAR DEM数组
        sar_dem = np.full((sar_az_size, sar_range_size), np.nan, dtype=np.float32)
        
        # 将像素坐标四舍五入为整数
        r_rounded = np.round(range_pixel).astype(int)
        a_rounded = np.round(azimuth_pixel).astype(int)
        
        # 创建掩码，只保留在SAR图像尺寸范围内的像素
        mask = (r_rounded >= 0) & (r_rounded < sar_range_size) & \
               (a_rounded >= 0) & (a_rounded < sar_az_size)
        
        # 填充SAR DEM
        if np.any(mask):
            # 确保 dem_h 是正确的形状
            dem_h_flat = dem_h.flatten()
            # 确保 mask 长度与 dem_h_flat 长度匹配
            if len(mask) == len(dem_h_flat):
                sar_dem[a_rounded[mask], r_rounded[mask]] = dem_h_flat[mask]
                print(f"填充了 {np.sum(mask)} 个像素到SAR DEM")
            else:
                print("错误：mask 长度与 DEM 数据长度不匹配")
        else:
            print("没有像素在SAR图像尺寸范围内")
        
        return sar_dem
    
    def simulate_slc(self, range_pixel, azimuth_pixel, slant_range):
        """
        从模拟SAR数据生成复数据的SLC
        :param range_pixel: 距离向像素坐标
        :param azimuth_pixel: 方位向像素坐标
        :param slant_range: 斜距
        :return: 复数据的SLC
        """
        # 获取SAR图像尺寸
        sar_az_size = self.azimuth_lines or 3645
        sar_range_size = self.range_samples or 3136
        
        # 创建复数据数组
        slc = np.zeros((sar_az_size, sar_range_size), dtype=np.complex64)
        
        # 计算相位
        phase = 4 * np.pi * slant_range / self.wavelength
        
        # 生成复信号
        signal = np.exp(1j * phase)
        
        # 四舍五入像素坐标
        r_rounded = np.round(range_pixel).astype(int)
        a_rounded = np.round(azimuth_pixel).astype(int)
        
        # 创建掩码，只处理有效的像素坐标
        mask = (r_rounded >= 0) & (r_rounded < sar_range_size) & (a_rounded >= 0) & (a_rounded < sar_az_size)
        
        # 将信号赋值到对应的SAR像素位置
        slc[a_rounded[mask], r_rounded[mask]] = signal[mask]
        
        return slc
    
    def coarse_registration(self, sim_slc, real_sar_data):
        """
        模拟SAR与真实SAR的粗配准
        :param sim_slc: 模拟SAR的SLC数据
        :param real_sar_data: 真实SAR的数据
        :return: (range_offset, azimuth_offset) 距离向和方位向的偏移量
        """
        # 确保输入是幅度图像
        if np.iscomplexobj(sim_slc):
            sim_amp = np.abs(sim_slc)
        else:
            sim_amp = sim_slc
        
        if np.iscomplexobj(real_sar_data):
            real_amp = np.abs(real_sar_data)
        else:
            real_amp = real_sar_data
        
        # 使用FFT-based互相关（更高效）
        f1 = np.fft.fft2(sim_amp)
        f2 = np.fft.fft2(real_amp)
        
        # 计算交叉功率谱
        cross_power = f1 * np.conj(f2)
        cross_power /= np.abs(cross_power) + 1e-12
        
        # 逆FFT得到互相关
        corr = np.fft.ifft2(cross_power)
        corr = np.abs(corr)
        
        # 找到互相关的最大值位置
        max_idx = np.unravel_index(np.argmax(corr), corr.shape)
        
        # 计算偏移量
        shift_y = max_idx[0]
        shift_x = max_idx[1]
        
        # 处理循环位移
        if shift_y > sim_amp.shape[0] // 2:
            shift_y -= sim_amp.shape[0]
        
        if shift_x > sim_amp.shape[1] // 2:
            shift_x -= sim_amp.shape[1]
        
        return shift_x, shift_y
    
    def geo2rdr_single(self, lat, lon, h):
        """
        单个 DEM 点的 DEM->SAR 映射
        :param lat: 纬度
        :param lon: 经度
        :param h: 高度
        :return: range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover
        """
        # 转换为 ECEF 坐标
        xyz = llh_to_xyz_numba(lat, lon, h)
        
        # 估计初始时间
        t_initial = self.t0
        
        # 求解 Zero-Doppler 时间
        t_zd = self._zero_doppler_time(xyz, t_initial)
        
        # 计算卫星位置和速度
        S = self.orbit.position(t_zd).reshape(3,)
        V = self.orbit.velocity(t_zd).reshape(3,)
        
        # 计算斜距
        slant_range = np.linalg.norm(xyz - S)
        
        # 计算距离向像素
        range_pixel = (slant_range - self.near_range) / self.range_spacing
        
        # 计算方位向时间和像素
        azimuth_time = t_zd - self.t0
        azimuth_pixel = azimuth_time * self.prf
        
        # 检测 Shadow 和 Layover（简化版）
        # 计算卫星速度在视线方向的分量
        r = xyz - S
        r_norm = np.linalg.norm(r)
        r_unit = r / r_norm
        V_r = np.dot(V, r_unit)
        layover = V_r > 0
        
        # 简化的阴影检测
        # 计算视线方向与地表的夹角
        radial_unit = xyz / np.linalg.norm(xyz)
        cos_theta = np.dot(r_unit, radial_unit)
        theta = np.arccos(cos_theta)
        shadow = theta > np.pi / 2
        
        # 使用 O(1) DEM 索引查找
        dem_index = self.get_dem_index(lat, lon)
        if dem_index:
            # 可以在这里添加基于 DEM 的更准确检测
            pass
        
        return range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover
    
    @njit
    def _geo2rdr_single_numba(self, xyz, t_initial, near_range, range_spacing, prf, wavelength, orbit_tmin, orbit_tmax):
        """Numba 优化的单个 DEM 点的 DEM->SAR 映射"""
        # 这里需要注意：Numba 不支持直接调用 Python 类方法，所以需要重构
        # 由于 orbit 类方法无法在 Numba 中直接使用，这里只优化计算密集的部分
        # 实际使用时，需要将此函数与主函数配合使用
        pass
    
    def geo2rdr_batch(self, chunk_size=10000, num_workers=None):
        """
        批量 DEM->SAR 映射
        :param chunk_size: 批处理大小
        :param num_workers: 并行处理线程数，默认使用 CPU 核心数
        :return: range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow_mask, layover_mask
        """
        import os
        
        # 自动设置线程数为 CPU 核心数的 80%
        if num_workers is None:
            num_workers = max(1, int(os.cpu_count() * 0.8))  # 至少使用 1 个核心
        
        # 初始化输出数组
        range_pixel = np.zeros(self.N)
        azimuth_pixel = np.zeros(self.N)
        slant_range = np.zeros(self.N)
        azimuth_time = np.zeros(self.N)
        shadow_mask = np.zeros(self.N, dtype=bool)
        layover_mask = np.zeros(self.N, dtype=bool)
        
        # 自动计算左右视
        if self.look_dir is None:
            self._calculate_look_dir()
        
        # 批量处理函数
        def process_chunk(start, end):
            """处理一个数据块"""
            local_range = np.zeros(end - start)
            local_azimuth = np.zeros(end - start)
            local_slant = np.zeros(end - start)
            local_time = np.zeros(end - start)
            local_shadow = np.zeros(end - start, dtype=bool)
            local_layover = np.zeros(end - start, dtype=bool)
            
            # 处理当前块内的所有点
            for i in range(start, end):
                idx = i - start
                xyz = self.xyz_array[i].reshape(3,)
                
                # 估计初始时间
                t_initial = self.t0
                
                # 求解 Zero-Doppler 时间
                t_zd = self._zero_doppler_time(xyz, t_initial)
                
                # 计算卫星位置和速度
                S = self.orbit.position(t_zd).reshape(3,)
                V = self.orbit.velocity(t_zd).reshape(3,)
                
                # 计算斜距
                r = xyz - S
                slant = np.linalg.norm(r)
                
                # 计算距离向像素
                range_pix = (slant - self.near_range) / self.range_spacing
                
                # 计算方位向时间和像素
                az_time = t_zd - self.t0
                az_pix = az_time * self.prf
                
                # 简化的 Shadow/Layover 检测
                r_unit = r / slant
                V_r = np.dot(V, r_unit)
                layover = V_r > 0
                
                radial_unit = xyz / np.linalg.norm(xyz)
                cos_theta = np.dot(r_unit, radial_unit)
                theta = np.arccos(cos_theta)
                shadow = theta > np.pi / 2
                
                # 保存结果
                local_range[idx] = range_pix
                local_azimuth[idx] = az_pix
                local_slant[idx] = slant
                local_time[idx] = az_time
                local_shadow[idx] = shadow
                local_layover[idx] = layover
            
            return start, end, (local_range, local_azimuth, local_slant, local_time, local_shadow, local_layover)
        
        # 调整 chunk_size 以获得最佳性能
        if self.N < 100000:
            chunk_size = min(5000, self.N)
        elif self.N < 1000000:
            chunk_size = min(10000, self.N)
        else:
            chunk_size = min(20000, self.N)
        
        # 使用并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交任务
            futures = []
            for start in range(0, self.N, chunk_size):
                end = min(start + chunk_size, self.N)
                futures.append(executor.submit(process_chunk, start, end))
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                start, end, (r_list, a_list, s_list, t_list, shadow_list, layover_list) = future.result()
                range_pixel[start:end] = r_list
                azimuth_pixel[start:end] = a_list
                slant_range[start:end] = s_list
                azimuth_time[start:end] = t_list
                shadow_mask[start:end] = shadow_list
                layover_mask[start:end] = layover_list
        
        # 使用工程级 Shadow/Layover 检测方法重新计算
        # 重塑数据为 2D 数组
        slant_range_2d = slant_range.reshape(self.dem_lat.shape)
        dem_h_2d = self.dem_h
        
        # 计算卫星高度（使用轨道中点位置）
        t_mid = (self.orbit.tmin + self.orbit.tmax) / 2
        S_mid = self.orbit.position(t_mid).reshape(3,)
        # 计算卫星高度（从 ECEF 坐标转换）
        _, _, sat_height = xyz_to_llh(np.array([S_mid]))
        sat_height = sat_height[0]
        
        # 执行工程级检测
        shadow_mask_2d, layover_mask_2d = self.shadow_layover_batch(slant_range_2d, dem_h_2d, sat_height)
        
        # 转换回 1D 数组
        shadow_mask = shadow_mask_2d.flatten()
        layover_mask = layover_mask_2d.flatten()
        
        return range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow_mask, layover_mask
    
    def geo2rdr_batch_vectorized(self):
        """
        批量处理 - 向量化版本，使用多进程
        :return: range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow_mask, layover_mask
        """
        # 批量处理 - 分块处理以避免内存问题
        chunk_size = 10000
        chunks = [(start, min(start + chunk_size, self.N)) for start in range(0, self.N, chunk_size)]
        
        # 使用多进程处理，限制进程数为CPU核心数-1
        import os
        import concurrent.futures
        num_workers = max(1, os.cpu_count() - 1)  # CPU核心数-1
        print(f"使用 {num_workers} 个进程进行并行处理")
        
        # 确保 xyz_array 是连续内存
        self.xyz_array = np.ascontiguousarray(self.xyz_array)
        
        # 准备参数列表
        args_list = [(chunk, self.xyz_array, self.t0, self.near_range, self.range_spacing, self.prf, self.orbit_config, self.wavelength) for chunk in chunks]
        
        # 使用进程池
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_chunk_vectorized, args_list))
        
        # 初始化输出数组
        range_pixel = np.zeros(self.N)
        azimuth_pixel = np.zeros(self.N)
        slant_range = np.zeros(self.N)
        azimuth_time = np.zeros(self.N)
        shadow_mask = np.zeros(self.N, dtype=bool)
        layover_mask = np.zeros(self.N, dtype=bool)
        
        # 收集结果
        for start, end, r_pix, a_pix, slant, az_time, shadow, layover in results:
            range_pixel[start:end] = r_pix
            azimuth_pixel[start:end] = a_pix
            slant_range[start:end] = slant
            azimuth_time[start:end] = az_time
            shadow_mask[start:end] = shadow
            layover_mask[start:end] = layover
        
        # 裁剪模拟SAR数据，只保留在真实SAR图像尺寸范围内的像素
        if self.azimuth_lines is not None and self.range_samples is not None:
            print(f"裁剪模拟SAR数据到真实SAR尺寸: {self.azimuth_lines} x {self.range_samples}")
            # 创建掩码，只保留在真实SAR尺寸范围内的像素
            mask = (range_pixel >= 0) & (range_pixel < self.range_samples) & \
                   (azimuth_pixel >= 0) & (azimuth_pixel < self.azimuth_lines)
            
            # 应用掩码
            range_pixel = range_pixel[mask]
            azimuth_pixel = azimuth_pixel[mask]
            slant_range = slant_range[mask]
            azimuth_time = azimuth_time[mask]
            shadow_mask = shadow_mask[mask]
            layover_mask = layover_mask[mask]
            
            print(f"裁剪后模拟SAR数据点数: {len(range_pixel)}")
        
        return range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow_mask, layover_mask
    
    def geo2rdr_batch_numba(self):
        """
        批量处理 - Numba优化版本，使用多进程
        :return: range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow_mask, layover_mask
        """
        # 批量处理 - 分块处理以避免内存问题
        chunk_size = 10000
        chunks = [(start, min(start + chunk_size, self.N)) for start in range(0, self.N, chunk_size)]
        
        # 使用多进程处理，限制进程数为CPU核心数-1
        import os
        import concurrent.futures
        num_workers = max(1, os.cpu_count() - 1)  # CPU核心数-1
        print(f"使用 {num_workers} 个进程进行并行处理")
        
        # 确保 xyz_array 是连续内存
        self.xyz_array = np.ascontiguousarray(self.xyz_array)
        
        # 准备参数列表
        args_list = [(chunk, self.xyz_array, self.t0, self.near_range, self.range_spacing, self.prf, self.orbit_config, self.wavelength) for chunk in chunks]
        
        # 使用进程池
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_chunk_numba, args_list))
        
        # 初始化输出数组
        range_pixel = np.zeros(self.N)
        azimuth_pixel = np.zeros(self.N)
        slant_range = np.zeros(self.N)
        azimuth_time = np.zeros(self.N)
        shadow_mask = np.zeros(self.N, dtype=bool)
        layover_mask = np.zeros(self.N, dtype=bool)
        
        # 收集结果
        for start, end, r_pix, a_pix, slant, az_time, shadow, layover in results:
            range_pixel[start:end] = r_pix
            azimuth_pixel[start:end] = a_pix
            slant_range[start:end] = slant
            azimuth_time[start:end] = az_time
            shadow_mask[start:end] = shadow
            layover_mask[start:end] = layover
        
        # 裁剪模拟SAR数据，只保留在真实SAR图像尺寸范围内的像素
        if self.azimuth_lines is not None and self.range_samples is not None:
            print(f"裁剪模拟SAR数据到真实SAR尺寸: {self.azimuth_lines} x {self.range_samples}")
            # 创建掩码，只保留在真实SAR尺寸范围内的像素
            mask = (range_pixel >= 0) & (range_pixel < self.range_samples) & \
                   (azimuth_pixel >= 0) & (azimuth_pixel < self.azimuth_lines)
            
            # 应用掩码
            range_pixel = range_pixel[mask]
            azimuth_pixel = azimuth_pixel[mask]
            slant_range = slant_range[mask]
            azimuth_time = azimuth_time[mask]
            shadow_mask = shadow_mask[mask]
            layover_mask = layover_mask[mask]
            
            print(f"裁剪后模拟SAR数据点数: {len(range_pixel)}")
        
        return range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow_mask, layover_mask
    
    def geo2rdr_vectorized(self, lat_array, lon_array, h_array):
        """
        向量化 DEM->SAR 映射
        :param lat_array: 纬度数组
        :param lon_array: 经度数组
        :param h_array: 高度数组
        :return: range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover
        """
        # 一次性预计算 ECEF 坐标
        xyz_array = llh_to_xyz(lat_array, lon_array, h_array)
        N = xyz_array.shape[0]
        
        # 初始化输出数组
        range_pixel = np.zeros(N)
        azimuth_pixel = np.zeros(N)
        slant_range = np.zeros(N)
        azimuth_time = np.zeros(N)
        shadow = np.zeros(N, dtype=bool)
        layover = np.zeros(N, dtype=bool)
        
        # 向量化处理：批量求解 Zero-Doppler 时间
        t_initial_array = np.full(N, self.t0)
        t_zd = self._zero_doppler_time_vectorized(xyz_array, t_initial_array)
        
        # 批量计算卫星位置和速度
        S = self.orbit.position(t_zd)
        V = self.orbit.velocity(t_zd)
        
        # 批量计算斜距
        r = xyz_array - S
        slant_range = np.linalg.norm(r, axis=1)
        
        # 批量计算距离向像素
        range_pixel = (slant_range - self.near_range) / self.range_spacing
        
        # 批量计算方位向时间和像素
        azimuth_time = t_zd - self.t0
        azimuth_pixel = azimuth_time * self.prf
        
        # 批量计算阴影和叠置
        # 计算视线单位向量
        r_unit = r / slant_range[:, None]
        
        # 计算卫星速度在视线方向的分量
        V_r = np.sum(V * r_unit, axis=1)
        layover = V_r > 0
        
        # 计算仰角
        radial_unit = xyz_array / np.linalg.norm(xyz_array, axis=1)[:, None]
        cos_theta = np.sum(r_unit * radial_unit, axis=1)
        theta = np.arccos(cos_theta)
        shadow = theta > np.pi / 2
        
        return range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover

# -------------------------------
#  Numba 优化的核心计算函数
# -------------------------------
@njit(fastmath=True, cache=True)
def zero_doppler_time_core_numba(xyz, t_initial, max_iter, tol, wavelength, tmin, tmax, times, positions, velocities, method):
    """Numba 优化的 Zero-Doppler 时间求解核心函数"""
    t = t_initial
    
    for i in range(max_iter):
        # 轨道插值 - 简化版，只处理单一点
        # 找到时间对应的轨道段
        idx = np.searchsorted(times, t) - 1
        idx = max(0, min(idx, len(times) - 2))
        
        # 线性插值卫星位置和速度
        dt = (t - times[idx]) / (times[idx+1] - times[idx])
        S = positions[idx] + dt * (positions[idx+1] - positions[idx])
        V = velocities[idx] + dt * (velocities[idx+1] - velocities[idx])
        
        # 计算视线方向
        r = xyz - S
        r_norm = np.linalg.norm(r)
        r_unit = r / r_norm
        
        # 计算多普勒频率
        doppler = -2 * np.dot(V, r_unit) / wavelength
        
        # 计算加速度（使用中心差分）
        dt_acc = 1e-6
        t_minus = max(tmin, t - dt_acc)
        t_plus = min(tmax, t + dt_acc)
        
        # 插值 t_minus 时的速度
        idx_minus = np.searchsorted(times, t_minus) - 1
        idx_minus = max(0, min(idx_minus, len(times) - 2))
        dt_minus = (t_minus - times[idx_minus]) / (times[idx_minus+1] - times[idx_minus])
        V_minus = velocities[idx_minus] + dt_minus * (velocities[idx_minus+1] - velocities[idx_minus])
        
        # 插值 t_plus 时的速度
        idx_plus = np.searchsorted(times, t_plus) - 1
        idx_plus = max(0, min(idx_plus, len(times) - 2))
        dt_plus = (t_plus - times[idx_plus]) / (times[idx_plus+1] - times[idx_plus])
        V_plus = velocities[idx_plus] + dt_plus * (velocities[idx_plus+1] - velocities[idx_plus])
        
        A = (V_plus - V_minus) / (2 * dt_acc)
        
        # 计算多普勒频率导数
        V_dot_r = np.dot(V, r_unit)
        term1 = np.dot(A, r_unit)
        term2 = np.dot(V, (-V + V_dot_r * r_unit)) / r_norm
        doppler_dot = -2 / wavelength * (term1 + term2)
        
        # 处理零导数情况
        if abs(doppler_dot) < 1e-10:
            # 简单处理，返回当前时间
            return t
        
        # 牛顿迭代
        t_new = t - doppler / doppler_dot
        
        # 检查时间是否在有效范围内
        if t_new < tmin or t_new > tmax:
            t_new = np.clip(t_new, tmin, tmax)
        
        # 检查收敛
        if abs(t_new - t) < tol:
            return t_new
        
        t = t_new
    
    # 如果不收敛，返回最后一次迭代的结果
    return t

def zero_doppler_vectorized_core_numba(xyz_array, S, V, S_plus, V_plus, wavelength):
    """Numba 优化的向量化 Zero-Doppler 计算核心函数"""
    n = len(xyz_array)
    doppler = np.empty(n)
    doppler_dot = np.empty(n)
    
    for i in range(n):
        # 计算视线方向
        r = xyz_array[i] - S[i]
        r_norm = np.linalg.norm(r)
        r_unit = r / r_norm
        
        # 计算多普勒频率
        doppler[i] = -2 * np.dot(V[i], r_unit) / wavelength
        
        # 计算t_plus的视线方向
        r_plus = xyz_array[i] - S_plus[i]
        r_norm_plus = np.linalg.norm(r_plus)
        r_unit_plus = r_plus / r_norm_plus
        
        # 计算t_plus的多普勒频率
        doppler_plus = -2 * np.dot(V_plus[i], r_unit_plus) / wavelength
        
        # 计算多普勒频率的导数
        dt = 1e-6
        doppler_dot[i] = (doppler_plus - doppler[i]) / dt
    
    return doppler, doppler_dot

@njit(parallel=True, fastmath=True, cache=True)
def zero_doppler_vectorized_core_analytical_numba(xyz_array, S, V, A, r_unit, r_norm, wavelength):
    """Numba 优化的向量化 Zero-Doppler 计算核心函数（使用解析导数）"""
    n = len(xyz_array)
    doppler = np.empty(n)
    doppler_dot = np.empty(n)
    
    for i in prange(n):
        # 计算多普勒频率
        doppler[i] = -2 * np.dot(V[i], r_unit[i]) / wavelength
        
        # 计算第一项：加速度与视线方向的点积
        term1 = np.dot(A[i], r_unit[i])
        
        # 计算第二项：速度与视线方向变化率的点积
        V_dot_r = np.dot(V[i], r_unit[i])
        term2 = np.dot(V[i], (-V[i] + V_dot_r * r_unit[i])) / r_norm[i]
        
        # 计算多普勒频率导数（解析解）
        doppler_dot[i] = -2 / wavelength * (term1 + term2)
    
    return doppler, doppler_dot

def zero_doppler_time_bisection_core_numba(xyz, t_min, t_max, max_iter, tol, wavelength, orbit):
    """Numba 优化的二分法求解 Zero-Doppler 时间核心函数"""
    # 这里需要注意：Numba 不支持直接调用 Python 类方法
    # 实际使用时，需要将轨道数据传递给函数并在 Numba 中实现插值
    return (t_min + t_max) / 2

@njit(parallel=True, fastmath=True, cache=True)
def compute_theta_numba(dem_h, sat_height, slant_range):
    """Numba 优化的仰角计算"""
    # 简化的仰角计算
    return np.arcsin((sat_height - dem_h) / slant_range)

@njit(parallel=True, fastmath=True, cache=True)
def compute_shadow_numba(theta):
    """Numba 优化的阴影检测"""
    # 简化的阴影检测
    return theta < 0

@njit(parallel=True, fastmath=True, cache=True)
def shadow_layover_batch_core_numba(slant_range, dem_h, sat_height):
    """Numba 优化的 Shadow/Layover 检测核心函数"""
    # 初始化掩码
    shadow_mask = np.zeros_like(dem_h, dtype=bool)
    layover_mask = np.zeros_like(dem_h, dtype=bool)
    
    # 计算 Layover：使用梯度法
    eps = 1e-6
    dR = np.diff(slant_range, axis=1)
    layover_mask[:, 1:] = dR < -eps
    
    # 计算仰角
    theta = compute_theta_numba(dem_h, sat_height, slant_range)
    
    # 计算 Shadow
    shadow_mask = compute_shadow_numba(theta)
    
    return shadow_mask, layover_mask

# -------------------------------
#  主函数
# -------------------------------
# 全局变量，用于多进程处理
worker_geo = None

def init_worker(geo_obj):
    """初始化worker进程"""
    global worker_geo
    worker_geo = geo_obj

def process_block(args):
    """处理单个数据块"""
    lat_block, lon_block, h_block, start, end = args
    # 转换为ECEF坐标
    P = llh_to_xyz(lat_block, lon_block, h_block)
    # 使用向量化处理
    range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover = worker_geo.geo2rdr_vectorized(lat_block, lon_block, h_block)
    return range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover, start, end

def process_dem_in_blocks_parallel(geo, lat, lon, h, block_size=None):
    """并行分块处理DEM数据"""
    total_points = lat.size
    
    # 自适应块大小
    if block_size is None:
        # 根据可用内存和DEM大小计算块大小
        import psutil
        available_memory = psutil.virtual_memory().available
        # 每个点大约需要 24 bytes (3 * float64)
        estimated_memory_per_point = 24
        max_block_size = available_memory // (estimated_memory_per_point * 10)  # 留10倍余量
        block_size = min(100000, max(10000, max_block_size))
    
    num_blocks = (total_points + block_size - 1) // block_size
    
    # 使用 Geo2Rdr 类中从 YAML 配置文件读取的 SAR 图像尺寸
    sar_az_size = geo.azimuth_lines
    sar_range_size = geo.range_samples
    
    # 如果从 YAML 配置文件中读取失败，使用默认值
    if sar_az_size is None or sar_range_size is None:
        sar_az_size = 3645  # 默认值，可根据实际情况调整
        sar_range_size = 3136  # 默认值，可根据实际情况调整
    
    sar_dem = np.full((sar_az_size, sar_range_size), np.nan, dtype=np.float32)
    all_r = []
    all_a = []
    all_R = []
    all_t = []
    all_shadow = []
    all_layover = []
    
    print(f"并行分块处理DEM数据，共{num_blocks}块，每块{block_size}点")
    
    # 准备任务
    tasks = []
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    h_flat = h.flatten()
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, total_points)
        tasks.append((lat_flat[start:end], lon_flat[start:end], h_flat[start:end], start, end))
    
    # 并行处理
    from multiprocessing import Pool
    cpu_count = os.cpu_count()
    # 使用80%的CPU核心
    used_cpu_count = max(1, int(cpu_count * 0.8))
    print(f"使用 {used_cpu_count} 个CPU核心并行处理")
    
    with Pool(processes=used_cpu_count, initializer=init_worker, initargs=(geo,)) as pool:
        results = pool.map(process_block, tasks)
    
    # 收集结果
    for r, a, R, t, shadow, layover, start, end in results:
        # 限制时间在轨道范围内
        t = np.clip(t, geo.orbit.tmin, geo.orbit.tmax)
        
        # 更新SAR DEM
        r_rounded = np.round(r).astype(int)
        a_rounded = np.round(a).astype(int)
        mask = (r_rounded >= 0) & (r_rounded < sar_range_size) & (a_rounded >= 0) & (a_rounded < sar_az_size)
        
        if np.any(mask):
            sar_dem[a_rounded[mask], r_rounded[mask]] = h_flat[start:end][mask]
        
        # 收集结果
        all_r.extend(r)
        all_a.extend(a)
        all_R.extend(R)
        all_t.extend(t)
        all_shadow.extend(shadow)
        all_layover.extend(layover)
    
    return sar_dem, np.array(all_r), np.array(all_a), np.array(all_R), np.array(all_t), np.array(all_shadow), np.array(all_layover)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DEM -> SAR 模拟工具')
    parser.add_argument('--yaml', type=str, required=True, help='YAML 配置文件路径')
    parser.add_argument('--dem', type=str, required=True, help='DEM 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--bbox', type=float, nargs=4, help='边界框 [min_lon, min_lat, max_lon, max_lat]')
    parser.add_argument('--real_sar', type=str, help='真实SAR数据文件路径')
    
    args = parser.parse_args()
    
    # 读取 DEM
    print(f"读取 DEM 文件: {args.dem}")
    bbox = args.bbox if args.bbox else None
    dem_h, dem_lat, dem_lon = read_dem(args.dem, bbox)
    
    # 初始化 Geo2Rdr
    print(f"初始化 Geo2Rdr，使用配置文件: {args.yaml}")
    geo2rdr = Geo2Rdr(args.yaml, dem_lat, dem_lon, dem_h)
    
    # 批量处理 - 使用并行分块处理
    print("开始批量处理...")
    start_time = time.time()
    
    # 生成模拟SAR DEM
    print("生成模拟SAR DEM...")
    sar_dem, range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover = process_dem_in_blocks_parallel(geo2rdr, dem_lat, dem_lon, dem_h)
    
    # 重塑模拟SAR的像素坐标为二维数组，与真实SAR数据形状一致
    sar_az_size = geo2rdr.azimuth_lines or 3645
    sar_range_size = geo2rdr.range_samples or 3136
    # 注意：这里我们使用SAR图像的尺寸来重塑模拟SAR的像素坐标
    # 实际应用中，可能需要根据DEM的实际覆盖范围来调整
    range_pixel = range_pixel[:sar_az_size * sar_range_size].reshape(sar_az_size, sar_range_size)
    azimuth_pixel = azimuth_pixel[:sar_az_size * sar_range_size].reshape(sar_az_size, sar_range_size)
    slant_range = slant_range[:sar_az_size * sar_range_size].reshape(sar_az_size, sar_range_size)
    print(f"重塑模拟SAR像素坐标为二维数组，形状: 方位向={azimuth_pixel.shape}, 距离向={range_pixel.shape}")
    
    end_time = time.time()
    print(f"处理完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 使用 Geo2Rdr 类中从 YAML 配置文件读取的 SAR 图像尺寸
    sar_az_size = geo2rdr.azimuth_lines
    sar_range_size = geo2rdr.range_samples
    
    # 如果从 YAML 配置文件中读取失败，使用默认值
    if sar_az_size is None or sar_range_size is None:
        sar_az_size = 3645  # 默认值，可根据实际情况调整
        sar_range_size = 3136  # 默认值，可根据实际情况调整
    
    print(f"SAR 图像尺寸: {sar_az_size} x {sar_range_size}")
    
    # 生成SLC
    print("生成SLC...")
    slc = geo2rdr.simulate_slc(range_pixel, azimuth_pixel, slant_range)
    
    # 粗配准
    coarse_range_offset = 0
    coarse_azimuth_offset = 0
    if args.real_sar:
        print("进行模拟SAR与真实SAR的粗配准...")
        # 读取真实SAR数据
        ext = os.path.splitext(args.real_sar)[1].lower()
        real_sar_data = None
        try:
            if ext in ['.tif', '.tiff']:
                from osgeo import gdal
                ds = gdal.Open(args.real_sar)
                if ds:
                    real_sar_data = ds.GetRasterBand(1).ReadAsArray()
                    print(f"成功读取GeoTIFF格式的真实SAR数据用于粗配准，形状: {real_sar_data.shape}")
        except Exception as e:
            print(f"读取真实SAR数据用于粗配准失败: {e}")
        
        if real_sar_data is not None:
            # 进行粗配准
            coarse_range_offset, coarse_azimuth_offset = geo2rdr.coarse_registration(slc, real_sar_data)
            print(f"粗配准结果: 距离向偏移={coarse_range_offset} 像素, 方位向偏移={coarse_azimuth_offset} 像素")
    
    # 计算偏移量并更新几何参数
    print("计算偏移量并更新几何参数...")
    # 初始化真实SAR像素坐标
    real_range_pixel = None
    real_azimuth_pixel = None
    
    if args.real_sar:
        print(f"读取真实SAR数据文件: {args.real_sar}")
        # 支持不同格式的真实SAR数据文件
        ext = os.path.splitext(args.real_sar)[1].lower()
        
        try:
            if ext in ['.h5', '.hdf5']:
                # 读取 HDF5 格式的真实SAR数据
                with h5py.File(args.real_sar, 'r') as f:
                    if 'range_pixel' in f and 'azimuth_pixel' in f:
                        real_range_pixel = f['range_pixel'][:]
                        real_azimuth_pixel = f['azimuth_pixel'][:]
                        print(f"成功读取真实SAR数据，形状: {real_range_pixel.shape}")
                        # 对真实SAR数据的像素坐标进行采样，使其与模拟SAR数据的点数相同
                        N = len(range_pixel)  # 模拟SAR数据的点数（由DEM大小决定）
                        
                        # 检查真实SAR数据的维度
                        if real_range_pixel.ndim == 2:
                            # 对于二维数组，计算总像素数
                            real_N = real_range_pixel.size
                            if real_N != N:
                                if real_N > N:
                                    # 随机采样
                                    # 将二维数组展平后采样，然后重塑回原形状
                                    flat_indices = np.random.choice(real_N, N, replace=False)
                                    real_range_pixel = real_range_pixel.flatten()[flat_indices]
                                    real_azimuth_pixel = real_azimuth_pixel.flatten()[flat_indices]
                                    print(f"对真实SAR数据进行采样，采样后形状: 方位向={real_azimuth_pixel.shape}, 距离向={real_range_pixel.shape}")
                                elif real_N < N:
                                    # 重复采样
                                    flat_indices = np.random.choice(real_N, N, replace=True)
                                    real_range_pixel = real_range_pixel.flatten()[flat_indices]
                                    real_azimuth_pixel = real_azimuth_pixel.flatten()[flat_indices]
                                    print(f"对真实SAR数据进行重复采样，采样后形状: 方位向={real_azimuth_pixel.shape}, 距离向={real_range_pixel.shape}")
                            else:
                                # 二维数组的总像素数与模拟SAR数据的点数相同，展平后使用
                                real_range_pixel = real_range_pixel.flatten()
                                real_azimuth_pixel = real_azimuth_pixel.flatten()
                                print("真实SAR数据的像素数量与模拟SAR数据的点数相同，展平后使用")
                        else:
                            # 对于一维数组，使用原来的采样逻辑
                            if len(real_range_pixel) != N:
                                if len(real_range_pixel) > N:
                                    # 随机采样
                                    indices = np.random.choice(len(real_range_pixel), N, replace=False)
                                    real_range_pixel = real_range_pixel[indices]
                                    real_azimuth_pixel = real_azimuth_pixel[indices]
                                    print(f"对真实SAR数据进行采样，采样后形状: 方位向={real_azimuth_pixel.shape}, 距离向={real_range_pixel.shape}")
                                elif len(real_range_pixel) < N:
                                    # 重复采样
                                    indices = np.random.choice(len(real_range_pixel), N, replace=True)
                                    real_range_pixel = real_range_pixel[indices]
                                    real_azimuth_pixel = real_azimuth_pixel[indices]
                                    print(f"对真实SAR数据进行重复采样，采样后形状: 方位向={real_azimuth_pixel.shape}, 距离向={real_range_pixel.shape}")
                            else:
                                print("真实SAR数据的像素数量与模拟SAR数据的点数相同，无需采样")
                    else:
                        print("真实SAR数据文件中缺少必要的字段")
            elif ext in ['.tif', '.tiff']:
                # 读取 GeoTIFF 格式的真实SAR数据
                try:
                    from osgeo import gdal
                    ds = gdal.Open(args.real_sar)
                    if not ds:
                        raise Exception(f"无法打开GeoTIFF文件: {args.real_sar}")
                    
                    # 读取SAR图像数据
                    sar_data = ds.GetRasterBand(1).ReadAsArray()
                    print(f"成功读取GeoTIFF格式的真实SAR数据，形状: {sar_data.shape}")
                    
                    # 生成像素坐标（保持二维）
                    rows, cols = sar_data.shape
                    real_azimuth_pixel, real_range_pixel = np.mgrid[0:rows, 0:cols]
                    print(f"生成的像素坐标形状: 方位向={real_azimuth_pixel.shape}, 距离向={real_range_pixel.shape}")
                except ImportError:
                    print("警告：GDAL 库未安装，无法读取 GeoTIFF 文件，使用默认像素坐标")
                    # 使用默认尺寸
                    rows, cols = 3645, 3136
                    real_azimuth_pixel, real_range_pixel = np.mgrid[0:rows, 0:cols]
                    print(f"使用默认像素坐标形状: 方位向={real_azimuth_pixel.shape}, 距离向={real_range_pixel.shape}")
            else:
                print(f"不支持的文件格式: {ext}")
        except Exception as e:
            print(f"读取真实SAR数据失败: {e}")
    

    # 计算偏移量
    range_offset, azimuth_offset = geo2rdr.compute_offset(
        range_pixel, azimuth_pixel, real_range_pixel, real_azimuth_pixel
    )
    
    # 更新几何参数
    geo2rdr.update_geometry(range_offset, azimuth_offset)
    
    # 重新生成模拟SAR DEM（使用更新后的参数）
    print("使用更新后的几何参数重新生成模拟SAR DEM...")
    sar_dem_updated, range_pixel_updated, azimuth_pixel_updated, slant_range_updated, azimuth_time_updated, shadow_updated, layover_updated = process_dem_in_blocks_parallel(geo2rdr, dem_lat, dem_lon, dem_h)
    
    # 重新生成SLC（使用更新后的参数）
    print("使用更新后的几何参数重新生成SLC...")
    slc_updated = geo2rdr.simulate_slc(range_pixel_updated, azimuth_pixel_updated, slant_range_updated)
    
    # 保存SLC
    output_prefix = os.path.splitext(args.output)[0]
    print(f"保存SLC到 {output_prefix}_slc_real.tif 和 {output_prefix}_slc_imag.tif")
    try:
        from osgeo import gdal, osr
        # 保存实部
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(f"{output_prefix}_slc_real.tif", slc_updated.shape[1], slc_updated.shape[0], 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(np.real(slc_updated))
        ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds = None
        
        # 保存虚部
        ds = driver.Create(f"{output_prefix}_slc_imag.tif", slc_updated.shape[1], slc_updated.shape[0], 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(np.imag(slc_updated))
        ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
        ds.SetProjection(srs.ExportToWkt())
        ds = None
    except Exception as e:
        print(f"保存SLC失败: {e}")
    
    # 保存结果
    print(f"保存结果到: {args.output}")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('range_pixel', data=range_pixel_updated)
        f.create_dataset('azimuth_pixel', data=azimuth_pixel_updated)
        f.create_dataset('slant_range', data=slant_range_updated)
        f.create_dataset('azimuth_time', data=azimuth_time_updated)
        f.create_dataset('shadow', data=shadow_updated)
        f.create_dataset('layover', data=layover_updated)
        f.create_dataset('dem_lat', data=dem_lat)
        f.create_dataset('dem_lon', data=dem_lon)
        f.create_dataset('dem_h', data=dem_h)
        # 添加模拟SAR DEM
        f.create_dataset('sar_dem', data=sar_dem_updated)
        # 添加 SAR 图像尺寸信息
        f.create_dataset('sar_az_size', data=sar_az_size)
        f.create_dataset('sar_range_size', data=sar_range_size)
        # 添加更新后的几何参数
        f.create_dataset('original_near_range', data=geo2rdr.original_near_range)
        f.create_dataset('updated_near_range', data=geo2rdr.near_range)
        f.create_dataset('original_t0', data=geo2rdr.original_t0)
        f.create_dataset('updated_t0', data=geo2rdr.t0)
        f.create_dataset('range_offset', data=range_offset)
        f.create_dataset('azimuth_offset', data=azimuth_offset)
        # 添加粗配准结果
        f.create_dataset('coarse_range_offset', data=coarse_range_offset)
        f.create_dataset('coarse_azimuth_offset', data=coarse_azimuth_offset)
        # 添加从YAML文件读取的真实SAR参数
        f.create_dataset('sar_azimuth_lines', data=geo2rdr.azimuth_lines if geo2rdr.azimuth_lines is not None else -1)
        f.create_dataset('sar_range_samples', data=geo2rdr.range_samples if geo2rdr.range_samples is not None else -1)
    
    print("完成！")

if __name__ == '__main__':
    main()