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
import multiprocessing as _mp

# 导入Numba
from numba import njit, prange
from functools import lru_cache

# -------------------------------
#  SAR lat/lon 网格写入（由 DEM->SAR 映射反推）
# -------------------------------
def write_sar_latlon_grids_from_dem_mapping(
    f_h5,
    *,
    dem_gt,
    dem_shape,
    range_pixel,
    azimuth_pixel,
    sar_shape,
    dtype="float32",
    compress="lzf",
    chunk=(512, 512),
    block_points=2_000_000,
):
    """
    在 HDF5 中写入 SAR 网格下的经纬度（sar_lat/sar_lon），并创建 lat_grid/lon_grid 软链接。

    说明（非常重要）：
    - 这里的 sar_lat/sar_lon 不是严格的 rdr2geo 求解，而是用 DEM->SAR(geo2rdr) 映射反推：
      对每个 DEM 像素中心 (lon,lat)，把它写到其对应的 SAR 像素 (az,rg) 上。
    - 叠置/阴影区域会出现“一像素多地面点”的情况；本实现默认“后写覆盖前写”（last-win）。
      如果需要更严格的选择规则（例如最小 slant_range），建议在后续版本中加入 tie-break。
    - 没有任何 DEM 点命中的 SAR 像素会保持 NaN。

    参数：
    - f_h5: 已打开的 h5py.File(..., 'w'/'a')
    - dem_gt: DEM geotransform(6)，假设为 EPSG:4326 下的 lon/lat 度
    - dem_shape: (nrows,ncols)
    - range_pixel/azimuth_pixel: 1D 数组，长度=DEM 点数，表示每个 DEM 点对应的 SAR 像素坐标（可为浮点）
    - sar_shape: (sar_az_size, sar_range_size)
    - dtype: 输出 dtype，建议 float32
    - compress: 'lzf'/'gzip'/'none'
    - chunk: HDF chunk 大小（影响写入性能）
    - block_points: 每次扫描的 DEM 点数量（影响内存峰值）
    """
    import numpy as np
    import h5py

    nrows, ncols = int(dem_shape[0]), int(dem_shape[1])
    sar_az_size, sar_range_size = int(sar_shape[0]), int(sar_shape[1])

    dt = np.float32 if str(dtype).lower() == "float32" else np.float64
    comp = None if str(compress).lower() == "none" else str(compress).lower()
    if comp not in (None, "lzf", "gzip"):
        raise ValueError(f"不支持的 compress: {compress}")

    # 创建 dataset（默认填 NaN）
    lat_ds = f_h5.create_dataset(
        "sar_lat",
        shape=(sar_az_size, sar_range_size),
        dtype=dt,
        chunks=tuple(chunk),
        compression=comp,
        shuffle=True,
        fillvalue=np.nan,
    )
    lon_ds = f_h5.create_dataset(
        "sar_lon",
        shape=(sar_az_size, sar_range_size),
        dtype=dt,
        chunks=tuple(chunk),
        compression=comp,
        shuffle=True,
        fillvalue=np.nan,
    )
    lat_ds.attrs["source"] = "from_dem_mapping_last_win"
    lon_ds.attrs["source"] = "from_dem_mapping_last_win"
    lat_ds.attrs["dem_geotransform"] = np.asarray(dem_gt, dtype=np.float64)
    lon_ds.attrs["dem_geotransform"] = np.asarray(dem_gt, dtype=np.float64)

    # 软链接（兼容外部工具期望的命名）
    try:
        f_h5["lat_grid"] = h5py.SoftLink("/sar_lat")
        f_h5["lon_grid"] = h5py.SoftLink("/sar_lon")
    except Exception:
        pass

    N = int(range_pixel.shape[0])
    block_points = max(100_000, int(block_points))

    gt0, gt1, gt2, gt3, gt4, gt5 = [float(x) for x in dem_gt]

    wrote = 0
    for i0 in range(0, N, block_points):
        i1 = min(N, i0 + block_points)
        rg = np.asarray(range_pixel[i0:i1], dtype=np.float64)
        az = np.asarray(azimuth_pixel[i0:i1], dtype=np.float64)

        r = np.rint(rg).astype(np.int32, copy=False)
        a = np.rint(az).astype(np.int32, copy=False)
        ok = (
            np.isfinite(rg) & np.isfinite(az)
            & (r >= 0) & (r < sar_range_size)
            & (a >= 0) & (a < sar_az_size)
        )
        if not np.any(ok):
            continue

        ok_pos = np.flatnonzero(ok).astype(np.int64)  # block 内的 DEM flat index
        idx = ok_pos + int(i0)  # DEM 全局 flat index
        row = idx // int(ncols)
        col = idx - row * int(ncols)

        # 像素中心
        c0 = col.astype(np.float64) + 0.5
        r0 = row.astype(np.float64) + 0.5
        lon = gt0 + gt1 * c0 + gt2 * r0
        lat = gt3 + gt4 * c0 + gt5 * r0

        # 写入 SAR 像素：
        # h5py 的 fancy indexing 要求索引单调递增；因此按 az 行分组、在每行内按 range 递增写入。
        a_ok = a[ok]
        r_ok = r[ok]
        lon_ok = lon.astype(dt, copy=False)
        lat_ok = lat.astype(dt, copy=False)

        # stable 排序：确保同一 (a,r) 的多次命中保持 DEM 顺序，从而 “last-win”
        key = a_ok.astype(np.int64) * int(sar_range_size) + r_ok.astype(np.int64)
        order = np.argsort(key, kind="mergesort")
        a_s = a_ok[order]
        r_s = r_ok[order]
        lon_s = lon_ok[order]
        lat_s = lat_ok[order]

        # 按 az 行分组写入
        uniq_a, idx0, cnt = np.unique(a_s, return_index=True, return_counts=True)
        for aa, j0, cc in zip(uniq_a.tolist(), idx0.tolist(), cnt.tolist()):
            j1 = j0 + cc
            rr = r_s[j0:j1]
            # rr 已非降序；保留每个 rr 的最后一次（last-win）
            keep = np.ones(rr.shape[0], dtype=bool)
            if rr.shape[0] >= 2:
                keep[:-1] = rr[1:] != rr[:-1]
            rr_u = rr[keep].astype(np.int64, copy=False)
            lon_u = lon_s[j0:j1][keep]
            lat_u = lat_s[j0:j1][keep]

            # 这里 rr_u 是严格递增，满足 h5py 索引要求
            lon_ds[int(aa), rr_u] = lon_u
            lat_ds[int(aa), rr_u] = lat_u
            wrote += int(rr_u.size)

    lat_ds.attrs["written_points"] = int(wrote)
    lon_ds.attrs["written_points"] = int(wrote)

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
                    A_valid = orbit.acceleration(t_zd_valid)
                    
                    # 计算第一项：加速度与视线方向的点积
                    term1 = np.sum(A_valid * r_unit_valid, axis=1)
                    
                    # 计算第二项：速度与视线方向变化率的点积
                    V_dot_r = np.sum(V_valid * r_unit_valid, axis=1)
                    term2 = np.sum(V_valid * (-V_valid + V_dot_r[:, None] * r_unit_valid), axis=1) / r_norm_valid
                    
                    # 计算多普勒频率导数
                    doppler_dot_valid = -2 / wavelength * (term1 + term2)
                except Exception:
                    # 回退到数值导数
                    dt = 1e-3
                    t_plus = t_zd_valid + dt
                    S_plus = orbit.position(t_plus)
                    V_plus = orbit.velocity(t_plus)
                    r_plus = r_valid - (S_plus - S[valid_mask])
                    r_norm_plus = np.linalg.norm(r_plus, axis=1)

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
                    A = orbit.acceleration(t_zd)
                    
                    # 计算第一项：加速度与视线方向的点积
                    term1 = np.sum(A * r_unit, axis=1)
                    
                    # 计算第二项：速度与视线方向变化率的点积
                    V_dot_r = np.sum(V * r_unit, axis=1)
                    term2 = np.sum(V * (-V + V_dot_r[:, None] * r_unit), axis=1) / r_norm
                    
                    # 计算多普勒频率导数
                    doppler_dot = -2 / wavelength * (term1 + term2)
                except Exception:
                    # 回退到数值导数
                    dt = 1e-3
                    t_plus = t_zd + dt
                    S_plus = orbit.position(t_plus)
                    V_plus = orbit.velocity(t_plus)
                    r_plus = xyz_chunk - S_plus
                    r_norm_plus = np.linalg.norm(r_plus, axis=1)

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
            # 注意：
            # 旧实现试图用 dict 缓存 (key=浮点时间) 来加速 position/velocity。
            # 但 Zero-Doppler 的牛顿迭代会产生大量“几乎不重复”的浮点时间，
            # 缓存命中率极低，反而引入 O(N) Python 循环 + 哈希开销，成为主要瓶颈。
            # 因此默认禁用该缓存（仍保留字段，方便将来按需启用/量化时间再缓存）。
            self._position_cache = None
            self._velocity_cache = None
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
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 0:
            t = t.reshape(1)

        if self.method == 'CUBIC':
            # scipy 的 interp1d 本身可向量化；这里不做逐点 try/except。
            out = np.empty((t.size, 3), dtype=np.float64)
            out[:, 0] = self.pos_interpolators[0](t)
            out[:, 1] = self.pos_interpolators[1](t)
            out[:, 2] = self.pos_interpolators[2](t)
            return out

        # HERMITE：纯 numpy 向量化评估（无 Python 循环/缓存）
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.h) - 1)
        dt = t - self.times[idx]

        y = self.positions[idx]
        yp = self.velocities[idx]
        c = self.c[idx]
        d = self.d[idx]
        out = y + yp * dt[:, None] + c * (dt[:, None] ** 2) + d * (dt[:, None] ** 3)

        # 极少数情况下会出现 NaN/Inf（例如轨道数据异常）；用端点位置兜底。
        bad = np.isnan(out).any(axis=1) | np.isinf(out).any(axis=1)
        if np.any(bad):
            out[bad] = self.positions[idx[bad]]
        return out
    
    def velocity(self, t):
        """计算给定时间的卫星速度"""
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 0:
            t = t.reshape(1)

        if self.method == 'CUBIC':
            out = np.empty((t.size, 3), dtype=np.float64)
            out[:, 0] = self.vel_interpolators[0](t)
            out[:, 1] = self.vel_interpolators[1](t)
            out[:, 2] = self.vel_interpolators[2](t)
            return out

        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.h) - 1)
        dt = t - self.times[idx]

        yp = self.velocities[idx]
        c = self.c[idx]
        d = self.d[idx]
        out = yp + 2.0 * c * dt[:, None] + 3.0 * d * (dt[:, None] ** 2)

        bad = np.isnan(out).any(axis=1) | np.isinf(out).any(axis=1)
        if np.any(bad):
            out[bad] = self.velocities[idx[bad]]
        return out

    def acceleration(self, t):
        """计算给定时间的卫星加速度（HERMITE 可解析计算，避免中心差分的 2 次 velocity 调用）"""
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 0:
            t = t.reshape(1)

        # 仅 HERMITE 有解析表达式：pos = y + yp*dt + c*dt^2 + d*dt^3
        # => acc = d2pos/dt2 = 2*c + 6*d*dt
        if self.method != 'HERMITE':
            # CUBIC：无解析加速度时，调用方应回退到数值差分
            raise NotImplementedError("acceleration() 仅在 HERMITE 模式下支持解析计算")

        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.h) - 1)
        dt = t - self.times[idx]
        c = self.c[idx]
        d = self.d[idx]
        out = 2.0 * c + 6.0 * d * dt[:, None]

        bad = np.isnan(out).any(axis=1) | np.isinf(out).any(axis=1)
        if np.any(bad):
            out[bad] = 0.0
        return out

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

    def acceleration(self, t):
        """计算卫星加速度（若轨道插值器支持解析加速度则直接返回，否则回退到数值差分）"""
        t = np.atleast_1d(t).astype(np.float64)
        if self.interpolator and hasattr(self.interpolator, "acceleration"):
            try:
                return self.interpolator.acceleration(t)
            except Exception:
                pass

        # 回退：中心差分（会多调用 2 次 velocity）
        dt = 1e-3
        v_minus = self.velocity(t - dt)
        v_plus = self.velocity(t + dt)
        return (v_plus - v_minus) / (2.0 * dt)

# -------------------------------
#  DEM 读取与处理
# -------------------------------
def read_dem_gt(dem_file, bbox=None):
    """
    读取 DEM 文件并可选裁剪（不生成 DEM_lat/DEM_lon 2D 网格）
    :param dem_file: DEM 文件路径
    :param bbox: 边界框 [min_lon, min_lat, max_lon, max_lat]
    :return: dem_h, dem_gt(6), (nrows, ncols)
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

        # 用像素中心坐标（避免 0.5 像素偏移），同时避免 linspace 的 off-by-one 误差
        # north-up 情况下通常 gt[2]=gt[4]=0
        # 注意：这里仅用于 bbox 裁剪的 1D 近似（假设 north-up）。如果 DEM 带旋转，bbox 裁剪会退化为近似。
        lats_1d = (gt[3] + gt[5] * (np.arange(rows, dtype=np.float64) + 0.5)).astype(np.float32)
        lons_1d = (gt[0] + gt[1] * (np.arange(cols, dtype=np.float64) + 0.5)).astype(np.float32)

        # 如果指定了边界框，优先用 1D 单调数组做索引裁剪，避免构造全量 meshgrid 再 mask
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox

            # lon 通常递增
            if lons_1d[0] <= lons_1d[-1]:
                c0 = int(np.searchsorted(lons_1d, min_lon, side='left'))
                c1 = int(np.searchsorted(lons_1d, max_lon, side='right'))
            else:
                # 极少数情况下递减
                lons_rev = lons_1d[::-1]
                c0r = int(np.searchsorted(lons_rev, max_lon, side='left'))
                c1r = int(np.searchsorted(lons_rev, min_lon, side='right'))
                c0 = cols - c1r
                c1 = cols - c0r

            # lat 可能递减（gt[5] < 0）
            if lats_1d[0] <= lats_1d[-1]:
                r0 = int(np.searchsorted(lats_1d, min_lat, side='left'))
                r1 = int(np.searchsorted(lats_1d, max_lat, side='right'))
            else:
                lats_rev = lats_1d[::-1]
                r0r = int(np.searchsorted(lats_rev, max_lat, side='left'))
                r1r = int(np.searchsorted(lats_rev, min_lat, side='right'))
                r0 = rows - r1r
                r1 = rows - r0r

            r0 = max(0, min(rows, r0))
            r1 = max(0, min(rows, r1))
            c0 = max(0, min(cols, c0))
            c1 = max(0, min(cols, c1))

            if (r1 > r0) and (c1 > c0):
                dem_h = dem_h[r0:r1, c0:c1]
                # 更新 geotransform 到裁剪后的左上角像素
                gt = (
                    gt[0] + c0 * gt[1] + r0 * gt[2],
                    gt[1],
                    gt[2],
                    gt[3] + c0 * gt[4] + r0 * gt[5],
                    gt[4],
                    gt[5],
                )
                rows, cols = dem_h.shape

        return dem_h, tuple(gt), (int(rows), int(cols))
    except ImportError:
        # 如果没有 GDAL，生成一个示例 DEM
        print("警告：GDAL 库未安装，生成示例 DEM")
        # 生成一个 100x100 的示例 DEM
        rows, cols = 100, 100
        dem_h = np.zeros((rows, cols), dtype=np.float32)
        # 添加一些地形起伏
        for i in range(rows):
            for j in range(cols):
                dem_h[i, j] = 1000 + 500 * np.sin(i/20) * np.cos(j/20)
        gt = (94.0, (95.0 - 94.0) / cols, 0.0, 31.0, 0.0, -(31.0 - 30.0) / rows)
        return dem_h, tuple(gt), (rows, cols)


def read_dem(dem_file, bbox=None):
    """
    兼容旧接口：返回 dem_h, DEM_lat(2D), DEM_lon(2D)。
    新代码建议使用 read_dem_gt() 避免构造巨大 2D 网格。
    """
    dem_h, gt, (rows, cols) = read_dem_gt(dem_file, bbox=bbox)
    # 构造 2D 网格（大 DEM 会非常耗内存）
    r = (np.arange(rows, dtype=np.float64) + 0.5)
    c = (np.arange(cols, dtype=np.float64) + 0.5)
    # lon = gt0 + gt1*c + gt2*r, lat = gt3 + gt4*c + gt5*r
    # 这里只在兼容模式使用
    cc, rr = np.meshgrid(c, r, indexing='xy')
    DEM_lon = (gt[0] + gt[1] * cc + gt[2] * rr).astype(np.float32)
    DEM_lat = (gt[3] + gt[4] * cc + gt[5] * rr).astype(np.float32)
    return dem_h, DEM_lat, DEM_lon

# -------------------------------
#  DEM -> SAR 映射计算
# -------------------------------
class Geo2Rdr:
    def __init__(
        self,
        yaml_file,
        dem_lat=None,
        dem_lon=None,
        dem_h=None,
        dem_gt=None,
        dem_shape=None,
        precompute_ecef=True,
    ):
        """
        初始化 Geo2Rdr 类
        :param yaml_file: YAML 配置文件
        :param dem_lat: DEM 纬度网格
        :param dem_lon: DEM 经度网格
        :param dem_h: DEM 高度数据
        :param precompute_ecef: 是否预先把 DEM 全部点转换为 ECEF（大 DEM 会非常耗内存/耗时）
        """
        self.yaml_file = yaml_file
        self.dem_lat = dem_lat
        self.dem_lon = dem_lon
        self.dem_h = dem_h
        self.dem_gt = dem_gt
        self.dem_shape = dem_shape
        
        # 读取 YAML 参数
        with open(yaml_file, 'r') as f:
            ycfg = yaml.safe_load(f)

        # 统一的 ISO8601 时间解析（返回 Unix epoch seconds）
        def _parse_time_iso8601(s):
            import datetime
            if s is None:
                return None
            if isinstance(s, (int, float)):
                return float(s)
            ss = str(s).strip()
            # 兼容 'Z'
            if ss.endswith('Z'):
                ss = ss[:-1] + '+00:00'
            return datetime.datetime.fromisoformat(ss).timestamp()
        
        # 从 master.yaml 解析参数（保持与 dem2sar_full.py 一致：时间用 epoch seconds）
        meta = ycfg.get('metadata', {}) or {}
        self.t0_str = meta.get('first_line_sensing_time', None)
        self.t1_str = meta.get('last_line_sensing_time', None)
        t0_meta_epoch = _parse_time_iso8601(self.t0_str) if self.t0_str else None
        t1_meta_epoch = _parse_time_iso8601(self.t1_str) if self.t1_str else None

        # 轨道数据（支持多种 YAML 结构）
        orbit_points = None
        if isinstance(ycfg.get('orbit_data', None), dict) and ('orbit_points' in ycfg['orbit_data']):
            orbit_points = ycfg['orbit_data'].get('orbit_points')
        elif isinstance(ycfg.get('orbit', None), list):
            orbit_points = ycfg.get('orbit')
        elif isinstance(ycfg.get('orbit_points', None), list):
            orbit_points = ycfg.get('orbit_points')

        orbit_rows = []
        orbit_time_is_epoch = False
        if isinstance(orbit_points, list) and orbit_points:
            for p in orbit_points:
                try:
                    tt = p.get('time')
                    t = _parse_time_iso8601(tt)
                    # 粗略判断是否为 epoch（秒级 1e9 量级）
                    if t is not None and t > 1e9:
                        orbit_time_is_epoch = True
                    pos = p.get('position', {})
                    vel = p.get('velocity', {})
                    orbit_rows.append([
                        float(t),
                        float(pos.get('x')), float(pos.get('y')), float(pos.get('z')),
                        float(vel.get('vx')), float(vel.get('vy')), float(vel.get('vz')),
                    ])
                except Exception:
                    continue
        elif isinstance(ycfg.get('orbit_data', None), (list, tuple)):
            # 允许直接给 Nx7 数组
            arr = np.asarray(ycfg.get('orbit_data'), dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 7:
                orbit_rows = arr.tolist()

        orbit_arr = np.asarray(orbit_rows, dtype=np.float64)
        if orbit_arr.size == 0:
            raise ValueError("YAML 中未解析到有效 orbit_points / orbit_data（需要 time/position/velocity）")
        # 按时间排序
        order = np.argsort(orbit_arr[:, 0])
        orbit_arr = orbit_arr[order]

        self.orbit_config = {
            'orbit_data': orbit_arr,
            'tmin': float(orbit_arr[0, 0]),
            'tmax': float(orbit_arr[-1, 0]),
        }

        # 传感器起止时间：尽量与 orbit 时间体系一致
        if orbit_time_is_epoch:
            self.t0 = float(t0_meta_epoch if t0_meta_epoch is not None else self.orbit_config['tmin'])
            self.t1 = float(t1_meta_epoch if t1_meta_epoch is not None else self.orbit_config['tmax'])
        else:
            # orbit 为相对秒：t0/t1 也必须在该范围内，否则会导致全部点出界、最终命中像素为 0
            if (t0_meta_epoch is not None) or (t1_meta_epoch is not None):
                print("警告：orbit_points.time 看起来是相对秒，但 metadata 时间是 ISO8601；将 t0/t1 退化为 orbit 相对时间范围。")
            self.t0 = float(ycfg.get('t0', self.orbit_config['tmin']))
            self.t1 = float(ycfg.get('t1', self.orbit_config['tmax']))
        
        # 系统参数（从 radar_parameters 部分读取）
        radar_params = ycfg.get('radar_parameters', {})
        self.prf = radar_params.get('prf', ycfg.get('prf', 1500.0))  # 先从 radar_parameters 读取，再从根级别读取
        self.near_range = radar_params.get('near_range', ycfg.get('near_range', 800000.0))  # 先从 radar_parameters 读取，再从根级别读取
        # range_spacing 在不同 YAML 中可能叫 range_spacing / range_pixel_spacing / range_sample_spacing
        self.range_spacing = radar_params.get(
            'range_spacing',
            radar_params.get('range_pixel_spacing', ycfg.get('range_spacing', ycfg.get('range_pixel_spacing', 6.25))),
        )
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
        """
        print(f"从 YAML 文件读取的真实 SAR 参数:")
        print(f"  方位向行数 (azimuth_lines): {self.azimuth_lines}")
        print(f"  距离向列数 (range_samples): {self.range_samples}")
        print(f"  数据格式 (data_format): {self.sar_data_format}")
        print(f"  字节序 (byte_order): {self.sar_byte_order}")
        print(f"  波段 (bands): {self.sar_bands}")
        """
        # 初始 R 和 t0 值
        self.original_near_range = self.near_range
        self.original_t0 = self.t0
        
        # 预计算 ECEF 坐标（仅在需要时启用；并行分块时不需要）
        self.xyz_array = None
        self.N = 0
        if precompute_ecef and (self.dem_lat is not None) and (self.dem_lon is not None) and (self.dem_h is not None):
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
        lat_c = None
        lon_c = None
        if (self.dem_lat is not None) and (self.dem_lon is not None):
            lat_c = float(np.mean(self.dem_lat))
            lon_c = float(np.mean(self.dem_lon))
        elif (self.dem_gt is not None) and (self.dem_shape is not None):
            # 用 geotransform + shape 估计 DEM 中心（像素中心）
            gt = self.dem_gt
            nrows, ncols = self.dem_shape
            r = (nrows * 0.5)
            c = (ncols * 0.5)
            lon_c = float(gt[0] + gt[1] * (c + 0.5) + gt[2] * (r + 0.5))
            lat_c = float(gt[3] + gt[4] * (c + 0.5) + gt[5] * (r + 0.5))
        else:
            return 1  # 默认右视
        
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
                # 优先用轨道插值器的解析加速度（HERMITE：2*c+6*d*dt）
                A = self.orbit.acceleration(t).reshape(3,)
                
                # 计算第一项：加速度与视线方向的点积
                term1 = np.dot(A, r_unit)
                
                # 计算第二项：速度与视线方向变化率的点积
                V_dot_r = np.dot(V, r_unit)
                term2 = np.dot(V, (-V + V_dot_r * r_unit)) / r_norm
                
                # 计算多普勒频率导数
                doppler_dot = -2 / self.wavelength * (term1 + term2)
            except Exception:
                # 回退：数值差分（更慢）
                dt = 1e-3
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
                # 优先用解析加速度（若可用）
                A = self.orbit.acceleration(t_zd)
                
                # 计算第一项：加速度与视线方向的点积
                term1 = np.sum(A * r_unit, axis=1)
                
                # 计算第二项：速度与视线方向变化率的点积
                V_dot_r = np.sum(V * r_unit, axis=1)
                term2 = np.sum(V * (-V + V_dot_r[:, None] * r_unit), axis=1) / r_norm
                
                # 计算多普勒频率导数
                doppler_dot = -2 / self.wavelength * (term1 + term2)
            except Exception:
                # 回退：数值差分（更慢）
                dt = 1e-3
                t_plus = t_zd + dt
                S_plus = self.orbit.position(t_plus)
                V_plus = self.orbit.velocity(t_plus)
                r_plus = xyz_array - S_plus
                r_norm_plus = np.linalg.norm(r_plus, axis=1)
                r_unit_plus = r_plus / r_norm_plus[:, None]
                doppler_plus = -2 * np.sum(V_plus * r_unit_plus, axis=1) / self.wavelength
                doppler_dot = (doppler_plus - doppler) / dt
            
            # 处理零导数情况：不要逐点二分（会极慢），后面统一批量处理
            mask0 = np.abs(doppler_dot) < 1e-10
            
            # 牛顿迭代
            t_new = t_zd - doppler / doppler_dot
            
            # 检查时间是否在有效范围内
            out_of_bounds = (t_new < self.orbit.tmin) | (t_new > self.orbit.tmax)
            bad = mask0 | out_of_bounds
            good = ~bad

            if np.any(good):
                t_zd[good] = t_new[good]

            if np.any(bad):
                t_zd[bad] = self._zero_doppler_time_bisection_batch(
                    xyz_array[bad],
                    self.orbit.tmin,
                    self.orbit.tmax,
                    max_iter=max_iter,
                    tol=tol,
                )
            
            # 检查收敛
            if np.any(good) and (np.max(np.abs(t_new[good] - t_zd[good])) < tol):
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

    def _zero_doppler_time_bisection_batch(self, xyz_array, t_min, t_max, max_iter=50, tol=1e-9):
        """
        批量二分法求解 Zero-Doppler 时间。
        目的：替代 _zero_doppler_time_vectorized() 里对坏点的逐点 Python for-loop 二分，
        在极端情况下能把耗时从“点数 * 迭代次数”降为“迭代次数（向量化）”。
        """
        xyz_array = np.asarray(xyz_array, dtype=np.float64)
        if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
            raise ValueError("xyz_array 必须为 (M,3)")
        m = xyz_array.shape[0]
        if m == 0:
            return np.zeros((0,), dtype=np.float64)

        t_lo = np.full(m, float(t_min), dtype=np.float64) if np.isscalar(t_min) else np.asarray(t_min, dtype=np.float64).copy()
        t_hi = np.full(m, float(t_max), dtype=np.float64) if np.isscalar(t_max) else np.asarray(t_max, dtype=np.float64).copy()

        # 计算边界 doppler
        S_lo = self.orbit.position(t_lo)
        V_lo = self.orbit.velocity(t_lo)
        r_lo = xyz_array - S_lo
        rn_lo = np.linalg.norm(r_lo, axis=1)
        ru_lo = r_lo / rn_lo[:, None]
        dop_lo = -2.0 * np.sum(V_lo * ru_lo, axis=1) / self.wavelength

        S_hi = self.orbit.position(t_hi)
        V_hi = self.orbit.velocity(t_hi)
        r_hi = xyz_array - S_hi
        rn_hi = np.linalg.norm(r_hi, axis=1)
        ru_hi = r_hi / rn_hi[:, None]
        dop_hi = -2.0 * np.sum(V_hi * ru_hi, axis=1) / self.wavelength

        ok = (dop_lo * dop_hi) <= 0.0
        # 对不满足括住条件的点，直接用中点（避免死循环）
        out = 0.5 * (t_lo + t_hi)
        if not np.any(ok):
            return out

        # 仅对 ok 点做二分
        for _ in range(int(max_iter)):
            t_mid = 0.5 * (t_lo + t_hi)
            S_mid = self.orbit.position(t_mid)
            V_mid = self.orbit.velocity(t_mid)
            r_mid = xyz_array - S_mid
            rn_mid = np.linalg.norm(r_mid, axis=1)
            ru_mid = r_mid / rn_mid[:, None]
            dop_mid = -2.0 * np.sum(V_mid * ru_mid, axis=1) / self.wavelength

            # 时间区间收敛
            if np.max((t_hi - t_lo)[ok]) < tol:
                break

            left = (dop_lo * dop_mid) <= 0.0
            left &= ok
            right = (~left) & ok

            t_hi[left] = t_mid[left]
            dop_hi[left] = dop_mid[left]

            t_lo[right] = t_mid[right]
            dop_lo[right] = dop_mid[right]

        out[ok] = 0.5 * (t_lo[ok] + t_hi[ok])
        return out
    
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
        # 约定：range_offset/azimuth_offset 为“把模拟 SAR roll 到真实 SAR”所需的像素位移
        # 与 dem2sar_full.py 保持一致：near_range += shift_x * range_spacing
        self.near_range += range_offset * self.range_spacing
        
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
        
        # 使用工程级 Shadow/Layover 检测方法重新计算（需要 2D DEM 网格）
        if (self.dem_lat is not None) and (self.dem_h is not None):
            try:
                slant_range_2d = slant_range.reshape(self.dem_lat.shape)
                dem_h_2d = self.dem_h

                # 计算卫星高度（使用轨道中点位置）
                t_mid = (self.orbit.tmin + self.orbit.tmax) / 2
                S_mid = self.orbit.position(t_mid).reshape(3,)
                _, _, sat_height = xyz_to_llh(np.array([S_mid]))
                sat_height = sat_height[0]

                shadow_mask_2d, layover_mask_2d = self.shadow_layover_batch(slant_range_2d, dem_h_2d, sat_height)
                shadow_mask = shadow_mask_2d.flatten()
                layover_mask = layover_mask_2d.flatten()
            except Exception as e:
                print(f"工程级 Shadow/Layover 检测跳过（reshape/计算失败）: {e}")
        
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
        # 初值必须落在轨道时间范围内，否则会导致大面积 out_of_bounds，甚至最终全 0 命中
        t0_init = float(np.clip(self.t0, self.orbit.tmin, self.orbit.tmax))
        t_initial_array = np.full(N, t0_init, dtype=np.float64)
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
_IN_LAT_FLAT = None
_IN_LON_FLAT = None
_IN_H_FLAT = None
_IN_SHMS = None
_DEM_GT = None
_DEM_NCOLS = 0
_OUT_RANGE = None
_OUT_AZ = None
_OUT_SLANT = None
_OUT_AZ_TIME = None
_OUT_SHADOW = None
_OUT_LAYOVER = None
_OUT_LEN = 0

def init_worker(
    yaml_file,
    out_range_raw,
    out_az_raw,
    out_slant_raw,
    out_az_time_raw,
    out_shadow_raw,
    out_layover_raw,
    out_len,
    in_lat_shm_name=None,
    in_lon_shm_name=None,
    in_h_shm_name=None,
    dem_gt=None,
    dem_ncols=None,
    in_dtype="float32",
):
    """初始化 worker 进程（共享内存输出）"""
    global worker_geo, _OUT_RANGE, _OUT_AZ, _OUT_SLANT, _OUT_AZ_TIME, _OUT_SHADOW, _OUT_LAYOVER, _OUT_LEN
    global _IN_LAT_FLAT, _IN_LON_FLAT, _IN_H_FLAT, _IN_SHMS, _DEM_GT, _DEM_NCOLS
    worker_geo = Geo2Rdr(yaml_file, precompute_ecef=False)
    _OUT_LEN = int(out_len)
    _OUT_RANGE = np.frombuffer(out_range_raw, dtype=np.float32, count=_OUT_LEN)
    _OUT_AZ = np.frombuffer(out_az_raw, dtype=np.float32, count=_OUT_LEN)
    _OUT_SLANT = np.frombuffer(out_slant_raw, dtype=np.float32, count=_OUT_LEN)
    _OUT_AZ_TIME = np.frombuffer(out_az_time_raw, dtype=np.float64, count=_OUT_LEN)
    _OUT_SHADOW = np.frombuffer(out_shadow_raw, dtype=np.uint8, count=_OUT_LEN)
    _OUT_LAYOVER = np.frombuffer(out_layover_raw, dtype=np.uint8, count=_OUT_LEN)

    if dem_gt is not None:
        _DEM_GT = tuple(dem_gt)
    if dem_ncols is not None:
        _DEM_NCOLS = int(dem_ncols)

    # 非 fork 情况下，输入大数组需要通过 shared_memory 显式传递
    if in_lat_shm_name and in_lon_shm_name and in_h_shm_name:
        from multiprocessing.shared_memory import SharedMemory
        import atexit
        dt = np.dtype(in_dtype)
        shm_lat = SharedMemory(name=in_lat_shm_name)
        shm_lon = SharedMemory(name=in_lon_shm_name)
        shm_h = SharedMemory(name=in_h_shm_name)
        _IN_SHMS = (shm_lat, shm_lon, shm_h)
        _IN_LAT_FLAT = np.ndarray((_OUT_LEN,), dtype=dt, buffer=shm_lat.buf)
        _IN_LON_FLAT = np.ndarray((_OUT_LEN,), dtype=dt, buffer=shm_lon.buf)
        _IN_H_FLAT = np.ndarray((_OUT_LEN,), dtype=dt, buffer=shm_h.buf)

        def _close_in_shm():
            try:
                for s in _IN_SHMS or ():
                    s.close()
            except Exception:
                pass

        atexit.register(_close_in_shm)
    elif in_h_shm_name:
        # 仅共享 dem_h（lat/lon 在 worker 内按 geotransform 生成）
        from multiprocessing.shared_memory import SharedMemory
        import atexit
        dt = np.dtype(in_dtype)
        shm_h = SharedMemory(name=in_h_shm_name)
        _IN_SHMS = (shm_h,)
        _IN_H_FLAT = np.ndarray((_OUT_LEN,), dtype=dt, buffer=shm_h.buf)

        def _close_in_shm():
            try:
                for s in _IN_SHMS or ():
                    s.close()
            except Exception:
                pass

        atexit.register(_close_in_shm)

def process_block(args):
    """处理单个数据块（只传 start/end，输入数组由 fork 继承）"""
    start, end = args
    # 优先使用共享输入数组；lat/lon 可以在这里按 geotransform 动态生成
    if _IN_H_FLAT is None:
        raise RuntimeError("worker 未初始化输入 dem_h 数组")
    h_block = _IN_H_FLAT[start:end]

    if (_IN_LAT_FLAT is not None) and (_IN_LON_FLAT is not None):
        lat_block = _IN_LAT_FLAT[start:end]
        lon_block = _IN_LON_FLAT[start:end]
    else:
        # 按 geotransform + flat index 生成 lat/lon（避免全量 DEM_lat/DEM_lon 网格）
        if (_DEM_GT is None) or (_DEM_NCOLS <= 0):
            raise RuntimeError("worker 缺少 DEM geotransform 或 ncols，无法生成 lat/lon")
        gt = _DEM_GT
        ncols = _DEM_NCOLS
        idx = np.arange(start, end, dtype=np.int64)
        rr = idx // ncols
        cc = idx - rr * ncols
        # 像素中心
        cc = cc.astype(np.float64) + 0.5
        rr = rr.astype(np.float64) + 0.5
        lon_block = (gt[0] + gt[1] * cc + gt[2] * rr).astype(np.float32)
        lat_block = (gt[3] + gt[4] * cc + gt[5] * rr).astype(np.float32)

    # 使用向量化处理
    r, a, slant, az_time, shadow, layover = worker_geo.geo2rdr_vectorized(lat_block, lon_block, h_block)

    # 写入共享输出（每个 worker 写 disjoint slice，无需锁）
    _OUT_RANGE[start:end] = r.astype(np.float32, copy=False)
    _OUT_AZ[start:end] = a.astype(np.float32, copy=False)
    _OUT_SLANT[start:end] = slant.astype(np.float32, copy=False)
    _OUT_AZ_TIME[start:end] = az_time
    _OUT_SHADOW[start:end] = shadow.astype(np.uint8, copy=False)
    _OUT_LAYOVER[start:end] = layover.astype(np.uint8, copy=False)
    return start, end

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

    print(f"并行分块处理DEM数据，共{num_blocks}块，每块{block_size}点")

    # 输入展平（fork 模式下可零拷贝继承；spawn 模式会退化为慢路径）
    lat_flat = lat.reshape(-1)
    lon_flat = lon.reshape(-1)
    h_flat = h.reshape(-1)

    # 共享内存输出（避免每个块把大数组 pickling 回主进程）
    from multiprocessing.sharedctypes import RawArray
    out_range_raw = RawArray('f', total_points)    # float32
    out_az_raw = RawArray('f', total_points)       # float32
    out_slant_raw = RawArray('f', total_points)    # float32
    out_az_time_raw = RawArray('d', total_points)  # float64
    out_shadow_raw = RawArray('B', total_points)   # uint8
    out_layover_raw = RawArray('B', total_points)  # uint8

    # 准备任务（只传索引范围）
    tasks = [(i * block_size, min((i + 1) * block_size, total_points)) for i in range(num_blocks)]

    # 并行处理
    from multiprocessing import Pool, get_start_method
    start_method = get_start_method()
    cpu_count = os.cpu_count() or 1
    used_cpu_count = max(1, int(cpu_count * 0.8))
    print(f"使用 {used_cpu_count} 个CPU核心并行处理 (start_method={start_method})")

    # fork 下：输入数组通过全局变量继承，速度快且不额外拷贝内存
    global _IN_LAT_FLAT, _IN_LON_FLAT, _IN_H_FLAT
    if start_method == 'fork':
        _IN_LAT_FLAT = lat_flat
        _IN_LON_FLAT = lon_flat
        _IN_H_FLAT = h_flat
        yaml_file = geo.yaml_file if hasattr(geo, 'yaml_file') else None
        if yaml_file is None:
            raise ValueError("Geo2Rdr 缺少 yaml_file，无法在 worker 进程中重建参数")

        with Pool(
            processes=used_cpu_count,
            initializer=init_worker,
            initargs=(yaml_file, out_range_raw, out_az_raw, out_slant_raw, out_az_time_raw, out_shadow_raw, out_layover_raw, total_points),
        ) as pool:
            # 只需要触发执行；返回的 (start,end) 无需处理
            for _ in pool.imap_unordered(process_block, tasks, chunksize=1):
                pass
    else:
        # spawn/forkserver 下：用 shared_memory 传递输入大数组（一次拷贝，后续各进程零拷贝读）
        print("start_method != fork：启用 shared_memory 传递输入数组，避免每个任务反复序列化大块数据。")
        from multiprocessing.shared_memory import SharedMemory

        # 尽量用 float32 降低共享内存占用
        lat_shm_arr = np.asarray(lat_flat, dtype=np.float32, order='C')
        lon_shm_arr = np.asarray(lon_flat, dtype=np.float32, order='C')
        h_shm_arr = np.asarray(h_flat, dtype=np.float32, order='C')

        shm_lat = SharedMemory(create=True, size=lat_shm_arr.nbytes)
        shm_lon = SharedMemory(create=True, size=lon_shm_arr.nbytes)
        shm_h = SharedMemory(create=True, size=h_shm_arr.nbytes)

        try:
            np.ndarray(lat_shm_arr.shape, dtype=lat_shm_arr.dtype, buffer=shm_lat.buf)[:] = lat_shm_arr
            np.ndarray(lon_shm_arr.shape, dtype=lon_shm_arr.dtype, buffer=shm_lon.buf)[:] = lon_shm_arr
            np.ndarray(h_shm_arr.shape, dtype=h_shm_arr.dtype, buffer=shm_h.buf)[:] = h_shm_arr

            yaml_file = geo.yaml_file if hasattr(geo, 'yaml_file') else None
            if yaml_file is None:
                raise ValueError("Geo2Rdr 缺少 yaml_file，无法在 worker 进程中重建参数")

            with Pool(
                processes=used_cpu_count,
                initializer=init_worker,
                initargs=(
                    yaml_file,
                    out_range_raw,
                    out_az_raw,
                    out_slant_raw,
                    out_az_time_raw,
                    out_shadow_raw,
                    out_layover_raw,
                    total_points,
                    shm_lat.name,
                    shm_lon.name,
                    shm_h.name,
                    None,  # dem_gt
                    None,  # dem_ncols
                    "float32",
                ),
            ) as pool:
                for _ in pool.imap_unordered(process_block, tasks, chunksize=1):
                    pass
        finally:
            # 父进程负责 unlink；子进程仅 close（由 OS 回收）
            shm_lat.close()
            shm_lon.close()
            shm_h.close()
            shm_lat.unlink()
            shm_lon.unlink()
            shm_h.unlink()

    # 组装输出 numpy view
    # 注意：这里返回的是 shared buffer 的 numpy view（不会额外拷贝一份全量数组）
    range_pixel = np.frombuffer(out_range_raw, dtype=np.float32, count=total_points)
    azimuth_pixel = np.frombuffer(out_az_raw, dtype=np.float32, count=total_points)
    slant_range = np.frombuffer(out_slant_raw, dtype=np.float32, count=total_points)
    azimuth_time = np.frombuffer(out_az_time_raw, dtype=np.float64, count=total_points)
    shadow = np.frombuffer(out_shadow_raw, dtype=np.uint8, count=total_points).astype(bool)
    layover = np.frombuffer(out_layover_raw, dtype=np.uint8, count=total_points).astype(bool)

    # 更新 SAR DEM（一次性向量化写入，比逐块 list.extend/循环更快）
    r_rounded = np.rint(range_pixel).astype(np.int32, copy=False)
    a_rounded = np.rint(azimuth_pixel).astype(np.int32, copy=False)
    mask = (r_rounded >= 0) & (r_rounded < sar_range_size) & (a_rounded >= 0) & (a_rounded < sar_az_size)
    if np.any(mask):
        sar_dem[a_rounded[mask], r_rounded[mask]] = h_flat[mask].astype(np.float32, copy=False)

    return sar_dem, range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover


def process_dem_in_blocks_parallel_gt(geo, dem_h, dem_gt, dem_shape, block_size=None):
    """
    并行分块处理 DEM（不构造 DEM_lat/DEM_lon 2D 网格）
    worker 内按 geotransform + flat index 动态生成 lat/lon。
    :param geo: Geo2Rdr
    :param dem_h: DEM 高程 (2D)
    :param dem_gt: geotransform(6)
    :param dem_shape: (nrows, ncols)
    """
    nrows, ncols = dem_shape
    total_points = int(dem_h.size)

    if block_size is None:
        # 默认按点数控制
        block_size = 100000
    block_size = max(1000, int(block_size))
    num_blocks = (total_points + block_size - 1) // block_size

    # SAR 尺寸
    sar_az_size = geo.azimuth_lines or 3645
    sar_range_size = geo.range_samples or 3136
    sar_dem = np.full((sar_az_size, sar_range_size), np.nan, dtype=np.float32)

    print(f"并行分块处理DEM数据（GT模式）：共{num_blocks}块，每块{block_size}点")

    # 输入：仅共享 dem_h.flatten()
    h_flat = np.asarray(dem_h, dtype=np.float32, order='C').reshape(-1)

    # 输出共享内存
    from multiprocessing.sharedctypes import RawArray
    out_range_raw = RawArray('f', total_points)
    out_az_raw = RawArray('f', total_points)
    out_slant_raw = RawArray('f', total_points)
    out_az_time_raw = RawArray('d', total_points)
    out_shadow_raw = RawArray('B', total_points)
    out_layover_raw = RawArray('B', total_points)

    tasks = [(i * block_size, min((i + 1) * block_size, total_points)) for i in range(num_blocks)]

    from multiprocessing import Pool, get_start_method
    start_method = get_start_method()
    cpu_count = os.cpu_count() or 1
    used_cpu_count = max(1, int(cpu_count * 0.8))
    print(f"使用 {used_cpu_count} 个CPU核心并行处理 (start_method={start_method})")

    global _IN_H_FLAT, _DEM_GT, _DEM_NCOLS
    _DEM_GT = tuple(dem_gt)
    _DEM_NCOLS = int(ncols)

    if start_method == 'fork':
        _IN_H_FLAT = h_flat
        yaml_file = geo.yaml_file if hasattr(geo, 'yaml_file') else None
        if yaml_file is None:
            raise ValueError("Geo2Rdr 缺少 yaml_file，无法在 worker 进程中重建参数")

        with Pool(
            processes=used_cpu_count,
            initializer=init_worker,
            initargs=(
                yaml_file,
                out_range_raw,
                out_az_raw,
                out_slant_raw,
                out_az_time_raw,
                out_shadow_raw,
                out_layover_raw,
                total_points,
                None,  # in_lat_shm_name
                None,  # in_lon_shm_name
                None,  # in_h_shm_name
                dem_gt,
                ncols,
                "float32",
            ),
        ) as pool:
            for _ in pool.imap_unordered(process_block, tasks, chunksize=1):
                pass
    else:
        # forkserver/spawn：用 shared_memory 传递 dem_h（lat/lon 由 worker 动态生成）
        from multiprocessing.shared_memory import SharedMemory
        print("start_method != fork：启用 shared_memory 传递 dem_h（lat/lon 在 worker 内生成）")
        shm_h = SharedMemory(create=True, size=h_flat.nbytes)
        try:
            np.ndarray(h_flat.shape, dtype=h_flat.dtype, buffer=shm_h.buf)[:] = h_flat
            yaml_file = geo.yaml_file if hasattr(geo, 'yaml_file') else None
            if yaml_file is None:
                raise ValueError("Geo2Rdr 缺少 yaml_file，无法在 worker 进程中重建参数")

            with Pool(
                processes=used_cpu_count,
                initializer=init_worker,
                initargs=(
                    yaml_file,
                    out_range_raw,
                    out_az_raw,
                    out_slant_raw,
                    out_az_time_raw,
                    out_shadow_raw,
                    out_layover_raw,
                    total_points,
                    None,  # in_lat_shm_name
                    None,  # in_lon_shm_name
                    shm_h.name,
                    dem_gt,
                    ncols,
                    "float32",
                ),
            ) as pool:
                for _ in pool.imap_unordered(process_block, tasks, chunksize=1):
                    pass
        finally:
            shm_h.close()
            shm_h.unlink()

    range_pixel = np.frombuffer(out_range_raw, dtype=np.float32, count=total_points)
    azimuth_pixel = np.frombuffer(out_az_raw, dtype=np.float32, count=total_points)
    slant_range = np.frombuffer(out_slant_raw, dtype=np.float32, count=total_points)
    azimuth_time = np.frombuffer(out_az_time_raw, dtype=np.float64, count=total_points)
    shadow = np.frombuffer(out_shadow_raw, dtype=np.uint8, count=total_points).astype(bool)
    layover = np.frombuffer(out_layover_raw, dtype=np.uint8, count=total_points).astype(bool)

    # 一次性向量化写入 SAR DEM
    r_rounded = np.rint(range_pixel).astype(np.int32, copy=False)
    a_rounded = np.rint(azimuth_pixel).astype(np.int32, copy=False)
    mask = (r_rounded >= 0) & (r_rounded < sar_range_size) & (a_rounded >= 0) & (a_rounded < sar_az_size)
    if np.any(mask):
        sar_dem[a_rounded[mask], r_rounded[mask]] = h_flat[mask]

    return sar_dem, range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DEM -> SAR 模拟工具')
    parser.add_argument('--yaml', type=str, required=True, help='YAML 配置文件路径')
    parser.add_argument('--dem', type=str, required=True, help='DEM 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--bbox', type=float, nargs=4, help='边界框 [min_lon, min_lat, max_lon, max_lat]')
    parser.add_argument('--real_sar', type=str, help='真实SAR数据文件路径')
    # 保存 SAR lat/lon 网格（用于 ll2sar/sar2ll 的双向转换）
    parser.set_defaults(save_sar_latlon=True)
    parser.add_argument('--no-save-sar-latlon', dest='save_sar_latlon', action='store_false',
                        help='不在 HDF 中保存 sar_lat/sar_lon（lat_grid/lon_grid）')
    parser.add_argument('--sar-latlon-dtype', default='float32', choices=['float32', 'float64'],
                        help='sar_lat/sar_lon 的 dtype（默认 float32，体积更小）')
    parser.add_argument('--sar-latlon-compress', default='lzf', choices=['lzf', 'gzip', 'none'],
                        help='sar_lat/sar_lon 的压缩方式（默认 lzf，较快）')
    parser.add_argument('--sar-latlon-chunk', type=int, nargs=2, default=[512, 512],
                        help='sar_lat/sar_lon 的 HDF chunk 大小，例如 512 512')
    parser.add_argument('--sar-latlon-block-points', type=int, default=2_000_000,
                        help='写 sar_lat/sar_lon 时扫描 DEM 点的分块大小（点数）')
    
    args = parser.parse_args()
    
    # 读取 DEM
    print(f"读取 DEM 文件: {args.dem}")
    bbox = args.bbox if args.bbox else None
    bbox_used = None  # 记录最终用于裁剪 DEM 的 bbox（若为 None 表示使用全 DEM）

    # 如果用户未显式指定 bbox，且 YAML 里有四角点，则优先按角点外扩裁剪 DEM（显著减少数据量）
    if bbox is None:
        try:
            with open(args.yaml, 'r') as f:
                ycfg = yaml.safe_load(f)
            corners = ycfg.get('corner_coordinates', None)
            if isinstance(corners, dict) and corners:
                lats = []
                lons = []
                for _, c in corners.items():
                    if isinstance(c, dict) and ('lat' in c) and ('lon' in c):
                        lats.append(float(c['lat']))
                        lons.append(float(c['lon']))
                if lats and lons:
                    min_lat, max_lat = min(lats), max(lats)
                    min_lon, max_lon = min(lons), max(lons)
                    margin_deg = 0.1
                    bbox = [min_lon - margin_deg, min_lat - margin_deg, max_lon + margin_deg, max_lat + margin_deg]
                    print(f"从 YAML 角点自动裁剪 DEM bbox: {bbox}")
        except Exception as e:
            print(f"从 YAML 自动提取 bbox 失败，改为全 DEM 处理: {e}")

    bbox_used = bbox
    dem_h, dem_gt, dem_shape = read_dem_gt(args.dem, bbox_used)
    
    # 初始化 Geo2Rdr
    print(f"初始化 Geo2Rdr，使用配置文件: {args.yaml}")
    # 并行分块计算时不需要把全 DEM 预计算为 ECEF（会极耗内存/耗时）
    geo2rdr = Geo2Rdr(args.yaml, dem_h=dem_h, dem_gt=dem_gt, dem_shape=dem_shape, precompute_ecef=False)
    
    # 批量处理 - 使用并行分块处理
    print("开始批量处理...")
    start_time = time.time()
    
    # 生成模拟SAR DEM
    print("生成模拟SAR DEM...")
    sar_dem, range_pixel, azimuth_pixel, slant_range, azimuth_time, shadow, layover = process_dem_in_blocks_parallel_gt(
        geo2rdr, dem_h, dem_gt, dem_shape
    )

    # 关键诊断：统计命中像素数（命中为 0 时，simulate_slc 会全 0）
    sar_az_size = geo2rdr.azimuth_lines or 3645
    sar_range_size = geo2rdr.range_samples or 3136
    valid = (
        np.isfinite(range_pixel) & np.isfinite(azimuth_pixel) &
        (range_pixel >= 0) & (range_pixel < sar_range_size) &
        (azimuth_pixel >= 0) & (azimuth_pixel < sar_az_size)
    )
    hit = int(np.sum(valid))
    print(f"geo2rdr 有效命中像素数: {hit} / {range_pixel.size}")
    if hit == 0:
        print("警告：没有任何 DEM 点落入 SAR 图像范围，simulate_slc 将为全 0。")
        print(f"  near_range={geo2rdr.near_range}  range_spacing={geo2rdr.range_spacing}  prf={geo2rdr.prf}")
        print(f"  t0={geo2rdr.t0}  orbit.tmin={geo2rdr.orbit.tmin}  orbit.tmax={geo2rdr.orbit.tmax}")
    
    # range_pixel/azimuth_pixel/slant_range 是“每个 DEM 点 → SAR 像素”的 1D 映射表，
    # 不应强行 reshape 成 SAR 图像网格（那会打乱对应关系，导致后续偏移估计/校正失真）。
    
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
    nonzero = int(np.count_nonzero(np.abs(slc) > 0))
    print(f"模拟SLC非零像素数: {nonzero} / {slc.size}")
    
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
    # 说明：
    # 旧流程把“DEM->SAR 的 1D 映射表”强行 reshape 成 SAR 网格后，再与真实像素网格做互相关估计偏移。
    # 这在几何上是不成立的，且会引入非常大的内存与计算开销。
    # 这里改为优先使用 coarse_registration 的结果作为偏移量（如果未提供真实 SAR，则偏移为 0）。
    print("计算偏移量并更新几何参数...")
    range_offset, azimuth_offset = coarse_range_offset, coarse_azimuth_offset
    
    # 更新几何参数
    geo2rdr.update_geometry(range_offset, azimuth_offset)
    
    # 重新生成模拟SAR DEM（使用更新后的参数）
    print("使用更新后的几何参数重新生成模拟SAR DEM...")
    sar_dem_updated, range_pixel_updated, azimuth_pixel_updated, slant_range_updated, azimuth_time_updated, shadow_updated, layover_updated = process_dem_in_blocks_parallel_gt(
        geo2rdr, dem_h, dem_gt, dem_shape
    )
    
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
        # 一致性检查：确保 DEM 与映射表长度一致
        try:
            if tuple(dem_h.shape) != tuple(dem_shape):
                raise ValueError(f"dem_h.shape={dem_h.shape} 与 dem_shape={dem_shape} 不一致")
            if int(range_pixel_updated.size) != int(dem_h.size):
                raise ValueError(f"range_pixel.size={range_pixel_updated.size} 与 dem_h.size={dem_h.size} 不一致")
            if int(azimuth_pixel_updated.size) != int(dem_h.size):
                raise ValueError(f"azimuth_pixel.size={azimuth_pixel_updated.size} 与 dem_h.size={dem_h.size} 不一致")
        except Exception as e:
            print(f"警告：DEM/映射表一致性检查失败: {e}")

        f.create_dataset('range_pixel', data=range_pixel_updated)
        f.create_dataset('azimuth_pixel', data=azimuth_pixel_updated)
        f.create_dataset('slant_range', data=slant_range_updated)
        f.create_dataset('azimuth_time', data=azimuth_time_updated)
        f.create_dataset('shadow', data=shadow_updated)
        f.create_dataset('layover', data=layover_updated)
        # SLC（拆分实部/虚部，便于单文件管理）
        # 形状为 (sar_az_size, sar_range_size)，dtype=float32
        f.create_dataset('slc_real', data=np.real(slc_updated).astype(np.float32, copy=False))
        f.create_dataset('slc_imag', data=np.imag(slc_updated).astype(np.float32, copy=False))
        # DEM：只保存高程 + geotransform + shape（避免写入巨大的 dem_lat/dem_lon 2D 网格）
        dem_h_c = np.asarray(dem_h, dtype=np.float32, order='C')
        f.create_dataset('dem_h', data=dem_h_c)
        f.create_dataset('dem_geotransform', data=np.array(dem_gt, dtype=np.float64))
        f.create_dataset('dem_shape', data=np.array(dem_shape, dtype=np.int32))
        # 记录 DEM 裁剪信息（若 bbox_used 为 None 表示未裁剪）
        if 'dem_crop_bbox' not in f:
            if bbox_used is None:
                f.create_dataset('dem_crop_bbox', data=np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64))
            else:
                f.create_dataset('dem_crop_bbox', data=np.array(bbox_used, dtype=np.float64))
        # NumPy 2.0 移除了 np.string_，这里用 bytes 存储路径（UTF-8）
        f.create_dataset('dem_source_file', data=np.bytes_(str(args.dem)))
        # 添加模拟SAR DEM
        sar_dem_ds = f.create_dataset('sar_dem', data=sar_dem_updated)
        # 在 sar_dem 上挂载其对应的 DEM 信息，便于后续追溯（裁剪后的 DEM）
        try:
            sar_dem_ds.attrs['dem_geotransform'] = np.array(dem_gt, dtype=np.float64)
            sar_dem_ds.attrs['dem_shape'] = np.array(dem_shape, dtype=np.int32)
            sar_dem_ds.attrs['dem_crop_bbox'] = np.array(bbox_used if bbox_used is not None else [np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        except Exception:
            pass
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

        # 在 HDF 中保存 SAR 网格下的经纬度（sar_lat/sar_lon；并兼容 lat_grid/lon_grid 名称）
        if getattr(args, "save_sar_latlon", True):
            try:
                print("写入 sar_lat/sar_lon（lat_grid/lon_grid）到 HDF...")
                write_sar_latlon_grids_from_dem_mapping(
                    f,
                    dem_gt=dem_gt,
                    dem_shape=dem_shape,
                    range_pixel=range_pixel_updated,
                    azimuth_pixel=azimuth_pixel_updated,
                    sar_shape=(sar_az_size, sar_range_size),
                    dtype=str(getattr(args, "sar_latlon_dtype", "float32")),
                    compress=str(getattr(args, "sar_latlon_compress", "lzf")),
                    chunk=tuple(int(x) for x in getattr(args, "sar_latlon_chunk", [512, 512])),
                    block_points=int(getattr(args, "sar_latlon_block_points", 2_000_000)),
                )
            except Exception as e:
                print(f"警告：写入 sar_lat/sar_lon 失败，将继续保存其他数据。原因: {e}")
    
    print("完成！")

if __name__ == '__main__':
    main()
