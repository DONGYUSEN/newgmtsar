import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS, Transformer
import os
import yaml
import tempfile
from typing import Optional, Tuple


def _require_scipy():
    try:
        from scipy.interpolate import griddata  # type: ignore
        from scipy.spatial import cKDTree  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少 SciPy：griddata/backward 模式需要 scipy。\n"
            "建议：优先使用 algo=gdal_geoloc（不需要 SciPy，且适合全分辨率 lat/lon 网格）。"
        ) from e
    return griddata, cKDTree


def _require_gdal():
    try:
        from osgeo import gdal, osr  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少 GDAL Python 绑定 (osgeo)。algo=gdal_geoloc 需要它。\n"
            "如果你用 conda：`conda install -c conda-forge gdal`"
        ) from e
    # 显式开启异常，避免 GDAL 4.0 行为变化，同时让错误可捕获而不是潜在崩溃。
    try:
        gdal.UseExceptions()
        osr.UseExceptions()
    except Exception:
        pass
    return gdal, osr

def geocode_sar(
    amp_or_ifg,
    lat,
    lon,
    out_file="geocoded.tif",
    output_crs="EPSG:4326",
    res=None,
    interp="bilinear",
    fill_nan_with_nearest=True,
    algo="backward",
    chunk_rows=256,
):
    """
    将 SAR 幅度或干涉图投影到规则经纬度或 UTM 网格并保存为 GeoTIFF

    Parameters
    ----------
    amp_or_ifg : np.ndarray
        SAR幅度或干涉图 (二维实数或复数)
    lat, lon : np.ndarray
        对应每个SAR像素的纬度和经度 (二维数组)
    out_file : str
        输出 GeoTIFF 文件路径
    output_crs : str
        输出投影，例如 'EPSG:4326'、'EPSG:32650' 或 'UTM'
        如果为 'UTM'，将根据数据中心经度自动计算 UTM 带（EPSG:326xx/327xx）
        如果为 'LATLON'，等价于 'EPSG:4326'
    res : float
        输出网格分辨率。
        - 若输出为 UTM/投影坐标（单位米），res 表示米。
        - 若输出为 EPSG:4326（单位度），res 表示度。
        经验规则：当 output_crs 为 EPSG:4326/LATLON 且 res >= 0.1 时，把 res 解释为“米”，并自动换算成度。
    interp : str
        插值方法（推荐 bilinear）：
        - bilinear：线性插值（在散点三角网格上分片线性，效果接近双线性；更稳，不易过冲）
        - nearest：最近邻（最快，但更粗糙）
        - cubic：三次插值（更平滑，但可能出现过冲/振铃，误差更大）
    fill_nan_with_nearest : bool
        若插值结果存在 NaN，是否用 nearest 再补一次（推荐 True）
    algo : str
        地理编码算法（默认 backward，推荐避免 griddata 的误差/瓶颈）：
        - backward：输出网格逐像素反查最近 SAR 像素索引 + 局部牛顿反解像素坐标 + 双线性采样
        - griddata：散点插值（Delaunay 三角剖分），点数大时非常慢且易过冲
        - gdal_geoloc：使用 GDAL geolocation arrays 做严格反向重采样（推荐全分辨率 lat/lon 网格）
    chunk_rows : int
        输出 GeoTIFF 的行块大小。backward 会分块计算并写盘，避免一次性占用大量内存。
    """

    assert amp_or_ifg.shape == lat.shape == lon.shape, "amp/lat/lon shape mismatch"

    # 复数处理
    is_complex = np.iscomplexobj(amp_or_ifg)
    if is_complex:
        values_real = np.real(amp_or_ifg).flatten()
        values_imag = np.imag(amp_or_ifg).flatten()
    else:
        values = amp_or_ifg.flatten()

    # CRS 别名
    if output_crs.upper() == "LATLON":
        output_crs = "EPSG:4326"

    # 处理 UTM 自动选择
    if output_crs.upper() == 'UTM':
        # 计算中心点经度，确定 UTM 带
        lon_mean = np.mean(lon)
        utm_zone = int(np.floor((lon_mean + 180) / 6) + 1)
        # 确定北半球或南半球
        lat_mean = np.mean(lat)
        hemisphere = 'north' if lat_mean >= 0 else 'south'
        # 构建 EPSG 代码
        if hemisphere == 'north':
            epsg_code = 32600 + utm_zone
        else:
            epsg_code = 32700 + utm_zone
        output_crs = f'EPSG:{epsg_code}'
        print(f"自动选择 UTM 带: {utm_zone}, 半球: {hemisphere}, EPSG: {output_crs}")

    lat_mean = float(np.mean(lat))
    is_geographic_out = output_crs.upper() in ("EPSG:4326", "WGS84")

    # 投影转换
    crs_out = CRS.from_string(output_crs)
    transformer = Transformer.from_crs("EPSG:4326", crs_out, always_xy=True)
    x_out, y_out = transformer.transform(lon, lat)

    # 输出网格范围
    x_min, x_max = x_out.min(), x_out.max()
    y_min, y_max = y_out.min(), y_out.max()

    # 输出分辨率
    if res is None:
        # 默认 5 米（符合常用需求）；对 EPSG:4326 会自动换算成度
        res = 5.0
    res = float(res)
    if is_geographic_out and res >= 0.1:
        # EPSG:4326/LATLON 下，res 以“米”理解更符合习惯
        res_y = res / 111_320.0
        res_x = res / (111_320.0 * max(0.1, np.cos(np.deg2rad(lat_mean))))
    else:
        res_x = res_y = res

    # 生成规则网格（仅保存 1D 坐标，避免 meshgrid 巨大内存）
    grid_x = np.arange(x_min, x_max, res_x, dtype=np.float64)
    grid_y = np.arange(y_max, y_min, -res_y, dtype=np.float64)  # 注意 Y 从上到下
    out_h = int(grid_y.size)
    out_w = int(grid_x.size)

    # 写入 GeoTIFF（分块写出，避免一次性占用大内存）
    transform = from_origin(x_min, y_max, res_x, res_y)
    # 复数用“实部/虚部两波段 float32”保存（更通用）
    dtype = "float32"
    count = 2 if is_complex else 1

    algo = (algo or "backward").lower()
    interp = (interp or "bilinear").lower()

    if interp == "bilinear":
        interp_mode = "bilinear"
    elif interp == "nearest":
        interp_mode = "nearest"
    elif interp == "cubic":
        interp_mode = "cubic"
    else:
        raise ValueError(f"不支持的插值方法: {interp}（可选 bilinear/nearest/cubic）")

    if algo not in ("backward", "griddata"):
        raise ValueError("algo 必须是 backward 或 griddata（gdal_geoloc 请走命令行文件模式）")

    # backward 算法需要把 SAR 像素坐标点建索引；点数过大将不可用
    n_points = int(x_out.size)
    if algo == "backward" and n_points > 10_000_000:
        raise ValueError(
            f"lat/lon 网格点数 {n_points:,} 太大，不适合 backward 反查索引。\n"
            "建议：使用 dem2sar 输出降采样 lat/lon 网格（--geocode-step>1），并在 geocoding 前用 meta 自动抽样 amp 对齐。"
        )

    def _bilinear_sample(img, u, v):
        h, w = img.shape
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        j0 = np.floor(u).astype(np.int64)
        i0 = np.floor(v).astype(np.int64)
        j0 = np.clip(j0, 0, w - 2)
        i0 = np.clip(i0, 0, h - 2)
        du = u - j0
        dv = v - i0
        p00 = img[i0, j0]
        p01 = img[i0, j0 + 1]
        p10 = img[i0 + 1, j0]
        p11 = img[i0 + 1, j0 + 1]
        return (
            (1 - du) * (1 - dv) * p00
            + du * (1 - dv) * p01
            + (1 - du) * dv * p10
            + du * dv * p11
        )

    def _invert_xy_to_uv(x_img, y_img, x_t, y_t, u0, v0, n_iter=3):
        """局部牛顿：在规则像素网格上，用 x(u,v),y(u,v) 的双线性插值反解 u,v。"""
        h, w = x_img.shape
        u = u0.astype(np.float64).copy()
        v = v0.astype(np.float64).copy()
        x_t = x_t.astype(np.float64)
        y_t = y_t.astype(np.float64)
        for _ in range(n_iter):
            j0 = np.floor(u).astype(np.int64)
            i0 = np.floor(v).astype(np.int64)
            j0 = np.clip(j0, 0, w - 2)
            i0 = np.clip(i0, 0, h - 2)
            du = u - j0
            dv = v - i0

            x00 = x_img[i0, j0]
            x01 = x_img[i0, j0 + 1]
            x10 = x_img[i0 + 1, j0]
            x11 = x_img[i0 + 1, j0 + 1]
            y00 = y_img[i0, j0]
            y01 = y_img[i0, j0 + 1]
            y10 = y_img[i0 + 1, j0]
            y11 = y_img[i0 + 1, j0 + 1]

            x_hat = (
                (1 - du) * (1 - dv) * x00
                + du * (1 - dv) * x01
                + (1 - du) * dv * x10
                + du * dv * x11
            )
            y_hat = (
                (1 - du) * (1 - dv) * y00
                + du * (1 - dv) * y01
                + (1 - du) * dv * y10
                + du * dv * y11
            )

            dxdU = (1 - dv) * (x01 - x00) + dv * (x11 - x10)
            dxdV = (1 - du) * (x10 - x00) + du * (x11 - x01)
            dydU = (1 - dv) * (y01 - y00) + dv * (y11 - y10)
            dydV = (1 - du) * (y10 - y00) + du * (y11 - y01)

            rx = x_t - x_hat
            ry = y_t - y_hat
            det = dxdU * dydV - dxdV * dydU
            good = np.abs(det) > 1e-12
            if not np.any(good):
                break

            du_step = np.zeros_like(u)
            dv_step = np.zeros_like(v)
            du_step[good] = (rx[good] * dydV[good] - ry[good] * dxdV[good]) / det[good]
            dv_step[good] = (-rx[good] * dydU[good] + ry[good] * dxdU[good]) / det[good]

            # 限制步长，避免跳出局部区域
            du_step = np.clip(du_step, -2.0, 2.0)
            dv_step = np.clip(dv_step, -2.0, 2.0)

            u = u + du_step
            v = v + dv_step
        return u, v

    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=count,
        dtype=dtype,
        crs=output_crs,
        transform=transform,
    ) as dst:
        if algo == "griddata":
            print("使用 griddata 插值（点数大时很慢）...")
            griddata, _ = _require_scipy()
            grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
            points = np.stack([x_out.flatten(), y_out.flatten()], axis=-1)
            if points.shape[0] > 5_000_000:
                print(
                    f"警告：插值点数 {points.shape[0]:,} 很大，griddata 可能非常慢且占用大量内存。\n"
                    "建议：改用 algo=backward，并使用降采样 lat/lon 网格。"
                )
            if interp_mode == "bilinear":
                method = "linear"
            else:
                method = interp_mode

            def _griddata_one(vals, m):
                out = griddata(points, vals, (grid_X, grid_Y), method=m)
                if fill_nan_with_nearest and np.isnan(out).any():
                    out2 = griddata(points, vals, (grid_X, grid_Y), method="nearest")
                    out = np.where(np.isnan(out), out2, out)
                return out

            if is_complex:
                grid_real = _griddata_one(values_real, method)
                grid_imag = _griddata_one(values_imag, method)
                dst.write(grid_real.astype(np.float32), 1)
                dst.write(grid_imag.astype(np.float32), 2)
            else:
                grid_out = _griddata_one(values, method).astype(np.float32)
                dst.write(grid_out, 1)
        else:
            print("使用 backward 反查 + 双线性采样（推荐）...")
            _, cKDTree = _require_scipy()
            # 建 KDTree（默认全点；如需更快可改成对下采样点建树）
            pts = np.stack([x_out.flatten(), y_out.flatten()], axis=-1).astype(np.float32)
            tree = cKDTree(pts)
            h_sar, w_sar = x_out.shape

            chunk_rows = int(chunk_rows)
            if chunk_rows <= 0:
                chunk_rows = 256

            if is_complex:
                amp_r = np.real(amp_or_ifg).astype(np.float32, copy=False)
                amp_i = np.imag(amp_or_ifg).astype(np.float32, copy=False)
            else:
                amp_f = amp_or_ifg.astype(np.float32, copy=False)

            from rasterio.windows import Window

            for row0 in range(0, out_h, chunk_rows):
                row1 = min(row0 + chunk_rows, out_h)
                # 当前块的输出坐标
                ys = grid_y[row0:row1]
                qx = np.tile(grid_x, ys.size).astype(np.float32, copy=False)
                qy = np.repeat(ys, out_w).astype(np.float32, copy=False)
                q = np.stack([qx, qy], axis=-1)
                try:
                    _, nn = tree.query(q, k=1, workers=-1)
                except TypeError:
                    # 兼容老版本 SciPy
                    _, nn = tree.query(q, k=1)
                i_nn = (nn // w_sar).astype(np.float64)
                j_nn = (nn % w_sar).astype(np.float64)

                # 局部反解 u/v
                u, v = _invert_xy_to_uv(x_out, y_out, q[:, 0].astype(np.float64), q[:, 1].astype(np.float64), j_nn, i_nn, n_iter=3)

                # 采样
                if interp_mode == "nearest":
                    ii = np.clip(np.rint(v).astype(np.int64), 0, h_sar - 1)
                    jj = np.clip(np.rint(u).astype(np.int64), 0, w_sar - 1)
                    if is_complex:
                        out_r = amp_r[ii, jj]
                        out_i = amp_i[ii, jj]
                    else:
                        out_v = amp_f[ii, jj]
                else:
                    if is_complex:
                        out_r = _bilinear_sample(amp_r, u, v).astype(np.float32)
                        out_i = _bilinear_sample(amp_i, u, v).astype(np.float32)
                    else:
                        out_v = _bilinear_sample(amp_f, u, v).astype(np.float32)

                if is_complex:
                    win = Window(0, row0, out_w, row1 - row0)
                    dst.write(out_r.reshape((row1 - row0, out_w)), 1, window=win)
                    dst.write(out_i.reshape((row1 - row0, out_w)), 2, window=win)
                else:
                    win = Window(0, row0, out_w, row1 - row0)
                    dst.write(out_v.reshape((row1 - row0, out_w)), 1, window=win)

    print(f"Geocoding 完成，结果保存到 {out_file}")


def read_data_file(file_path):
    """
    读取数据文件，支持 .npy 和 .tif 文件
    
    Parameters
    ----------
    file_path : str
        文件路径，支持 .npy 或 .tif 格式
    
    Returns
    -------
    np.ndarray
        读取的数据数组
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.tif') or file_path.endswith('.tiff'):
        with rasterio.open(file_path) as ds:
            if ds.count >= 2:
                real = ds.read(1).astype(np.float32)
                imag = ds.read(2).astype(np.float32)
                return real + 1j * imag
            return ds.read(1)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def _read_latlon_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    if not isinstance(meta, dict) or "files" not in meta:
        raise ValueError(f"latlon meta 文件格式不正确: {meta_path}")
    shape = meta.get("shape") or {}
    if isinstance(shape, dict):
        nrows = int(shape.get("nrows"))
        ncols = int(shape.get("ncols"))
    else:
        nrows = ncols = None

    files = meta["files"] or {}
    # 兼容两种键：
    # - 新：files.lat / files.lon
    # - 旧：files.lat_bin / files.lon_bin
    lat_name = files.get("lat") or files.get("lat_bin")
    lon_name = files.get("lon") or files.get("lon_bin")
    if lat_name is None or lon_name is None:
        raise ValueError(f"meta 文件缺少 files.lat/files.lon（或 files.lat_bin/files.lon_bin）: {meta_path}")
    base_dir = os.path.dirname(os.path.abspath(meta_path))
    lat_path = os.path.join(base_dir, lat_name)
    lon_path = os.path.join(base_dir, lon_name)
    return meta, (nrows, ncols), lat_path, lon_path


def read_latlon_grid(lat_path, lon_path=None):
    """
    读取 lat/lon 网格，支持：
    - .npy / .tif：直接读取 2D 数组
    - dem2sar 的二进制分块输出：
      - 传入 *_latlon_grid_meta.yaml（推荐）：自动定位并 memmap 两个 .bin
      - 或传入 lat/lon 的 .bin：自动寻找同前缀的 *_latlon_grid_meta.yaml 来获取 shape

    返回：
    - lat, lon: 2D 数组（可能是 np.memmap）
    - meta: dict 或 None（若来自 meta.yaml）
    """
    meta = None

    if lat_path.endswith("_latlon_grid_meta.yaml") or lat_path.endswith("_latlon_grid_meta.yml"):
        meta, shape, lat_bin_path, lon_bin_path = _read_latlon_meta(lat_path)
        # 根据后缀决定读取方式
        if lat_bin_path.endswith(".bin"):
            if shape[0] is None or shape[1] is None:
                raise ValueError(f"meta 缺少 shape，无法 memmap: {lat_path}")
            lat = np.memmap(lat_bin_path, dtype="<f4", mode="r", shape=shape, order="C")
            lon = np.memmap(lon_bin_path, dtype="<f4", mode="r", shape=shape, order="C")
        elif lat_bin_path.endswith((".tif", ".tiff", ".npy")):
            lat = read_data_file(lat_bin_path)
            lon = read_data_file(lon_bin_path)
        else:
            raise ValueError(f"不支持的 lat/lon 文件类型: {lat_bin_path}")
        return lat, lon, meta

    if lat_path.endswith(".bin"):
        # 从 .bin 推断 meta 文件路径
        if "_lat_grid_f32le.bin" in lat_path:
            meta_path = lat_path.replace("_lat_grid_f32le.bin", "_latlon_grid_meta.yaml")
        else:
            base, _ = os.path.splitext(lat_path)
            meta_path = base + "_latlon_grid_meta.yaml"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"找不到 latlon meta 文件（无法从 .bin 推断 shape）: {meta_path}")
        meta, shape, lat_bin_path, lon_bin_path = _read_latlon_meta(meta_path)

        # 若用户同时提供了 lon_path，则优先使用
        if lon_path is not None and lon_path.endswith(".bin"):
            lon_bin_path = lon_path

        lat = np.memmap(lat_bin_path, dtype="<f4", mode="r", shape=shape, order="C")
        lon = np.memmap(lon_bin_path, dtype="<f4", mode="r", shape=shape, order="C")
        return lat, lon, meta

    # 传统路径：lat/lon 单独文件
    if lon_path is None:
        raise ValueError("读取 lat/lon 网格需要同时提供 lon_path（或使用 *_latlon_grid_meta.yaml）")
    lat = read_data_file(lat_path)
    lon = read_data_file(lon_path)
    return lat, lon, meta


def maybe_subsample_amp_to_match_grid(amp, meta, lat_shape):
    """
    当 lat/lon 网格是降采样输出时，根据 meta 里的像素索引从原始 amp 中抽取对应子网格。
    - amp: 原始 SAR 幅度/干涉图数组（2D 或 complex 2D）
    - meta: dem2sar 输出的 *_latlon_grid_meta.yaml 内容
    - lat_shape: (nrows, ncols) 输出网格 shape
    """
    if amp.shape == lat_shape or meta is None:
        return amp
    src_shape = meta.get("source_sar_shape") or {}
    try:
        src_h = int(src_shape.get("nrows"))
        src_w = int(src_shape.get("ncols"))
        if amp.shape != (src_h, src_w):
            print(
                "警告：amp 的 shape 与 meta.source_sar_shape 不一致。\n"
                f"  amp.shape={amp.shape}, meta.source_sar_shape={(src_h, src_w)}\n"
                "这通常意味着：amp 已经被裁剪/多视/重采样过。\n"
                "此时不能直接用原始 pixel_indices 去抽样或会产生公里级偏移；建议重新生成匹配该 amp 网格的 lat/lon。"
            )
    except Exception:
        pass
    pix = meta.get("pixel_indices") or {}
    a_idx = pix.get("azimuth_lines")
    r_idx = pix.get("range_pixels")
    if a_idx is None or r_idx is None:
        raise ValueError("amp/latlon shape 不匹配且 meta 中缺少 pixel_indices，无法自动对齐")
    a_idx = np.asarray(a_idx, dtype=np.int64)
    r_idx = np.asarray(r_idx, dtype=np.int64)
    sub = amp[np.ix_(a_idx, r_idx)]
    if sub.shape != lat_shape:
        raise ValueError(f"根据 meta 索引抽取后的 amp shape={sub.shape} 仍与 lat/lon shape={lat_shape} 不一致")
    return sub


def resolve_latlon_paths(lat_file_or_meta: str, lon_file_or_auto: Optional[str]):
    """
    从命令行参数解析出实际 lat/lon 文件路径（用于 gdal_geoloc）。
    返回: (lat_path, lon_path, meta, shape)
    """
    meta = None
    shape = None

    if lat_file_or_meta.endswith(("_latlon_grid_meta.yaml", "_latlon_grid_meta.yml")):
        meta, shape, lat_path, lon_path = _read_latlon_meta(lat_file_or_meta)
        if lon_file_or_auto is not None and lon_file_or_auto.upper() != "AUTO":
            lon_path = lon_file_or_auto
        return lat_path, lon_path, meta, shape

    if lon_file_or_auto is None or lon_file_or_auto.upper() == "AUTO":
        raise ValueError("当 lat 参数不是 *_latlon_grid_meta.yaml 时，必须显式提供 lon 文件路径（不能用 AUTO）")

    return lat_file_or_meta, lon_file_or_auto, meta, shape


def _sample_latlon_mean(
    lat_path: str,
    lon_path: str,
    stride: int = 200,
    *,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """从 lat/lon 栅格抽样估计均值，用于 UTM 自动分带与经纬度米->度换算。"""
    stride = max(1, int(stride))

    # .bin：用 memmap 抽样，避免 rasterio 尝试打开 raw bin
    if lat_path.endswith(".bin") or lon_path.endswith(".bin"):
        if shape is None or shape[0] is None or shape[1] is None:
            raise ValueError("lat/lon 为 .bin 时需要提供 shape（来自 *_latlon_grid_meta.yaml）")
        lat_mm = np.memmap(lat_path, dtype="<f4", mode="r", shape=shape, order="C")
        lon_mm = np.memmap(lon_path, dtype="<f4", mode="r", shape=shape, order="C")
        lat_s = np.asarray(lat_mm[::stride, ::stride], dtype=np.float64)
        lon_s = np.asarray(lon_mm[::stride, ::stride], dtype=np.float64)
        return float(np.nanmean(lat_s)), float(np.nanmean(lon_s))

    with rasterio.open(lat_path) as ds_lat, rasterio.open(lon_path) as ds_lon:
        h = ds_lat.height
        w = ds_lat.width
        win_h = max(1, h // stride)
        win_w = max(1, w // stride)
        lat = ds_lat.read(
            1,
            out_shape=(win_h, win_w),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float64)
        lon = ds_lon.read(
            1,
            out_shape=(win_h, win_w),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float64)
    return float(np.nanmean(lat)), float(np.nanmean(lon))


def _auto_utm_epsg_from_latlon(lat_path: str, lon_path: str, *, shape: Optional[Tuple[int, int]] = None) -> str:
    lat_mean, lon_mean = _sample_latlon_mean(lat_path, lon_path, stride=250, shape=shape)
    utm_zone = int(np.floor((lon_mean + 180.0) / 6.0) + 1)
    hemisphere_north = lat_mean >= 0
    epsg_code = (32600 + utm_zone) if hemisphere_north else (32700 + utm_zone)
    return f"EPSG:{epsg_code}"


def _make_raw_f32_vrt(bin_path: str, shape: Tuple[int, int], out_vrt: str) -> str:
    """为 float32 little-endian 的 .bin 生成 VRT（GDAL 可读）。"""
    nrows, ncols = int(shape[0]), int(shape[1])
    line_offset = ncols * 4
    # 为了兼容 GDAL 的 VRT RawRasterBand 安全限制：
    # - 尽量让 SourceFilename 是 VRT 的 sibling 文件，并设置 relativeToVRT=1。
    # 这样无需用户额外设置 GDAL_VRT_RAWRASTERBAND_ALLOWED_SOURCE。
    base = os.path.basename(bin_path)
    xml = f"""<VRTDataset rasterXSize="{ncols}" rasterYSize="{nrows}">
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">
    <SourceFilename relativeToVRT="1">{base}</SourceFilename>
    <ByteOrder>LSB</ByteOrder>
    <ImageOffset>0</ImageOffset>
    <PixelOffset>4</PixelOffset>
    <LineOffset>{line_offset}</LineOffset>
  </VRTRasterBand>
</VRTDataset>
"""
    with open(out_vrt, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_vrt


def _stage_bin_as_sibling(src_bin: str, tmp_dir: str) -> str:
    """
    将 raw .bin 以“同目录兄弟文件”的方式提供给 VRT，规避 GDAL RawRasterBand 路径限制。
    默认用 symlink；若失败（极少数系统），回退为复制（可能很慢）。
    """
    src_bin = os.path.abspath(src_bin)
    dst_bin = os.path.join(tmp_dir, os.path.basename(src_bin))
    if os.path.exists(dst_bin):
        return dst_bin
    try:
        os.symlink(src_bin, dst_bin)
    except Exception:
        # 回退复制：大文件会慢，但至少可用
        import shutil
        shutil.copyfile(src_bin, dst_bin)
    return dst_bin


def geocode_gdal_geoloc(
    amp_file: str,
    lat_path: str,
    lon_path: str,
    out_file: str,
    output_crs: str,
    res: Optional[float],
    interp: str,
    *,
    shape: Optional[Tuple[int, int]] = None,
):
    """
    使用 GDAL geolocation arrays（-geoloc）做反向重采样。
    优点：不需要 KDTree，不需要 griddata/Delaunay，适合全分辨率 lat/lon 网格。
    """
    gdal, osr = _require_gdal()

    # CRS 别名
    if output_crs.upper() == "LATLON":
        output_crs = "EPSG:4326"
    if output_crs.upper() == "UTM":
        output_crs = _auto_utm_epsg_from_latlon(lat_path, lon_path, shape=shape)
        print(f"自动选择 UTM EPSG: {output_crs}")

    # 分辨率：默认 5m（与旧逻辑一致）
    if res is None:
        res = 5.0
    res = float(res)

    is_geographic_out = output_crs.upper() in ("EPSG:4326", "WGS84")
    if is_geographic_out and res >= 0.1:
        lat_mean, _ = _sample_latlon_mean(lat_path, lon_path, stride=250, shape=shape)
        res_y = res / 111_320.0
        res_x = res / (111_320.0 * max(0.1, np.cos(np.deg2rad(lat_mean))))
    else:
        res_x = res_y = res

    interp = (interp or "bilinear").lower()
    resample_map = {"nearest": gdal.GRA_NearestNeighbour, "bilinear": gdal.GRA_Bilinear, "cubic": gdal.GRA_Cubic}
    if interp not in resample_map:
        raise ValueError(f"不支持的插值方法: {interp}（可选 bilinear/nearest/cubic）")

    # 支持 .bin：先生成对应 VRT
    tmp_dir = tempfile.mkdtemp(prefix="geoloc_")
    try:
        # lat/lon: 支持 .tif/.tiff/.vrt；若为 .bin 则生成临时 VRT
        lat_ds_path = lat_path
        lon_ds_path = lon_path
        if lat_path.endswith(".bin") or lon_path.endswith(".bin"):
            if shape is None or shape[0] is None or shape[1] is None:
                raise ValueError("lat/lon 为 .bin 时需要从 meta 提供 shape")
            if lat_path.endswith(".bin"):
                lat_vrt = os.path.join(tmp_dir, "lat.vrt")
                lat_bin_local = _stage_bin_as_sibling(lat_path, tmp_dir)
                _make_raw_f32_vrt(lat_bin_local, shape, lat_vrt)
                lat_ds_path = lat_vrt
            if lon_path.endswith(".bin"):
                lon_vrt = os.path.join(tmp_dir, "lon.vrt")
                lon_bin_local = _stage_bin_as_sibling(lon_path, tmp_dir)
                _make_raw_f32_vrt(lon_bin_local, shape, lon_vrt)
                lon_ds_path = lon_vrt

        # amp: algo=gdal_geoloc 需要文件输入；若为 .npy 则先写临时 GeoTIFF
        amp_ds_path = amp_file
        if amp_file.endswith(".npy"):
            arr = np.load(amp_file)
            amp_tif = os.path.join(tmp_dir, "amp.tif")
            if np.iscomplexobj(arr):
                real = np.real(arr).astype(np.float32, copy=False)
                imag = np.imag(arr).astype(np.float32, copy=False)
                profile = dict(driver="GTiff", height=real.shape[0], width=real.shape[1], count=2, dtype="float32")
                with rasterio.open(amp_tif, "w", **profile) as dst:
                    dst.write(real, 1)
                    dst.write(imag, 2)
            else:
                arr_f = arr.astype(np.float32, copy=False)
                profile = dict(driver="GTiff", height=arr_f.shape[0], width=arr_f.shape[1], count=1, dtype="float32")
                with rasterio.open(amp_tif, "w", **profile) as dst:
                    dst.write(arr_f, 1)
            amp_ds_path = amp_tif

        # 基于输入数据创建 VRT，并附加 GEOLOCATION 元数据
        amp_vrt = os.path.join(tmp_dir, "amp.vrt")
        gdal.Translate(amp_vrt, amp_ds_path, format="VRT")
        ds = gdal.Open(amp_vrt, gdal.GA_Update)
        if ds is None:
            raise RuntimeError(f"无法打开输入幅度文件: {amp_file}")

        # GDAL 的 GEOLOCATION.SRS 期望 WKT；传 EPSG:4326 字符串可能触发 "missing [" 并导致崩溃。
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetMetadataItem("SRS", srs.ExportToWkt(), "GEOLOCATION")
        ds.SetMetadataItem("X_DATASET", lon_ds_path, "GEOLOCATION")
        ds.SetMetadataItem("X_BAND", "1", "GEOLOCATION")
        ds.SetMetadataItem("Y_DATASET", lat_ds_path, "GEOLOCATION")
        ds.SetMetadataItem("Y_BAND", "1", "GEOLOCATION")
        ds.SetMetadataItem("PIXEL_OFFSET", "0", "GEOLOCATION")
        ds.SetMetadataItem("LINE_OFFSET", "0", "GEOLOCATION")
        ds.SetMetadataItem("PIXEL_STEP", "1", "GEOLOCATION")
        ds.SetMetadataItem("LINE_STEP", "1", "GEOLOCATION")
        ds.FlushCache()
        ds = None

        # Warp
        # 关键：对 geolocation arrays 的反查变换，GDAL 默认会启用“近似变换器”，
        # 在非线性较强时会表现为分块/错位（看起来像东一块西一块）。
        # 将 errorThreshold 设为 0（或极小）可显著减少这类分块伪影。
        warp_opts = gdal.WarpOptions(
            format="GTiff",
            dstSRS=output_crs,
            xRes=res_x,
            yRes=res_y,
            resampleAlg=resample_map[interp],
            geoloc=True,
            multithread=True,
            errorThreshold=0.0,
            creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
            outputType=gdal.GDT_Float32,
        )
        out = gdal.Warp(out_file, amp_vrt, options=warp_opts)
        if out is None:
            raise RuntimeError("gdal.Warp(geoloc=True) 失败")
        out.FlushCache()
        out = None
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


# ===============================
# 命令行运行示例
# ===============================

if __name__ == "__main__":
    import sys
    import numpy as np
    import time

    # 检查命令行参数
    if len(sys.argv) < 4:
        print("使用方法:")
        print("  python geocoding.py <amp_file> <lat_file_or_meta> <lon_file_or_AUTO> [out_file] [output_crs] [res] [interp] [algo]")
        print("  ")
        print("参数说明:")
        print("  <amp_file>: SAR幅度或干涉图文件 (.npy 或 .tif)")
        print("  <lat_file_or_meta>:")
        print("    - 传统方式：纬度文件 (.npy 或 .tif)")
        print("    - 二进制方式：dem2sar 输出的 *_latlon_grid_meta.yaml（推荐）")
        print("    - 或者：纬度 .bin（需同目录存在 *_latlon_grid_meta.yaml 用于提供 shape）")
        print("  <lon_file_or_AUTO>:")
        print("    - 传统方式：经度文件 (.npy 或 .tif)")
        print("    - 二进制方式：可传 'AUTO'（从 meta 自动定位 lon.bin），或显式传 lon.bin")
        print("  [out_file]: 输出GeoTIFF文件路径 (默认: geocoded.tif)")
        print("  [output_crs]: 输出投影 (默认: LATLON)")
        print("               - LATLON：输出经纬度（等价 EPSG:4326）")
        print("               - UTM：输出 UTM 米制网格，自动计算投影带（EPSG:326xx/327xx）")
        print("               - EPSG:326XX / EPSG:327XX：指定 UTM 带（北/南半球）")
        print("  [res]: 输出网格分辨率 (默认: 自动)")
        print("         - UTM/投影：单位米")
        print("         - LATLON/EPSG:4326：当 res>=0.1 时按“米”理解并自动换算成度（例如 1.5、5）")
        print("  [interp]: 插值方法 (默认: bilinear)")
        print("           可选: bilinear(推荐), nearest, cubic")
        print("  [algo]: 地理编码算法 (默认: backward)")
        print("         - backward：反查最近 SAR 像素 + 局部反解 + 双线性采样（推荐，避免 griddata 瓶颈）")
        print("         - griddata：散点插值（点数大时非常慢且可能过冲）")
        print("         - gdal_geoloc：GDAL geolocation arrays 反向重采样（推荐全分辨率 lat/lon 网格；不建 KDTree）")
        print("  ")
        print("示例:")
        print("  python geocoding.py amp.tif out_latlon_grid_meta.yaml AUTO out_utm.tif UTM 5 bilinear backward")
        print("  python geocoding.py amp.tif out_latlon_grid_meta.yaml AUTO out_ll.tif LATLON 5 bilinear backward")
        print("  python geocoding.py amp.tif lat.tif lon.tif out_ll.tif EPSG:4326 1.5 bilinear backward")
        sys.exit(1)

    # 记录开始时间
    start_time = time.time()

    # 解析命令行参数
    amp_file = sys.argv[1]
    lat_file = sys.argv[2]
    lon_file = sys.argv[3]
    out_file = sys.argv[4] if len(sys.argv) > 4 else "geocoded.tif"
    output_crs = sys.argv[5] if len(sys.argv) > 5 else "LATLON"
    res = None
    interp = "bilinear"
    algo = "backward"
    if len(sys.argv) > 6:
        # 兼容旧用法：第 6 个参数可能是 res（float）
        try:
            res = float(sys.argv[6])
        except ValueError:
            interp = sys.argv[6]
    if len(sys.argv) > 7:
        interp = sys.argv[7]
    if len(sys.argv) > 8:
        algo = sys.argv[8]

    print(f"输入文件:")
    print(f"  幅度/干涉图: {amp_file}")
    print(f"  纬度: {lat_file}")
    print(f"  经度: {lon_file}")
    print(f"输出设置:")
    print(f"  输出文件: {out_file}")
    print(f"  输出投影: {output_crs}")
    print(f"  分辨率: {res if res is not None else '自动'}")
    print(f"  插值方法: {interp}")
    print(f"  算法: {algo}")
    print("")

    algo_l = (algo or "backward").lower()

    # algo=gdal_geoloc：走 GDAL Warp(-geoloc) 文件路径，不把全图读进 numpy，也不建 KDTree
    if algo_l == "gdal_geoloc":
        print("开始地理编码（gdal_geoloc）...")
        lat_path, lon_path, meta, shape = resolve_latlon_paths(lat_file, lon_file)
        if meta is not None:
            try:
                print("lat/lon meta:")
                print(f"  geocode: {meta.get('geocode')}")
                print(f"  source_sar_shape: {meta.get('source_sar_shape')}")
                print(f"  latlon_shape: {meta.get('shape')}")
            except Exception:
                pass
        geocode_gdal_geoloc(amp_file, lat_path, lon_path, out_file, output_crs, res, interp, shape=shape)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n总运行时间: {elapsed_time:.2f} 秒")
        sys.exit(0)

    # 读取数据：amp 支持 .npy/.tif；lat/lon 支持 .npy/.tif 或 dem2sar 的 meta.yaml/.bin
    print("读取数据...")
    amp = read_data_file(amp_file)
    lon_arg = None if lon_file.upper() == "AUTO" else lon_file
    lat, lon, meta = read_latlon_grid(lat_file, lon_arg)
    if meta is not None:
        try:
            print("lat/lon meta:")
            print(f"  geocode: {meta.get('geocode')}")
            print(f"  source_sar_shape: {meta.get('source_sar_shape')}")
            print(f"  latlon_shape: {lat.shape}")
        except Exception:
            pass

    # 如果 lat/lon 网格是降采样输出，自动按 meta 的像素索引抽样 amp 以对齐 shape
    amp = maybe_subsample_amp_to_match_grid(amp, meta, lat.shape)

    print(f"数据形状: {amp.shape}")
    print("")

    # 执行地理编码
    print("开始地理编码...")
    geocode_sar(amp, lat, lon, out_file, output_crs, res, interp=interp, algo=algo)

    # 计算并输出运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")
