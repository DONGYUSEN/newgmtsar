#!/usr/bin/env python3
"""
DEM处理模块
功能：
  1) 从 master.yaml 的角点范围推算 DEM 覆盖范围（可选）
  2) 从 ESA STEP 镜像下载 SRTMGL1 (1 arc-second) 并拼接（可选）
  3) 对 DEM 做裁剪、重采样
  4) 输出经纬度(EPSG:4326) 与 UTM 两种投影的 GeoTIFF + VRT

设计目标：
  - 不引入 SciPy 等重依赖（避免在生产环境报错）
  - 不生成“随机占位 DEM”（这会导致后续所有结果不可用）
  - 支持：当 YAML 没有角点时，仍可对本地 DEM 全量处理
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import time
import urllib.request
import urllib.error
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

_GDAL = None
_OSR = None


def gdal_osr():
    """延迟导入 GDAL Python 绑定，确保 `-h` 等不依赖 GDAL 的场景也能运行。"""
    global _GDAL, _OSR
    if _GDAL is None or _OSR is None:
        try:
            from osgeo import gdal as _gdal, osr as _osr  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "缺少 GDAL Python 绑定 (osgeo)。\n"
                "- 如果你用 conda：`conda install -c conda-forge gdal`\n"
                "- 或系统安装 GDAL 后再装 python 绑定。\n"
                "mkdem.py 需要 GDAL 来裁剪/投影/写 GeoTIFF。"
            ) from e
        _osr.UseExceptions()
        _GDAL, _OSR = _gdal, _osr
    return _GDAL, _OSR

SRTMGL1_BASE_URL = "https://step.esa.int/auxdata/dem/SRTMGL1"
SRTM_NODATA = -32768


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"无法解析为浮点数: {x!r}") from e


def read_master_bbox(yaml_file: str) -> Optional[Tuple[float, float, float, float]]:
    """从master的YAML文件中读取边界框
    
    Args:
        yaml_file: YAML文件路径
        
    Returns:
        (min_lon, max_lon, min_lat, max_lat): 边界框坐标
        如果 YAML 不包含角点字段，则返回 None
    """
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少依赖 PyYAML：请执行 `python3 -m pip install pyyaml`，或使用 --bbox/--src-dem 避免读取 YAML。"
        ) from e

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # 从corner_coordinates中提取边界框
    corners_dict = data.get('corner_coordinates', {})
    if not corners_dict:
        return None
    
    # 提取所有角点的经纬度
    lons = []
    lats = []
    for corner_name, corner_data in corners_dict.items():
        if not isinstance(corner_data, dict):
            raise ValueError(f"corner_coordinates.{corner_name} 不是 dict: {corner_data!r}")
        if 'lon' not in corner_data or 'lat' not in corner_data:
            raise ValueError(f"corner_coordinates.{corner_name} 缺少 lon/lat: {corner_data!r}")
        lons.append(_safe_float(corner_data['lon']))
        lats.append(_safe_float(corner_data['lat']))
    
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    return min_lon, max_lon, min_lat, max_lat


def expand_bbox_km(
    bbox: Tuple[float, float, float, float],
    margin_km: float,
) -> Tuple[float, float, float, float]:
    """按公里扩展经纬度边界框。"""
    if margin_km <= 0:
        return bbox
    min_lon, max_lon, min_lat, max_lat = bbox
    mid_lat = 0.5 * (min_lat + max_lat)
    dlat = margin_km / 111.32
    coslat = math.cos(math.radians(mid_lat))
    if abs(coslat) < 1e-6:
        dlon = 180.0
    else:
        dlon = margin_km / (111.32 * coslat)
    return (min_lon - dlon, max_lon + dlon, min_lat - dlat, max_lat + dlat)


def srtm_tile_id(lat: int, lon: int) -> str:
    """返回 SRTM HGT tile 的基础名，例如 N39E116。lat/lon 为整数格网（tile 的西南角）。"""
    lat_tag = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_tag = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    return f"{lat_tag}{lon_tag}"


def download_url(
    url: str,
    dst: str,
    *,
    timeout_s: float = 60.0,
    retries: int = 3,
    progress_prefix: str = "",
) -> None:
    """最小依赖下载器（urllib），失败会抛异常。"""
    last_err: Optional[BaseException] = None
    dst_part = f"{dst}.part"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mkdem.py"})
            with urllib.request.urlopen(req, timeout=timeout_s) as r, open(dst_part, "wb") as f:
                total = r.headers.get("Content-Length")
                total_size = int(total) if total and total.isdigit() else -1
                downloaded = 0
                printed_pct = -1
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        if pct >= printed_pct + 10:
                            printed_pct = pct
                            print(
                                f"{progress_prefix}下载进度: {pct}% "
                                f"({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)",
                                flush=True,
                            )
            os.replace(dst_part, dst)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            try:
                if os.path.exists(dst_part):
                    os.remove(dst_part)
            except OSError:
                pass
            if attempt < retries:
                print(f"{progress_prefix}下载失败，重试 {attempt}/{retries}: {e}", flush=True)
                time.sleep(1.5 * attempt)
            continue
    raise RuntimeError(f"下载失败: {url}") from last_err


def download_srtm_tile(
    lat: int,
    lon: int,
    cache_dir: str,
    *,
    keep_zip: bool = False,
    timeout_s: float = 60.0,
    retries: int = 3,
) -> str:
    """下载单个SRTM瓦片
    
    Args:
        lat: 纬度（整数）
        lon: 经度（整数）
        cache_dir: 缓存目录
        keep_zip: 是否保留下载的 zip
        
    Returns:
        tile_file: 下载的瓦片文件路径
    """
    os.makedirs(cache_dir, exist_ok=True)
    tile_id = srtm_tile_id(lat, lon)
    tile_file = os.path.join(cache_dir, f"{tile_id}.hgt")
    if os.path.exists(tile_file):
        print(f"使用缓存瓦片: {tile_id}", flush=True)
        return tile_file

    zip_file = os.path.join(cache_dir, f"{tile_id}.SRTMGL1.hgt.zip")
    url = f"{SRTMGL1_BASE_URL}/{Path(zip_file).name}"
    print(f"下载 SRTMGL1: {tile_id}  {url}", flush=True)

    if not os.path.exists(zip_file):
        download_url(
            url,
            zip_file,
            timeout_s=timeout_s,
            retries=retries,
            progress_prefix=f"[{tile_id}] ",
        )

    with zipfile.ZipFile(zip_file, "r") as zf:
        members = zf.namelist()
        target_member = None
        expected = f"{tile_id}.HGT"
        for member in members:
            if Path(member).name.upper() == expected:
                target_member = member
                break
        if target_member is None:
            for member in members:
                if Path(member).suffix.lower() == ".hgt":
                    target_member = member
                    break
        if target_member is None:
            raise RuntimeError(f"zip 内未找到 HGT 文件: {zip_file}")

        extracted = zf.extract(target_member, cache_dir)
        extracted_abs = os.path.abspath(extracted)
        tile_abs = os.path.abspath(tile_file)
        if extracted_abs != tile_abs:
            shutil.move(extracted_abs, tile_abs)
            extracted_dir = os.path.dirname(extracted_abs)
            if extracted_dir and extracted_dir != os.path.abspath(cache_dir):
                try:
                    shutil.rmtree(extracted_dir)
                except OSError:
                    pass

    if not os.path.exists(tile_file):
        raise RuntimeError(f"解压后未找到 {tile_file}，请检查 zip 内容: {zip_file}")
    if not keep_zip:
        try:
            os.remove(zip_file)
        except OSError:
            pass
    return tile_file


def download_dem(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    output_dir: str,
    *,
    hgt_dir: Optional[str] = None,
    keep_tiles: bool = False,
    keep_zip: bool = False,
    download_workers: int = 4,
    timeout_s: float = 60.0,
    retries: int = 3,
) -> str:
    """下载DEM数据
    
    Args:
        min_lon: 最小经度
        max_lon: 最大经度
        min_lat: 最小纬度
        max_lat: 最大纬度
        output_dir: 输出目录
        
    Returns:
        dem_tif_file: DEM TIF文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    gdal, _ = gdal_osr()

    # 如果未指定 hgt_dir:
    # keep_tiles=True 时缓存到输出目录；否则使用临时目录并在结束后删除。
    tmp_dir = None
    if hgt_dir is None:
        if keep_tiles:
            tile_dir = os.path.join(output_dir, "srtm_hgt_cache")
            os.makedirs(tile_dir, exist_ok=True)
        else:
            tmp_dir = tempfile.mkdtemp(prefix="mkdem_hgt_")
            tile_dir = tmp_dir
    else:
        tile_dir = hgt_dir
        os.makedirs(tile_dir, exist_ok=True)
    
    # 计算需要的SRTM瓦片
    min_lat_int = int(math.floor(min_lat))
    max_lat_int = int(math.ceil(max_lat))
    min_lon_int = int(math.floor(min_lon))
    max_lon_int = int(math.ceil(max_lon))
    
    # 下载所有需要的瓦片（并行）
    tile_coords = [(lat, lon) for lat in range(min_lat_int, max_lat_int) for lon in range(min_lon_int, max_lon_int)]
    if not tile_coords:
        raise RuntimeError(
            f"bbox 计算得到 0 个瓦片，请检查范围: "
            f"lon[{min_lon:.6f},{max_lon:.6f}] lat[{min_lat:.6f},{max_lat:.6f}]"
        )
    workers = max(1, min(int(download_workers), len(tile_coords)))
    print(f"待下载瓦片数: {len(tile_coords)}，下载线程数: {workers}", flush=True)

    tiles = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                download_srtm_tile,
                lat,
                lon,
                tile_dir,
                keep_zip=keep_zip,
                timeout_s=timeout_s,
                retries=retries,
            ): (lat, lon)
            for lat, lon in tile_coords
        }
        for future in as_completed(future_map):
            lat, lon = future_map[future]
            try:
                tiles.append(future.result())
            except Exception as e:
                failures.append((lat, lon, str(e)))

    if failures:
        detail = "\n".join([f"{srtm_tile_id(lat, lon)}: {err}" for lat, lon, err in failures])
        raise RuntimeError(f"部分瓦片下载失败:\n{detail}")
    tiles = sorted(set(tiles))
    
    # 合并瓦片
    dem_tif_file = os.path.join(output_dir, "dem.tif")
    if len(tiles) == 1:
        # 单瓦片也走 Translate，确保输出为规范 GeoTIFF（避免直接 copy .hgt）。
        out_ds = gdal.Translate(
            dem_tif_file,
            tiles[0],
            format="GTiff",
            outputType=gdal.GDT_Int16,
            noData=SRTM_NODATA,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE"],
        )
        if not out_ds:
            raise RuntimeError(f"单瓦片转换失败: {tiles[0]}")
        out_ds = None
    else:
        # 多个瓦片，使用gdal.BuildVRT和gdal.Translate合并
        vrt_file = os.path.join(output_dir, "dem.vrt")
        vrt_ds = gdal.BuildVRT(vrt_file, tiles, srcNodata=SRTM_NODATA, VRTNodata=SRTM_NODATA)
        if not vrt_ds:
            raise RuntimeError("BuildVRT 合并失败")
        vrt_ds = None
        out_ds = gdal.Translate(
            dem_tif_file,
            vrt_file,
            format="GTiff",
            outputType=gdal.GDT_Int16,
            noData=SRTM_NODATA,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE"],
        )
        if not out_ds:
            raise RuntimeError("Translate 合并输出失败")
        out_ds = None
        try:
            os.remove(vrt_file)
        except OSError:
            pass

    # 若使用临时目录，则用完后删除 hgt/zip 及目录本身
    if tmp_dir is not None:
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass
    
    print(f"DEM下载完成: {dem_tif_file}")
    return dem_tif_file


def remove_geoid(dem_tif_file: str, output_dir: str) -> str:
    """移除大地水准面，使高度相对于WGS84椭球面
    
    Args:
        dem_tif_file: DEM TIF文件路径
        output_dir: 输出目录
        
    Returns:
        dem_wgs84_tif: 相对于WGS84椭球面的DEM TIF文件路径
    """
    # 注意：
    # 大多数公开 DEM（如 SRTM）高度是“正高”（相对大地水准面）。而部分雷达几何模型期望“椭球高”。
    # 如果你的后续链路确实需要椭球高，应在这里引入 EGM96/EGM2008 等模型做垂向改正。
    # 这里默认不做改正，保持输入 DEM 高度基准不变。
    # 保留这个函数是为了将来接入 EGM96/EGM2008 时不破坏调用方接口。
    print("大地水准面移除：当前为 no-op（未做垂向改正）")
    return dem_tif_file


def fill_nodata_inplace(ds, *, max_search_dist_px: int = 100, smoothing_iters: int = 0) -> int:
    """使用 GDAL FillNodata 填充空洞（可选），避免引入 SciPy 依赖。返回填充像元个数（粗略）。"""
    gdal, _ = gdal_osr()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = SRTM_NODATA
        band.SetNoDataValue(nodata)
    arr = band.ReadAsArray()
    if arr is None:
        return 0
    mask = (arr == nodata)
    missing = int(mask.sum())
    if missing == 0:
        return 0
    driver = gdal.GetDriverByName("MEM")
    mask_ds = driver.Create("", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.WriteArray((~mask).astype("uint8") * 255)
    gdal.FillNodata(targetBand=band, maskBand=mask_band, maxSearchDist=max_search_dist_px, smoothingIterations=smoothing_iters)
    return missing


def warp_crop_latlon(
    src: str,
    dst_tif: str,
    *,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    res_deg: Optional[float] = None,
    resample: str = "bilinear",
) -> str:
    """裁剪/重采样到 EPSG:4326。"""
    gdal, _ = gdal_osr()
    resample_map = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
    }
    if resample not in resample_map:
        raise ValueError(f"不支持的重采样方法: {resample}，可选 nearest/bilinear/cubic")

    warp_kwargs = dict(
        format="GTiff",
        dstSRS="EPSG:4326",
        resampleAlg=resample_map[resample],
        srcNodata=SRTM_NODATA,
        dstNodata=SRTM_NODATA,
        multithread=True,
    )
    if bbox is not None:
        min_lon, max_lon, min_lat, max_lat = bbox
        warp_kwargs["outputBounds"] = (min_lon, min_lat, max_lon, max_lat)
    if res_deg is not None:
        warp_kwargs["xRes"] = res_deg
        warp_kwargs["yRes"] = res_deg

    out_ds = gdal.Warp(dst_tif, src, **warp_kwargs)
    if not out_ds:
        raise RuntimeError(f"gdal.Warp 失败: {dst_tif}")
    out_ds.FlushCache()
    out_ds = None
    return dst_tif


def convert_to_utm(
    dem_tif_file: str,
    output_dir: str,
    resolution_m: float = 30.0,
    *,
    utm_zone: Optional[int] = None,
    utm_south: Optional[bool] = None,
    resample: str = "bilinear",
    fill_nodata: bool = True,
    fill_max_search_dist_px: int = 100,
    fill_smoothing_iters: int = 0,
) -> str:
    """将DEM转换为UTM投影
    
    Args:
        dem_tif_file: DEM TIF文件路径
        output_dir: 输出目录
        resolution_m: 输出分辨率（米）
        utm_zone: 指定 UTM 带号（1-60），不指定则根据中心经度自动计算
        utm_south: 指定是否南半球（True=南半球，False=北半球），不指定则根据中心纬度判断
        resample: 重采样方法 nearest/bilinear/cubic
        fill_nodata: 是否在投影后用 GDAL FillNodata 填洞（默认 True）
        fill_max_search_dist_px: FillNodata 最大搜索半径（像素）
        fill_smoothing_iters: FillNodata 平滑迭代次数（0 表示不平滑）
        
    Returns:
        dem_utm_tif: UTM投影DEM TIF文件路径
    """
    print(f"将DEM转换为UTM投影: {dem_tif_file}")
    gdal, osr = gdal_osr()
    
    # 打开输入DEM文件
    in_ds = gdal.Open(dem_tif_file)
    if not in_ds:
        raise ValueError(f"无法打开输入DEM文件: {dem_tif_file}")
    
    # 获取输入DEM的地理信息
    in_gt = in_ds.GetGeoTransform()
    
    # 计算输入DEM的中心点，用于确定UTM带
    width = in_ds.RasterXSize
    height = in_ds.RasterYSize
    center_lon = in_gt[0] + (width / 2) * in_gt[1] + (height / 2) * in_gt[2]
    center_lat = in_gt[3] + (width / 2) * in_gt[4] + (height / 2) * in_gt[5]
    
    # 确定UTM带
    if utm_zone is None:
        utm_zone = int((center_lon + 180) / 6) + 1
    if utm_zone < 1 or utm_zone > 60:
        raise ValueError(f"UTM 带号非法: {utm_zone} (应为 1-60)")
    if utm_south is None:
        utm_south = center_lat < 0
    hemi_txt = "S" if utm_south else "N"
    print(f"UTM带: {utm_zone}{hemi_txt}  (center_lon={center_lon:.6f}, center_lat={center_lat:.6f})")
    
    # 创建UTM空间参考
    utm_srs = osr.SpatialReference()
    utm_srs.SetUTM(utm_zone, not utm_south)  # True=北半球
    utm_srs.SetWellKnownGeogCS('WGS84')
    
    # 构建输出文件路径
    dem_utm_tif = os.path.join(output_dir, "dem_utm.tif")
    
    resample_map = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
    }
    if resample not in resample_map:
        raise ValueError(f"不支持的重采样方法: {resample}，可选 nearest/bilinear/cubic")

    print("使用gdal.Warp进行坐标转换...")
    out_ds = gdal.Warp(
        dem_utm_tif,
        in_ds,
        format="GTiff",
        xRes=resolution_m,
        yRes=resolution_m,
        dstSRS=utm_srs.ExportToWkt(),
        resampleAlg=resample_map[resample],
        srcNodata=SRTM_NODATA,
        dstNodata=SRTM_NODATA,
        multithread=True,
    )
    if not out_ds:
        raise RuntimeError("gdal.Warp 转换为 UTM 失败")

    if fill_nodata:
        filled = fill_nodata_inplace(
            out_ds,
            max_search_dist_px=fill_max_search_dist_px,
            smoothing_iters=fill_smoothing_iters,
        )
        if filled:
            print(f"FillNodata: 尝试填充空洞像元数={filled}")

    band = out_ds.GetRasterBand(1)
    try:
        min_val, max_val = band.ComputeRasterMinMax()
        print(f"转换后DEM值范围: [{min_val:.2f}, {max_val:.2f}]")
    except Exception:
        pass

    out_ds.FlushCache()
    out_ds = None
    in_ds = None
    
    print(f"UTM转换完成: {dem_utm_tif}")
    return dem_utm_tif


def create_vrt(tif_file: str, output_dir: str) -> str:
    """为TIF文件创建VRT文件
    
    Args:
        tif_file: TIF文件路径
        output_dir: 输出目录
        
    Returns:
        vrt_file: VRT文件路径
    """
    # 提取文件名（不含扩展名）
    base_name = os.path.basename(tif_file).replace('.tif', '')
    vrt_file = os.path.join(output_dir, f"{base_name}.vrt")
    
    # 使用gdalbuildvrt创建VRT文件
    gdal, _ = gdal_osr()
    gdal.BuildVRT(vrt_file, [tif_file])
    
    print(f"VRT创建完成: {vrt_file}")
    return vrt_file


def parse_bbox_arg(bbox_str: str) -> Tuple[float, float, float, float]:
    """解析 --bbox 'min_lon,max_lon,min_lat,max_lat'。"""
    parts = [p.strip() for p in bbox_str.replace(" ", "").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox 需要 4 个数: min_lon,max_lon,min_lat,max_lat")
    min_lon, max_lon, min_lat, max_lat = map(_safe_float, parts)
    if max_lon <= min_lon or max_lat <= min_lat:
        raise ValueError(f"--bbox 范围不合法: {bbox_str}")
    return (min_lon, max_lon, min_lat, max_lat)


def _dataset_bounds_wgs84(path: str) -> Optional[Tuple[float, float, float, float]]:
    """读取数据范围并转换为 WGS84 经纬度 bbox。"""
    gdal, osr = gdal_osr()
    ds = gdal.Open(path)
    if not ds:
        return None
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        ds = None
        return None

    w = ds.RasterXSize
    h = ds.RasterYSize
    # 四角（考虑旋转项）
    corners_px = [(0, 0), (w, 0), (0, h), (w, h)]
    corners_xy = []
    for px, py in corners_px:
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        corners_xy.append((x, y))

    proj = ds.GetProjectionRef() or ""
    ds = None
    if not proj:
        # 无投影信息，保守按经纬度解释
        lons = [c[0] for c in corners_xy]
        lats = [c[1] for c in corners_xy]
        return (min(lons), max(lons), min(lats), max(lats))

    srs_src = osr.SpatialReference()
    srs_src.ImportFromWkt(proj)
    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)
    if hasattr(srs_src, "SetAxisMappingStrategy"):
        srs_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if hasattr(srs_wgs84, "SetAxisMappingStrategy"):
        srs_wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srs_src, srs_wgs84)

    lons = []
    lats = []
    for x, y in corners_xy:
        lon, lat, _ = ct.TransformPoint(x, y)
        lons.append(lon)
        lats.append(lat)
    return (min(lons), max(lons), min(lats), max(lats))


def _bbox_contains(outer: Tuple[float, float, float, float], inner: Tuple[float, float, float, float], tol: float = 1e-5) -> bool:
    return (
        outer[0] <= inner[0] + tol and
        outer[1] >= inner[1] - tol and
        outer[2] <= inner[2] + tol and
        outer[3] >= inner[3] - tol
    )


def find_local_dem_candidate(
    master_yaml: str,
    output_dir: str,
    bbox_wgs84: Optional[Tuple[float, float, float, float]],
    hgt_dir: Optional[str] = None,
) -> Optional[str]:
    """本地 DEM 自动发现：优先覆盖 bbox 的候选。"""
    search_dirs = []
    master_dir = os.path.dirname(os.path.abspath(master_yaml))
    out_dir_abs = os.path.abspath(output_dir)
    cwd = os.getcwd()
    for d in [out_dir_abs, master_dir, cwd]:
        if d and d not in search_dirs:
            search_dirs.append(d)
    if hgt_dir:
        hgt_abs = os.path.abspath(hgt_dir)
        if hgt_abs not in search_dirs:
            search_dirs.append(hgt_abs)

    candidate_names = [
        "dem_latlon.tif",
        "dem.tif",
        "dem_latlon.vrt",
        "dem.vrt",
        "dem_utm.tif",
        "dem_utm.vrt",
    ]

    candidates = []
    for d in search_dirs:
        for name in candidate_names:
            p = os.path.join(d, name)
            if os.path.exists(p) and p not in candidates:
                candidates.append(p)

    if not candidates:
        return None

    if bbox_wgs84 is None:
        return candidates[0]

    # 优先选可覆盖 bbox 的候选
    for p in candidates:
        bounds = _dataset_bounds_wgs84(p)
        if bounds is None:
            continue
        if _bbox_contains(bounds, bbox_wgs84):
            return p
    return None


def process_dem(
    master_yaml: str,
    output_dir: str,
    *,
    src_dem: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    margin_km: float = 10.0,
    out_crs: str = "both",
    utm_resolution_m: float = 30.0,
    latlon_res_arcsec: Optional[float] = None,
    resample: str = "bilinear",
    hgt_dir: Optional[str] = None,
    keep_tiles: bool = False,
    download_workers: int = 4,
    download_timeout_s: float = 60.0,
    download_retries: int = 3,
    make_vrt: bool = True,
    fill_nodata: bool = True,
    fill_max_search_dist_px: int = 100,
    fill_smoothing_iters: int = 0,
) -> Dict[str, str]:
    """处理DEM的完整流程
    
    Args:
        master_yaml: master的YAML文件路径
        output_dir: 输出目录
        src_dem: 本地 DEM（有则不下载）
        bbox: 指定裁剪范围 (min_lon,max_lon,min_lat,max_lat)，不指定则尝试从 YAML 读取
        margin_km: bbox 扩展边界（公里），用于留出缓冲
        out_crs: 输出投影: latlon / utm / both
        utm_resolution_m: UTM 输出分辨率（米）
        latlon_res_arcsec: 经纬度输出分辨率（角秒），不指定则保持源 DEM 分辨率
        resample: 重采样方法 nearest/bilinear/cubic
        hgt_dir: 下载的原始 HGT 文件存放目录。不指定则使用临时目录，并在拼接完成后删除。
        keep_tiles: 下载 SRTM 时是否保留瓦片（作为缓存）
        download_workers: 下载并行线程数
        download_timeout_s: 单次下载超时（秒）
        download_retries: 下载重试次数
        make_vrt: 是否为输出 tif 生成对应 vrt
        fill_nodata: 是否对输出 DEM 执行 GDAL FillNodata（默认 True）
        fill_max_search_dist_px: FillNodata 最大搜索半径（像素）
        fill_smoothing_iters: FillNodata 平滑迭代次数（0 表示不平滑）
        
    Returns:
        包含各文件路径的字典
    """
    print("=== DEM处理流程 ===")

    os.makedirs(output_dir, exist_ok=True)
    out_crs = out_crs.lower()
    if out_crs not in {"latlon", "utm", "both"}:
        raise ValueError("--out-crs 必须是 latlon/utm/both")

    if bbox is None:
        print(f"1. 尝试从 YAML 读取角点范围: {master_yaml}")
        bbox = read_master_bbox(master_yaml)
        if bbox is not None:
            print(f"   YAML bbox: {bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}")
        else:
            print("   YAML 不包含 corner_coordinates，将不做自动裁剪（除非你显式 --bbox）")

    bbox_expanded = expand_bbox_km(bbox, margin_km) if bbox is not None else None
    if bbox_expanded is not None:
        print(f"   bbox 扩展({margin_km} km): {bbox_expanded[0]:.6f},{bbox_expanded[1]:.6f},{bbox_expanded[2]:.6f},{bbox_expanded[3]:.6f}")

    # 2. 获取源 DEM：显式本地 -> 自动本地 -> 下载
    if src_dem:
        print(f"2. 使用本地 DEM: {src_dem}")
        if not os.path.exists(src_dem):
            raise FileNotFoundError(f"--src-dem 文件不存在: {src_dem}")
        src_path = src_dem
    else:
        print("2. 未指定 --src-dem，先尝试自动发现本地 DEM")
        local_dem = find_local_dem_candidate(
            master_yaml=master_yaml,
            output_dir=output_dir,
            bbox_wgs84=bbox_expanded,
            hgt_dir=hgt_dir,
        )
        if local_dem is not None:
            print(f"   命中本地 DEM: {local_dem}")
            src_path = local_dem
        else:
            if bbox_expanded is None:
                raise ValueError("未提供 --src-dem，且 YAML 无角点范围；也未发现本地 DEM。请提供 --bbox 或 --src-dem。")
            print("   本地未找到可用 DEM，开始下载并拼接 SRTMGL1")
            min_lon, max_lon, min_lat, max_lat = bbox_expanded
            src_path = download_dem(
                min_lon,
                max_lon,
                min_lat,
                max_lat,
                output_dir,
                hgt_dir=hgt_dir,
                keep_tiles=keep_tiles,
                keep_zip=keep_tiles,
                download_workers=download_workers,
                timeout_s=download_timeout_s,
                retries=download_retries,
            )

    # 3. 输出经纬度 DEM (EPSG:4326)，并（可选）按 bbox 裁剪
    print("3. 生成经纬度 DEM (EPSG:4326)")
    latlon_res_deg = None
    if latlon_res_arcsec is not None:
        if latlon_res_arcsec <= 0:
            raise ValueError("--latlon-res-arcsec 必须 > 0")
        latlon_res_deg = latlon_res_arcsec / 3600.0

    dem_latlon_tif = os.path.join(output_dir, "dem_latlon.tif")
    warp_crop_latlon(
        src_path,
        dem_latlon_tif,
        bbox=bbox_expanded,
        res_deg=latlon_res_deg,
        resample=resample,
    )

    if fill_nodata:
        gdal, _ = gdal_osr()
        ds = gdal.Open(dem_latlon_tif, gdal.GA_Update)
        if not ds:
            raise RuntimeError(f"无法以更新模式打开 DEM: {dem_latlon_tif}")
        filled = fill_nodata_inplace(
            ds,
            max_search_dist_px=fill_max_search_dist_px,
            smoothing_iters=fill_smoothing_iters,
        )
        if filled:
            print(f"FillNodata(latlon): 尝试填充空洞像元数={filled}")
        ds.FlushCache()
        ds = None

    # 4. 高程基准处理（默认不做垂向改正）
    print("4. 高程基准处理（默认不做大地水准面改正）")
    dem_latlon_final = remove_geoid(dem_latlon_tif, output_dir)

    result: Dict[str, str] = {"dem_latlon_tif": dem_latlon_final}
    if make_vrt:
        result["dem_latlon_vrt"] = create_vrt(dem_latlon_final, output_dir)

    if out_crs in {"utm", "both"}:
        print("5. 生成 UTM DEM")
        dem_utm_tif = convert_to_utm(
            dem_latlon_final,
            output_dir,
            utm_resolution_m,
            resample=resample,
            fill_nodata=fill_nodata,
            fill_max_search_dist_px=fill_max_search_dist_px,
            fill_smoothing_iters=fill_smoothing_iters,
        )
        result["dem_utm_tif"] = dem_utm_tif
        if make_vrt:
            result["dem_utm_vrt"] = create_vrt(dem_utm_tif, output_dir)
    
    print("\n=== DEM处理完成 ===")
    print(f"所有结果已输出到: {output_dir}")
    
    return result


def main():
    """主函数 - 命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="mkdem.py: 下载/裁剪/投影 DEM（面向 DInSAR/GMTSAR 工作流）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "master_yaml",
        help="master 的 YAML 参数文件路径（优先从 corner_coordinates 读取范围；没有角点时可配合 --bbox 或 --src-dem）",
    )
    parser.add_argument("-o", "--output-dir", default=".", help="输出目录")
    parser.add_argument(
        "-s",
        "--src-dem",
        default=None,
        help="显式指定本地 DEM 文件路径（GeoTIFF/HGT/VRT 等 GDAL 可读格式）。不指定时会先自动查找本地 DEM，找不到再下载。",
    )
    parser.add_argument(
        "-b",
        "--bbox",
        default=None,
        help="手动指定裁剪范围：min_lon,max_lon,min_lat,max_lat。优先级高于 YAML 角点。",
    )
    parser.add_argument(
        "-m",
        "--margin-km",
        type=float,
        default=10.0,
        help="在 bbox 基础上额外扩展的缓冲距离（公里）。用于避免边缘裁剪过紧。",
    )
    parser.add_argument(
        "-C",
        "--out-crs",
        choices=["latlon", "utm", "both"],
        default="both",
        help="输出投影类型：latlon=仅输出 EPSG:4326；utm=仅输出 UTM；both=两者都输出。",
    )
    parser.add_argument("-u", "--utm-resolution", type=float, default=30.0, help="UTM 输出分辨率（米）")
    parser.add_argument(
        "-a",
        "--latlon-res-arcsec",
        type=float,
        default=None,
        help="经纬度输出分辨率（角秒）。不指定则保持源 DEM 分辨率（SRTMGL1 通常为 1 arc-second）。",
    )
    parser.add_argument(
        "-r",
        "--resample",
        choices=["nearest", "bilinear", "cubic"],
        default="bilinear",
        help="裁剪/投影时的重采样方法。DEM 一般建议 bilinear；需要保留离散值时用 nearest。",
    )
    parser.add_argument(
        "-k",
        "--keep-tiles",
        action="store_true",
        default=False,
        help="保留下载的 SRTM 原始瓦片（hgt/zip）作为缓存（不开启则使用临时目录，流程结束后清理）。",
    )
    parser.add_argument(
        "-t",
        "--hgt-dir",
        default=None,
        help="下载的原始 .hgt 文件存放目录。未指定时使用临时目录，并在拼接 dem.tif 后自动删除这些 .hgt。",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="下载 SRTM 瓦片的并行线程数。",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=60.0,
        help="单次 HTTP 下载超时（秒）。",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="单瓦片下载失败后的重试次数。",
    )
    parser.add_argument(
        "--no-fill-nodata",
        action="store_false",
        dest="fill_nodata",
        default=True,
        help="禁用 GDAL FillNodata（默认启用；启用时会对输出 DEM 尝试填补 NoData 空洞）。",
    )
    parser.add_argument(
        "--fill-maxdist",
        type=int,
        default=100,
        help="FillNodata 最大搜索半径（像素）。值越大越能跨越空洞，但速度更慢、也更可能引入不合理填充值。",
    )
    parser.add_argument(
        "--fill-smooth",
        type=int,
        default=0,
        help="FillNodata 平滑迭代次数（0 表示不平滑）。",
    )
    parser.add_argument(
        "--no-vrt",
        action="store_true",
        default=False,
        help="不生成输出 GeoTIFF 对应的 VRT。",
    )
    
    args = parser.parse_args()
    
    bbox = parse_bbox_arg(args.bbox) if args.bbox else None

    result = process_dem(
        args.master_yaml,
        args.output_dir,
        src_dem=args.src_dem,
        bbox=bbox,
        margin_km=args.margin_km,
        out_crs=args.out_crs,
        utm_resolution_m=args.utm_resolution,
        latlon_res_arcsec=args.latlon_res_arcsec,
        resample=args.resample,
        hgt_dir=args.hgt_dir,
        keep_tiles=args.keep_tiles,
        download_workers=args.download_workers,
        download_timeout_s=args.download_timeout,
        download_retries=args.download_retries,
        make_vrt=not args.no_vrt,
        fill_nodata=args.fill_nodata,
        fill_max_search_dist_px=args.fill_maxdist,
        fill_smoothing_iters=args.fill_smooth,
    )
    
    # 打印结果
    print("\n生成的文件:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
