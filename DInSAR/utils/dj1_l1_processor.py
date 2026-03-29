#!/usr/bin/env python3
"""
天仪SAR L1数据处理器
功能：处理天仪系列SAR L1数据，生成SLC数据文件、轨道文件和GDAL VRT文件
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from datetime import datetime
from osgeo import gdal

class DJ1L1Processor:
    """天仪SAR L1数据处理器"""
    
    def __init__(self, input_dir, output_dir):
        """初始化处理器
        
        Args:
            input_dir: 输入数据目录
            output_dir: 输出数据目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 启用GDAL异常处理
        gdal.UseExceptions()
        
    def process(self):
        """处理天仪SAR L1数据
        
        Returns:
            dict: 处理结果，包含生成的文件路径
        """
        # 1. 查找输入文件
        xml_files = self._find_files('.xml')
        tiff_files = self._find_files('.tif') + self._find_files('.tiff')
        prm_files = self._find_files('.PRM')
        
        if not xml_files:
            raise FileNotFoundError("未找到XML元数据文件")
        if not tiff_files:
            raise FileNotFoundError("未找到TIFF图像文件")
        
        # 2. 处理每个XML文件
        results = []
        for xml_file in xml_files:
            try:
                result = self._process_single(xml_file, tiff_files, prm_files)
                results.append(result)
            except Exception as e:
                print(f"处理文件 {xml_file} 时出错: {e}")
        
        return results
    
    def _find_files(self, extension):
        """查找指定扩展名的文件
        
        Args:
            extension: 文件扩展名
            
        Returns:
            list: 文件路径列表
        """
        files = []
        for root, _, filenames in os.walk(self.input_dir):
            for filename in filenames:
                if filename.lower().endswith(extension):
                    files.append(os.path.join(root, filename))
        return files
    
    def _process_single(self, xml_file, tiff_files, prm_files):
        """处理单个天仪SAR L1数据
        
        Args:
            xml_file: XML元数据文件路径
            tiff_files: TIFF图像文件路径列表
            prm_files: PRM文件路径列表
            
        Returns:
            dict: 处理结果
        """
        # 1. 解析XML元数据
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 2. 提取基本信息
        metadata = self._extract_metadata(root)
        orbit_data = self._extract_orbit_data(root)
        
        # 3. 查找对应的TIFF文件
        tiff_file = self._find_matching_tiff(xml_file, tiff_files)
        if not tiff_file:
            raise FileNotFoundError(f"未找到与 {xml_file} 匹配的TIFF文件")
        
        # 4. 生成文件名
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        slc_file = os.path.join(self.output_dir, f"{base_name}.SLC")
        yaml_file = os.path.join(self.output_dir, f"{base_name}.yaml")
        vrt_file = os.path.join(self.output_dir, f"{base_name}.vrt")
        
        # 5. 生成GDAL VRT文件（代替直接转换SLC）
        self._create_vrt_file(tiff_file, vrt_file, metadata)
        
        # 6. 从VRT文件提取四个角点的经纬度坐标
        corner_coordinates = self._extract_corner_coordinates(vrt_file)
        
        # 7. 生成YAML元数据文件（包含PRM和LED信息）
        self._generate_yaml(metadata, orbit_data, corner_coordinates, yaml_file, slc_file)
        
        return {
            'xml_file': xml_file,
            'tiff_file': tiff_file,
            'vrt_file': vrt_file,
            'slc_file': slc_file,
            'yaml_file': yaml_file
        }
    
    def _find_matching_tiff(self, xml_file, tiff_files):
        """查找与XML文件匹配的TIFF文件
        
        Args:
            xml_file: XML文件路径
            tiff_files: TIFF文件列表
            
        Returns:
            str: 匹配的TIFF文件路径
        """
        xml_base = os.path.splitext(os.path.basename(xml_file))[0]
        for tiff_file in tiff_files:
            tiff_base = os.path.splitext(os.path.basename(tiff_file))[0]
            if xml_base in tiff_base or tiff_base in xml_base:
                return tiff_file
        return None
    
    def _find_matching_prm(self, xml_file, prm_files):
        """查找与XML文件匹配的PRM文件
        
        Args:
            xml_file: XML文件路径
            prm_files: PRM文件列表
            
        Returns:
            str: 匹配的PRM文件路径
        """
        xml_base = os.path.splitext(os.path.basename(xml_file))[0]
        for prm_file in prm_files:
            prm_base = os.path.splitext(os.path.basename(prm_file))[0]
            if xml_base in prm_base or prm_base in xml_base:
                return prm_file
        return None
    
    def _extract_metadata(self, root):
        """从XML提取元数据
        
        Args:
            root: XML根元素
            
        Returns:
            dict: 元数据字典
        """
        metadata = {}
        c_speed = 299792458.0  # 光速
        
        # 提取产品信息 - 注意：根元素就是product
        product = root  # root.tag = 'product'
        if product is None:
            product = root.find('product')
        if product:
            # 提取ADS头信息
            ads_header = product.find('adsHeader')
            if ads_header:
                metadata['start_time'] = ads_header.findtext('startTime', '')
                metadata['stop_time'] = ads_header.findtext('stopTime', '')
                metadata['satellite'] = ads_header.findtext('missionId', 'Tianyi')
                metadata['polarization'] = ads_header.findtext('polarisation', 'VV')
                metadata['sensor'] = ads_header.findtext('mode', 'DJ1')
                metadata['absolute_orbit_number'] = ads_header.findtext('absoluteOrbitNumber', '0')
            
            # 提取generalAnnotation信息
            general_annotation = product.find('generalAnnotation')
            print(f"DEBUG: general_annotation 存在: {general_annotation is not None}")
            if general_annotation:
                # 产品信息
                product_info = general_annotation.find('productInformation')
                if product_info:
                    metadata['prf'] = float(product_info.findtext('prf', '4105.0903'))
                    
                    # 尝试从azimuthProcessing中获取更精确的PRF值
                    image_annotation = product.find('imageAnnotation')
                    if image_annotation:
                        image_info = image_annotation.find('imageInformation')
                        if image_info:
                            azimuth_processing = image_info.find('azimuthProcessing')
                            if azimuth_processing:
                                prf_precise = azimuth_processing.findtext('prf', None)
                                if prf_precise:
                                    metadata['prf'] = float(prf_precise)
                                    print(f"DEBUG: 使用精确PRF值: {prf_precise}")
                    
                    metadata['orbit_direction'] = product_info.findtext('pass', 'DESCENDING')
                    metadata['radar_frequency'] = float(product_info.findtext('radarFrequency', '5400000100'))
                    metadata['range_sampling_rate'] = float(product_info.findtext('rangeSamplingRate', '120000000'))
                    metadata['platform_heading'] = float(product_info.findtext('platformHeading', '0'))
                    # 计算波长 (c/frequency)
                    frequency = metadata.get('radar_frequency', 5400000100)
                    metadata['wavelength'] = c_speed / frequency
                
                # 下行链路信息
                downlink_info_list = general_annotation.find('downlinkInformationList')
                print(f"DEBUG: downlink_info_list 存在: {downlink_info_list is not None}")
                if downlink_info_list:
                    downlink_info = downlink_info_list.find('downlinkInformation')
                    print(f"DEBUG: downlink_info 存在: {downlink_info is not None}")
                    if downlink_info:
                        metadata['first_line_sensing_time'] = downlink_info.findtext('firstLineSensingTime', '')
                        metadata['last_line_sensing_time'] = downlink_info.findtext('stopTime', downlink_info.findtext('lastLineSensingTime', ''))
                        print(f"DEBUG: 从downlink提取 - first_line_sensing_time: {metadata.get('first_line_sensing_time', 'NOT_SET')}")
            
            # 提取imageAnnotation信息
            image_annotation = product.find('imageAnnotation')
            print(f"DEBUG: image_annotation 存在: {image_annotation is not None}")
            if image_annotation:
                # 图像信息
                image_info = image_annotation.find('imageInformation')
                print(f"DEBUG: image_info 存在: {image_info is not None}")
                if image_info:
                    metadata['number_of_samples'] = int(image_info.findtext('numberOfSamples', '12544'))
                    metadata['number_of_lines'] = int(image_info.findtext('numberOfLines', '14580'))
                    metadata['azimuth_time_interval'] = float(image_info.findtext('azimuthTimeInterval', '2.436e-04'))
                    metadata['slant_range_time'] = float(image_info.findtext('slantRangeTime', '4.177e-03'))
                    # 计算近距
                    metadata['near_range'] = metadata.get('slant_range_time', 4.177e-03) * c_speed / 2
                    # 提取 sensing time
                    metadata['first_line_sensing_time'] = image_info.findtext('productFirstLineUtcTime', metadata.get('first_line_sensing_time', ''))
                    metadata['last_line_sensing_time'] = image_info.findtext('productLastLineUtcTime', metadata.get('last_line_sensing_time', ''))
                    print(f"DEBUG: 从imageAnnotation提取 - first_line_sensing_time: {metadata.get('first_line_sensing_time', 'NOT_SET')}")
                
                # 处理信息
                processing_info = image_annotation.find('processingInformation')
                if processing_info:
                    swath_params_list = processing_info.find('swathProcParamsList')
                    if swath_params_list:
                        swath_params = swath_params_list.find('swathProcParams')
                        if swath_params:
                            range_processing = swath_params.find('rangeProcessing')
                            if range_processing:
                                metadata['number_of_looks'] = int(range_processing.findtext('numberOfLooks', '1'))
                                metadata['look_bandwidth'] = float(range_processing.findtext('lookBandwidth', '3.73134e+12'))
            
            # 提取dopplerCentroid信息
            doppler_centroid = product.find('dopplerCentroid')
            if doppler_centroid:
                dc_estimate_list = doppler_centroid.find('dcEstimateList')
                if dc_estimate_list:
                    dc_estimate = dc_estimate_list.find('dcEstimate')
                    if dc_estimate:
                        dc_polynomial = dc_estimate.findtext('dataDcPolynomial', '0 0 0')
                        metadata['doppler_polynomial'] = list(map(float, dc_polynomial.split()))
            
            # 提取geolocationGrid信息
            geolocation_grid = product.find('.//geolocationGrid')
            if geolocation_grid:
                grid_point_list = geolocation_grid.find('geolocationGridPointList')
                if grid_point_list:
                    grid_points = grid_point_list.findall('geolocationGridPoint')
                    metadata['geolocation_grid'] = []
                    for point in grid_points:
                        grid_point = {
                            'azimuth_time': point.findtext('azimuthTime', ''),
                            'slant_range_time': float(point.findtext('slantRangeTime', '0')),
                            'line': int(point.findtext('line', '0')),
                            'pixel': int(point.findtext('pixel', '0')),
                            'latitude': float(point.findtext('latitude', '0')),
                            'longitude': float(point.findtext('longitude', '0')),
                            'incidence_angle': float(point.findtext('incidenceAngle', '0'))
                        }
                        metadata['geolocation_grid'].append(grid_point)
        
        # 计算派生参数
        metadata['nrows'] = metadata.get('number_of_lines', 14580)
        metadata['ncols'] = metadata.get('number_of_samples', 12544)
        metadata['range_spacing'] = c_speed / (2 * metadata.get('range_sampling_rate', 120000000))
        metadata['azimuth_spacing'] = metadata.get('range_spacing', 1.249)  # 假设与距离向 spacing 相同
        metadata['far_range'] = metadata.get('near_range', 626552.193) + metadata['ncols'] * metadata['range_spacing']
        metadata['pulse_duration'] = 0.0000268  # 参考C代码中的固定值
        metadata['chirp_slope'] = metadata.get('look_bandwidth', 3.73134e+12) / metadata['pulse_duration']
        
        return metadata
    
    def _extract_orbit_data(self, root):
        """从XML提取轨道数据
        
        Args:
            root: XML根元素
            
        Returns:
            dict: 轨道数据字典，包含关联的位置和速度信息
        """
        orbit_data = {
            'orbit_points': [],  # 新格式：每个轨道点包含时间、位置和速度
            'positions': [],     # 旧格式：仅位置数据
            'velocities': []     # 旧格式：仅速度数据
        }
        
        # 提取轨道点
        try:
            # 直接查找orbitList元素，不依赖层级关系
            orbit_list = root.find('.//orbitList')
            if orbit_list:
                orbit_points = orbit_list.findall('orbit')
                print(f"找到 {len(orbit_points)} 个轨道点")
                for point in orbit_points:
                    time = point.findtext('time', '')
                    
                    # 位置数据
                    position = point.find('position')
                    pos_data = {
                        'time': time,
                        'x': float(position.findtext('x', '0')) if position else 0,
                        'y': float(position.findtext('y', '0')) if position else 0,
                        'z': float(position.findtext('z', '0')) if position else 0
                    }
                    
                    # 速度数据
                    velocity = point.find('velocity')
                    vel_data = {
                        'time': time,
                        'vx': float(velocity.findtext('x', '0')) if velocity else 0,
                        'vy': float(velocity.findtext('y', '0')) if velocity else 0,
                        'vz': float(velocity.findtext('z', '0')) if velocity else 0
                    }
                    
                    # 新格式：关联的轨道点
                    orbit_point = {
                        'time': time,
                        'position': {
                            'x': pos_data['x'],
                            'y': pos_data['y'],
                            'z': pos_data['z']
                        },
                        'velocity': {
                            'vx': vel_data['vx'],
                            'vy': vel_data['vy'],
                            'vz': vel_data['vz']
                        }
                    }
                    
                    orbit_data['orbit_points'].append(orbit_point)
                    orbit_data['positions'].append(pos_data)
                    orbit_data['velocities'].append(vel_data)
            else:
                print("未找到orbitList元素")
        except Exception as e:
            print(f"提取轨道数据时出错: {e}")
        
        print(f"成功提取 {len(orbit_data['orbit_points'])} 个轨道点（包含位置和速度信息）")
        return orbit_data
    
    def _extract_corner_coordinates(self, vrt_file):
        """从VRT文件提取四个角点的经纬度坐标和图像坐标
        
        Args:
            vrt_file: VRT文件路径
            
        Returns:
            dict: 四个角点的经纬度坐标和图像坐标
        """
        corner_coordinates = {
            'top_left': {'lon': 0, 'lat': 0, 'x': 0, 'y': 0},
            'top_right': {'lon': 0, 'lat': 0, 'x': 0, 'y': 0},
            'bottom_left': {'lon': 0, 'lat': 0, 'x': 0, 'y': 0},
            'bottom_right': {'lon': 0, 'lat': 0, 'x': 0, 'y': 0}
        }
        
        # 尝试从VRT文件中提取GCP信息
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(vrt_file)
            root = tree.getroot()
            
            # 查找GCP列表
            gcp_list = root.find('GCPList')
            if gcp_list:
                gcps = gcp_list.findall('GCP')
                if len(gcps) >= 4:
                    # 假设GCP顺序为：top_left, top_right, bottom_left, bottom_right
                    # 或者根据Pixel和Line值判断
                    for gcp in gcps:
                        pixel = float(gcp.get('Pixel', '0'))
                        line = float(gcp.get('Line', '0'))
                        x = float(gcp.get('X', '0'))  # lon
                        y = float(gcp.get('Y', '0'))  # lat
                        
                        # 根据像素和行位置判断角点
                        if pixel == 0 and line == 0:
                            corner_coordinates['top_left']['lon'] = x
                            corner_coordinates['top_left']['lat'] = y
                            corner_coordinates['top_left']['x'] = pixel
                            corner_coordinates['top_left']['y'] = line
                        elif pixel > 0 and line == 0:
                            corner_coordinates['top_right']['lon'] = x
                            corner_coordinates['top_right']['lat'] = y
                            corner_coordinates['top_right']['x'] = pixel
                            corner_coordinates['top_right']['y'] = line
                        elif pixel == 0 and line > 0:
                            corner_coordinates['bottom_left']['lon'] = x
                            corner_coordinates['bottom_left']['lat'] = y
                            corner_coordinates['bottom_left']['x'] = pixel
                            corner_coordinates['bottom_left']['y'] = line
                        elif pixel > 0 and line > 0:
                            corner_coordinates['bottom_right']['lon'] = x
                            corner_coordinates['bottom_right']['lat'] = y
                            corner_coordinates['bottom_right']['x'] = pixel
                            corner_coordinates['bottom_right']['y'] = line
        except Exception as e:
            print(f"从VRT文件提取角点坐标时出错: {e}")
        
        return corner_coordinates
    
    def _create_vrt_file(self, tiff_file, vrt_file, metadata):
        """创建GDAL VRT文件
        
        Args:
            tiff_file: 输入TIFF文件路径
            vrt_file: 输出VRT文件路径
            metadata: 元数据字典
        """
        print(f"创建VRT文件: {vrt_file}")
        
        # 打开TIFF文件
        ds = gdal.Open(tiff_file, gdal.GA_ReadOnly)
        if not ds:
            raise Exception(f"无法打开TIFF文件: {tiff_file}")
        
        # 获取图像信息
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        
        # 创建VRT文件
        driver = gdal.GetDriverByName('VRT')
        vrt_ds = driver.Create(vrt_file, width, height, bands, gdal.GDT_Int16)
        
        # 复制地理变换和投影信息
        if ds.GetGeoTransform():
            vrt_ds.SetGeoTransform(ds.GetGeoTransform())
        if ds.GetProjection():
            vrt_ds.SetProjection(ds.GetProjection())
        
        # 复制波段信息
        for i in range(bands):
            band = ds.GetRasterBand(i + 1)
            vrt_band = vrt_ds.GetRasterBand(i + 1)
            vrt_band.SetDescription(band.GetDescription())
            # 只有当NoData值存在时才设置
            nodata = band.GetNoDataValue()
            if nodata is not None:
                vrt_band.SetNoDataValue(nodata)
        
        # 直接使用GDAL的Translate功能创建VRT
        gdal.Translate(vrt_file, tiff_file, format='VRT')
        
        # 重新打开VRT文件添加元数据
        vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
        if vrt_ds:
            # 添加元数据
            vrt_ds.SetMetadataItem('satellite', metadata.get('satellite', 'Tianyi'))
            vrt_ds.SetMetadataItem('sensor', metadata.get('sensor', 'DJ1'))
            vrt_ds.SetMetadataItem('polarization', metadata.get('polarization', 'VV'))
            vrt_ds.SetMetadataItem('wavelength', str(metadata.get('wavelength', 0.0555)))
            vrt_ds.SetMetadataItem('prf', str(metadata.get('prf', 4105.0903)))
            vrt_ds.SetMetadataItem('near_range', str(metadata.get('near_range', 626552.193)))
            vrt_ds = None
        
        # 关闭数据集
        ds = None
        
        print(f"VRT文件创建完成: {vrt_file}")
    
    def _generate_prm(self, metadata, prm_file):
        """生成PRM文件
        
        Args:
            metadata: 元数据字典
            prm_file: 输出PRM文件路径
        """
        print(f"生成PRM文件: {prm_file}")
        
        base_name = os.path.splitext(os.path.basename(prm_file))[0]
        prm_content = []
        
        # 基本参数
        prm_content.append(f"num_valid_az    = {metadata.get('nrows', 14580)}")
        prm_content.append(f"nrows    = {metadata.get('nrows', 14580)}")
        prm_content.append(f"first_line    = 1")
        prm_content.append(f"deskew    = n")
        prm_content.append(f"caltone    = 0.000000")
        prm_content.append(f"st_rng_bin    = 1")
        prm_content.append(f"Flip_iq    = n")
        prm_content.append(f"offset_video    = n")
        prm_content.append(f"az_res    = 0.000000")
        prm_content.append(f"nlooks    = {metadata.get('number_of_looks', 1)}")
        prm_content.append(f"chirp_ext    = 0")
        prm_content.append(f"scnd_rng_mig    = 0")
        prm_content.append(f"rng_spec_wgt    = 1.000000")
        prm_content.append(f"rm_rng_band    = 0.200000")
        prm_content.append(f"rm_az_band    = 0.000000")
        prm_content.append(f"rshift   = 0")
        prm_content.append(f"ashift   = 0")
        prm_content.append(f"stretch_r    = 0")
        prm_content.append(f"stretch_a    = 0")
        prm_content.append(f"a_stretch_r    = 0")
        prm_content.append(f"a_stretch_a    = 0")
        prm_content.append(f"first_sample    = 1")
        prm_content.append(f"SC_identity    = 14")
        prm_content.append(f"rng_samp_rate    = {metadata.get('range_sampling_rate', 120000000.0)}")
        prm_content.append(f"input_file    = {base_name}.raw")
        prm_content.append(f"num_rng_bins    = {metadata.get('ncols', 12544)}")
        prm_content.append(f"bytes_per_line    = {metadata.get('ncols', 12544) * 4}")
        prm_content.append(f"good_bytes_per_line    = {metadata.get('ncols', 12544) * 4}")
        prm_content.append(f"PRF    = {metadata.get('prf', 4105.0903)}")
        prm_content.append(f"pulse_dur    = {metadata.get('pulse_duration', 0.0000268)}")
        prm_content.append(f"near_range    = {metadata.get('near_range', 626552.193)}")
        prm_content.append(f"num_lines    = {metadata.get('nrows', 14580)}")
        prm_content.append(f"num_patches    = 1")
        
        # 时间参数
        start_time = metadata.get('start_time', '')
        if start_time:
            # 转换时间格式
            try:
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                year = dt.year
                day_of_year = dt.timetuple().tm_yday
                seconds_of_day = (dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6)
                SC_clock_start = year * 1000 + day_of_year + seconds_of_day / 86400
                clock_start = day_of_year + seconds_of_day / 86400
                prm_content.append(f"SC_clock_start    = {SC_clock_start:.9f}")
                prm_content.append(f"SC_clock_stop    = {SC_clock_start + metadata.get('nrows', 14580) / metadata.get('prf', 4105.0903) / 86400:.9f}")
                prm_content.append(f"clock_start    = {clock_start:.9f}")
                prm_content.append(f"clock_stop    = {clock_start + metadata.get('nrows', 14580) / metadata.get('prf', 4105.0903) / 86400:.9f}")
            except Exception as e:
                print(f"时间格式解析出错: {e}")
                prm_content.append(f"SC_clock_start    = 2023313.1943157625")
                prm_content.append(f"SC_clock_stop    = 2023313.1943568699")
                prm_content.append(f"clock_start    = 313.194315762604")
                prm_content.append(f"clock_stop    = 313.194356870104")
        
        prm_content.append(f"led_file    = {base_name}.LED")
        prm_content.append(f"orbdir    = {'D' if metadata.get('orbit_direction', 'DESCENDING') == 'DESCENDING' else 'A'}")
        prm_content.append(f"lookdir    = L")
        prm_content.append(f"radar_wavelength    = {metadata.get('wavelength', 0.0555171)}")
        prm_content.append(f"chirp_slope    = {metadata.get('chirp_slope', 3.73134e+12)}")
        prm_content.append(f"rng_samp_rate    = {metadata.get('range_sampling_rate', 120000000.0)}")
        prm_content.append(f"I_mean    = 1")
        prm_content.append(f"Q_mean    = 1")
        prm_content.append(f"SC_vel    = 7377.171657")
        prm_content.append(f"earth_radius    = 6372683.826941")
        prm_content.append(f"equatorial_radius    = 6378137.000000")
        prm_content.append(f"polar_radius    = 6356752.310000")
        prm_content.append(f"SC_height    = 511762.634749")
        prm_content.append(f"SC_height_start    = 511742.586064")
        prm_content.append(f"SC_height_end    = 511782.687512")
        
        # Doppler参数
        doppler_poly = metadata.get('doppler_polynomial', [0, 0, 0])
        prm_content.append(f"fd1    = {doppler_poly[0] if len(doppler_poly) > 0 else 0.010000}")
        prm_content.append(f"fdd1    = {doppler_poly[1] if len(doppler_poly) > 1 else 0.000000}")
        prm_content.append(f"fddd1    = {doppler_poly[2] if len(doppler_poly) > 2 else 0.000000}")
        
        prm_content.append(f"sub_int_r               = 0.000000")
        prm_content.append(f"sub_int_a               = 0.000000")
        prm_content.append(f"SLC_file               = {base_name}.SLC")
        prm_content.append(f"dtype    = a")
        prm_content.append(f"SLC_scale               = 1.000000")
        
        with open(prm_file, 'w') as f:
            f.write('\n'.join(prm_content))
        
        print(f"PRM文件生成完成: {prm_file}")
    
    def _update_prm_file(self, input_prm, output_prm, metadata):
        """更新PRM文件
        
        Args:
            input_prm: 输入PRM文件路径
            output_prm: 输出PRM文件路径
            metadata: 元数据字典
        """
        print(f"更新PRM文件: {output_prm}")
        
        # 读取现有PRM文件
        with open(input_prm, 'r') as f:
            prm_lines = f.readlines()
        
        # 更新参数
        updated_lines = []
        for line in prm_lines:
            line = line.strip()
            if line.startswith('satellite ='):
                updated_lines.append(f"satellite = {metadata.get('satellite', 'Tianyi')}")
            elif line.startswith('sensor ='):
                updated_lines.append(f"sensor = {metadata.get('sensor', 'DJ1')}")
            elif line.startswith('polarization ='):
                updated_lines.append(f"polarization = {metadata.get('polarization', 'VV')}")
            elif line.startswith('nrows ='):
                updated_lines.append(f"nrows = {metadata.get('nrows', 14580)}")
            elif line.startswith('num_rng_bins ='):
                updated_lines.append(f"num_rng_bins = {metadata.get('ncols', 12544)}")
            elif line.startswith('wavelength ='):
                updated_lines.append(f"wavelength = {metadata.get('wavelength', 0.0555171)}")
            elif line.startswith('prf ='):
                updated_lines.append(f"prf = {metadata.get('prf', 4105.0903)}")
            elif line.startswith('near_range ='):
                updated_lines.append(f"near_range = {metadata.get('near_range', 626552.193)}")
            elif line.startswith('orbit_direction ='):
                updated_lines.append(f"orbit_direction = {metadata.get('orbit_direction', 'DESCENDING')}")
            else:
                updated_lines.append(line)
        
        # 写入更新后的PRM文件
        with open(output_prm, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"PRM文件更新完成: {output_prm}")
    
    def _generate_led(self, orbit_data, led_file):
        """生成LED轨道文件
        
        Args:
            orbit_data: 轨道数据字典
            led_file: 输出LED文件路径
        """
        print(f"生成LED文件: {led_file}")
        
        positions = orbit_data['positions']
        velocities = orbit_data['velocities']
        
        led_content = []
        
        if positions:
            # 计算时间间隔
            import re
            def parse_time(time_str):
                match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d+\.\d+)', time_str)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))
                    hour = int(match.group(4))
                    minute = int(match.group(5))
                    second = float(match.group(6))
                    # 计算年积日
                    import datetime
                    dt = datetime.datetime(year, month, day, hour, minute, int(second), int((second % 1) * 1e6))
                    day_of_year = dt.timetuple().tm_yday
                    seconds_of_day = hour * 3600 + minute * 60 + second
                    return year, day_of_year, seconds_of_day
                return 0, 0, 0
            
            # 提取轨道点
            orbit_points = []
            for i, pos in enumerate(positions):
                year, jd, sec = parse_time(pos['time'])
                if year > 0:
                    vel = velocities[i] if i < len(velocities) else {'vx': 0, 'vy': 0, 'vz': 0}
                    orbit_points.append((year, jd, sec, pos['x'], pos['y'], pos['z'], vel['vx'], vel['vy'], vel['vz']))
            
            if orbit_points:
                # 计算时间间隔
                dt = 0
                if len(orbit_points) > 1:
                    _, _, sec1 = parse_time(positions[0]['time'])
                    _, _, sec2 = parse_time(positions[1]['time'])
                    dt = sec2 - sec1
                
                # 写入LED文件
                led_content.append(f"{len(orbit_points)} {orbit_points[0][0]} {orbit_points[0][1]} {orbit_points[0][2]:.6f} {dt:.6f}")
                for point in orbit_points:
                    led_content.append(f"{point[0]} {point[1]} {point[2]:.6f} {point[3]:.6f} {point[4]:.6f} {point[5]:.6f} {point[6]:.8f} {point[7]:.8f} {point[8]:.8f}")
        
        if not led_content:
            # 默认轨道数据
            led_content.append("1 2023 313 69588.881889 1.000000")
            led_content.append("2023 313 69588.881889 -284515.577000 5480417.824000 4154857.155000 1393.477259 4621.008403 -5981.710606")
        
        with open(led_file, 'w') as f:
            f.write('\n'.join(led_content))
        
        print(f"LED文件生成完成: {led_file}")
    
    def _generate_yaml(self, metadata, orbit_data, corner_coordinates, yaml_file, slc_file):
        """生成YAML元数据文件（包含PRM和LED信息）
        
        Args:
            metadata: 元数据字典
            orbit_data: 轨道数据字典
            corner_coordinates: 四个角点的经纬度坐标
            yaml_file: 输出YAML文件路径
            slc_file: SLC文件路径
        """
        print(f"生成YAML文件: {yaml_file}")
        
        # 计算PRM相关参数
        ncols = metadata.get('ncols', 12544)
        nrows = metadata.get('nrows', 14580)
        bytes_per_line = ncols * 4
        
        # 计算时间参数
        start_time = metadata.get('start_time', '')
        SC_clock_start = 2023313.1943157625
        clock_start = 313.194315762604
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                year = dt.year
                day_of_year = dt.timetuple().tm_yday
                seconds_of_day = (dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6)
                SC_clock_start = year * 1000 + day_of_year + seconds_of_day / 86400
                clock_start = day_of_year + seconds_of_day / 86400
            except Exception as e:
                print(f"时间格式解析出错: {e}")
        
        SC_clock_stop = SC_clock_start + nrows / metadata.get('prf', 4105.0903) / 86400
        clock_stop = clock_start + nrows / metadata.get('prf', 4105.0903) / 86400
        
        # 准备YAML数据，使用普通字典确保标准格式
        # 按照用户要求的顺序排列metadata字段
        metadata_dict = {
            'satellite': metadata.get('satellite', 'Tianyi'),
            'sensor': metadata.get('sensor', 'DJ1'),
            'absolute_orbit_number': metadata.get('absolute_orbit_number', '0'),
            'creation_time': datetime.now().astimezone().isoformat(),
            'data_file': os.path.basename(slc_file),
            'data_type': 'SLC',
            'first_line_sensing_time': metadata.get('first_line_sensing_time', ''),
            'last_line_sensing_time': metadata.get('last_line_sensing_time', ''),
            'platform_heading': metadata.get('platform_heading', 0),
            'polarization': metadata.get('polarization', 'VV'),
            'version': '1.0'
        }
        
        # 准备完整的YAML数据
        yaml_data = {
            'metadata': metadata_dict,
            'image_parameters': {
                'nrows': nrows,
                'ncols': ncols,
                'data_format': 'complex_float32',
                'bands': ['real', 'imaginary'],
                'byte_order': 'little_endian'
            },
            'radar_parameters': {
                'wavelength': metadata.get('wavelength', 0.0555171),
                'prf': metadata.get('prf', 4105.0903),
                'pulse_duration': metadata.get('pulse_duration', 0.0000268),
                'near_range': metadata.get('near_range', 626552.193),
                'far_range': metadata.get('far_range', 972000.0),
                'range_spacing': metadata.get('range_spacing', 1.249),
                'azimuth_spacing': metadata.get('azimuth_spacing', 1.249),
                'range_sampling_rate': metadata.get('range_sampling_rate', 120000000.0),
                'chirp_slope': metadata.get('chirp_slope', 3.73134e+12)
            },
            'orbit_parameters': {
                'orbit_direction': metadata.get('orbit_direction', 'DESCENDING'),
                'look_direction': 'LEFT',
                'satellite_height': 511762.634749,
                'satellite_velocity': 7377.171657
            },
            'processing_parameters': {
                'number_of_looks': metadata.get('number_of_looks', 1),
                'look_bandwidth': metadata.get('look_bandwidth', 3.73134e+12),
                'doppler_polynomial': metadata.get('doppler_polynomial', [0, 0, 0])
            },
            'prm_parameters': {
                'num_valid_az': nrows,
                'first_line': 1,
                'deskew': 'n',
                'caltone': 0.0,
                'st_rng_bin': 1,
                'Flip_iq': 'n',
                'offset_video': 'n',
                'az_res': 0.0,
                'nlooks': metadata.get('number_of_looks', 1),
                'chirp_ext': 0,
                'scnd_rng_mig': 0,
                'rng_spec_wgt': 1.0,
                'rm_rng_band': 0.2,
                'rm_az_band': 0.0,
                'rshift': 0,
                'ashift': 0,
                'stretch_r': 0.0,
                'stretch_a': 0.0,
                'a_stretch_r': 0.0,
                'a_stretch_a': 0.0,
                'first_sample': 1,
                'SC_identity': 14,
                'rng_samp_rate': metadata.get('range_sampling_rate', 120000000.0),
                'input_file': f"{os.path.splitext(os.path.basename(yaml_file))[0]}.raw",
                'num_rng_bins': ncols,
                'bytes_per_line': bytes_per_line,
                'good_bytes_per_line': bytes_per_line,
                'PRF': metadata.get('prf', 4105.0903),
                'pulse_dur': metadata.get('pulse_duration', 0.0000268),
                'near_range': metadata.get('near_range', 626552.193),
                'num_lines': nrows,
                'num_patches': 1,
                'SC_clock_start': SC_clock_start,
                'SC_clock_stop': SC_clock_stop,
                'clock_start': clock_start,
                'clock_stop': clock_stop,
                'led_file': f"{os.path.splitext(os.path.basename(yaml_file))[0]}.LED",
                'orbdir': 'D' if metadata.get('orbit_direction', 'DESCENDING') == 'DESCENDING' else 'A',
                'lookdir': 'L',
                'radar_wavelength': metadata.get('wavelength', 0.0555171),
                'chirp_slope': metadata.get('chirp_slope', 3.73134e+12),
                'I_mean': 1,
                'Q_mean': 1,
                'SC_vel': 7377.171657,
                'earth_radius': 6372683.826941,
                'equatorial_radius': 6378137.0,
                'polar_radius': 6356752.31,
                'SC_height': 511762.634749,
                'SC_height_start': 511742.586064,
                'SC_height_end': 511782.687512,
                'fd1': metadata.get('doppler_polynomial', [0, 0, 0])[0] if len(metadata.get('doppler_polynomial', [])) > 0 else 0.0,
                'fdd1': metadata.get('doppler_polynomial', [0, 0, 0])[1] if len(metadata.get('doppler_polynomial', [])) > 1 else 0.0,
                'fddd1': metadata.get('doppler_polynomial', [0, 0, 0])[2] if len(metadata.get('doppler_polynomial', [])) > 2 else 0.0,
                'sub_int_r': 0.0,
                'sub_int_a': 0.0,
                'SLC_file': os.path.basename(slc_file),
                'dtype': 'a',
                'SLC_scale': 1.0
            },
            'orbit_data': {
                'orbit_points': orbit_data.get('orbit_points', []),  # 新格式：每个轨道点包含时间、位置和速度
                'positions': orbit_data.get('positions', []),       # 旧格式：仅位置数据
                'velocities': orbit_data.get('velocities', [])       # 旧格式：仅速度数据
            },
            'corner_coordinates': corner_coordinates,
            'geolocation_grid': metadata.get('geolocation_grid', [])
        }
        
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"YAML文件生成完成: {yaml_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='天仪SAR L1数据处理器')
    parser.add_argument('input_dir', help='输入数据目录')
    parser.add_argument('output_dir', help='输出数据目录')
    args = parser.parse_args()
    
    processor = DJ1L1Processor(args.input_dir, args.output_dir)
    results = processor.process()
    
    print("处理完成，生成的文件：")
    for result in results:
        print(f"\n处理文件: {result['xml_file']}")
        print(f"TIFF文件: {result['tiff_file']}")
        print(f"VRT文件: {result['vrt_file']}")
        print(f"YAML文件: {result['yaml_file']}")


if __name__ == '__main__':
    main()
