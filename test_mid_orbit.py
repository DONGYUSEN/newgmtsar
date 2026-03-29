import numpy as np
from datetime import datetime

# 地球参数
a = 6378137.0
e2 = 0.006694379999999999

# 坐标转换函数
def xyz_to_llh(x, y, z):
    """ECEF 转经纬度高度"""
    lon = np.arctan2(y, x)
    p = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, p*(1-e2))
    
    for _ in range(6):
        N = a / np.sqrt(1 - e2*np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p*(1-e2*N/(N+h)))
    
    N = a / np.sqrt(1 - e2*np.sin(lat)**2)
    h = p / np.cos(lat) - N
    
    return np.rad2deg(lat), np.rad2deg(lon), h

# 计算轨道方向
def compute_orbit_direction(pos, vel):
    """计算轨道方向（升轨/降轨）"""
    # 计算角动量向量
    angular_momentum = np.cross(pos, vel)
    # 轨道倾角的Z分量
    z_component = angular_momentum[2]
    return "升轨" if z_component > 0 else "降轨"

# 计算轨道法线
def compute_orbit_normal(pos, vel):
    """计算轨道法线"""
    normal = np.cross(pos, vel)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else normal

# 时间转换
def parse_time(time_str):
    return datetime.fromisoformat(time_str).timestamp()

# 线性插值函数
def interpolate_orbit(orbit_points, target_time):
    """在轨道点之间线性插值"""
    times = np.array([parse_time(p['time']) for p in orbit_points])
    idx = np.searchsorted(times, target_time)
    
    if idx == 0:
        return orbit_points[0]['position'], orbit_points[0]['velocity']
    if idx >= len(times):
        return orbit_points[-1]['position'], orbit_points[-1]['velocity']
    
    # 线性插值
    t1 = times[idx-1]
    t2 = times[idx]
    alpha = (target_time - t1) / (t2 - t1)
    
    p1 = np.array([orbit_points[idx-1]['position']['x'], 
                   orbit_points[idx-1]['position']['y'], 
                   orbit_points[idx-1]['position']['z']])
    p2 = np.array([orbit_points[idx]['position']['x'], 
                   orbit_points[idx]['position']['y'], 
                   orbit_points[idx]['position']['z']])
    
    v1 = np.array([orbit_points[idx-1]['velocity']['vx'], 
                   orbit_points[idx-1]['velocity']['vy'], 
                   orbit_points[idx-1]['velocity']['vz']])
    v2 = np.array([orbit_points[idx]['velocity']['vx'], 
                   orbit_points[idx]['velocity']['vy'], 
                   orbit_points[idx]['velocity']['vz']])
    
    pos = p1 + alpha * (p2 - p1)
    vel = v1 + alpha * (v2 - v1)
    
    return pos, vel

# 轨道数据（从master.yaml中提取）
orbit_points = [
    {'time': '2023-11-10T04:39:48.881889', 'position': {'x': -130136.876, 'y': 5934436.683, 'z': 3487221.764}, 'velocity': {'vx': 1486.7669206, 'vy': 3855.2051902, 'vz': -6482.893421}},
    {'time': '2023-11-10T04:39:49.999993', 'position': {'x': -128649.749, 'y': 5938288.17, 'z': 3480736.739}, 'velocity': {'vx': 1487.4859307, 'vy': 3847.770812, 'vz': -6487.1596532}},
    {'time': '2023-11-10T04:39:50.000008', 'position': {'x': -127161.905, 'y': 5942132.222999999, 'z': 3474247.448}, 'velocity': {'vx': 1488.2020479, 'vy': 3840.3316329, 'vz': -6491.4179502}},
    {'time': '2023-11-10T04:39:51.000002', 'position': {'x': -125673.346, 'y': 5945968.834, 'z': 3467753.903}, 'velocity': {'vx': 1488.9152704, 'vy': 3832.8876668999997, 'vz': -6495.6683041}},
    {'time': '2023-11-10T04:39:51.999997', 'position': {'x': -124184.075, 'y': 5949797.996, 'z': 3461256.115}, 'velocity': {'vx': 1489.6255962, 'vy': 3825.4389282999996, 'vz': -6499.9107066999995}},
    {'time': '2023-11-10T04:39:52.433358', 'position': {'x': -123439.19, 'y': 5951713.015, 'z': 3458007.22}, 'velocity': {'vx': 1489.980755, 'vy': 3821.71251, 'vz': -6502.03191}}  # 估算值
]

# sensor_start 和 sensor_end 时间
sensor_start = parse_time('2023-11-10T04:39:48.881889')
sensor_end = parse_time('2023-11-10T04:39:52.433358')

# 计算中间时间
mid_time = (sensor_start + sensor_end) / 2
mid_time_str = datetime.fromtimestamp(mid_time).isoformat()
print(f"中间时间: {mid_time_str}")

# 插值得到中间时刻的位置和速度
mid_pos, mid_vel = interpolate_orbit(orbit_points, mid_time)
print(f"中间时刻卫星位置 (ECEF):")
print(f"  x: {mid_pos[0]:.2f}")
print(f"  y: {mid_pos[1]:.2f}")
print(f"  z: {mid_pos[2]:.2f}")

print(f"\n中间时刻卫星速度:")
print(f"  vx: {mid_vel[0]:.2f}")
print(f"  vy: {mid_vel[1]:.2f}")
print(f"  vz: {mid_vel[2]:.2f}")

# 转换为经纬度
lat, lon, h = xyz_to_llh(mid_pos[0], mid_pos[1], mid_pos[2])
print(f"\n中间时刻卫星位置 (经纬度):")
print(f"  纬度: {lat:.6f}°")
print(f"  经度: {lon:.6f}°")
print(f"  高度: {h:.2f} 米")

# 计算轨道方向（升轨/降轨）
orbit_direction = compute_orbit_direction(mid_pos, mid_vel)
print(f"\n轨道方向: {orbit_direction}")

# DEM中心点位置
print(f"\nDEM中心点位置:")
print(f"  纬度: 29.500000°")
print(f"  经度: 95.000000°")

# 计算DEM中心的ECEF坐标
def llh_to_xyz(lat, lon, h=0):
    """经纬度转ECEF"""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    N = a / np.sqrt(1 - e2*np.sin(lat_rad)**2)
    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + h) * np.sin(lat_rad)
    return np.array([x, y, z])

dem_center_xyz = llh_to_xyz(29.5, 95.0)
print(f"\nDEM中心ECEF坐标:")
print(f"  x: {dem_center_xyz[0]:.2f}")
print(f"  y: {dem_center_xyz[1]:.2f}")
print(f"  z: {dem_center_xyz[2]:.2f}")

# 计算视线向量 dr = P - S
dr = dem_center_xyz - mid_pos
print(f"\n视线向量 (P - S):")
print(f"  x: {dr[0]:.2f}")
print(f"  y: {dr[1]:.2f}")
print(f"  z: {dr[2]:.2f}")

# 计算视线方向单位向量
dr_norm = np.linalg.norm(dr)
dr_unit = dr / dr_norm
print(f"\n视线方向单位向量:")
print(f"  x: {dr_unit[0]:.6f}")
print(f"  y: {dr_unit[1]:.6f}")
print(f"  z: {dr_unit[2]:.6f}")

# 使用S_end - S_start作为轨道前进方向
# 选择轨道数据中的起始点和结束点
S_start = np.array([orbit_points[0]['position']['x'], 
                   orbit_points[0]['position']['y'], 
                   orbit_points[0]['position']['z']])
S_end = np.array([orbit_points[-1]['position']['x'], 
                 orbit_points[-1]['position']['y'], 
                 orbit_points[-1]['position']['z']])
orbit_dir = S_end - S_start
orbit_dir = orbit_dir / np.linalg.norm(orbit_dir)
print(f"\n轨道前进方向单位向量:")
print(f"  x: {orbit_dir[0]:.6f}")
print(f"  y: {orbit_dir[1]:.6f}")
print(f"  z: {orbit_dir[2]:.6f}")

# 距离向方向（使用视线方向）
range_dir = dr_unit

# 计算叉积
cross_vec = np.cross(orbit_dir, range_dir)
print(f"\n叉积:")
print(f"  x: {cross_vec[0]:.6f}")
print(f"  y: {cross_vec[1]:.6f}")
print(f"  z: {cross_vec[2]:.6f}")

# 根据叉积的z分量符号确定左右视
look_dir = -1 if cross_vec[2] > 0 else 1
print(f"\nlook_dir: {look_dir}")
print(f"视线方向: {'右视' if look_dir == 1 else '左视'}")

# 分析相对方向
sat_lat = lat
sat_lon = lon
dem_lat = 29.5
dem_lon = 95.0

if sat_lat > dem_lat:
    lat_dir = "北"
elif sat_lat < dem_lat:
    lat_dir = "南"
else:
    lat_dir = "同一纬度"

if sat_lon > dem_lon:
    lon_dir = "东"
elif sat_lon < dem_lon:
    lon_dir = "西"
else:
    lon_dir = "同一经度"

print(f"\n卫星相对于DEM中心的方向: {lat_dir}{lon_dir}")
