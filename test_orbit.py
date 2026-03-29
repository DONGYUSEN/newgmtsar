import numpy as np

# 地球参数
a = 6378137.0
e2 = 0.00669437999014

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

# 测试数据
pos = np.array([17023.540000000, 6272732.458000001, 2839517.427000000])
vel = np.array([1542.8240422000001, 3113.1685167000001, -6859.0153277999998])

# 转换为经纬度
lat, lon, h = xyz_to_llh(pos[0], pos[1], pos[2])
print(f"卫星位置：")
print(f"  纬度: {lat:.6f}°")
print(f"  经度: {lon:.6f}°")
print(f"  高度: {h:.2f} 米")

# 计算轨道方向
orbit_dir = compute_orbit_direction(pos, vel)
print(f"\n轨道方向: {orbit_dir}")

# 计算速度方向
vel_norm = np.linalg.norm(vel)
vel_unit = vel / vel_norm
print(f"\n速度向量（单位向量）:")
print(f"  x: {vel_unit[0]:.6f}")
print(f"  y: {vel_unit[1]:.6f}")
print(f"  z: {vel_unit[2]:.6f}")

# 计算轨道法线
orbit_normal = compute_orbit_normal(pos, vel)
print(f"\n轨道法线:")
print(f"  x: {orbit_normal[0]:.6f}")
print(f"  y: {orbit_normal[1]:.6f}")
print(f"  z: {orbit_normal[2]:.6f}")

# 分析DEM位置（假设DEM中心在 95°E, 29.5°N）
dem_center_lat = 29.5
dem_center_lon = 95.0

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

dem_center_xyz = llh_to_xyz(dem_center_lat, dem_center_lon)
print(f"\nDEM中心ECEF坐标:")
print(f"  x: {dem_center_xyz[0]:.2f}")
print(f"  y: {dem_center_xyz[1]:.2f}")
print(f"  z: {dem_center_xyz[2]:.2f}")

# 计算视线向量 dr = P - S
dr = dem_center_xyz - pos
print(f"\n视线向量 (P - S):")
print(f"  x: {dr[0]:.2f}")
print(f"  y: {dr[1]:.2f}")
print(f"  z: {dr[2]:.2f}")

# 计算look_dir
look_dir = np.sign(np.dot(dr, orbit_normal))
print(f"\nlook_dir: {look_dir}")
print(f"视线方向: {'右视' if look_dir == 1 else '左视'}")
