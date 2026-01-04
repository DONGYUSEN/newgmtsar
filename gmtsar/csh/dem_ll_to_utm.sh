#!/usr/bin/env bash
set -e

# ===================== 功能说明与使用帮助 =====================
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] INPUT_DEM OUTPUT_DEM Reselution

功能：将地理坐标系的DEM网格文件（.grd）自动转换为UTM投影坐标系
适配：最新版GMTSAR/GMT（GMT 6.4+），修复经纬度提取和投影解析错误

参数说明：
  INPUT_DEM    输入的地理坐标系DEM文件（如 dem.grd），必须为GMT grd格式
  OUTPUT_DEM   输出的UTM投影DEM文件（如 dem_utm.grd），脚本会自动创建
  Reselution   输出的UTM投影DEM文件分辨率(Meter)

示例：
  $(basename "$0") dem.grd dem_utm.grd          # 基础用法
  $(basename "$0") ./data/input_dem.grd ./result/utm_dem.grd  # 指定路径

注意：
  1. 输入文件必须存在且为有效的GMT grd格式
  2. 脚本会自动计算DEM中心点的UTM带和南北半球
  3. 输出文件若已存在会被覆盖
EOF
    exit 1
}

echo "将地理坐标系的DEM网格文件（.grd）自动转换为UTM投影坐标系!"

# ===================== 参数校验 =====================
if [ $# -ne 3 ]; then
    echo "错误：参数数量不正确！"
    usage
fi

INPUT_DEM="$1"
OUTPUT_DEM="$2"
Resolution="$3"

if [ ! -f "$INPUT_DEM" ]; then
    echo "错误：输入文件 '$INPUT_DEM' 不存在！"
    usage
fi

# ===================== 核心处理逻辑 =====================
# ========= 1. 读取 DEM 中心点经纬度（适配实际输出格式） =========
echo "正在读取DEM文件：$INPUT_DEM"
# 提取grdinfo -C的输出并按制表符/空格分割
read lon_min lon_max lat_min lat_max <<< $(
    gmt grdinfo -C $INPUT_DEM | awk '{print $2,$3,$4,$5}'
)
# 计算中心点经纬度（保留6位小数）
lon_center=$(echo "scale=6; ($lon_min + $lon_max)/2.0" | bc -l)
lat_center=$(echo "scale=6; ($lat_min + $lat_max)/2.0" | bc -l)

echo "DEM中心点经纬度：$lon_center , $lat_center"

# ========= 2. 计算 UTM zone（确保为整数且有效） =========
utm_zone=$(echo "scale=0; ($lon_center + 180)/6 + 1" | bc)
# 验证UTM带是否为有效整数（1-60）
if ! [[ "$utm_zone" =~ ^[0-9]+$ && "$utm_zone" -ge 1 && "$utm_zone" -le 60 ]]; then
    echo "错误：计算出的UTM带 $utm_zone 无效（应在1-60之间）！"
    exit 1
fi

# ========= 3. 判断南北半球 =========
if (( $(echo "$lat_center >= 0" | bc -l) )); then
    hemi="N"
    utm_hemi=""  # 北半球无需后缀
else
    hemi="S"
    utm_hemi="S" # 南半球加S后缀
fi

echo "自动识别的UTM投影带：$utm_zone $hemi"

# ========= 4. 构造新版GMT兼容的UTM投影字符串 =========
JPROJ="-Ju${utm_zone}${utm_hemi}/1:1 -Fe" # -R$lon_min/$lon_max/$lat_min/$lat_max "
echo "使用的投影参数：$JPROJ"

# ========= 5. 投影 DEM =========
echo "正在将DEM转换为UTM投影，输出文件：$OUTPUT_DEM"
gmt grdproject "$INPUT_DEM" \
    $JPROJ \
    -D$3 \
    -G"$OUTPUT_DEM" 
 #
 #   -Vv
# -R94.5/95.5/29/30.5 -JU46 -D5m 

# 验证输出文件是否生成
if [ -f "$OUTPUT_DEM" ]; then
    echo -e "\n✅ 转换完成！UTM投影DEM已保存至：$OUTPUT_DEM"
    #echo "输出文件信息："
    #gmt grdinfo "$OUTPUT_DEM" | grep -E "x_min|y_min|units|projection"
else
    echo -e "\n❌ 转换失败！未生成输出文件"
    exit 1
fi
