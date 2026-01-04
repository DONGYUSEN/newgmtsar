#!/bin/bash
# GMT 6.5实现SAR相位图去水平/垂直条纹的脚本（修复版）
# 依赖：GMT ≥ 6.5、bc（系统自带）
# 使用方法：chmod +x phase_denoise_gmt65.sh && ./phase_denoise_gmt65.sh phase.grd phase_denoised.grd

# ========== 参数配置 ==========
INPUT_GRD=$1
OUTPUT_GRD=$2
TEMP_PREFIX="temp_phase_"  # 临时文件前缀
GENERATE_PNG=$3  # 第三个参数，如果设置为"png"则生成PNG
LOW_FREQ_RADIUS=500        # 调整为适合您网格尺寸的值
ANGLE_TOLERANCE=10
PI=$(gmt math -Q pi =)  # GMT 6.5内置pi，精准

# ========== 前置检查 ==========
set -euo pipefail  # 开启严格模式，提前暴露错误
if [ $# -lt 2 ]; then
    echo "用法：$0 <输入GRD文件> <输出GRD文件> [png]"
    echo "      第三个参数为'png'时生成PNG图像"
    exit 1
fi

# 检查GMT版本
GMT_VER=$(gmt --version | awk -F. '{print $1$2}')
if [ ${GMT_VER} -lt 65 ]; then
    echo "错误：GMT版本需≥6.5，当前为$(gmt --version)"
    exit 1
fi

# 检查输入文件
if [ ! -f "${INPUT_GRD}" ]; then
    echo "错误：输入文件${INPUT_GRD}不存在！"
    exit 1
fi

# 清理函数：在退出时删除临时文件
cleanup() {
    echo "清理临时文件..."
    rm -f ${TEMP_PREFIX}*.grd 2>/dev/null || true
}
trap cleanup EXIT

# ========== 步骤1：获取网格信息 ==========
echo "===== 步骤1：获取网格信息 ====="
# 使用grdinfo直接获取准确信息
XMIN=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $1}')
XMAX=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $2}')
YMIN=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $3}')
YMAX=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $4}')
INC_X=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $7}')
INC_Y=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $8}')
NX=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $11}')
NY=$(gmt grdinfo ${INPUT_GRD} -C | awk '{print $12}')

echo "网格尺寸: ${NX} x ${NY}"
echo "网格范围: X: ${XMIN} 到 ${XMAX}, Y: ${YMIN} 到 ${YMAX}"
echo "网格间距: X方向: ${INC_X}, Y方向: ${INC_Y}"

# 计算中心坐标
CX=$(echo "${XMIN} ${XMAX}" | awk '{print ($1+$2)/2}')
CY=$(echo "${YMIN} ${YMAX}" | awk '{print ($1+$2)/2}')
echo "网格中心坐标: (${CX}, ${CY})"

# ========== 步骤2：数据预处理 ==========
echo "===== 步骤2：数据预处理 ====="
# 直接复制输入文件
cp ${INPUT_GRD} ${TEMP_PREFIX}cleaned.grd
echo "数据预处理完成"

# ========== 步骤3：创建与输入网格相同尺寸的掩膜 ==========
echo "===== 步骤3：创建掩膜 ====="

# 方法：先创建一个与输入网格相同尺寸的基础网格
echo "创建基础网格..."
gmt grdmath ${INPUT_GRD} 0 MUL = ${TEMP_PREFIX}base.grd

# 获取基础网格信息，确保与输入一致
BASE_INFO=$(gmt grdinfo ${TEMP_PREFIX}base.grd -C)
BASE_NX=$(echo ${BASE_INFO} | awk '{print $11}')
BASE_NY=$(echo ${BASE_INFO} | awk '{print $12}')
BASE_XMIN=$(echo ${BASE_INFO} | awk '{print $1}')
BASE_XMAX=$(echo ${BASE_INFO} | awk '{print $2}')
BASE_YMIN=$(echo ${BASE_INFO} | awk '{print $3}')
BASE_YMAX=$(echo ${BASE_INFO} | awk '{print $4}')

echo "基础网格尺寸: ${BASE_NX} x ${BASE_NY}"
echo "基础网格范围: X: ${BASE_XMIN} 到 ${BASE_XMAX}, Y: ${BASE_YMIN} 到 ${BASE_YMAX}"

# 基于基础网格创建坐标网格
echo "创建X坐标网格..."
gmt grdmath ${TEMP_PREFIX}base.grd X = ${TEMP_PREFIX}x_grid.grd

echo "创建Y坐标网格..."
gmt grdmath ${TEMP_PREFIX}base.grd Y = ${TEMP_PREFIX}y_grid.grd

# 计算距离
echo "计算距离网格..."
gmt grdmath ${TEMP_PREFIX}x_grid.grd ${CX} SUB DUP MUL ${TEMP_PREFIX}y_grid.grd ${CY} SUB DUP MUL ADD SQRT = ${TEMP_PREFIX}distance.grd

# 创建低频掩膜
echo "创建低频掩膜..."
gmt grdmath ${TEMP_PREFIX}distance.grd ${LOW_FREQ_RADIUS} LE = ${TEMP_PREFIX}low_freq_mask.grd

# 计算角度
echo "计算角度网格..."
gmt grdmath ${TEMP_PREFIX}x_grid.grd ${CX} SUB ${TEMP_PREFIX}y_grid.grd ${CY} SUB ATAN2 57.29578 MUL = ${TEMP_PREFIX}angle_deg.grd

# 创建水平条纹掩膜
echo "创建水平条纹掩膜..."
gmt grdmath ${TEMP_PREFIX}angle_deg.grd ABS ${ANGLE_TOLERANCE} LE = ${TEMP_PREFIX}horiz_mask.grd

# 创建垂直条纹掩膜
echo "创建垂直条纹掩膜..."
gmt grdmath ${TEMP_PREFIX}angle_deg.grd 90 SUB ABS ${ANGLE_TOLERANCE} LE = ${TEMP_PREFIX}vert_mask1.grd
gmt grdmath ${TEMP_PREFIX}angle_deg.grd -90 SUB ABS ${ANGLE_TOLERANCE} LE = ${TEMP_PREFIX}vert_mask2.grd
gmt grdmath ${TEMP_PREFIX}vert_mask1.grd ${TEMP_PREFIX}vert_mask2.grd ADD = ${TEMP_PREFIX}vert_mask.grd

# 合并条纹掩膜
echo "合并条纹掩膜..."
gmt grdmath ${TEMP_PREFIX}horiz_mask.grd ${TEMP_PREFIX}vert_mask.grd ADD = ${TEMP_PREFIX}stripes_mask.grd

# 创建非条纹掩膜
echo "创建非条纹掩膜..."
gmt grdmath ${TEMP_PREFIX}stripes_mask.grd 0 EQ = ${TEMP_PREFIX}no_stripes_mask.grd

# 创建最终掩膜
echo "创建最终掩膜..."
gmt grdmath ${TEMP_PREFIX}low_freq_mask.grd ${TEMP_PREFIX}no_stripes_mask.grd MUL = ${TEMP_PREFIX}mask.grd

# 清理中间网格
rm -f ${TEMP_PREFIX}base.grd ${TEMP_PREFIX}x_grid.grd ${TEMP_PREFIX}y_grid.grd ${TEMP_PREFIX}distance.grd ${TEMP_PREFIX}angle_deg.grd
rm -f ${TEMP_PREFIX}horiz_mask.grd ${TEMP_PREFIX}vert_mask*.grd ${TEMP_PREFIX}stripes_mask.grd ${TEMP_PREFIX}no_stripes_mask.grd

echo "掩膜构造完成"

# ========== 步骤4：应用滤波 ==========
echo "===== 步骤4：应用滤波 ====="

# 验证网格尺寸是否一致
echo "验证网格尺寸..."
CLEANED_SIZE=$(gmt grdinfo ${TEMP_PREFIX}cleaned.grd -C | awk '{print $11 "x" $12}')
MASK_SIZE=$(gmt grdinfo ${TEMP_PREFIX}mask.grd -C | awk '{print $11 "x" $12}')

echo "清理后网格尺寸: ${CLEANED_SIZE}"
echo "掩膜网格尺寸: ${MASK_SIZE}"

if [ "${CLEANED_SIZE}" != "${MASK_SIZE}" ]; then
    echo "错误：网格尺寸不匹配！"
    echo "尝试重新调整掩膜尺寸..."
    
    # 如果尺寸不匹配，调整掩膜尺寸
    gmt grdsample ${TEMP_PREFIX}mask.grd -G${TEMP_PREFIX}mask_resized.grd -R${INPUT_GRD} -I${INC_X}/${INC_Y}
    mv ${TEMP_PREFIX}mask_resized.grd ${TEMP_PREFIX}mask.grd
    
    # 再次检查尺寸
    MASK_SIZE=$(gmt grdinfo ${TEMP_PREFIX}mask.grd -C | awk '{print $11 "x" $12}')
    echo "调整后掩膜网格尺寸: ${MASK_SIZE}"
fi

echo "应用掩膜..."
gmt grdmath ${TEMP_PREFIX}cleaned.grd ${TEMP_PREFIX}mask.grd MUL = ${TEMP_PREFIX}filtered.grd

echo "计算平均值用于填充..."
MEAN_VAL=$(gmt grdinfo ${TEMP_PREFIX}filtered.grd -L1 | grep -oP 'mean:\s*\K[0-9.-]+' || echo "0.0")
echo "平均值: ${MEAN_VAL}"

echo "创建平均值网格..."
gmt grdmath ${TEMP_PREFIX}cleaned.grd 0 MUL ${MEAN_VAL} ADD = ${TEMP_PREFIX}mean_grid.grd

echo "填充被掩膜的区域..."
# 如果掩膜为0，使用平均值；否则使用滤波值
gmt grdmath ${TEMP_PREFIX}mask.grd 0 EQ ${TEMP_PREFIX}mean_grid.grd ${TEMP_PREFIX}filtered.grd IFELSE = ${TEMP_PREFIX}filled.grd

# ========== 步骤5：后处理 ==========
echo "===== 步骤5：后处理（相位范围限制） ====="

TWO_PI=$(echo "2 * ${PI}" | bc -l)

echo "限制相位范围到[-π, π]..."
# 应用模2π运算
gmt grdmath ${TEMP_PREFIX}filled.grd ${TWO_PI} MOD = ${TEMP_PREFIX}mod_result.grd

# 调整范围
gmt grdmath ${TEMP_PREFIX}mod_result.grd ${PI} GT ${TEMP_PREFIX}mod_result.grd ${TWO_PI} SUB ${TEMP_PREFIX}mod_result.grd IFELSE = ${TEMP_PREFIX}adjusted1.grd
gmt grdmath ${TEMP_PREFIX}adjusted1.grd ${PI} NEG LT ${TEMP_PREFIX}adjusted1.grd ${TWO_PI} ADD ${TEMP_PREFIX}adjusted1.grd IFELSE = ${OUTPUT_GRD}

# 备用方案
if [ ! -f "${OUTPUT_GRD}" ] || ! gmt grdinfo "${OUTPUT_GRD}" > /dev/null 2>&1; then
    echo "使用备用方案..."
    cp ${TEMP_PREFIX}filled.grd ${OUTPUT_GRD}
fi

# ========== 完成信息 ==========
echo "===== 处理完成！ ====="
echo "输入文件：${INPUT_GRD}"
echo "输出文件：${OUTPUT_GRD}"
echo "关键参数：低频保护半径=${LOW_FREQ_RADIUS}，角度容差=${ANGLE_TOLERANCE}°"

echo "===== 统计数据 ====="
echo "输入文件:"
gmt grdinfo ${INPUT_GRD} -L1 | grep -E "min:|max:|mean:" || true
echo "输出文件:"
gmt grdinfo ${OUTPUT_GRD} -L1 | grep -E "min:|max:|mean:" || true

echo "临时文件已清理"

# ========== 生成PNG图像（如果指定） ==========
if [ "${GENERATE_PNG}" = "png" ]; then
    echo "===== 生成PNG图像 ====="
    
    # 获取输出网格的范围
    XMIN=$(gmt grdinfo ${OUTPUT_GRD} -C | awk '{print $1}')
    XMAX=$(gmt grdinfo ${OUTPUT_GRD} -C | awk '{print $2}')
    YMIN=$(gmt grdinfo ${OUTPUT_GRD} -C | awk '{print $3}')
    YMAX=$(gmt grdinfo ${OUTPUT_GRD} -C | awk '{print $4}')
    ZMIN=$(gmt grdinfo ${OUTPUT_GRD} -C | awk '{print $9}')
    ZMAX=$(gmt grdinfo ${OUTPUT_GRD} -C | awk '{print $10}')
    
    echo "数据范围: ${ZMIN} 到 ${ZMAX}"
    
    # 创建PNG文件名
    PNG_FILE="${OUTPUT_GRD%.grd}.png"
    
    # 创建rainbow色标
    echo "创建rainbow色标..."
    gmt makecpt -Crainbow -T${ZMIN}/${ZMAX}/0.01 -Z > ${TEMP_PREFIX}color.cpt
    
    # 生成PNG图像
    echo "生成PNG图像: ${PNG_FILE}"
    gmt begin ${PNG_FILE%.png} png
        # 绘制网格图像
        gmt grdimage ${OUTPUT_GRD} -C${TEMP_PREFIX}color.cpt -R${XMIN}/${XMAX}/${YMIN}/${YMAX} -JX15c/10c -Baf -Bx+l"X" -By+l"Y" -BWSne
        
        # 添加色标条
        gmt colorbar -C${TEMP_PREFIX}color.cpt -Baf -By+l"Phase (rad)" -DJMR+w10c/0.5c+h+o0.5c/0c+ml
        
        # 添加标题
        gmt text -F+f16p,Helvetica-Bold+jTC << EOF
${XMIN} ${YMAX} 10 0 0 CB Phase Denoised Result
EOF
    gmt end
    
    echo "PNG图像已生成: ${PNG_FILE}"
fi
