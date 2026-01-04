#!/bin/bash
# GMT 6.5实现SAR相位图去水平/垂直条纹并生成PNG图像的脚本
# 依赖：GMT ≥ 6.5、bc（系统自带）
# 使用方法：chmod +x phase_denoise_gmt65.sh && ./phase_denoise_gmt65.sh phase.grd phase_denoised.grd [png]

# ========== 参数配置 ==========
INPUT_GRD=$1
OUTPUT_GRD=$2
GENERATE_PNG=$3  # 第三个参数，如果设置为"png"则生成PNG
TEMP_PREFIX="temp_phase_"  # 临时文件前缀
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

# ... (前面的所有步骤保持不变，直到完成信息部分) ...

# ========== 完成信息 ==========
echo "===== 处理完成！ ====="
echo "输入文件：${INPUT_GRD}"
echo "输出文件：${OUTPUT_GRD}"
echo "关键参数：低频保护半径=${LOW_FREQ_RADIUS}，角度容差=${ANGLE_TOLERANCE}°"

# 输出网格信息
echo "===== 输出文件基本信息 ====="
gmt grdinfo ${OUTPUT_GRD} | grep -E "v_min|v_max|n_columns|n_rows" || true

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

# 列出生成的临时文件
echo "===== 临时文件列表 ====="
ls -la ${TEMP_PREFIX}*.grd 2>/dev/null | wc -l | awk '{print "生成了 "$1" 个临时文件"}'
