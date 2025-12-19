#!/bin/bash
# 数据筛选脚本：剔除x偏移量（第二列）和y偏移量（第四列）的异常值
# 作者：ysdong@cug.edu.cn
# 日期：2025年12月18日
# 支持算法：IQR（四分位距）、Z-score（标准正态分布）、modified_zscore（修正Z-score，抗极端值）

# ------------ 参数校验：bash语法适配 ------------
if [ $# -eq 0 ]; then
    echo " "
    echo "Usage: filter_offset.sh input_file output_file [算法选项]"
    echo "  算法选项：iqr（默认）、zscore、modified_zscore"
    echo "  示例1：./filter_offset.sh data.txt result.txt  # 默认用修正Z-score"
    echo "  示例2：./filter_offset.sh data.txt result.txt zscore  # 用Z-score法"
    echo " "
    exit 1
fi

# ===================== 核心配置参数（可根据需求调整） =====================
input_file="$1"                # 命令行第1个参数：输入文件
output_file="$2"               # 命令行第2个参数：输出文件
outlier_method="${3:-modified_zscore}"  # 异常值检测算法（默认修正Z-score）
x_offset_col=2                 # x偏移量列（固定第二列）
y_offset_col=4                 # y偏移量列（固定第四列）
iqr_coeff=1.5                  # IQR法系数（1.5为常用值，值越大筛选越宽松）
z_threshold=3.0                # Z-score法阈值（|Z|>3为异常，正态分布99.7%置信区间）
modified_z_threshold=3.29      # 修正Z-score阈值（|MZ|>3.29为异常，更抗极端值）
# 临时文件：拆分x/y到独立文件（避免数据混合统计）
x_temp="x_offset_temp.txt"
y_temp="y_offset_temp.txt"

# ===================== 算法校验与初始化 =====================
# 校验输入的算法是否支持
if [ "$outlier_method" != "iqr" ] && [ "$outlier_method" != "zscore" ] && [ "$outlier_method" != "modified_zscore" ]; then
    echo "错误：不支持的算法！仅支持 iqr、zscore、modified_zscore"
    exit 1
fi
echo "=== 初始化信息 ==="
echo "输入文件：$input_file"
echo "输出文件：$output_file"
echo "使用算法：$outlier_method"
echo "==================="

# ===================== 数据预处理 =====================
# 1. 检查输入文件存在性
if [ ! -f "$input_file" ]; then
    echo "错误：输入文件 $input_file 不存在！"
    exit 1
fi

# 2. 拆分x/y偏移量到独立临时文件（仅保留5列有效数据）
awk '{if(NF==5) print $2}' "$input_file" > "$x_temp"
awk '{if(NF==5) print $4}' "$input_file" > "$y_temp"

# 3. 检查偏移量数据有效性（避免空文件导致后续计算错误）
if [ ! -s "$x_temp" -o ! -s "$y_temp" ]; then
    echo "错误：输入文件中无有效数据（需每行5列，列间用空格/制表符分隔）！"
    rm -f "$x_temp" "$y_temp"
    exit 1
fi
echo "临时文件生成成功：x偏移量（$x_temp）、y偏移量（$y_temp）"

# ===================== 异常值检测核心算法（三种算法实现） =====================
# 算法1：IQR法（原算法，保留供对比）
# 输入：单列数据文件；输出：下限 上限
calc_iqr_bounds() {
    local data_file="$1"
    # 排序（保留重复数据，统计更准确）
    sort -n "$data_file" > sorted.tmp
    local n=$(wc -l < sorted.tmp)
    if [ "$n" -eq 0 ]; then
        echo "0 0"
        rm -f sorted.tmp
        return
    fi
    
    # 计算Q1（25%分位数）、Q3（75%分位数）位置（线性插值法）
    local q1_pos=$(echo "scale=2; ($n + 1) * 0.25" | bc)
    local q3_pos=$(echo "scale=2; ($n + 1) * 0.75" | bc)
    
    # 提取整数/小数部分 + 对应位置数值
    local q1_int=$(echo "$q1_pos" | cut -d. -f1)
    local q1_dec=$(echo "$q1_pos" | cut -d. -f2)
    local q3_int=$(echo "$q3_pos" | cut -d. -f1)
    local q3_dec=$(echo "$q3_pos" | cut -d. -f2)
    local val_q1=$(sed -n "$q1_int p" sorted.tmp)
    local val_q3=$(sed -n "$q3_int p" sorted.tmp)
    local val_q1_next=$(sed -n "$((q1_int + 1)) p" sorted.tmp)
    local val_q3_next=$(sed -n "$((q3_int + 1)) p" sorted.tmp)
    
    # 线性插值计算Q1/Q3
    local q1=$(echo "scale=4; $val_q1 + $q1_dec * 0.01 * ($val_q1_next - $val_q1)" | bc)
    local q3=$(echo "scale=4; $val_q3 + $q3_dec * 0.01 * ($val_q3_next - $val_q3)" | bc)
    local iqr=$(echo "scale=4; $q3 - $q1" | bc)
    
    # 计算边界（处理IQR=0的极端情况）
    local lower upper
    if [ "$(echo "$iqr == 0" | bc)" -eq 1 ]; then
        lower="$q1"
        upper="$q3"
    else
        lower=$(echo "scale=4; $q1 - $iqr_coeff * $iqr" | bc)
        upper=$(echo "scale=4; $q3 + $iqr_coeff * $iqr" | bc)
    fi
    
    rm -f sorted.tmp
    echo "$lower $upper"
}

# 算法2：Z-score法（基于正态分布，适合近似正态分布数据）
# 输入：单列数据文件；输出：下限 上限（|Z|<=阈值为正常）
calc_zscore_bounds() {
    local data_file="$1"
    # 用awk计算均值（mean）和标准差（std），比bash循环更高效
    local stats=$(awk '
        BEGIN { sum=0; sum_sq=0; n=0 }
        { sum+=$1; sum_sq+=$1*$1; n++ }
        END {
            mean=sum/n;
            std=sqrt((sum_sq - sum*sum/n)/n);
            print mean, std
        }' "$data_file")
    local mean=$(echo "$stats" | awk '{print $1}')
    local std=$(echo "$stats" | awk '{print $2}')
    
    # 处理标准差为0的极端情况（所有值相同）
    if [ "$(echo "$std == 0" | bc)" -eq 1 ]; then
        lower="$mean"
        upper="$mean"
    else
        lower=$(echo "scale=4; $mean - $z_threshold * $std" | bc)
        upper=$(echo "scale=4; $mean + $z_threshold * $std" | bc)
    fi
    
    echo "$lower $upper $mean $std"  # 额外返回均值和标准差用于日志
}

# 算法3：修正Z-score法（基于中位数，抗极端值能力强，推荐优先使用）
# 输入：单列数据文件；输出：下限 上限（|修正Z|<=阈值为正常）
calc_modified_zscore_bounds() {
    local data_file="$1"
    # 步骤1：计算中位数（median）
    local median=$(sort -n "$data_file" | awk -v n=$(wc -l < "$data_file") '
        NR == int((n+1)/2) {print $1; exit}  # 奇数行取中间行，偶数行取上中间行
    ')
    
    # 步骤2：计算中位数绝对偏差（MAD）
    local mad=$(awk -v med="$median" '{print sqrt(($1 - med)^2)}' "$data_file" | sort -n | \
                awk -v n=$(wc -l < "$data_file") '
                    NR == int((n+1)/2) {print $1; exit}
                ')
    
    # 步骤3：计算边界（0.6745为正态分布下的常数，用于标准化MAD）
    local lower upper
    if [ "$(echo "$mad == 0" | bc)" -eq 1 ]; then
        lower="$median"
        upper="$median"
    else
        # local scale_factor=$(echo "scale=4; 0.6745 / $mad" | bc)
        local scale_factor=$(echo "scale=4; 0.6745 / $mad" | bc)
        lower=$(echo "scale=4; $median - $modified_z_threshold / $scale_factor" | bc)
        upper=$(echo "scale=4; $median + $modified_z_threshold / $scale_factor" | bc)
    fi
    
    echo "$lower $upper $median $mad"  # 额外返回中位数和MAD用于日志
}

# ===================== 计算异常值边界（根据选择的算法调用对应函数） =====================
echo -e "\n=== 异常值边界计算（算法：$outlier_method） ==="
# 计算x偏移量边界
if [ "$outlier_method" == "iqr" ]; then
    x_bounds=$(calc_iqr_bounds "$x_temp")
    x_lower=$(echo "$x_bounds" | awk '{print $1}')
    x_upper=$(echo "$x_bounds" | awk '{print $2}')
    echo "x偏移量（第二列）- IQR法：Q1=$x_q1, Q3=$x_q3, 边界[$x_lower, $x_upper]"
elif [ "$outlier_method" == "zscore" ]; then
    x_stats=$(calc_zscore_bounds "$x_temp")
    x_lower=$(echo "$x_stats" | awk '{print $1}')
    x_upper=$(echo "$x_stats" | awk '{print $2}')
    x_mean=$(echo "$x_stats" | awk '{print $3}')
    x_std=$(echo "$x_stats" | awk '{print $4}')
    echo "x偏移量（第二列）- Z-score法：均值=$x_mean, 标准差=$x_std, 边界[$x_lower, $x_upper]"
else  # modified_zscore
    x_stats=$(calc_modified_zscore_bounds "$x_temp")
    x_lower=$(echo "$x_stats" | awk '{print $1}')
    x_upper=$(echo "$x_stats" | awk '{print $2}')
    x_median=$(echo "$x_stats" | awk '{print $3}')
    x_mad=$(echo "$x_stats" | awk '{print $4}')
    echo "x偏移量（第二列）- 修正Z-score法：中位数=$x_median, MAD=$x_mad, 边界[$x_lower, $x_upper]"
fi

# 计算y偏移量边界（逻辑同x）
if [ "$outlier_method" == "iqr" ]; then
    y_bounds=$(calc_iqr_bounds "$y_temp")
    y_lower=$(echo "$y_bounds" | awk '{print $1}')
    y_upper=$(echo "$y_bounds" | awk '{print $2}')
    echo "y偏移量（第四列）- IQR法：Q1=$y_q1, Q3=$y_q3, 边界[$y_lower, $y_upper]"
elif [ "$outlier_method" == "zscore" ]; then
    y_stats=$(calc_zscore_bounds "$y_temp")
    y_lower=$(echo "$y_stats" | awk '{print $1}')
    y_upper=$(echo "$y_stats" | awk '{print $2}')
    y_mean=$(echo "$y_stats" | awk '{print $3}')
    y_std=$(echo "$y_stats" | awk '{print $4}')
    echo "y偏移量（第四列）- Z-score法：均值=$y_mean, 标准差=$y_std, 边界[$y_lower, $y_upper]"
else  # modified_zscore
    y_stats=$(calc_modified_zscore_bounds "$y_temp")
    y_lower=$(echo "$y_stats" | awk '{print $1}')
    y_upper=$(echo "$y_stats" | awk '{print $2}')
    y_median=$(echo "$y_stats" | awk '{print $3}')
    y_mad=$(echo "$y_stats" | awk '{print $4}')
    echo "y偏移量（第四列）- 修正Z-score法：中位数=$y_median, MAD=$y_mad, 边界[$y_lower, $y_upper]"
fi

# ===================== 筛选有效数据（核心逻辑：双列边界同时满足） =====================
echo -e "\n=== 开始筛选数据 ==="
awk -v x_l="$x_lower" -v x_u="$x_upper" -v y_l="$y_lower" -v y_u="$y_upper" -v method="$outlier_method" '
    BEGIN { count=0 }
    {
        # 仅保留5列有效数据，且x/y偏移量均在边界内
        if(NF==5 && $2 >= x_l && $2 <= x_u && $4 >= y_l && $4 <= y_u) {
            print $0;
            count++
        }
    }
#    END { print "筛选出有效数据行数：" count }
' "$input_file" > "$output_file"

# ===================== 结果统计与清理 =====================
original_count=$(wc -l < "$input_file")
filtered_count=$(wc -l < "$output_file")
removed_count=$((original_count - filtered_count))

echo -e "\n=== 筛选结果汇总 ==="
echo "原始数据总行数：$original_count"
echo "筛选后有效行数：$filtered_count"
echo "剔除异常值行数：$removed_count"
echo "有效数据保存路径：$output_file"

# 清理临时文件（避免残留）
rm -f "$x_temp" "$y_temp"
echo -e "\n临时文件已清理，脚本执行完成！"
exit 0
