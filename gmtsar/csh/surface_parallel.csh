#!/bin/csh -f
# surface_parallel.csh - GMT Surface分块并行处理脚本
# 使用方法：csh surface_parallel.csh input.xyz output.grd

# 设置错误处理
# set echo
# set verbose

# 检查参数
if ($#argv < 2) then
    echo "用法: $0 <输入文件> <REGION> <输出文件>"
    echo "示例: $0 temp.rat 0/14256/0/28428 pixel.grd"
    exit 1
endif

# ========== 参数配置 ==========
set INPUT_FILE = $1
set REGION = $2
set OUTPUT_FILE = $3

# 区域参数（根据您的数据调整）
set INC = "1/2"          # 网格间隔
set TENSION = 0.5        # 张力因子

# 分块参数
set NX_BLOCKS = 3        # X方向分块数
set NY_BLOCKS = 3        # Y方向分块数
set OVERLAP_X = 300      # X方向重叠像素
set OVERLAP_Y = 300      # Y方向重叠像素

# 计算区域边界
set XMIN = `echo $REGION | awk -F'/' '{print $1}'`
set XMAX = `echo $REGION | awk -F'/' '{print $2}'`
set YMIN = `echo $REGION | awk -F'/' '{print $3}'`
set YMAX = `echo $REGION | awk -F'/' '{print $4}'`

# 计算实际网格大小
set DX = `echo $INC | awk -F'/' '{if(NF==2) print $1; else print $1}'`
set DY = `echo $INC | awk -F'/' '{if(NF==2) print $2; else print $1}'`

# 计算总像素数
set TOTAL_NX = `echo "($XMAX - $XMIN) / $DX" | bc`
set TOTAL_NY = `echo "($YMAX - $YMIN) / $DY" | bc`

# 计算每个块的像素数（不考虑重叠）
set BLOCK_NX = `echo "$TOTAL_NX / $NX_BLOCKS" | bc`
set BLOCK_NY = `echo "$TOTAL_NY / $NY_BLOCKS" | bc`

echo "=============================================="
echo "           GMT Surface 分块并行处理           "
echo "=============================================="
echo "输入文件:    $INPUT_FILE"
echo "输出文件:    $OUTPUT_FILE"
echo "区域:        $REGION"
echo "网格间隔:    $INC"
echo "张力因子:    $TENSION"
echo ""
echo "分块配置:    ${NX_BLOCKS} x ${NY_BLOCKS}"
echo "重叠区域:    ${OVERLAP_X} x ${OVERLAP_Y} 像素"
echo "总网格:      ${TOTAL_NX} x ${TOTAL_NY}"
echo "每块大小:    ${BLOCK_NX} x ${BLOCK_NY} (不含重叠)"
echo "=============================================="

# ========== 步骤1: 创建临时目录 ==========
set TEMP_DIR = "surface_blocks_$$"
set MERGE_DIR = "${TEMP_DIR}/merged"
set BLOCK_DIR = "${TEMP_DIR}/blocks"

echo "\n步骤1: 创建临时目录..."
mkdir -p $TEMP_DIR
mkdir -p $MERGE_DIR
mkdir -p $BLOCK_DIR

if ($status != 0) then
    echo "错误: 无法创建临时目录"
    exit 1
endif

echo "临时目录: $TEMP_DIR"

# ========== 步骤2: 分块并行处理 ==========
echo "\n步骤2: 开始分块并行处理..."

set block_count = 0
set max_concurrent = 9  # 最大并发作业数

# 循环处理所有块
set i = 0
while ($i < $NX_BLOCKS)
    set j = 0
    while ($j < $NY_BLOCKS)
        # 计算块的起始和结束位置（带重叠）
        set block_xmin_tmp = `echo "$XMIN + $i * $BLOCK_NX * $DX - $OVERLAP_X * $DX" | bc`
        set block_xmax_tmp = `echo "$XMIN + ($i + 1) * $BLOCK_NX * $DX + $OVERLAP_X * $DX" | bc`
        set block_ymin_tmp = `echo "$YMIN + $j * $BLOCK_NY * $DY - $OVERLAP_Y * $DY" | bc`
        set block_ymax_tmp = `echo "$YMIN + ($j + 1) * $BLOCK_NY * $DY + $OVERLAP_Y * $DY" | bc`
        
        # 确保不超出边界
        # set block_xmin = `echo "$block_xmin_tmp < $XMIN ? $XMIN : $block_xmin_tmp" | bc`
        if ($block_xmin_tmp < $XMIN) then
            set block_xmin = $XMIN
        else
            set block_xmin = $block_xmin_tmp
        endif

        #set block_xmax = `echo "$block_xmax_tmp > $XMAX ? $XMAX : $block_xmax_tmp" | bc`
        if ($block_xmax_tmp > $XMAX) then
            set block_xmax = $XMAX
        else
            set block_xmax = $block_xmax_tmp
        endif

        # set block_ymin = `echo "$block_ymin_tmp < $YMIN ? $YMIN : $block_ymin_tmp" | bc`
        if ($block_ymin_tmp < $YMIN) then
            set block_ymin = $YMIN
        else
            set block_ymin = $block_ymin_tmp
        endif

        # set block_ymax = `echo "$block_ymax_tmp > $YMAX ? $YMAX : $block_ymax_tmp" | bc`
        if ($block_ymax_tmp > $YMAX) then
            set block_ymax = $YMAX
        else
            set block_ymax = $block_ymax_tmp
        endif
        
        set block_region = "${block_xmin}/${block_xmax}/${block_ymin}/${block_ymax}"
        set block_output = "$BLOCK_DIR/block_${i}_${j}.grd"
        
        @ block_count++
        echo "启动块 $i,$j (总第${block_count}个): $block_region"
        
        # 后台运行surface命令
        gmt surface $INPUT_FILE -R$block_region  -I$INC -T$TENSION -G$block_output  -N1000  -r  -V >& $BLOCK_DIR/block_${i}_${j}.log& 
        
        # 检查当前后台作业数
        @ current_jobs = `jobs -p | wc -l`
        if ($current_jobs >= $max_concurrent) then
            echo "达到最大并发数($max_concurrent)，等待..."
            wait  # 等待所有当前后台作业完成
        endif
        
        @ j++
    end
    @ i++
end

echo "\n等待所有剩余块处理完成..."
wait  # 等待所有后台进程完成

echo "所有分块处理完成！"

# ========== 步骤3: 检查处理结果 ==========
echo "\n步骤3: 检查处理结果..."

set success_count = 0
set i = 0
while ($i < $NX_BLOCKS)
    set j = 0
    while ($j < $NY_BLOCKS)
        set block_file = "$BLOCK_DIR/block_${i}_${j}.grd"
        if (-e $block_file) then
            @ success_count++
            echo "块 $i,$j: 成功 ($block_file)"
        else
            echo "块 $i,$j: 失败 (文件不存在)"
        endif
        @ j++
    end
    @ i++
end

if ($success_count == 0) then
    echo "错误: 所有块处理失败！"
    echo "检查日志文件:"
    ls -la $BLOCK_DIR/*.log
    exit 1
endif

echo "成功处理 ${success_count}/$block_count 个块"

# ========== 步骤4: 裁剪重叠区域 ==========
echo "\n步骤4: 裁剪重叠区域..."

set cropped_files = ()
set i = 0
while ($i < $NX_BLOCKS)
    set j = 0
    while ($j < $NY_BLOCKS)
        # 计算实际区域（去掉重叠）
        set actual_xmin = `echo "$XMIN + $i * $BLOCK_NX * $DX" | bc`
        set actual_xmax = `echo "$XMIN + ($i + 1) * $BLOCK_NX * $DX" | bc`
        set actual_ymin = `echo "$YMIN + $j * $BLOCK_NY * $DY" | bc`
        set actual_ymax = `echo "$YMIN + ($j + 1) * $BLOCK_NY * $DY" | bc`
        
        # 最后一个块调整到实际边界
        if ($i == ($NX_BLOCKS - 1)) then
            set actual_xmax = $XMAX
        endif
        if ($j == ($NY_BLOCKS - 1)) then
            set actual_ymax = $YMAX
        endif
        
        set actual_region = "${actual_xmin}/${actual_xmax}/${actual_ymin}/${actual_ymax}"
        set block_input = "$BLOCK_DIR/block_${i}_${j}.grd"
        set cropped_output = "$MERGE_DIR/block_${i}_${j}_cropped.grd"
        
        if (-e $block_input) then
            echo "裁剪块 $i,$j: $actual_region"
            gmt grdcut $block_input -R$actual_region -G$cropped_output -V
            
            if (-e $cropped_output) then
                set cropped_files = ($cropped_files $cropped_output)
            endif
        endif
        
        @ j++
    end
    @ i++
end

if ($#cropped_files == 0) then
    echo "错误: 没有裁剪后的文件"
    exit 1
endif

echo "裁剪完成，共 $#cropped_files 个文件"

# ========== 步骤5: 合并所有块 ==========
echo "\n步骤5: 合并所有块..."

if ($#cropped_files == 1) then
    # 只有一个文件，直接复制
    echo "只有一个块，直接复制..."
    cp $cropped_files[1] $OUTPUT_FILE
else
    # 使用grdblend合并多个文件
    echo "使用grdblend合并 $#cropped_files 个文件..."
    
    # 构建grdblend命令
    set blend_cmd = "gmt grdblend"
    foreach file ($cropped_files)
        set blend_cmd = "$blend_cmd $file"
    end
    
    set blend_cmd = "$blend_cmd -R$REGION -I$INC -G$OUTPUT_FILE -V"
    
    echo "执行命令: $blend_cmd"
    eval $blend_cmd
    
    if ($status != 0) then
        echo "错误: grdblend合并失败"
        exit 1
    endif
endif

# ========== 步骤6: 验证结果 ==========
echo "\n步骤6: 验证结果..."

if (-e $OUTPUT_FILE) then
    echo "输出文件创建成功: $OUTPUT_FILE"
    echo "\n网格信息:"
    gmt grdinfo $OUTPUT_FILE
    
    # 检查文件大小
    set file_size = `ls -lh $OUTPUT_FILE | awk '{print $5}'`
    echo "文件大小: $file_size"
else
    echo "错误: 输出文件未创建"
    exit 1
endif

# ========== 步骤7: 可选清理 ==========
echo "\n步骤7: 清理临时文件..."

echo "是否清理临时文件? (y/n)"
set cleanup = $<
if ($cleanup == 'y' || $cleanup == 'Y') then
    echo "清理临时目录: $TEMP_DIR"
    rm -rf $TEMP_DIR
else
    echo "保留临时目录: $TEMP_DIR"
    echo "块文件位置: $BLOCK_DIR"
    echo "裁剪文件位置: $MERGE_DIR"
endif

echo "\n=============================================="
echo "处理完成！"
echo "输出文件: $OUTPUT_FILE"
echo "总耗时: 计算中..."
echo "=============================================="

exit 0