#!/bin/csh -f
# ysdong, Dec, 2025 
# 修复：1. 避免删除程序运行目录文件 2. 统一转换为绝对路径
# 使用方法：./脚本名 <zip文件1> [zip文件2 ...] <目录A路径>

# ===================== 1. 检查输入参数 =====================
if ($#argv < 2) then
    echo "【错误】参数数量错误！"
    echo "正确用法: $0 <zip文件1> [zip文件2 ...] <目录A路径>"
    echo "示例: $0 file1.zip file2.zip /data/dirA"
    exit 1
endif

# ===================== 2. 分离参数并转换为绝对路径 =====================
# 2.1 提取目录A并转换为绝对路径（核心：避免相对路径导致的目录混淆）
set dir_A_input = "$argv[$#argv]"
# 转换为绝对路径（无论输入是相对/绝对路径，都转为绝对路径）
if (-d "$dir_A_input") then
    set dir_A = `cd "$dir_A_input" && pwd`  # 目录存在则直接取绝对路径
else
    # 目录不存在则先创建父目录，再取绝对路径
    set dir_A_parent = `dirname "$dir_A_input"`
    mkdir -p "$dir_A_parent"
    set dir_A = `cd "$dir_A_parent" && pwd`/`basename "$dir_A_input"`

endif
# 确保raw子目录存在（绝对路径）
mkdir -p "$dir_A/raw"
set dir_topo = $dir_A/topo
mkdir -p "$dir_topo"

# 2.2 提取所有zip文件并转换为绝对路径
set zip_files = ""           # 初始化zip文件列表
foreach arg ($argv)
    if ("$arg" != "$dir_A_input") then
        # 转换每个zip文件为绝对路径
        if (-f "$arg") then
            set zip_dir = `dirname "$arg"`
            set zip_base = `basename "$arg"`
            set abs_zip = `cd "$zip_dir" && pwd`/"$zip_base"
        else
            set abs_zip = "$arg"  # 不存在的文件保留原路径，后续检查
        endif
        set zip_files = "$zip_files $abs_zip"
    endif
end

# 移除zip_files开头的空格
set zip_files = `echo "$zip_files" | sed 's/^ //'`

# 检查是否有有效的zip文件
if ("$zip_files" == "") then
    echo "【错误】未检测到有效的zip文件参数！"
    exit 1
endif

# ===================== 3. 批量处理每个zip文件（基于绝对路径） =====================
foreach zip_file ($zip_files)
    echo "\n========================================"
    echo "【信息】开始处理zip文件: $zip_file"
    echo "========================================"

    # 检查zip文件是否存在（绝对路径检查，更准确）
    if (! -f "$zip_file") then
        echo "【警告】zip文件 $zip_file 不存在，跳过该文件！"
        continue
    endif

    # -------------------- 提取name1（第17-24位，仅基于文件名） --------------------
    set temp_name = "`basename "$zip_file"`"
    set name1 = `echo "$temp_name" | cut -c17-24`
    if ("$name1" == "") then
        echo "【警告】从 $zip_file 提取name失败（结果为空），跳过该文件！"
        continue
    endif
    echo "【信息】提取的name: $name1"

    # -------------------- 解压zip文件（绝对路径解压，避免路径混淆） --------------------
    echo "【信息】正在解压 $zip_file 到 $dir_A ..."
    unzip -q "$zip_file" -d "$dir_A"
    if ($status != 0) then
        echo "【警告】$zip_file 解压失败，跳过该文件！"
        # 清理：仅针对dir_A下的临时文件（绝对路径，绝不涉及运行目录）
        find "$dir_A" -mindepth 1 -maxdepth 1 ! -path "$dir_A/raw" -exec rm -rf {} +
        continue
    endif

    # -------------------- 查找目标文件（绝对路径查找） --------------------
    set tiff_file = `find "$dir_A" -type f -path "*/measurement/*.tiff" -print | head -n 1`
    #set xml_file = `find  "$dir_A" -maxdepth 2 -type f -path "*/annotation/*.xml" -print | head -n 1`
    #echo $xml_file 
    set xml_file = `find "$dir_A" -type f -path "*/annotation/*.xml" | grep '/annotation/[^/]*.xml$'  | head -n 1`
    set png_file = `find "$dir_A" -type f -path "*/preview/quick-look.png" -print | head -n 1`
    set kml_file = `find "$dir_A" -type f -path "*/preview/map-overlay.kml" -print | head -n 1`

    # 校验必选文件
    set error_flag = 0
    if ("$tiff_file" == "") then
        echo "【警告】未找到 $zip_file 中的measurement/*.tiff文件！"
        set error_flag = 1
    endif
    if ("$xml_file" == "") then
        echo "【警告】未找到 $zip_file 中的annotation/*.xml文件！"
        set error_flag = 1
    endif
    if ($error_flag == 1) then
        # 清理：仅操作dir_A下的文件，与运行目录隔离
        find "$dir_A" -mindepth 1 -maxdepth 1 ! -path "$dir_A/raw" ! -path  "$dir_topo" -exec rm -rf {} +
        continue
    endif

    # -------------------- 移动并重命名文件（绝对路径操作） --------------------
    echo "【信息】移动tiff文件: $tiff_file -> $dir_A/raw/$name1.tiff"
    mv "$tiff_file" "$dir_A/raw/$name1.tiff"
    echo "【信息】移动xml文件: $xml_file -> $dir_A/raw/$name1.xml"
    mv "$xml_file" "$dir_A/raw/$name1.xml"

    if ("$png_file" != "") then
        echo "【信息】移动png文件: $png_file -> $dir_A/raw/$name1.png"
        mv "$png_file" "$dir_A/raw/$name1.png"
    else
        echo "【提示】未找到 $zip_file 中的preview/quick-look.png文件"
    endif
    if ("$kml_file" != "") then
        echo "【信息】移动kml文件: $kml_file -> $dir_A/raw/$name1.kml"
        mv "$kml_file" "$dir_A/raw/$name1.kml"
    else
        echo "【提示】未找到 $zip_file 中的preview/map-overlay.kml文件"
    endif

    # -------------------- 清理解压临时文件（关键：仅清理dir_A下的非raw目录） --------------------
    echo "【信息】清理 $zip_file 解压的临时文件..."
    # 使用绝对路径限定范围，! -path "$dir_A/raw" 确保不删除raw目录
    find "$dir_A" -mindepth 1 -maxdepth 1 ! -path "$dir_A/raw" ! -path "$dir_topo" -exec rm -rf {} +
    echo "【信息】$zip_file 处理完成！"
end
# -------------------- make dem --------------------
cd "$dir_topo" 

#!/bin/csh

# 从XML文件中提取coordinates标签内的内容
set coords_str = `sed -n '/<coordinates>/ { s/<coordinates>//; s/<\/coordinates>//; s/^[[:space:]]*//; s/[[:space:]]*$//; p }' "$dir_A/raw/$name1.kml"`

# 将坐标字符串转换为数组（格式：lon0 lat0 lon1 lat1 lon2 lat2 lon3 lat3）
set coords = ( `echo $coords_str | tr ' ,' ' '` )

# 提取所有经度和纬度值
set lon = ( $coords[1] $coords[3] $coords[5] $coords[7] )
set lat = ( $coords[2] $coords[4] $coords[6] $coords[8] )

# 计算经度的最大最小值
set lonmax = `echo $lon | awk '{max=$1; for(i=1;i<=NF;i++) if($i>max) max=$i; print max}'`
set lonmin = `echo $lon | awk '{min=$1; for(i=1;i<=NF;i++) if($i<min) min=$i; print min}'`

# 计算纬度的最大最小值
set latmax = `echo $lat | awk '{max=$1; for(i=1;i<=NF;i++) if($i>max) max=$i; print max}'`
set latmin = `echo $lat | awk '{min=$1; for(i=1;i<=NF;i++) if($i<min) min=$i; print min}'`

# 保留1位小数并加0.1
set lonmax_final = `echo $lonmax | awk '{printf "%.1f", $1 + 0.1}'`
set lonmin_final = `echo $lonmin | awk '{printf "%.1f", $1 - 0.1}'`
set latmax_final = `echo $latmax | awk '{printf "%.1f", $1 + 0.1}'`
set latmin_final = `echo $latmin | awk '{printf "%.1f", $1 - 0.1}'`

# make dem
make_dem.csh $lonmin_final $lonmax_final $latmin_final $latmax_final 1
cd "$dir_A"
pop_config.csh DJ1 > config.DJ1.txt

# ===================== 4. 处理结束 =====================
echo "\n【信息】所有文件处理完毕！"
echo "【结果】所有文件已保存至: $dir_A"
exit 0
