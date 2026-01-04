#!/bin/csh -f
# 用于从zip文件直接生成geotiff，便于下一步使用  
  
if ($#argv < 2) then
    echo ""
    echo "用法: DJ1_SLC2Geotiff.csh  zipfile output_dir"
    echo ""
    echo "例子: DJ1_SLC2Geotiff.csh  20241110.zip 20241110 "
    echo ""
    echo "     Put the data and orbit files in the raw folder, put DEM in the topo folder"
    echo ""
    exit 1
  endif

date
set OMP_NUM_THREADS = 12

# 移除zip_files开头的空格
set zip_files = `echo "$1" | sed 's/^ //'`
set temp_name = "`basename "$zip_file"`"
set master = `echo "$temp_name" | cut -c17-24`


# 导入数据
echo "从zip文件中导入SLC，并生成dem"
import_dj1.csh $1 $2

# 生成topo_ra 等
echo "生成topo_ra"
cd $2
cd topo
cp ../SLC/$master.PRM master.PRM 
ln -s ../raw/$master.LED . 
dem2topo_ra.csh master.PRM dem.grd 1

echo "SLC_2_amp, and log2"
cd ../SLC
set rng = `gmt grdinfo ../topo/topo_ra.grd | grep x_inc | awk '{print $7}'`
slc2amp.csh $master.PRM $rng $master.grd 
gmt grdmath $master.grd MUL LOG2 100 ADD  = final-master.grd

set AMAX2 = `gmt grdinfo -L2 final-master.grd | grep stdev | awk '{ print 3*$5 }'`
set tmp_c1 = `gmt grdinfo final-master.grd -T+a`
set tmp_zmin = `echo $tmp_c1 | awk -F'[-T/]' '{print $3}'`
set tmp_zmax = `echo $tmp_c1 | awk -F'[-T/]' '{print $4}'`
gmt grd2cpt final-master.grd -Cgray -T$tmp_zmin/$tmp_zmax/0.1 -Z -Di  > final-master.cpt
echo "N  255   255   254" >> final-master.cpt
# geocoding 地理编码
echo "将结果投影到经纬度坐标下，geocode.csh"
proj_ra2ll.csh trans.dat final-master.grd final-master_ll.grd 8
# 生成geotiff
echo "生成geotiff"
grd2geotiff.csh final-master_ll final-master.cpt  




