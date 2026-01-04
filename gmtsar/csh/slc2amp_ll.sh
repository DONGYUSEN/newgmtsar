#!/bin/csh -f
# ysdong@cug.edu.cn,2025/12/25
#
unset noclobber
# 
# Converts a complex SLC file to a real amplitude grd, and project to ll
# file using optional filter and a PRM file
#
# define the filters
#
  set sharedir = `gmtsar_sharedir.csh`
  set fil1 = $sharedir/filters/gauss5x3
  set fil2 = $sharedir/filters/gauss9x5 
#
# check for number of arguments
#
  if ($#argv != 2 ) then
    echo ""
    echo "Usage: slc2ampll.csh name rng_dec" 
    echo "       need SLC PRM LED trans.dat in the folder"
    echo "       rng_dec is range decimation"
    echo "       e.g. 1 for ERS ENVISAT ALOS FBD" 
    echo "            2 for ALOS FBS DJ1 GF3 " 
    echo "            4 for TSX"
    echo ""     output geo_name.tif
    echo "Example: slc2ampll.csh 20241102 2" 
    echo ""
    exit 1
  endif 
#
# filter the amplitudes done in conv
# check the input and output filename before 
#
set name = $1

cd topo
  
echo "=== 逐个检查文件存在性 ==="
# 检查 .SLC 文件
if (-e "${name}.SLC") then
    echo "✅ 存在文件：${name}.SLC"
else
    ln -s ../raw/${name}.SLC ."
endif

# 检查 .PRM 文件
if (-e "${name}.PRM") then
    echo "✅ 存在文件：${name}.PRM"
else
    ln -s ../raw/${name}.PRM ."
endif

# 检查 .LED 文件
if (-e "${name}.LED") then
    echo "✅ 存在文件：${name}.LED"
else
    ln -s ../raw/${name}.LED ."
endif


# 检查 .trans.dat 文件
if (-e "trans.dat") then
    echo "✅ 存在文件：trans.dat"
else
    echo "❌ 缺失文件: trans.dat"
    exit 1
endif
  

    echo " range decimation is:" $2
    conv 2 $2 $fil1 $1.PRM test22.grd =bf
    gmt grdmath test22.grd LOG2 100 ADD = test2.grd
    proj_ra2ll.csh  trans.dat  test2.grd test2_ll.grd 8
    gmt grd2cpt  test2.grd  -Cgray  -Z  > test2.cpt
    set filename = "geo_$1.tif"
    gmt grdimage test2_ll.grd -Ctest2.cpt -JX5c -A$filename
    mv  test2_ll.grd geo_$1.grd
    rm test2*

# 
# get the zmin and zmax value
#
  # gmt grdmath $3=bf 1 MUL = $3
