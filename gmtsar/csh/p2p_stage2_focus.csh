#!/bin/csh -f
#
# p2p_processing.csh 的 Stage-2A
# Stage-2A for p2p_processing.csh
# 功能：准备 SLC 输入并在需要时执行 L0->L1 聚焦
# Function: prepare SLC inputs and perform L0->L1 focusing when required
#

if ($#argv != 7) then
  echo ""
  echo "用法 / Usage: p2p_stage2_focus.csh SAT master aligned skip_master iono separate_focus data_level"
  echo "参数说明 / Args:"
  echo "  skip_master: 0=都处理 both, 1=跳过master skip master, 2=跳过aligned skip aligned"
  echo "  iono: 0|1"
  echo "  separate_focus: 0|1"
  echo "  data_level: 0=L0输入, 1=L1输入"
  echo ""
  exit 1
endif

set SAT = $1
set master = $2
set aligned = $3
set skip_master = $4
set iono = $5
set separate_focus = $6
set data_level = $7

if ($skip_master != 0 && $skip_master != 1 && $skip_master != 2) then
  echo "错误参数：skip_master=$skip_master / Invalid parameter: skip_master=$skip_master"
  exit 1
endif

if ($iono != 0 && $iono != 1) then
  echo "错误参数：iono=$iono / Invalid parameter: iono=$iono"
  exit 1
endif

if ($separate_focus != 0 && $separate_focus != 1) then
  echo "错误参数：separate_focus=$separate_focus / Invalid parameter: separate_focus=$separate_focus"
  exit 1
endif

if ($data_level != 0 && $data_level != 1) then
  echo "错误参数：data_level=$data_level / Invalid parameter: data_level=$data_level"
  exit 1
endif

set is_raw_sat = 0
if ($SAT == "ERS" || $SAT == "ENVI" || $SAT == "ALOS" || $SAT == "CSK_RAW") then
  set is_raw_sat = 1
endif

if ($SAT == "S1_TOPS") then
  exit 0
endif

if ($data_level == 1) then
  # L1/SLC 模式：直接使用现有 PRM/SLC/LED，不做聚焦
  # L1/SLC mode: use existing PRM/SLC/LED and skip focusing.
  if ($skip_master == 0 || $skip_master == 2) then
    if (! -f ../raw/$master.PRM || ! -f ../raw/$master.SLC || ! -f ../raw/$master.LED) then
      echo "raw/ 缺少 master 的 L1 文件 / Missing master L1 files in raw/: $master.PRM/.SLC/.LED"
      exit 2
    endif
    cp ../raw/$master.PRM .
    rm -f $master.SLC $master.LED
    ln -s ../raw/$master.SLC .
    ln -s ../raw/$master.LED .
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    if (! -f ../raw/$aligned.PRM || ! -f ../raw/$aligned.SLC || ! -f ../raw/$aligned.LED) then
      echo "raw/ 缺少 aligned 的 L1 文件 / Missing aligned L1 files in raw/: $aligned.PRM/.SLC/.LED"
      exit 2
    endif
    cp ../raw/$aligned.PRM .
    rm -f $aligned.SLC $aligned.LED
    ln -s ../raw/$aligned.SLC .
    ln -s ../raw/$aligned.LED .
  endif
  exit 0
endif

if ($is_raw_sat == 1) then
  if ($skip_master == 0 || $skip_master == 2) then
    if (! -f ../raw/$master.PRM || ! -f ../raw/$master.raw || ! -f ../raw/$master.LED) then
      echo "raw/ 缺少 master 的 L0 文件 / Missing master L0 files in raw/: $master.PRM/.raw/.LED"
      exit 2
    endif
    cp ../raw/$master.PRM .
    rm -f $master.raw $master.LED
    ln -s ../raw/$master.raw .
    ln -s ../raw/$master.LED .
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    if (! -f ../raw/$aligned.PRM || ! -f ../raw/$aligned.raw || ! -f ../raw/$aligned.LED) then
      echo "raw/ 缺少 aligned 的 L0 文件 / Missing aligned L0 files in raw/: $aligned.PRM/.raw/.LED"
      exit 2
    endif
    cp ../raw/$aligned.PRM .
    rm -f $aligned.raw $aligned.LED
    ln -s ../raw/$aligned.raw .
    ln -s ../raw/$aligned.LED .
  endif
else
  if ($skip_master == 0 || $skip_master == 2) then
    if (! -f ../raw/$master.PRM || ! -f ../raw/$master.SLC || ! -f ../raw/$master.LED) then
      echo "raw/ 缺少 master 的 L1 文件 / Missing master L1 files in raw/: $master.PRM/.SLC/.LED"
      exit 2
    endif
    cp ../raw/$master.PRM .
    rm -f $master.SLC $master.LED
    ln -s ../raw/$master.SLC .
    ln -s ../raw/$master.LED .
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    if (! -f ../raw/$aligned.PRM || ! -f ../raw/$aligned.SLC || ! -f ../raw/$aligned.LED) then
      echo "raw/ 缺少 aligned 的 L1 文件 / Missing aligned L1 files in raw/: $aligned.PRM/.SLC/.LED"
      exit 2
    endif
    cp ../raw/$aligned.PRM .
    rm -f $aligned.SLC $aligned.LED
    ln -s ../raw/$aligned.SLC .
    ln -s ../raw/$aligned.LED .
  endif
endif

if ($is_raw_sat == 1 && $iono == 1) then
  # 为电离层估计统一 fd1/chirp_ext 参数
  # Normalize fd1/chirp_ext for ionosphere estimation.
  if (-f $master.PRM) then
    sed "s/.*fd1.*/fd1 = 0.0000/g" $master.PRM > tmp
    sed "s/.*chirp_ext.*/chirp_ext = 0/g" tmp > tmp2
    mv tmp2 $master.PRM
    rm -f tmp
  endif
  if (-f $aligned.PRM) then
    sed "s/.*fd1.*/fd1 = 0.0000/g" $aligned.PRM > tmp
    sed "s/.*chirp_ext.*/chirp_ext = 0/g" tmp > tmp2
    mv tmp2 $aligned.PRM
    rm -f tmp
  endif
endif

if ($is_raw_sat == 1) then
  if ($skip_master == 0 || $skip_master == 2) then
    sarp.csh $master.PRM
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    sarp.csh $aligned.PRM
  endif
endif

if ($separate_focus == 1) then
  if ($is_raw_sat == 1) then
    echo ""
    echo "L0->L1 聚焦完成（separate_focus=1）/ L0->L1 focus completed (separate_focus=1)"
    echo "已按配置跳过配准；将 separate_focus=0 后可继续 Stage-2 / Alignment skipped; set separate_focus=0 to continue Stage-2"
    echo ""
    exit 10
  else
    echo "separate_focus=1 仅对 L0 原始数据卫星生效（ERS/ENVI/ALOS/CSK_RAW），当前继续执行配准 / separate_focus=1 is only for L0 raw satellites; continue alignment."
  endif
endif

exit 0
