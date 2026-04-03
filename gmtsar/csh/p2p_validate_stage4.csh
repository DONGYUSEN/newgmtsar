#!/bin/csh -f
# Stage-4 输入校验器 / Stage-4 input validator.

if ($#argv != 6) then
  echo ""
  echo "用法 / Usage: p2p_validate_stage4.csh SAT ref rep topo_phase shift_topo iono"
  echo "参数说明 / Args: topo_phase=0|1, shift_topo=0|1, iono=0|1"
  echo ""
  exit 1
endif

set SAT = $1
set ref = $2
set rep = $3
set topo_phase = $4
set shift_topo = $5
set iono = $6

if ("$topo_phase" != "0" && "$topo_phase" != "1") then
  echo "错误参数：topo_phase=$topo_phase / Invalid parameter: topo_phase=$topo_phase"
  exit 1
endif

if ("$iono" != "0" && "$iono" != "1") then
  echo "错误参数：iono=$iono / Invalid parameter: iono=$iono"
  exit 1
endif

if (! -f SLC/$ref.PRM || ! -f SLC/$ref.SLC || ! -f SLC/$ref.LED) then
  echo "缺少阶段4输入文件 / Missing stage-4 input: SLC/$ref.PRM/.SLC/.LED"
  exit 1
endif

if (! -f SLC/$rep.PRM || ! -f SLC/$rep.SLC || ! -f SLC/$rep.LED) then
  echo "缺少阶段4输入文件 / Missing stage-4 input: SLC/$rep.PRM/.SLC/.LED"
  exit 1
endif

if (! -f raw/$ref.PRM || ! -f raw/$rep.PRM) then
  echo "缺少阶段4输入文件 / Missing stage-4 input: raw/$ref.PRM or raw/$rep.PRM"
  exit 1
endif

if ("$topo_phase" == "1") then
  if ("$shift_topo" != "0" && "$shift_topo" != "1") then
    echo "错误参数：shift_topo=$shift_topo / Invalid parameter: shift_topo=$shift_topo"
    exit 1
  endif

  if ("$shift_topo" == "1") then
    if (! -f topo/topo_shift.grd) then
      echo "缺少阶段4输入文件 / Missing stage-4 input: topo/topo_shift.grd"
      exit 1
    endif
  else
    if (! -f topo/topo_ra.grd) then
      echo "缺少阶段4输入文件 / Missing stage-4 input: topo/topo_ra.grd"
      exit 1
    endif
  endif
endif

if ("$iono" == "1") then
  if (! -f SLC_L/$ref.PRM || ! -f SLC_L/$ref.SLC || ! -f SLC_L/$ref.LED) then
    echo "缺少 iono 输入文件 / Missing iono input: SLC_L/$ref.PRM/.SLC/.LED"
    exit 1
  endif
  if (! -f SLC_L/$rep.PRM || ! -f SLC_L/$rep.SLC || ! -f SLC_L/$rep.LED) then
    echo "缺少 iono 输入文件 / Missing iono input: SLC_L/$rep.PRM/.SLC/.LED"
    exit 1
  endif
  if (! -f SLC_H/$ref.PRM || ! -f SLC_H/$ref.SLC || ! -f SLC_H/$ref.LED) then
    echo "缺少 iono 输入文件 / Missing iono input: SLC_H/$ref.PRM/.SLC/.LED"
    exit 1
  endif
  if (! -f SLC_H/$rep.PRM || ! -f SLC_H/$rep.SLC || ! -f SLC_H/$rep.LED) then
    echo "缺少 iono 输入文件 / Missing iono input: SLC_H/$rep.PRM/.SLC/.LED"
    exit 1
  endif
endif

exit 0
