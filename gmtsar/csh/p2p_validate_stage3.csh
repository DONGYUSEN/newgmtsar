#!/bin/csh -f
# Stage-3 输入校验器 / Stage-3 input validator.

if ($#argv != 5) then
  echo ""
  echo "用法 / Usage: p2p_validate_stage3.csh SAT master aligned topo_phase shift_topo"
  echo "参数说明 / Args: topo_phase=0|1, shift_topo=0|1(仅 topo_phase=1 时生效)"
  echo ""
  exit 1
endif

set SAT = $1
set master = $2
set aligned = $3
set topo_phase = $4
set shift_topo = $5

if ("$topo_phase" != "0" && "$topo_phase" != "1") then
  echo "错误参数：topo_phase=$topo_phase / Invalid parameter: topo_phase=$topo_phase"
  exit 1
endif

if (! -f SLC/$master.PRM) then
  echo "缺少阶段3输入文件 / Missing stage-3 input: SLC/$master.PRM"
  exit 1
endif

if (! -f raw/$master.LED) then
  echo "缺少阶段3输入文件 / Missing stage-3 input: raw/$master.LED"
  exit 1
endif

if ("$topo_phase" == "1") then
  if ("$shift_topo" != "0" && "$shift_topo" != "1") then
    echo "错误参数：shift_topo=$shift_topo / Invalid parameter: shift_topo=$shift_topo"
    exit 1
  endif

  if (! -f topo/dem.grd) then
    echo "缺少 DEM 文件 / DEM file not found: topo/dem.grd"
    exit 1
  endif

  if (! -f SLC/$aligned.PRM) then
    echo "缺少阶段3输入文件 / Missing stage-3 input: SLC/$aligned.PRM"
    exit 1
  endif
endif

exit 0
