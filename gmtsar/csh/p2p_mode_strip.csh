#!/bin/csh -f
# 通用 strip 模式入口 / Common strip-mode entry.

set supported_strip_sat = (ERS ENVI ALOS ALOS_SLC ALOS2 ALOS2_SCAN S1_STRIP ENVI_SLC CSK_RAW CSK_SLC CSG TSX RS2 GF3 LT1 DJ1)

if ($#argv < 3 || $#argv > 5) then
  echo ""
  echo "用法 / Usage: p2p_mode_strip.csh SAT master aligned [config] [rg:za]"
  echo "            : p2p_mode_strip.csh SAT master aligned [rg:za]"
  echo "说明 / Note: 该入口用于 strip 类卫星 (非 S1_TOPS)"
  echo "支持的 strip 卫星 / Supported strip SAT: $supported_strip_sat"
  echo ""
  exit 1
endif

set SAT = $1
set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."
if ($?PATH) then
  setenv PATH "$script_dir":"$PATH"
else
  setenv PATH "$script_dir"
endif

if ($SAT == "S1_TOPS") then
  echo "SAT=S1_TOPS 应使用 p2p_mode_tops.csh / SAT=S1_TOPS should use p2p_mode_tops.csh"
  exit 1
endif

set sat_ok = 0
foreach sat_name ($supported_strip_sat)
  if ("$SAT" == "$sat_name") then
    set sat_ok = 1
  endif
end
if ($sat_ok == 0) then
  echo "不支持的 strip 卫星 / Unsupported strip SAT: $SAT"
  echo "支持的 strip 卫星 / Supported strip SAT: $supported_strip_sat"
  exit 1
endif

if (! -f "$script_dir/p2p_processing.csh") then
  echo "未找到核心处理脚本 / Core processing script not found: $script_dir/p2p_processing.csh"
  exit 1
endif

if ($#argv == 5) then
  csh -f $script_dir/p2p_processing.csh $1 $2 $3 $4 $5
else if ($#argv == 4) then
  csh -f $script_dir/p2p_processing.csh $1 $2 $3 $4
else
  csh -f $script_dir/p2p_processing.csh $1 $2 $3
endif

exit $status
