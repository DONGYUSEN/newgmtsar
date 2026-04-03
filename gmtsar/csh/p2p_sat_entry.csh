#!/bin/csh -f
# 卫星独立入口统一路由 / Unified satellite entry router.

if ($#argv != 4 && $#argv != 5) then
  echo ""
  echo "用法 / Usage: p2p_sat_entry.csh SAT mode master aligned [config]"
  echo ""
  echo "模式 / Mode: strip | tops"
  echo "说明 / Note: mode=strip 用于条带模式, mode=tops 用于 TOPS 模式"
  echo ""
  exit 1
endif

set SAT = $1
set mode = $2
set master = $3
set aligned = $4

set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

if ($mode != "strip" && $mode != "tops") then
  echo "不支持的模式 / Unsupported mode: $mode"
  exit 1
endif

if ($mode == "tops" && $SAT != "S1_TOPS") then
  echo "tops 模式当前仅支持 SAT=S1_TOPS / tops mode currently supports only SAT=S1_TOPS"
  exit 1
endif

if ($mode == "strip") then
  if ($#argv == 5) then
    csh -f $script_dir/p2p_mode_strip.csh $SAT $master $aligned $5
  else
    csh -f $script_dir/p2p_mode_strip.csh $SAT $master $aligned
  endif
else if ($mode == "tops") then
  if ($#argv == 5) then
    csh -f $script_dir/p2p_mode_tops.csh $SAT $master $aligned $5
  else
    csh -f $script_dir/p2p_mode_tops.csh $SAT $master $aligned
  endif
endif

exit $status
