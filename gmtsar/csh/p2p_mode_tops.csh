#!/bin/csh -f
# 通用 TOPS 模式入口 / Common tops-mode entry.

if ($#argv != 3 && $#argv != 4) then
  echo ""
  echo "用法 / Usage: p2p_mode_tops.csh SAT master aligned [config]"
  echo "说明 / Note: 当前仅支持 SAT=S1_TOPS"
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

if ($SAT != "S1_TOPS") then
  echo "tops 模式当前仅支持 SAT=S1_TOPS / tops mode currently supports only SAT=S1_TOPS"
  exit 1
endif

if ($#argv == 4) then
  csh -f $script_dir/p2p_processing.csh $1 $2 $3 $4
else
  csh -f $script_dir/p2p_processing.csh $1 $2 $3
endif

exit $status
