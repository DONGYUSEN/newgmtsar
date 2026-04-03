#!/bin/csh -f
# Stage-3 卫星插件分发器 / Stage-3 hook dispatcher for satellite plugins.

if ($#argv != 7) then
  echo ""
  echo "用法 / Usage: p2p_hook_stage3.csh SAT when master aligned conf topo_phase shift_topo"
  echo "参数说明 / Args: when=pre|post"
  echo ""
  exit 1
endif

set SAT = $1
set when = $2
set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

if ($when != "pre" && $when != "post") then
  echo "无效的 Stage-3 Hook 阶段 / Invalid stage-3 hook phase: $when"
  exit 1
endif

set hook_impl = "p2p_hook_stage3_"$SAT".csh"
if (-f "$script_dir/$hook_impl") then
  csh -f "$script_dir/$hook_impl" $argv
  exit $status
endif

which $hook_impl >& /dev/null
if ($status == 0) then
  $hook_impl $argv
  exit $status
endif

exit 0
