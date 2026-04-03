#!/bin/csh -f
# Stage-4 卫星插件分发器 / Stage-4 hook dispatcher for satellite plugins.

if ($#argv != 7) then
  echo ""
  echo "用法 / Usage: p2p_hook_stage4.csh SAT when ref rep conf topo_phase iono"
  echo "参数说明 / Args: when=pre|post"
  echo ""
  exit 1
endif

set SAT = $1
set when = $2
set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

if ($when != "pre" && $when != "post") then
  echo "无效的 Stage-4 Hook 阶段 / Invalid stage-4 hook phase: $when"
  exit 1
endif

set hook_impl = "p2p_hook_stage4_"$SAT".csh"
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
