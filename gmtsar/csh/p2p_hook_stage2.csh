#!/bin/csh -f
# Stage-2 卫星插件分发器 / Stage-2 hook dispatcher for satellite plugins.
#
# pre 钩子返回码约定 / Status contract (pre hook):
#   0  -> 继续默认阶段逻辑 / continue default stage logic
#   10 -> 跳过默认阶段逻辑 / skip default stage logic
#   20 -> 插件已处理该阶段 / hook already handled stage logic
#  other -> 错误 / error

if ($#argv != 7) then
  echo ""
  echo "用法 / Usage: p2p_hook_stage2.csh SAT when master aligned conf data_level skip_master"
  echo "参数说明 / Args: when=pre|post"
  echo ""
  exit 1
endif

set SAT = $1
set when = $2
set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

if ($when != "pre" && $when != "post") then
  echo "无效的 Stage-2 Hook 阶段 / Invalid stage-2 hook phase: $when"
  exit 1
endif

set hook_impl = "p2p_hook_stage2_"$SAT".csh"
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
