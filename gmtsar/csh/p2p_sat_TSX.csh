#!/bin/csh -f
# 卫星独立入口（条带模式）/ Satellite-specific entry (strip mode): TSX

set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

csh -f $script_dir/p2p_sat_entry.csh TSX strip $argv
exit $status
