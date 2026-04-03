#!/bin/csh -f
# 卫星独立入口（条带模式）/ Satellite-specific entry (strip mode): RS2

set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

csh -f $script_dir/p2p_sat_entry.csh RS2 strip $argv
exit $status
