#!/bin/csh -f
# 卫星独立入口（条带模式）/ Satellite-specific entry (strip mode): ALOS

set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

csh -f $script_dir/p2p_sat_entry.csh ALOS strip $argv
exit $status
