#!/bin/csh -f
# 卫星独立入口（TOPS模式）/ Satellite-specific entry (tops mode): S1_TOPS

set script_dir = $0:h
if ("$script_dir" == "$0") set script_dir = "."

csh -f $script_dir/p2p_sat_entry.csh S1_TOPS tops $argv
exit $status
