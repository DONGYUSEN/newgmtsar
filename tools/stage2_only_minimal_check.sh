#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/stage2_only_minimal_check.sh SAT MASTER ALIGNED CONFIG [WORKDIR]

Description:
  Stage-2 only minimal regression precheck.
  It checks config validity and required file completeness only.
  No processing commands are executed.

Supported SAT:
  RS2, DJ1, S1_TOPS
USAGE
}

if [[ $# -ne 4 && $# -ne 5 ]]; then
  usage
  exit 1
fi

sat="$1"
master="$2"
aligned="$3"
config="$4"
workdir="${5:-$PWD}"

if [[ ! -d "$workdir" ]]; then
  echo "ERROR: workdir not found: $workdir"
  exit 2
fi

raw_dir="$workdir/raw"
topo_dir="$workdir/topo"

if [[ ! -f "$config" ]]; then
  echo "ERROR: config not found: $config"
  exit 2
fi

if [[ ! -d "$raw_dir" ]]; then
  echo "ERROR: raw directory not found: $raw_dir"
  exit 2
fi

if [[ ! -d "$topo_dir" ]]; then
  echo "ERROR: topo directory not found: $topo_dir"
  exit 2
fi

if [[ "$sat" != "RS2" && "$sat" != "DJ1" && "$sat" != "S1_TOPS" ]]; then
  echo "ERROR: unsupported SAT: $sat"
  echo "Supported SAT: RS2, DJ1, S1_TOPS"
  exit 2
fi

get_cfg_value() {
  local key="$1"
  awk -v k="$key" '
    $0 !~ /^[[:space:]]*#/ && $1 == k && $2 == "=" { print $3; exit }
  ' "$config"
}

contains_stage() {
  local list="$1"
  local target="$2"
  [[ ",${list}," == *",${target},"* ]]
}

scene_to_tops_l1_id() {
  local scene="$1"
  echo "$scene" | awk '{ print "S1_"substr($1,16,8)"_"substr($1,25,6)"_F"substr($1,7,1)}'
}

proc_stage="$(get_cfg_value proc_stage)"
skip_stage="$(get_cfg_value skip_stage)"
data_level="$(get_cfg_value data_level)"
skip_master="$(get_cfg_value skip_master)"
separate_focus="$(get_cfg_value separate_focus)"
correct_iono="$(get_cfg_value correct_iono)"

[[ -z "$proc_stage" ]] && proc_stage="1"
[[ -z "$skip_stage" ]] && skip_stage=""
[[ -z "$data_level" ]] && data_level="0"
[[ -z "$skip_master" ]] && skip_master="0"
[[ -z "$separate_focus" ]] && separate_focus="0"
[[ -z "$correct_iono" ]] && correct_iono="0"

cfg_errors=0
echo "[Config] proc_stage=$proc_stage skip_stage=$skip_stage data_level=$data_level skip_master=$skip_master separate_focus=$separate_focus correct_iono=$correct_iono"

if [[ "$proc_stage" != "2" ]]; then
  echo "ERROR: proc_stage must be 2 for stage-2 only minimal regression."
  cfg_errors=1
fi

if ! contains_stage "$skip_stage" "1"; then
  echo "ERROR: skip_stage must include 1 for stage-2 only minimal regression."
  cfg_errors=1
fi

if [[ "$data_level" != "1" ]]; then
  echo "ERROR: data_level must be 1 for current L1-first workflow."
  cfg_errors=1
fi

if [[ "$skip_master" != "0" && "$skip_master" != "1" && "$skip_master" != "2" ]]; then
  echo "ERROR: skip_master must be one of 0,1,2."
  cfg_errors=1
fi

if [[ "$separate_focus" == "1" ]]; then
  echo "ERROR: separate_focus must be 0 in stage-2 L1 mode."
  cfg_errors=1
fi

if [[ "$cfg_errors" -ne 0 ]]; then
  exit 2
fi

require_master=0
require_aligned=0
if [[ "$skip_master" == "0" || "$skip_master" == "2" ]]; then
  require_master=1
fi
if [[ "$skip_master" == "0" || "$skip_master" == "1" ]]; then
  require_aligned=1
fi

missing=0
checked=0

check_file() {
  local f="$1"
  ((checked+=1))
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    ((missing+=1))
  fi
}

if [[ "$sat" == "RS2" || "$sat" == "DJ1" ]]; then
  if [[ "$require_master" -eq 1 ]]; then
    check_file "$raw_dir/$master.PRM"
    check_file "$raw_dir/$master.SLC"
    check_file "$raw_dir/$master.LED"
  fi
  if [[ "$require_aligned" -eq 1 ]]; then
    check_file "$raw_dir/$aligned.PRM"
    check_file "$raw_dir/$aligned.SLC"
    check_file "$raw_dir/$aligned.LED"
  fi
fi

if [[ "$sat" == "S1_TOPS" ]]; then
  master_id="$(scene_to_tops_l1_id "$master")"
  aligned_id="$(scene_to_tops_l1_id "$aligned")"
  echo "[S1_TOPS] master scene -> $master_id"
  echo "[S1_TOPS] aligned scene -> $aligned_id"

  if [[ "$require_master" -eq 1 ]]; then
    check_file "$raw_dir/$master_id.PRM"
    check_file "$raw_dir/$master_id.SLC"
    check_file "$raw_dir/$master_id.LED"
  fi
  if [[ "$require_aligned" -eq 1 ]]; then
    check_file "$raw_dir/$aligned_id.PRM"
    check_file "$raw_dir/$aligned_id.SLC"
    check_file "$raw_dir/$aligned_id.LED"
  fi

  if [[ "$correct_iono" == "1" ]]; then
    check_file "$topo_dir/dem.grd"
    if [[ "$require_master" -eq 1 ]]; then
      check_file "$raw_dir/$master.tiff"
      check_file "$raw_dir/$master.xml"
      check_file "$raw_dir/$master.EOF"
    fi
    if [[ "$require_aligned" -eq 1 ]]; then
      check_file "$raw_dir/$aligned.tiff"
      check_file "$raw_dir/$aligned.xml"
      check_file "$raw_dir/$aligned.EOF"
      check_file "$raw_dir/a.grd"
      check_file "$raw_dir/r.grd"
    fi
  fi
fi

if [[ "$missing" -ne 0 ]]; then
  echo "FAILED: checked=$checked missing=$missing"
  exit 2
fi

echo "PASS: checked=$checked missing=0"
exit 0
