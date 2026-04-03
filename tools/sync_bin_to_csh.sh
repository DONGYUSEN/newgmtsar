#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/sync_bin_to_csh.sh [--check] [--dry-run]

Options:
  --check    Only check consistency; do not copy files.
  --dry-run  Show what would be copied.
  -h, --help Show this help message.
USAGE
}

check_only=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      check_only=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="$repo_root/bin"
dst_dir="$repo_root/gmtsar/csh"

if [[ ! -d "$src_dir" || ! -d "$dst_dir" ]]; then
  echo "Missing source or destination directory." >&2
  exit 1
fi

shopt -s nullglob
src_files=("$src_dir"/*.csh)
shopt -u nullglob

if [[ ${#src_files[@]} -eq 0 ]]; then
  echo "No .csh files found under $src_dir" >&2
  exit 1
fi

same_count=0
diff_count=0
missing_count=0
copied_count=0

for src_file in "${src_files[@]}"; do
  base_name="$(basename "$src_file")"
  dst_file="$dst_dir/$base_name"

  if [[ ! -f "$dst_file" ]]; then
    ((missing_count+=1))
    echo "[MISSING] $base_name"
    if [[ $check_only -eq 0 ]]; then
      if [[ $dry_run -eq 1 ]]; then
        echo "  -> would copy $src_file -> $dst_file"
      else
        cp -f "$src_file" "$dst_file"
        chmod --reference="$src_file" "$dst_file"
        ((copied_count+=1))
      fi
    fi
    continue
  fi

  if cmp -s "$src_file" "$dst_file"; then
    ((same_count+=1))
  else
    ((diff_count+=1))
    echo "[DIFF] $base_name"
    if [[ $check_only -eq 0 ]]; then
      if [[ $dry_run -eq 1 ]]; then
        echo "  -> would copy $src_file -> $dst_file"
      else
        cp -f "$src_file" "$dst_file"
        chmod --reference="$src_file" "$dst_file"
        ((copied_count+=1))
      fi
    fi
  fi
done

echo "Summary: same=$same_count diff=$diff_count missing=$missing_count copied=$copied_count"

if [[ $check_only -eq 1 && ( $diff_count -gt 0 || $missing_count -gt 0 ) ]]; then
  exit 2
fi

exit 0
