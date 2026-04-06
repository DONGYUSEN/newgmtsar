#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo ""
  echo "Usage: make_dem_from_hgt.sh W E S N [hgt_dir]"
  echo "  If hgt_dir is provided, use local HGT/HGT.ZIP files in that directory."
  echo "  If hgt_dir is omitted, download SRTMGL1 HGT ZIP tiles from ESA and mosaic."
  echo ""
  exit 1
fi

W="$1"
E="$2"
S="$3"
N="$4"
HGT_DIR="${5:-}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1" >&2; exit 1; }
}

need_cmd awk
need_cmd gdalbuildvrt
need_cmd gdalwarp
need_cmd gdal_translate
need_cmd wget

floor_int() {
  awk -v v="$1" 'BEGIN{iv=int(v); if (v < 0 && v != iv) iv=iv-1; print iv}'
}

ceil_int() {
  awk -v v="$1" 'BEGIN{iv=int(v); if (v > 0 && v != iv) iv=iv+1; print iv}'
}

lon_start="$(floor_int "$W")"
lon_end_excl="$(ceil_int "$E")"
lat_start="$(floor_int "$S")"
lat_end_excl="$(ceil_int "$N")"

if [[ "$lon_start" -ge "$lon_end_excl" || "$lat_start" -ge "$lat_end_excl" ]]; then
  echo "ERROR: invalid region W/E/S/N = $W $E $S $N" >&2
  exit 1
fi

tmp_dir="$(mktemp -d /tmp/gmtsar_hgt_XXXXXX)"
trap 'rm -rf "$tmp_dir"' EXIT

cache_dir="${HOME:-/tmp}/.gmtsar_hgt_cache"
mkdir -p "$cache_dir"

declare -a tile_sources=()
missing_local=0

for ((lat=lat_start; lat<lat_end_excl; lat++)); do
  if ((lat >= 0)); then
    lat_tag="$(printf "N%02d" "$lat")"
  else
    lat_tag="$(printf "S%02d" "$((-lat))")"
  fi

  for ((lon=lon_start; lon<lon_end_excl; lon++)); do
    if ((lon >= 0)); then
      lon_tag="$(printf "E%03d" "$lon")"
    else
      lon_tag="$(printf "W%03d" "$((-lon))")"
    fi
    tile="${lat_tag}${lon_tag}"

    if [[ -n "$HGT_DIR" ]]; then
      if [[ -f "$HGT_DIR/$tile.hgt" ]]; then
        tile_sources+=("$HGT_DIR/$tile.hgt")
        continue
      fi
      if [[ -f "$HGT_DIR/$tile.hgt.zip" ]]; then
        tile_sources+=("/vsizip/$HGT_DIR/$tile.hgt.zip/$tile.hgt")
        continue
      fi
      if [[ -f "$HGT_DIR/$tile.SRTMGL1.hgt.zip" ]]; then
        tile_sources+=("/vsizip/$HGT_DIR/$tile.SRTMGL1.hgt.zip/$tile.hgt")
        continue
      fi
      echo "ERROR: missing local HGT tile for $tile in $HGT_DIR" >&2
      missing_local=1
      continue
    fi

    zip_file="$cache_dir/$tile.SRTMGL1.hgt.zip"
    if [[ ! -s "$zip_file" ]]; then
      url="https://step.esa.int/auxdata/dem/SRTMGL1/$tile.SRTMGL1.hgt.zip"
      echo "Downloading $tile from ESA: $url"
      if ! wget -q -O "$zip_file" "$url"; then
        rm -f "$zip_file"
        echo "ERROR: failed to download $url" >&2
        exit 1
      fi
    fi
    tile_sources+=("/vsizip/$zip_file/$tile.hgt")
  done
done

if [[ "$missing_local" -ne 0 ]]; then
  exit 1
fi

if [[ "${#tile_sources[@]}" -eq 0 ]]; then
  echo "ERROR: no HGT tiles resolved" >&2
  exit 1
fi

gdalbuildvrt "$tmp_dir/dem.vrt" "${tile_sources[@]}"
gdalwarp -overwrite -te "$W" "$S" "$E" "$N" -te_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear "$tmp_dir/dem.vrt" "$tmp_dir/dem_clip.tif"
gdal_translate -of netCDF "$tmp_dir/dem_clip.tif" dem_ortho.grd
