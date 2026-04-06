#!/bin/csh -f
#
#  Eric Lindsey, July 2022
#
# Script to create DEM for GMTSAR, relative to WGS84 ellipsoid
#
  if ($#argv != 4 && $#argv != 5 && $#argv != 6) then
    echo ""
    echo "Usage: make_dem.csh W E S N [mode] [hgt_dir]"
    echo "      Uses GMT server to download SRTM 1-arcsec data (@earth_relief_xxs)"
    echo "      and removes the EGM96 geoid to make heights relative to WGS84."
    echo ""
    echo "      mode 1:SRTM-1s 2:SRTM-3s 3:earth_relief_15s"
    echo "      hgt_dir (optional): local directory for HGT/HGT.ZIP tiles (mode=1)"
    echo ""
    echo "Example: make_dem.csh -115 -112 32 35 2"
    echo ""
    exit 1
  endif
#
  echo ""
  echo "START: make_dem.csh"
  echo ""

  if ($#argv >= 5) then
    set mode = $5
  else
    set mode = 1
  endif

  set hgt_dir = ""
  if ($#argv == 6) then
    set hgt_dir = "$6"
  endif

#
# get region in GMT format
#
  set R = "-R$1/$2/$3/$4"
#
# need to set this for the distribution
#
  set sharedir = `gmtsar_sharedir.csh`
#
# get srtm data
#
  if ($mode == 1) then
    if ("$hgt_dir" != "") then
      echo "make_dem.csh mode=1: use local HGT directory $hgt_dir"
      make_dem_from_hgt.sh $1 $2 $3 $4 "$hgt_dir"
    else
      echo "make_dem.csh mode=1: no HGT directory specified, download HGT from ESA"
      make_dem_from_hgt.sh $1 $2 $3 $4
    endif
    if ($status != 0 || ! -f dem_ortho.grd) then
      echo "ERROR: failed to build dem_ortho.grd from HGT tiles"
      exit 1
    endif
  else if ($mode == 2) then
    gmt grdcut @earth_relief_03s $R -Gdem_ortho.grd 
  else 
    set local_relief = ""
    if (-f "$sharedir/earth_relief_15s.grd") then
      set local_relief = "$sharedir/earth_relief_15s.grd"
    else if (-f "$sharedir/earth_relief_15s_host_test.grd") then
      set local_relief = "$sharedir/earth_relief_15s_host_test.grd"
    else
    endif

    if ("$local_relief" != "") then
      set local_ok = `gmt grdinfo -C "$local_relief" | awk -v W=$1 -v E=$2 -v S=$3 -v N=$4 '{if (W >= $2 && E <= $3 && S >= $4 && N <= $5) print 1; else print 0}'`
      if ("$local_ok" == "1") then
        gmt grdcut "$local_relief" $R -Gdem_ortho.grd
      else
        gmt grdcut @earth_relief_15s $R -Gdem_ortho.grd
      endif
    else
      gmt grdcut @earth_relief_15s $R -Gdem_ortho.grd
    endif
  endif
#
# resample and remove geoid
#
  gmt grdsample $sharedir/geoid_egm96_icgem.grd -Rdem_ortho.grd -Ggeoid_resamp.grd # -Vq
  # gmt grdmath -Vq dem_ortho.grd geoid_resamp.grd ADD = dem.grd # 原来用的加法，我试一试减法
  
  gmt grdmath -Vq dem_ortho.grd geoid_resamp.grd SUB = dem.grd # 确认残余干涉条纹与DEM无关。
#
# clean up 
#
  rm geoid_resamp.grd
#
  echo ""
  echo "created dem.grd, heights relative to WGS84 ellipsoid"
  echo ""
  echo "END: make_dem.csh"
  echo ""
