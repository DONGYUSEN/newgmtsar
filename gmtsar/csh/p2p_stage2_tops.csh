#!/bin/csh -f
#
# Stage 2 helper for S1_TOPS in p2p_processing.csh
# Must be run from SLC/ directory.
#

if ($#argv != 6) then
  echo ""
  echo "Usage: p2p_stage2_tops.csh master_prm aligned_prm master_scene aligned_scene skip_master iono"
  echo ""
  exit 1
endif

set master = $1
set aligned = $2
set master_scene = $3
set aligned_scene = $4
set skip_master = $5
set iono = $6

if ($skip_master == 0 || $skip_master == 2) then
  if (! -f ../raw/$master.PRM || ! -f ../raw/$master.SLC || ! -f ../raw/$master.LED) then
    echo "Missing TOPS L1 files for master in raw/: "$master".PRM/.SLC/.LED"
    exit 2
  endif
  cp ../raw/$master.PRM .
  ln -s ../raw/$master.SLC .
  ln -s ../raw/$master.LED .
endif

if ($skip_master == 0 || $skip_master == 1) then
  if (! -f ../raw/$aligned.PRM || ! -f ../raw/$aligned.SLC || ! -f ../raw/$aligned.LED) then
    echo "Missing TOPS L1 files for aligned in raw/: "$aligned".PRM/.SLC/.LED"
    exit 2
  endif
  cp ../raw/$aligned.PRM .
  ln -s ../raw/$aligned.SLC .
  ln -s ../raw/$aligned.LED .
endif

if ($iono == 1) then
  if (! -f ../topo/dem.grd) then
    echo "Missing ../topo/dem.grd for TOPS iono processing"
    exit 2
  endif
  if ($skip_master == 0 || $skip_master == 2) then
    if (! -f ../raw/$master_scene.tiff || ! -f ../raw/$master_scene.xml || ! -f ../raw/$master_scene.EOF) then
      echo "Missing TOPS files for master scene in raw/: "$master_scene".tiff/.xml/.EOF"
      exit 2
    endif
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    if (! -f ../raw/$aligned_scene.tiff || ! -f ../raw/$aligned_scene.xml || ! -f ../raw/$aligned_scene.EOF) then
      echo "Missing TOPS files for aligned scene in raw/: "$aligned_scene".tiff/.xml/.EOF"
      exit 2
    endif
  endif

  if ($skip_master == 0 || $skip_master == 2) then
    ln -s ../raw/$master_scene.tiff .
    split_spectrum $master.PRM > params1
    mv high.tiff ../SLC_H/$master_scene.tiff
    mv low.tiff ../SLC_L/$master_scene.tiff
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    ln -s ../raw/$aligned_scene.tiff .
    split_spectrum $aligned.PRM > params2
    mv high.tiff ../SLC_H/$aligned_scene.tiff
    mv low.tiff ../SLC_L/$aligned_scene.tiff
  endif

  cd ../SLC_L
  if ($skip_master == 0 || $skip_master == 2) then
    ln -s ../raw/$master_scene.xml .
    ln -s ../raw/$master_scene.EOF .
    ln -s ../topo/dem.grd .
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    ln -s ../raw/$aligned_scene.xml .
    ln -s ../raw/$aligned_scene.EOF .
    ln -s ../raw/a.grd .
    ln -s ../raw/r.grd .
    ls ../raw/offset*.dat >& /dev/null
    if ($status == 0) then
      ln -s ../raw/offset*.dat .
    endif
  endif

  if ($skip_master == 0) then
    align_tops.csh $master_scene $master_scene.EOF $aligned_scene $aligned_scene.EOF dem.grd 1
  else if ($skip_master == 1) then
    align_tops.csh $master_scene 0 $aligned_scene $aligned_scene.EOF dem.grd 1
  else if ($skip_master == 2) then
    align_tops.csh $master_scene $master_scene.EOF $aligned_scene 0 dem.grd 1
  endif

  if ($skip_master == 0 || $skip_master == 2) then
    set wl1 = `grep low_wavelength ../SLC/params1 | awk '{print $3}'`
    sed "s/.*wavelength.*/radar_wavelength    = $wl1/g" $master.PRM > tmp
    mv tmp $master.PRM
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    set wl2 = `grep low_wavelength ../SLC/params2 | awk '{print $3}'`
    sed "s/.*wavelength.*/radar_wavelength    = $wl2/g" $aligned.PRM > tmp
    mv tmp $aligned.PRM
  endif

  cd ../SLC_H
  if ($skip_master == 0 || $skip_master == 2) then
    ln -s ../raw/$master_scene.xml .
    ln -s ../raw/$master_scene.EOF .
    ln -s ../topo/dem.grd .
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    ln -s ../raw/$aligned_scene.xml .
    ln -s ../raw/$aligned_scene.EOF .
    ln -s ../raw/a.grd .
    ln -s ../raw/r.grd .
    ls ../raw/offset*.dat >& /dev/null
    if ($status == 0) then
      ln -s ../raw/offset*.dat .
    endif
  endif

  if ($skip_master == 0) then
    align_tops.csh $master_scene $master_scene.EOF $aligned_scene $aligned_scene.EOF dem.grd 1
  else if ($skip_master == 1) then
    align_tops.csh $master_scene 0 $aligned_scene $aligned_scene.EOF dem.grd 1
  else if ($skip_master == 2) then
    align_tops.csh $master_scene $master_scene.EOF $aligned_scene 0 dem.grd 1
  endif

  if ($skip_master == 0 || $skip_master == 2) then
    set wh1 = `grep high_wavelength ../SLC/params1 | awk '{print $3}'`
    sed "s/.*wavelength.*/radar_wavelength    = $wh1/g" $master.PRM > tmp
    mv tmp $master.PRM
  endif
  if ($skip_master == 0 || $skip_master == 1) then
    set wh2 = `grep high_wavelength ../SLC/params2 | awk '{print $3}'`
    sed "s/.*wavelength.*/radar_wavelength    = $wh2/g" $aligned.PRM > tmp
    mv tmp $aligned.PRM
  endif

  cd ../SLC
endif

exit 0
