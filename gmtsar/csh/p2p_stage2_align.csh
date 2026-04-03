#!/bin/csh -f
#
# p2p_processing.csh 的 Stage-2B（非 S1_TOPS 分支）
# Stage-2B for p2p_processing.csh (non-S1_TOPS branch)
# 功能：在聚焦/准备完成后执行配准与重采样
# Function: run alignment and resampling after focus/preparation
#

if ($#argv != 5) then
  echo ""
  echo "用法 / Usage: p2p_stage2_align.csh SAT master aligned skip_master iono"
  echo "参数说明 / Args: skip_master=0|1|2, iono=0|1"
  echo ""
  exit 1
endif

set SAT = $1
set master = $2
set aligned = $3
set skip_master = $4
set iono = $5

if ($skip_master != 0 && $skip_master != 1 && $skip_master != 2) then
  echo "错误参数：skip_master=$skip_master / Invalid parameter: skip_master=$skip_master"
  exit 1
endif

if ($iono != 0 && $iono != 1) then
  echo "错误参数：iono=$iono / Invalid parameter: iono=$iono"
  exit 1
endif

# skip_master=1 时，主影像应已在当前目录（通常来自前一次运行）
# For skip_master=1, master artifacts are expected to already exist (typically from a previous run).
if ($skip_master == 1) then
  if (! -f $master.PRM || ! -f $master.SLC || ! -f $master.LED) then
    echo "缺少 master 参考文件（skip_master=1 需要复用已有 master）/ Missing master reference files for skip_master=1: $master.PRM/.SLC/.LED"
    exit 2
  endif
  if ($iono == 1) then
    if (! -f ../SLC_L/$master.PRM || ! -f ../SLC_L/$master.SLC || ! -f ../SLC_L/$master.LED) then
      echo "缺少 SLC_L master 文件（skip_master=1, iono=1）/ Missing SLC_L master files for skip_master=1, iono=1: ../SLC_L/$master.PRM/.SLC/.LED"
      exit 2
    endif
    if (! -f ../SLC_H/$master.PRM || ! -f ../SLC_H/$master.SLC || ! -f ../SLC_H/$master.LED) then
      echo "缺少 SLC_H master 文件（skip_master=1, iono=1）/ Missing SLC_H master files for skip_master=1, iono=1: ../SLC_H/$master.PRM/.SLC/.LED"
      exit 2
    endif
  endif
endif

if ($iono == 1) then
  if ($skip_master == 0 || $skip_master == 2) then
    if (-f ../raw/ALOS_fbd2fbs_log_$aligned) then
      split_spectrum $master.PRM 1 > params1
    else
      split_spectrum $master.PRM > params1
    endif
    mv SLCH ../SLC_H/$master.SLC
    mv SLCL ../SLC_L/$master.SLC

    cd ../SLC_L
    set wl1 = `grep low_wavelength ../SLC/params1 | awk '{print $3}'`
    cp ../SLC/$master.PRM .
    rm -f $master.LED
    ln -s ../raw/$master.LED .
    sed "s/.*wavelength.*/radar_wavelength    = $wl1/g" $master.PRM > tmp
    mv tmp $master.PRM
    cd ../SLC_H
    set wh1 = `grep high_wavelength ../SLC/params1 | awk '{print $3}'`
    cp ../SLC/$master.PRM .
    rm -f $master.LED
    ln -s ../raw/$master.LED .
    sed "s/.*wavelength.*/radar_wavelength    = $wh1/g" $master.PRM > tmp
    mv tmp $master.PRM
    cd ../SLC
  endif

  if ($skip_master == 0 || $skip_master == 1) then
    if (-f ../raw/ALOS_fbd2fbs_log_$master) then
      split_spectrum $aligned.PRM 1 > params2
    else
      split_spectrum $aligned.PRM > params2
    endif
    mv SLCH ../SLC_H/$aligned.SLC
    mv SLCL ../SLC_L/$aligned.SLC

    cd ../SLC_L
    set wl2 = `grep low_wavelength ../SLC/params2 | awk '{print $3}'`
    cp ../SLC/$aligned.PRM .
    rm -f $aligned.LED
    ln -s ../raw/$aligned.LED .
    sed "s/.*wavelength.*/radar_wavelength    = $wl2/g" $aligned.PRM > tmp
    mv tmp $aligned.PRM
    cd ../SLC_H
    set wh2 = `grep high_wavelength ../SLC/params2 | awk '{print $3}'`
    cp ../SLC/$aligned.PRM .
    rm -f $aligned.LED
    ln -s ../raw/$aligned.LED .
    sed "s/.*wavelength.*/radar_wavelength    = $wh2/g" $aligned.PRM > tmp
    mv tmp $aligned.PRM
    cd ../SLC
  endif
endif

if ($skip_master == 0 || $skip_master == 1) then
  cp $aligned.PRM $aligned.PRM0
  SAT_baseline $master.PRM $aligned.PRM0 >> $aligned.PRM
  if ($SAT == "ALOS2_SCAN") then
    xcorr2 $master.PRM $aligned.PRM -xsearch 32 -ysearch 256 -nx 32 -ny 128
    awk '{print $4}' < freq_xcorr.dat > tmp.dat
    set amedian = `sort -n tmp.dat | awk ' { a[i++]=$1; } END { print a[int(i/2)]; }'`
    set amax = `echo $amedian | awk '{print $1+3}'`
    set amin = `echo $amedian | awk '{print $1-3}'`
    awk '{if($4 > '$amin' && $4 < '$amax') print $0}' < freq_xcorr.dat > freq_alos2.dat
    fitoffset.csh 2 3 freq_alos2.dat 10 >> $aligned.PRM
  else if ($SAT == "ERS" || $SAT == "ENVI" || $SAT == "ALOS" || $SAT == "CSK_RAW" || $SAT == "ALOS_SLC") then
    xcorr2 $master.PRM $aligned.PRM -xsearch 128 -ysearch 128 -nx 20 -ny 50
    fitoffset.csh 3 3 freq_xcorr.dat 18 >> $aligned.PRM
  else if ($SAT == "DJ1" ) then
    echo "                     ..配准 DJ1            "
    set OMP_NUM_THREADS = 10
    xcorr2 $master.PRM $aligned.PRM -xsearch 256 -ysearch 256 -nx 30 -ny 30 -noshift
    filter_offset.csh freq_xcorr.dat  output.txt
    mv output.txt freq_xcorr.dat
    fitoffset.csh 3 3 freq_xcorr.dat 40 >> $aligned.PRM
  else if ($SAT == "LT1") then
    echo "             ..配准 LT1(多尺度配准技术)            "
    echo "大窗口，粗配准，2048*2048, 获得总体的偏差：  "
    xcorr $master.PRM $aligned.PRM -nx 4 -ny 4 -nointerp -noshift -xsearch 2048 -ysearch 2048
    #filter_offset.csh freq_xcorr.dat  output.txt
    #mv output.txt freq_xcorr.dat
    #fitoffset.csh 1 1 freq_xcorr.dat
    fitoffset.csh 1 1 freq_xcorr.dat >> $aligned.PRM
    update_PRM $aligned.PRM SC_identity 12
  else
    xcorr2 $master.PRM $aligned.PRM -noshift -xsearch 128 -ysearch 128 -nx 20 -ny 50
    fitoffset.csh 2 2  freq_xcorr.dat 18 >> $aligned.PRM
  endif
  echo "                     ..重采样    "

  if ($SAT == "LT1") then
    #resamp $master.PRM $aligned.PRM $aligned.PRMresamp $aligned.SLCresamp 4
    #rm $aligned.SLC
    #mv $aligned.SLCresamp $aligned.SLC
    #cp $aligned.PRMresamp $aligned.PRM
    echo "Loop 2"
    xcorr2 $master.PRM $aligned.PRM -nx 60 -ny 60  -xsearch 256 -ysearch 256 
    filter_offset.csh freq_xcorr.dat  output.txt
    mv output.txt freq_xcorr.dat
    fitoffset.csh 2 2 freq_xcorr.dat
    fitoffset.csh 2 2 freq_xcorr.dat 40 >> $aligned.PRM
    resamp $master.PRM $aligned.PRM $aligned.PRMresamp $aligned.SLCresamp 4
    #rm $aligned.SLC
    #mv $aligned.SLCresamp $aligned.SLC
    #cp $aligned.PRMresamp $aligned.PRM

    echo "Final check!  "
    #xcorr2 $master.PRM $aligned.PRM -nx 1200 -ny 1  -xsearch 128 -ysearch 256

    # keep original behavior: LT1 branch exits early in current implementation
    #exit 20
  else
    resamp $master.PRM $aligned.PRM $aligned.PRMresamp $aligned.SLCresamp 4
  endif
  rm $aligned.SLC
  mv $aligned.SLCresamp $aligned.SLC
  cp $aligned.PRMresamp $aligned.PRM

  echo ".....完成重采样"
  slc2amp.csh $master.PRM 2 master.grd
  slc2amp.csh $aligned.PRM 2 slave.grd
  rm master.grd slave.grd

  if ($iono == 1) then
    cd ../SLC_L
    cp $aligned.PRM $aligned.PRM0
    if ($SAT == "ALOS2_SCAN") then
      ln -s ../SLC/freq_alos2.dat
      fitoffset.csh  2 3 freq_alos2.dat 10 >> $aligned.PRM
    else if ($SAT == "ERS" || $SAT == "ENVI" || $SAT == "ALOS" || $SAT == "CSK_RAW" || $SAT == "TSX") then
      ln -s ../SLC/freq_xcorr.dat .
      fitoffset.csh 3 3 freq_xcorr.dat 18 >> $aligned.PRM
    else
      ln -s ../SLC/freq_xcorr.dat .
      fitoffset.csh 2 2 freq_xcorr.dat 18 >> $aligned.PRM
    endif
    resamp_omp $master.PRM $aligned.PRM $aligned.PRMresamp $aligned.SLCresamp 4
    rm $aligned.SLC
    mv $aligned.SLCresamp $aligned.SLC
    cp $aligned.PRMresamp $aligned.PRM

    cd ../SLC_H
    cp $aligned.PRM $aligned.PRM0
    if ($SAT == "ALOS2_SCAN") then
      ln -s ../SLC/freq_alos2.dat
      fitoffset.csh  2 3 freq_alos2.dat 10 >> $aligned.PRM
    else if ($SAT == "ERS" || $SAT == "ENVI" || $SAT == "ALOS" || $SAT == "CSK_RAW") then
      ln -s ../SLC/freq_xcorr.dat .
      fitoffset.csh 3 3 freq_xcorr.dat 18 >> $aligned.PRM
    else
      ln -s ../SLC/freq_xcorr.dat .
      fitoffset.csh 2 2 freq_xcorr.dat 18 >> $aligned.PRM
    endif
    resamp_omp $master.PRM $aligned.PRM $aligned.PRMresamp $aligned.SLCresamp 4
    rm $aligned.SLC
    mv $aligned.SLCresamp $aligned.SLC
    cp $aligned.PRMresamp $aligned.PRM
    cd ../SLC
  endif
endif

exit 0
