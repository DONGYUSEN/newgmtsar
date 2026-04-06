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
    # LT1 often has PRF/DC mismatch between acquisitions.
    # Normalize parameters before coarse correlation to stabilize offsets.
    set prf_m = `grep PRF $master.PRM | awk 'NR==1{print $3}'`
    set prf_s = `grep PRF $aligned.PRM | awk 'NR==1{print $3}'`
    if ("x$prf_m" == "x" || "x$prf_s" == "x") then
      echo "错误：无法读取 LT1 PRF 参数 / ERROR: failed to read LT1 PRF"
      exit 1
    endif
    set prf_diff_pct = `echo "$prf_m $prf_s" | awk '{m=$1+0;s=$2+0; if(m==0){print 0}else{d=s-m;if(d<0)d=-d; printf("%.4f",100.0*d/m)}}'`
    set prf_need_norm = `echo "$prf_m $prf_s" | awk '{m=$1+0;s=$2+0; if(m==0){print 0}else{d=s-m;if(d<0)d=-d; if(d/m>0.0005) print 1; else print 0}}'`
    echo "LT1 PRF check: master=$prf_m aligned=$prf_s diff=${prf_diff_pct}%"
    if ($prf_need_norm == 1) then
      echo "LT1 PRF normalization: resample aligned SLC to master PRF=$prf_m"
      samp_slc.csh $aligned $prf_m 0
      if ($status != 0) then
        echo "错误：LT1 PRF 归一化失败 / ERROR: LT1 PRF normalization failed"
        exit 1
      endif
    endif

    set fd1_m = `grep fd1 $master.PRM | awk 'NR==1{print $3}'`
    set fd1_s = `grep fd1 $aligned.PRM | awk 'NR==1{print $3}'`
    set fdd1_m = `grep fdd1 $master.PRM | awk 'NR==1{print $3}'`
    set fdd1_s = `grep fdd1 $aligned.PRM | awk 'NR==1{print $3}'`
    set fddd1_m = `grep fddd1 $master.PRM | awk 'NR==1{print $3}'`
    set fddd1_s = `grep fddd1 $aligned.PRM | awk 'NR==1{print $3}'`
    if ("x$fd1_m" == "x") set fd1_m = 0
    if ("x$fd1_s" == "x") set fd1_s = 0
    if ("x$fdd1_m" == "x") set fdd1_m = 0
    if ("x$fdd1_s" == "x") set fdd1_s = 0
    if ("x$fddd1_m" == "x") set fddd1_m = 0
    if ("x$fddd1_s" == "x") set fddd1_s = 0
    set fd1_ref = $fd1_m
    set fdd1_ref = $fdd1_m
    set fddd1_ref = $fddd1_m
    update_PRM $aligned.PRM fd1 $fd1_ref
    update_PRM $aligned.PRM fdd1 $fdd1_ref
    update_PRM $aligned.PRM fddd1 $fddd1_ref
    set dc_diff = `echo "$fd1_m $fd1_s $fdd1_m $fdd1_s $fddd1_m $fddd1_s" | awk '{d1=$2-$1; if(d1<0)d1=-d1; d2=$4-$3; if(d2<0)d2=-d2; d3=$6-$5; if(d3<0)d3=-d3; printf("d_fd1=%.6f d_fdd1=%.6f d_fddd1=%.6f",d1,d2,d3)}'`
    echo "LT1 DC normalization: slave -> master (fd1=$fd1_ref fdd1=$fdd1_ref fddd1=$fddd1_ref), $dc_diff"

    echo "     大窗口，粗配准，2048*2048, 获得总体的偏差：  "
    xcorr2 $master.PRM $aligned.PRM -nx 30 -ny 30 -noshift -xsearch 512 -ysearch 512
    if ($status != 0) then
      echo "错误：LT1 粗配准 xcorr 失败 / ERROR: LT1 coarse xcorr failed"
      exit 1
    endif
    fitoffset.csh 2 2 freq_xcorr.dat 
    fitoffset.csh 2 2 freq_xcorr.dat 20 >> $aligned.PRM
    update_PRM $aligned.PRM SC_identity 12 
    
  else
    xcorr2 $master.PRM $aligned.PRM -noshift -xsearch 128 -ysearch 128 -nx 20 -ny 50
    fitoffset.csh 2 2  freq_xcorr.dat 18 >> $aligned.PRM
  endif
  echo "                     ..重采样    "

  resamp $master.PRM $aligned.PRM $aligned.PRMresamp $aligned.SLCresamp 4
  
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
