#!/bin/csh -f
# $Id$
#  modified by ysdong, 2025.12.26
#  Xiaohua Xu, Jan, 2018
#
#  Automatically perform two-path processing on raw(1.0)/SLC(1.1) data
#  

  if ($#argv < 3 || $#argv > 5) then
    echo ""
    echo "用法:"
    echo "  1) 自动多视（推荐）"
    echo "     p2p_processing.csh SAT master_image aligned_image [configuration_file]"
    echo "  2) 手动多视（命令行指定 rg:za）"
    echo "     p2p_processing.csh SAT master_image aligned_image [configuration_file] rg:za"
    echo "  3) 手动多视（配置文件指定）"
    echo "     在 configuration_file 中设置:"
    echo "       multilook_mode = manual"
    echo "       multilook_rg_az = rg:za    (或 range_dec / azimuth_dec)"
    echo ""
    echo "示例:"
    echo "  自动: p2p_processing.csh DJ1 20241110 20241121 config.DJ1.txt"
    echo "  手动: p2p_processing.csh DJ1 20241110 20241121 config.DJ1.txt 2:2"
    echo "  手动: p2p_processing.csh S1_TOPS master aligned 8:2"
    echo ""
    echo "    Put the data and orbit files in the raw folder, put DEM in the topo folder"
    echo "    The SAT needs to be specified, choices with in ERS, ENVI, ALOS, ALOS_SLC, ALOS2, ALOS2_SCAN"
    echo "    S1_STRIP, S1_TOPS, ENVI_SLC, CSK_RAW, CSK_SLC, CSG, TSX, RS2, GF3, LT1, DJ1"
    echo ""
    echo "    常用卫星手动多视参考值 range_dec:azimuth_dec（写入 config 的 range_dec/azimuth_dec）:"
    echo "      S1_TOPS -> 8:2"
    echo "      ALOS2_SCAN -> 4:8"
    echo "      RS2/TSX -> 1:1 或 2:2"
    echo "      ERS/ENVI/ALOS/ALOS_SLC/ALOS2/S1_STRIP/ENVI_SLC/CSK_RAW/CSK_SLC/CSG/GF3/LT1/DJ1 -> 2:2"
    echo "    注：以上为常用起始值，最终应以相干性、解缠稳定性与目标分辨率需求微调。"
    echo ""
    echo "    Make sure the files from the same date have the same stem, e.g. aaaa.tif aaaa.xml aaaa.cos aaaa.EOF, etc"
    echo ""
    echo "    If the configuration file is left blank, the program will generate one "
    echo "    with default parameters "
    echo ""
    echo "    参数优先级: 命令行 rg:za > multilook_rg_az > range_dec/azimuth_dec"
    echo "    rg:za 为手动多视（range_dec:azimuth_dec），例如 8:2、2:2、1:1"
    echo "    只要进入手动模式（命令行或配置文件），系统会自动严格按手动 rg:za 执行"
    echo "    force_rgaz 为兼容旧配置保留项，通常不需要设置"
    echo ""
    exit 1
  endif

# start
# parse optional config / rg:za argument(s)
  set conf = ""
  set cli_rgza = ""
  if ($#argv >= 4) then
    set arg4_is_rgza = `echo "$4" | awk -F: '{if(NF==2 && $1~/^[0-9]+$/ && $2~/^[0-9]+$/) print 1; else print 0}'`
    if ($#argv == 4) then
      if ($arg4_is_rgza == 1) then
        set cli_rgza = "$4"
      else
        if(! -f "$4" ) then
          echo " no configure file: $4"
          echo " Leave it blank to generate config file with default values."
          exit 1
        endif
        set conf = "$4"
      endif
    else if ($#argv == 5) then
      if(! -f "$4" ) then
        echo " no configure file: $4"
        echo " Leave it blank to generate config file with default values."
        exit 1
      endif
      set conf = "$4"
      set arg5_is_rgza = `echo "$5" | awk -F: '{if(NF==2 && $1~/^[0-9]+$/ && $2~/^[0-9]+$/) print 1; else print 0}'`
      if ($arg5_is_rgza != 1) then
        echo "错误参数：rg:za=$5（应为整数:整数，如 8:2） / Invalid rg:za: $5"
        exit 1
      endif
      set cli_rgza = "$5"
    endif
  endif

  # Ensure bundled csh helpers are discoverable when running by path.
  set script_dir = $0:h
  if ("$script_dir" == "$0") set script_dir = "."
  if ($?PATH) then
    setenv PATH "$script_dir":"$PATH"
  else
    setenv PATH "$script_dir"
  endif

# Avoid Conda/system mixed runtime libraries causing GMT/GDAL crashes.
  if (-x /usr/bin/gmt) then
    alias gmt /usr/bin/gmt
  endif
  if ($?LD_LIBRARY_PATH) then
    set _gmtsar_ld_clean = ""
    foreach _gmtsar_ldp (`echo "$LD_LIBRARY_PATH" | tr ':' ' '`)
      if ("$_gmtsar_ldp" !~ "*miniforge3*" && "$_gmtsar_ldp" !~ "*mambaforge*" && "$_gmtsar_ldp" !~ "*anaconda*" && "$_gmtsar_ldp" !~ "*conda*") then
        if ("x$_gmtsar_ld_clean" == "x") then
          set _gmtsar_ld_clean = "$_gmtsar_ldp"
        else
          set _gmtsar_ld_clean = "${_gmtsar_ld_clean}:$_gmtsar_ldp"
        endif
      endif
    end
    if ("x$_gmtsar_ld_clean" == "x") then
      unsetenv LD_LIBRARY_PATH
    else
      setenv LD_LIBRARY_PATH "$_gmtsar_ld_clean"
    endif
  endif
  if ($?GDAL_DRIVER_PATH) then
    if ("$GDAL_DRIVER_PATH" =~ "*miniforge3*" || "$GDAL_DRIVER_PATH" =~ "*mambaforge*" || "$GDAL_DRIVER_PATH" =~ "*anaconda*" || "$GDAL_DRIVER_PATH" =~ "*conda*") then
      unsetenv GDAL_DRIVER_PATH
    endif
  endif
  if ($?GDAL_DATA) then
    if ("$GDAL_DATA" =~ "*miniforge3*" || "$GDAL_DATA" =~ "*mambaforge*" || "$GDAL_DATA" =~ "*anaconda*" || "$GDAL_DATA" =~ "*conda*") then
      unsetenv GDAL_DATA
    endif
  endif
  if ($?PROJ_LIB) then
    if ("$PROJ_LIB" =~ "*miniforge3*" || "$PROJ_LIB" =~ "*mambaforge*" || "$PROJ_LIB" =~ "*anaconda*" || "$PROJ_LIB" =~ "*conda*") then
      unsetenv PROJ_LIB
    endif
  endif
  if ($?CONDA_PREFIX || $?CONDA_DEFAULT_ENV) then
    echo "检测到 Conda 环境，已清理 GMT/GDAL 运行库路径，避免混链崩溃。"
  endif
  
#
#  Read parameters from the configure file
#
date
setenv GMT_MEMORY_LIMIT 8192
setenv OMP_NUM_THREADS 10

  set SAT = `echo $1`
  if ("$conf" == "") then
    csh -f "$script_dir/pop_config.csh" $SAT > config.$SAT.txt
    set conf = `echo "config.$SAT.txt"`
  endif
  # conf may need to be changed later on
  set stage = `awk '$1=="proc_stage" && $2=="=" {print $3; exit}' $conf`
  set s_stages = `awk '$1=="skip_stage" && $2=="=" {print $3; exit}' $conf | awk -F, '{print $1,$2,$3,$4,$5,$6}'`
  set skip_1 = 0
  set skip_2 = 0 
  set skip_3 = 0 
  set skip_4 = 0 
  set skip_5 = 0 
  set skip_6 = 0 
  foreach line (`echo $s_stages`)
    if ($line == 1) set skip_1 = 1
    if ($line == 2) set skip_2 = 1
    if ($line == 3) set skip_3 = 1
    if ($line == 4) set skip_4 = 1
    if ($line == 5) set skip_5 = 1
    if ($line == 6) set skip_6 = 1
  end
  if ("x$s_stages" != "x") then
    echo ""
    echo "Skipping stage $s_stages ..."
  endif
  set skip_master = `awk '$1=="skip_master" && $2=="=" {print $3; exit}' $conf`
  if ($skip_master == "") set skip_master = 0
  if ($skip_master == 2) then
    set skip_4 = 1
    set skip_5 = 1
    set skip_6 = 1
    echo "Skipping stage 4,5,6 as skip_master is set to 2 ..."
  endif
  set num_patches = `grep num_patches $conf | awk '{print $3}'`
  set near_range = `grep near_range $conf | awk '{print $3}'`
  set earth_radius = `grep earth_radius $conf | awk '{print $3}'`
  set fd = `grep fd1 $conf | awk '{print $3}'`
  set topo_phase = `grep topo_phase $conf | awk '{print $3}'`
  set topo_interp_mode = `grep topo_interp_mode $conf | awk '{print $3}'`
  if ( "x$topo_interp_mode" == "x" ) then
    set topo_interp_mode = 0
  endif
  set shift_topo = `grep shift_topo $conf | awk '{print $3}'`
  set switch_master = `grep switch_master $conf | awk '{print $3}'`
  set filter = `grep filter_wavelength $conf | awk '{print $3}'` 
  set compute_phase_gradient = `grep compute_phase_gradient $conf | awk '{print $3}'` 
  set iono = `grep correct_iono $conf | awk '{print $3}'`
  if ( "x$iono" == "x" ) then 
    set iono = 0
  endif
  set iono_filt_rng = `grep iono_filt_rng $conf | awk '{print $3}'`
  set iono_filt_azi = `grep iono_filt_azi $conf | awk '{print $3}'`
  set iono_dsamp = `grep iono_dsamp $conf | awk '{print $3}'`
  set iono_skip_est = `grep iono_skip_est $conf | awk '{print $3}'`
  set spec_div = `grep spec_div $conf | awk '{print $3}'`
  if ( "x$spec_div" == "x" ) then
    set spec_div = 0
  endif
  set spec_mode = `grep spec_mode $conf | awk '{print $3}'`
  #  set filter = 200
  #  echo " "
  #  echo "WARNING filter wavelength was not set in config.txt file"
  #  echo "        please specify wavelength (e.g., filter_wavelength = 200)"
  #  echo "        remove filter1 = gauss_alos_200m"
  #endif
  set dec = `grep dec_factor $conf | awk '{print $3}'` 
  if ("x$dec" == "x") then
    set dec = 1
  endif
  set threshold_snaphu = `grep threshold_snaphu $conf | awk '{print $3}'`
  set threshold_geocode = `grep threshold_geocode $conf | awk '{print $3}'`
  set region_cut = `grep region_cut $conf | awk '{print $3}'`
  set mask_water = `grep mask_water $conf | awk '{print $3}'`
  set switch_land = `grep switch_land $conf | awk '{print $3}'`
  set defomax = `grep defomax $conf | awk '{print $3}'`
  set range_dec = `grep range_dec $conf | awk '{print $3}'`
  set azimuth_dec = `grep azimuth_dec $conf | awk '{print $3}'`
  set multilook_mode = `awk '$1=="multilook_mode" && $2=="=" {print $3; exit}' $conf`
  set multilook_rg_az = `awk '$1=="multilook_rg_az" && $2=="=" {print $3; exit}' $conf`
  set force_rgaz = `awk '$1=="force_rgaz" && $2=="=" {print $3; exit}' $conf`
  if ("x$multilook_mode" == "x") then
    set multilook_mode = "auto"
  endif
  if ("x$force_rgaz" == "x") then
    set force_rgaz = 0
  endif
  set multilook_mode = `echo "$multilook_mode" | tr 'A-Z' 'a-z'`
  set force_ok = `echo "$force_rgaz" | awk '{if($1==0 || $1==1) print 1; else print 0}'`
  if ($force_ok != 1) then
    echo "错误参数：force_rgaz=$force_rgaz（仅允许 0 或 1） / Invalid force_rgaz=$force_rgaz"
    exit 1
  endif

  set range_dec_eff = ""
  set azimuth_dec_eff = ""
  set multilook_source = "auto"
  set multilook_mode_eff = "$multilook_mode"

  if ("$multilook_mode" != "auto") then
    if ("$multilook_mode" != "manual") then
    echo "警告：未知 multilook_mode=$multilook_mode，回退为 auto / WARNING: unknown multilook_mode=$multilook_mode, fallback to auto"
    set multilook_mode = "auto"
    set multilook_mode_eff = "auto"
    endif
  endif

  if ("$cli_rgza" != "") then
    set range_dec_eff = `echo "$cli_rgza" | awk -F: '{print $1}'`
    set azimuth_dec_eff = `echo "$cli_rgza" | awk -F: '{print $2}'`
    set multilook_mode_eff = "manual"
    set multilook_source = "cli"
  else if ("$multilook_mode" == "manual") then
    if ("$multilook_rg_az" != "") then
      set rgza_ok = `echo "$multilook_rg_az" | awk -F: '{if(NF==2 && $1~/^[0-9]+$/ && $2~/^[0-9]+$/) print 1; else print 0}'`
      if ($rgza_ok != 1) then
        echo "错误参数：multilook_rg_az=$multilook_rg_az（应为整数:整数） / Invalid multilook_rg_az=$multilook_rg_az"
        exit 1
      endif
      set range_dec_eff = `echo "$multilook_rg_az" | awk -F: '{print $1}'`
      set azimuth_dec_eff = `echo "$multilook_rg_az" | awk -F: '{print $2}'`
      set multilook_source = "config(multilook_rg_az)"
    else if ("$range_dec" != "") then
      if ("$azimuth_dec" != "") then
        set range_dec_eff = "$range_dec"
        set azimuth_dec_eff = "$azimuth_dec"
        set multilook_source = "config(range_dec,azimuth_dec)"
      else
        echo "错误：multilook_mode=manual 时 range_dec 存在但 azimuth_dec 缺失"
        exit 1
      endif
    else
      echo "错误：multilook_mode=manual 但未提供 multilook_rg_az 或 range_dec/azimuth_dec"
      exit 1
    endif
  else
    # legacy compatibility: if both old keys exist and mode not explicitly set, treat as manual
    if ("$range_dec" != "") then
      if ("$azimuth_dec" != "") then
        if ("$multilook_rg_az" == "") then
          if ("$multilook_mode" == "auto") then
            set range_dec_eff = "$range_dec"
            set azimuth_dec_eff = "$azimuth_dec"
            set multilook_mode_eff = "manual"
            set multilook_source = "legacy_config(range_dec,azimuth_dec)"
          endif
        endif
      endif
    endif
  endif

  if ("$range_dec_eff" != "") then
    if ("$azimuth_dec_eff" == "") then
      echo "错误：range_dec 与 azimuth_dec 需要同时存在 / range_dec and azimuth_dec must be provided together"
      exit 1
    endif
  else
    if ("$azimuth_dec_eff" != "") then
      echo "错误：range_dec 与 azimuth_dec 需要同时存在 / range_dec and azimuth_dec must be provided together"
      exit 1
    endif
  endif

  if ("$range_dec_eff" != "") then
    if ("$azimuth_dec_eff" != "") then
      set rgok = `echo "$range_dec_eff $azimuth_dec_eff" | awk '{if($1>=1 && $2>=1) print 1; else print 0}'`
      if ($rgok != 1) then
        echo "错误：无效多视参数 range_dec=$range_dec_eff azimuth_dec=$azimuth_dec_eff（需 >=1）"
        exit 1
      endif
    endif
  endif
  set SLC_factor = `grep SLC_factor $conf | awk '{print $3}'`
  set near_interp = `grep near_interp $conf | awk '{print $3}'`
  set data_level = `awk '$1=="data_level" && $2=="=" {print $3; exit}' $conf`
  if ( "x$data_level" == "x" ) then
    set data_level = 0
  endif
  if ($data_level != 0 && $data_level != 1) then
    echo "Wrong parameter: data_level "$data_level
    exit 1
  endif
  set separate_focus = `awk '$1=="separate_focus" && $2=="=" {print $3; exit}' $conf`
  if ( "x$separate_focus" == "x" ) then
    set separate_focus = 0
  endif
  if ($data_level == 1) then
    # L1/SLC mode: stage-1 pre_proc is skipped and stage starts from stage-2
    if ($stage == 1) then
      set stage = 2
    endif
    set skip_1 = 1
    if ($separate_focus == 1) then
      echo "separate_focus is ignored when data_level = 1"
      set separate_focus = 0
    endif
  endif
  set master = ` echo $2 `
  set aligned =  ` echo $3 `
  echo ""
  

#
#  combine preprocess parameters
#  
  set commandline = ""
  if (!($earth_radius == "")) then 
    set commandline = "$commandline -radius $earth_radius"
  endif
  if (!($num_patches == "")) then  
    set commandline = "$commandline -npatch $num_patches"
  endif
  if (!($SLC_factor == "")) then  
    set commandline = "$commandline -SLC_factor $SLC_factor"
  endif
  if (!($spec_div == 0)) then
    set commandline = "$commandline -ESD $spec_mode"
  endif
  if (!($skip_master == "")) then
    set commandline = "$commandline -skip_master $skip_master"
  endif
  


#############################
# 1 - start from preprocess #
#############################
#
#   make sure the files exist
#
  if ($stage == 1 && $skip_1 == 0) then
    echo ""
    echo "PREPROCESS - START"
    echo ""
    echo "Working on images $master $aligned ..."
    if ($SAT == "ALOS" || $SAT == "ALOS2" || $SAT == "ALOS_SLC" || $SAT == "ALOS2_SCAN") then
      if(! -f raw/$master ) then
        echo " no file  raw/"$master
        exit
      endif
      if(! -f raw/$aligned ) then
        echo " no file  raw/"$aligned
        exit
      endif
    else if ($SAT == "ENVI_SLC") then
      if(! -f raw/$master.N1 && ! -f raw/$master.E1 && ! -f raw/$master.E2) then
        echo " no file  raw/"$master
        exit
      endif
      if(! -f raw/$aligned.N1 && ! -f raw/$aligned.E1 && ! -f raw/$aligned.E2 ) then
        echo " no file  raw/"$aligned
        exit
      endif
    else if ($SAT == "ERS") then
      if(! -f raw/$master.dat ) then
        echo " no file  raw/"$master.dat
        exit
      endif
      if(! -f raw/$aligned.dat ) then
        echo " no file  raw/"$aligned.dat
        exit
      endif
      if(! -f raw/$master.ldr ) then
        echo " no file  raw/"$master.ldr
        exit
      endif
      if(! -f raw/$aligned.ldr ) then
        echo " no file  raw/"$aligned.ldr
        exit
      endif
    else if ($SAT == "ENVI") then
      if(! -f raw/$master.baq ) then
        echo " no file  raw/"$master.baq
        exit
      endif
      if(! -f raw/$aligned.baq ) then
        echo " no file  raw/"$aligned.baq
        exit
      endif
    else if ($SAT == "S1_STRIP" || $SAT == "S1_TOPS"|| $SAT == "DJ1") then
      if(! -f raw/$master.xml ) then
        echo " no file  raw/"$master".xml"
        exit
      endif
      if(! -f raw/$master.tiff ) then
        echo " no file  raw/"$master".tiff"
        exit
      endif
      if(! -f raw/$aligned.xml ) then
        echo " no file  raw/"$aligned".xml"
        exit
      endif
      if(! -f raw/$aligned.tiff ) then
        echo " no file  raw/"$aligned".tiff"
        exit
      endif
      if ($SAT == "S1_TOPS") then
        if(! -f raw/$master.EOF ) then
          echo " no file  raw/"$master".EOF"
        endif
        if(! -f raw/$aligned.EOF ) then
          echo " no file  raw/"$aligned".EOF"
        endif
      endif
    else if ($SAT == "CSK_RAW" || $SAT == "CSK_SLC" || $SAT == "CSG") then
      if(! -f raw/$master.h5 ) then
        echo " no file  raw/"$master".h5"
        exit
      endif
      if(! -f raw/$aligned.h5 ) then
        echo " no file  raw/"$aligned".h5"
        exit
      endif
    else if ($SAT == "RS2") then
      if(! -f raw/$master.xml ) then
        echo " no file  raw/"$master".xml"
        exit
      endif
      if(! -f raw/$master.tif ) then
        echo " no file  raw/"$master".tif"
        exit
      endif
      if(! -f raw/$aligned.xml ) then
        echo " no file  raw/"$aligned".xml"
        exit
      endif
      if(! -f raw/$aligned.tif ) then
        echo " no file  raw/"$aligned".tif"
        exit
      endif
    else if ($SAT == "TSX") then
      if(! -f raw/$master.xml ) then
        echo " no file  raw/"$master".xml"
        exit
      endif
      if(! -f raw/$aligned.xml ) then
        echo " no file  raw/"$aligned".xml"
        exit
      endif
      if(! -f raw/$master.cos ) then
        echo " no file  raw/"$master".cos"
        exit
      endif
      if(! -f raw/$aligned.cos ) then
        echo " no file  raw/"$aligned".cos"
        exit
      endif
    else if ($SAT == "GF3") then
      if(! -f raw/$master.xml ) then
        echo " no file  raw/"$master".xml"
        exit
      endif
      if(! -f raw/$aligned.xml ) then
        echo " no file  raw/"$aligned".xml"
        exit 
      endif
      if(! -f raw/$master.tiff ) then
        echo " no file  raw/"$master".tiff"
        exit 
      endif
      if(! -f raw/$aligned.tiff ) then
        echo " no file  raw/"$aligned".tiff"
        exit
      endif
    else if ($SAT == "LT1") then
      if(! -f raw/$master.xml ) then
        echo " no file  raw/"$master".xml"
        exit
      endif
      if(! -f raw/$aligned.xml ) then
        echo " no file  raw/"$aligned".xml"
        exit 
      endif
      if(! -f raw/$master.tiff ) then
        echo " no file  raw/"$master".tiff"
        exit 
      endif
      if(! -f raw/$aligned.tiff ) then
        echo " no file  raw/"$aligned".tiff"
        exit
      endif
    endif

#
#  Start preprocessing
#
    echo "1.                      数据准备阶段 ......  "
    if ($SAT == "S1_TOPS") then
      set master = `echo $master | awk '{ print "S1_"substr($1,16,8)"_"substr($1,25,6)"_F"substr($1,7,1)}'`
      set aligned = `echo $aligned | awk '{ print "S1_"substr($1,16,8)"_"substr($1,25,6)"_F"substr($1,7,1)}'`
    endif
    if ($skip_master == 0 || $skip_master == 2) then
      rm -f raw/$master.PRM*
      rm -f raw/$master.SLC
      rm -f raw/$master.LED
    endif
    if ($skip_master == 0 || $skip_master == 1) then
      rm -f raw/$aligned.PRM*
      rm -f raw/$aligned.SLC
      rm -f raw/$aligned.LED
    endif
    if ($SAT == "S1_TOPS") then
      set master = ` echo $2 `
      set aligned = `echo $3`
    endif
    cd raw
    echo "                      -------- "
    echo "pre_proc.csh $SAT $master $aligned $commandline"
    pre_proc.csh $SAT $master $aligned $commandline   
    cd ..
    echo " "
    echo "PREPROCESS - END"
    echo ""
  endif
 
#############################################
# 2 - start from focus and align SLC images #
#############################################
# 

  mkdir -p SLC
  if ($iono == 1) then
    mkdir -p SLC_L 
    mkdir -p SLC_H
  endif

  if ($SAT == "S1_TOPS") then
    set master = `echo $master | awk '{ print "S1_"substr($1,16,8)"_"substr($1,25,6)"_F"substr($1,7,1)}'`
    set aligned = `echo $aligned | awk '{ print "S1_"substr($1,16,8)"_"substr($1,25,6)"_F"substr($1,7,1)}'`
  endif

  if ($stage <= 2 && $skip_2 == 0) then 
    # 阶段2插件钩子 / Stage-2 plugin hook
    set run_stage2 = 1
    p2p_hook_stage2.csh $SAT pre $master $aligned $conf $data_level $skip_master
    if ($status == 10) then
      echo "阶段2由 Hook 跳过 (SAT=$SAT) / Stage-2 skipped by hook (SAT=$SAT)"
      set run_stage2 = 0
    else if ($status == 20) then
      echo "阶段2由 Hook 接管处理 (SAT=$SAT) / Stage-2 handled by hook (SAT=$SAT)"
      set run_stage2 = 0
    else if ($status != 0) then
      echo "错误：阶段2前置 Hook 失败 / ERROR: stage-2 pre-hook failed"
      exit 1
    endif

    if ($run_stage2 == 1) then
    #cleanup.csh SLC
    if ($skip_master == 0 || $skip_master == 2) then
      rm -f SLC/$master.PRM*
      rm -f SLC/$master.SLC
      rm -f SLC/$master.LED
    endif
    if ($skip_master == 0 || $skip_master == 1) then
      rm -f SLC/$aligned.PRM*
      rm -f SLC/$aligned.SLC
      rm -f SLC/$aligned.LED
    endif
    if ($iono == 1) then
      if ($skip_master == 0 || $skip_master == 2) then
        rm -f SLC/$2.tiff
        rm -f SLC/$2.xml
        rm -f SLC/$2.EOF
        rm -f SLC_L/$master.PRM*
        rm -f SLC_L/$master.SLC
        rm -f SLC_L/$master.LED
        rm -f SLC_L/$2.tiff
        rm -f SLC_L/$2.xml
        rm -f SLC_L/$2.EOF
        rm -f SLC_H/$master.PRM*
        rm -f SLC_H/$master.SLC
        rm -f SLC_H/$master.LED
        rm -f SLC_H/$2.tiff
        rm -f SLC_H/$2.xml
        rm -f SLC_H/$2.EOF
      endif
      if ($skip_master == 0 || $skip_master == 1) then
        rm -f SLC/$3.tiff
        rm -f SLC/$3.xml
        rm -f SLC/$3.EOF
        rm -f SLC_L/$aligned.PRM*
        rm -f SLC_L/$aligned.SLC
        rm -f SLC_L/$aligned.LED
        rm -f SLC_L/$3.tiff
        rm -f SLC_L/$3.xml
        rm -f SLC_L/$3.EOF
        rm -f SLC_H/$aligned.PRM*
        rm -f SLC_H/$aligned.SLC
        rm -f SLC_H/$aligned.LED
        rm -f SLC_H/$3.tiff
        rm -f SLC_H/$3.xml
        rm -f SLC_H/$3.EOF
      endif
    endif


#
# focus and align SLC images 
# 
    echo " "
    echo "ALIGN.CSH - START"
    echo "2.                     数据配准阶段 ......  "
    cd SLC
    if ($SAT != "S1_TOPS") then
      p2p_stage2_focus.csh $SAT $master $aligned $skip_master $iono $separate_focus $data_level
      if ($status == 10) then
        cd ..
        echo ""
        echo "ALIGN.CSH - END"
        echo ""
        exit 0
      else if ($status != 0) then
        echo "ERROR: p2p_stage2_focus.csh failed"
        exit 1
      endif

      p2p_stage2_align.csh $SAT $master $aligned $skip_master $iono
      set stage2_align_status = $status
      if ($stage2_align_status == 20) then
        # keep original behavior: LT1 branch exits early from stage2
        exit 0
      else if ($stage2_align_status != 0) then
        echo "ERROR: p2p_stage2_align.csh failed"
        exit 1
      endif
      if ($SAT == "LT1") then
        if (! -s freq_xcorr.dat) then
          echo "ERROR: LT1 stage-2 produced empty/missing freq_xcorr.dat"
          exit 1
        endif
      endif

    else if ($SAT == "S1_TOPS") then
      p2p_stage2_tops.csh $master $aligned $2 $3 $skip_master $iono
      if ($status != 0) then
        echo "ERROR: p2p_stage2_tops.csh failed"
        exit 1
      endif
    endif
    
    echo "3.                     裁减一下图，减少工作量"
    if ($region_cut != "") then
      echo "Cutting SLC image to $region_cut"
      if ($skip_master == 0 || $skip_master == 2) then
        cut_slc $master.PRM junk1 $region_cut
        mv junk1.PRM $master.PRM 
        mv junk1.SLC $master.SLC
      endif
      if ($skip_master == 0 || $skip_master == 1) then
        cut_slc $aligned.PRM junk2 $region_cut
        mv junk2.PRM $aligned.PRM
        mv junk2.SLC $aligned.SLC
      endif

      if ($iono == 1) then
        cd ../SLC_L
        if ($skip_master == 0 || $skip_master == 2) then
          cut_slc $master.PRM junk1 $region_cut
          mv junk1.PRM $master.PRM
          mv junk1.SLC $master.SLC
        endif
        if ($skip_master == 0 || $skip_master == 1) then
          cut_slc $aligned.PRM junk2 $region_cut
          mv junk2.PRM $aligned.PRM
          mv junk2.SLC $aligned.SLC
        endif 
        cd ../SLC_H
        if ($skip_master == 0 || $skip_master == 2) then
          cut_slc $master.PRM junk1 $region_cut
          mv junk1.PRM $master.PRM
          mv junk1.SLC $master.SLC
        endif
        if ($skip_master == 0 || $skip_master == 1) then
          cut_slc $aligned.PRM junk2 $region_cut
          mv junk2.PRM $aligned.PRM
          mv junk2.SLC $aligned.SLC
        endif
      endif
    endif

      cd ..
      echo ""
      echo "ALIGN.CSH - END"
      echo ""  
    endif

    p2p_hook_stage2.csh $SAT post $master $aligned $conf $data_level $skip_master
    if ($status != 0) then
      echo "错误：阶段2后置 Hook 失败 / ERROR: stage-2 post-hook failed"
      exit 1
    endif
  endif
##################################
# 3 - start from make topo_ra  #
##################################
#
  if ($stage <= 3 && $skip_3 == 0) then
    # 阶段3插件钩子 / Stage-3 plugin hook
    set run_stage3 = 1
    p2p_hook_stage3.csh $SAT pre $master $aligned $conf $topo_phase $shift_topo
    if ($status == 10) then
      echo "阶段3由 Hook 跳过 (SAT=$SAT) / Stage-3 skipped by hook (SAT=$SAT)"
      set run_stage3 = 0
    else if ($status == 20) then
      echo "阶段3由 Hook 接管处理 (SAT=$SAT) / Stage-3 handled by hook (SAT=$SAT)"
      set run_stage3 = 0
    else if ($status != 0) then
      echo "错误：阶段3前置 Hook 失败 / ERROR: stage-3 pre-hook failed"
      exit 1
    endif

    if ($run_stage3 == 1) then
      echo ""
      echo "STAGE-3 - START / 阶段3开始：地形相位与几何准备"
      p2p_validate_stage3.csh $SAT $master $aligned $topo_phase $shift_topo
      if ($status != 0) then
        echo "错误：阶段3输入校验失败 / ERROR: stage-3 input validation failed"
        exit 1
      endif

#
# clean up
#
      cleanup.csh topo
#
# make topo_ra if there is dem.grd
#
      if ("$topo_phase" == "1") then
        echo " "
        echo "4.                     将DEM转换到雷达坐标（斜距-方位） / Convert DEM to radar coordinates (range-azimuth)"
        echo "DEM2TOPO_RA.CSH - START / DEM2TOPO_RA.CSH - 开始"
        echo "需要用户提供 DEM 文件 / USER SHOULD PROVIDE DEM FILE"
        cd topo
        cp ../SLC/$master.PRM master.PRM
        rm -f $master.LED
        ln -s ../raw/$master.LED .
        if ($topo_interp_mode == 1) then
          dem2topo_ra.csh master.PRM dem.grd 1
        else
          dem2topo_ra.csh master.PRM dem.grd
        endif
        if (! -f topo_ra.grd) then
          echo "DEM2TOPO 失败：未生成 topo_ra.grd / DEM2TOPO failed: topo_ra.grd was not generated"
          exit 1
        endif
        cd ..
        echo "DEM2TOPO_RA.CSH - END / DEM2TOPO_RA.CSH - 结束"
#
# shift topo_ra
#
        echo "5.                     拓扑偏移与幅度图准备 / Topo shift and amplitude preparation"
        if (! -f SLC/$aligned.PRM) then
          echo "缺少阶段3输入文件 / Missing stage-3 input: SLC/$aligned.PRM"
          exit 1
        endif
        if (! -f topo/topo_ra.grd) then
          echo "缺少阶段3输入文件 / Missing stage-3 input: topo/topo_ra.grd"
          exit 1
        endif
        if ("$shift_topo" == "1") then
          echo " "
          echo "OFFSET_TOPO - START / OFFSET_TOPO - 开始"
          cd SLC
          set rng_samp_rate = `grep rng_samp_rate $master.PRM | awk 'NR == 1 {printf("%d", $3)}'`
          set rng = `gmt grdinfo ../topo/topo_ra.grd | grep x_inc | awk '{print $7}'`
          slc2amp.csh $master.PRM $rng amp-$master.grd
          slc2amp.csh $aligned.PRM $rng amp-$aligned.grd
          gmt grdmath amp-$master.grd amp-$aligned.grd ADD 0.5 MUL LOG2 100 ADD = final-amp.grd
          cd ..
          cd topo
          rm -f final-amp.grd amp-$master.grd
          ln -s ../SLC/final-amp.grd .
          ln -s ../SLC/amp-$master.grd .
          echo "配准 SAR 图像与 DEM / Align SAR amplitude and DEM"
          if ($SAT == "LT1") then
            offset_topo2 final-amp.grd topo_ra.grd 0 0 128 topo_shift.grd
          else
            offset_topo2 final-amp.grd topo_ra.grd 0 0 64 topo_shift.grd
          endif
          if (! -f topo_shift.grd) then
            echo "OFFSET_TOPO 失败：未生成 topo_shift.grd / OFFSET_TOPO failed: topo_shift.grd was not generated"
            exit 1
          endif
          cd ../SLC
          gmt grdmath amp-$master.grd amp-$aligned.grd ADD 0.5 MUL 0.5 POW LOG2 100 ADD FLIPUD = final-amp.grd
          cd ..
          echo "OFFSET_TOPO - END / OFFSET_TOPO - 结束"
        else
          cd SLC
          set rng_samp_rate = `grep rng_samp_rate $master.PRM | awk 'NR == 1 {printf("%d", $3)}'`
          set rng = `gmt grdinfo ../topo/topo_ra.grd | grep x_inc | awk '{print $7}'`
          slc2amp.csh $master.PRM $rng amp-$master.grd
          slc2amp.csh $aligned.PRM $rng amp-$aligned.grd
          gmt grdmath amp-$master.grd amp-$aligned.grd ADD 0.5 MUL 0.5 POW LOG2 100 ADD FLIPUD = final-amp.grd
          rm amp-$aligned.grd
          cd ..
          cd topo
          rm -f amp-$master.grd final-amp.grd
          ln -s ../SLC/amp-$master.grd .
          ln -s ../SLC/final-amp.grd .
          cd ..
          echo "不做 topo_ra 偏移 / NO TOPO_RA SHIFT"
        endif
      else
        echo "不去除地形相位（topo_phase=0）/ NO TOPO_RA IS SUBTRACTED (topo_phase=0)"
      endif

      echo "STAGE-3 - END / 阶段3结束"
    endif

    p2p_hook_stage3.csh $SAT post $master $aligned $conf $topo_phase $shift_topo
    if ($status != 0) then
      echo "错误：阶段3后置 Hook 失败 / ERROR: stage-3 post-hook failed"
      exit 1
    endif
  endif

##################################################
# 4 - start from make and filter interferograms  #
##################################################
#

#
# select the master
#    
  if ("$switch_master" == "0") then
    set ref = $master
    set rep = $aligned
  else if ("$switch_master" == "1") then
    set ref = $aligned
    set rep = $master
  else
    echo "错误参数：switch_master=$switch_master / Invalid parameter: switch_master=$switch_master"
    exit 1
  endif

  if ($stage <= 4 && $skip_4 == 0) then
    # 阶段4插件钩子 / Stage-4 plugin hook
    set run_stage4 = 1
    p2p_hook_stage4.csh $SAT pre $ref $rep $conf $topo_phase $iono
    if ($status == 10) then
      echo "阶段4由 Hook 跳过 (SAT=$SAT) / Stage-4 skipped by hook (SAT=$SAT)"
      set run_stage4 = 0
    else if ($status == 20) then
      echo "阶段4由 Hook 接管处理 (SAT=$SAT) / Stage-4 handled by hook (SAT=$SAT)"
      set run_stage4 = 0
    else if ($status != 0) then
      echo "错误：阶段4前置 Hook 失败 / ERROR: stage-4 pre-hook failed"
      exit 1
    endif

    if ($run_stage4 == 1) then
      echo ""
      echo "STAGE-4 - START / 阶段4开始：干涉图生成与滤波"
      p2p_validate_stage4.csh $SAT $ref $rep $topo_phase $shift_topo $iono
      if ($status != 0) then
        echo "错误：阶段4输入校验失败 / ERROR: stage-4 input validation failed"
        exit 1
      endif
#
# clean up
#
      mkdir -p intf
#    cleanup.csh intf
# 
# make and filter interferograms
# 
    echo "6.                     开始干涉处理和滤波处理 / Start interferogram generation and filtering"
    echo " "
    echo "INTF.CSH, FILTER.CSH - START / INTF.CSH, FILTER.CSH - 开始"
    cd intf/
    set ref_id  = `grep SC_clock_start ../raw/$ref.PRM | awk '{printf("%d",int($3))}' `
    set rep_id  = `grep SC_clock_start ../raw/$rep.PRM | awk '{printf("%d",int($3))}' `
    mkdir -p $ref_id"_"$rep_id
    cd $ref_id"_"$rep_id
    rm -f $ref.LED $rep.LED $ref.SLC $rep.SLC
    ln -s ../../SLC/$ref.LED . 
    ln -s ../../SLC/$rep.LED .
    ln -s ../../SLC/$ref.SLC . 
    ln -s ../../SLC/$rep.SLC .
    cp ../../SLC/$ref.PRM . 
    cp ../../SLC/$rep.PRM .

    # Resolve effective multilook and geocoding square pixel size.
    set ref_rng_samp_rate = `grep rng_samp_rate $ref.PRM | awk 'NR==1{print $3}'`
    set ref_prf = `grep PRF $ref.PRM | awk 'NR==1{print $3}'`
    set ref_sc_vel = `grep SC_vel $ref.PRM | awk 'NR==1{print $3}'`
    set ref_sc_height = `grep SC_height $ref.PRM | awk 'NR==1{print $3}'`
    set ref_earth_radius = `grep earth_radius $ref.PRM | awk 'NR==1{print $3}'`

    set dr_ground_m = `echo "$ref_rng_samp_rate" | awk '{if($1>0){printf("%.6f",1.556*299792458.0/(2.0*$1));}else{print "";}}'`
    if ("x$dr_ground_m" == "x") set dr_ground_m = 10

    set da_ground_m = `echo "$ref_sc_vel $ref_sc_height $ref_earth_radius $ref_prf" | awk '{gv=$1; if($2>0 && $3>0 && $1>0){gv=$1/sqrt(1.0+$2/$3)}; if($4>0 && gv>0){printf("%.6f",gv/$4);} else {print "";}}'`
    if ("x$da_ground_m" == "x") set da_ground_m = 10

    if ("$range_dec_eff" == "" || "$azimuth_dec_eff" == "") then
      set range_dec_eff = `echo "$dr_ground_m $da_ground_m $dec" | awk '{dr=$1; da=$2; dc=int($3+0); if(dc<1) dc=1; t=(dr>da)?dr:da; rg=int(t/dr+0.5); if(rg<1) rg=1; if(rg>64) rg=64; print rg*dc;}'`
      set azimuth_dec_eff = `echo "$dr_ground_m $da_ground_m $dec" | awk '{dr=$1; da=$2; dc=int($3+0); if(dc<1) dc=1; t=(dr>da)?dr:da; az=int(t/da+0.5); if(az<1) az=1; if(az>64) az=64; print az*dc;}'`
      set multilook_mode_eff = "auto"
      set multilook_source = "auto(prm_approx)"
    endif

    # Guard auto/manual values so filter.csh never derives idec/jdec=0.
    # Manual rg:za is always strict (no auto adjustment). Auto mode keeps guard.
    set force_rgaz_eff = 0
    if ("$multilook_mode_eff" == "manual") then
      set force_rgaz_eff = 1
    endif

    if ($force_rgaz_eff == 1) then
      echo "Multilook manual mode: keep rg:az exactly as requested"
    else if ($force_rgaz == 1 && "$multilook_mode_eff" != "manual") then
      echo "Multilook force mode ignored in auto mode (force_rgaz=1, mode=$multilook_mode_eff)"
    endif

    if ($force_rgaz_eff != 1) then
      set ml_adjust = `echo "$range_dec_eff $azimuth_dec_eff $ref_rng_samp_rate $ref_prf" | awk '{rg=int($1+0); az=int($2+0); rs=$3+0; prf=$4+0; if(rg<1) rg=1; if(az<1) az=1; az_lks=(prf<1000)?1:4; if(rs>110000000) dec_rng=4; else if(rs>20000000) dec_rng=2; else dec_rng=1; if((az%2)!=0) az_lks=1; if((rg%2)!=0) dec_rng=1; if(az<az_lks) az=az_lks; if(rg<dec_rng) rg=dec_rng; if(az_lks>1 && (az%az_lks)!=0) az=int((az+az_lks-1)/az_lks)*az_lks; if(dec_rng>1 && (rg%dec_rng)!=0) rg=int((rg+dec_rng-1)/dec_rng)*dec_rng; print rg, az;}'`
      set range_dec_eff = `echo "$ml_adjust" | awk '{print $1}'`
      set azimuth_dec_eff = `echo "$ml_adjust" | awk '{print $2}'`
    else
      # strict manual mode keeps user-requested rg:az unchanged
    endif

    set geo_pix_m = `echo "$dr_ground_m $da_ground_m $range_dec_eff $azimuth_dec_eff" | awk '{dr=$1*$3; da=$2*$4; g=(dr>da)?dr:da; if(g<=0) g=60; printf("%.3f",g)}'`
    if ("x$geo_pix_m" == "x") set geo_pix_m = 60

    echo "Multilook resolved: mode=$multilook_mode_eff source=$multilook_source range_dec=$range_dec_eff azimuth_dec=$azimuth_dec_eff"
    echo "Ground spacing estimate: dr=${dr_ground_m}m da=${da_ground_m}m => geocode square pixel=${geo_pix_m}m"
    echo "multilook_mode = $multilook_mode_eff" > multilook.meta
    echo "multilook_source = $multilook_source" >> multilook.meta
    echo "range_dec_eff = $range_dec_eff" >> multilook.meta
    echo "azimuth_dec_eff = $azimuth_dec_eff" >> multilook.meta
    echo "force_rgaz = $force_rgaz" >> multilook.meta
    echo "force_rgaz_eff = $force_rgaz_eff" >> multilook.meta
    echo "dr_ground_m = $dr_ground_m" >> multilook.meta
    echo "da_ground_m = $da_ground_m" >> multilook.meta
    echo "geo_pix_m = $geo_pix_m" >> multilook.meta

    if ("$topo_phase" == "1") then
      echo "生成干涉图并去除地形相位 / Generate interferogram with topographic phase removal"
      if ("$shift_topo" == "1") then
        rm -f topo_shift.grd
        ln -s ../../topo/topo_shift.grd .
        intf.csh $ref.PRM $rep.PRM -topo topo_shift.grd  
        filter.csh $ref.PRM $rep.PRM $filter $dec $range_dec_eff $azimuth_dec_eff $compute_phase_gradient $force_rgaz_eff
      else 
        rm -f topo_ra.grd
        ln -s ../../topo/topo_ra.grd . 
        intf.csh $ref.PRM $rep.PRM -topo topo_ra.grd 
        filter.csh $ref.PRM $rep.PRM $filter $dec $range_dec_eff $azimuth_dec_eff $compute_phase_gradient $force_rgaz_eff
      endif
    else
      echo "仅生成干涉图（不去除地形相位） / Generate interferogram without topographic phase removal"
      intf.csh $ref.PRM $rep.PRM
      filter.csh $ref.PRM $rep.PRM $filter $dec $range_dec_eff $azimuth_dec_eff $compute_phase_gradient $force_rgaz_eff
    endif
    cd ../..

    if ("$iono" == "1") then
    echo "6:                     开始电离层链路（高/低频子带） / Start ionosphere branch (high/low sub-bands)"
      if (-e iono_phase ) rm -r iono_phase
      mkdir -p iono_phase
      cd iono_phase 
      mkdir -p intf_o intf_h intf_l iono_correction

      set new_incx = `echo $range_dec_eff $iono_dsamp | awk '{print $1*$2}'`
      set new_incy = `echo $azimuth_dec_eff $iono_dsamp | awk '{print $1*$2}'`

      echo ""
      cd intf_h
      ln -s ../../SLC_H/*.SLC .
      ln -s ../../SLC_H/*.LED .
      cp ../../SLC_H/*.PRM .
      cp ../../SLC/params* .
      if ("$topo_phase" == "1") then
        if ("$shift_topo" == "1") then
          rm -f topo_shift.grd
          ln -s ../../topo/topo_shift.grd .
          intf.csh $ref.PRM $rep.PRM -topo topo_shift.grd  
          filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy 0 $force_rgaz_eff
        else 
          rm -f topo_ra.grd
          ln -s ../../topo/topo_ra.grd . 
          intf.csh $ref.PRM $rep.PRM -topo topo_ra.grd 
          filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy 0 $force_rgaz_eff
        endif
      else
        echo "不进行地形相位去除 / NO TOPOGRAPHIC PHASE REMOVAL PERFORMED"
        intf.csh $ref.PRM $rep.PRM
        filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy 0 $force_rgaz_eff
      endif
      cp phase.grd phasefilt.grd
      if ($iono_skip_est == 0) then
        if ($mask_water == 1 || $switch_land == 1) then
          set rcut = `gmt grdinfo phase.grd -I- | cut -c3-20`
          cd ../../topo
          landmask.csh $rcut
          cd ../iono_phase/intf_h
          ln -s ../../topo/landmask_ra.grd .
        endif
        snaphu_interp.csh 0.05 0
      endif
      cd ..

      echo ""
      cd intf_l
      ln -s ../../SLC_L/*.SLC .
      ln -s ../../SLC_L/*.LED .
      cp ../../SLC_L/*.PRM .
      cp ../../SLC/params* .
      if ("$topo_phase" == "1") then
        if ("$shift_topo" == "1") then
          rm -f topo_shift.grd
          ln -s ../../topo/topo_shift.grd .
          intf.csh $ref.PRM $rep.PRM -topo topo_shift.grd
          filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy 0 $force_rgaz_eff
        else 
          rm -f topo_ra.grd
          ln -s ../../topo/topo_ra.grd . 
          intf.csh $ref.PRM $rep.PRM -topo topo_ra.grd 
          filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy 0 $force_rgaz_eff
        endif
      else
        echo "不进行地形相位去除 / NO TOPOGRAPHIC PHASE REMOVAL PERFORMED"
        intf.csh $ref.PRM $rep.PRM
        filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy 0 $force_rgaz_eff
      endif
      cp phase.grd phasefilt.grd
      if ($iono_skip_est == 0) then
        if ($mask_water == 1 || $switch_land == 1) ln -s ../../topo/landmask_ra.grd .
        snaphu_interp.csh 0.05 0
      endif
      cd ..

      echo ""
      cd intf_o
      ln -s ../../SLC/*.SLC .
      ln -s ../../SLC/*.LED .
      cp ../../SLC/*.PRM .
      if ("$topo_phase" == "1") then
        if ("$shift_topo" == "1") then
          rm -f topo_shift.grd
          ln -s ../../topo/topo_shift.grd .
          intf.csh $ref.PRM $rep.PRM -topo topo_shift.grd
          filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy
        else
          rm -f topo_ra.grd
          ln -s ../../topo/topo_ra.grd .
          intf.csh $ref.PRM $rep.PRM -topo topo_ra.grd
          filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy
        endif
      else
        echo "不进行地形相位去除 / NO TOPOGRAPHIC PHASE REMOVAL PERFORMED"
        intf.csh $ref.PRM $rep.PRM
        filter.csh $ref.PRM $rep.PRM 500 $dec $new_incx $new_incy
      endif
      cp phase.grd phasefilt.grd
      if ($iono_skip_est == 0) then
        if ($mask_water == 1 || $switch_land == 1) ln -s ../../topo/landmask_ra.grd .
        snaphu_interp.csh 0.05 0
      endif
      cd ../iono_correction
      echo ""

      if ($iono_skip_est == 0) then
        estimate_ionospheric_phase.csh ../intf_h ../intf_l ../intf_o ../../intf/$ref_id"_"$rep_id $iono_filt_rng $iono_filt_azi
      
        cd ../../intf/$ref_id"_"$rep_id
        mv phasefilt.grd phasefilt_non_corrected.grd
        gmt grdsample ../../iono_phase/iono_correction/ph_iono_orig.grd -Rphasefilt_non_corrected.grd -Gph_iono.grd
        gmt grdmath phasefilt_non_corrected.grd ph_iono.grd SUB PI ADD 2 PI MUL MOD PI SUB = phasefilt.grd
        gmt grdimage phasefilt.grd -JX6.5i -Bxaf+lRange -Byaf+lAzimuth -BWSen -Cphase.cpt -X1.3i -Y3i -P -K > phasefilt.ps
        gmt psscale -Rphasefilt.grd -J -DJTC+w5i/0.2i+h -Cphase.cpt -Bxa1.57+l"Phase" -By+lrad -O >> phasefilt.ps
        gmt psconvert -Tf -P -A -Z phasefilt.ps
        #rm phasefilt.ps
      endif
      cd ../../
    endif

      echo "INTF.CSH, FILTER.CSH - END / INTF.CSH, FILTER.CSH - 结束"
      echo "STAGE-4 - END / 阶段4结束"
    endif

    p2p_hook_stage4.csh $SAT post $ref $rep $conf $topo_phase $iono
    if ($status != 0) then
      echo "错误：阶段4后置 Hook 失败 / ERROR: stage-4 post-hook failed"
      exit 1
    endif
  endif


################################
# 5 - start from unwrap phase  #
################################
#
  if ($stage <= 5 && $skip_5 == 0) then
    if ($threshold_snaphu != 0 ) then
      cd intf
      set ref_id  = `grep SC_clock_start ../raw/$ref.PRM | awk '{printf("%d",int($3))}' `
      set rep_id  = `grep SC_clock_start ../raw/$rep.PRM | awk '{printf("%d",int($3))}' `
      cd $ref_id"_"$rep_id
      echo "7.                     终于开始相位解缠了，采用多线程方式，提高速度！" #一堆问题？
#
# landmask
#
      if ($mask_water == 1 || $switch_land == 1) then
        set r_cut = `gmt grdinfo phase.grd -I- | cut -c3-20`
        cd ../../topo
        if (! -f landmask_ra.grd) then
          landmask.csh $r_cut
        endif
        cd ../intf
        cd $ref_id"_"$rep_id
        ln -s ../../topo/landmask_ra.grd .
      endif
#
      echo " "
      echo "SNAPHU.CSH - START"
      echo "threshold_snaphu: $threshold_snaphu"
#
      if ($near_interp == 1) then
        snaphu_interp.csh $threshold_snaphu $defomax
      else
        snaphu.csh $threshold_snaphu $defomax
      endif
#
      #echo "SNAPHU.CSH - END"
      echo "相位解缠结束：SNAPHU.CSH - END"
      
     # if ($SAT == "DJ1" || $SAT == "LT1" ||  $SAT == "GF3" ) then 
        #  echo "测试功能：在解缠后相位中去除 DJ1、LT1、GF3 平行干涉条纹......"
        
        #  gmt grdtrend  unwrap.grd -N3+r -Tphase_trend.grd  # N5 = a+bx+cy+dx^2+ey^2; N3 = a+bx+cy
        #  gmt grdmath unwrap.grd phase_trend.grd SUB = unwrap_rm_trend.grd
        #  cp unwrap.grd unwrap.org.grd
        #  mv unwrap_rm_trend.grd unwrap.grd
        #  echo "测试功能：在原始相位中去除相位趋势......"
        #  echo "原始相位文件保留为unwrap.org.grd, 趋势文件保留为phase_trend.grd"
          #gmt grdmath phase.grd phase_trend.grd SUB = phase_rm_trend.grd
          #gmt grdmath phase_rm_trend 2 PI MUL MOD = wrapped_phase_0_2pi.grd -fg
          #mv phase.grd phase.org.grd
          #mv wrapped_phase_0_2pi.grd phase.grd
        #  gmt grdmath phasefilt.grd phase_trend.grd SUB = phasefilt_rm_trend.grd
        #  gmt grdmath phasefilt_rm_trend.grd 2 PI MUL MOD = wrapped_phase_0_2pi.grd -fg
        #  mv phasefilt.grd phasefilt.org.grd
        #  mv wrapped_phase_0_2pi.grd phasefilt.grd
        #  echo "原始filt相位文件保留为：phasefilt.org.grd"
      # endif

      cd ../..
    else 
      echo ""
      echo "SKIP UNWRAP PHASE"
    endif
  endif

###########################
# 6 - start from geocode  #
###########################
#
  if ($stage <= 6 && $skip_6 == 0) then
    if ($threshold_geocode != 0 ) then
      echo "8.                     地理编码，成图！"
      cd intf
      set ref_id  = `grep SC_clock_start ../raw/$ref.PRM | awk '{printf("%d",int($3))}' `
      set rep_id  = `grep SC_clock_start ../raw/$rep.PRM | awk '{printf("%d",int($3))}' `
      cd $ref_id"_"$rep_id
      echo " "
      echo "GEOCODE.CSH - START"
      if (-f raln.grd) rm raln.grd 
      if (-f ralt.grd) rm ralt.grd
      if (-f trans.dat)  rm trans.dat
      if ($topo_phase == 1) then
        ln -s  ../../topo/trans.dat . 
        echo "threshold_geocode: $threshold_geocode"
        geocode.csh $threshold_geocode
      else 
        echo "topo_ra is needed to geocode"
        exit 1
      endif
      echo "GEOCODE.CSH - END"
      cd ../..
    else
      echo ""
      echo "SKIP GEOCODE"
      echo ""
    endif
  endif
#
# end  
  date
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
