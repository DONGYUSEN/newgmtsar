#!/bin/csh -f
#
# Run on linux machines
#
# Written 11/28/2023 by Xiaohua to fix the problem caused by ESA moving its service to
# fix the problem caused by ESA moving its service to a new website
#   https://step.esa.int/auxdata/orbits/Sentinel-1
#
# Written 04/05/2022 by Katherine Guns with aid from code snippets by Xiaohua Xu 
# and from ESA's website pages:
#   https://scihub.copernicus.eu/userguide/BatchScripting
#   https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/ODataAPI#URI_Components
#   https://scihub.copernicus.eu/gnss/#/home

if ($#argv != 2) then
    echo ""
    echo "Usage: download_sentinel_orbits.csh safefilelist mode"
    echo "  Downloads precise or restituted orbits for specific Sentinel-1 *.SAFE data files  "
    echo ""
    echo "safefilelist:"
    echo "    absolutepathto/filename1.SAFE"
    echo "    absolutepathto/filename2.SAFE"
    echo "    ......"
    echo "mode:"
    echo "    mode 1 = precise orbits (POEORB)"
    echo "            (most users should choose precise orbits)"
    echo "    mode 2 = temporary (restituted) orbits (RESORB)"
    echo "            (only recent data (~last couple weeks) requires restituted"
    echo "            orbits, because precise orbits are not yet finalized)"
    echo ""
    echo "Example: download_sentinel_orbits.csh SAFEfile.list 1"
    echo ""
    echo "Note: "
    echo "  (1) Files listed in safefilelist should be the .SAFE directory with absolute path."
    echo ""
    exit 1
endif

set s1_orbit_dir = "/Work/s1orbit"
set s1_orbit_dir_env = `printenv S1_ORBIT_DIR`
if ("x$s1_orbit_dir_env" != "x") then
    set s1_orbit_dir = "$s1_orbit_dir_env"
endif
if (! -d "$s1_orbit_dir") then
    echo "提示: S1轨道目录不存在: $s1_orbit_dir"
    echo "请先创建目录，或设置环境变量 S1_ORBIT_DIR 指向已存在目录。"
    exit 1
endif
if (! -r "$s1_orbit_dir" || ! -w "$s1_orbit_dir") then
    echo "提示: S1轨道目录不可读写: $s1_orbit_dir"
    echo "请检查目录权限。"
    exit 1
endif
echo "Using orbit cache directory: $s1_orbit_dir"


#-------------------------
# PRECISE ORBITS (POEORB)
#-------------------------

if ($2 == 1) then
    echo " Downloading Precise Orbits (POEORB)..."
    #start working with SAFE file list
    foreach line (` awk -F"/" '{print $(NF)}' $1`)   #pull the name of the SAFE file from end of path
      set orbittype="POEORB"
      echo " "
      echo "------------------------------------------ "
      echo " "
      echo "Finding orbits for ${line}..."
      set date1 = `echo $line | awk -F'/' '{print $NF}' | awk -F"_" '{print substr($6,1,8)}' `                
      set SAT1 = `echo $line | awk -F'/' '{print $NF}' | awk -F"_" '{print $1}' `                 

      # get the orbit file names 
      set n1 = ` date --date="$date1 - 1 day" +%Y%m%d `
      set n2 = ` date --date="$date1 + 1 day" +%Y%m%d `
      set yr = `echo $n1 | awk -F"_" '{print substr($1,1,4)}'`
      set mo = `echo $n1 | awk -F"_" '{print substr($1,5,2)}'`

      echo "Required orbit file dates: ${n1} to  ${n2}..." 
  
      wget https://step.esa.int/auxdata/orbits/Sentinel-1/$orbittype/$SAT1/$yr/$mo -O tmp_orbit.html

      set orbit = `grep $n1 tmp_orbit.html | grep $n2 | awk -F'"' '{print $2}'`
      if ("x" == $orbit"x") then
        echo "[ERROR]: No precise orbit exist for date $date1 "
        exit 1
      endif
      set file = `echo $orbit | awk '{print substr($1,1,length($1)-4)}'`

      if (-e $file) then
        echo "Using local precise orbit $file"
      else if (-e "$s1_orbit_dir/$file") then
        ln -sf "$s1_orbit_dir/$file" "$file"
        echo "Using cached precise orbit $file from $s1_orbit_dir"
      else
        echo "Downloading precise orbit $file ..."
        wget https://step.esa.int/auxdata/orbits/Sentinel-1/$orbittype/$SAT1/$yr/$mo/$orbit
        unzip $orbit $file
        rm $orbit
        if (-e $file) cp -f "$file" "$s1_orbit_dir/$file"
      endif
      rm tmp_orbit.html
    end
      
endif

#----------------------------
# RESTITUTED ORBITS (RESORB)
#----------------------------

if ($2 == 2) then
    echo " Downloading temporary Restituted Orbits (RESORB)..."
    #start working with SAFE file list
    foreach line (` awk -F"/" '{print $(NF)}' $1`)   #pull the name of the SAFE file from end of path
      set orbittype="RESORB"
      echo " "
      echo "------------------------------------------ "
      echo " "
      echo "Finding orbits for ${line}..."
      set date1 = `echo $line | awk -F'/' '{print $NF}' | awk -F"_" '{print substr($6,1,8)}' `                
      set yr = `echo $date1 | awk -F"_" '{print substr($1,1,4)}'`
      set mo = `echo $date1 | awk -F"_" '{print substr($1,5,2)}'`
      set datetime1 = `echo $line | awk -F'/' '{print $NF}' | awk -F"_" '{printf "%s:%s:%s %s-%s-%s",substr($6,10,2),substr($6,12,2),substr($6,14,2),substr($6,1,4),substr($6,5,2),substr($6,7,2)}' ` 
      set datetime2 = `echo $line | awk -F'/' '{print $NF}' | awk -F"_" '{printf "%s:%s:%s %s-%s-%s",substr($7,10,2),substr($7,12,2),substr($7,14,2),substr($7,1,4),substr($7,5,2),substr($7,7,2)}' ` 
      set SAT1 = ` echo $line | awk -F'/' '{print $NF}' | awk -F"_" '{print $1}' `                 

      wget https://step.esa.int/auxdata/orbits/Sentinel-1/$orbittype/$SAT1/$yr/$mo -O tmp_orbit.html

      awk -F'"' 'NR>4 {print $2}' tmp_orbit.html | grep $date1 > tmp_orbit.list 
      set start = `date --date="$datetime1 - 50 minute" +%s`
      set end = `date --date="$datetime2 + 50 minute" +%s`
      foreach rec (`cat tmp_orbit.list`)
        set t1 = `echo $rec | awk -F"_" '{printf "%s:%s:%s %s-%s-%s",substr($7,11,2),substr($7,13,2),substr($7,15,2),substr($7,2,4),substr($7,6,2),substr($7,8,2)}' `
        set t2 = `echo $rec | awk -F"_" '{printf "%s:%s:%s %s-%s-%s",substr($8,10,2),substr($8,12,2),substr($8,14,2),substr($8,1,4),substr($8,5,2),substr($8,7,2)}'`
        set tstart = `date --date="$t1" +%s`
        set tend = `date --date="$t2" +%s`

        set orbit = `echo $rec`
        set file = `echo $orbit | awk '{print substr($1,1,length($1)-4)}'`

        set crita = `echo $tstart $start | awk '{printf("%d",$1/$2)}'`
        set critb = `echo $tend $end | awk '{printf("%d",$1/$2)}'`
        if ($crita == 0 && $critb == 1) then
          if (-e $file) then
            echo "Using restituted orbit $file from local directory"
          else if (-e "$s1_orbit_dir/$file") then
            ln -sf "$s1_orbit_dir/$file" "$file"
            echo "Using restituted orbit $file from cache $s1_orbit_dir"
          else
            echo "Downloading restituted orbit $file ..."
            wget https://step.esa.int/auxdata/orbits/Sentinel-1/$orbittype/$SAT1/$yr/$mo/$orbit
            unzip $orbit $file
            rm $orbit
            if (-e $file) cp -f "$file" "$s1_orbit_dir/$file"
          endif
        endif
      end
      rm tmp_orbit.list tmp_orbit.html
    end
endif
