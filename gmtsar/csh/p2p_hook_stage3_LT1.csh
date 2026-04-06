#!/bin/csh -f
# LT1 Stage-3 hook: auto-prepare topo/dem.grd when missing.

if ($#argv != 7) then
  echo ""
  echo "用法 / Usage: p2p_hook_stage3_LT1.csh SAT when master aligned conf topo_phase shift_topo"
  echo ""
  exit 1
endif

set SAT = $1
set when = $2
set master = $3
set aligned = $4
set conf = $5
set topo_phase = $6
set shift_topo = $7

if ("$when" != "pre") exit 0
if ("$topo_phase" != "1") exit 0
if (-f topo/dem.grd) then
  echo "LT1 stage-3 hook: reuse existing DEM topo/dem.grd"
  exit 0
endif

set xml_file = ""
if (-f raw/$master.xml) then
  set xml_file = raw/$master.xml
else if (-f raw/$master.meta.xml) then
  set xml_file = raw/$master.meta.xml
endif

if ("$xml_file" == "") then
  echo "LT1 stage-3 hook ERROR: missing raw scene xml (raw/$master.xml or raw/$master.meta.xml)"
  exit 1
endif

set tmp_bounds = /tmp/lt1_dem_bounds_$$.txt
perl -0777 -ne 'if (/<sceneInfo>(.*?)<\/sceneInfo>/s) { $sec=$1; while ($sec =~ /<sceneCornerCoord\b[^>]*>.*?<lat>([^<]+)<\/lat>.*?<lon>([^<]+)<\/lon>.*?<\/sceneCornerCoord>/sg) { push @lat, $1; push @lon, $2; } if (scalar(@lon)==0 || scalar(@lat)==0) { exit 2; } ($lonmin,$lonmax)=($lon[0],$lon[0]); for (@lon) { $lonmin=$_ if $_<$lonmin; $lonmax=$_ if $_>$lonmax; } ($latmin,$latmax)=($lat[0],$lat[0]); for (@lat) { $latmin=$_ if $_<$latmin; $latmax=$_ if $_>$latmax; } $pad=0.1; $w=$lonmin-$pad; $e=$lonmax+$pad; $s=$latmin-$pad; $n=$latmax+$pad; $e=$w+0.01 if $e <= $w; $n=$s+0.01 if $n <= $s; printf("%.2f %.2f %.2f %.2f\n", $w, $e, $s, $n); } else { exit 2; }' "$xml_file" >! $tmp_bounds

set bounds = (`cat $tmp_bounds`)
rm -f $tmp_bounds

if ($status != 0 || $#bounds != 4) then
  echo "LT1 stage-3 hook ERROR: failed to derive DEM bounds from $xml_file"
  exit 1
endif

set west = $bounds[1]
set east = $bounds[2]
set south = $bounds[3]
set north = $bounds[4]

echo "LT1 stage-3 hook: auto-generate DEM with make_dem.csh $west $east $south $north 1"
mkdir -p topo
set oldpwd = $PWD
cd topo
make_dem.csh $west $east $south $north 1
set rc = $status
cd "$oldpwd"

if ($rc != 0 || ! -f topo/dem.grd) then
  echo "LT1 stage-3 hook ERROR: auto DEM generation failed (topo/dem.grd missing)"
  exit 1
endif

echo "LT1 stage-3 hook: DEM ready at topo/dem.grd"
exit 0
