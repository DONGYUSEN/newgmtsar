#!/bin/csh -f
#
#  ysdong@cug, 2026
#
unset noclobber
#
# script to convert a grd file to a geotiff file
#
if ($#argv < 5 || $#argv > 5) then
 echo " "
 echo "Usage: grdcomb2tiff.csh grd_file1 cptfile1 grd_file2 cptfile2  output"
 echo "grd_file2 以40%的透明度叠加到grd_file1之上。"
 echo "Example: grdcomb2tiff.csh final-amp_ll final-amp.cpt phasefilt_ll phase.cpt phase_amp_ll"
 echo " "
 exit 1
endif 
#
#
set DX = `gmt grdinfo $1.grd -C | cut -f8`
set DPI = `gmt math -Q $DX INV RINT = `
echo $DPI
gmt set COLOR_MODEL = hsv
gmt set PAPER_MEDIA = tabloid
#
  gmt grdimage $1.grd  -C$2 -Jx1id -P -Y2i -X2i -Q -K -V >  $5.ps
  gmt grdimage $3.grd  -C$4 -Jx1id -t60 -Q -O -V >> $5.ps
#
#   now make the geotiff 
#
echo "Make $5.tiff"
gmt psconvert $5.ps -W+g+t"$1" -E$DPI -P -A
#gmt psconvert $5.ps -TG -E$DPI -P -A
rm -f $5.ps 
#
