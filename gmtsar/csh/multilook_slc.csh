#!/bin/csh -f
# ysdong
# script for LT1 multibook 
#

if ($#argv != 3) then
  echo ""
  echo "Usage: multilook_slc.csh file(without .SLC or .PRM)  range_down_rate(2,4,8) azimuth_down_rate(2,4,8,16) "
  echo ""
  echo "Example: multilook_slc.csh 20240802  2 4"
  echo ""
  echo "Note: "
  echo ""
  exit 1
endif


set input = $1
set rdown = $2
set adown = $3

cp $input.PRM $input.PRM_backup
set prf = `grep PRF $input.PRM | awk '{print $3}'`
set rng_samp_rate = `grep rng_samp_rate $input.PRM | head -1 | awk '{print $3}'`
set num_valid_az = `grep num_valid_az $input.PRM | awk '{print $3}'`  # nrows num_lines
set num_rng_bins = `grep num_rng_bins $input.PRM | awk '{print $3}'`  # bytes_per_line good_bytes_per_line

set newprf = `echo $prf $adown | awk '{printf("%f", $1/$2)}'`
set new_rng_samp_rate = `echo $rng_samp_rate $rdown | awk '{printf("%f", $1/$2)}'`
set new_num_valid_az = `echo $num_valid_az $adown | awk '{printf("%d", $1/$2)}'`
set new_num_rng_bins = `echo $num_rng_bins $rdown | awk '{printf("%d", $1/$2)}'`
set bytes = `echo $new_num_rng_bins | awk '{printf("%d",$1*4)}'` # bytes_per_line good_bytes_per_line


update_PRM $input.PRM PRF $newprf
update_PRM $input.PRM rng_samp_rate $new_rng_samp_rate
update_PRM $input.PRM num_valid_az $new_num_valid_az
update_PRM $input.PRM nrows $new_num_valid_az
update_PRM $input.PRM num_lines $new_num_valid_az
update_PRM $input.PRM num_rng_bins $new_num_rng_bins 
update_PRM $input.PRM bytes_per_line $bytes
update_PRM $input.PRM good_bytes_per_line  $bytes
update_PRM $input.PRM num_patches 1

multilook $input.SLC  $num_rng_bins $rdown $adown temp.slc
mv temp.slc $input.SLC
rm $input.PRM_backup
# cat temp.slc.meta
rm temp.slc.meta



