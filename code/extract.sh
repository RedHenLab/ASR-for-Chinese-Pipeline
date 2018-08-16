#!/bin/sh

#This script is to find all the Chinese data and move them into another folder


cd /mnt/rds/redhen/gallina/tv/2018
for date in 2018-0{1,2,3,4,5,6} 
do
   cd $date
   for file in `ls`
   do  
   	find ./$file -name '*_CN_*.mp4' -exec cp {} /mnt/rds/redhen/gallina/Singularity/data/$date \;
   	echo $file
   done
   cd ..
done
exit 
