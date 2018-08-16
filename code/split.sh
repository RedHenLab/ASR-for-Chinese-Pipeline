#!/bin/bash
cd /mnt/rds/redhen/gallina/Singularity/data/wav
echo 'Split Begins'
for file in `ls`
do
   for wav in `ls $file`
   do 
	mkdir -p ../split/$file/${wav%%.*}
	sox $file/$wav  ../split/$file/${wav%%.*}/$wav  trim 0 10 : newfile : restart \;
        echo $file/$wav' split  done'
   done
done

exit
