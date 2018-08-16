#!/bin/sh
cd /mnt/rds/redhen/gallina/Singularity/data
for file in `ls`
do
  if [ "$file" != "wav" ]
  then
 	for mp4 in `ls $file`
	do
	   ffmpeg -i ./$file/$mp4 wav/$file/${mp4%%.*}.wav
	done
  fi
done
