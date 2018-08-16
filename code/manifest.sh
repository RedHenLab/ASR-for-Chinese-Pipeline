#! /usr/bin/env bash

cd ..
BASEDIR=$(pwd)
for month in `ls data/split`
do
  for file in `ls data/split/$month`
  do
    mkdir -p $BASEDIR/manifest/$month 
    python code/manifest.py \
     --target_dir=$BASEDIR/data/split/$month/$file  \
     --manifest_path=$BASEDIR/manifest/$month/$file
  done
done

