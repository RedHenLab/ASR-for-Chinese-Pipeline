#!/bin/bash -xv
#
# Chinese pipeline: extract text from speech
#
# Changelog
#
#       2018-12-07 The words date and file are a reserved in bash, use DAT and FIL for variables; check for completed files (FFS)
#

echo -e "\n\tStarting infer.sh for automated speech recognition ...\n"

cd ..
PWDDIR=$( pwd ) # The directory of /code 
BASEDIR='/mnt/rds/redhen/gallina'
FAILED="/tmp/check-recordings-daemon-sysdown"
TO=`cat $BASEDIR/Singularity/Chinese_Pipeline/e-mail`

# Move to the main tv storage directory N days ago and list the contents
if [ -z "$1" ] ; then DAY=0 ; else DAY=${1:0:10} ; fi
if [ "$( echo "$1" | egrep '^[0-9]+$' )" ] ; then DAY="$1"
  elif [ "${#1}" -eq "7" ] ; then cd $BASEDIR/tv/${1%-*}/$1 ; DAY=""
  elif [ "$1" = "here" ] ; then DAY="$( pwd )" DAY=${DAY##*/} DAY="$[$[$(date +%s)-$(date -d "$DAY" +%s)]/86400]"
  elif [ "$1" = "+" ] ; then DAY=`pwd` ; DAY=${DAY##*/}
    DAY="$[$[$(date +%s)-$(date -ud "$DAY" +%s)]/86400]" ; DAY=$[DAY-$2]
  elif [ "$1" = "-" ] ; then DAY=`pwd` ; DAY=${DAY##*/}
    DAY="$[$[$(date +%s)-$(date -ud "$DAY" +%s)]/86400]" ; DAY=$[DAY+$2]
  elif [ "${#DAY}" -eq "10" ] ; then DAY="$[$[$(date +%s)-$(date -ud "$DAY" +%s)]/86400]"
  else echo "$1?"
fi #;  echo "DAY is $DAY ; 1 is $1 ; 2 is $2"
     
if [ -n "$DAY" ] ; then DIR="$BASEDIR/tv/$(date -ud "-$DAY day" +%Y)/$(date -ud "-$DAY day" +%Y-%m)/$(date -ud "-$DAY day" +%F)" ; fi
if [ -d $DIR ] ; then cd $DIR ; else echo "No $DIR" ; exit ; fi

echo -e "\tWorking on `pwd`\n"

DAT=$(basename `pwd`) # get the date name of this day
MONTH=${DAT:0:7}
YEAR=${DAT:0:4}

# Generate a list of files to process -- exclude _CN_ files in English
# Using find $DIR will generate a list with full path -- find . a list without path
find . -name '*_CN_*.mp4' ! -iname "*CGTN*" -exec cp {} $PWDDIR/temp_data/ \;

# Working directory -- it's unclear we really need to copy the files -- a downside is that unprocessed files remain in this location
cd $PWDDIR/temp_data

# If there are no files for a particular day, alert us with an e-mail
if [ -z "$(ls -A $DAT*.mp4)" ]; then
       echo -e "\n\tInfer.sh reports that $DAT doesn't have any Chinese files.\n\tPlease intervene as needed." > $FAILED
       /usr/bin/mail -s "No Chinese files on $DAT\n" $TO < $FAILED
       exit 0
fi

# Initialize counters
n=0 m=0

for FIL in $DAT*.mp4 ; do n=$[n+1]

  # Skip existing files
  if [ -f $PWDDIR'/new_text/'$YEAR/$MONTH/$DAT/${FIL%.*}.txt ] ; then echo -e "\t${FIL%.*}.txt has already been processed" ; m=$[m+1] ; continue ; fi

  # Extract and split the audio
  ffmpeg -hide_banner -nostats -loglevel 0 -i $FIL ./${FIL%%.*}.wav
  mkdir -p ${FIL%%.*}
  sox -V1 ${FIL%%.*}.wav  ${FIL%%.*}/${FIL%%.*}.wav  trim 0 10 : newfile : restart \;
  echo $FIL' split completed' 
  rm ${FIL%%.*}.wav
  rm $FIL
done

# Completed
if [ "$m" -eq "$n" ] ; then exit ; fi

# Create manifests
for FIL in `ls -d $DAT*` ; do

   # Skip existing files
   if [ -f $PWDDIR'/new_text/'$YEAR/$MONTH/$DAT/${FIL%.*}.txt ] ; then echo -e "\tSkipping manifest for $FIL" ; continue ; fi

   python ../code/manifest.py \
     --target_dir=$PWDDIR/temp_data/$FIL  \
     --manifest_path=$PWDDIR/temp_manifest/$FIL
done

cd $PWDDIR/temp_manifest

# Run the automated speech-to-text python script
for manifest in $DAT* ; do
         echo -e "\n\tRunning ASR on $manifest ...\n"
         mkdir -p $PWDDIR'/new_text/'$YEAR/$MONTH/$DAT
         CUDA_VISIBLE_DEVICES=0 \
         python -u ../code/infer.py \
            --batch_size=10 \
            --trainer_count=1 \
            --beam_size=300 \
            --num_proc_bsearch=2 \
            --num_conv_layers=2 \
            --num_rnn_layers=3 \
            --rnn_layer_size=1024 \
            --alpha=2.4 \
            --beta=5.0 \
            --cutoff_prob=0.99 \
            --cutoff_top_n=40 \
            --use_gru=True \
            --use_gpu=True \
            --share_rnn_weights=False \
            --infer_manifest=$PWDDIR'/temp_manifest/'$manifest \
            --mean_std_path=$BASEDIR'/models/mean_std.npz' \
            --vocab_path=$BASEDIR'/models/vocab.txt' \
            --model_path=$BASEDIR'/models/params.tar.gz'  \
            --lang_model_path=$BASEDIR'/models/zhidao_giga.klm' \
            --decoding_method='ctc_beam_search' \
            --specgram_type='linear' \
            --output_file=$PWDDIR'/new_text/'$YEAR/$MONTH/$DAT/$manifest'.txt' \
            --input_file=$BASEDIR'/tv/'$YEAR/$MONTH/$DAT/$manifest'.txt'
            if [ $? -ne 0 ]; then
               echo "Failed in inference!"
               rm $PWDDIR'/new_text/'$YEAR/$MONTH/$DAT/$manifest'.txt'
               continue
            fi
            echo $manifest' is done'
            rm $manifest
            rm -rf $PWDDIR/temp_data/$manifest
done

exit 0
