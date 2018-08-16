#! /usr/bin/env bash
#This is a script to run the main inference task
cd ..
BASEDIR=$(pwd )
cd code
for month in `ls ../manifest`
do
  for manifest in `ls ../manifest/$month`
  do
    mkdir -p ../text/$month
    CUDA_VISIBLE_DEVICES=0 \
    python -u infer.py \
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
      --infer_manifest=$BASEDIR'/manifest/'$month/$manifest \
      --mean_std_path='/mnt/rds/redhen/gallina/models/mean_std.npz' \
      --vocab_path='/mnt/rds/redhen/gallina/models/vocab.txt' \
      --model_path='/mnt/rds/redhen/gallina/models/params.tar.gz'  \
      --lang_model_path='/mnt/rds/redhen/gallina/models/zhidao_giga.klm' \
      --decoding_method='ctc_beam_search' \
      --specgram_type='linear' \
      --output_file=$BASEDIR'/text/'$month/$manifest'.txt'
      if [ $? -ne 0 ]; then
          echo "Failed in inference!"
          exit 1
      fi
      echo $manifest' is done'
      done

done

exit 0
