#! /usr/bin/env bash
#This is a script to run the main inference task
# infer
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
--infer_manifest='manifest.ex' \
--mean_std_path='mean_std.npz' \
--vocab_path='vocab.txt' \
--model_path='params.tar.gz'  \
--lang_model_path='/mnt/rds/redhen/gallina/models/zhidao_giga.klm' \
--decoding_method='ctc_beam_search' \
--specgram_type='linear' \
--output_file='output/1.txt'
if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0
