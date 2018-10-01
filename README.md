# Automatic Speech Recognition for Speech to Text on Chinese

This is [Zhaoqing Xu](https://xuzhaoqing.github.io) and [Shuwei Xu](https://cynthiasuwi.github.io/)'s  Google Summer of Code project for Red Hen Lab.

The aim of this project is to develop a working Speech to Text module for the Red Hen Lab’s Chinese Pipeline. In this project, a Speech-to-Text conversion engine on Chinese is established, resulting in a working application.

#### Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preprocessing-for-training)
3. [Training](#training)
4. [Inference](#inference)
5. [Running Code at CWRU HPC](#running-code-at-cwru-hpc)
6. [Results](#results)
7. [Acknowledgments](#acknowledgments)

## Getting Started
This project is based on the open source project [DeepSpeech2 on PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech#getting-started) released by Baidu. For the configuration, you may choose to build your own environment from scratch or use our singularity recipe. I strongly recommend you to use our recipe to avoid potential problems. 
### Prerequisites
*for the Red Hen Lab participants, all the configuration has been set up on the server*

* Python == 2.7  
* Singularity == 2.4.1 
* CUDA == 7.5

for anyone who doesn't want to use singularity, you could check this [webpage](https://github.com/PaddlePaddle/DeepSpeech#installation) to install PaddlePaddle and DeepSpeech by hand, but we strongly recommend you to use the container of singularity to avoid the potential problems.

## Data Preparation
### Data Description
What we use in this project is the Chinese video news data collected by Red Hen Lab, including Xinwenlianbo(新闻联播), Wanjianxinwen(晚间新闻), etc. The video lengths vary from 20 minutes to 30 minutes.  
### Format Conversion
Firstly we need to convert the video to audio so that we could apply our model. 
The tool we use here is a built-in linux application: FFmpeg, the usage is :
`ffmpeg -i file.mp4 file.wav`
You could check in my transform.sh to convert all of your videos into audios.
### Split the audio
Since the model is intended to process short audios, we should split the audio into short time. Here I set the split-time with 10s. 
The tool we use here is also a built-in linux application: sox, the usage here is:
`sox trim 0 10 : newfile : restart`
You could check in my split.sh to split your audios into 10 seconds peices.
### Prepare the Manifest
Manifest is a file that includes the information of data, including the location and the duration. It's acturally a json file that looks like:
```
{"audio_filepath": "data/split/2018-01/2018-01-31_0410_CN_HNTV1_午间新闻/2018-01-31_0410_CN_HNTV1_午间新闻001.wav", "duration": 10.0, "text": ""}
{"audio_filepath": "data/split/2018-01/2018-01-31_0410_CN_HNTV1_午间新闻/2018-01-31_0410_CN_HNTV1_午间新闻002.wav", "duration": 10.0, "text": ""}
{"audio_filepath": "data/split/2018-01/2018-01-31_0410_CN_HNTV1_午间新闻/2018-01-31_0410_CN_HNTV1_午间新闻003.wav", "duration": 10.0, "text": ""}
{"audio_filepath": "data/split/2018-01/2018-01-31_0410_CN_HNTV1_午间新闻/2018-01-31_0410_CN_HNTV1_午间新闻004.wav", "duration": 10.0, "text": ""}
{"audio_filepath": "data/split/2018-01/2018-01-31_0410_CN_HNTV1_午间新闻/2018-01-31_0410_CN_HNTV1_午间新闻005.wav", "duration": 10.0, "text": ""}
```
You could check in my manifest.sh and manifest.py to see how to implement this.

## Training
Since we don't have the ground truth(or label), we use the model trained by Baidu, which sort of harms our results. If you have the labels already, you could check this [webpage](https://github.com/PaddlePaddle/DeepSpeech#training-for-mandarin-language) for more things about training. 
## Inference
To get the inference of your data, we need the following files:
### Infer Manifest
This is the manifest that generates from the [Prepare the Manifest](#prepare-the-manifest)
### Mean & Stddev
This is the file that perform z-score normalization which includes audio features. 
```
python code/compute_mean_std.py \
--num_samples 2000 \
--specgram_type linear \
--manifest_paths {your manifest path here} \
--output_path {your expected output path}
```
You could check in the `compute_mean.sh` and `compute_mean_std.py` to see more details. 
### Vocabulary
This is the file to count all the words(in Chinese we say characters, same thing) in your data. Note that all the generated words are from the vocabulary, so if you didn't put the expected word here, it's impossible to generate it.
We just use the vocab.txt that Baidu provides. You could find the vocab.txt in code directory.
### Speech Model
We used [Aishell Model](https://github.com/PaddlePaddle/DeepSpeech#speech-model-released) here, which is trained on the Aishell Dataset for 151h.
### Language Model
We used [Mandarin LM Large](https://github.com/PaddlePaddle/DeepSpeech#language-model-released) here, which has no pruning and about 3.7 billion n-grams.

**Note: Though we could build vocab.txt mean_std of our data, I highyly recommend you use the mean_std and vocab.txt from Baidu Speech Model. The reason is not clear, but the performance is largely different.**

After preparing for all the files, you could fill them in the right place in infer.sh:
```shell
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
            --infer_manifest=$PWDDIR'/manifest/'$month/$manifest \
            --mean_std_path=$BASEDIR'/models/mean_std.npz' \
            --vocab_path=$BASEDIR'/models/vocab.txt' \
            --model_path=$BASEDIR'/models/params.tar.gz'  \
            --lang_model_path=$BASEDIR'/models/zhidao_giga.klm' \
            --decoding_method='ctc_beam_search' \
            --specgram_type='linear' \
            --output_file=$PWDDIR'/text/'$month/$manifest'.txt'
```

## Running Code at CWRU HPC
### Get into the container
```shell
module load singularity/2.5.1
module load cuda/7.5
export SINGULARITY_BINDPATH="/mnt"
srun -p gpu -C gpup100 --mem=100gb --gres=gpu:2 --pty bash
cd /mnt/rds/redhen/gallina/Singularity/Chinese_Pipeline
singularity shell -e --nv Chinese_Pipeline.simg
```
### Run the infer.sh
```
cd code
sh infer.sh
```
### Check the results
```
cd ../text
```
all the results are in the `/mnt/rds/redhen/gallina/Singularity/Chinese_Pipeline/text`
## Results
The example of results is like this:
```
20180112204720.000|20180112204730.000|ISO_01 老人们也在逐一排除最后的视频最后十>八点五十八分零三秒头部的也要会和微型面包车被警告锁定
20180112204730.000|20180112204740.000|ISO_01 哪些已成了男人的世界里心里装的是的>是的我们有一个重要会议重要
20180112204740.000|20180112204750.000|ISO_01 小面包车车主将二人生的轨迹给大家此>前经过的公交车里的行车记录仪正好拍下了地的微型面包车在路口闯红灯
20180112204750.000|20180112204800.000|ISO_01 宝马的目的了有的还没有与女友的女儿>只能期望的目的没有指标的一面也有
```

## Acknowledgments
* [Google Summer of Code 2018](https://summerofcode.withgoogle.com/)
* [Red Hen Lab](http://www.redhenlab.org/)
* [DeepSpeech2 on PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech)
* [DeepSpeech2 Paper](http://proceedings.mlr.press/v48/amodei16.pdf)
