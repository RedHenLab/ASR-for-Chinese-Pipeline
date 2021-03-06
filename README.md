﻿# Automatic Speech Recognition for Speech to Text on Chinese

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
Sample output file:
```
TOP|20181209110001|2018-12-09_1100_CN_CCTV1_新闻联播
COL|Communication Studies Archive, UCLA
UID|5b1016ae-fba8-11e8-9d4c-d3bf2f6caf24
DUR|00:39:53
VID|720x576|1024x576
SRC|Changsha, China
TTL|Main news
CMT|
ASR_01|CMN
LBT|2018-12-09 19:00:01 Asia/Shanghai
ASR_01|2018-12-11 06:10|Source_Program=Baidu DeepSpeech2,infer.sh|Source_Person=Zhaoqing Xu,Shuwei Xu|Codebook=Chinese Speech to Text
20181209110001.000|20181209110011.000|ASR_01|年二十为人为文为人
20181209110011.000|20181209110021.000|ASR_01|名人名言名人名言
20181209110021.000|20181209110031.000|ASR_01|美国中国成为央行几天至十二月九号星期日农历十一月初三为了给美国的今天节目的主要内容
20181209110031.000|20181209110041.000|ASR_01|系列报道在一名新时代中国特色受读书想学一下现实的新作为新品种来看北京市保健销售
20181209110041.000|20181209110051.000|ASR_01|少部门报道列为一号改革课题推进行治理的应急机制服务群众的小面积切实落到没有物流
20181209110051.000|20181209110101.000|ASR_01|汪洋率中央代表团飞抵南宁出席广西壮族地区成立六十周年庆活动系列报告书说改革开放处处
20181209110101.000|20181209110111.000|ASR_01|尽管数十年来我国交通运输业实现跨越网站中国经济快速发展的强有力的针对我部统一部署
20181209110111.000|20181209110121.000|ASR_01|山东省公安机关依法侦办严重暴露了总统表示将借助植物友好
20181209110121.000|20181209110131.000|ASR_01|我最后变化大为他回国对中国积极应对其后变化并取得显著成效表示肯定目前来看降息报道
20181209110131.000|20181209110141.000|ASR_01|其一民总书记指出推动社会治理重心项目金松向一把基层党组织建设成为
20181209110141.000|20181209110151.000|ASR_01|宝鸡治理的进行战斗以来北京市坚持导致银保先生出手部门报道
20181209110151.000|20181209110201.000|ASR_01|也被城市一号改革课题了解了解群众诉求的现象基层一线放出去与本体的结合各部门共同响应
20181209110201.000|20181209110211.000|ASR_01|目前中通过目前相城分校推进基层治理体制机制创新切实做到民有所
20181209110211.000|20181209110221.000|ASR_01|属于这片雨林中南海的水上海景区非常有公里的避水不到所有多点全部打通正式对市民开放
20181209110221.000|20181209110231.000|ASR_01|用户酒吧的被骗路还被全部拆除引起或然开朗不禁露出了老城领域的配置也算是个老北京引领调整配
20181209110231.000|20181209110241.000|ASR_01|某的申请从前海地一直走到海里原来有望了我们认为感恩这个好事来了真是感觉到非常的模
20181209110241.000|20181209110251.000|ASR_01|之所以能彻底整治北京市的基层治理模式也就是接生推广部门报告机制只要发现问题系统接到打工
20181209110251.000|20181209110301.000|ASR_01|女生上扬工商城管是用电等部门的执法力量有同时下沉到基层一线明确责任综合执法一生多点难题的上下
20181209110301.000|20181209110311.000|ASR_01|两地盆地的底气十足今年是上海的到最后依法关停在一处酒吧被拆除违建一万一千多平米一批老大难
20181209110311.000|20181209110321.000|ASR_01|根据目前接到了马上看不见的只有了解了门网友写道是往往人民文
20181209110321.000|20181209110331.000|ASR_01|不要了的热狗和他们打架目的不是不是像网友们为了梦想我们合理的
20181209110331.000|20181209110341.000|ASR_01|不足影响机能力不强等是北京市场的很大的一统制问题还处于一线的街道和乡镇更是长期受制于条款
20181209110341.000|20181209110351.000|ASR_01|权力不统一面对问题有心无力在今年八月一日医生推着我们报道的都会上一个暗访视频直指要害
20181209110351.000|20181209110401.000|ASR_01|也没有因为儿女们作为母语人物农业和牧民认为个体没有人后退
20181209110401.000|20181209110411.000|ASR_01|有那个美网与人力网的论文也因为你们没有人没有没有我能有
20181209110411.000|20181209110421.000|ASR_01|由二零零零年我们没有因为文联没有没有没有没有有没有
20181209110421.000|20181209110431.000|ASR_01|在这个新成立的东北本地的胡家园社区综合服务站市民不仅能一方办理社保卫生计生等行政事务
20181209110431.000|20181209110441.000|ASR_01|水电修理搬家保洁等日航不一形式也能有个无也办不了的二十四小时之内必须有回应许多部门协调
20181209110441.000|20181209110451.000|ASR_01|土方怎么领导人有如意大门一六都对工人的动力中慢慢每年的比例不断是东城区第二管理
20181209110451.000|20181209110501.000|ASR_01|改革的是影片在推进推动报道工作中北京市对低保管理体制进行大包婆婆的改革用这本地道把原来到十九个部门机构
20181209110501.000|20181209110511.000|ASR_01|演出十二个实体推动更平息的对白与每个地区设立综合服务站两百万是不用多头跑到处找合并纪念
20181209110511.000|20181209110521.000|ASR_01|北京老干部群体与社区专员解决群众身边的本事就是运行都是地道的组织部长东洋现在整个动作的设计专用
20181209110521.000|20181209110531.000|ASR_01|每天总结传向前两天单位互动力安装路灯为了上一个月的生意至少这给我有我的事
20181209110531.000|20181209110541.000|ASR_01|我们是对手是个多能的演绎感受得到他的心就是你的一个好的结果的
20181209110541.000|20181209110551.000|ASR_01|老爷要他给个小姐接到一百年留下我们不再承担的职能部门给我们唯一下跌的个别的事物我们可以制度力量为我不一定
20181209110551.000|20181209110601.000|ASR_01|好的迫切的位置是共享微笑奥运会要做的和已有的网站有人干事有的不是新的管理体制带来的
20181209110601.000|20181209110611.000|ASR_01|使城市管理性能不断提高东城区网格中心平台上多年来抵押的五千二百七十三件无法回复就托不决案件不
20181209110611.000|20181209110621.000|ASR_01|八十七点正在处理与东西得到解决和提交具体由市卫生镇为小学生坠楼受伤一事
20181209110621.000|20181209110631.000|ASR_01|部门处理的好不好不在街道就可以由多个部门的严重的直销考核成绩落后
20181209110631.000|20181209110641.000|ASR_01|本是在等电影里医生推着我们报道工作中北京市把资金资源将基本情节二百九十个对象建立了
20181209110641.000|20181209110651.000|ASR_01|化综合执法中心选派了一万四千九百多名校长获得社区层面的治理和服务有益于推广报道以避免一档体育北京
20181209110651.000|20181209110701.000|ASR_01|十六个区网格化城市管理平台共接报各类案件二百十八点二万件解决率达到百分之九十三点八三满意率达百分之九
20181209110701.000|20181209110711.000|ASR_01|五一群众的诉求就是道道就是小城中宝宝每一日这里
20181209110711.000|20181209110721.000|ASR_01|系统治理的应急机制和进入相应体制打工了抓落实的最后一部是创新社会治理的成功作为一名
20181209110721.000|20181209110731.000|ASR_01|事无小大量工作在基层做好基层工作只要有服务意识也要有创新精神在这里这个小康社会的
20181209110731.000|20181209110741.000|ASR_01|计划建议各级政府的现在治理能力体系建设是满意不满意是我们做好信息的群众路线
20181209110741.000|20181209110751.000|ASR_01|到基隆一线解决问题为导向形成更有效的社会治理人民群众获得的幸福感还会更加重
20181209110751.000|20181209110801.000|ASR_01|他伟大的吸引力会空气和一股不要命的行为通过这些主要与光明为何等的重要的
20181209110801.000|20181209110811.000|ASR_01|就好像五岁的南宁出席广西东中市的一名六十周年庆祝活动中央代表团在机场受到广西各族行动的热烈欢迎
20181209110811.000|20181209110821.000|ASR_01|南宁无锡国际机场洋溢着欢乐喜庆的节庆气氛协调奥运的行动中心感谢一些一平同志为核心的党中要对广西的亲切
20181209110821.000|20181209110831.000|ASR_01|怀等标语格外醒目带领重要的儿童家长的中国医疗队的一美元工业互动的新春的中国主要数据图书记都要同等
20181209110831.000|20181209110841.000|ASR_01|部部长邀请经过人大常委会委员长爱马特里结果这些副主席马的结果这些副主席国家民委主任巴特
20181209110841.000|20181209110851.000|ASR_01|中央纪委委员因为政治工作部主任刘华一同走下前提中央在和成员同期抵达宝鸡中毒的地区地委书记
20181209110851.000|20181209110901.000|ASR_01|新设广西中毒自治区人民就不主席财务等在机场列队欢迎都要在团结号上五层包机离开北京前往南宁
20181209110901.000|20181209110911.000|ASR_01|代表团有六十三人组成成员包括中央和国家机关有关部门军队有关单位其他四个自治区和对口具有广西的广东省
20181209110911.000|20181209110921.000|ASR_01|关注的人的酒后下午中央政治局常委全国政协主席中央代表团团长汪洋率中的
20181209110921.000|20181209110931.000|ASR_01|而偷袭向广西中毒的地区赠送纪念品现在共同性的体式上汪洋和广西中毒的地区地委书记目前
20181209110931.000|20181209110941.000|ASR_01|我会习近平总书记平时特别节目新年总书记建设作为广西公园复兴理想的梯次特点端多大其一人身后
20181209110941.000|20181209110951.000|ASR_01|中国主要的女委员工物业不同的重要的团部团长说出了来这个东西出没总书记十分关心广电广告
20181209110951.000|20181209111001.000|ASR_01|人民为庆祝广西壮族自治区成立六十周年新能提升近一倍广西三河作为民族合睦人民幸福的美好祝福一
20181209111001.000|20181209111011.000|ASR_01|对广西站在新起点创造新辉煌的因此期望力争治理广大中性人及已经开拓进取确保孩子的潜力一般
20181209111011.000|20181209111021.000|ASR_01|国内做足文章同学一道进入全面小康社会创造无愧于时代的新业绩重要的和更多的业
20181209111021.000|20181209111031.000|ASR_01|共计实现体现了中央对广西的一起关怀全国各族人民对广西的申请后议彰显了广西的地方特色和民俗风貌
20181209111031.000|20181209111041.000|ASR_01|全国人大常委会委员长中央代表团组团长爱把气理解读正从纪念品清单中共中央对
20181209111041.000|20181209111051.000|ASR_01|美国国家主席中央代表团团长汪洋酒后下午率代表团部分成员分别和广西军区武警广西总队被误会绝对的
20181209111051.000|20181209111101.000|ASR_01|政法系统代表代表的东阳国务院中央军委代表进行主席大家质疑成都信和亲切问候带给了广西军区
20181209111101.000|20181209111111.000|ASR_01|和作为部队官兵的表示汪洋希望广西新区全体官兵骑摩托车习近平新时代中国特色社会主义思想和党的十九大精神
20181209111111.000|20181209111121.000|ASR_01|有货车电瓶强思想毫不动摇坚持党对经济的绝对领导是行好党和人民给予的新时代实名的未建设正给广西
20181209111121.000|20181209111131.000|ASR_01|为实现两个一百年的目标和中华民族伟大复兴的中国梦作出新的更大贡献在规格五金网系统对官兵的表示汪洋希望
20181209111131.000|20181209111141.000|ASR_01|全体官兵牢固树立三大意识坚决听等指挥坚决维护新兴品种数据核心地位建议维护党中央权威和集中统一领导为有
20181209111141.000|20181209111151.000|ASR_01|和管制因为主西部则是永远作党和给你的东城卫士为维护广西工作团及社会稳定的大好局面再立新功我娘还会
20181209111151.000|20181209111201.000|ASR_01|有的被地区政法系统代表中国中央政治局常委全国政协主席中央代表团团长汪洋最后下午在南宁一生多会
20181209111201.000|20181209111211.000|ASR_01|广西中毒自治区离退休老同志和各族各界人民代表并与他们合影影片重要的还不断长出来有钱把这
20181209111211.000|20181209111221.000|ASR_01|马巴特尔刘华等参加会议活动的消息每天出版的人民日报将发表社论题目是
20181209111221.000|20181209111231.000|ASR_01|写包回大笔买入发展芯片将这类一部广西壮族自治区成立六十周
20181209111231.000|20181209111241.000|ASR_01|老数年我国交通运输业实现跨越是网站服务能力和水平提升成为国民经济快速发展的强有力支撑
20181209111241.000|20181209111251.000|ASR_01|给开启向交通强国迈进的新政今年的述说改革开放数年来看交通人数往往是网站
20181209111251.000|20181209111301.000|ASR_01|二女儿若阿纳和儿子的合
20181209111301.000|20181209111311.000|ASR_01|农民因为一秒钟的爱一马二零五年难以满意我们会利用有的还可能出
20181209111311.000|20181209111321.000|ASR_01|还有的孩子还想要里子的命案阿林阿信母亲的
20181209111321.000|20181209111331.000|ASR_01|外面还有一对母子母女们一样也是一名黑
20181209111331.000|20181209111341.000|ASR_01|如黑马和冷门瑞士洛克到二零一六年共和路人的
20181209111341.000|20181209111351.000|ASR_01|目前是多么的公路还不到位工作迈入一百五十多名当地人感天动
20181209111351.000|20181209111401.000|ASR_01|美女的世界中也有的人家用因为一名职业盗用于是后面有一
20181209111401.000|20181209111411.000|ASR_01|女王也会跑板上面有一个要用一部农民农业
20181209111411.000|20181209111421.000|ASR_01|有男女两名一一名有多名还有一名女孩一名美丽的女友二人没了母亲为什么没
20181209111421.000|20181209111431.000|ASR_01|每天的两百多名美方也有了近二十名作为一名有为交通设施的配置和运用
20181209111431.000|20181209111441.000|ASR_01|有的妈妈也因为一名女性因为我们没有你们两个女人一样没
20181209111441.000|20181209111451.000|ASR_01|那两名面临运营目的又有一名女性人民运动的每一
20181209111451.000|20181209111501.000|ASR_01|大龄青年却还会和二零一零年的二十六名魔女中的两人也生猛
20181209111501.000|20181209111511.000|ASR_01|我们有能力什么名利你们一定会为目前没有通过一定的模式也不一样
20181209111511.000|20181209111521.000|ASR_01|明明卖给你的明星美亚的钢铁动脉铁路项目中
20181209111521.000|20181209111531.000|ASR_01|中国美国两名有摩托运营改善为培养人们明明明明就没有一般女人面
20181209111531.000|20181209111541.000|ASR_01|目的另一方面为妹妹们会养也有名流都会出现
20181209111541.000|20181209111551.000|ASR_01|不用海不沿边的湖南省商业上世纪九十年代就写了一个意外的起落不定又开始感到了
20181209111551.000|20181209111601.000|ASR_01|空警一号创新发展走向世界迈入了上述视频
20181209111601.000|20181209111611.000|ASR_01|实体的表示之后国内的庞大的地盘也有的没的名人卫冕的姑娘们为了不当的农业
20181209111611.000|20181209111621.000|ASR_01|那么同样我们也能应用于铁路机场优势长线建成综合保税区打造开放平均已走廊
20181209111621.000|20181209111631.000|ASR_01|过去一年长沙县引进航空物流高端服务领域重大项目三十多个总投资超过数百亿元而几年前这里经济发展的主力
20181209111631.000|20181209111641.000|ASR_01|还是机械制造市场营运于竞争吉林朝着这个经济结构我们的想法的外地
20181209111641.000|20181209111651.000|ASR_01|到了二零一年之后工程地铁行业市场运行产品具体化的临空顶级海马物流领域完善和严格
20181209111651.000|20181209111701.000|ASR_01|以一名平时被他发现完成给出口总额五十四点五一亿美元同比增长百分之六十二点三如今世界五百强企业中
20181209111701.000|20181209111711.000|ASR_01|三十五家都在这里落户六十年的改革之路往长沙天从一个平凡的现场一路进入全国百强协调地发展
20181209111711.000|20181209111721.000|ASR_01|中西部排名第一提供工业特别是黑体工业占比逐渐扩大十八大以来以现代服务业为主的平盘产业表现突出
20181209111721.000|20181209111731.000|ASR_01|根据公安部统一部署目前山东省公安机关依法侦办的一起严重暴力案件中中周某病
20181209111731.000|20181209111741.000|ASR_01|葛某高等十名嫌疑人被警方依法采取刑事强制措施月四日至七日山东省平
20181209111741.000|20181209111751.000|ASR_01|同时发生以及少数打着可以与人体后的人组织的聚集事件一点一少数人暴力一平打造车辆造成严重人身伤害
20181209111751.000|20181209111801.000|ASR_01|和重大经济损失对社会公共秩序等重大维和二二二六月二十日山东平度人员与某公
20181209111801.000|20181209111811.000|ASR_01|王某军蓝光的策划已经非法集体上榜在当地党委和政府也是劝阻过程中与某东北某军故意碰
20181209111811.000|20181209111821.000|ASR_01|声称遭到殴打后到医院检查两人均为法一与某中国某军蓝宝光王某刚等人借机录制
20181209111821.000|20181209111831.000|ASR_01|不不是视频山东平度人员十月五日七时五十分级制又于东湖青岛和山东其他地市及安徽
20181209111831.000|20181209111841.000|ASR_01|江苏河南河北辽宁等小学部分人员陆续到达平度市人民会堂广场聚集人员约三百人并将不会
20181209111841.000|20181209111851.000|ASR_01|国际一部分的工具运到现场一点何某王某李某进等人通过现场演说微信群等方式让学生
20181209111851.000|20181209111901.000|ASR_01|调和暴力犯罪和特有的文化获得的养女儿没有美国的球队也有了周
20181209111901.000|20181209111911.000|ASR_01|干部的具体工作市委市政府先后派行政干部主动共同对话但具体人员提出必须对作用
20181209111911.000|20181209111921.000|ASR_01|平度声援人员给与补偿的模拟足球并对于女排长十月六日十四时根本周某中
20181209111921.000|20181209111931.000|ASR_01|某冰和某包养某星的疯狂实施暴力袭警打砸车辆的行为一时十一分钟共导致二十四名执勤民警
20181209111931.000|20181209111941.000|ASR_01|品种受伤造成经济损失达八百二十五万元的过程中现场民警其中规范的明确化保持了
20181209111941.000|20181209111951.000|ASR_01|请客或主管的行业一手数有微网站为黑客人员是和暴力一名大巴车辆的主要运作
20181209111951.000|20181209112001.000|ASR_01|周某郑某因和某方等四人已被机关分别以涉嫌妨害公务罪故意伤害罪一动乱公共
20181209112001.000|20181209112011.000|ASR_01|我只学会平静是对的一把牌局行为强制措施公安机关表示事件是极少数不法分子表示国
20181209112011.000|20181209112021.000|ASR_01|与挑战执法权威由组织游客划有预谋实施的一起严重暴力犯罪案件对任何北把犯罪行为
20181209112021.000|20181209112031.000|ASR_01|机关都将坚决依法予以打击切实维护社会正常秩序切实保障人民群众生命财产安全
20181209112031.000|20181209112041.000|ASR_01|本来消息明天出版的人民日报将刊发表有的还对往日工业维护合法权益我们说一声
20181209112041.000|20181209112051.000|ASR_01|是党和国家的宝贵财富是建设中国特色社会主义的重要力量维护军人军属合法权益做好退役军人服务保证
20181209112051.000|20181209112101.000|ASR_01|作关系国防和军队建设国企改革发展稳定全局关系人民群众切身利益而言军人转业到地方
20181209112101.000|20181209112111.000|ASR_01|安置好也有使用地板会好他们的作用绝大多数年轻人都得以理性合法的形式表达诉求
20181209112111.000|20181209112121.000|ASR_01|但也必须看到有极少数人一些从事违法犯罪活动甚至借助楼市保密系统呈现出有组织
20181209112121.000|20181209112131.000|ASR_01|壁画形象有些东西关路是当地国宝对于公平正义要问是不是严重损害可以真人的形象
20181209112131.000|20181209112141.000|ASR_01|损害的人民群众切身利益已经成为影响一些地方社会稳定的突出因素引起广大人民群众的强烈不满近日
20181209112141.000|20181209112151.000|ASR_01|有关部们对于山东平度等给极少数打着退役军人其后奥利金等违法犯罪行为依法采取行动顺应了广
20181209112151.000|20181209112201.000|ASR_01|对群众意愿我们是社会主义法治国家法制维护人民权益维护社会公平正义维护国家安全稳定
20181209112201.000|20181209112211.000|ASR_01|重要保障法治社会不同的宝贝被定岗为病人谁都不能为所欲为班级
20181209112211.000|20181209112221.000|ASR_01|纪录片我们一起走过改革开放和中日晚继续在央视综合频道八点档播出两年一十五平均价格最低的
20181209112221.000|20181209112231.000|ASR_01|旅游情人心情展现广大官兵为实现党在新时代的成就目标把人民军队全面退出世界一流绝对不是奋斗的一
20181209112231.000|20181209112241.000|ASR_01|第十六集我的中国心讲述高台作为引进外资人才的活动房为改革开放作出的贡献呈现异国两地的成功
20181209112241.000|20181209112251.000|ASR_01|让两岸三地中国梦品牌行列国家统计局一公布数据显示十月份我国物价总体保持平稳
20181209112251.000|20181209112301.000|ASR_01|还是全国居民消费价格同比上涨百分之二二涨幅比上月回落了三个百分点外工业品方面十一月份全国工业生产者
20181209112301.000|20181209112311.000|ASR_01|出厂价格同比上涨百分之二点七第五届中国工业大奖地点在北京一共有十二家企业和不幸好中国标准动车组的
20181209112311.000|20181209112321.000|ASR_01|是一个项目获中国工业大奖与往届相比新兴产业高技术产业或到企业和项目大幅提升占到总数的一半中国工业
20181209112321.000|20181209112331.000|ASR_01|表示经国务院批准设立的我国工业领域的最高奖项每两年评选表彰一次秘密账户由昆仑山与上街买
20181209112331.000|20181209112341.000|ASR_01|五点组成的海军第十一批护航编队从广东站将其行以榜样应该是马里海域也比比分视频行业队执行
20181209112341.000|20181209112351.000|ASR_01|五月五日二十二年以来海军已经派出一百多次建平两万多名国民很大护航任务历时一月中国运营网上
20181209112351.000|20181209112401.000|ASR_01|我们的李鸿海一度引领着我拥有的影响无定名也变了我拥有能将一个模型里有也失败
20181209112401.000|20181209112411.000|ASR_01|村的主演陈明获得优秀面临将回顾和行动便会大夜化名很可能是营业我又有何用
20181209112411.000|20181209112421.000|ASR_01|大型文博派主题目标张宝的病儿已将于今晚在央视一频道播出首集节目中故宫博物院将协样式雷建筑
20181209112421.000|20181209112431.000|ASR_01|将李白书上阳台帖经铁路职工用户被申请国应向多位演员让作为国宝守护人还是朋友
20181209112431.000|20181209112441.000|ASR_01|国家保障的信息进行公文运营我们来共同
20181209112441.000|20181209112451.000|ASR_01|乘客饱受教于时候指数配置物流领域的深刻改革的媒体位置误
20181209112451.000|20181209112501.000|ASR_01|友好条约有一定的象征意义乌克兰也会于本月六号已通过决议终止无有要约也有九九年生效的物有很多约承认
20181209112501.000|20181209112511.000|ASR_01|树林时期就已划定的两国边界东西通过和平方式解决一切争端多配备一部绿色园引入会者会员的话说这个有
20181209112511.000|20181209112521.000|ASR_01|由于是以老母生病问题和两国战略合作的关键业务退出跳远的决定将进一步损害两国关系一些大业主
20181209112521.000|20181209112531.000|ASR_01|说位置友好条约将给两国关系制造法律俄罗斯与乌克兰上个月二十五号在克海洋发生冲突乌克兰海军的
20181209112531.000|20181209112541.000|ASR_01|柳传志集团上大致四人被格方扣留乌克兰也部部分地区进度战争状态帮忙俄罗斯礼包具体指出要是女顾客老人面临的是
20181209112541.000|20181209112551.000|ASR_01|行为指控不能被视为战书这对安的刑事调查已经提交到我们的电波安全局总部而且俄罗斯与乌克兰也没有出
20181209112551.000|20181209112601.000|ASR_01|于是同图或网站中国期货变化大会正在波兰卡会举行有一个人是肯定中国第一部
20181209112601.000|20181209112611.000|ASR_01|对气候变化问题并取得成效国际变化框架公约秘书处执行秘书安徽披露大巴后再回
20181209112611.000|20181209112621.000|ASR_01|本报记者提问时对中国采取的减排措施表示肯定俄文和德国未降低碳还本所多出的能力得到了越来越
20181209112621.000|20181209112631.000|ASR_01|中国设计成为了一年能源领域的领军国家也能够与六个国很好他们的优秀经验中国友人亦有明确的优惠
20181209112631.000|20181209112641.000|ASR_01|可持续发展和应对气候变化的未来大学的合并为日后对付因为他着迷于东方十一月发布的报告显示
20181209112641.000|20181209112651.000|ASR_01|二一四年中国单位国内生产总值二氧化碳排放比二零零五年下降约百分之四十六日一天三年完成的一定目标兑现了对国际
20181209112651.000|20181209112701.000|ASR_01|给的承诺我们之后变化会议主席波兰环境部副部长一行然后在地上采访时对中国弟弟努力优化本国能源
20181209112701.000|20181209112711.000|ASR_01|机构在起动中的一个宏大又切实可行的能源政策很重要中国已公布了多个相关同时通过
20181209112711.000|20181209112721.000|ASR_01|母女的秘密和能源使用表明我们对世界影响最大的日报的引领作用的王后伊朗总统鲁哈尼改革未来
20181209112721.000|20181209112731.000|ASR_01|一场国际会议上发的讲话也给美国对伊朗的支配行为是经济恐怖主义目的就是制造恐慌打击投资者对以往的信用
20181209112731.000|20181209112741.000|ASR_01|你还能说这开伊朗将影响地铁安全的病特别伊朗外长要以后也表示美国向中东地区出口的武器已经远远超过对比
20181209112741.000|20181209112751.000|ASR_01|的需求正在把这一地区变成火药同王某应当选择德国主要给人们一不要运动联盟主
20181209112751.000|20181209112801.000|ASR_01|股他们鲍尔网的文化与加强党的团结和不可保的一名一名蒙面临某名老人一百个人一名网名
20181209112801.000|20181209112811.000|ASR_01|男女这一灵活的通过在基民盟内部秘书长一职仅次于帮主任面对的是马克与阿伯他们来自
20181209112811.000|20181209112821.000|ASR_01|不同阵营的人们可以表明客户的抱怨与李敏和党的问题为帮助还和年轻成员更长了名人
20181209112821.000|20181209112831.000|ASR_01|一位国防军方说已对当地发现了一条真主党酒店的同一般的也正是一系列的法定地方一般的政府与
20181209112831.000|20181209112841.000|ASR_01|对你还说当天已经在预定地域向昆明一个一般的真主党的目标人员开国一列从本月初巴西对行动
20181209112841.000|20181209112851.000|ASR_01|全部已经摧毁了两条伟大的制度党的跨境地道不过一般的推算五号表示作为真主党地到底是一样的说法
20181209112851.000|20181209112901.000|ASR_01|韩国政府做好全部针对西部海域边界附近区域的运河水路调查与当地技术调查团找到了可供传播行
20181209112901.000|20181209112911.000|ASR_01|航道汉江和临江路海口水域有很快的利用与上月政治处临时航班随后转向民间传播开
20181209112911.000|20181209112921.000|ASR_01|当地时间起好事夜位于墨西哥首都墨西哥城附近的波布卡不被困火灾接连两个大
20181209112921.000|20181209112931.000|ASR_01|沪上口喷出的火车会和整体在业务上发出要演的白宫获胜被当时形成的火车会生成到两千米的高空在落到火
20181209112931.000|20181209112941.000|ASR_01|中美体育文法没有造成人员伤亡今年的宁波宝宝网的协助更多新利益你还可以关注央视新闻
20181209112941.000|20181209112951.000|ASR_01|网面对民警说两名门将没有文明没有文明没
20181209112951.000|20181209113001.000|ASR_01|为了老人的二儿
20181209113001.000|20181209113011.000|ASR_01|二
20181209113011.000|20181209113021.000|ASR_01|要有一名迷一名用户跳离应提供
20181209113021.000|20181209113031.000|ASR_01|进入母乳应运用领会
20181209113031.000|20181209113041.000|ASR_01|因为迷宫会有数名艺人有
20181209113041.000|20181209113051.000|ASR_01|人民网友们上演名为米亚
20181209113051.000|20181209113101.000|ASR_01|因为我们的一位人士的响应一名运营人员
20181209113101.000|20181209113111.000|ASR_01|民有民营民有民营运营中的网民中
20181209113111.000|20181209113121.000|ASR_01|网络应用的网路用户想用母亲的名
20181209113121.000|20181209113131.000|ASR_01|原因一没有目的的
20181209113131.000|20181209113141.000|ASR_01|如果一个网络一个运动项目奥伟美二人体的一定有那个人也
20181209113141.000|20181209113151.000|ASR_01|我们的目的没有人会认为是一名一名一物一名一名一名
20181209113151.000|20181209113201.000|ASR_01|能够拥有一名幼儿园
20181209113201.000|20181209113211.000|ASR_01|有木有有木有有木有某个领域的应用
20181209113211.000|20181209113221.000|ASR_01|我们没有把外文名英文名和定义有欧美明星名人
20181209113221.000|20181209113231.000|ASR_01|因为有你有你们的
20181209113231.000|20181209113241.000|ASR_01|能容人容物容易与一般的信息安
20181209113241.000|20181209113251.000|ASR_01|文人一起打中一名儿童身体变化
20181209113251.000|20181209113301.000|ASR_01|农民人
20181209113301.000|20181209113311.000|ASR_01|老子爱民如子的努力两年多的女友
20181209113311.000|20181209113321.000|ASR_01|小女孩我宁可永远也不可能顺利引入零度等温保
20181209113321.000|20181209113331.000|ASR_01|那么我们来进行的农民还没有完全过去补充的空气又前来助阵未来两三天内蒙古中西部的北部华北黄淮江淮
20181209113331.000|20181209113341.000|ASR_01|会话体会导游机的配备更新后也会下降和导流洞及对角的和的会超过十度并为我国中东部的气温偏低的情况而仍然的
20181209113341.000|20181209113351.000|ASR_01|持续一段时间区间内运行联动线未来将会对它们的北部到贵州的中国这一现实的最高气温同样也是很难不大一名后
20181209113351.000|20181209113401.000|ASR_01|华北京的最高气温大多的都是只有一二个而将难于普遍不会超过一部要到时候才不会有后面被安全开的情况还会
20181209113401.000|20181209113411.000|ASR_01|有大小每方面要晚到明天我国依然是南多北少的格局不过的变化大于教学的区域会向北转移到湖北的一部
20181209113411.000|20181209113421.000|ASR_01|男人还有安徽的大部项目的西部和北部山东东南部和外观的中西部湖南西部一带还不下去等于没买到后天河
20181209113421.000|20181209113431.000|ASR_01|山东安徽江苏北部这一带的还会出现与大学或药学人物东莞公安会的北部江苏北部还有当时出现而南方大范围的阴雨
20181209113431.000|20181209113441.000|ASR_01|汇集世界上又有十二月以来南方多地都是教学会有频繁的特别是长江中下一百二十日南昌杭州上海
20181209113441.000|20181209113451.000|ASR_01|降水运输普遍都达到了把整片红红的是要贯穿整个行情不过罗森南方降水就会被围的教练现在的给了有望
20181209113451.000|20181209113501.000|ASR_01|明显明白阳光想了解更多天气信息你可以将屏幕上方的密码我们广州城市人群
20181209113501.000|20181209113511.000|ASR_01|什么多名宁夏水的运水工二零零二十一到零下十度长春和影像是一个留下一
20181209113511.000|20181209113521.000|ASR_01|多名明师的生母母母母母当儿童模特提名二十九的一手
20181209113521.000|20181209113531.000|ASR_01|没有进行影像希望的印象十分凌晨影响上到零下五度说明多空双
20181209113531.000|20181209113541.000|ASR_01|不到一个完整的多名名下拥有的灵动与民间多为林木果木木妈妈什么都
20181209113541.000|20181209113551.000|ASR_01|宁夏中部的巴东人人都能因奥运的奥运从树木的
20181209113551.000|20181209113601.000|ASR_01|养小女儿的耳朵的耳朵莫名莫名失踪二百多年人还是可能会
20181209113601.000|20181209113611.000|ASR_01|十三中的多名女网友的男友和好友的女儿郑州都没有水
20181209113611.000|20181209113621.000|ASR_01|儿童和北上水路岛和安子女向学生中的密度上海小女
20181209113621.000|20181209113631.000|ASR_01|武汉的小雨林木的木门
20181209113631.000|20181209113641.000|ASR_01|小武公岛务工杭州小雪中的欧洲小屋十二到十八个
20181209113641.000|20181209113651.000|ASR_01|海北小雨时就到一农民南宁小颖一保障二东海口小雨小到牙
20181209113651.000|20181209113701.000|ASR_01|杭州小米和阿宝说我送了多名黑魔术苏澳门十二月十六日
20181209113701.000|20181209113711.000|ASR_01|运用名人名言人文
20181209113711.000|20181209113721.000|ASR_01|我想你肯定会的德林都没人应
20181209113721.000|20181209113731.000|ASR_01|因为这样的一位母亲的女人都会有
20181209113731.000|20181209113741.000|ASR_01|并被
20181209113741.000|20181209113751.000|ASR_01|母母母母母母母母没进行多元
20181209113751.000|20181209113801.000|ASR_01|打印输出的运营运营运营
20181209113801.000|20181209113811.000|ASR_01|也有人把你
20181209113811.000|20181209113821.000|ASR_01|一一一一一一一一一一一一一一一一一一
20181209113821.000|20181209113831.000|ASR_01|一一一一一一一一一一一一
20181209113831.000|20181209113841.000|ASR_01|你和一个女人和这个成品
20181209113841.000|20181209113851.000|ASR_01|一一一一一一一一一一一一一一一
20181209113851.000|20181209113901.000|ASR_01|平行的另一名
20181209113901.000|20181209113911.000|ASR_01|原为一名秘密男女
20181209113911.000|20181209113921.000|ASR_01|而我的女儿我们是我们
20181209113921.000|20181209113931.000|ASR_01|有一密码门要有一名小人物
20181209113931.000|20181209113941.000|ASR_01|老处女的行为都有一个红
20181209113941.000|20181209113951.000|ASR_01|人民网
20181209113951.000|20181209114001.000|ASR_01|老王
END|20181209113954|2018-12-09_1100_CN_CCTV1_新闻联播
```

## Acknowledgments
* [Google Summer of Code 2018](https://summerofcode.withgoogle.com/)
* [Red Hen Lab](http://www.redhenlab.org/)
* [DeepSpeech2 on PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech)
* [DeepSpeech2 Paper](http://proceedings.mlr.press/v48/amodei16.pdf)
