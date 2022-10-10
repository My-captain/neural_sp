#!/bin/bash

#   生成所有utt的绝对路径>> local/tmp/wav.flist
#   最终生成spk2utt utt2spk wav.scp text输出到AiShell/对应的train/dev目录下

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <audio-path> <text-path>"
  echo " $0 /export/a05/xna/data/data_aishell/wav /export/a05/xna/data/data_aishell/transcript"
  exit 1;
fi

aishell_audio_dir=$1
aishell_text=$2/aishell_transcript_v0.8.txt

train_dir=${data}/local/train
dev_dir=${data}/local/dev
test_dir=${data}/local/test
tmp_dir=${data}/local/tmp
mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir
mkdir -p $tmp_dir

# data directory check
if [ ! -d $aishell_audio_dir ] || [ ! -f $aishell_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
# 将所有音频的绝对路径整合到local/tmp/wav.flist
find $aishell_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 141925 ] && \
  echo Warning: expected 141925 data data files, found $n

# 搜索出wav.flist中对应数据集的前缀，并分流到对应的数据集中
grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

#rm -r $tmp_dir

# Transcriptions preparation
for dir in $train_dir $dev_dir $test_dir; do
  echo Preparing $dir transcriptions
  #  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}'
  # sed -e(expression)用于替换.wav后缀
  # awk -F用于指定分隔符，$NF表示最后一个字段

  # 提取出所有utt的id序列 >> local/train/utt.list
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
  # utt_id与speaker的对应关系 >> local/train/utt2spk_all
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
  # 生成utt-id与绝对路径对应关系 >> local/train/wav.scp_all
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  # 根据utt-id序列，筛选出utt-id对应的转录文本 >> local/train/transcripts.txt
  utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
  # 重写utt.list（有可能部分utt的转录文本缺失）
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  # 筛选出utt-id与speaker的对应关系 >> local/train/utt2spk
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
  # 生成utt-id与绝对路径对应关系 >> local/train/wav.scp
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  # 去重后输出utt-id对应的转录文本 >> local/train/text
  sort -u $dir/transcripts.txt > $dir/text
  # 生成speaker-id与utt-id列表 >> local/train/spk2utt
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p ${data}/train ${data}/dev ${data}/test

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f ${data}/train/$f || exit 1;
  cp $dev_dir/$f ${data}/dev/$f || exit 1;
  cp $test_dir/$f ${data}/test/$f || exit 1;
done

echo "$0: AISHELL data preparation succeeded"
exit 0;
