#!/usr/bin/env bash
# zliu-elliot   iai.xmu.edu.cn
echo ============================================================================
echo "                               ChinaPopSong                               "
echo ============================================================================

# 全局变量
stage=0
stop_stage=2
unit=char
corpus=popsong
export data=/home/zliu-elliot/dataset/PopSong

train_set="train_sp"
dev_set="dev_sp"
test_set="test_sp"

# 前置脚本
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
  echo ============================================================================
  echo "                           数据预处理 (stage:0)                            "
  echo ============================================================================
  ./local/pop_data_prep.sh ${data}/wav ${data}/transcript

  # remove space in text
  for x in train dev test; do
      cp ${data}/${x}/text ${data}/${x}/text.org
      paste -d " " <(cut -f 1 -d" " ${data}/${x}/text.org) <(cut -f 2- -d" " ${data}/${x}/text.org | tr -d " ") > ${data}/${x}/text
      rm ${data}/${x}/text.org
  done

  touch ${data}/.done_stage_0 && echo "数据预处理已完成。"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train dev test; do
        # train_cmd被定义为run.pl -mem 2G（用于指定占用内存的上限）
        echo "开始$x"
        steps/make_fbank.sh --nj 32 --cmd "run.pl --mem 2G" --write_utt2num_frames true ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
        utils/fix_data_dir.sh ${data}/${x}
    done

    speed_perturb_3way.sh ${data} train ${train_set}
    echo "speed_perturb_3way结束"
    cp -rf ${data}/dev ${data}/${dev_set}
    cp -rf ${data}/test ${data}/${test_set}

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;
    echo "compute-cmvn-stats结束"

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 32 ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1 && echo "特征提取已完成。"
fi

dict=${data}/dict/${train_set}.txt;
mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2 ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    make_vocab.sh --unit ${unit} --speed_perturb true \
      ${data} ${dict} ${data}/${train_set}/text || exit 1;

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} ${data}/${x} ${dict} > ${data}/dataset/${x}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2 && echo "Finish creating dataset for ASR (stage: 2)."
fi


