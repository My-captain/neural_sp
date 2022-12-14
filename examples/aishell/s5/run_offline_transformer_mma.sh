#!/usr/bin/env bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                AISHELL-1                                 "
echo ============================================================================

stage=0
stop_stage=2
gpu=1
benchmark=true
deterministic=false
pin_memory=false
stdout=false
wandb_id=""
corpus=aishell1

### vocabulary
unit=char

#########################
# ASR configuration
#########################
conf=conf/asr/transformer_mma_hie_subsample8_ma4H_ca4H_w16_from4L.yaml
conf2=
asr_init=
external_lm=

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=/media/zliu-elliot/Jarvis/neural_sp/transformer_mma_hie_subsample8_ma4H_ca4H_w16_from4L/${corpus}

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/media/zliu-elliot/Jarvis/aishell

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_sp
dev_set=dev_sp
test_set="test_sp"

use_wandb=false
if [ ! -z ${wandb_id} ]; then
    use_wandb=true
    wandb login ${wandb_id}
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${data}
    local/download_and_untar.sh ${data} "www.openslr.org/resources/33" data_aishell
    local/download_and_untar.sh ${data} "www.openslr.org/resources/33" resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    echo ${data}/data_aishell/wav
    echo ${data}/data_aishell/transcript

    local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
    # remove space in text
    for x in train dev test; do
        cp ${data}/${x}/text ${data}/${x}/text.org
        paste -d " " <(cut -f 1 -d" " ${data}/${x}/text.org) <(cut -f 2- -d" " ${data}/${x}/text.org | tr -d " ") \
            > ${data}/${x}/text
        rm ${data}/${x}/text.org
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train dev test; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
        utils/fix_data_dir.sh ${data}/${x}
    done

    speed_perturb_3way.sh ${data} train ${train_set}
    cp -rf ${data}/dev ${data}/${dev_set}
    cp -rf ${data}/test ${data}/${test_set}

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1
fi

dict=${data}/dict/${train_set}.txt; mkdir -p ${data}/dict
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
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2 && echo "Finish creating dataset for ASR (stage: 2)."
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    export OMP_NUM_THREADS=${n_gpus}
    echo "CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
        ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --train_set ${data}/dataset/${train_set}.tsv \
        --dev_set ${data}/dataset/${dev_set}.tsv \
        --eval_sets ${data}/dataset/${test_set}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --model_save_dir ${model}/lm \
        --stdout ${stdout} \
        --resume ${lm_resume} || exit 1;"
#    CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
#        ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py --local_world_size=${n_gpus} \
#        --corpus ${corpus} \
#        --config ${lm_conf} \
#        --n_gpus ${n_gpus} \
#        --cudnn_benchmark ${benchmark} \
#        --cudnn_deterministic ${deterministic} \
#        --train_set ${data}/dataset/${train_set}.tsv \
#        --dev_set ${data}/dataset/${dev_set}.tsv \
#        --eval_sets ${data}/dataset/${test_set}.tsv \
#        --unit ${unit} \
#        --dict ${dict} \
#        --model_save_dir ${model}/lm \
#        --stdout ${stdout} \
#        --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    export OMP_NUM_THREADS=${n_gpus}

    echo "CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
        ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --use_wandb ${use_wandb} \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --pin_memory ${pin_memory} \
        --train_set ${data}/dataset/${train_set}.tsv \
        --dev_set ${data}/dataset/${dev_set}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --external_lm ${external_lm} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;"
    CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
        ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --use_wandb ${use_wandb} \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --pin_memory ${pin_memory} \
        --train_set ${data}/dataset/${train_set}.tsv \
        --dev_set ${data}/dataset/${dev_set}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --external_lm ${external_lm} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
