#!/usr/bin/env bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                              Switchboard                                  "
echo ============================================================================

stage=0
stop_stage=5
gpu=
benchmark=true
deterministic=false
pin_memory=false
speed_perturb=true  # default
stdout=false
wandb_id=""
corpus=swbd

### vocabulary
unit=wp           # word/wp/word_char
vocab=10000
wp_type=bpe       # bpe/unigram (for wordpiece)
unit_sub1=char
wp_type_sub1=bpe  # bpe/unigram (for wordpiece)
vocab_sub1=1000
unit_sub2=char
wp_type_sub2=bpe  # bpe/unigram (for wordpiece)
vocab_sub2=300

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las_3mtl.yaml
conf2=
asr_init=

### path to save the model
model=/n/work2/inaguma/results/${corpus}

### path to the model directory to resume training
resume=

### path to save preproecssed data
export data=/n/work2/inaguma/corpus/${corpus}

### path to original data
SWBD_AUDIOPATH=/n/rd21/corpora_7/swb
EVAL2000_AUDIOPATH=/n/rd21/corpora_7/hub5_english/LDC2002S09
EVAL2000_TRANSPATH=/n/rd21/corpora_7/hub5_english/LDC2002T43
RT03_PATH=
FISHER_PATH=/n/rd7/fisher_english

### data size
datasize=swbd

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_nodev_${datasize}
dev_set=dev
test_set="eval2000"
if [ ${speed_perturb} = true ]; then
    train_set=train_nodev_sp_${datasize}
    dev_set=dev_sp
    test_set="eval2000_sp"
fi

# main
if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi
# sub1
if [ ${unit_sub1} = char ]; then
    vocab_sub1=
fi
if [ ${unit_sub1} != wp ]; then
    wp_type_sub1=
fi
# sub2
if [ ${unit_sub2} = char ] || [ ${unit_sub2} = phone ]; then
    vocab_sub2=
fi
if [ ${unit_sub2} != wp ]; then
    wp_type_sub2=
fi

use_wandb=false
if [ ! -z ${wandb_id} ]; then
    use_wandb=true
    wandb login ${wandb_id}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # prepare swbd data and put it under data/train_swbd
    local/swbd1_data_download.sh ${SWBD_AUDIOPATH} || exit 1;
    local/swbd1_prepare_dict.sh || exit 1;
    local/swbd1_data_prep.sh ${SWBD_AUDIOPATH} || exit 1;

    # prepare fisher data and put it under data/train_fisher
    local/fisher_data_prep.sh ${FISHER_PATH}
    local/fisher_swbd_prepare_dict.sh
    utils/fix_data_dir.sh ${data}/train_fisher

    # nomalization
    cp ${data}/train_fisher/text ${data}/train_fisher/text.tmp.0
    cut -f 2- -d " " ${data}/train_fisher/text.tmp.0 | \
        sed -e 's/\[laughter\]-/[laughter]/g' | \
        sed -e 's/\[noise\]-/[noise]/g' > ${data}/train_fisher/text.tmp.1
    paste -d " " <(cut -f 1 -d " " ${data}/train_fisher/text.tmp.0) \
        <(cat ${data}/train_fisher/text.tmp.1) > ${data}/train_fisher/text
    rm ${data}/train_fisher/text.tmp*

    # eval2000
    local/eval2000_data_prep.sh ${EVAL2000_AUDIOPATH} ${EVAL2000_TRANSPATH} || exit 1;
    [ ! -z ${RT03_PATH} ] && local/rt03_data_prep.sh ${RT03_PATH}

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_${datasize}_spfalse ]; then
        for x in train_swbd eval2000; do
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
            utils/fix_data_dir.sh ${data}/${x}
        done

        # Use the first 4k sentences as dev set.
        utils/subset_data_dir.sh --first ${data}/train_swbd 4000 ${data}/dev || exit 1;  # 5hr 6min
        n=$[$(cat ${data}/train_swbd/segments | wc -l) - 4000]
        utils/subset_data_dir.sh --last ${data}/train_swbd ${n} ${data}/${train_set}.tmp || exit 1;

        # Finally, the full training set:
        utils/data/remove_dup_utts.sh 300 ${data}/${train_set}.tmp ${data}/train_nodev_swbd || exit 1;  # 286hr
        rm -rf ${data}/*.tmp

        if [ ${datasize} = fisher_swbd ]; then
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/train_fisher ${data}/log/make_fbank/train_fisher ${data}/fbank || exit 1;
            utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} ${data}/train_nodev_swbd ${data}/train_fisher || exit 1;
        fi
    fi

    if [ ${speed_perturb} = true ]; then
        speed_perturb_3way.sh ${data} train_nodev_${datasize} ${train_set}
        cp -rf ${data}/dev ${data}/${dev_set}
        cp -rf ${data}/eval2000 ${data}/eval2000_sp
    fi

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${datasize} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1_${datasize}_sp${speed_perturb} && echo "Finish feature extranction (stage: 1)."
fi

# main
dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
nlsyms=${data}/dict/nlsyms_${train_set}.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, main)                  "
    echo ============================================================================

    echo "make a non-linguistic symbol list"
    cut -f 2- -d " " ${data}/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
    cat ${nlsyms}

    make_vocab.sh --unit ${unit} --nlsyms ${nlsyms} --speed_perturb ${speed_perturb} \
        --vocab ${vocab} --wp_type ${wp_type} --wp_model ${wp_model} \
        ${data} ${dict} ${data}/${train_set}/text || exit 1;

    # normalize eval2000
    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    paste -d " " <(awk '{print $1}' ${data}/${test_set}/text) <(cat ${data}/${test_set}/text | cut -f 2- -d " " | awk '{ print tolower($0) }' | \
        perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g') \
        > ${data}/${test_set}/text.tmp
    mv ${data}/${test_set}/text.tmp ${data}/${test_set}/text

    grep -v '^en_' ${data}/${test_set}/text > ${data}/${test_set}/text.swbd
    grep -v '^sw_' ${data}/${test_set}/text > ${data}/${test_set}/text.ch

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
        for x in ${train_set} ${dev_set}; do
            cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${x}_${datasize}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${x}_${datasize}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab}_${datasize}.txt || exit 1;
            # NOTE: speed perturbation is not considered
        done
        for set in "swbd" "ch"; do
            cut -f 2- -d " " ${data}/${test_set}/text.${set} | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${test_set}_${set}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${test_set}_${set}.txt ${dict} ${test_set}_${set} \
                >> ${data}/dict/oov_rate/word${vocab}_${datasize}.txt || exit 1;
        done
        cat ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
        ${data}/${train_set} ${dict} > ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

# sub1
dict_sub1=${data}/dict/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.txt
wp_model_sub1=${data}/dict/${train_set}_${wp_type_sub1}${vocab_sub1}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit_sub1}${wp_type_sub1}${vocab_sub1}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, sub1)                  "
    echo ============================================================================

    make_vocab.sh --unit ${unit_sub1} --nlsyms ${nlsyms} --speed_perturb ${speed_perturb} \
        ${data} ${dict_sub1} ${data}/${train_set}/text || exit 1;
    # NOTE: bpe is not supported here
    # --vocab ${vocab_sub1} --wp_type ${wp_type_sub1} --wp_model ${wp_model_sub1} \

    echo "Making dataset tsv files for ASR ..."
    make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit_sub1} --nlsyms ${nlsyms} --wp_model ${wp_model_sub1} \
        ${data}/${train_set} ${dict_sub1} > ${data}/dataset/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit_sub1} --nlsyms ${nlsyms} --wp_model ${wp_model_sub1} \
            ${data}/${x} ${dict_sub1} > ${data}/dataset/${x}_${datasize}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${datasize}_${unit_sub1}${wp_type_sub1}${vocab_sub1}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

# sub2
dict_sub2=${data}/dict/${train_set}_${unit_sub2}${wp_type_sub2}${vocab_sub2}.txt
wp_model_sub2=${data}/dict/${train_set}_${wp_type_sub2}${vocab_sub2}
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit_sub2}${wp_type_sub2}${vocab_sub2}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, sub2)                  "
    echo ============================================================================

    if [ ${unit_sub2} = phone ]; then
        lexicon=${data}/local/dict_nosp/lexicon.txt
        unk=spn
        for x in ${train_set} ${dev_set} ${test_set}; do
            map2phone.py --text ${data}/${train_set}/text --lexicon ${lexicon} --unk ${unk} > ${data}/${train_set}/text.phone
        done
        make_vocab.sh --unit ${unit_sub2} --speed_perturb true \
            ${data} ${dict_sub2} ${data}/${train_set}/text.phone || exit 1;
    else
        make_vocab.sh --unit ${unit_sub2} --nlsyms ${nlsyms} --speed_perturb ${speed_perturb} \
            ${data} ${dict_sub2} ${data}/${train_set}/text || exit 1;
        # NOTE: bpe is not supported here
    fi

    echo "Making dataset tsv files for ASR ..."
    if [ ${unit_sub2} = phone ]; then
        text="text.phone"
    else
        text="text"
    fi
    make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit_sub2} --wp_model ${wp_model_sub2} --text ${data}/${train_set}/${text} \
        ${data}/${train_set} ${dict_sub2} > ${data}/dataset/${train_set}_${unit_sub2}${wp_type_sub2}${vocab_sub2}.tsv || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit_sub2} --nlsyms ${nlsyms} --wp_model ${wp_model_sub2} --text ${data}/${x}/${text}  \
            ${data}/${x} ${dict_sub2} > ${data}/dataset/${x}_${datasize}_${unit_sub2}${wp_type_sub2}${vocab_sub2}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${datasize}_${unit_sub2}${wp_type_sub2}${vocab_sub2}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

mkdir -p ${model}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    export OMP_NUM_THREADS=${n_gpus}
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
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --train_set_sub1 ${data}/dataset/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv \
        --train_set_sub2 ${data}/dataset/${train_set}_${unit_sub2}${wp_type_sub2}${vocab_sub2}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --dev_set_sub1 ${data}/dataset/${dev_set}_${datasize}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv \
        --dev_set_sub2 ${data}/dataset/${dev_set}_${datasize}_${unit_sub2}${wp_type_sub2}${vocab_sub2}.tsv \
        --unit ${unit} \
        --unit_sub1 ${unit_sub1} \
        --unit_sub2 ${unit_sub2} \
        --nlsyms ${nlsyms} \
        --dict ${dict} \
        --dict_sub1 ${dict_sub1} \
        --dict_sub2 ${dict_sub2} \
        --wp_model ${wp_model}.model \
        --wp_model_sub1 ${wp_model_sub1}.model \
        --wp_model_sub2 ${wp_model_sub2}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi