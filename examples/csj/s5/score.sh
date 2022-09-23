#!/usr/bin/env bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
model1=
model2=
model3=
model_bwd=
gpu=
stdout=false
n_threads=1
eval_set="dev eval1 eval2 eval3"

### path to save preproecssed data
data=/n/work2/inaguma/corpus/csj

unit=
metric=edit_distance
batch_size=1
beam_width=10
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.0
length_norm=true  ###
coverage_penalty=0.0
coverage_threshold=0.0
gnmt_decoding=false
eos_threshold=1.0
lm=
lm_second=
lm_bwd=
lm_weight=0.3
lm_second_weight=0.3
lm_bwd_weight=0.3
ctc_weight=0.0  # 1.0 for joint CTC-attention means decoding with CTC
softmax_smoothing=1.0
resolving_unk=false
fwd_bwd_attention=false
bwd_attention=false
reverse_lm_rescoring=false
asr_state_carry_over=false
lm_state_carry_over=true
n_average=10  # for Transformer
oracle=false
longform_max_n_frames=0
mma_delay_threshold=-1  # for MMA

# for streaming
streaming_encoding=false
block_sync=false
block_size=40
lc_chunk_size_left=-1
lc_chunk_size_current=-1
vad_free=false
blank_threshold=40  # 400ms
spike_threshold=0.1
n_accum_frames=1600  # 16s

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    # CPU
    n_gpus=0
    export OMP_NUM_THREADS=${n_threads}
else
    n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
fi

for set in ${eval_set}; do
    recog_dir=$(dirname ${model})/decode_${set}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}
    if [ ! -z ${unit} ]; then
        recog_dir=${recog_dir}_${unit}
    fi
    if [ ${length_norm} = true ]; then
        recog_dir=${recog_dir}_norm
    fi
    if [ ${metric} != 'edit_distance' ]; then
        recog_dir=${recog_dir}_${metric}
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ]; then
        recog_dir=${recog_dir}_lm${lm_weight}
    fi
    if [ ! -z ${lm_second} ] && [ ${lm_second_weight} != 0 ]; then
        recog_dir=${recog_dir}_rescore${lm_second_weight}
    fi
    if [ ! -z ${lm_bwd} ] && [ ${lm_bwd_weight} != 0 ]; then
        recog_dir=${recog_dir}_bwd${lm_bwd_weight}
    fi
    if [ ${ctc_weight} != 0.0 ]; then
        recog_dir=${recog_dir}_ctc${ctc_weight}
    fi
    if [ ${softmax_smoothing} != 1.0 ]; then
        recog_dir=${recog_dir}_smooth${softmax_smoothing}
    fi
    if [ ${gnmt_decoding} = true ]; then
        recog_dir=${recog_dir}_gnmt
    fi
    if [ ${resolving_unk} = true ]; then
        recog_dir=${recog_dir}_resolvingOOV
    fi
    if [ ${fwd_bwd_attention} = true ]; then
        recog_dir=${recog_dir}_fwdbwd
    fi
    if [ ${bwd_attention} = true ]; then
        recog_dir=${recog_dir}_bwd
    fi
    if [ ${reverse_lm_rescoring} = true ]; then
        recog_dir=${recog_dir}_revLM
    fi
    if [ ${asr_state_carry_over} = true ]; then
        recog_dir=${recog_dir}_ASRcarryover
    fi
    if [ ${longform_max_n_frames} != 0 ]; then
        recog_dir=${recog_dir}_longform${longform_max_n_frames}
    fi
    if [ ${streaming_encoding} = true ]; then
        recog_dir=${recog_dir}_streaming_encoding${block_size}
    fi
    if [ ${block_sync} = true ]; then
        recog_dir=${recog_dir}_blocksync${block_size}
    fi
    if [ ${n_average} != 1 ]; then
        recog_dir=${recog_dir}_average${n_average}
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ] && [ ${lm_state_carry_over} = true ]; then
        recog_dir=${recog_dir}_LMcarryover
    fi
    if [ ${oracle} = true ]; then
        recog_dir=${recog_dir}_oracle
    fi
    if [ ${mma_delay_threshold} != -1 ]; then
        recog_dir=${recog_dir}_epswait${mma_delay_threshold}
    fi
    if [ ! -z ${model3} ]; then
        recog_dir=${recog_dir}_ensemble4
    elif [ ! -z ${model2} ]; then
        recog_dir=${recog_dir}_ensemble3
    elif [ ! -z ${model1} ]; then
        recog_dir=${recog_dir}_ensemble2
    fi
    mkdir -p ${recog_dir}

    if [ $(echo ${model} | grep 'train_sp') ]; then
        if [ $(echo ${model} | grep 'all') ]; then
            recog_set=${data}/dataset/${set}_sp_all_wpbpe10000.tsv
        elif [ $(echo ${model} | grep 'aps_other') ]; then
            recog_set=${data}/dataset/${set}_sp_aps_other_wpbpe10000.tsv
        elif [ $(echo ${model} | grep 'sps') ]; then
            recog_set=${data}/dataset/${set}_sp_sps_wpbpe10000.tsv
        fi
    else
        if [ $(echo ${model} | grep 'all') ]; then
            recog_set=${data}/dataset/${set}_all_wpbpe10000.tsv
        elif [ $(echo ${model} | grep 'aps_other') ]; then
            recog_set=${data}/dataset/${set}_aps_other_wpbpe10000.tsv
        elif [ $(echo ${model} | grep 'sps') ]; then
            recog_set=${data}/dataset/${set}_sps_wpbpe10000.tsv
        fi
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/eval.py \
        --recog_n_gpus ${n_gpus} \
        --recog_sets ${recog_set} \
        --recog_dir ${recog_dir} \
        --recog_unit ${unit} \
        --recog_metric ${metric} \
        --recog_model ${model} ${model1} ${model2} ${model3} \
        --recog_model_bwd ${model_bwd} \
        --recog_batch_size ${batch_size} \
        --recog_beam_width ${beam_width} \
        --recog_max_len_ratio ${max_len_ratio} \
        --recog_min_len_ratio ${min_len_ratio} \
        --recog_length_penalty ${length_penalty} \
        --recog_length_norm ${length_norm} \
        --recog_coverage_penalty ${coverage_penalty} \
        --recog_coverage_threshold ${coverage_threshold} \
        --recog_gnmt_decoding ${gnmt_decoding} \
        --recog_eos_threshold ${eos_threshold} \
        --recog_lm ${lm} \
        --recog_lm_second ${lm_second} \
        --recog_lm_bwd ${lm_bwd} \
        --recog_lm_weight ${lm_weight} \
        --recog_lm_second_weight ${lm_second_weight} \
        --recog_lm_bwd_weight ${lm_bwd_weight} \
        --recog_ctc_weight ${ctc_weight} \
        --recog_softmax_smoothing ${softmax_smoothing} \
        --recog_resolving_unk ${resolving_unk} \
        --recog_fwd_bwd_attention ${fwd_bwd_attention} \
        --recog_bwd_attention ${bwd_attention} \
        --recog_reverse_lm_rescoring ${reverse_lm_rescoring} \
        --recog_asr_state_carry_over ${asr_state_carry_over} \
        --recog_lm_state_carry_over ${lm_state_carry_over} \
        --recog_n_average ${n_average} \
        --recog_oracle ${oracle} \
        --recog_longform_max_n_frames ${longform_max_n_frames} \
        --recog_streaming_encoding ${streaming_encoding} \
        --recog_block_sync ${block_sync} \
        --recog_block_sync_size ${block_size} \
        --recog_mma_delay_threshold ${mma_delay_threshold} \
        --recog_ctc_vad ${vad_free} \
        --recog_ctc_vad_blank_threshold ${blank_threshold} \
        --recog_ctc_vad_spike_threshold ${spike_threshold} \
        --recog_ctc_vad_n_accum_frames ${n_accum_frames} \
        --recog_stdout ${stdout} || exit 1;

    grep RTF ${recog_dir}/decode.log

    if [ ${metric} = 'edit_distance' ]; then
        # remove <unk>
        cat ${recog_dir}/ref.trn | sed 's:<unk>::g' > ${recog_dir}/ref.trn.filt
        cat ${recog_dir}/hyp.trn | sed 's:<unk>::g' > ${recog_dir}/hyp.trn.filt

        echo ${set}
        # WER
        echo 'WER' > ${recog_dir}/RESULTS
        sclite -r ${recog_dir}/ref.trn.filt trn -h ${recog_dir}/hyp.trn.filt trn -i rm -o all stdout > ${recog_dir}/result.txt
        grep -e Avg -e SPKR -m 2 ${recog_dir}/result.txt >> ${recog_dir}/RESULTS
        # CER
        echo 'CER' >> ${recog_dir}/RESULTS
        # add space
        paste -d " " <(cat ${recog_dir}/ref.trn.filt | cut -f 1 -d "(" | LC_ALL=en_US.UTF-8 sed -e "s/ //g" | LC_ALL=en_US.UTF-8 sed -e 's/\(.\)/ \1/g') <(cat ${recog_dir}/ref.trn.filt | sed -e 's/.*\((.*)\)/\1/g') \
            > ${recog_dir}/ref.trn.filt.char
        paste -d " " <(cat ${recog_dir}/hyp.trn.filt | cut -f 1 -d "(" | LC_ALL=en_US.UTF-8 sed -e "s/ //g" | LC_ALL=en_US.UTF-8 sed -e 's/\(.\)/ \1/g') <(cat ${recog_dir}/hyp.trn.filt | sed -e 's/.*\((.*)\)/\1/g') \
            > ${recog_dir}/hyp.trn.filt.char
        sclite -r ${recog_dir}/ref.trn.filt.char trn -h ${recog_dir}/hyp.trn.filt.char trn -i rm -o all stdout > ${recog_dir}/result.char.txt
        grep -e Avg -e SPKR -m 2 ${recog_dir}/result.char.txt >> ${recog_dir}/RESULTS
        cat ${recog_dir}/RESULTS
    fi
done
