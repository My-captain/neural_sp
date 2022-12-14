# PopSong

## 数据预处理
1. data_prepare.py
   - 结合多个数据集并整理成kaldi格式
     - wav/
       - train/
         - Singer1~n
           - utt1.wav~~
       - dev/
         - Singer1~n
           - utt1.wav~~
       - test/
         - Singer1~n
           - utt1.wav~~
     - transcript
       - transcript.txt


## Transformer-MMA



### Train
- 配置文件： conf/asr/transformer_mma_hie_subsample8_ma4H_ca4H_w16_from4L.yaml
- train.py参数：
```editorconfig
--local_world_size 1
--corpus popsong
--use_wandb false 
--config /home/zliu-elliot/Desktop/computerMusic/alignment/neural_sp/examples/popsong/s5/conf/asr/transformer_mma_hie_subsample8_ma4H_ca4H_w16_from4L.yaml 
--config2
--n_gpus 1
--cudnn_benchmark true 
--cudnn_deterministic false 
--pin_memory false 
--train_set /media/zliu-elliot/Jarvis/PopSong/dataset/train_sp.tsv 
--dev_set /media/zliu-elliot/Jarvis/PopSong/dataset/dev_sp.tsv 
--unit char
--dict /media/zliu-elliot/Jarvis/PopSong/dict/train_sp.txt 
--model_save_dir /media/zliu-elliot/Jarvis/neural_sp/PopSong/asr/transformer_mma_hie_debug/
--asr_init --external_lm --stdout false --resume
--max_n_frames 4000
```
- eval.py参数：
```editorconfig
--recog_n_gpus 1
--recog_sets /home/zliu-elliot/dataset/PopSong/dataset/test_sp.tsv
--recog_dir /media/zliu-elliot/Thor/neural_sp/PopSong/asr/openSinger_popCS_m4singer/train_sp/decode_dev_beam10_lp0.0_cp0.0_0.0_1.0_average10
--recog_first_n_utt 0
--recog_unit
--recog_metric edit_distance
--recog_model /media/zliu-elliot/Thor/neural_sp/PopSong/asr/openSinger_popCS_m4singer/train_sp/conv2Ltransformer256dmodel2048dff12L4Hpeadd_max_pool8_transformer256dmodel2048dff6L1Hpe1dconv3Lmocha_ma1H_ca4H_w16_bias-2.0_qua1.0_share_from4L_HD0.5_noam_lr5.0_bs32_ls0.1_warmup25000_accum8_ctc0.3_27FM2_50TM2/model.epoch-20   
--recog_model_bwd 
--recog_batch_size 1
--recog_beam_width 10
--recog_max_len_ratio 1.0
--recog_min_len_ratio 0.0
--recog_length_penalty 0.0
--recog_length_norm false
--recog_coverage_penalty 0.0
--recog_coverage_threshold 0.0
--recog_gnmt_decoding false
--recog_eos_threshold 1.0
--recog_lm 
--recog_lm_second 
--recog_lm_bwd 
--recog_lm_weight 0.3
--recog_lm_second_weight 0.3
--recog_ctc_weight 0.0
--recog_softmax_smoothing 1.0
--recog_resolving_unk false
--recog_fwd_bwd_attention false
--recog_bwd_attention false
--recog_reverse_lm_rescoring false
--recog_asr_state_carry_over false
--recog_lm_state_carry_over true
--recog_n_average 10
--recog_oracle false
--recog_longform_max_n_frames 0
--recog_streaming_encoding false
--recog_block_sync false
--recog_block_sync_size 40
--recog_mma_delay_threshold -1
--recog_ctc_vad false
--recog_ctc_vad_blank_threshold 40
--recog_ctc_vad_spike_threshold 0.1
--recog_ctc_vad_n_accum_frames 1600
--recog_stdout false
```

eval（MacOS）
```editorconfig
--recog_n_gpus 0
--recog_sets /Users/mr.robot/dataset/PopSong/dataset/test_sp.tsv
--recog_dir /Users/mr.robot/dataset/PopSong/testResult/decode_dev_beam10_lp0.0_cp0.0_0.0_1.0_average10
--recog_first_n_utt 0
--recog_unit
--recog_metric edit_distance
--recog_model /Users/mr.robot/model/transformer_mma_1head/model-avg10
--recog_model_bwd
--recog_batch_size 1
--recog_beam_width 10
--recog_max_len_ratio 1.0
--recog_min_len_ratio 0.0
--recog_length_penalty 0.0
--recog_length_norm false
--recog_coverage_penalty 0.0
--recog_coverage_threshold 0.0
--recog_gnmt_decoding false
--recog_eos_threshold 1.0
--recog_lm
--recog_lm_second
--recog_lm_bwd
--recog_lm_weight 0.3
--recog_lm_second_weight 0.3
--recog_ctc_weight 0.0
--recog_softmax_smoothing 1.0
--recog_resolving_unk false
--recog_fwd_bwd_attention false
--recog_bwd_attention false
--recog_reverse_lm_rescoring false
--recog_asr_state_carry_over false
--recog_lm_state_carry_over true
--recog_n_average 10
--recog_oracle false
--recog_longform_max_n_frames 0
--recog_streaming_encoding false
--recog_block_sync false
--recog_block_sync_size 40
--recog_mma_delay_threshold -1
--recog_ctc_vad false
--recog_ctc_vad_blank_threshold 40
--recog_ctc_vad_spike_threshold 0.1
--recog_ctc_vad_n_accum_frames 1600
--recog_stdout false
```

```text
TransformerEncoder(
    **ConvEncoder**(
        Conv2dBlock(
            Dropout(p=0.0, inplace=False)
            Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        Conv2dBlock(
            Dropout(p=0.0, inplace=False)
            Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=1, ceil_mode=True)
        )
        Linear(in_features=1280, out_features=256, bias=True)
    ) -> batch, T//2, 256
    **PositionalEncoding**(
        # PositionalEncoding直接与token的embedding相加
        Dropout(p=0.0, inplace=False)
    )
    
    TransformerEncoderBlock(
        LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        MultiheadAttentionMechanism(
            Dropout(p=0.0, inplace=False)
            (w_key): Linear(in_features=256, out_features=256, bias=True)
            (w_value): Linear(in_features=256, out_features=256, bias=True)
            (w_query): Linear(in_features=256, out_features=256, bias=True)
            (w_out): Linear(in_features=256, out_features=256, bias=True)
        ) -> batch, T//2, 256
        Dropout(p=0.1, inplace=False) + 残差（在LayerNorm之前）
        LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        PositionwiseFeedForward(
            Linear(in_features=256, out_features=2048, bias=True)
            ReLU()
            Dropout(p=0.1, inplace=False)
            Linear(in_features=2048, out_features=256, bias=True)
        )
        Dropout(p=0.1, inplace=False) + 残差（在LayerNorm之前）
    )  *12    ->   batch, T//8, 256
    （12个相同的TransformerEncoderBlock
      每4个中间有一个MaxPool1d降采样，沿着time轴作降采样）
    
    LayerNorm((256,), eps=1e-12, elementwise_affine=True)
)

TransformerDecoder(
    PositionalEncoding(
        CausalConv1d(
            (conv1d): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(2,))
        )
        LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        ReLU()
        Dropout(p=0.1, inplace=False)
        CausalConv1d(
            (conv1d): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(2,))
        )
        LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        ReLU()
        Dropout(p=0.1, inplace=False)
        CausalConv1d(
            (conv1d): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(2,))
        )
        LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        ReLU()
        Dropout(p=0.1, inplace=False)
    )

    TransformerDecoderBlock(
        (norm1): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (self_attention): MultiheadAttentionMechanism(
            (dropout_attn): Dropout(p=0.0, inplace=False)
            (w_key): Linear(in_features=256, out_features=256, bias=True)
            (w_value): Linear(in_features=256, out_features=256, bias=True)
            (w_query): Linear(in_features=256, out_features=256, bias=True)
            (w_out): Linear(in_features=256, out_features=256, bias=True)
        )
        Dropout(p=0.1, inplace=False) + 残差（在LayerNorm之前）
        (norm3): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        PositionwiseFeedForward(
            Linear(in_features=256, out_features=2048, bias=True)
            ReLU()
            Dropout(p=0.1, inplace=False)
            Linear(in_features=2048, out_features=256, bias=True)
        )
        Dropout(p=0.1, inplace=False) + 残差（在PositionwiseFeedForward之前）
    ) * 3 (前三层只有self-attention)
    
    TransformerDecoderBlock(
        (norm1): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (self_attention): MultiheadAttentionMechanism(
            (dropout_attn): Dropout(p=0.0, inplace=False)
            (w_key): Linear(in_features=256, out_features=256, bias=True)
            (w_value): Linear(in_features=256, out_features=256, bias=True)
            (w_query): Linear(in_features=256, out_features=256, bias=True)
            (w_out): Linear(in_features=256, out_features=256, bias=True)
        )
        Dropout(p=0.1, inplace=False) + 残差（在LayerNorm之前）
        (norm3): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        PositionwiseFeedForward(
            Linear(in_features=256, out_features=2048, bias=True)
            ReLU()
            Dropout(p=0.1, inplace=False)
            Linear(in_features=2048, out_features=256, bias=True)
        )
        Dropout(p=0.1, inplace=False) + 残差（在norm3之前）
    ) * 3 (前三层只有self-attention)
    
    TransformerDecoderBlock(
        (norm1): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (self_attention): MultiheadAttentionMechanism(
            (dropout_attn): Dropout(p=0.0, inplace=False)
            (w_key): Linear(in_features=256, out_features=256, bias=True)
            (w_value): Linear(in_features=256, out_features=256, bias=True)
            (w_query): Linear(in_features=256, out_features=256, bias=True)
            (w_out): Linear(in_features=256, out_features=256, bias=True)
        )
        Dropout(p=0.1, inplace=False) + 残差（在norm1之前）
        
        (norm2): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (src_attn): MMA(
            MonotonicEnergy
            ChunkEnergy
            ChunkwiseAttention
        )
        Dropout(p=0.1, inplace=False) + 残差（在norm2之前）
        
        (norm3): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        PositionwiseFeedForward(
            Linear(in_features=256, out_features=2048, bias=True)
            ReLU()
            Dropout(p=0.1, inplace=False)
            Linear(in_features=2048, out_features=256, bias=True)
        )
        Dropout(p=0.1, inplace=False) + 残差（在PositionwiseFeedForward之前）
    )
)

)
```
