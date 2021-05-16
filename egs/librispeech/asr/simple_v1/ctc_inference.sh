export CUDA_VISIBLE_DEVICES=2
python3 local_snowfall/nnlm_asr_inference.py \
        --num_workers 0 \
        --ngpu 1 \
        --asr_train_config exp/kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave/config.yaml \
        --asr_model_file exp/kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave/valid.acc.ave_10best.pth
