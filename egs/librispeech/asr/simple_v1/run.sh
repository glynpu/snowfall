#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

stage=7

if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.org/resources/11" data/local/lm
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh data/local/lm data/local/dict_nosp
fi

if [ $stage -le 3 ]; then
  local/prepare_lang.sh \
    --position-dependent-phones false \
    data/local/dict_nosp \
    "<UNK>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 4 ]; then
  # Build G
  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=1 \
    data/local/lm/lm_tgmed.arpa >data/lang_nosp/G_uni.fst.txt

  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    data/local/lm/lm_tgmed.arpa >data/lang_nosp/G.fst.txt

  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=4 \
    data/local/lm/lm_fglarge.arpa >data/lang_nosp/G_4_gram.fst.txt

  echo ""
  echo "To load G:"
  echo "Use::"
  echo "  with open('data/lang_nosp/G.fst.txt') as f:"
  echo "    G = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi

if [ $stage -le 5 ]; then
  python3 ./prepare.py
fi

if [ $stage -le 6 ]; then
  # python3 ./train.py # ctc training
  # python3 ./mmi_bigram_train.py # ctc training + bigram phone LM
  #  python3 ./mmi_mbr_train.py

  # Single node, multi-GPU training
  # Adapting to a multi-node scenario should be straightforward.
  ngpus=2
  python3 -m torch.distributed.launch --nproc_per_node=$ngpus ./mmi_bigram_train.py --world_size $ngpus
fi

if [ $stage -le 7 ]; then
  # python3 ./decode.py # ctc decoding
  # python3 ./mmi_bigram_decode.py --epoch 9
  #  python3 ./mmi_mbr_decode.py
  mkdir -p exp/
  export CUDA_VISIBLE_DEVICES=3
  ln -sf /home/storage14/guoliyong/open-source/snowfall/egs/librispeech/asr/simple_v1/exp-conformer-noam-mmi-att-musan-sa-vgg ./
  ln -sf /ceph-ly/open-source/lm_resocre_snowfall/snowfall/egs/librispeech/asr/simple_v1/exp/data ./exp/
  ln -sf /ceph-ly/open-source/to_submit/espnet_snowfall/snowfall/egs/librispeech/asr/simple_v1/data ./
  output_dir=result_dir
  mkdir -p $output_dir
  python3 ./mmi_att_transformer_decode.py \
    --output-dir $output_dir \
    --num-paths -1 \
    --max-duration 300 \
    --attention-dim 512 \
    --use-lm-rescoring True \
    --avg 15 \
    --epoch 19
fi

if [ $stage -le 8 ]; then
  export CUDA_VISIBLE_DEVICES=3
  python3 ./mmi_att_transformer_decode.py \
    --output-dir $output_dir \
    --num-paths 1000 \
    --max-duration 300 \
    --attention-dim 512 \
    --use-lm-rescoring True \
    --avg 15 \
    --epoch 19
fi
