#!/bin/sh

python train.py \
  --src_vocab_fpath vocab.bpe.32000.refined \
  --trg_vocab_fpath vocab.bpe.32000.refined \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern train.tok.clean.bpe.32000.en-de \
  --use_token_batch True \
  --batch_size 4096 \
  --sort_type pool \
  --pool_size 200000
