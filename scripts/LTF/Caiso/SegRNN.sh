#!/bin/bash
model_name=SegRNN

seq_len=96

python -u run.py \
  --gpu 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Caiso/ \
  --data_path Caiso.csv \
  --model_id Caiso_96_720 \
  --model $model_name \
  --data Caiso \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --seg_len 24 \
  --enc_in 321 \
  --d_model 512 \
  --dropout 0 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

