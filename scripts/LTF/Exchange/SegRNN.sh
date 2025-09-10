#!/bin/bash
model_name=SegRNN

seq_len=96

python -u run.py \
  --gpu 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_14 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 14 \
  --seg_len 24 \
  --enc_in 8 \
  --d_model 64 \
  --dropout 0 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

