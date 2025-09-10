#!/bin/bash
model_name=MultiPatchFormer

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
  --seq_len 96 \
  --label_len 14 \
  --pred_len 14 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1