#!/bin/bash
model_name=MultiPatchFormer

python -u run.py \
  --gpu 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Norpool/ \
  --data_path Norpool.csv \
  --model_id Norpool_96_720 \
  --model $model_name \
  --data Norpool \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --enc_in 18 \
  --dec_in 18 \
  --c_out 18 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1