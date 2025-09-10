#!/bin/bash
model_name=iTransformer

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
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 18 \
  --dec_in 18 \
  --c_out 18 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1