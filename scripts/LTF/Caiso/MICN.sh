#!/bin/bash
model_name=MICN

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
  --seq_len 96 \
  --label_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1