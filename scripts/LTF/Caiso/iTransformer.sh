#!/bin/bash
model_name=iTransformer

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
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1