#!/bin/bash

model_name="CnDiff"
description="run"
device=1
cond=True
tphi=True

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model $model_name \
  --data MSL \
  --device $device \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 10 \
  --n_emb 2 \
  --n_heads 8 \
  --attn_dropout 0.1 \
  --mlp_ratio 1 \
  --n_depth 3 \
  --use_cond $cond \
  --use_tphi $tphi

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model $model_name \
  --data PSM \
  --device $device \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --n_emb 2 \
  --n_heads 8 \
  --attn_dropout 0.1 \
  --mlp_ratio 1 \
  --n_depth 3 \
  --use_cond $cond \
  --use_tphi $tphi

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model $model_name \
  --data SMAP \
  --device $device \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --n_emb 2 \
  --n_heads 8 \
  --attn_dropout 0.1 \
  --mlp_ratio 1 \
  --n_depth 3 \
  --use_cond $cond \
  --use_tphi $tphi


python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model $model_name \
  --data SMD \
  --device $device \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 38 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --n_emb 2 \
  --n_heads 8 \
  --attn_dropout 0.1 \
  --mlp_ratio 1 \
  --n_depth 3 \
  --use_cond $cond \
  --use_tphi $tphi



python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --model $model_name \
  --data SWAT \
  --device $device \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --n_emb 2 \
  --n_heads 8 \
  --attn_dropout 0.1 \
  --mlp_ratio 1 \
  --n_depth 3 \
  --use_cond $cond \
  --use_tphi $tphi