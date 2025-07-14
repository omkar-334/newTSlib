#!/bin/bash

model_name=KANAD

# Original KANAD parameters from config.toml
batch_size=1024
train_epochs=100
learning_rate=0.01

# Run MSL dataset
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model $model_name \
  --data MSL \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --learning_rate $learning_rate

# Run PSM dataset
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model $model_name \
  --data PSM \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --learning_rate $learning_rate

# Run SMAP dataset
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model $model_name \
  --data SMAP \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --learning_rate $learning_rate

# Run SMD dataset
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model $model_name \
  --data SMD \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --learning_rate $learning_rate 

# Run SWaT dataset
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model $model_name \
  --data SWaT \
  --wandb False \
  --features M \
  --pred_len 96 \
  --d_model 96 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --learning_rate $learning_rate


# model_name=KANAD

# attn_dropout=0.1
# hidden_dim=128
# mlp_ratio=4
# n_depth=4
# n_emb=128
# n_heads=8

# python -u run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --root_path ./dataset/MSL \
#   --model_id MSL \
#   --model $model_name \
#   --data MSL \
#   --wandb False \
#   --features M \
#   --pred_len 96 \
#   --d_model 96 \
#   --c_out 55 \
#   --anomaly_ratio 1 \
#   --batch_size 128 \
#   --train_epochs 10 \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads


# python -u run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --root_path ./dataset/PSM \
#   --model_id PSM \
#   --model $model_name \
#   --data PSM \
#   --wandb False \
#   --features M \
#   --pred_len 96 \
#   --d_model 96 \
#   --c_out 25 \
#   --anomaly_ratio 1 \
#   --batch_size 128 \
#   --train_epochs 3 \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads

# python -u run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --root_path ./dataset/SMAP \
#   --model_id SMAP \
#   --model $model_name \
#   --data SMAP \
#   --wandb False \
#   --features M \
#   --pred_len 96 \
#   --d_model 96 \
#   --c_out 25 \
#   --anomaly_ratio 1 \
#   --batch_size 128 \
#   --train_epochs 3 \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads

# python -u run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --root_path ./dataset/SMD \
#   --model_id SMD \
#   --model $model_name \
#   --data SMD \
#   --wandb False \
#   --features M \
#   --pred_len 96 \
#   --d_model 96 \
#   --c_out 38 \
#   --anomaly_ratio 0.5 \
#   --batch_size 128 \
#   --train_epochs 10 \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads

# python -u run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --root_path ./dataset/SWaT \
#   --model_id SWaT \
#   --model $model_name \
#   --data SWaT \
#   --wandb False \
#   --features M \
#   --pred_len 96 \
#   --d_model 96 \
#   --c_out 51 \
#   --anomaly_ratio 1 \
#   --batch_size 128 \
#   --train_epochs 3 \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads