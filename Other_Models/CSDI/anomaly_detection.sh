#!/bin/bash

model_name=CSDI
device=1
batch_size=16
train_epochs=10

# MSL (55 channels)
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
  --d_model 64 \
  --c_out 55 \
  --enc_in 55 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu $device

# PSM (25 channels)
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model $model_name \
  --data PSM \
  --wandb False \
  --features M \
  --pred_len 25 \
  --d_model 25 \
  --c_out 25 \
  --enc_in 25 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu $device

# SMAP (25 channels)
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model $model_name \
  --data SMAP \
  --wandb False \
  --features M \
  --pred_len 25 \
  --d_model 25 \
  --c_out 25 \
  --enc_in 25 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu $device

# SMD (38 channels)
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model $model_name \
  --data SMD \
  --wandb False \
  --features M \
  --pred_len 38 \
  --d_model 38 \
  --c_out 38 \
  --enc_in 38 \
  --anomaly_ratio 0.5 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu $device

# SWaT (51 channels)
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWaT \
  --model $model_name \
  --data SWaT \
  --wandb False \
  --features M \
  --pred_len 51 \
  --d_model 51 \
  --c_out 51 \
  --enc_in 51 \
  --anomaly_ratio 1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu $device 