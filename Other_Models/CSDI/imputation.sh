#!/bin/bash

model_name=CSDI
device=1
batch_size=8
train_epochs=10

# ETTh1
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --d_model 96 \
    --gpu $device
done

# ETTh2
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --gpu $device \
    --d_model 96
done

# ETTm1
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --gpu $device \
    --d_model 96
done

# ETTm2
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 7 \
    --gpu $device \
    --d_model 96
done

# ECL (321 channels)
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 321 \
    --enc_in 321 \
    --filename CSDI \
    --d_model 96 \
    --batch_size $batch_size \
    --train_epochs $train_epochs \
    --wandb False \
    --gpu $device
done

# Weather (21 channels)
for rate in 0.125 0.25 0.375 0.5
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_$rate \
    --mask_rate $rate \
    --model $model_name \
    --data custom \
    --features M \
    --pred_len 96 \
    --c_out 21 \
    --enc_in 21 \
    --filename CSDI \
    --d_model 96 \
    --batch_size $batch_size \
    --train_epochs $train_epochs \
    --wandb False \
    --gpu $device
done 