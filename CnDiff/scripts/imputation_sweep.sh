#!/bin/bash

model_name=CnDiff
device=1
cond=1
tphi=True

n_depth=4
n_emb=2
n_heads=8

for attn_dropout in 0.05 0.1 0.15
do
  for mlp_ratio in 3 3.5 4
  do
    for timesteps in 80 100 120
    do
      for hidden_dim in 512 576
      do
        for rate in 0.125 0.25 0.375 0.5
        do
          echo "Running with dropout=$attn_dropout, mlp_ratio=$mlp_ratio, timesteps=$timesteps, hidden_dim=$hidden_dim, mask_rate=$rate"
          
          python -u run.py \
            --task_name imputation \
            --is_training 1 \
            --root_path ./dataset/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_mask_${rate}_do${attn_dropout}_mlp${mlp_ratio}_ts${timesteps}_hd${hidden_dim} \
            --mask_rate $rate \
            --model $model_name \
            --data custom \
            --features M \
            --pred_len 96 \
            --c_out 7 \
            --gpu $device \
            --d_model 96 \
            --attn_dropout $attn_dropout \
            --hidden_dim $hidden_dim \
            --mlp_ratio $mlp_ratio \
            --n_depth $n_depth \
            --n_emb $n_emb \
            --n_heads $n_heads \
            --timesteps $timesteps
        done
      done
    done
  done
done
