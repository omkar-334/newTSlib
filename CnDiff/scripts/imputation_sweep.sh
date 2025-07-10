#!/bin/bash

model_name=CnDiff
device=1
cond=1
tphi=1
tphi_loss=True

for attn_dropout in  0.1 0.2
do
  for mlp_ratio in 2 4
  do
    for timesteps in 50 100 200
    do
      for hidden_dim in 256 512
      do
        for rate in 0.125 0.25 0.375 0.5
        do
          for n_depth in 1 2 4
          do
            for n_emb in 2 4
            do
              for n_heads in 1 2 4 8
              do


                python -u run.py \
                --task_name imputation \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTh2.csv \
                --model_id ETTh2_mask_${rate} \
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
                --timesteps $timesteps \
                --tphi_loss $tphi_loss \
                --use_cond $cond \
                --use_tphi $tphi \
                --normalize False
              done
            done
          done
        done
      done
    done
  done
done
