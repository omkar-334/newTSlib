#!/bin/bash

model_name=CnDiff

cond=1
tphi=2
tphi_loss=True
normalize=True

attn_dropout=0.1
hidden_dim=64
mlp_ratio=2
n_depth=2
n_emb=2
n_heads=4
timesteps=200

filename=ltf_ECL
device=1

# Electricity
pred_len=168

for seq_len in 336; do
  for n_depth in 2 4; do
    for hidden_dim in 32; do
      for mlp_ratio in 1 2; do
        for n_emb in 1 2; do
          for n_heads in 2 4 ; do
            python -u run.py \
              --task_name long_term_forecast \
              --is_training 1 \
              --root_path ./dataset/electricity/ \
              --data_path electricity.csv \
              --model_id ECL_s${seq_len}_p${pred_len} \
              --model $model_name \
              --data custom \
              --features M \
              --seq_len $seq_len \
              --label_len 0 \
              --pred_len $pred_len \
              --e_layers 2 \
              --d_layers 1 \
              --factor 3 \
              --enc_in 321 \
              --dec_in 321 \
              --c_out 321 \
              --use_cond $cond \
              --use_tphi $tphi \
              --tphi_loss $tphi_loss \
              --des 'Exp' \
              --itr 1 \
              --normalize $normalize \
              --attn_dropout $attn_dropout \
              --hidden_dim $hidden_dim \
              --mlp_ratio $mlp_ratio \
              --n_depth $n_depth \
              --n_emb $n_emb \
              --n_heads $n_heads \
              --filename $filename \
              --gpu $device \
              --timesteps $timesteps
          done
        done
      done
    done
  done
done