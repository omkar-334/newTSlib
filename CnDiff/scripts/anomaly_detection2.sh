#!/bin/bash

model_name="CnDiff"
description="try2"
cond=1
tphi=2
device=1

attn_dropout=0.3
mlp_ratio=1
n_depth=2
n_emb=4
n_heads=16
# timesteps=200
normalize

for tphi_loss in True
do
  for normalize in True
  do
    for hidden_dim in 256 512 768
    do
      for timesteps in 100 150 200
      do
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
          --batch_size 128 \
          --train_epochs 10 \
          --use_cond $cond \
          --use_tphi $tphi \
          --attn_dropout $attn_dropout \
          --hidden_dim $hidden_dim \
          --mlp_ratio $mlp_ratio \
          --n_depth $n_depth \
          --n_emb $n_emb \
          --n_heads $n_heads \
          --timesteps $timesteps \
          --tphi_loss $tphi_loss \
          --normalize $normalize
      done
    done
  done
done

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
#   --use_cond $cond \
#   --use_tphi $tphi \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads \
#   --timesteps $timesteps \
#   --tphi_loss $tphi_loss \
#   --normalize $normalize

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
#   --use_cond $cond \
#   --use_tphi $tphi \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads \
#   --timesteps $timesteps \
#   --tphi_loss $tphi_loss \
#   --normalize $normalize

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
#   --use_cond $cond \
#   --use_tphi $tphi \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads \
#   --timesteps $timesteps  \
#   --tphi_loss $tphi_loss \
#   --normalize $normalize

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
#   --use_cond $cond \
#   --use_tphi $tphi \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads \
#   --timesteps $timesteps \
#   --tphi_loss $tphi_loss \
#   --normalize $normalize

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
#   --use_cond $cond \
#   --use_tphi $tphi \
#   --attn_dropout $attn_dropout \
#   --hidden_dim $hidden_dim \
#   --mlp_ratio $mlp_ratio \
#   --n_depth $n_depth \
#   --n_emb $n_emb \
#   --n_heads $n_heads \
#   --timesteps $timesteps \
#   --tphi_loss $tphi_loss \
#   --normalize $normalize