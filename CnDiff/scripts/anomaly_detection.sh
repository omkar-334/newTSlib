#!/bin/bash

model_name="CnDiff"
description="run"
cond=1
tphi=True
# anomaly_detection_SMAP_CnDiff_attndropout0.1_hiddendim128_mlp-ratio3_n-depth3_n-emb4_n-heads8_timesteps100_run
# anomaly_detection_SMAP_CnDiff_attndropout0.2_hiddendim128_mlp-ratio1_n-depth2_n-emb4_n-heads4_timesteps100_run
# anomaly_detection_SMAP_CnDiff_attndropout0.1_hiddendim128_mlp-ratio1_n-depth1_n-emb8_n-heads16_timesteps200_run

# anomaly_detection_SMAP_CnDiff_attndropout0.1_hiddendim128_mlp-ratio3_n-depth3_n-emb4_n-heads8_timesteps100_test---accuracy - 0.9356126515491693

# anomaly_detection_SMAP_CnDiff_attndropout0.1_hiddendim128_mlp-ratio3_n-depth3_n-emb4_n-heads8_timesteps100_test---accuracy - 0.9364382203262984 normalize - True
attn_dropout=0.1
# 0.1
hidden_dim=128
# 512
mlp_ratio=3
# 1
n_depth=3
# 2
n_emb=4
# 2
n_heads=8
# 8
timesteps=100
# 100
# 100
# attn_dropout=0.1
# # 0.1
# hidden_dim=512
# # 512
# mlp_ratio=3
# # 1
# n_depth=4
# # 2
# n_emb=2
# # 2
# n_heads=8
# # 8
# timesteps=100
# # 100


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
  --tphi_loss True \
  --normalize True

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
  --batch_size 128 \
  --train_epochs 3 \
  --use_cond $cond \
  --use_tphi $tphi \
  --attn_dropout $attn_dropout \
  --hidden_dim $hidden_dim \
  --mlp_ratio $mlp_ratio \
  --n_depth $n_depth \
  --n_emb $n_emb \
  --n_heads $n_heads \
  --timesteps $timesteps \
  --tphi_loss True \
  --normalize True

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
  --batch_size 128 \
  --train_epochs 3 \
  --use_cond $cond \
  --use_tphi $tphi \
  --attn_dropout $attn_dropout \
  --hidden_dim $hidden_dim \
  --mlp_ratio $mlp_ratio \
  --n_depth $n_depth \
  --n_emb $n_emb \
  --n_heads $n_heads \
  --timesteps $timesteps  \
  --tphi_loss False \
  --normalize True 


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
  --tphi_loss True \
  --normalize True



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
  --batch_size 128 \
  --train_epochs 3 \
  --use_cond $cond \
  --use_tphi $tphi \
  --attn_dropout $attn_dropout \
  --hidden_dim $hidden_dim \
  --mlp_ratio $mlp_ratio \
  --n_depth $n_depth \
  --n_emb $n_emb \
  --n_heads $n_heads \
  --timesteps $timesteps \
  --tphi_loss True \
  --normalize True