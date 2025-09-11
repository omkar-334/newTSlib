#!/bin/bash

model_name="CnDiff"
filename='final_new_ad'
description="new"

cond=1
tphi=1
tphi_loss=True

# anomaly_detection_MSL_CnDiff_attndropout0.3_hiddendim32_mlp-ratio1_n-depth3_n-emb2_n-heads4_timesteps200_sweep_use_cond1_use_tphi2_normalizeTrue_tphi-lossTrue

# attn_dropout=0.3
# hidden_dim=32
# mlp_ratio=1
# n_depth=3
# n_emb=2
# n_heads=4
# timesteps=200
# normalize=True

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
#   --train_epochs 100 \
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
#   --des $description \
#   --filename $filename \
#   --normalize $normalize


# anomaly_detection_PSM_CnDiff_attndropout0.3_hiddendim32_mlp-ratio2_n-depth4_n-emb4_n-heads4_timesteps100_sweep_use_cond1_use_tphi2_normalizeTrue_tphi-lossTrue

attn_dropout=0.1
hidden_dim=32
mlp_ratio=4
n_depth=4
n_emb=1
n_heads=4
timesteps=200
normalize=True

for hidden_dim in 32 64; do
  for n_heads in 1 2 4; do
    for n_depth in 2 3 4; do
      for n_emb in 1 2 4; do
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
          --des $description \
          --filename $filename \
          --learning_rate 0.001 \
          --gpu 0 \
          --normalize $normalize \
          --filename 'final_ad_psm_sweep' \
          --train_epochs 100
      done
    done
  done
done


# # anomaly_detection_SMAP_CnDiff_attndropout0.3_hiddendim64_mlp-ratio2_n-depth4_n-emb4_n-heads4_timesteps200_sweep_use_cond1_use_tphi2_normalizeTrue_tphi-lossTrue
# attn_dropout=0.3
# hidden_dim=64
# mlp_ratio=2
# n_depth=4
# n_emb=4
# n_heads=4
# timesteps=200
# normalize=True

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
#   --batch_size 64 \
#   --train_epochs 100 \
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
#   --des $description \
#   --filename $filename \
#   --normalize $normalize

# # anomaly_detection_SMD_CnDiff_attndropout0.3_hiddendim32_mlp-ratio2_n-depth3_n-emb2_n-heads1_timesteps100_sweep_use_cond1_use_tphi2_normalizeFalse_tphi-lossTrue"

# attn_dropout=0.3
# hidden_dim=32
# mlp_ratio=2
# n_depth=3
# n_emb=2
# n_heads=1
# timesteps=100
# normalize=False

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
#   --train_epochs 100 \
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
#   --des $description \
#   --filename $filename \
#   --patience 3 \
#   --normalize $normalize


# # anomaly_detection_SWaT_CnDiff_attndropout0.3_hiddendim32_mlp-ratio1_n-depth3_n-emb4_n-heads1_timesteps100_sweep_use_cond1_use_tphi2_normalizeTrue_tphi-lossTrue


# attn_dropout=0.3
# hidden_dim=32
# mlp_ratio=1
# n_depth=3
# n_emb=4
# n_heads=1
# timesteps=100
# normalize=True


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
#   --train_epochs 100 \
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
#   --des $description \
#   --filename $filename \
#   --normalize $normalize