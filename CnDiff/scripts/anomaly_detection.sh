#!/bin/bash

model_name="CnDiff"
filename='msl_sweep'
description="lr3_sweepsmd"

cond=1
tphi=2
tphi_loss=True

    # "anomaly_detection_MSL_CnDiff_attndropout0.3_hiddendim128_mlp-ratio1_n-depth2_n-emb4_n-heads8_timesteps200_lr3_use_cond1_use_tphi2_normalizeTrue_tphi-lossFalse": {
# 
# attn_dropout=0.1
# hidden_dim=128
# mlp_ratio=2
# n_depth=6
# n_emb=8 
# n_heads=4
# timesteps=200

# attn_dropout=0.1 #0.3
# hidden_dim=128
# mlp_ratio=1
# n_depth=3
# n_emb=4
# n_heads=16
# timesteps=200

attn_dropout=0.3
hidden_dim=128
mlp_ratio=1
n_depth=2
n_emb=4
n_heads=8
timesteps=200
normalize=True

for hidden_dim in 32 64 128; do
  for n_depth in 2 3 4; do
    for n_heads in 1 4 8; do
      for mlp_ratio in 1 2; do
        for timesteps in 100 200; do
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
            --train_epochs 100 \
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
            --normalize $normalize
        done
      done
    done
  done
done



# # anomaly_detection_PSM_CnDiff_attndropout0.0_hiddendim256_mlp-ratio1_n-depth2_n-emb4_n-heads32_timesteps200_cond_use_cond1_use_tphi2_normalizeTrue_tphi-lossTrue
# attn_dropout=0.0
# hidden_dim=128
# mlp_ratio=1
# n_depth=2
# n_emb=4
# n_heads=32
# timesteps=200
# normalize=True

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
#     --filename $filename \
#   --normalize $normalize


# attn_dropout=0.1 #0.3
# hidden_dim=128
# mlp_ratio=1
# n_depth=3
# n_emb=4
# n_heads=16
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


# attn_dropout=0.1
# hidden_dim=128
# mlp_ratio=2
# n_depth=6
# n_emb=8 
# n_heads=4
# timesteps=200
# normalize=False

# device=0


# for attn_dropout in 0.05 0.1 0.2; do
#   for mlp_ratio in 1 2 3; do
#     for n_depth in 4 6; do
#       for n_heads in 4 8; do
#         python -u run.py \
#           --task_name anomaly_detection \
#           --is_training 1 \
#           --root_path ./dataset/SMD \
#           --model_id SMD \
#           --model $model_name \
#           --data SMD \
#           --wandb False \
#           --features M \
#           --pred_len 96 \
#           --d_model 96 \
#           --c_out 38 \
#           --anomaly_ratio 0.5 \
#           --batch_size 128 \
#           --train_epochs 100 \
#           --use_cond $cond \
#           --use_tphi $tphi \
#           --attn_dropout $attn_dropout \
#           --hidden_dim $hidden_dim \
#           --mlp_ratio $mlp_ratio \
#           --n_depth $n_depth \
#           --n_emb $n_emb \
#           --n_heads $n_heads \
#           --timesteps $timesteps \
#           --tphi_loss $tphi_loss \
#           --des $description \
#           --filename $filename \
#           --patience 3 \
#           --gpu $device \
#           --normalize $normalize
#       done
#     done
#   done
# done

# attn_dropout=0.1
# hidden_dim=256
# mlp_ratio=4
# n_depth=4
# n_emb=4
# n_heads=16
# timesteps=10
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