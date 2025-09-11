#!/bin/bash

model="CnDiff"
use_cond=1
use_tphi=2

description="newmethod"
filename="classification_sweep"

device=0

datasets=(
  # "EthanolConcentration"
  # "FaceDetection"
  # "Handwriting" 
  # "Heartbeat"
  "JapaneseVowels"
  # "PEMS-SF"
  # "SelfRegulationSCP1"
  # "SelfRegulationSCP2"
  "SpokenArabicDigits"
  # "UWaveGestureLibrary"
)

attn_dropout=0.1
# 0.1
hidden_dim=32
# 512
mlp_ratio=2
# 1
n_depth=4
# 2
n_emb=2
# 2
n_heads=4
# 8
timesteps=100
# 100

for dataset in "${datasets[@]}"; do
  for hidden_dim in 32 64 128; do
    for n_depth in 2 3 4; do
      for n_heads in 4 8; do
        for mlp_ratio in 2 4; do
          for timesteps in 10 50 100 200; do
            python -u run.py \
              --task_name classification \
              --is_training 1 \
              --root_path ./dataset/$dataset/ \
              --model_id $dataset \
              --model $model \
              --data UEA \
              --batch_size 16 \
              --train_epochs 100 \
              --des $description \
              --learning_rate 0.001 \
              --n_emb $n_emb \
              --n_heads $n_heads \
              --attn_dropout $attn_dropout \
              --mlp_ratio $mlp_ratio \
              --n_depth $n_depth \
              --hidden_dim $hidden_dim \
              --use_tphi $use_tphi \
              --wandb False \
              --normalize True \
              --use_cond $use_cond --gpu $device --tphi_loss True --timesteps $timesteps --train_epochs 100
          done
        done
      done
    done
  done
done