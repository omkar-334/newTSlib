#!/bin/bash

model="CnDiffOld"
use_cond=1
classifier=1

description="base"


datasets=(
  "EthanolConcentration"
  "FaceDetection"
  "Handwriting"
  "Heartbeat"
  "JapaneseVowels"
  "PEMS-SF"
  "SelfRegulationSCP1"
  "SelfRegulationSCP2"
  "SpokenArabicDigits"
  "UWaveGestureLibrary"
)

attn_dropout=0.1
# 0.1
hidden_dim=512
# 512
mlp_ratio=3
# 1
n_depth=4
# 2
n_emb=2
# 2
n_heads=8
# 8
timesteps=100
# 100


for dataset in "${datasets[@]}"; do
  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/$dataset/ \
    --model_id $dataset \
    --model $model \
    --data UEA \
    --gpu "$device" \
    --batch_size 16 \
    --train_epochs 1 \
    --des $description \
    --learning_rate 0.001 \
    --n_emb $n_emb \
    --n_heads $n_heads \
    --attn_dropout $attn_dropout \
    --mlp_ratio $mlp_ratio \
    --n_depth $mlp_ratio \
    --use_tphi True \
    --classifier $classifier \
    --wandb False \
    --use_cond $use_cond
done


    # echo "Running dataset: $dataset"
    # python -u run.py \
    #   --task_name classification \
    #   --is_training 1 \
    #   --root_path ./dataset/"$dataset"/ \
    #   --model_id "$dataset" \
    #   --model "$model" \
    #   --data UEA \
    #   --gpu "$device" \
    #   --batch_size 16 \
    #   --train_epochs 100 \
    #   --des "$description" \
    #   --learning_rate 0.001 \
    #   --n_emb 4 \
    #   --n_heads 16 \
    #   --attn_dropout 0.2 \
    #   --mlp_ratio 2 \
    #   --n_depth 4 \
    #   --use_cond True \
    #   --use_tphi True \
    #   --wandb False