#!/bin/bash

model_name="CnDiff"
description="run"
device=1

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

for dataset in "${datasets[@]}"; do
  echo "Running dataset: $dataset"
  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/"$dataset"/ \
    --model_id "$dataset" \
    --model "$model_name" \
    --data UEA \
    --gpu "$device" \
    --batch_size 16 \
    --train_epochs 100 \
    --des "$description" \
    --learning_rate 0.001 \
    --n_emb 4 \
    --n_heads 16 \
    --attn_dropout 0.2 \
    --mlp_ratio 2 \
    --n_depth 4 \
    --use_cond True \
    --use_tphi True \
    --wandb False
done
