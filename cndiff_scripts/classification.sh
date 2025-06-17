#!/bin/bash

model_name="CnDiff"
description="run"
device=1

datasets=(
  "EthanolConcentration"
  "FaceDetection"
  "Handwriting"s
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
    --n_emb 2 \
    --n_heads 8 \
    --attn_dropout 0.1 \
    --mlp_ratio 1 \
    --n_depth 2 \
    --use_cond True \
    --use_tphi True \
    --wandb True
done
