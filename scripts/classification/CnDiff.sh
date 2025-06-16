#!/bin/bash

model_name="CnDiff"
description="run"
device=0

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
    --batch_size 16 \
    --des "$description" \
    --learning_rate 0.001 \
    --device "cuda:$device" \
    --gpu "$device" \
    --train_epochs 100
done
