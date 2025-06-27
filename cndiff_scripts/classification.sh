#!/bin/bash

# model_name="CnDiff"
description="base"
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
  for model in "CnDiff" "CnDiffOld" ; do
    for classifier in 1 2; do
      for use_cond in 1 2; do
        python -u run.py \
          --task_name classification \
          --is_training 1 \
          --root_path ./dataset/"$dataset"/ \
          --model_id "$dataset" \
          --model "$model" \
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
          --use_tphi True \
          --classifier $classifier \
          --wandb False \
          --use_cond "$use_cond"
      done
    done
  done
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