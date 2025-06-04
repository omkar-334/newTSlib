#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <task_subdir> [gpu_number]"
    exit 1
fi

TASK_DIR="$1"
GPU_NUMBER="${2:-0}"  # Default to GPU 0 if not specified

if [ ! -d "$TASK_DIR" ]; then
    echo "Error: Directory $TASK_DIR does not exist."
    exit 1
fi

echo "Running all .sh scripts under $TASK_DIR using GPU $GPU_NUMBER..."

find "$TASK_DIR" -type f -iname "*.sh" | while read -r script; do
    echo "Running: $script with CUDA_VISIBLE_DEVICES=$GPU_NUMBER"
    CUDA_VISIBLE_DEVICES="$GPU_NUMBER" bash "$script"
done
