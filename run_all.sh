#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <task_subdir>"
    exit 1
fi

TASK_DIR="$1"

if [ ! -d "$TASK_DIR" ]; then
    echo "Error: Directory $TASK_DIR does not exist."
    exit 1
fi

echo "Running all .sh scripts under $TASK_DIR..."

find "$TASK_DIR" -type f -iname "*.sh" | while read script; do
    echo "Running: $script"
    bash "$script"
done
