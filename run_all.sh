#!/bin/bash

TASK_SUBDIR=$1

if [ -z "$TASK_SUBDIR" ]; then
  echo "Usage: $0 <task_subdir>"
  exit 1
fi

TASK_PATH="./scripts/$TASK_SUBDIR"

if [ ! -d "$TASK_PATH" ]; then
  echo "Directory $TASK_PATH does not exist."
  exit 1
fi

# Make all files executable
chmod +x "$TASK_PATH"/*

# Run all files
for script in "$TASK_PATH"/*; do
  if [ -f "$script" ]; then
    echo "Running $script..."
    "$script"
    echo "Finished $script"
    echo "-----------------"
  fi
done
