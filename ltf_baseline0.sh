#!/bin/bash
set -e

echo ">>> Running all scripts in ECL_script"
for f in scripts/LTF/ECL_script/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Running all scripts in ETTh1"
for f in scripts/LTF/ETTh1/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Running all scripts in Exchange_script"
for f in scripts/LTF/Exchange_script/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Finished running ECL, ETTh1, Exchange scripts!"
