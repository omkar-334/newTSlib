#!/bin/bash
set -e

echo ">>> Running all scripts in ECL_script"
for f in scripts/LTF/ECL_script/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Running all scripts in ETTm1"
for f in scripts/LTF/ETTm1/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Running all scripts in Exchange_script"
for f in scripts/LTF/Exchange_script/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Running all scripts in Traffic_script"
for f in scripts/LTF/Traffic_script/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Running all scripts in Weather_script"
for f in scripts/LTF/Weather_script/*.sh; do
    echo "Running $f"
    bash "$f"
done

echo ">>> Finished running ECL, ETTm1 scripts!"
