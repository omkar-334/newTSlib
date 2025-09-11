#!/bin/bash
set -e

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

echo ">>> Finished running remaining scripts ( Traffic, Weather)!"
