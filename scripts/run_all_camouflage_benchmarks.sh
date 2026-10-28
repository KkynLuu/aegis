#!/usr/bin/env bash
# Driver script to run 10-seed camouflage stress tests across all benchmark datasets:
#   1. CTU-13 Scenario 11 (Rbot IRC)
#   2. CTU-13 Scenario 9 (Dense Volumetric Spam)
#   Followed by generating the consolidated 4-panel publication figure.

set -e

PYTHON_BIN=".pixi/envs/default/bin/python"

echo "=========================================================="
echo " Starting Multi-Dataset Camouflage Benchmark Execution"
echo "=========================================================="

echo "[1/3] Running CTU-13 Scenario 11 (10 seeds, 35 epochs)..."
$PYTHON_BIN benchmarks/benchmark_camouflage.py --dataset ctu13_11 --num_seeds 10 --epochs 35

echo "[2/3] Running CTU-13 Scenario 9 (10 seeds, 35 epochs)..."
$PYTHON_BIN benchmarks/benchmark_camouflage.py --dataset ctu13_9 --num_seeds 10 --epochs 35

echo "[3/3] Generating Consolidated 4-Panel Publication Figure..."
$PYTHON_BIN scripts/plot_multidataset_camouflage.py

echo "=========================================================="
echo " Multi-Dataset Camouflage Benchmark Execution Complete!"
echo "=========================================================="
