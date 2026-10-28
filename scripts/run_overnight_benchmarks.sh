#!/usr/bin/env bash
# ==============================================================================
# AEGIS Overnight Benchmarking Suite for IEEE TIFS
# Executes all required re-runs with fixed code, bug-free camouflage, and
# verified standalone ablation architectures.
#
# Prevents system sleep using caffeinate when invoked with:
#   caffeinate -dimsu bash scripts/run_overnight_benchmarks.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.pixi/envs/default/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

export PYTHONPATH=".:$PYTHONPATH"
export PYTHONUNBUFFERED="1"

echo "=============================================================================="
echo "  STARTING AEGIS OVERNIGHT BENCHMARK EXECUTION"
echo "  Timestamp: $(date)"
echo "  Python:    $PYTHON_BIN"
echo "  Directory: $REPO_ROOT"
echo "=============================================================================="

mkdir -p benchmarks/logs

# ------------------------------------------------------------------------------
# STEP 2: MoE Dual-Expert Standalone Ablation Study (Table IV Verification)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "  [STAGE 1/3] STEP 2: MoE DUAL-EXPERT STANDALONE ABLATION (35 EPOCHS, 5 SEEDS)"
echo "=============================================================================="
"$PYTHON_BIN" benchmarks/benchmark_aegis_ablation.py \
    --seeds 42 43 44 45 46 \
    --epochs 35 \
    --device mps 2>&1 | tee benchmarks/logs/ablation_overnight.log

# ------------------------------------------------------------------------------
# STEP 3: Multi-Dataset Topological Camouflage Sweeps (Figure 2 Verification)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "  [STAGE 2/3] STEP 3: MULTI-DATASET CAMOUFLAGE SWEEPS (ALL 4 DATASETS, 5 SEEDS)"
echo "=============================================================================="
"$PYTHON_BIN" benchmarks/run_aegis_camouflage_sweeps.py \
    --seeds 42 43 44 45 46 \
    --device mps 2>&1 | tee benchmarks/logs/camouflage_overnight.log

# ------------------------------------------------------------------------------
# STEP 4: Cross-Scenario Zero-Shot Botnet Transferability (Table V Verification)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "  [STAGE 3/3] STEP 4: CROSS-SCENARIO TRANSFER (S10 -> S11, 35 EPOCHS, 5 SEEDS)"
echo "=============================================================================="
"$PYTHON_BIN" benchmarks/run_cross_scenario_eval.py \
    --num_seeds 5 \
    --epochs 35 \
    --batch_size 32 \
    --device mps 2>&1 | tee benchmarks/logs/cross_scenario_overnight.log

# ------------------------------------------------------------------------------
# STEP 5: Verification and Final Figure Re-plotting
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "  [STAGE 4/4] REGENERATING ALL PUBLICATION FIGURES & VERIFYING METRICS"
echo "=============================================================================="
"$PYTHON_BIN" scripts/plot_multidataset_camouflage.py

echo ""
echo "=============================================================================="
echo "  [SUCCESS] ALL OVERNIGHT BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "  Completion Time: $(date)"
echo "=============================================================================="
