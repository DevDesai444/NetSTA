#!/bin/bash
# Run the full NetSTA benchmark suite. Resulting tables and plots land in
# results/. Individual scripts return non-zero on hard failures; the master
# script uses `set -e` so a fatal error halts the suite for visibility.
set -e

# Run from the repo root regardless of where the user invoked this from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

mkdir -p results/plots

echo "=== NetSTA Comprehensive Benchmarking ==="
echo ""
echo "[1/6] Baseline Comparisons..."
python3 scripts/benchmark_baselines.py --num-circuits 1000 --epochs 200
echo ""
echo "[2/6] Robustness Analysis (5 seeds)..."
python3 scripts/benchmark_robustness.py --num-circuits 1000 --epochs 200 --seeds 42,123,456,789,1024
echo ""
echo "[3/6] Scaling Analysis..."
python3 scripts/benchmark_scaling.py --epochs 200
echo ""
echo "[4/6] Ablation Study..."
python3 scripts/benchmark_ablation.py --num-circuits 1000 --epochs 200
echo ""
echo "[5/6] Generalization Tests..."
python3 scripts/benchmark_generalization.py --num-circuits 1000 --epochs 200
echo ""
echo "[6/6] Training Curves..."
python3 scripts/benchmark_training_curves.py --num-circuits 1000 --epochs 200
echo ""
echo "=== All benchmarks complete. Results saved to results/ ==="
echo "Run 'python3 scripts/compile_results.py' to generate the final report."
