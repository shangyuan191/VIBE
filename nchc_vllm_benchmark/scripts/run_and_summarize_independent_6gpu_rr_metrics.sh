#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========== Run Independent 6GPU RR Benchmark with Metrics =========="
bash "$SCRIPT_DIR/run_independent_6gpu_rr_metrics.sh"

echo ""
echo "========== Summarize Results =========="
bash "$SCRIPT_DIR/summarize_independent_6gpu_rr_metrics.sh"

python3 "$SCRIPT_DIR/plot_benchmark_metrics.py" \
  --only-png \
  --pattern "$ROOT_DIR/results/independent_6gpu_rr_70B_c*_metrics.json" \
  --out-png "$ROOT_DIR/results/independent_6gpu_rr_70B_report.png" \
  --title "Independent 6-GPU Round-Robin Benchmark | Llama-3.1-70B-Instruct"