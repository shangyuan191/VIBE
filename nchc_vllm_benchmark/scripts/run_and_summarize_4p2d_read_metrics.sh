#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========== Run 4P2D READ Mode Benchmark with Metrics =========="
bash "$SCRIPT_DIR/run_4p2d_read_metrics.sh"

echo ""
echo "========== Summarize Results =========="
bash "$SCRIPT_DIR/summarize_4p2d_read_metrics.sh"

python3 "$SCRIPT_DIR/plot_benchmark_metrics.py" \
  --only-png \
  --pattern "$ROOT_DIR/results/4p2d_read_70B_c*_metrics.json" \
  --out-png "$ROOT_DIR/results/4p2d_read_70B_report.png" \
  --title "4P2D READ Mode Benchmark | Llama-3.1-70B-Instruct"