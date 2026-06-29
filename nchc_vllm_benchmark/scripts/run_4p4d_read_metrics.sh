#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="meta-llama/Llama-3.1-70B-Instruct"
URL="http://127.0.0.1:30000/v1/chat/completions"

mkdir -p "$ROOT_DIR/results"

for C in 16 32 64 128 256 512
do
  echo ""
  echo "========================================"
  echo "Running 4P4D READ Mode with Metrics"
  echo "Concurrency: ${C}"
  echo "========================================"

  python3 "$SCRIPT_DIR/quick_bench_openai.py" \
    --url "${URL}" \
    --metrics-urls \
      "http://127.0.0.1:20005/metrics" \
      "http://127.0.0.1:40005/metrics" \
    --model "${MODEL}" \
    --name "$ROOT_DIR/results/4p4d_read_70B_c${C}_metrics" \
    --num-requests 1024 \
    --concurrency "${C}" \
    --input-chars 12000 \
    --max-tokens 128 \
    --metrics-interval 0.2
done