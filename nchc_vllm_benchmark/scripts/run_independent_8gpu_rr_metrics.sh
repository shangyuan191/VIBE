#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="meta-llama/Llama-3.1-70B-Instruct"

URLS=(
  "http://127.0.0.1:8000/v1/chat/completions"
  "http://127.0.0.1:8001/v1/chat/completions"
  "http://127.0.0.1:8002/v1/chat/completions"
  "http://127.0.0.1:8003/v1/chat/completions"
  "http://127.0.0.1:8004/v1/chat/completions"
  "http://127.0.0.1:8005/v1/chat/completions"
  "http://127.0.0.1:8006/v1/chat/completions"
  "http://127.0.0.1:8007/v1/chat/completions"
)

mkdir -p "$ROOT_DIR/results"

for C in 16 32 64 128 256 512
do
  echo ""
  echo "========================================"
  echo "Running Independent 8GPU RR with Metrics"
  echo "Concurrency: ${C}"
  echo "========================================"

  python3 "$SCRIPT_DIR/quick_bench_openai.py" \
    --urls "${URLS[@]}" \
    --model "${MODEL}" \
    --name "$ROOT_DIR/results/independent_8gpu_rr_70B_c${C}_metrics" \
    --num-requests 1024 \
    --concurrency "${C}" \
    --input-chars 12000 \
    --max-tokens 128 \
    --metrics-interval 0.2

done
