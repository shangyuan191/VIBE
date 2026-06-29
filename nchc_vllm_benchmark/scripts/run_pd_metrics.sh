#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <experiment_name>"
  echo ""
  echo "Example:"
  echo "  $0 4p4d_read"
  echo "  $0 2p6d_read"
  echo "  $0 3p5d_read"
  echo "  $0 5p3d_read"
  echo "  $0 6p2d_read"
  exit 1
fi

EXP_NAME="$1"

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
URL="${URL:-http://127.0.0.1:30000/v1/chat/completions}"

METRICS_URLS="${METRICS_URLS:-http://127.0.0.1:20005/metrics http://127.0.0.1:40005/metrics}"

NUM_REQUESTS="${NUM_REQUESTS:-1024}"
INPUT_CHARS="${INPUT_CHARS:-12000}"
MAX_TOKENS="${MAX_TOKENS:-128}"
METRICS_INTERVAL="${METRICS_INTERVAL:-0.2}"

CONCURRENCY_LIST="${CONCURRENCY_LIST:-16 32 64 128 256 512}"

mkdir -p "$ROOT_DIR/results"

for C in ${CONCURRENCY_LIST}
do
  echo ""
  echo "========================================"
  echo "Running ${EXP_NAME} with Metrics"
  echo "Concurrency: ${C}"
  echo "URL: ${URL}"
  echo "Metrics URLs: ${METRICS_URLS}"
  echo "========================================"

  python3 "$SCRIPT_DIR/quick_bench_openai.py" \
    --url "${URL}" \
    --metrics-urls ${METRICS_URLS} \
    --model "${MODEL}" \
    --name "$ROOT_DIR/results/${EXP_NAME}_70B_c${C}_metrics" \
    --num-requests "${NUM_REQUESTS}" \
    --concurrency "${C}" \
    --input-chars "${INPUT_CHARS}" \
    --max-tokens "${MAX_TOKENS}" \
    --metrics-interval "${METRICS_INTERVAL}"
done