#!/bin/bash

set -e

echo "=========================="
echo "Check Router"
echo "=========================="

curl -s http://127.0.0.1:30000/health
echo ""

echo "=========================="
echo "Check Prefill Metrics"
echo "=========================="

curl -s http://127.0.0.1:20005/metrics | grep kv_cache_usage_perc
echo ""

echo "=========================="
echo "Check Decode Metrics"
echo "=========================="

curl -s http://127.0.0.1:40005/metrics | grep kv_cache_usage_perc
echo ""

echo "=========================="
echo "Run Inference"
echo "=========================="

curl --max-time 120 \
http://127.0.0.1:30000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model":"meta-llama/Llama-3.1-70B-Instruct",
  "messages":[
    {
      "role":"user",
      "content":"Say hello in one sentence."
    }
  ],
  "max_tokens":32,
  "temperature":0
}'

echo ""