#!/bin/bash

set -e

curl --max-time 120 http://127.0.0.1:30000/v1/chat/completions \
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