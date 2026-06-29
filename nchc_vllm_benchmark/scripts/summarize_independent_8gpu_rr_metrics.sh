#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

for C in 16 32 64 128 256 512
do
    echo ""
    echo "===== independent_8gpu_rr_70B_c${C}_metrics ====="

    python3 - <<EOF_PY
import json

p = r"${ROOT_DIR}/results/independent_8gpu_rr_70B_c${C}_metrics.json"

with open(p) as f:
        d = json.load(f)

m = d.get("metrics_summary", {}).get("overall", {})
derived = d.get("metrics_summary", {}).get("derived", {})

def fmt_unit(x, unit, n=4):
    if x is None:
        return "None"
    if isinstance(x, (int, float)):
        return f"{round(x, n)} {unit}"
    return str(x)

def fmt_percent_from_ratio(x, n=2):
    if x is None:
        return "None"
    if isinstance(x, (int, float)):
        return f"{round(x * 100, n)} %"
    return str(x)

print("completed:", fmt_unit(d.get("completed"), "requests", 0))
print("failed:", fmt_unit(d.get("failed"), "requests", 0))
print("throughput:", fmt_unit(d.get("request_throughput_req_s"), "req/s", 2))

tok = d.get("token_throughput", {})

print("prompt_tokens_total:", fmt_unit(tok.get("prompt_tokens_total"), "tokens", 0))
print("generation_tokens_total:", fmt_unit(tok.get("generation_tokens_total"), "tokens", 0))
print("total_tokens_total:", fmt_unit(tok.get("total_tokens_total"), "tokens", 0))

print("prompt_token_throughput:", fmt_unit(tok.get("prompt_tokens_per_second"), "tok/s", 2))
print("generation_token_throughput:", fmt_unit(tok.get("generation_tokens_per_second"), "tok/s", 2))
print("total_token_throughput:", fmt_unit(tok.get("total_tokens_per_second"), "tok/s", 2))

print("prefill_workload_token_throughput:", fmt_unit(tok.get("prompt_tokens_per_second"), "tok/s", 2))
print("decode_workload_token_throughput:", fmt_unit(tok.get("generation_tokens_per_second"), "tok/s", 2))

print("mean_ttft:", fmt_unit(d.get("ttft_s", {}).get("mean"), "s"))
print("p99_ttft:", fmt_unit(d.get("ttft_s", {}).get("p99"), "s"))
print("mean_e2e:", fmt_unit(d.get("e2e_s", {}).get("mean"), "s"))
print("p99_e2e:", fmt_unit(d.get("e2e_s", {}).get("p99"), "s"))

print("mean_decode_time:", fmt_unit(d.get("decode_time_s", {}).get("mean"), "s"))
print("p99_decode_time:", fmt_unit(d.get("decode_time_s", {}).get("p99"), "s"))
print("mean_itl:", fmt_unit(d.get("itl_s", {}).get("mean"), "s/token"))
print("p99_itl:", fmt_unit(d.get("itl_s", {}).get("p99"), "s/token"))

kv = m.get("vllm:kv_cache_usage_perc", {})
running = m.get("vllm:num_requests_running", {})
waiting = m.get("vllm:num_requests_waiting", {})

print("kv_cache_mean:", fmt_percent_from_ratio(kv.get("mean")))
print("kv_cache_peak:", fmt_percent_from_ratio(kv.get("max")))
print("running_peak:", fmt_unit(running.get("max"), "requests", 0))
print("waiting_peak:", fmt_unit(waiting.get("max"), "requests", 0))

print("prefix_hit_rate:", fmt_percent_from_ratio(derived.get("prefix_cache_hit_rate_total")))
print("external_prefix_hit_rate:", fmt_percent_from_ratio(derived.get("external_prefix_cache_hit_rate_total")))
print("prompt_cache_hit_rate:", fmt_percent_from_ratio(derived.get("prompt_cache_hit_rate_total")))
EOF_PY

done