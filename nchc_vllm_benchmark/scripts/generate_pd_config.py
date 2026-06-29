#!/usr/bin/env python3

import argparse
from pathlib import Path

BASE_ENV = """PD_MODEL=meta-llama/Llama-3.1-70B-Instruct
HF_HOME=/app/model
HOST_HF_CACHE=./hf_cache
VLLM_IMAGE=vllm/vllm-openai-rocm:nightly
ROUTER_IMAGE=vllm/vllm-router:nightly
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.85
DTYPE=bfloat16
VLLM_MORIIO_CONNECTOR_READ_MODE=1
"""

COMPOSE_TEMPLATE = """services:
  router:
    image: ${{ROUTER_IMAGE}}
    container_name: vllm-router-moriio
    network_mode: host
    command: >
      vllm-router
      --host 0.0.0.0
      --port 30000
      --vllm-pd-disaggregation
      --kv-connector moriio
      --vllm-discovery-address 0.0.0.0:36367

  prefill:
    image: ${{VLLM_IMAGE}}
    container_name: vllm-{exp_name}-prefill
    network_mode: host
    ipc: host
    group_add:
      - video
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    devices:
      - /dev/kfd
      - /dev/dri
    volumes:
      - ${{HOST_HF_CACHE}}:${{HF_HOME}}
    environment:
      VLLM_MORIIO_CONNECTOR_READ_MODE: ${{VLLM_MORIIO_CONNECTOR_READ_MODE}}
      HF_HOME: ${{HF_HOME}}
      VLLM_ROCM_USE_AITER: "1"
      HIP_VISIBLE_DEVICES: "{prefill_gpus}"
    command: >
      ${{PD_MODEL}}
      --tensor-parallel-size {p}
      --host 0.0.0.0
      --port 20005
      --dtype ${{DTYPE}}
      --gpu-memory-utilization ${{GPU_MEMORY_UTILIZATION}}
      --max-model-len ${{MAX_MODEL_LEN}}
      --kv-transfer-config
      '{{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_connector_extra_config":{{"backend":"xgmi","read_mode":true,"proxy_ip":"127.0.0.1","proxy_ping_port":"36367","http_port":"20005","handshake_port":"6301","notify_port":"6105"}}}}'
    depends_on:
      - router

  decode:
    image: ${{VLLM_IMAGE}}
    container_name: vllm-{exp_name}-decode
    network_mode: host
    ipc: host
    group_add:
      - video
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    devices:
      - /dev/kfd
      - /dev/dri
    volumes:
      - ${{HOST_HF_CACHE}}:${{HF_HOME}}
    environment:
      VLLM_MORIIO_CONNECTOR_READ_MODE: ${{VLLM_MORIIO_CONNECTOR_READ_MODE}}
      HF_HOME: ${{HF_HOME}}
      VLLM_ROCM_USE_AITER: "1"
      HIP_VISIBLE_DEVICES: "{decode_gpus}"
    command: >
      ${{PD_MODEL}}
      --tensor-parallel-size {d}
      --host 0.0.0.0
      --port 40005
      --dtype ${{DTYPE}}
      --gpu-memory-utilization ${{GPU_MEMORY_UTILIZATION}}
      --max-model-len ${{MAX_MODEL_LEN}}
      --kv-transfer-config
      '{{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_connector_extra_config":{{"backend":"xgmi","read_mode":true,"proxy_ip":"127.0.0.1","proxy_ping_port":"36367","http_port":"40005","handshake_port":"7301","notify_port":"7501"}}}}'
    depends_on:
      - router
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--mode", default="read")
    args = parser.parse_args()

    if args.p + args.d != 8:
        raise ValueError("p + d must equal 8 for single-node 8-GPU MI300X setup")

    exp_name = f"{args.p}p{args.d}d_{args.mode}"

    prefill_gpus = ",".join(str(i) for i in range(args.p))
    decode_gpus = ",".join(str(i) for i in range(args.p, args.p + args.d))

    Path("configs").mkdir(exist_ok=True)
    Path("compose").mkdir(exist_ok=True)

    env_path = Path(f"configs/.env.{args.p}p{args.d}d")
    compose_path = Path(f"compose/docker-compose.{args.p}p{args.d}d-{args.mode}.yml")

    env_path.write_text(BASE_ENV)

    compose = COMPOSE_TEMPLATE.format(
        exp_name=exp_name,
        p=args.p,
        d=args.d,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
    )
    compose_path.write_text(compose)

    print(f"Generated: {env_path}")
    print(f"Generated: {compose_path}")
    print()
    print(f"Experiment name: {exp_name}")
    print(f"Prefill GPUs: {prefill_gpus}, TP={args.p}")
    print(f"Decode GPUs:  {decode_gpus}, TP={args.d}")

if __name__ == "__main__":
    main()