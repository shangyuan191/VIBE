# VIBE

**V**LLM **I**nference **B**enchmarking **E**ngine

VIBE is the umbrella repository. It now serves as the landing page for the main benchmarking framework and for the imported NCHC benchmark assets.

## Where To Start

- Main imported benchmark project: [nchc_vllm_benchmark](nchc_vllm_benchmark)
- NCHC benchmark README: [nchc_vllm_benchmark/README.md](nchc_vllm_benchmark/README.md)

If you are looking for the benchmark runs, compose layouts, scripts, notes, and results, start in the NCHC subdirectory above.

## Repository Layout

- [nchc_vllm_benchmark](nchc_vllm_benchmark) contains the imported NCHC vLLM benchmark suite.
- The subproject includes:
  - [compose](nchc_vllm_benchmark/compose) for deployment manifests.
  - [configs](nchc_vllm_benchmark/configs) for environment presets.
  - [scripts](nchc_vllm_benchmark/scripts) for benchmark execution and reporting.
  - [note](nchc_vllm_benchmark/note) for experiment logs and run instructions.
  - [results](nchc_vllm_benchmark/results) for sample metric outputs and plots.

## What This Repo Contains

- vLLM benchmarking workflows.
- Prefill / Decode disaggregation experiments.
- Throughput, TTFT, ITL, E2E, and KV cache analysis.
- Docker Compose based deployment examples.
- ROCm-oriented MI300X benchmark assets.

## Notes

- The imported benchmark content has been sanitized to avoid local credential paths.
- The detailed operational steps live in the NCHC subproject notes, not in this root README.
