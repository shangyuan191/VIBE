# ShangYuan vLLM Benchmark

This repo collects the NCHC vLLM benchmark assets for independent serving, PD disaggregation, and follow-up comparisons. The detailed run logs live in the note files; this README is only the landing page and quick navigation map.

## What is here

- [compose](compose) for the `docker-compose.*.yml` files.
- [configs](configs) for the `.env*` environment presets.
- [scripts](scripts) for benchmark generation, validation, and plotting.
- [results](results) for sample JSON and PNG outputs.
- [note](note) for the chronological experiment logs.

## Quick start

1. Pick the experiment family you want to run.
2. Open the matching note below for the exact environment, command sequence, and troubleshooting details.
3. Start the compose stack or benchmark script from this repo root.
4. Compare the generated artifacts in `results/` with the examples already checked in.

## Experiment map

| Experiment | Main entry point | Read this note |
| --- | --- | --- |
| PD disaggregation bring-up and debugging | `compose/docker-compose.4p4d-read.yml`, `moriio_connector.py`, `moriio_engine.py` | [2026-05-28 PD disaggregation debug log](note/2026-05-28_pd-disaggregation-debug-log.md) |
| GLM-5-FP8 serving example and workload setup | `scripts/run_pd_metrics.sh` and the note commands | [2026-05-21 GLM-5-FP8 serving note](note/2026-05-21_glm5-fp8-serving-and-prefill-decode-benchmarks.md) |
| Independent 8-GPU baseline | `scripts/run_and_summarize_independent_8gpu_rr_metrics.sh` | [2026-06-11 independent 8-GPU baseline](note/2026-06-11_independent-8gpu-baseline-and-4p4d-benchmark.md) |
| 4P4D READ benchmark and comparison | `scripts/run_and_summarize_4p4d_read_metrics.sh` | [2026-06-18 4P4D vs independent comparison](note/2026-06-18_4p4d-vs-independent-8gpu-comparison.md) |
| Independent 6-GPU baseline and 2P4D / 4P2D comparison | `scripts/run_and_summarize_independent_6gpu_rr_metrics.sh`, `scripts/run_and_summarize_2p4d_read_metrics.sh`, `scripts/run_and_summarize_4p2d_read_metrics.sh` | [2026-06-25 6-GPU and P/D ratio comparison](note/2026-06-25_independent-6gpu-and-2p4d-4p2d-comparison.md) |

## Recommended run order

If you are trying to understand the project from scratch, read the notes in this order:

1. [2026-05-28 PD disaggregation debug log](note/2026-05-28_pd-disaggregation-debug-log.md)
2. [2026-06-04 independent 8-GPU baseline and model validation](note/2026-06-04_independent-8gpu-baseline-and-model-validation.md)
3. [2026-06-11 independent 8-GPU baseline and 4P4D benchmark](note/2026-06-11_independent-8gpu-baseline-and-4p4d-benchmark.md)
4. [2026-06-18 4P4D vs independent 8-GPU comparison](note/2026-06-18_4p4d-vs-independent-8gpu-comparison.md)
5. [2026-06-25 independent 6-GPU and 2P4D / 4P2D comparison](note/2026-06-25_independent-6gpu-and-2p4d-4p2d-comparison.md)

## Useful outputs

- `results/` contains the checked-in benchmark summaries and plots.
- `scripts/plot_benchmark_metrics.py` turns metric JSON files into report images.
- The benchmark shell scripts now live under `scripts/` and write outputs under `results/`.

## Notes

- The repo keeps the low-level commands in the note files so the README stays short.
- If you change a benchmark layout or output naming scheme, update both the note and the corresponding script together.
