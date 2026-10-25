#!/bin/bash

# Per-model wrapper: serve the v21 matched single-stage control (Qwen2.5-32B,
# one pass over the union of the v21 Stage 1-3 shards; trained by
# SFT/autotrain/run_sft_qwen25_32b_v21_single_stage.sh) on 2xH100 and run the
# full suite the v21 chain checkpoints were scored on: AthenaBench (incl.
# Canonical TAA) + CyberMetric 2K/10K + CyberSOCEval malware/TI + MMLU-Pro,
# under one warm vLLM session. Tears the server down on exit.
#
# Serving settings are identical to serve_and_bench_qwen25_32b_v21_*.sh
# (--tp 2 --max-len 32768) so the control is scored exactly like the chain.
# MMLU-Pro is opted in here because the paper reports it for every 32B
# checkpoint; drop --include-mmlu-pro to reproduce the CTI-only sweep.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_single_stage.sh [extra-flags]
#   # LR-sensitivity run (pushed with the -lr5e-06 suffix):
#   MODEL_ALIAS=athena-cti-sft-qwen25-32b-v21-single-stage-lr5e-06-vllm \
#     BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_single_stage.sh
#
# Pre-flight: the orchestrator aborts if the HF repo does not exist yet;
# confirm the training run's push completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"
MODEL_ALIAS="${MODEL_ALIAS:-athena-cti-sft-qwen25-32b-v21-single-stage-vllm}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model "${MODEL_ALIAS}" \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    --include-mmlu-pro \
    "$@"
