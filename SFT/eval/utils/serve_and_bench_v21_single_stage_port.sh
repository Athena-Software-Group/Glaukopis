#!/bin/bash

# Serve one checkpoint from the cross-model single-pass comparison and run the
# full suite under one warm vLLM session: AthenaBench (incl. Canonical TAA),
# CyberMetric 2K/10K, CyberSOCEval malware/TI and MMLU-Pro. Tears the server
# down on exit.
#
# Use it for both sides of each comparison, so they share one serving stack:
#   single pass  athena-cti-sft-<model>-v21-single-stage-vllm
#   layered      athena-cti-sft-{foundation-8b,llama31-8b,qwen25-14b}-v21-recalibrate-vllm
#                athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm
#
# Serving matches the v21 chain wrappers: --tp 2 --max-len 32768, except Qwen3,
# which keeps its automatic context length and serves with thinking on and the
# qwen3 reasoning parser, like its base.
#
# Usage:
#   conda activate vllm
#   MODEL_ALIAS=<alias> BENCH_CONDA_ENV=ctibench \
#     bash serve_and_bench_v21_single_stage_port.sh [extra-flags]
#
# Pre-flight: the orchestrator aborts if the HF repo does not exist yet.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"
: "${MODEL_ALIAS:?Set MODEL_ALIAS to a v21 single-pass or layered comparator alias}"

SERVE_ARGS=( --tp 2 --max-len 32768 )
case "${MODEL_ALIAS}" in
    *qwen3-30b-a3b-thinking*)
        export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:---reasoning-parser qwen3}"
        SERVE_ARGS=( --tp 2 ) ;;
esac

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model "${MODEL_ALIAS}" \
    "${SERVE_ARGS[@]}" \
    --cybermetric-size 2000,10000 \
    --include-mmlu-pro \
    "$@"
