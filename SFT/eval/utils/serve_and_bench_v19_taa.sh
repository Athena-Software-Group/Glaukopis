#!/bin/bash

# Per-model wrapper: serve v19-taa (v19-core + v19 Stage 3 TAA Classic
# narrow drill on the v19_taa shard, recipe byte-identical to v18-taa
# / v16) on 2xH100 and run the full benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session. Tears the server down on exit.
#
# Stage sign-off question (v19_plan.txt §5.2, README §1): does the
# v18.1 TAA Classic narrow drill recipe reproduce on top of v19-core
# (TAA Classic >= 40.0) without regressing the stage-1+2 axes (MCQ,
# RMS, ATE, VSP, RCM, CKT) by more than 2 pp? CSE axes are still
# expected at base levels until Stage 4.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen2.5-14B-on-2xH100 defaults baked in. Any
# extra flags are forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_taa.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_taa.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_taa.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_taa.sh --skip-cybersoceval
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-
# cti-sft-qwen25-14b-v19-taa is not yet on Hugging Face. Confirm the
# v19-taa autotrain push (which chains off v19-core) has completed
# before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v19-taa-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
