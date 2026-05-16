#!/bin/bash

# Per-model wrapper: serve v19-core (Qwen2.5-14B-Instruct + v19 Stage 1+2
# Core chain -- Phase A broad re-anchor on the v19_core_a shard then
# Phase B axis catalog drill on the v19_core_b shard, recipe byte-
# identical to v18.1-core) on 2xH100 and run the full benchmark suite
# (AthenaBench + CyberMetric 2K + CyberMetric 10K + CyberSOCEval
# malware/TI) under one warm vLLM session. Tears the server down on exit.
#
# Stage sign-off question (v19_plan.txt §5.1, README §1): does the v18.1
# two-phase Core recipe regenerate the v18-1-core baseline (CKT >= 70.0,
# RMS >= 55.0, ATE >= 60.0, VSP >= 80.0, RCM >= 67.5) when built from a
# clean v19 checkout with the v19-named shards? This stage is a
# diagnostic checkpoint; TAA Classic / CSE axes are expected to be at
# Qwen2.5-14B-Instruct base levels until Stages 3 and 4 install them.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen2.5-14B-on-2xH100 defaults baked in. Any
# extra flags are forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_core.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_core.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_core.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_core.sh --skip-cybersoceval
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-
# cti-sft-qwen25-14b-v19-core is not yet on Hugging Face. Confirm the
# v19-core autotrain push has completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v19-core-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
