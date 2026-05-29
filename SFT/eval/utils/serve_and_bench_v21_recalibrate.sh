#!/bin/bash

# Per-model wrapper: serve v21-recalibrate (v21-cse + the off-plan v21
# Stage 4 three-shard interleaved touch-up with the v18.2-style prob mix
# 0.25/0.40/0.35 at lr 1e-6, cutoff 16384, packing off) on 2xH100 and
# run the full benchmark suite (AthenaBench + CyberMetric 2K +
# CyberMetric 10K + CyberSOCEval malware/TI) under one warm vLLM
# session. Tears the server down on exit.
#
# Recalibrate is OFF-PLAN vs v21_plan.txt §3 (which defines only Core /
# TAA / CSE for v18.1 parity); it is benched only if v21-cse sign-off
# exposes the same Phase B / catalog erosion v20-cse showed against
# v18.2. Treat as a diagnostic comparator, not a ship candidate.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen2.5-14B-on-2xH100 defaults baked in. Any
# extra flags are forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v21_recalibrate.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v21_recalibrate.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v21_recalibrate.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v21_recalibrate.sh --skip-cybersoceval
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-
# cti-sft-qwen25-14b-v21-recalibrate is not yet on Hugging Face. Confirm
# the v21-recalibrate autotrain push (which chains off v21-cse) has
# completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v21-recalibrate-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
