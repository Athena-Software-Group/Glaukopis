#!/bin/bash

# Per-model wrapper: serve v20-cse (v20-taa + v20 Stage 4 CyberSOCEval
# letter-set narrow drill on the v20_cse shard; recipe byte-identical to
# v19-cse) on 2xH100 and run the full benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session. Tears the server down on exit.
#
# Stage sign-off question (v20_plan.txt §5.3): does the v19 CSE letter-
# set narrow drill recipe reproduce on top of v20-taa (CSE-TI >= 34.0,
# CSE-Malware >= 20.0) without regressing the stage 1-3 axes (MCQ, RMS,
# ATE, VSP, RCM, CKT, TAA Classic) by more than 2 pp? Stage 5
# (Recalibrate) is expected to recover any residual axis erosion via the
# interleaved replay.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen2.5-14B-on-2xH100 defaults baked in. Any
# extra flags are forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v20_cse.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v20_cse.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v20_cse.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v20_cse.sh --skip-cybersoceval
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-
# cti-sft-qwen25-14b-v20-cse is not yet on Hugging Face. Confirm the
# v20-cse autotrain push (which chains off v20-taa) has completed
# before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v20-cse-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
