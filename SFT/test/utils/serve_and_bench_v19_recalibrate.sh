#!/bin/bash

# Per-model wrapper: serve v19-recalibrate (Qwen2.5-14B, v19-cse + Stage 5
# equal-weight 3-shard interleaved replay, probs 0.33/0.33/0.34, lr 1e-6,
# cutoff 16384, packing off -- the v19 published headline) on 2xH100 and
# run the full benchmark suite (AthenaBench + CyberMetric 2K + CyberMetric
# 10K + CyberSOCEval malware/TI) under one warm vLLM session. Tears the
# server down on exit.
#
# Headline gate (v19_plan.txt §5.4, carried verbatim from v18.2 §7.4):
#   RMS >= 54.0, MCQ >= 62.0, TAA Classic >= 40.0, CSE-TI >= 34.0,
#   CSE-Malware >= 20.0, ATE >= 62.0, RCM >= 67.5, VSP >= 80.0,
#   CyberMetric-2K >= 85.5, CyberMetric-10K >= 81.0.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite orchestrator)
# with the Qwen2.5-14B-on-2xH100 defaults baked in. Any extra flags are
# forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_recalibrate.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_recalibrate.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_recalibrate.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_recalibrate.sh --skip-cybersoceval
#
# Wallclock estimate (2xH100, Qwen2.5-14B): roughly the v18.2 envelope
# (v18.2 finished the full sweep in ~8 min; bench wallclock is determined
# by inference, not training).
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-cti-sft-
# qwen25-14b-v19-recalibrate is not yet on Hugging Face. Confirm the
# v19-recalibrate autotrain push has completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v19-recalibrate-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
