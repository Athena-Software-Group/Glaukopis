#!/bin/bash

# Per-model wrapper: serve v19-cse (v19-taa + v19 Stage 4 CSE letter-set
# drill on the v19_cse shard, recipe byte-identical to v18-cse / v17.1)
# on 2xH100 and run the full benchmark suite (AthenaBench + CyberMetric
# 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one warm vLLM
# session. Tears the server down on exit.
#
# Stage sign-off question (v19_plan.txt §5.3, README §1): does the
# v18.1 CSE letter-set drill recipe reproduce on top of v19-taa
# (CSE-TI >= 34.0, CSE-Malware >= 20.0) while keeping the v19-core
# gains within 2 pp on MCQ + TAA + RCM? Stage-3 RMS / ATE / VSP erosion
# is expected here and is the target of Stage 5 (v19-recalibrate); do
# NOT grade this checkpoint against the headline RMS/ATE/VSP gates.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen2.5-14B-on-2xH100 defaults baked in. Any
# extra flags are forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_cse.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_cse.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_cse.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v19_cse.sh --skip-cybersoceval
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-
# cti-sft-qwen25-14b-v19-cse is not yet on Hugging Face. Confirm the
# v19-cse autotrain push (which chains off v19-taa) has completed
# before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v19-cse-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
