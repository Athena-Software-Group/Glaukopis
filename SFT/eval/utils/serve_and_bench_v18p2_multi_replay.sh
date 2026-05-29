#!/bin/bash

# Per-model wrapper: serve v18.2 (Qwen2.5-14B, v18.1-cse + Stage 4 multi-shard
# replay -- the v18.2 ship candidate) on 2xH100 and run the full benchmark
# suite (AthenaBench + CyberMetric 2K + CyberMetric 10K + CyberSOCEval
# malware/TI) under one warm vLLM session. Tears the server down on exit.
#
# v18.2 supersedes the cse-rms experiment (v18.1-cse-rms): it adds Phase A
# (MCQ coverage) and standalone TAA shards alongside the Phase B catalog
# replay to protect MCQ and TAA Classic axes that regressed in cse-rms
# while preserving the Phase B RMS/ATE/VSP recovery and the CSE drill gain.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite orchestrator)
# with the Qwen2.5-14B-on-2xH100 defaults baked in. Any extra flags are
# forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2_multi_replay.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2_multi_replay.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2_multi_replay.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2_multi_replay.sh --skip-cybersoceval
#
# Wallclock estimate (2xH100, Qwen2.5-14B):
#   Athena (~50 min) + CM-2K (~30 min) + CM-10K (~2.2 h) + CSE (~3.5 h)
#   ~ 7 h total per model.
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-cti-sft-
# qwen25-14b-v18-2 is not yet on Hugging Face. Confirm the v18.2 autotrain
# push has completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v18-2-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
