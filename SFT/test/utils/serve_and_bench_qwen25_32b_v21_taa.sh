#!/bin/bash

# Per-model wrapper: serve v21-TAA (Qwen2.5-32B, v21-core + TAA Classic
# narrow drill) on 2xH100 and run the full benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session. Tears the server down on exit.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite orchestrator)
# with the Qwen2.5-32B-on-2xH100 defaults baked in. Any extra flags are
# forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_taa.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_taa.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_taa.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_taa.sh --skip-cybersoceval
#
# Context window:
#   Qwen2.5-32B-Instruct native ctx is 32768; --max-len pinned at 32768
#   (matches the v21 SFT cutoff envelope; CSE-TI rows that exceed are
#   caught by the client-side ctx-overflow path in pipelines/models.py).
#
# Wallclock estimate (2xH100, Qwen2.5-32B dense):
#   Athena (~1.2 h) + CM-2K (~45 min) + CM-10K (~3.5 h) + CSE (~5.5 h)
#   ~ 11 h total per model.
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-cti-sft-
# qwen25-32b-v21-taa is not yet on Hugging Face. Watch for the v21-TAA
# autotrain run on the SFT box to push first (Stage 2 of the v21 chain).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-32b-v21-taa-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
