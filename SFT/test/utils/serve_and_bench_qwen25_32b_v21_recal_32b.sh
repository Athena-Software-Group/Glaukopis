#!/bin/bash

# Per-model wrapper: serve v21-recal-32b (Qwen2.5-32B, 32B-recipe
# variant of the off-plan Stage 4 touch-up; lr 3e-6, mix 0.15/0.60/0.25
# Phase A/B/TAA, max-samples 3600, cutoff 16384, packing off) on 2xH100
# and run the full benchmark suite (AthenaBench + CyberMetric 2K +
# CyberMetric 10K + CyberSOCEval malware/TI) under one warm vLLM
# session. Tears the server down on exit.
#
# recal-32b is OFF-PLAN vs v21_plan.txt §3 and a parallel branch off
# v21-cse alongside the existing qwen25-32b-v21-recalibrate (which uses
# the 14B recipe verbatim). Naming reflects RECIPE PROVENANCE, not chain
# position. Background: the 14B-recipe port (lr 1e-6, mix 0.25/0.40/0.35)
# failed to recover VSP at the 32B scale (post-cse VSP 78.9 -> post-recal
# VSP 75.7, vs the 14B chain's 72.9 -> 83.1 lift). The 32B recipe re-
# weights the interleave toward Phase B (the VSP/RMS catalog shard) and
# bumps the LR 3x to clear the 32B + adamw_8bit optimizer noise floor
# while holding step count and wall-time constant. Treat as a diagnostic
# A/B against qwen25-32b-v21-recalibrate; if VSP recovers without
# sacrificing the cse-stage gains, this becomes the 32B ship candidate.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen2.5-32B-on-2xH100 defaults baked in. Any
# extra flags are forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_recal_32b.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_recal_32b.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_recal_32b.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen25_32b_v21_recal_32b.sh --skip-cybersoceval
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
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-
# cti-sft-qwen25-32b-v21-recal-32b is not yet on Hugging Face. Confirm
# the recal-32b push (run_sft_qwen25_32b_v21_recal_32b.sh, which
# branches off v21-cse) has completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-32b-v21-recal-32b-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
