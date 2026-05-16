#!/bin/bash

# Per-model wrapper: serve v19-recalibrate-v18p2mix (Qwen2.5-14B, v19-cse +
# Stage 5 3-shard interleaved replay with the v18.2 prob mix 0.25/0.40/0.35
# at v18.2-matched step count, lr 1e-6, cutoff 16384, packing off -- the
# prob-mix-isolation variant of the v19 recalibrate) on 2xH100 and run the
# full benchmark suite (AthenaBench + CyberMetric 2K + CyberMetric 10K +
# CyberSOCEval malware/TI) under one warm vLLM session. Tears the server
# down on exit.
#
# Headline gate (v19_plan.txt §5.4, carried verbatim from v18.2 §7.4):
#   RMS >= 54.0, MCQ >= 62.0, TAA Classic >= 40.0, CSE-TI >= 34.0,
#   CSE-Malware >= 20.0, ATE >= 62.0, RCM >= 67.5, VSP >= 80.0,
#   CyberMetric-2K >= 85.5, CyberMetric-10K >= 81.0.
#
# Comparison framing (the entire purpose of this variant):
#   - vs v19-recalibrate (same v19-cse base, equal-weight 0.33/0.33/0.34):
#       delta isolates the prob-mix contribution.
#   - vs v18-2 (v18.1-cse base, same 0.25/0.40/0.35 mix, same step count):
#       delta isolates the base-checkpoint contribution.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite orchestrator)
# with Qwen2.5-14B-on-2xH100 defaults: --tp 2 --max-len 32768.
#
# Usage:
#   conda activate vllm
#   cd ~/Glaukopis/SFT/test
#   BENCH_CONDA_ENV=ctibench bash utils/serve_and_bench_v19_recalibrate_v18p2mix.sh
#
#   # Clean re-bench (overwrite prior response files):
#   BENCH_CONDA_ENV=ctibench bash utils/serve_and_bench_v19_recalibrate_v18p2mix.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash utils/serve_and_bench_v19_recalibrate_v18p2mix.sh --skip-cybersoceval
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-cti-sft-
# qwen25-14b-v19-recalibrate-v18p2mix is not yet on Hugging Face. Confirm the
# v19-recalibrate-v18p2mix autotrain push has completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v19-recalibrate-v18p2mix-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
