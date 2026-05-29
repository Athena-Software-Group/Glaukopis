#!/bin/bash

# Per-model wrapper: serve v18.2.2 (Qwen2.5-14B, v18.1-cse + Stage 4 multi-shard
# replay with the v18.2 prob mix at HALF the step count -- the v18.2.2 ship
# candidate) on 2xH100 and run the full benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one warm
# vLLM session. Tears the server down on exit.
#
# v18.2.2 supersedes v18.2.1 by reverting to v18.2's prob mix (0.25/0.40/0.35)
# while cutting --max-samples 3000 -> 1500 per dataset. The trade-ratio
# analysis (plan §8.2.3) showed v18.2.1's MCQ-for-RMS exchange (|dRMS/dMCQ|
# = 0.45) was half as efficient as v18.2's (0.86), implying v18.2 was on the
# better side of the diminishing-returns curve and that v18.2.1 pushed past
# the optimum on both probs and step count. v18.2.2 tests the inverse
# hypothesis: the v18.2 prob mix is correct and the regression is over-
# exposure damage. Half the steps should preserve the RMS gain while
# reducing MCQ damage and protect ATE/RCM from sliding below their floors.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite orchestrator)
# with the Qwen2.5-14B-on-2xH100 defaults baked in. Any extra flags are
# forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2p2_multi_replay.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2p2_multi_replay.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2p2_multi_replay.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p2p2_multi_replay.sh --skip-cybersoceval
#
# Wallclock estimate (2xH100, Qwen2.5-14B): roughly the v18.2 envelope
# (v18.2 finished the full sweep in ~8 min; v18.2.1 in ~8 min; v18.2.2 should
# be similar -- bench wallclock is determined by inference, not training).
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-cti-sft-
# qwen25-14b-v18-2-2 is not yet on Hugging Face. Confirm the v18.2.2 autotrain
# push has completed before launching.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v18-2-2-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
