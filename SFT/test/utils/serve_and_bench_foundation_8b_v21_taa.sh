#!/bin/bash

# Per-model wrapper: serve v21-TAA (Foundation-Sec-8B, v21-core + TAA
# Classic narrow drill) on 2xH100 and run the full benchmark suite
# (AthenaBench + CyberMetric 2K + CyberMetric 10K + CyberSOCEval
# malware/TI) under one warm vLLM session. Tears the server down on exit.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Foundation-Sec-8B-on-2xH100 defaults baked in.
# Any extra flags are forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_foundation_8b_v21_taa.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Wallclock estimate (2xH100, 8B dense):
#   Athena (~30 min) + CM-2K (~20 min) + CM-10K (~1.5 h) + CSE (~2.5 h)
#   ~ 4.5-5 h total per model.
#
# Pre-flight: aborts at the orchestrator if asg-ai/athena-cti-sft-
# foundation-8b-v21-taa is not yet on Hugging Face. Watch for the
# v21-TAA autotrain run on the SFT box to push first.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-foundation-8b-v21-taa-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
