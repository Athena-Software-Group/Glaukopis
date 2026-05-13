#!/bin/bash

# Per-model wrapper: serve v18.1-CSE (Qwen2.5-14B, v18.1-taa + CyberSOCEval
# letter-set narrow drill -- the headline v18.1 model) on 2xH100 and run the
# full benchmark suite (AthenaBench + CyberMetric 2K + CyberMetric 10K +
# CyberSOCEval malware/TI) under one warm vLLM session. Tears the server
# down on exit.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite orchestrator)
# with the Qwen2.5-14B-on-2xH100 defaults baked in. Any extra flags are
# forwarded to the orchestrator verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p1_cse.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Examples:
#   # Full sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p1_cse.sh
#
#   # Clean re-bench (deletes prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p1_cse.sh --mode overwrite
#
#   # Athena + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_v18p1_cse.sh --skip-cybersoceval
#
# Wallclock estimate (2xH100, Qwen2.5-14B):
#   Athena (~50 min) + CM-2K (~30 min) + CM-10K (~2.2 h) + CSE (~3.5 h)
#   ~ 7 h total per model.
#
# Pre-flight: this script aborts at the orchestrator if asg-ai/athena-cti-sft-
# qwen25-14b-v18-1-cse is not yet on Hugging Face. Watch for the v18.1-CSE
# autotrain run on the SFT box to push first (~T+15h after v18.1-core lands).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen25-14b-v18-1-cse-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
