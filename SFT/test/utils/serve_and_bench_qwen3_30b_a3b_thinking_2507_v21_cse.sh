#!/bin/bash

# Per-model wrapper: serve asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
# on 2xH100 and run the full baseline benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session. Tears the server down on exit.
#
# Model lineage:
#   Base : asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa
#          (v21 chain Stage 2 output on the 30.5B/3.3B-active MoE).
#   SFT  : v21 CSE letter-set drill held byte-identical to the
#          Qwen2.5-32B v21-cse recipe. See SFT/autotrain/
#          run_sft_qwen3_30b_a3b_thinking_v21_final.sh.
#
# This is the v18.1 ship-equivalent checkpoint on the MoE arch -- the
# headline candidate for the Qwen3 chain port. Recalibrate (Stage 4) is
# an off-plan touch-up that may or may not improve on this checkpoint.
#
# Inference semantic ('-no-think' alias suffix):
#   See serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_core.sh for the
#   full enable_thinking=True training / -no-think serving rationale.
#
# Reasoning parser:
#   Defaults to no --reasoning-parser. Override:
#     EXTRA_SERVE_FLAGS="--reasoning-parser qwen3" bash <this-script>
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2. Weights ~60 GB bf16 resident.
#
# Wallclock estimate (2xH100, MoE 3.3B active, empty 6-token trace):
#   Athena (~25 min) + CM-2K (~10 min) + CM-10K (~45 min) + CSE (~50 min)
#   ~ 2 - 2.5 h total.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_cse.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"
export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:-}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-no-think-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
