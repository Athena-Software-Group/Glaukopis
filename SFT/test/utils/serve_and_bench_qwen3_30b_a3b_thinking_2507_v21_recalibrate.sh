#!/bin/bash

# Per-model wrapper: serve asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recalibrate
# on 2xH100 and run the full baseline benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session. Tears the server down on exit.
#
# Model lineage:
#   Upstream base : Qwen/Qwen3-30B-A3B-Thinking-2507 (MoE, 30.5B total /
#                   3.3B active per token, pure-thinking July 2025 split).
#   Stage-4 parent: asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
#                   (Qwen3-MoE v21 chain Core -> TAA -> CSE).
#   SFT           : v21 Recalibrate 14B-recipe 3-shard interleave touch-
#                   up (probs 0.25/0.40/0.35, lr 1e-6, max-samples 2400,
#                   eff_bs 8, 1 epoch). OFF-CHAIN on the Qwen3-MoE port
#                   (the default chain ships the 32B-tuned recal-32b
#                   recipe at Stage 4 -- see README-21.md §"Qwen3-30B-
#                   A3B-Thinking-2507 MoE port"). This variant exists
#                   for off-chain A/B against the on-chain recal-32b
#                   ship-candidate. Launcher: SFT/autotrain/
#                   run_sft_qwen3_30b_a3b_thinking_v21_recalibrate.sh.
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
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_recalibrate.sh [extra-flags]
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
    --model athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recalibrate-no-think-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
