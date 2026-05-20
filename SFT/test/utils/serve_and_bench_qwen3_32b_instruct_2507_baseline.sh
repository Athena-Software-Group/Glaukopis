#!/bin/bash

# Per-model wrapper: serve Qwen/Qwen3-32B-Instruct-2507 on 2xH100 and run
# the full baseline benchmark suite (AthenaBench + CyberMetric 2K +
# CyberMetric 10K + CyberSOCEval malware/TI) under one warm vLLM session.
# Tears the server down on exit.
#
# Purpose: establish pre-SFT scores for the Qwen3-32B Instruct-2507 base
# before committing to a full v21 chain port. Use these numbers to decide
# whether the v21 recipe is worth the 40-60h training cost at 32B.
#
# Why Qwen3-32B-Instruct-2507 (vs Qwen3-32B):
#   - Qwen3-32B (the April 2025 release) is the hybrid instruct+thinking
#     model and emits `<think>...</think>` traces unless the chat template
#     is told otherwise. That makes it a poor SFT base for v21 (whose
#     dataset envelopes contain no reasoning traces) and an awkward
#     baseline (scores depend on the trace-suppression flag).
#   - Qwen3-32B-Instruct-2507 (July 2025 split) has no thinking mode at
#     all -- pure chat instruct, same architecture, ~7 months newer
#     pre-train than Qwen2.5-32B. Cleanest 32B reference point for v21.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite orchestrator)
# with the Qwen3-32B-on-2xH100 defaults baked in. Any extra flags are
# forwarded to the orchestrator verbatim.
#
# Hardware notes (8xH100 80GB host):
#   --tp 2 is the right pick for 32B bf16 (~64 GB weights -> ~32 GB/rank,
#   leaving ~48 GB/rank for KV cache at --max-num-seqs 32). --tp 1 also
#   fits (16 GB headroom) but starves KV cache on the long-context
#   cybersoceval-TI rows. --tp 4 wastes half the host on a model that
#   doesn't need it.
#
# Context window:
#   Qwen3-32B-Instruct-2507's native ctx is 262144 (256K) with YaRN. The
#   orchestrator's auto-pick (49152 when cybersoceval is selected, 16384
#   otherwise) sits well inside that envelope, so no --max-len override
#   is needed. The cybersoceval-TI prompts top out near 32K tokens; the
#   49152 cap leaves ~17K of generation headroom on the worst row.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_32b_instruct_2507_baseline.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#   --max-len N                            (override the auto-pick)
#
# Examples:
#   # Full baseline sweep, resume mode (default):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_32b_instruct_2507_baseline.sh
#
#   # Clean re-bench (delete prior response files):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_32b_instruct_2507_baseline.sh --mode overwrite
#
#   # Smoke test against the first 8 rows of each suite:
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_32b_instruct_2507_baseline.sh --rows 8
#
#   # AthenaBench + CyberMetric only (skip the slow CyberSOCEval suite):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_32b_instruct_2507_baseline.sh --skip-cybersoceval
#
# Wallclock estimate (2xH100, Qwen3-32B-Instruct-2507):
#   Athena (~1.2 h) + CM-2K (~45 min) + CM-10K (~3.5 h) + CSE (~5.5 h)
#   ~ 11 h total. ~1.6x the Qwen2.5-14B wrapper's ~7h owing to the 2.3x
#   parameter count (token throughput scales sublinearly under tp=2).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model qwen3-32b-instruct-2507-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
