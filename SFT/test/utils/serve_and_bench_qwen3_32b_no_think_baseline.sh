#!/bin/bash

# Per-model wrapper: serve Qwen/Qwen3-32B (hybrid, served with `<think>`
# trace disabled) on 2xH100 and run the full baseline benchmark suite
# (AthenaBench + CyberMetric 2K + CyberMetric 10K + CyberSOCEval
# malware/TI) under one warm vLLM session. Tears the server down on exit.
#
# Purpose: establish pre-SFT scores for the Qwen3-32B base before
# committing to a v21 chain port. The '-no-think' alias forwards
# `chat_template_kwargs.enable_thinking=False` on every request so the
# chat template skips the reasoning preamble -- the behavioural
# equivalent of a pure-instruct 32B for short-answer / multi-turn-MCQ
# tasks, without the trace eating the generation budget.
#
# Why not Qwen3-32B-Instruct-2507:
#   That repo does not exist. Qwen's July 2025 pure-instruct '-2507'
#   splits were only published for 4B dense, 30B-A3B MoE, and 235B-A22B
#   MoE. There is no 32B dense Instruct-2507. The closest behavioural
#   substitute at the dense 32B size is Qwen3-32B with thinking disabled
#   -- same weights as a hypothetical 32B-Instruct-2507 would have been
#   pre-2507 post-train, just with the dual-mode chat template instead
#   of a single-mode one. For a future v21 SFT port to Qwen3-32B, the
#   SFT recipe would train on this same base and the data envelope
#   would naturally lack `<think>` traces, so the SFT'd model converges
#   on single-mode behaviour anyway.
#
# Companion baseline: serve_and_bench_qwen3_30b_a3b_instruct_2507_baseline.sh
# benches the actual 2507 MoE variant (different architecture shape but
# matches the 'pure instruct, newer pre-train' pitch).
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen3-32B-on-2xH100 defaults baked in. Any
# extra flags are forwarded to the orchestrator verbatim.
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2 is the right pick for 32B bf16 (~64 GB weights -> ~32 GB/rank,
#   leaving ~48 GB/rank for KV cache at --max-num-seqs 32).
#
# Context window:
#   Qwen3-32B's native ctx is 41K (262K with YaRN). The orchestrator's
#   auto-pick for sweeps that include cybersoceval is 49152, above
#   the 40960 native ceiling, so we explicitly cap --max-len at 40960
#   to avoid vLLM rejecting the serve. The 32K-32.7K cybersoceval-TI
#   prompts still fit with ~8K generation headroom; longer rows are
#   caught by the client-side ctx-overflow path in pipelines/models.py.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_32b_no_think_baseline.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#   --max-len N                            (override the 40960 cap below)
#
# Wallclock estimate (2xH100, Qwen3-32B dense):
#   Athena (~1.2 h) + CM-2K (~45 min) + CM-10K (~3.5 h) + CSE (~5.5 h)
#   ~ 11 h total.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model qwen3-32b-no-think-vllm \
    --tp 2 --max-len 40960 \
    --cybermetric-size 2000,10000 \
    "$@"
