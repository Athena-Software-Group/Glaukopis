#!/bin/bash

# Per-model wrapper: serve Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE,
# pure-instruct July 2025 split) on 2xH100 and run the full baseline
# benchmark suite (AthenaBench + CyberMetric 2K + CyberMetric 10K +
# CyberSOCEval malware/TI) under one warm vLLM session. Tears the
# server down on exit.
#
# Purpose: establish a pure-instruct ~30B-class baseline that is the
# actual realisation of the 'Instruct-2507' line (no hybrid mode, no
# `<think>` traces possible by construction). Pairs with the
# serve_and_bench_qwen3_32b_no_think_baseline.sh wrapper, which benches
# the dense 32B hybrid base with thinking disabled at request time.
# Together the two wrappers bracket the ~30B-class pre-SFT envelope:
#   - dense 32B (this wrapper's sibling): same arch shape as our v21
#     SFT lineage, behavioural-equivalent to pure instruct via the
#     -no-think alias trick.
#   - MoE 30B-A3B (this wrapper): cheaper to serve, newer pre-train,
#     pure instruct by design, but architecturally different (sparse
#     routing, 128 experts/8 routed) so the SFT story would diverge.
#
# Architecture:
#   - 30.5B total params, 3.3B active per token. 128 experts, top-8
#     routing. ~60 GB bf16 weights resident; serve-side memory profile
#     is similar to dense 32B because all expert weights must reside in
#     VRAM. Decode throughput is ~5-7x faster than dense 32B owing to
#     the 3.3B active path.
#   - 262K native context (262144), so the orchestrator's 49152 auto-pick
#     fits cleanly. No --max-len override needed.
#   - Pure non-thinking model: the Instruct-2507 chat template has no
#     enable_thinking knob -- it simply never emits `<think>` blocks.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Qwen3-30B-A3B-MoE-on-2xH100 defaults baked
# in. Any extra flags are forwarded to the orchestrator verbatim.
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2 is the cleanest pick for MoE serving on this hardware. MoE
#   sharding under vLLM splits experts across ranks; tp=1 leaves only
#   ~20 GB for KV cache (60 GB weights vs 80 GB VRAM), which throttles
#   batch concurrency on the cybersoceval-TI rows. tp=2 gives ~50 GB
#   KV-cache headroom per rank.
#
# vLLM version requirement:
#   Qwen3-MoE serving needs vllm>=0.8.5 (per the model card). If the
#   serve fails with a model_type 'qwen3_moe' not recognised error,
#   upgrade vllm in the local conda env.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_instruct_2507_baseline.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Wallclock estimate (2xH100, Qwen3-30B-A3B MoE):
#   Athena (~25-35 min) + CM-2K (~10-15 min) + CM-10K (~50-70 min) +
#   CSE (~1.5-2 h)
#   ~ 3.5-4.5 h total. Significantly faster than dense 32B because the
#   active path is 3.3B (vs 32B), and decode is bandwidth-bound on the
#   active params, not the resident params.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model qwen3-30b-a3b-instruct-2507-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
