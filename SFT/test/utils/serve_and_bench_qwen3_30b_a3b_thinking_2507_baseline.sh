#!/bin/bash

# Per-model wrapper: serve Qwen/Qwen3-30B-A3B-Thinking-2507 (MoE,
# pure-thinking July 2025 split) on 2xH100 and run the full baseline
# benchmark suite (AthenaBench + CyberMetric 2K + CyberMetric 10K +
# CyberSOCEval malware/TI) under one warm vLLM session. Tears the
# server down on exit.
#
# Purpose: establish a thinking-mode baseline for the ~30B-class Qwen3
# MoE family after the Instruct-2507 sibling (this wrapper's pair)
# scored 35.8 avg with collapsed AthenaBench performance (CKT 18.1,
# RMS 7.0, CyberMetric 20.0) -- below Qwen2.5-14B-Instruct. The
# hypothesis is that the AthenaBench MCQ/structured tasks are exactly
# the kind of multi-step reasoning where Qwen3's thinking variant was
# trained to dominate, and the Instruct-2507's non-thinking head was
# the wrong evaluation surface.
#
# Architecture:
#   - Same MoE base as Qwen3-30B-A3B-Instruct-2507: 30.5B total, 3.3B
#     active per token, 128 experts top-8 routing, 262K native ctx.
#   - Post-training: thinking-only. ALWAYS emits a <think>...</think>
#     trace before the final answer; no enable_thinking switch.
#
# Server-side reasoning parser:
#   The serve cmd passes `--reasoning-parser deepseek_r1` via EXTRA_SERVE_FLAGS
#   so vLLM splits the response: thinking trace -> reasoning_content,
#   final answer -> content. The bench client reads content only, so it
#   sees clean short-answer output (no <think> wrapping the regex match).
#   On vllm>=0.10 the dedicated `qwen3` parser also works and routes
#   identically -- swap if you see the trace leak into content on serve
#   logs (Qwen/Qwen3-30B-A3B-Thinking-2507 HF discussion #2 documents
#   the version-dependent behaviour).
#
# Client-side max_tokens floor:
#   pipelines/models.py VLLMModel detects the 'thinking' substring in
#   the alias (qwen3-30b-a3b-thinking-2507-vllm matches) and bumps the
#   per-request max_tokens to 8192 minimum. Without this the per-task
#   caps in TASK_MAX_NEW_TOKENS (MCQ=128, RCM/RMS/TAA=256) would
#   truncate every row mid-trace and collapse accuracy below random.
#   8192 leaves comfortable headroom for both the trace (~2-5k tokens
#   typical) and the answer; the floor is only applied per request so
#   prompts already requesting >8192 are unaffected.
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2. Same envelope as the Instruct-2507 sibling -- weights are
#   identical, ~60 GB bf16 resident, leaves ~50 GB/rank for KV cache.
#
# vLLM version requirement:
#   vllm>=0.8.5 for Qwen3-MoE support and the deepseek_r1 reasoning
#   parser. vllm>=0.10 if switching to --reasoning-parser qwen3.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_baseline.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Wallclock estimate (2xH100, Qwen3-30B-A3B-Thinking MoE):
#   ~2.5-3.5x the Instruct-2507 sibling owing to the per-row thinking
#   trace. Each row generates ~2-5k extra tokens for the trace before
#   the answer, even though only the answer is bench-visible.
#   Athena (~1.5 h) + CM-2K (~40 min) + CM-10K (~2.5 h) + CSE (~4 h)
#   ~ 8.5-9 h total.
#
# Smoke test recommended:
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_baseline.sh --rows 8
#   Confirms the reasoning parser is wired correctly (bench content
#   field should be a terse answer, not a <think>...</think> blob) and
#   the max_tokens floor lets the trace fit before the answer.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

# Enable the deepseek_r1 reasoning parser so the <think> trace lands in
# reasoning_content and the bench-visible `content` field is just the
# final answer. --enable-reasoning is required on vllm<0.10; on >=0.10
# the parser flag alone is enough but the legacy flag is silently
# accepted, so we always pass both for portability.
export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:---enable-reasoning --reasoning-parser deepseek_r1}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model qwen3-30b-a3b-thinking-2507-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
