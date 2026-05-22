#!/bin/bash

# Per-model wrapper: serve asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core
# on 2xH100 and run the full baseline benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session. Tears the server down on exit.
#
# Model lineage:
#   Base : Qwen/Qwen3-30B-A3B-Thinking-2507 (MoE, 30.5B total / 3.3B
#          active per token, pure-thinking July 2025 split).
#   SFT  : v21 Core (Phase A + Phase B) held byte-identical to the
#          Qwen2.5-32B v21-core recipe (cutoff 8192 packing on -> 16384
#          packing off, eff_bs 16/8, adamw_8bit, Liger). Sparse-arch port
#          of the v21 chain Stage 1. See SFT/autotrain/
#          run_sft_qwen3_30b_a3b_thinking_v21_core.sh.
#
# Inference semantic ('-no-think' alias suffix):
#   The base model is pure-thinking (always emits <think>...</think>).
#   The SFT was run with --enable_thinking True (the new run_train.sh
#   default), so the reasoning template injected empty <think>\n\n</think>
#   into the loss/response_ids for every CTI row. The trained model
#   should autonomously emit a ~6-token empty thought then the answer.
#   The '-no-think' suffix in the alias triggers VLLMModel to forward
#   chat_template_kwargs.enable_thinking=False per request -- belt-and-
#   suspenders against template drift, and -- more importantly --
#   suppresses VLLMModel's '-thinking' 8192-token floor so the per-task
#   caps in TASK_MAX_NEW_TOKENS (MCQ=128, RCM/RMS/TAA=256) apply.
#
# Reasoning parser:
#   Defaults to no --reasoning-parser. If the SFT'd model occasionally
#   leaks a real (non-empty) <think></think> trace on OOD prompts, the
#   parser can be re-enabled at launch:
#     EXTRA_SERVE_FLAGS="--reasoning-parser qwen3" bash <this-script>
#   The parser is harmless on output without <think> tags (the trace
#   path is just never taken), so this is a safe override.
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2. Weights ~60 GB bf16 resident, ~50 GB/rank for KV cache.
#   On a single B300 (288 GB) tp=1 would also work and saves one
#   all-reduce per layer; set --tp 1 at launch if benching on the
#   training host between runs.
#
# Wallclock estimate (2xH100, MoE 3.3B active, ~6-token empty trace):
#   Athena (~25 min) + CM-2K (~10 min) + CM-10K (~45 min) + CSE (~50 min)
#   ~ 2 - 2.5 h total. ~3.5x faster than the Thinking-2507 baseline since
#   the SFT'd model emits a ~6-token empty <think> + answer instead of a
#   2-5k token real <think> trace + answer.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_core.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Smoke test recommended:
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_core.sh --rows 8
#   Confirms the SFT'd model emits direct answers (bench `content` field
#   should be terse with at most an empty 6-token <think></think>, NOT a
#   multi-kB trace). If a real trace leaks, set
#   EXTRA_SERVE_FLAGS="--reasoning-parser qwen3" as the override above.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

# No --reasoning-parser by default: the SFT trained the model to emit an
# empty 6-token <think></think> + answer. Override at launch if the
# smoke test shows a real (non-empty) trace leaking into the bench-
# visible `content` field on OOD prompts.
export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:-}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core-no-think-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
