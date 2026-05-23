#!/bin/bash

# Per-model wrapper: serve asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
# on 2xH100 and run the full baseline benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session, with the Qwen3 thinking template ENABLED at request
# time. Sibling of serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_cse.sh
# (which serves the same checkpoint with enable_thinking=False); the two
# wrappers together form the matched-conditions A/B on the v21-cse SFT
# checkpoint under thinking-on vs no-think serving.
#
# Cache isolation: the two wrappers use distinct aliases that resolve to
# the same HF repo. Caches key off the alias (see
# pipelines/models.alias_to_safe_name) so the two runs land in separate
# responses/<alias-sanitized>/ slots by construction -- no manual
# isolation needed before launching.
#
# Inference semantic (no '-no-think' alias suffix):
#   The athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm alias
#   omits the '-no-think' substring so VLLMModel:
#     1. does NOT forward chat_template_kwargs.enable_thinking=False, so
#        the chat template is free to inject its <think> preamble.
#     2. detects 'thinking' in the alias and raises max_new_tokens to the
#        8192 floor so any emitted trace doesn't truncate at the per-task
#        cap (128 for MCQ, 1024 for MMLU-Pro).
#   The v21 SFT trained this checkpoint with the empty-thought pattern
#   (template injects <think>\n\n</think> on every CTI row), so the model
#   should keep emitting an empty trace under thinking-on serving and the
#   CTI numbers should land within noise of the -no-think sibling. The
#   point of running this is to verify that empirically and to give the
#   matched-conditions comparison against the qwen3-30b-a3b-thinking-2507-
#   vllm base under the same inference path -- isolating the SFT's
#   contribution under identical serving semantics.
#
# Reasoning parser:
#   Defaults to --reasoning-parser qwen3 so if the model DOES emit a
#   non-empty trace it lands in reasoning_content and the bench-visible
#   content field stays clean. Override by passing EXTRA_SERVE_FLAGS=""
#   (or any other value) before the script.
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2. Weights ~60 GB bf16 resident.
#
# Wallclock estimate (2xH100, MoE 3.3B active):
#   If the model honors the empty-thought pattern under thinking-on
#   serving (expected) wallclock should match the -no-think sibling
#   (~2 - 2.5 h total: Athena ~25 min + CM-2K ~10 min + CM-10K ~45 min +
#   CSE ~50 min). If the model emits substantive traces wallclock will
#   blow up proportionally to the trace length (could 3-5x).
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_cse_thinking_on.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"
export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:---reasoning-parser qwen3}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
