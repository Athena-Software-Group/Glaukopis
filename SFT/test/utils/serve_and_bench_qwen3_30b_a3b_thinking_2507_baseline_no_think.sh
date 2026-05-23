#!/bin/bash

# Per-model wrapper: serve Qwen/Qwen3-30B-A3B-Thinking-2507 (MoE,
# pure-thinking July 2025 split) on 2xH100 under the matched-conditions
# no-think inference path and run the full baseline benchmark suite
# (AthenaBench + CyberMetric 2K + CyberMetric 10K + CyberSOCEval
# malware/TI) under one warm vLLM session. Tears the server down on exit.
#
# Purpose: matched-conditions A/B baseline against the v21 Qwen3-MoE SFT
# chain. The sibling serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_{
# core,taa,cse,recal_32b,recalibrate}.sh wrappers all serve the SFT'd
# checkpoints under the '-no-think-vllm' alias scheme (thinking disabled
# at request time, per-task TASK_MAX_NEW_TOKENS caps applied without the
# thinking-mode 8192-token floor). The original
# serve_and_bench_qwen3_30b_a3b_thinking_2507_baseline.sh wrapper served
# the base on its fair footing -- thinking-on with the 8192 floor and
# the qwen3 reasoning parser -- which is a useful capability ceiling
# but is NOT directly comparable to the v21 SFT numbers. This wrapper
# fills that gap: same base HF repo, same '-no-think' inference path
# the v21 SFT bench wrappers use, so the deltas in §"Qwen3-30B-A3B-
# Thinking-2507 MoE port" of README-21.md become apples-to-apples.
#
# Expected outcome: the base under matched no-think conditions should
# collapse relative to its thinking-on baseline because Thinking-2507
# was NOT trained on the empty-thought pattern that the v21 SFT
# instills. It will emit a substantive trace despite
# enable_thinking=False being forwarded, the trace will be truncated
# at the per-task cap (MCQ=128, RCM/RMS/TAA=256, MMLU-Pro=1024 via
# the standalone MMLU-Pro wrapper), and accuracy on most short-answer
# axes should fall well below random. That collapse vs the SFT'd v21
# stages IS the signal -- it isolates the SFT's contribution to
# functioning under a no-trace inference budget.
#
# Inference semantic ('-no-think' alias suffix):
#   The alias 'qwen3-30b-a3b-thinking-2507-no-think-vllm' (registered
#   in SFT/test/pipelines/models.py) points at the same Qwen/Qwen3-30B-
#   A3B-Thinking-2507 HF repo as the thinking-on baseline alias above,
#   but the '-no-think' substring triggers two behaviours in VLLMModel:
#     1. Forwards `chat_template_kwargs.enable_thinking=False` per
#        request so the chat template skips the reasoning preamble
#        (belt-and-suspenders -- the base may still emit a trace
#        because it was not trained on the empty-thought pattern).
#     2. Opts the alias OUT of the '-thinking' 8192-token floor so
#        the per-task TASK_MAX_NEW_TOKENS caps apply unmodified.
#
# Reasoning parser:
#   Defaults to NO --reasoning-parser, mirroring the v21-cse / v21-recal-
#   32b / v21-recalibrate sibling wrappers exactly for matched-conditions
#   parity. Adding the parser would route any leaked <think> trace to
#   reasoning_content and clean the bench-visible content field --
#   "rescuing" exactly the failure mode the v21 SFT is designed to
#   solve, which would undermine the A/B. Override at launch only if
#   debugging:
#     EXTRA_SERVE_FLAGS="--reasoning-parser qwen3" bash <this-script>
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2. Weights ~60 GB bf16 resident, ~30 GB/rank, leaves ~50 GB/
#   rank for KV cache. Same envelope as the v21-cse wrapper.
#
# Wallclock estimate (2xH100, MoE 3.3B active, truncated trace per row):
#   Comparable to v21-cse wrapper since per-task caps bound the per-row
#   token budget the same way. Expect ~2.5-3 h end-to-end vs the
#   thinking-on baseline's ~8.5-9 h (which let traces consume up to
#   8192 tokens per row).
#   Athena (~25-30 min) + CM-2K (~10-12 min) + CM-10K (~50 min) +
#   CSE (~55 min) ~ 2.5-3 h total.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_baseline_no_think.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# MMLU-Pro is intentionally NOT included in this wrapper. Run it via
# the standalone generic wrapper to keep suite scope decoupled:
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_mmlu_pro.sh \
#       qwen3-30b-a3b-thinking-2507-no-think-vllm
#
# Smoke test recommended:
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_baseline_no_think.sh --rows 4
#   Confirms the alias is honoured (vLLM-ready log line should show
#   "thinking=disabled" and NOT "thinking=floor8192"), and that the
#   bench-visible content field shows truncated <think> tokens or junk
#   answers -- the expected collapse signature.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

# No reasoning parser by default -- matches the v21-cse / v21-recal-32b
# sibling wrappers exactly for matched-conditions A/B parity. See header
# for the rationale.
export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:-}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model qwen3-30b-a3b-thinking-2507-no-think-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
