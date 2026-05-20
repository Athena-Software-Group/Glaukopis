#!/bin/bash

# Per-model wrapper: serve mistralai/Mistral-Small-3.2-24B-Instruct-2506
# (24B dense, June 2025) on 2xH100 and run the full baseline benchmark
# suite (AthenaBench + CyberMetric 2K + CyberMetric 10K + CyberSOCEval
# malware/TI) under one warm vLLM session. Tears the server down on exit.
#
# Purpose: pre-SFT baseline for a 32B-class candidate for the v21 SFT
# recipe (Glaukopis). Sits between Qwen2.5-14B (14B v21-recalibrate at
# 62.3 avg) and Qwen3-32B (~32B baseline pending) on the param axis; if
# it benches close to or above the Qwen2.5-14B-v21-recalibrate post-SFT
# number, it has meaningful headroom for an SFT port.
#
# Architecture:
#   24B dense, multimodal (text + vision). Served text-only here -- the
#   bench harness only sends /v1/chat/completions with text, and the
#   serve cmd passes --limit-mm-per-prompt image=0 via EXTRA_SERVE_FLAGS
#   to skip vision-encoder KV pre-allocation (same pattern as the Gemma 4
#   31B-it baseline). Native context 128K. Mistral V11 tokenizer; HF
#   format works with vLLM's default tokenizer mode -- no
#   `--tokenizer-mode mistral` needed for plain chat completions.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator) with the Mistral-Small-3.2-on-2xH100 defaults baked in.
# Any extra flags are forwarded to the orchestrator verbatim.
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2. 24B bf16 ~= 48 GB weights -> ~24 GB/rank, leaving ~56 GB/rank
#   for KV cache at --max-num-seqs 32. Plenty of headroom.
#
# Context window:
#   Native ctx is 128K; we cap --max-len at 40960 for parity with the
#   other 24B-32B baselines so cybersoceval-TI rows (~32K-32.7K) fit
#   with 8K generation headroom and KV cache stays bounded. Longer rows
#   are caught by the client-side ctx-overflow path in
#   pipelines/models.py (one-shot drop to floor and retry).
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_mistral_small_3p2_24b_instruct_2506_baseline.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#   --max-len N                            (override the 40960 cap below)
#
# Wallclock estimate (2xH100, 24B dense):
#   Athena (~50 min) + CM-2K (~30 min) + CM-10K (~2.5 h) + CSE (~4 h)
#   ~ 7.5-8.5 h total.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

# Skip vision-encoder KV pre-allocation for the text-only bench. The
# Mistral-Small-3.2 vision tower would otherwise reserve multi-GB per
# rank for image features we never send.
#
# vLLM 0.7+ takes --limit-mm-per-prompt as a JSON object; the older
# key=value syntax errors with "Value image=0 cannot be converted to
# <function loads>". JSON form works on both old and new vLLM.
export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:---limit-mm-per-prompt '{\"image\":0}'}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model mistral-small-3.2-24b-instruct-2506-vllm \
    --tp 2 --max-len 40960 \
    --cybermetric-size 2000,10000 \
    "$@"
