#!/bin/bash

# Per-model wrapper: serve fdtn-ai/Foundation-Sec-8B-Instruct (Cisco
# SFT+RLHF cybersecurity model on the Llama-3.1-8B architecture, Aug
# 2025) on 2xH100 and run the full baseline benchmark suite (AthenaBench
# + CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under
# one warm vLLM session. Tears the server down on exit.
#
# Purpose: re-confirm the pre-SFT baseline for the Foundation-Sec-8B
# port of the v21 chain (recipe lift from Qwen2.5-14B). Prior bench had
# this at 57.7 avg (CKT 77.8, ATE 36.4, RCM 65.3, RMS combined 28.3,
# VSP 65.1, TAA Classic 57.5, CSE Malware 49.7 / TI 55.1, CM 83.7). Re-run
# after the recent bench-client fixes (vLLM ctx-overflow regex now handles
# both old "at least P" and new "exact P" 400 messages; no more silent
# truncation of CSE-TI rows that exceed served ctx).
#
# Chat template:
#   Foundation-Sec-8B-Instruct ships its own jinja template with custom
#   '<|system|>' / '<|user|>' / '<|assistant|>' markers and a baked-in
#   Cisco system prompt. vLLM picks it up from tokenizer_config.json --
#   no `--chat-template` override needed at serve time. This is the
#   SAME starting checkpoint used by the v21 chain on the SFT box;
#   post-SFT the chain rewrites the saved template to llama3 (LF's
#   default), but the pre-SFT base still uses Cisco's custom one.
#
# Hardware notes (2xH100 80GB test server):
#   --tp 2. 8B bf16 ~= 16 GB weights -> 8 GB/rank, leaving ~72 GB/rank
#   for KV cache at --max-num-seqs 32. Trivially fits at any --max-len.
#
# Wraps run_foundation_8b_baselines.sh (the generic multi-suite
# orchestrator). Any extra flags are forwarded verbatim.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_foundation_8b_instruct_baseline.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Wallclock estimate (2xH100, 8B dense):
#   Athena (~30 min) + CM-2K (~20 min) + CM-10K (~1.5 h) + CSE (~2.5 h)
#   ~ 4.5-5 h total.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model foundation-8b-instruct-vllm \
    --tp 2 --max-len 32768 \
    --cybermetric-size 2000,10000 \
    "$@"
