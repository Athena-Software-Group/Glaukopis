#!/bin/bash

# Per-model wrapper: serve asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b
# on 2xH100 and run the full baseline benchmark suite (AthenaBench +
# CyberMetric 2K + CyberMetric 10K + CyberSOCEval malware/TI) under one
# warm vLLM session. Tears the server down on exit.
#
# Model lineage:
#   Base : Qwen/Qwen3-30B-A3B-Thinking-2507 (MoE, 30.5B total / 3.3B
#          active per token, pure-thinking July 2025 split).
#   SFT  : v21 recal-32b recipe held byte-identical to the Qwen2.5-32B
#          headline (athena-cti-sft-qwen25-32b-v21-recal-32b: Total 66.3,
#          Weighted 65.3). 3-shard Phase-B-heavy interleave (0.15/0.60/
#          0.25), lr 3e-6, cutoff 16384, packing off, eff_bs 8, 1 epoch,
#          adamw_8bit, Liger. First Qwen3-family SFT in the codebase.
#          See SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_recal_32b.sh.
#
# Inference semantic ('-no-think' alias suffix):
#   The base model is pure-thinking (always emits <think>...</think>),
#   but the SFT trained it with --enable_thinking False so the qwen3
#   chat template skipped the reasoning preamble on every CTI prompt.
#   The SFT'd checkpoint should emit direct answers on the CTI surface.
#   The '-no-think' suffix in the alias triggers VLLMModel to forward
#   chat_template_kwargs.enable_thinking=False per request -- belt-and-
#   suspenders: even if the saved chat_template defaults to thinking-on,
#   the override forces the no-think branch. Critically, '-no-think'
#   ALSO suppresses VLLMModel's '-thinking' 8192-token floor, so the
#   per-task caps in TASK_MAX_NEW_TOKENS (MCQ=128, RCM/RMS/TAA=256)
#   apply as designed; expected throughput matches non-thinking 30B-MoE
#   (~25x faster per-row wallclock than the Thinking-2507 baseline).
#
# Reasoning parser:
#   Defaults to no --reasoning-parser. If the SFT'd model occasionally
#   leaks a stray <think></think> wrapper on out-of-distribution
#   prompts, the parser can be re-enabled at launch:
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
# Wallclock estimate (2xH100, MoE 3.3B active, no thinking trace):
#   Athena (~25 min) + CM-2K (~10 min) + CM-10K (~45 min) + CSE (~50 min)
#   ~ 2 - 2.5 h total. ~3.5x faster than the Thinking-2507 baseline since
#   the SFT'd model emits 1-3 answer tokens per MCQ row instead of a
#   2-5k token <think> trace + answer.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_recal_32b.sh [extra-flags]
#
# Common extra flags (see run_foundation_8b_baselines.sh --help):
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --skip-athena | --skip-cybermetric | --skip-cybersoceval
#   --rows N                               (smoke-test against first N rows)
#   --cybermetric-size 2000,10000          (default; pass 2000 alone for short)
#
# Smoke test recommended:
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_recal_32b.sh --rows 8
#   Confirms the SFT'd model emits direct answers (bench `content` field
#   should be terse, NOT a <think>...</think> blob). If a trace appears,
#   set EXTRA_SERVE_FLAGS="--reasoning-parser qwen3" as the override above.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper.
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

# No --reasoning-parser by default: the SFT was run with --enable_thinking
# False so the trained model should emit direct answers and the parser
# would be a no-op. Override at launch if the smoke test shows the trace
# leaking into the bench-visible `content` field (e.g. on OOD prompts).
export EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:-}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b-no-think-vllm \
    --tp 2 \
    --cybermetric-size 2000,10000 \
    "$@"
