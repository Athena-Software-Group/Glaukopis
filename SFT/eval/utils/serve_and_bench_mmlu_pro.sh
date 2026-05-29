#!/bin/bash

# Generic MMLU-Pro wrapper: serve any -vllm-suffixed alias on the local
# 2xH100 box and run only the MMLU-Pro suite (TIGER-Lab MMLU-Pro 12K,
# reasoning benchmark) against it under one warm vLLM session. Tears the
# server down on exit.
#
# Wraps run_foundation_8b_baselines.sh with the standard 2xH100 defaults
# (--tp 2 --max-len 32768, matching the Qwen2.5 native context cap so
# the same wrapper works for Qwen2.5-14B/32B and any other -vllm alias
# whose native ctx is >= 32768). All other CTI suites are skipped; only
# --include-mmlu-pro is passed through.
#
# Unlike the per-model wrappers (serve_and_bench_v21_recalibrate.sh
# etc.), the model alias is required as the first positional argument
# so MMLU-Pro can be run against the best candidate of the day without
# adding a new file per checkpoint. Use the per-model wrappers for the
# CTI suite sweep; use this one for the MMLU-Pro spot-check.
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_mmlu_pro.sh \
#       <model-alias-vllm> [extra-flags]
#
# Extra flags forwarded verbatim to run_foundation_8b_baselines.sh:
#   --tp N                                 (default 2; override for 1x or 4x GPU hosts)
#   --max-len N                            (default 32768; raise for non-Qwen2.5 bases
#                                           with longer native ctx, e.g. Llama-3.1 131K,
#                                           though MMLU-Pro prompts are <2K so the floor
#                                           dominates either way)
#   --rows N                               (optional; smoke-test against first N rows
#                                           before the full 12K sweep)
#   --mode resume|overwrite|retry-errors   (default: resume)
#   --dry-run                              (print the bench command and exit)
#
# Examples:
#   # Full MMLU-Pro sweep against v21-recalibrate (Qwen2.5-14B):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_mmlu_pro.sh \
#       athena-cti-sft-qwen25-14b-v21-recalibrate-vllm
#
#   # Smoke-probe (2 rows) before committing to the full run:
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_mmlu_pro.sh \
#       athena-cti-sft-qwen25-14b-v21-recalibrate-vllm --rows 2
#
#   # Clean re-bench (delete prior response file):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_mmlu_pro.sh \
#       athena-cti-sft-qwen25-14b-v21-recalibrate-vllm --mode overwrite
#
#   # Single-GPU host (e.g. 1xH100 dev box):
#   BENCH_CONDA_ENV=ctibench bash serve_and_bench_mmlu_pro.sh \
#       athena-cti-sft-qwen25-14b-v21-recalibrate-vllm --tp 1
#
# Wallclock estimate (2xH100, Qwen2.5-14B, --tp 2 --max-len 32768):
#   vLLM cold start ~3-4 min (14B weights + cudagraph capture) +
#   MMLU-Pro 12K rows @ batch 64 ~5-10 min = ~10-15 min total.
#
# Pre-flight: the orchestrator aborts if the HF repo for the alias is
# not yet pushed. Confirm the autotrain push has completed before
# launching.

set -u

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '3,57p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

MODEL_ALIAS="$1"; shift

if [[ "${MODEL_ALIAS}" != *-vllm ]]; then
    echo "[FAIL] model alias must end with '-vllm' (got: ${MODEL_ALIAS})" >&2
    echo "       The -vllm suffix routes through VLLMModel; only -vllm aliases" >&2
    echo "       have an HF repo id resolvable by serve_vllm.sh." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this wrapper. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

exec bash "${SCRIPT_DIR}/run_foundation_8b_baselines.sh" \
    --model "${MODEL_ALIAS}" \
    --tp 2 --max-len 32768 \
    --skip-athena --skip-cybermetric --skip-cybersoceval \
    --include-mmlu-pro \
    "$@"
