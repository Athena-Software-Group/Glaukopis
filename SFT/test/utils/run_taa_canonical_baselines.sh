#!/bin/bash

# Backfill TAA Canonical (athena-taa-canonical) scores for the six reference
# models cited in the v21 ship-comparison sheet but not yet graded on the
# canonical-attribution axis. Thin chain over the per-model scripts so each
# model can be re-run in isolation when needed. Total wallclock ~15-20 min.
#
# Per-model scripts (chained in this order):
#   1. run_taa_canonical_gpt5_2_high.sh            (OpenAI gpt-5.2 effort=high)
#   2. run_taa_canonical_gemini_3_flash.sh         (gemini-3-flash-preview)
#   3. run_taa_canonical_gemini_2_5_flash.sh       (gemini-2.5-flash)
#   4. run_taa_canonical_deepseek_v4_pro.sh        (DeepSeek-V4-Pro via HF Router)
#   5. run_taa_canonical_deepseek_v3_2_exp.sh      (DeepSeek-V3.2-Exp via HF Router)
#   6. serve_and_bench_qwen25_14b_taa_canonical.sh (Qwen2.5-14B local vLLM)
#
# Phases:
#   - hosted (1-5): sequential HTTP-only calls; no GPU.
#   - local  (6) : warms a single vLLM session (--tp 2 on 2xH100), benches,
#                  tears down. Skip with --skip-vllm to run hosted-only.
#
# Each per-model script forwards --tasks "athena-taa-canonical" --version 1
# --overwrite --yes (unless --no-overwrite) so pre-existing canonical
# responses are replaced with a clean baseline.
#
# Usage:
#   conda activate ctibench
#   bash run_taa_canonical_baselines.sh [--rows N] [--batch N] [--no-overwrite]
#                                       [--skip-openai|--skip-gemini|--skip-hf|--skip-vllm]
#                                       [--reasoning-effort low|medium|high|xhigh]
#                                       [--dry-run]
#
# Environment (only the families you actually run need their key set):
#   OPENAI_API_KEY     gpt5.2
#   GEMINI_API_KEY     gemini-2.5-flash, gemini-3-flash
#   HF_TOKEN           deepseek-v4-pro-hf, deepseek-v3.2-exp-hf
#                      (HUGGINGFACE_TOKEN also accepted)
#   BENCH_CONDA_ENV    set to 'ctibench' if launching from the vllm env
#
# Logs / artifacts:
#   SFT/test/utils/taa_canonical_baselines_<UTC>.log   combined sweep log
#   SFT/test/responses/<display>/athena-taa-canonical/ per-row responses
#   SFT/test/responses/<display>/summary_athena_*.json per-model summary

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=_load_dotenv.sh
source "${SCRIPT_DIR}/_load_dotenv.sh"

ROWS=""
BATCH_HOSTED="16"
BATCH_VLLM="64"
OVERWRITE=1
REASONING_EFFORT="high"
DRY_RUN=0
SKIP_OPENAI=0
SKIP_GEMINI=0
SKIP_HF=0
SKIP_VLLM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows)             ROWS="$2"; shift 2 ;;
        --batch)            BATCH_HOSTED="$2"; BATCH_VLLM="$2"; shift 2 ;;
        --no-overwrite)     OVERWRITE=0; shift ;;
        --skip-openai)      SKIP_OPENAI=1; shift ;;
        --skip-gemini)      SKIP_GEMINI=1; shift ;;
        --skip-hf)          SKIP_HF=1; shift ;;
        --skip-vllm)        SKIP_VLLM=1; shift ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

UTC="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG="${SCRIPT_DIR}/taa_canonical_baselines_${UTC}.log"
echo "[info] log              : ${LOG}"
echo "[info] task             : athena-taa-canonical"
echo "[info] rows             : ${ROWS:-all}"
echo "[info] batch (hosted)   : ${BATCH_HOSTED}"
echo "[info] batch (vllm)     : ${BATCH_VLLM}"
echo "[info] overwrite        : $([[ ${OVERWRITE} -eq 1 ]] && echo yes || echo no)"
echo "[info] reasoning-effort : ${REASONING_EFFORT} (gpt5.2 only)"

# Pre-flight: required API keys per enabled family. The per-model scripts
# also pre-flight their own key, but surfacing missing keys here means we
# fail fast before any sub-script writes a partial log.
[[ ${SKIP_OPENAI} -eq 0 && -z "${OPENAI_API_KEY:-}" ]] && \
    { echo "[FAIL] OPENAI_API_KEY required for gpt5.2 (or pass --skip-openai)" >&2; exit 2; }
[[ ${SKIP_GEMINI} -eq 0 && -z "${GEMINI_API_KEY:-}" ]] && \
    { echo "[FAIL] GEMINI_API_KEY required for gemini-* (or pass --skip-gemini)" >&2; exit 2; }
[[ ${SKIP_HF} -eq 0 && -z "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ]] && \
    { echo "[FAIL] HF_TOKEN/HUGGINGFACE_TOKEN required for deepseek-*-hf (or pass --skip-hf)" >&2; exit 2; }

COMMON_FLAGS=()
[[ ${OVERWRITE} -eq 0 ]] && COMMON_FLAGS+=(--no-overwrite)
[[ ${DRY_RUN}   -eq 1 ]] && COMMON_FLAGS+=(--dry-run)
[[ -n "${ROWS}" ]]      && COMMON_FLAGS+=(--rows "${ROWS}")

run_step() {
    # Chain a per-model script; tee its output to the sweep log. Non-zero
    # exits propagate as a warning rather than aborting the chain so a
    # provider hiccup on one model doesn't block the rest of the sweep.
    local script="$1"; shift
    echo
    bash "${SCRIPT_DIR}/${script}" "$@" 2>&1 | tee -a "${LOG}" \
        || echo "[WARN] ${script} exited non-zero; continuing chain" | tee -a "${LOG}"
}

# ------------------------------------------------------------------
# Phase 1: hosted models (HTTP-only; no GPU)
# ------------------------------------------------------------------
[[ ${SKIP_OPENAI} -eq 0 ]] && run_step run_taa_canonical_gpt5_2_high.sh \
    --batch "${BATCH_HOSTED}" --reasoning-effort "${REASONING_EFFORT}" "${COMMON_FLAGS[@]}"
[[ ${SKIP_GEMINI} -eq 0 ]] && run_step run_taa_canonical_gemini_3_flash.sh \
    --batch "${BATCH_HOSTED}" "${COMMON_FLAGS[@]}"
[[ ${SKIP_GEMINI} -eq 0 ]] && run_step run_taa_canonical_gemini_2_5_flash.sh \
    --batch "${BATCH_HOSTED}" "${COMMON_FLAGS[@]}"
[[ ${SKIP_HF}     -eq 0 ]] && run_step run_taa_canonical_deepseek_v4_pro.sh \
    --batch "${BATCH_HOSTED}" "${COMMON_FLAGS[@]}"
[[ ${SKIP_HF}     -eq 0 ]] && run_step run_taa_canonical_deepseek_v3_2_exp.sh \
    --batch "${BATCH_HOSTED}" "${COMMON_FLAGS[@]}"

# ------------------------------------------------------------------
# Phase 2: local vLLM (Qwen2.5-14B-Instruct on 2xH100)
# ------------------------------------------------------------------
[[ ${SKIP_VLLM} -eq 0 ]] && run_step serve_and_bench_qwen25_14b_taa_canonical.sh \
    --batch "${BATCH_VLLM}" "${COMMON_FLAGS[@]}"

echo
echo "[done] TAA Canonical baselines complete; log=${LOG}"
echo
echo "=================================================================="
echo "  Headline summary: TAA Canonical (acc / plaus / combined) per model"
echo "=================================================================="
( cd "${BENCH_DIR}" && python "${SCRIPT_DIR}/_print_taa_canonical_summary.py" \
    gpt-5.2-high \
    gemini-3-flash-preview \
    gemini-2.5-flash \
    deepseek-ai_DeepSeek-V4-Pro \
    deepseek-ai_DeepSeek-V3.2-Exp \
    Qwen_Qwen2.5-14B-Instruct \
    2>&1 | tee -a "${LOG}" ) \
    || echo "[WARN] summary printer failed (non-fatal); per-model summary_athena_*.json still on disk" | tee -a "${LOG}"
