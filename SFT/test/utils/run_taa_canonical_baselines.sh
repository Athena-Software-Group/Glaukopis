#!/bin/bash

# Backfill TAA Canonical (athena-taa-canonical) scores for the six reference
# models cited in the v21 ship-comparison sheet but not yet graded on the
# canonical-attribution axis. Single-task sweep; total wallclock ~15-20 min.
#
# Models exercised (in order):
#   1. gpt5.2 --reasoning-effort high          -> display gpt-5.2-high      (OpenAI)
#   2. gemini-3-flash                          -> display gemini-3-flash-preview
#   3. gemini-2.5-flash                        -> display gemini-2.5-flash
#   4. deepseek-v4-pro-hf                      -> display deepseek-ai_DeepSeek-V4-Pro
#   5. deepseek-v3.2-exp-hf                    -> display deepseek-ai_DeepSeek-V3.2-Exp
#   6. qwen2.5-14b-vllm                        -> display Qwen_Qwen2.5-14B-Instruct
#
# Phases:
#   - hosted (1-5): sequential HTTP-only calls; no GPU.
#   - local  (6) : warms a single vLLM session (--tp 2 on 2xH100), benches,
#                  tears down. Skip with --skip-vllm to run hosted-only.
#
# Each run forwards --tasks "athena-taa-canonical" --version 1 --overwrite --yes
# so any pre-existing canonical responses are replaced with a clean baseline.
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
RUN_BENCH="${SCRIPT_DIR}/run_benchmark.sh"
SERVE_VLLM="${SCRIPT_DIR}/serve_vllm.sh"

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
        -h|--help)          sed -n '3,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# Pre-flight: required API keys per enabled family. Surface missing keys
# now rather than mid-run (where they raise a stack trace from inside
# pipelines/models.py).
[[ ${SKIP_OPENAI} -eq 0 && -z "${OPENAI_API_KEY:-}" ]] && \
    { echo "[FAIL] OPENAI_API_KEY required for gpt5.2 (or pass --skip-openai)" >&2; exit 2; }
[[ ${SKIP_GEMINI} -eq 0 && -z "${GEMINI_API_KEY:-}" ]] && \
    { echo "[FAIL] GEMINI_API_KEY required for gemini-* (or pass --skip-gemini)" >&2; exit 2; }
[[ ${SKIP_HF} -eq 0 && -z "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ]] && \
    { echo "[FAIL] HF_TOKEN/HUGGINGFACE_TOKEN required for deepseek-*-hf (or pass --skip-hf)" >&2; exit 2; }

MODE_FLAGS=()
[[ ${OVERWRITE} -eq 1 ]] && MODE_FLAGS+=(--overwrite --yes)
ROWS_FLAGS=()
[[ -n "${ROWS}" ]] && ROWS_FLAGS+=(--rows "${ROWS}")

bench_one() {
    local label="$1"; shift
    local alias="$1"; shift
    local batch="$1"; shift
    # any remaining args are forwarded verbatim (e.g. --reasoning-effort high)
    echo
    echo "=================================================================="
    echo "  TAA Canonical / ${label}  (alias=${alias})"
    echo "=================================================================="
    if [[ ${DRY_RUN} -eq 1 ]]; then
        local _mode="${MODE_FLAGS[*]:-}"
        local _rows="${ROWS_FLAGS[*]:-}"
        echo "[dry-run] ${RUN_BENCH} ${alias} --tasks athena-taa-canonical --version 1 --batch ${batch} ${_mode} ${_rows} $*"
        return 0
    fi
    bash "${RUN_BENCH}" "${alias}" \
        --tasks "athena-taa-canonical" --version 1 \
        --batch "${batch}" "${MODE_FLAGS[@]}" "${ROWS_FLAGS[@]}" "$@" \
        2>&1 | tee -a "${LOG}"
}

# ------------------------------------------------------------------
# Phase 1: hosted models (HTTP-only; no GPU)
# ------------------------------------------------------------------
[[ ${SKIP_OPENAI} -eq 0 ]] && bench_one "gpt-5.2-high" "gpt5.2" "${BATCH_HOSTED}" \
    --reasoning-effort "${REASONING_EFFORT}"
[[ ${SKIP_GEMINI} -eq 0 ]] && bench_one "gemini-3-flash-preview" "gemini-3-flash" "${BATCH_HOSTED}"
[[ ${SKIP_GEMINI} -eq 0 ]] && bench_one "gemini-2.5-flash"       "gemini-2.5-flash" "${BATCH_HOSTED}"
[[ ${SKIP_HF}     -eq 0 ]] && bench_one "DeepSeek-V4-Pro"        "deepseek-v4-pro-hf"  "${BATCH_HOSTED}"
[[ ${SKIP_HF}     -eq 0 ]] && bench_one "DeepSeek-V3.2-Exp"      "deepseek-v3.2-exp-hf" "${BATCH_HOSTED}"

# ------------------------------------------------------------------
# Phase 2: local vLLM (Qwen2.5-14B-Instruct on 2xH100)
#
# Wraps serve_and_bench.sh so the vLLM server is launched, the canonical
# TAA sweep runs against it, and the server is torn down on exit. The
# alias resolves to Qwen/Qwen2.5-14B-Instruct via models.py line 145.
# --max-len 32768 matches the model's native ctx; --max-num-seqs 32
# matches BATCH_VLLM=64 with a small queue margin (see serve_vllm.sh
# header table for the per-family sizing rationale).
# ------------------------------------------------------------------
if [[ ${SKIP_VLLM} -eq 0 ]]; then
    echo
    echo "=================================================================="
    echo "  TAA Canonical / Qwen2.5-14B-Instruct  (alias=qwen2.5-14b-vllm)"
    echo "=================================================================="
    if [[ ${DRY_RUN} -eq 1 ]]; then
        _mode="${MODE_FLAGS[*]:-}"
        _rows="${ROWS_FLAGS[*]:-}"
        echo "[dry-run] serve_and_bench.sh qwen2.5-14b-vllm --tp 2 --max-len 32768 \\"
        echo "          --extra '--gpu-memory-utilization 0.92 --max-num-seqs 32' \\"
        echo "          -- --tasks athena-taa-canonical --version 1 --batch ${BATCH_VLLM} ${_mode} ${_rows}"
    else
        env BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}" \
            bash "${SCRIPT_DIR}/serve_and_bench.sh" qwen2.5-14b-vllm \
            --tp 2 --max-len 32768 \
            --extra "--gpu-memory-utilization 0.92 --max-num-seqs 32" \
            -- --tasks "athena-taa-canonical" --version 1 \
               --batch "${BATCH_VLLM}" "${MODE_FLAGS[@]}" "${ROWS_FLAGS[@]}" \
            2>&1 | tee -a "${LOG}"
    fi
fi

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
