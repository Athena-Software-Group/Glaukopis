#!/bin/bash

# TAA Canonical (athena-taa-canonical) sweep against DeepSeek-V3.2-Exp
# routed through HF Inference Providers. Pure HTTP; no GPU. Per-model
# carve-out of the six-model run_taa_canonical_baselines.sh orchestrator --
# runnable independently when only the V3.2-Exp row needs a re-bench.
#
# Resolves to:
#   alias    : deepseek-v3.2-exp-hf
#   model    : deepseek-ai/DeepSeek-V3.2-Exp
#   display  : deepseek-ai_DeepSeek-V3.2-Exp
#
# Wallclock: ~2-4 min (single task, 100 rows; depends on provider warmup).
#
# Usage:
#   conda activate ctibench
#   bash run_taa_canonical_deepseek_v3_2_exp.sh [--rows N] [--batch N]
#                                               [--no-overwrite] [--dry-run]
#
# Environment:
#   HF_TOKEN              required (HUGGINGFACE_TOKEN also accepted;
#                         loaded from SFT/.env if not already exported).
#                         Token must have 'Inference Providers' scope.
#   HF_INFERENCE_ENDPOINT_URL  optional; when set, bypasses provider
#                         auto-routing and targets a dedicated endpoint URL.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BENCH="${SCRIPT_DIR}/run_benchmark.sh"

ROWS=""
BATCH="16"
OVERWRITE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows)         ROWS="$2"; shift 2 ;;
        --batch)        BATCH="$2"; shift 2 ;;
        --no-overwrite) OVERWRITE=0; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ]] && \
    { echo "[FAIL] HF_TOKEN/HUGGINGFACE_TOKEN required for deepseek-v3.2-exp-hf" >&2; exit 2; }

MODE_FLAGS=()
[[ ${OVERWRITE} -eq 1 ]] && MODE_FLAGS+=(--overwrite --yes)
ROWS_FLAGS=()
[[ -n "${ROWS}" ]] && ROWS_FLAGS+=(--rows "${ROWS}")

echo "=================================================================="
echo "  TAA Canonical / DeepSeek-V3.2-Exp  (alias=deepseek-v3.2-exp-hf)"
echo "=================================================================="

if [[ ${DRY_RUN} -eq 1 ]]; then
    _mode="${MODE_FLAGS[*]:-}"
    _rows="${ROWS_FLAGS[*]:-}"
    echo "[dry-run] ${RUN_BENCH} deepseek-v3.2-exp-hf --tasks athena-taa-canonical --version 1 --batch ${BATCH} ${_mode} ${_rows}"
    exit 0
fi

exec bash "${RUN_BENCH}" deepseek-v3.2-exp-hf \
    --tasks "athena-taa-canonical" --version 1 \
    --batch "${BATCH}" "${MODE_FLAGS[@]}" "${ROWS_FLAGS[@]}"
