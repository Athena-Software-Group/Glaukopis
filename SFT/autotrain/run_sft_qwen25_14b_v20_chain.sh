#!/bin/bash

# v20 chain runner: TAA -> CSE -> Recalibrate (Stages 3 -> 4 -> 5).
# Stage 1+2 (Core) is run separately via run_sft_qwen25_14b_v20_core.sh
# and must have already pushed to HF before this wrapper starts (the
# default base model for the first chained stage is the v20-core repo).
#
# Each stage launches its own run_sft_qwen25_14b_v20_<stage>.sh, which
# trains, merges, and pushes to HF before exiting. The next stage will
# only kick off if the prior stage exited 0 AND the prior stage's HF
# repo is readable -- a defensive guard against a successful local
# train + silent push failure that would otherwise burn ~6-8h of GPU.
#
# Usage:
#   ./run_sft_qwen25_14b_v20_chain.sh
#       [--start-stage taa|cse|recalibrate]   # default: taa
#       [--include-core]                      # also run Stage 1+2 first
#       [--report-to wandb|none]              # forwarded to every stage
#       [--offload | --no-offload]            # forwarded to every stage
#       [--probs P_A,P_B,P_TAA]               # recalibrate only
#       [--max-samples N]                     # recalibrate only
#       [--lr LR]                             # recalibrate only
#       [--skip-readiness-check]              # skip pre-stage HF probe
#       [--dry-run]
#
# Estimated wall-time (8xH100 80GB for TAA/CSE, 4xH100 for Recalibrate):
#   TAA   ~6-8 h    -> athena-cti-sft-qwen25-14b-v20-taa
#   CSE   ~4-6 h    -> athena-cti-sft-qwen25-14b-v20-cse
#   Recal ~95-115 m -> athena-cti-sft-qwen25-14b-v20-recalibrate
#   Total ~11-15 h sequential.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_STAGE="taa"
INCLUDE_CORE=0
REPORT_TO="wandb"
OFFLOAD=""
PROBS=""
MAX_SAMPLES=""
LR=""
SKIP_READINESS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-stage)          START_STAGE="$2";    shift 2 ;;
        --include-core)         INCLUDE_CORE=1;      shift ;;
        --report-to)            REPORT_TO="$2";      shift 2 ;;
        --offload)              OFFLOAD="--offload"; shift ;;
        --no-offload)           OFFLOAD="--no-offload"; shift ;;
        --probs)                PROBS="$2";          shift 2 ;;
        --max-samples)          MAX_SAMPLES="$2";    shift 2 ;;
        --lr)                   LR="$2";             shift 2 ;;
        --skip-readiness-check) SKIP_READINESS=1;    shift ;;
        --dry-run)              DRY_RUN=1;           shift ;;
        -h|--help) sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${START_STAGE}" in taa|cse|recalibrate) ;;
    *) echo "--start-stage must be taa, cse, or recalibrate (got '${START_STAGE}')" >&2; exit 1 ;;
esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done
: "${HF_USERNAME:?Set HF_USERNAME in SFT/.env}"

CHAIN_TS="$(date +"%Y-%m-%d-%H-%M-%S")"
CHAIN_LOG_DIR="${SFT_DIR}/saves/v20_chain_${CHAIN_TS}"
mkdir -p "${CHAIN_LOG_DIR}"
CHAIN_LOG="${CHAIN_LOG_DIR}/chain.log"

COMMON_FLAGS=( --report-to "${REPORT_TO}" )
[[ -n "${OFFLOAD}" ]] && COMMON_FLAGS+=( "${OFFLOAD}" )
[[ ${DRY_RUN} -eq 1 ]] && COMMON_FLAGS+=( --dry-run )

probe_hf_repo() {
    local repo="$1" label="$2"
    [[ ${SKIP_READINESS} -eq 1 ]] && { echo "[readiness] SKIPPED for ${label} (${repo})"; return 0; }
    [[ ${DRY_RUN} -eq 1 ]] && { echo "[readiness] dry-run: would probe ${repo}"; return 0; }
    echo "[readiness] probing ${repo} ..."
    python - "${repo}" <<'PY' || { echo "[readiness] FAILED: ${label} base repo not readable" >&2; exit 2; }
import sys
from huggingface_hub import HfApi
HfApi().model_info(sys.argv[1])
print(f"[readiness] OK: {sys.argv[1]}")
PY
}

run_stage() {
    local label="$1" script="$2"; shift 2
    local stage_log="${CHAIN_LOG_DIR}/${label}.log"
    echo "============================================================"
    echo "=== v20 chain :: ${label}  start $(date -u +%FT%TZ) ==="
    echo "=== log: ${stage_log}"
    echo "============================================================"
    local stage_start; stage_start=$(date +%s)
    bash "${SCRIPT_DIR}/${script}" "${COMMON_FLAGS[@]}" "$@" 2>&1 | tee "${stage_log}"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - stage_start ))
    printf '=== v20 chain :: %s  exit=%d  elapsed=%dh %dm ===\n' \
        "${label}" "${rc}" $((elapsed/3600)) $(((elapsed%3600)/60))
    return ${rc}
}

{
    echo "v20 chain start  : $(date -u +%FT%TZ)"
    echo "  start-stage    : ${START_STAGE}"
    echo "  include-core   : ${INCLUDE_CORE}"
    echo "  report-to      : ${REPORT_TO}"
    echo "  offload        : ${OFFLOAD:-auto}"
    echo "  chain log dir  : ${CHAIN_LOG_DIR}"
    echo

    if [[ ${INCLUDE_CORE} -eq 1 ]]; then
        run_stage "core" "run_sft_qwen25_14b_v20_core.sh"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-core" "v20-core"
    fi

    if [[ "${START_STAGE}" == "taa" ]]; then
        [[ ${INCLUDE_CORE} -eq 0 ]] && probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-core" "v20-core (TAA base)"
        run_stage "taa" "run_sft_qwen25_14b_v20_taa.sh"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-taa" "v20-taa"
    fi

    if [[ "${START_STAGE}" == "taa" || "${START_STAGE}" == "cse" ]]; then
        [[ "${START_STAGE}" == "cse" ]] && probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-taa" "v20-taa (CSE base)"
        run_stage "cse" "run_sft_qwen25_14b_v20_cse.sh"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-cse" "v20-cse"
    fi

    if [[ "${START_STAGE}" == "recalibrate" ]]; then
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-cse" "v20-cse (Recalibrate base)"
    fi

    RECAL_FLAGS=()
    [[ -n "${PROBS}"       ]] && RECAL_FLAGS+=( --probs       "${PROBS}" )
    [[ -n "${MAX_SAMPLES}" ]] && RECAL_FLAGS+=( --max-samples "${MAX_SAMPLES}" )
    [[ -n "${LR}"          ]] && RECAL_FLAGS+=( --lr          "${LR}" )
    run_stage "recalibrate" "run_sft_qwen25_14b_v20_recalibrate.sh" "${RECAL_FLAGS[@]}"

    echo
    echo "v20 chain finish : $(date -u +%FT%TZ)"
    echo "Headline checkpoint: ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-recalibrate"
} 2>&1 | tee "${CHAIN_LOG}"
