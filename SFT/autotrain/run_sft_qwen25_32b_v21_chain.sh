#!/bin/bash

# v21 chain runner (Qwen2.5-32B): TAA -> CSE -> Recalibrate
# (Stages 2 -> 3 -> 4). Stage 1 (Core, Phase A+B) is run separately via
# run_sft_qwen25_32b_v21_core.sh and must have already pushed to HF
# before this wrapper starts (the default base model for the first
# chained stage is the v21-core repo at the 32B scale).
#
# Each stage launches its own run_sft_qwen25_32b_v21_<stage>.sh, which
# trains, merges, and pushes to HF before exiting. The next stage will
# only kick off if the prior stage exited 0 AND the prior stage's HF
# repo is readable -- a defensive guard against a successful local
# train + silent push failure that would otherwise burn ~9-17h of GPU
# at the 32B scale.
#
# Forked from run_sft_qwen25_14b_v21_chain.sh; sole differences are the
# stage script names (14b -> 32b) and the HF repo names probed between
# stages. The recalibrate stage is off-plan for v21 (v21_plan.txt §3
# defines only Core/TAA/CSE); it is included here for parity with the
# 14B v21 chain that shipped at 62.3 Total via Recalibrate, and because
# the 32B port is expected to show the same Stage 3 (CSE) VSP erosion
# the 14B chain trade-off exposed. Pass --start-stage taa --stop-stage
# cse to mirror the v18.1 three-stage ship topology.
#
# Usage:
#   ./run_sft_qwen25_32b_v21_chain.sh
#       [--start-stage taa|cse|recalibrate]   # default: taa
#       [--stop-stage  taa|cse|recalibrate]   # default: recalibrate
#       [--include-core]                      # also run Stage 1 first
#       [--report-to wandb|none]              # forwarded to every stage
#       [--offload | --no-offload]            # forwarded to every stage
#       [--skip-eval]                         # forwarded to taa/cse stages
#       [--probs P_A,P_B,P_TAA]               # recalibrate only
#       [--max-samples N]                     # recalibrate only
#       [--lr LR]                             # recalibrate only
#       [--skip-readiness-check]              # skip pre-stage HF probe
#       [--dry-run]
#
# Estimated wall-time (8xH100 80GB SXM is the only supported topology
# for the 32B v21 chain -- 4xH100 OOMs at Stage 2/3 cutoff=4096 packing=on
# and Stage 4 cutoff=16384 packing=off):
#   TAA   ~13-17 h  -> athena-cti-sft-qwen25-32b-v21-taa
#   CSE   ~9-13  h  -> athena-cti-sft-qwen25-32b-v21-cse
#   Recal ~3-4   h  -> athena-cti-sft-qwen25-32b-v21-recalibrate
#   Total ~25-34 h sequential (Stages 2 -> 3 -> 4).
#   With --include-core add Stage 1 ~26-30 h (Phase A+B) for ~51-64 h
#   end-to-end Core -> Recalibrate.
#
# Estimated wall-time (8xRTX PRO 6000 96GB):
#   TAA   ~17-22 h  (ZeRO-3 PCIe Gen5 vs NVLink collective tax)
#   CSE   ~13-18 h  (same geometry; smaller corpus)
#   Recal ~4-6   h  (Phase B geometry; offload-on default at 32B)
#   Total ~34-46 h sequential.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_STAGE="taa"
STOP_STAGE="recalibrate"
INCLUDE_CORE=0
REPORT_TO="wandb"
OFFLOAD=""
SKIP_EVAL=0
PROBS=""
MAX_SAMPLES=""
LR=""
SKIP_READINESS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-stage)          START_STAGE="$2";    shift 2 ;;
        --stop-stage)           STOP_STAGE="$2";     shift 2 ;;
        --include-core)         INCLUDE_CORE=1;      shift ;;
        --report-to)            REPORT_TO="$2";      shift 2 ;;
        --offload)              OFFLOAD="--offload"; shift ;;
        --no-offload)           OFFLOAD="--no-offload"; shift ;;
        --skip-eval)            SKIP_EVAL=1;         shift ;;
        --probs)                PROBS="$2";          shift 2 ;;
        --max-samples)          MAX_SAMPLES="$2";    shift 2 ;;
        --lr)                   LR="$2";             shift 2 ;;
        --skip-readiness-check) SKIP_READINESS=1;    shift ;;
        --dry-run)              DRY_RUN=1;           shift ;;
        -h|--help) sed -n '3,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${START_STAGE}" in taa|cse|recalibrate) ;;
    *) echo "--start-stage must be taa, cse, or recalibrate (got '${START_STAGE}')" >&2; exit 1 ;;
esac
case "${STOP_STAGE}" in taa|cse|recalibrate) ;;
    *) echo "--stop-stage must be taa, cse, or recalibrate (got '${STOP_STAGE}')" >&2; exit 1 ;;
esac

# Encode stage ordering for start/stop comparisons.
stage_rank() {
    case "$1" in taa) echo 1 ;; cse) echo 2 ;; recalibrate) echo 3 ;;
                 *) echo 99 ;; esac
}
START_RANK=$(stage_rank "${START_STAGE}")
STOP_RANK=$(stage_rank "${STOP_STAGE}")
if [[ ${START_RANK} -gt ${STOP_RANK} ]]; then
    echo "--start-stage (${START_STAGE}) must not be later than --stop-stage (${STOP_STAGE})" >&2
    exit 1
fi

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done
: "${HF_USERNAME:?Set HF_USERNAME in SFT/.env}"

CHAIN_TS="$(date +"%Y-%m-%d-%H-%M-%S")"
CHAIN_LOG_DIR="${SFT_DIR}/saves/v21_chain_${CHAIN_TS}"
mkdir -p "${CHAIN_LOG_DIR}"
CHAIN_LOG="${CHAIN_LOG_DIR}/chain.log"

# --skip-eval is honoured by run_sft_qwen25_32b_v21_plus_taa.sh and
# run_sft_qwen25_32b_v21_final.sh; the recalibrate stage already disables
# eval unconditionally (see its header for the 3-shard interleave reason),
# so the flag is not forwarded to it.
COMMON_FLAGS=( --report-to "${REPORT_TO}" )
[[ -n "${OFFLOAD}" ]] && COMMON_FLAGS+=( "${OFFLOAD}" )
[[ ${DRY_RUN} -eq 1 ]] && COMMON_FLAGS+=( --dry-run )

STAGE_FLAGS=()
[[ ${SKIP_EVAL} -eq 1 ]] && STAGE_FLAGS+=( --skip-eval )

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
    echo "=== v21 chain :: ${label}  start $(date -u +%FT%TZ) ==="
    echo "=== log: ${stage_log}"
    echo "============================================================"
    local stage_start; stage_start=$(date +%s)
    bash "${SCRIPT_DIR}/${script}" "${COMMON_FLAGS[@]}" "$@" 2>&1 | tee "${stage_log}"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - stage_start ))
    printf '=== v21 chain :: %s  exit=%d  elapsed=%dh %dm ===\n' \
        "${label}" "${rc}" $((elapsed/3600)) $(((elapsed%3600)/60))
    return ${rc}
}


{
    echo "v21 chain start  : $(date -u +%FT%TZ)"
    echo "  start-stage    : ${START_STAGE}"
    echo "  stop-stage     : ${STOP_STAGE}"
    echo "  include-core   : ${INCLUDE_CORE}"
    echo "  report-to      : ${REPORT_TO}"
    echo "  offload        : ${OFFLOAD:-auto}"
    echo "  skip-eval      : $([[ ${SKIP_EVAL} -eq 1 ]] && echo on || echo off)"
    echo "  chain log dir  : ${CHAIN_LOG_DIR}"
    echo

    if [[ ${INCLUDE_CORE} -eq 1 ]]; then
        run_stage "core" "run_sft_qwen25_32b_v21_core.sh" "${STAGE_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-core" "v21-core"
    fi

    # Stage 2: TAA Classic narrow drill (run_sft_qwen25_32b_v21_plus_taa.sh).
    if [[ ${START_RANK} -le 1 && ${STOP_RANK} -ge 1 ]]; then
        if [[ ${START_RANK} -eq 1 && ${INCLUDE_CORE} -eq 0 ]]; then
            probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-core" "v21-core (TAA base)"
        fi
        run_stage "taa" "run_sft_qwen25_32b_v21_plus_taa.sh" "${STAGE_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-taa" "v21-taa"
    fi

    # Stage 3: CSE letter-set drill (run_sft_qwen25_32b_v21_final.sh).
    if [[ ${START_RANK} -le 2 && ${STOP_RANK} -ge 2 ]]; then
        if [[ ${START_RANK} -eq 2 ]]; then
            probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-taa" "v21-taa (CSE base)"
        fi
        run_stage "cse" "run_sft_qwen25_32b_v21_final.sh" "${STAGE_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-cse" "v21-cse"
    fi

    # Stage 4 (off-plan): Recalibrate touch-up.
    if [[ ${STOP_RANK} -ge 3 ]]; then
        if [[ ${START_RANK} -eq 3 ]]; then
            probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-cse" "v21-cse (Recalibrate base)"
        fi
        RECAL_FLAGS=()
        [[ -n "${PROBS}"       ]] && RECAL_FLAGS+=( --probs       "${PROBS}" )
        [[ -n "${MAX_SAMPLES}" ]] && RECAL_FLAGS+=( --max-samples "${MAX_SAMPLES}" )
        [[ -n "${LR}"          ]] && RECAL_FLAGS+=( --lr          "${LR}" )
        run_stage "recalibrate" "run_sft_qwen25_32b_v21_recalibrate.sh" "${RECAL_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-recalibrate" "v21-recalibrate"
    fi

    echo
    echo "v21 chain finish : $(date -u +%FT%TZ)"
    case "${STOP_STAGE}" in
        taa)         echo "Headline checkpoint: ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-taa" ;;
        cse)         echo "Headline checkpoint: ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-cse  (v18.1 ship-equivalent)" ;;
        recalibrate) echo "Headline checkpoint: ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-recalibrate  (off-plan extension)" ;;
    esac
} 2>&1 | tee "${CHAIN_LOG}"
