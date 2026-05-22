#!/bin/bash

# v21 chain runner (Qwen3-30B-A3B-Thinking-2507 MoE):
# TAA -> CSE -> Recal-32b (Stages 2 -> 3 -> 4).
# Stage 1 (Core, Phase A+B) is run separately via
# run_sft_qwen3_30b_a3b_thinking_v21_core.sh (or with --include-core
# here) and must have already pushed to HF before the chained stages
# kick off (the default base model for the first chained stage is the
# Qwen3-MoE v21-core repo).
#
# Each stage launches its own run_sft_qwen3_30b_a3b_thinking_v21_<stage>.sh,
# which trains, merges, and pushes to HF before exiting. The next stage
# will only kick off if the prior stage exited 0 AND the prior stage's
# HF repo is readable -- a defensive guard against a successful local
# train + silent push failure that would otherwise burn ~5-10 h of GPU
# time at the Qwen3-MoE scale.
#
# Why Stage 4 is the 32B-tuned recal-32b recipe (not the 14B-recipe
# recalibrate): the 14B-recipe Recalibrate (lr 1e-6, probs 0.25/0.40/
# 0.35, max-samples 2400) sits at the adamw_8bit optimizer noise floor
# at 32B+ parameter scale -- the dense Qwen2.5-32B port (README-21.md
# §"Qwen2.5-32B port") confirmed that the 14B recipe drifts VSP the
# wrong way (78.9 -> 75.7) instead of recovering it. The Qwen3-MoE
# parent is peer-scale (30.5B total / 3.3B active per token) and uses
# the same adamw_8bit + Liger + ZeRO-3 footprint, so this chain ships
# the 32B-tuned recipe (lr 3e-6, probs 0.15/0.60/0.25, max-samples
# 3600) at Stage 4 by default. The 14B-recipe variant remains
# available as a standalone launcher
# (run_sft_qwen3_30b_a3b_thinking_v21_recalibrate.sh) for off-chain
# A/B work but is no longer on the default chain path. Pass
# --start-stage taa --stop-stage cse to mirror the v18.1 three-stage
# ship topology.
#
# Usage:
#   ./run_sft_qwen3_30b_a3b_thinking_v21_chain.sh
#       [--start-stage taa|cse|recal_32b]    # default: taa
#       [--stop-stage  taa|cse|recal_32b]    # default: recal_32b
#       [--include-core]                     # also run Stage 1 first
#       [--report-to wandb|none]             # forwarded to every stage
#       [--offload | --no-offload]           # forwarded to every stage
#       [--skip-eval]                        # forwarded to taa/cse stages
#       [--probs P_A,P_B,P_TAA]              # recal_32b only
#       [--max-samples N]                    # recal_32b only
#       [--lr LR]                            # recal_32b only
#       [--skip-readiness-check]             # skip pre-stage HF probe
#       [--dry-run]
#
# Estimated wall-time (8xB300 288GB SXM target; sparse 3.3B-active MoE
# fwd path + Liger + adamw_8bit, no offload):
#   TAA       ~7-10 h  -> athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa
#   CSE       ~5-7  h  -> athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
#   Recal-32b ~1.5-2 h -> athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b
#   Total ~14-19 h sequential (Stages 2 -> 3 -> 4).
#   With --include-core add Stage 1 ~14-18 h (Phase A+B) for ~28-37 h
#   end-to-end Core -> Recal-32b.
#
# Estimated wall-time (8xH100 80GB SXM fallback; --offload may be needed
# for Phase B and Recal-32b at cutoff=16384 packing=off):
#   TAA       ~13-17 h  (matches 32B chain at this stage)
#   CSE       ~9-13  h
#   Recal-32b ~3-4   h
#   Total ~25-34 h sequential.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_STAGE="taa"
STOP_STAGE="recal_32b"
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
        -h|--help) sed -n '3,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${START_STAGE}" in taa|cse|recal_32b) ;;
    *) echo "--start-stage must be taa, cse, or recal_32b (got '${START_STAGE}')" >&2; exit 1 ;;
esac
case "${STOP_STAGE}" in taa|cse|recal_32b) ;;
    *) echo "--stop-stage must be taa, cse, or recal_32b (got '${STOP_STAGE}')" >&2; exit 1 ;;
esac

stage_rank() {
    case "$1" in taa) echo 1 ;; cse) echo 2 ;; recal_32b) echo 3 ;;
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
CHAIN_LOG_DIR="${SFT_DIR}/saves/qwen3_30b_a3b_thinking_v21_chain_${CHAIN_TS}"
mkdir -p "${CHAIN_LOG_DIR}"
CHAIN_LOG="${CHAIN_LOG_DIR}/chain.log"

# --skip-eval is honoured by the plus_taa and final launchers; recal_32b
# disables eval unconditionally (see its header for the 3-shard interleave
# reason), so the flag is not forwarded to it.
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
    echo "=== v21 chain (Qwen3-MoE) :: ${label}  start $(date -u +%FT%TZ) ==="
    echo "=== log: ${stage_log}"
    echo "============================================================"
    local stage_start; stage_start=$(date +%s)
    bash "${SCRIPT_DIR}/${script}" "${COMMON_FLAGS[@]}" "$@" 2>&1 | tee "${stage_log}"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - stage_start ))
    printf '=== v21 chain (Qwen3-MoE) :: %s  exit=%d  elapsed=%dh %dm ===\n' \
        "${label}" "${rc}" $((elapsed/3600)) $(((elapsed%3600)/60))
    return ${rc}
}


{
    echo "v21 chain start  : $(date -u +%FT%TZ)  (Qwen3-30B-A3B-Thinking-2507 MoE)"
    echo "  start-stage    : ${START_STAGE}"
    echo "  stop-stage     : ${STOP_STAGE}"
    echo "  include-core   : ${INCLUDE_CORE}"
    echo "  report-to      : ${REPORT_TO}"
    echo "  offload        : ${OFFLOAD:-default (off on 8xB300)}"
    echo "  skip-eval      : $([[ ${SKIP_EVAL} -eq 1 ]] && echo on || echo off)"
    echo "  chain log dir  : ${CHAIN_LOG_DIR}"
    echo

    if [[ ${INCLUDE_CORE} -eq 1 ]]; then
        run_stage "core" "run_sft_qwen3_30b_a3b_thinking_v21_core.sh" "${STAGE_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core" "v21-core"
    fi

    # Stage 2: TAA Classic narrow drill.
    if [[ ${START_RANK} -le 1 && ${STOP_RANK} -ge 1 ]]; then
        if [[ ${START_RANK} -eq 1 && ${INCLUDE_CORE} -eq 0 ]]; then
            probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core" "v21-core (TAA base)"
        fi
        run_stage "taa" "run_sft_qwen3_30b_a3b_thinking_v21_plus_taa.sh" "${STAGE_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa" "v21-taa"
    fi

    # Stage 3: CSE letter-set drill.
    if [[ ${START_RANK} -le 2 && ${STOP_RANK} -ge 2 ]]; then
        if [[ ${START_RANK} -eq 2 ]]; then
            probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa" "v21-taa (CSE base)"
        fi
        run_stage "cse" "run_sft_qwen3_30b_a3b_thinking_v21_final.sh" "${STAGE_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse" "v21-cse"
    fi

    # Stage 4 (off-plan): Recal-32b touch-up (32B-tuned recipe).
    if [[ ${STOP_RANK} -ge 3 ]]; then
        if [[ ${START_RANK} -eq 3 ]]; then
            probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse" "v21-cse (Recal-32b base)"
        fi
        RECAL_FLAGS=()
        [[ -n "${PROBS}"       ]] && RECAL_FLAGS+=( --probs       "${PROBS}" )
        [[ -n "${MAX_SAMPLES}" ]] && RECAL_FLAGS+=( --max-samples "${MAX_SAMPLES}" )
        [[ -n "${LR}"          ]] && RECAL_FLAGS+=( --lr          "${LR}" )
        run_stage "recal_32b" "run_sft_qwen3_30b_a3b_thinking_v21_recal_32b.sh" "${RECAL_FLAGS[@]}"
        probe_hf_repo "${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b" "v21-recal-32b"
    fi

    echo
    echo "v21 chain finish : $(date -u +%FT%TZ)"
    case "${STOP_STAGE}" in
        taa)       echo "Headline checkpoint: ${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa" ;;
        cse)       echo "Headline checkpoint: ${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse  (v18.1 ship-equivalent)" ;;
        recal_32b) echo "Headline checkpoint: ${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b  (off-plan extension; 32B-tuned recipe)" ;;
    esac
} 2>&1 | tee "${CHAIN_LOG}"
