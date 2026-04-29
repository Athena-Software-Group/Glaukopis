#!/bin/bash

# Two-phase full-parameter SFT of Qwen2.5-14B-Instruct for the v8-large
# corpus. Mirrors run_abaligned_sft_qwen25_32b_v8.sh exactly except for
# the base model swap; pairs with run_abaligned_sft_qwen25_14b_v8small.sh
# to produce the small-vs-large recipe ablation at 14B.
#
# Why 14B + v8-large is in the matrix:
#   v8-large was sized for 32B's parameter budget. Running it at 14B tests
#   whether the recipe's gains are intrinsic (data quality / curriculum)
#   or dependent on >=32B capacity. A meaningful 14B-large vs 32B-large
#   delta justifies 32B in future studies; a small delta argues for 14B
#   as the production size. 14B is also large enough that v8-large's
#   1.17M-row Phase A at 1 epoch sits below the catastrophic-forgetting
#   threshold (v7 14B trained ~700K example-passes successfully).
#
# Phase shape (identical to 32B v8 launcher):
#   Phase A -- broad knowledge re-anchor.
#     - Datasets   : ift_data_2026_04_26_combined_v7,tulu_3_sft_mixture,alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 4096, packing on
#     - Effective batch 16
#   Phase B -- format + long-context specialization.
#     - Datasets   : ift_data_2026_04_29_json_v8,ift_data_2026_04_29_longctx_v8
#     - 1 epoch, lr 5e-6, cutoff 16384, packing OFF
#     - Effective batch 8 (cutoff 4x => half the effective batch)
#     - --model points at Phase A's output dir
#
# Only Phase B's final merged model is pushed to HF.
#
# Usage:
#   ./run_abaligned_sft_qwen25_14b_v8.sh [--repo-id USER/NAME]
#                                        [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                        [--report-to wandb|none]
#                                        [--phase a|b|both]   # default: both
#                                        [--offload | --no-offload]
#                                        [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
PHASE_A_DIR=""
PHASE_B_DIR=""
REPORT_TO="wandb"
PHASE="both"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --phase-a-dir)  PHASE_A_DIR="$2";  shift 2 ;;
        --phase-b-dir)  PHASE_B_DIR="$2";  shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --phase)        PHASE="$2";        shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        -h|--help) sed -n '3,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|both) ;; *) echo "--phase must be a|b|both" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v8"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${PHASE_A_DIR}" ]] && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v8_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]] && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v8_phase_b_${TIMESTAMP}"

PHASE_A_DATASETS="ift_data_2026_04_26_combined_v7,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_04_29_json_v8,ift_data_2026_04_29_longctx_v8"

if [[ "${PHASE}" != "a" ]]; then
    for ds in ift_data_2026_04_29_json_v8 ift_data_2026_04_29_longctx_v8; do
        if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
            echo "[FAIL] Phase B dataset missing: SFT/data/${ds}.json" >&2
            echo "       Generate via tmpl_gen + stitch_long_context.py and run" >&2
            echo "       tmpl_gen/scripts/dedup_against_evals.py before training." >&2
            exit 2
        fi
    done
fi

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 4 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))

A_BATCH=1
A_GA=$(( 16 / (A_BATCH * EFFECTIVE_GPUS) )); [[ ${A_GA} -lt 1 ]] && A_GA=1
B_BATCH=1
B_GA=$(( 8 / (B_BATCH * EFFECTIVE_GPUS) ));  [[ ${B_GA} -lt 1 ]] && B_GA=1

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 5 --save_only_model True --enable_liger_kernel True --optim adamw_8bit"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_phase_a() {
    echo "=== v8 Phase A (Qwen2.5-14B): broad knowledge re-anchor (cutoff=4096, packing=on) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen2.5-14B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 4096 --save-steps 200 --eval-steps 200 --packing true \
        --max-samples 250000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_b() {
    echo "=== v8 Phase B (Qwen2.5-14B): format + long-context (cutoff=16384, packing=off) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${B_BATCH} --grad-accum ${B_GA} \
        --cutoff 16384 --save-steps 100 --eval-steps 100 --packing false \
        --max-samples 50000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_B_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}"
echo "  phase A dir  : ${PHASE_A_DIR}"
echo "  phase B dir  : ${PHASE_B_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

[[ "${PHASE}" == "a" || "${PHASE}" == "both" ]] && run_phase_a
[[ "${PHASE}" == "b" || "${PHASE}" == "both" ]] && run_phase_b
