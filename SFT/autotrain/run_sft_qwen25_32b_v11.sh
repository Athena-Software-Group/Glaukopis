#!/bin/bash

# v11 single-pass full-parameter SFT of Qwen2.5-32B-Instruct on the
# unified v11 corpus (ift_data_2026_05_03_v11_train.json, ~198,644 rows
# after held-out val split).
#
# Per v11_plan.txt §6.2: same dataset list and deltas as the 14B
# launcher (run_sft_qwen25_14b_v11.sh). The 32B launcher reuses the
# single-pass shape rather than the two-phase Phase A/B chain that v8
# 32B used; v11's unified manifest plus stratified shuffle removes the
# RMS catalog-collapse pressure that originally motivated the split.
#
# Naming migration (per v11_plan.txt §0): "abaligned" suffix dropped
# from script name + HF repo id + harness alias.
#
# Corpus shape: identical to run_sft_qwen25_14b_v11.sh; see that script
# for the full v10 -> v11 delta list.
#
# Training shape (single phase):
#   - Base model     : Qwen/Qwen2.5-32B-Instruct
#   - Template       : qwen
#   - 1 epoch (default), cosine, 5% warmup, bf16
#   - lr 1e-5
#   - cutoff_len 8192, packing on
#   - effective batch 16 across visible GPUs
#   - DeepSpeed ZeRO-3 with offload forced on (32B weight footprint)
#   - --eval_dataset wired to ift_data_2026_05_03_v11_val
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-qwen25-32b-v11
#
# Usage:
#   ./run_sft_qwen25_32b_v11.sh [--repo-id USER/NAME]
#                               [--output-dir DIR]
#                               [--report-to wandb|none]
#                               [--epochs N] [--lr FLOAT]
#                               [--offload | --no-offload]
#                               [--dry-run] [--extra "..."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
EXTRA_USER=""
EPOCHS="1"
LR="1e-05"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)    REPO_ID="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        --report-to)  REPORT_TO="$2";   shift 2 ;;
        --epochs)     EPOCHS="$2";      shift 2 ;;
        --lr)         LR="$2";          shift 2 ;;
        --extra)      EXTRA_USER="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        --offload)    OFFLOAD="on";     shift ;;
        --no-offload) OFFLOAD="off";    shift ;;
        -h|--help) sed -n '3,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-32b-v11"
fi

DATASET_NAME="ift_data_2026_05_03_v11_train"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
VAL_NAME="ift_data_2026_05_03_v11_val"
VAL_FILE="${SFT_DIR}/data/${VAL_NAME}.json"
if [[ ! -f "${DATASET_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "[FAIL] training/validation dataset not found:" >&2
    [[ ! -f "${DATASET_FILE}" ]] && echo "       ${DATASET_FILE}" >&2
    [[ ! -f "${VAL_FILE}" ]]     && echo "       ${VAL_FILE}" >&2
    echo "       Generate locally then split with build_val_slice.py;" >&2
    echo "       or rsync from the build host. See" >&2
    echo "       SFT/autotrain/run_sft_qwen25_14b_v11.sh for the recipe." >&2
    exit 2
fi

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 8 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
BATCH_DEFAULT=1
GRAD_ACCUM_DEFAULT=$(( 16 / (BATCH_DEFAULT * EFFECTIVE_GPUS) ))
[[ ${GRAD_ACCUM_DEFAULT} -lt 1 ]] && GRAD_ACCUM_DEFAULT=1
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * EFFECTIVE_GPUS ))

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit --eval_dataset ${VAL_NAME} --val_size 0"
EXTRA_ALL="${EXTRA_DEFAULT}${EXTRA_USER:+ ${EXTRA_USER}}"

RUN_TRAIN_ARGS=(
    --model        "Qwen/Qwen2.5-32B-Instruct"
    --dataset      "${DATASET_NAME},tulu_3_sft_mixture,alpaca_en_demo"
    --template     "qwen"
    --finetuning   "full"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "8192"
    --save-steps   "500"
    --eval-steps   "500"
    --packing      "true"
    --max-samples  "240000"
    --report-to    "${REPORT_TO}"
    --push-to-hf   "${REPO_ID}"
    --extra        "${EXTRA_ALL}"
)
[[ -n "${OUTPUT_DIR}" ]] && RUN_TRAIN_ARGS+=( --output-dir "${OUTPUT_DIR}" )
[[ ${DRY_RUN} -eq 1 ]]   && RUN_TRAIN_ARGS+=( --dry-run )

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done

echo "=== v11 (Qwen2.5-32B-Instruct, single-pass) full SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset      : ${DATASET_FILE}"
echo "  val          : ${VAL_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  cutoff_len   : 8192  (packing on)"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
