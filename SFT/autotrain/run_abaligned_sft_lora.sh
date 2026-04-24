#!/bin/bash

# Launch LoRA SFT of Llama-3.1-8B-Instruct on the ORIGINAL AthenaBench-aligned
# 2026-04-22 ift dataset (ift_data_2026_04_22, 138,343 rows). Counterpart to
# the retired full-parameter recipe that produced
# hf://asg-ai/athena-cti-sft-llama31-8b-abaligned (combined 52.0 on the 6-task
# AthenaBench suite), re-implemented with a low-rank adapter to isolate the
# method-vs-data contribution to that result.
#
# Why this script exists:
#   The three SFT versions pushed so far sit on a monotonic regression curve:
#       abaligned (full-param, 04-22 data) : 52.0 combined
#       abaligned-v3 (full-param, 04-23 trimmed-v3 + alpaca mix) : 45.3
#       abaligned-v4 (LoRA, 04-23 abaligned-v4, no MCQ/TAA)      : 38.9
#   Both changes (dataset trim + full->LoRA) confound the comparison. This
#   script pins the dataset at the 04-22 original so the only free variable
#   vs the winning run is the training method. Its combined score tells us
#   how much of the 52.0 -> 38.9 drop was caused by switching to LoRA on its
#   own vs the subsequent dataset changes.
#
# What stays fixed vs the retired full-param recipe:
#   - Base model: meta-llama/Llama-3.1-8B-Instruct
#   - Dataset: ift_data_2026_04_22 (138,343 rows; 8% AB.* AthenaBench-aligned
#     addendum, 92% Sophia core + YN/X cross-framework templates)
#   - No alpaca_en_demo mix-in (the mix-in was an addition in v3, not part of
#     the original abaligned recipe)
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - Effective global batch 16 (4 per-device x 2 grad_accum x 2 GPUs)
#   - cutoff_len 2048, save_steps 500
#
# What changes vs the retired full-param recipe:
#   - --finetuning lora (was full)
#   - lr 5e-5 (was 1e-5; LoRA needs ~5-10x higher LR than full-param because
#     the low-rank update has less gradient signal per step)
#   - No DeepSpeed ZeRO-3 (LoRA trainable params are <1% of model, plain DDP
#     is fine and avoids the end-of-train optimizer-reload OOM pattern)
#   - LoRA rank=16, alpha=32, dropout=0.05, target=all (same as v4, so v4
#     and this run differ only in dataset and epoch count)
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-lora
#
# Usage:
#   ./run_abaligned_sft_lora.sh [--repo-id USER/NAME] [--output-dir DIR]
#                               [--report-to wandb|none]
#                               [--epochs N] [--lr FLOAT]
#                               [--dry-run] [--extra "..."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
EXTRA_USER=""
EPOCHS="3"
LR="5e-05"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)    REPO_ID="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        --report-to)  REPORT_TO="$2";   shift 2 ;;
        --epochs)     EPOCHS="$2";      shift 2 ;;
        --lr)         LR="$2";          shift 2 ;;
        --extra)      EXTRA_USER="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        -h|--help) sed -n '3,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        set -a; source "${env_file}"; set +a
    fi
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-lora"
fi

DATASET_NAME="ift_data_2026_04_22"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       This file is gitignored (~320 MB). Transfer it to this host" >&2
    echo "       before running, e.g.:" >&2
    echo "         rsync -avP workstation:Glaukopis/SFT/data/${DATASET_NAME}.json \\" >&2
    echo "               ${SFT_DIR}/data/" >&2
    exit 2
fi

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

if [[ "${GPU_COUNT}" -ge 2 ]]; then
    BATCH_DEFAULT="4"
    GRAD_ACCUM_DEFAULT="2"
else
    BATCH_DEFAULT="4"
    GRAD_ACCUM_DEFAULT="4"
fi
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * (GPU_COUNT > 0 ? GPU_COUNT : 1) ))

export LORA_RANK_DEFAULT=16
export LORA_ALPHA_DEFAULT=32
export LORA_DROPOUT_DEFAULT=0.05
export LORA_TARGET_DEFAULT=all

EXTRA_DEFAULT="--save_total_limit 5 --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False --save_only_model True"

if [[ -n "${EXTRA_USER}" ]]; then
    EXTRA_ALL="${EXTRA_DEFAULT} ${EXTRA_USER}"
else
    EXTRA_ALL="${EXTRA_DEFAULT}"
fi

RUN_TRAIN_ARGS=(
    --model        "meta-llama/Llama-3.1-8B-Instruct"
    --dataset      "${DATASET_NAME}"
    --template     "llama3"
    --finetuning   "lora"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "2048"
    --save-steps   "500"
    --max-samples  "200000"
    --report-to    "${REPORT_TO}"
    --push-to-hf   "${REPO_ID}"
    --extra        "${EXTRA_ALL}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
    RUN_TRAIN_ARGS+=( --output-dir "${OUTPUT_DIR}" )
fi
if [[ ${DRY_RUN} -eq 1 ]]; then
    RUN_TRAIN_ARGS+=( --dry-run )
fi

export FORCE_TORCHRUN=1

for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    if [[ -z "${!var:-}" ]]; then
        unset "${var}"
    fi
done

echo "=== AthenaBench-aligned (original 04-22 data) LoRA SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset file : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective batch ~= ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  lora         : rank=16 alpha=32 dropout=0.05 target=all"
echo "  method       : LoRA (no DeepSpeed), DDP"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
