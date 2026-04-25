#!/bin/bash

# Launch LoRA SFT of Llama-3.1-8B-Instruct on the v5 dataset
# (ift_data_2026_04_24_v5, 170,500 rows). LoRA counterpart to
# run_abaligned_sft_v5.sh (full-parameter + DeepSpeed ZeRO-3); pinned to
# the same dataset, template, epochs, effective batch, packing, and
# save/eval cadence so the only free variable vs the 45.28-combined
# full-param v5 result is the optimisation method.
#
# Why this script exists (and why r=64, not r=16):
#   The 04-22 LoRA replication (run_abaligned_sft_lora.sh) at r=16 was
#   benchmarked at 42.1 combined on AthenaBench v1, vs the original
#   full-param 'abaligned' baseline at 52.0 on the same data -- a 9.9
#   point regression concentrated in open-generation tasks (VSP -20.6,
#   TAA -16.0, ATE -14.0). With v5 being ~23% larger than 04-22 and the
#   addendum skewing further toward open generation, r=16 would likely
#   regress worse than 42.1. r=64 (alpha=128) is the smallest rank that
#   has a credible shot at recovering generation tasks; it is also the
#   default rank in utils/run_train.sh, so this run reuses that default
#   rather than overriding via env vars.
#
# What stays fixed vs run_abaligned_sft_v5.sh:
#   - Base model: meta-llama/Llama-3.1-8B-Instruct
#   - Dataset: ift_data_2026_04_24_v5 + alpaca_en_demo mix-in (identical
#     dataset config so the LoRA-vs-full delta is not confounded by data)
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - lr 5e-5 (LoRA needs ~5x higher LR than the full-param 1e-5 because
#     the low-rank update has less gradient signal per step; matches the
#     prior abaligned-lora recipe)
#   - cutoff_len 2048, packing on, save_steps + eval_steps = 1500
#   - Effective global batch 16 (4 per-device x 2 grad_accum x 2 GPUs).
#     Same gradient noise scale as the v5 full-param run; LoRA's lower
#     activation memory lets us double per-device batch from 2 to 4
#     without exceeding 80 GB.
#
# What changes vs run_abaligned_sft_v5.sh:
#   - --finetuning lora (was full)
#   - LoRA rank=64, alpha=128, dropout=0.05, target=all (run_train.sh
#     defaults; no env-var overrides)
#   - No DeepSpeed: LoRA trainable params are <2% of model weights,
#     plain DDP is faster and avoids the end-of-train ZeRO-3 OOM that
#     forced --save_only_model in the full-param recipe.
#   - load_best_model_at_end=True (no DeepSpeed = no validator conflict
#     with --save_only_model, so we can keep best-of-eval selection).
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v5-lora
#     (upload_to_hf.py runs llamafactory-cli export to merge the adapter
#     into base weights before upload, so SFT/test consumers see a full
#     model and don't need to know LoRA was used.)
#
# Usage:
#   ./run_abaligned_sft_v5_lora.sh [--repo-id USER/NAME] [--output-dir DIR]
#                                  [--report-to wandb|none]
#                                  [--epochs N] [--lr FLOAT]
#                                  [--dry-run] [--extra "..."]

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
        -h|--help) sed -n '3,58p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v5-lora"
fi

DATASET_NAME="ift_data_2026_04_24_v5"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       This file is gitignored (~166 MB). Transfer it to this host" >&2
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

# LoRA frees enough activation memory that we can run per-device batch 4
# (vs full-param's 2) and still fit a 2048-cutoff packed batch on 80GB.
# Effective batch is held at 16 to mirror the v5 full-param trajectory.
if [[ "${GPU_COUNT}" -ge 2 ]]; then
    BATCH_DEFAULT="4"
    GRAD_ACCUM_DEFAULT="2"
else
    BATCH_DEFAULT="4"
    GRAD_ACCUM_DEFAULT="4"
fi
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * (GPU_COUNT > 0 ? GPU_COUNT : 1) ))

# load_best_model_at_end is safe here (no DeepSpeed) -- pick the lowest
# eval_loss checkpoint for the HF push rather than the last step.
EXTRA_DEFAULT="--save_total_limit 5 --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False --save_only_model True"

if [[ -n "${EXTRA_USER}" ]]; then
    EXTRA_ALL="${EXTRA_DEFAULT} ${EXTRA_USER}"
else
    EXTRA_ALL="${EXTRA_DEFAULT}"
fi

RUN_TRAIN_ARGS=(
    --model        "meta-llama/Llama-3.1-8B-Instruct"
    --dataset      "${DATASET_NAME},alpaca_en_demo"
    --template     "llama3"
    --finetuning   "lora"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "2048"
    --save-steps   "1500"
    --eval-steps   "1500"
    --packing      "true"
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

echo "=== AthenaBench-aligned v5 (04-22 core + 04-24 addendum) LoRA SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset file : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective batch ~= ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  packing      : true  (cutoff_len=2048)"
echo "  eval / save  : every 1500 steps  (paired so each eval has a recoverable checkpoint)"
echo "  lora         : rank=64 alpha=128 dropout=0.05 target=all (run_train.sh defaults)"
echo "  method       : LoRA (no DeepSpeed), DDP"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
