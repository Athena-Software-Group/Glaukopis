#!/bin/bash

# Launch LoRA SFT of Llama-3.1-8B-Instruct on the AthenaBench-aligned v4 ift
# dataset (ift_data_2026_04_23_abaligned_v4) via LLaMA-Factory. Counterpart
# to run_abaligned_sft.sh (full-parameter + DeepSpeed ZeRO-3); this one keeps
# the base weights frozen and trains a rank-16 LoRA adapter.
#
# Why LoRA for v4:
#   The v3 full-parameter run produced a checkpoint that mode-collapsed on
#   MCQ (A-bias) and TAA (Lazarus-bias). The TAA collapse happened on a task
#   that was not in v3 training at all, implicating the SFT procedure itself
#   (full-param updates over 3 epochs) in degrading base-model world
#   knowledge. LoRA keeps the base weights frozen and concentrates the
#   update in a low-rank subspace, which almost never produces this kind
#   of collapse on a narrow domain dataset.
#
# What else changed vs v3:
#   - Dataset: ift_data_2026_04_23_abaligned_v4 (15,115 rows, no MCQ, no TAA)
#   - No alpaca_en_demo mixin (its contribution was un-ablated and it is
#     the most suspected collapse driver after full-param itself)
#   - 1 epoch (v3 did 3; LoRA on a narrow set rarely benefits from more)
#   - LoRA rank=16, alpha=32, dropout=0.05, target=all
#   - LR=5e-5 (half of the LoRA default 1e-4, reflecting narrow-domain SFT)
#   - No DeepSpeed: LoRA trainable params are tiny, plain DDP is fine
#   - Final merged model is pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v4
#     (upload_to_hf.py merges the adapter before upload, so SFT/eval
#      eval pulls a full merged model and needs no change).
#
# Usage:
#   ./run_abaligned_sft_v4.sh [--repo-id USER/NAME] [--output-dir DIR]
#                             [--report-to wandb|none]
#                             [--dry-run] [--extra "..."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
EXTRA_USER=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)    REPO_ID="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        --report-to)  REPORT_TO="$2";   shift 2 ;;
        --extra)      EXTRA_USER="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        -h|--help) sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v4"
fi

DATASET_NAME="ift_data_2026_04_23_abaligned_v4"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       This file is gitignored (37MB). Transfer it to this host" >&2
    echo "       before running, e.g.:" >&2
    echo "         rsync -avP workstation:Glaukopis/SFT/data/${DATASET_NAME}.json \\" >&2
    echo "               ${SFT_DIR}/data/" >&2
    exit 2
fi

# LoRA + 2x H100 80GB: base weights are frozen (16 GB bf16), trainable
# adapter is <1% of parameters, so there is no need for ZeRO sharding.
# per_device_train_batch_size=4 x grad_accum=2 x 2 GPUs = global batch 16
# (same effective batch as v3, for apples-to-apples loss comparison).
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

# LoRA hyperparameters: override the r=64/a=128 defaults in run_train.sh
# via env vars (not --extra) so the llamafactory-cli command line and
# train_config.json snapshot each have a single unambiguous value per flag.
# rank 16, alpha 32, dropout 0.05, target all linear layers.
export LORA_RANK_DEFAULT=16
export LORA_ALPHA_DEFAULT=32
export LORA_DROPOUT_DEFAULT=0.05
export LORA_TARGET_DEFAULT=all

# save_only_model=True keeps adapters small and avoids the end-of-train
# optimizer reload OOM pattern seen under ZeRO-3 (not an issue here, but
# cheap to keep on). load_best_model_at_end=True so the output dir ends
# up pointing at the minimum-eval checkpoint when we push to HF.
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
    --epochs       "1"
    --lr           "5e-05"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "2048"
    --save-steps   "200"
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

echo "=== AthenaBench-aligned v4 LoRA SFT (LLaMA-Factory) ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset file : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective batch ~= ${EFFECTIVE_BATCH})"
echo "  lora         : rank=16 alpha=32 dropout=0.05 target=all"
echo "  method       : LoRA (no DeepSpeed), DDP"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
