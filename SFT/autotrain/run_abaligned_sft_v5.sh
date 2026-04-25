#!/bin/bash

# Launch LoRA SFT of Llama-3.1-8B-Instruct on the v5 dataset
# (ift_data_2026_04_24_v5, 170,500 rows = 138,343 04-22 abaligned core +
# 32,157 04-24 detection/exploit/PoC addendum). Throughput-optimised
# successor to run_abaligned_sft_lora.sh, intended to land in <12 h on a
# dual-H100 80GB host.
#
# Why this script exists:
#   The 04-22 LoRA replication (run_abaligned_sft_lora.sh, 3 epochs, no
#   packing, per-device batch 4 x grad_accum 2 x 2 GPUs = eff 16) takes
#   ~24 h end-to-end. v5 keeps the recipe but turns on the throughput
#   knobs that were left at safe defaults in the replication run, and
#   layers in the addendum so the model gets exposure to Sigma rules,
#   ExploitDB entries, and GitHub PoCs alongside the original core.
#
# What stays fixed vs run_abaligned_sft_lora.sh:
#   - Base model: meta-llama/Llama-3.1-8B-Instruct
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - LoRA rank=16, alpha=32, dropout=0.05, target=all
#   - lr 5e-5, cutoff_len 2048, save_steps 500
#   - No DeepSpeed (plain DDP)
#
# What changes vs run_abaligned_sft_lora.sh:
#   - Dataset: ift_data_2026_04_24_v5 (170,500 rows, was 138,343)
#   - Sequence packing on (was off): packs short Alpaca rows up to
#     cutoff_len, eliminating padding waste and cutting optimizer steps
#     by roughly 2-3x for this corpus.
#   - per-device batch 8, grad_accum 1 (was 4 / 2): with packing, each
#     batch slot already carries up to 2048 tokens, so the larger
#     per-device batch fits in 80 GB without OOM. Effective batch on
#     2 GPUs becomes 16 (matches the replication run, so the optimizer
#     trajectory stays comparable).
#   - eval_steps 1500 (was 500): full 17K-row val pass every 500 steps
#     was the second-largest wall-clock contributor; tripling the
#     interval reclaims ~1-2 h without losing the early-stopping signal.
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v5-lora
#
# Usage:
#   ./run_abaligned_sft_v5.sh [--repo-id USER/NAME] [--output-dir DIR]
#                             [--report-to wandb|none]
#                             [--epochs N] [--lr FLOAT]
#                             [--dry-run] [--extra "..."]

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
        -h|--help) sed -n '3,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

if [[ "${GPU_COUNT}" -ge 2 ]]; then
    BATCH_DEFAULT="8"
    GRAD_ACCUM_DEFAULT="1"
else
    BATCH_DEFAULT="8"
    GRAD_ACCUM_DEFAULT="2"
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
echo "  eval         : every 1500 steps  (save every 500)"
echo "  lora         : rank=16 alpha=32 dropout=0.05 target=all"
echo "  method       : LoRA (no DeepSpeed), DDP"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
