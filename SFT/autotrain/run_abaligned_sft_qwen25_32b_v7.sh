#!/bin/bash

# Launch full-parameter SFT of Qwen2.5-32B-Instruct on the consolidated v7
# dataset via LLaMA-Factory + DeepSpeed ZeRO-3. Trains on
# ift_data_2026_04_26_combined_v7 (180,533 rows: v5 broad SFT coverage +
# v7 RMS-only addendum, pre-merged into one file) plus alpaca_en_demo
# (instruction-following baseline). Total ~181.5k rows.
#
# Why this script exists (Qwen2.5-14B v7 -> Qwen2.5-32B v7 capacity test):
#   The Qwen2.5-14B v7 SFT extended the Llama-3.1-8B v7 capacity test
#   (run_abaligned_sft_qwen25_14b_v7.sh). CyberMetric literature reports
#   Qwen2.5-32B-Instruct as a notably stronger zero-shot performer than
#   the 14B sibling. This script tests whether the additional capacity
#   absorbs the v7 curriculum without eroding the model's broad cyber
#   knowledge baseline (CyberMetric) while still recovering AthenaBench
#   task specializations the v7 dataset is shaped for.
#
# What stays fixed vs run_abaligned_sft_qwen25_14b_v7.sh:
#   - Dataset: ift_data_2026_04_26_combined_v7 + alpaca_en_demo
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - lr 1e-5
#   - cutoff_len 4096, packing on, save_only_model=True
#   - save/eval every 200 steps
#   - Template: qwen
#
# What changes vs run_abaligned_sft_qwen25_14b_v7.sh:
#   - Base model: Qwen/Qwen2.5-32B-Instruct (was Qwen/Qwen2.5-14B-Instruct)
#   - DeepSpeed ZeRO-3 CPU offload auto-enabled on <8 GPUs (was <2). Full
#     SFT of 32B in bf16 needs ~112 GB/GPU of weights+grads+Adam state on
#     4 GPUs without offload, which exceeds H100 80GB. Pass --no-offload
#     only when running on >=8 x 80GB GPUs.
#   - per-device batch fixed at 1; grad_accum derived from GPU_COUNT to
#     hold the effective batch at 16 across cluster sizes (matches the
#     14B v7 run for direct comparability):
#       1 GPU  -> grad_accum 16
#       2 GPUs -> grad_accum 8
#       4 GPUs -> grad_accum 4
#       8 GPUs -> grad_accum 2
#   - Liger Kernel enabled by default. Qwen2.5's 152K vocab makes the
#     fp32 cross-entropy buffer ~2.32 GiB at cutoff_len=4096, which OOMs
#     after ZeRO-3 weights+grads+Adam state on 8x80GB. Liger's fused
#     linear CE never materializes the full logits tensor.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exported to keep
#     the allocator from fragmenting the ~430 MB reserved-but-unused
#     pool that pushes us over the edge on the larger vocab.
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-qwen25-32b-abaligned-v7
#
# Usage:
#   ./run_abaligned_sft_qwen25_32b_v7.sh [--repo-id USER/NAME] [--output-dir DIR]
#                                        [--report-to wandb|none]
#                                        [--epochs N] [--lr FLOAT]
#                                        [--offload | --no-offload]
#                                        [--dry-run] [--extra "..."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
EXTRA_USER=""
EPOCHS="3"
LR="1e-05"
DRY_RUN=0
OFFLOAD="auto"    # auto | on | off

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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-32b-abaligned-v7"
fi

DATASET_NAME="ift_data_2026_04_26_combined_v7"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       The combined v7 file (~193 MB) is gitignored. Either" >&2
    echo "       regenerate locally with tmpl_gen (Section A of" >&2
    echo "       tmpl_gen/templates/04262026/Sophia-CTI-Templates-Combined-v7.txt" >&2
    echo "       documents the build pipeline) or transfer from another host:" >&2
    echo "         rsync -avP workstation:Glaukopis/SFT/data/$(basename "${DATASET_FILE}") \\" >&2
    echo "               ${SFT_DIR}/data/" >&2
    exit 2
fi

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 8 ]]; then
        OFFLOAD="on"
        echo "[info] detected ${GPU_COUNT} GPU(s); auto-enabling ZeRO-3 CPU offload."
        echo "       Full SFT of 32B in bf16 needs >=8 x 80GB GPUs without offload."
        echo "       Pass --no-offload to force the on-GPU config."
    else
        OFFLOAD="off"
    fi
fi

if [[ "${OFFLOAD}" == "on" ]]; then
    DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
else
    DS_CONFIG="examples/deepspeed/ds_z3_config.json"
fi
if [[ ! -f "${SFT_DIR}/${DS_CONFIG}" ]]; then
    echo "[FAIL] deepspeed config missing: ${SFT_DIR}/${DS_CONFIG}" >&2
    exit 2
fi

# Qwen2.5-32B is ~2.3x the parameter count of 14B; activations dominate
# at cutoff_len=4096 with packing. Hold effective batch at 16 (matches
# the 14B v7 run) by deriving grad_accum from GPU_COUNT.
TARGET_EFFECTIVE_BATCH=16
BATCH_DEFAULT="1"
EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
GRAD_ACCUM_DEFAULT=$(( TARGET_EFFECTIVE_BATCH / (BATCH_DEFAULT * EFFECTIVE_GPUS) ))
if [[ "${GRAD_ACCUM_DEFAULT}" -lt 1 ]]; then
    GRAD_ACCUM_DEFAULT=1
fi
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * EFFECTIVE_GPUS ))

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 10 --save_only_model True --enable_liger_kernel True"

if [[ -n "${EXTRA_USER}" ]]; then
    EXTRA_ALL="${EXTRA_DEFAULT} ${EXTRA_USER}"
else
    EXTRA_ALL="${EXTRA_DEFAULT}"
fi

RUN_TRAIN_ARGS=(
    --model        "Qwen/Qwen2.5-32B-Instruct"
    --dataset      "${DATASET_NAME},alpaca_en_demo"
    --template     "qwen"
    --finetuning   "full"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "4096"
    --save-steps   "200"
    --eval-steps   "200"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    if [[ -z "${!var:-}" ]]; then
        unset "${var}"
    fi
done

echo "=== AthenaBench-aligned v7 (combined v5 broad coverage + v7 RMS addendum) full SFT [Qwen2.5-32B] ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset      : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective batch ~= ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  packing      : true  (cutoff_len=4096)"
echo "  eval / save  : every 200 steps"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}"
echo "  cpu offload  : ${OFFLOAD}"
echo "  liger kernel : on  (fused linear CE; required for Qwen2.5 152K vocab)"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  method       : full-parameter SFT (DeepSpeed ZeRO-3)"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
