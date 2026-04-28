#!/bin/bash

# Launch full-parameter SFT of Qwen2.5-14B-Instruct on the consolidated v7
# dataset via LLaMA-Factory + DeepSpeed ZeRO-3. Trains on
# ift_data_2026_04_26_combined_v7 (180,533 rows: v5 broad SFT coverage +
# v7 RMS-only addendum, pre-merged into one file) plus alpaca_en_demo
# (instruction-following baseline). Total ~181.5k rows.
#
# Why this script exists (Llama-3.1-8B v7 -> Qwen2.5-14B v7 capacity test):
#   The Llama-3.1-8B v7 SFT (run_abaligned_sft_v7.sh) recovered athena-rms
#   (62.64% strict F1, +56pp over v0) but regressed CyberSOCEval relative
#   to its base model (~50-60% drop in strict accuracy on both malware and
#   threat-intel tasks). The diagnosed cause was narrow-curriculum format
#   drift: v7 compressed the model's output distribution onto the six
#   AthenaBench task envelopes at the cost of the more general "produce
#   the JSON schema the system prompt asks for" capability the base
#   Instruct model had.
#
#   This script tests whether a ~75% larger model (14B vs 8B) absorbs the
#   same v7 curriculum without the same forgetting trade-off. The hypothesis
#   is that the additional parameter capacity leaves more room to retain
#   broad instruction-following + cyber breadth alongside the AthenaBench
#   task specializations.
#
# What stays fixed vs run_abaligned_sft_v7.sh:
#   - Dataset: ift_data_2026_04_26_combined_v7 + alpaca_en_demo
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - lr 1e-5
#   - DeepSpeed ZeRO-3 (CPU offload auto-enabled on <2 GPUs)
#   - cutoff_len 4096, packing on, save_only_model=True
#   - Effective batch ~16, save/eval every 200 steps
#
# What changes vs run_abaligned_sft_v7.sh:
#   - Base model: Qwen/Qwen2.5-14B-Instruct (was meta-llama/Llama-3.1-8B-Instruct)
#   - Template: qwen (was llama3)
#   - per-device batch lowered to keep packed-activation memory bounded
#     on 80 GB GPUs at the larger parameter count:
#       <4 GPUs: batch 1, grad_accum 16 (was 1 / 8)
#       >=4 GPUs: batch 1, grad_accum 4 (was 2 / 2)
#     Effective batch stays at 16 in both branches.
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v7
#
# Usage:
#   ./run_abaligned_sft_qwen25_14b_v7.sh [--repo-id USER/NAME] [--output-dir DIR]
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
        -h|--help) sed -n '3,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v7"
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
    if [[ "${GPU_COUNT}" -lt 2 ]]; then
        OFFLOAD="on"
        echo "[info] detected ${GPU_COUNT} GPU(s); auto-enabling ZeRO-3 CPU offload."
        echo "       Pass --no-offload to force the on-GPU config (will OOM on <4 x 80GB for 14B)."
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

# Qwen2.5-14B has ~75% more parameters than Llama-3.1-8B; halve the
# per-device batch vs the v7 Llama config and double grad_accum to keep
# effective batch at 16 without OOMing 80 GB GPUs at cutoff_len=4096
# packing.
if [[ "${GPU_COUNT}" -ge 4 ]]; then
    BATCH_DEFAULT="1"
    GRAD_ACCUM_DEFAULT="4"
else
    BATCH_DEFAULT="1"
    GRAD_ACCUM_DEFAULT="16"
fi
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * (GPU_COUNT > 0 ? GPU_COUNT : 1) ))

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 10 --save_only_model True"

if [[ -n "${EXTRA_USER}" ]]; then
    EXTRA_ALL="${EXTRA_DEFAULT} ${EXTRA_USER}"
else
    EXTRA_ALL="${EXTRA_DEFAULT}"
fi

RUN_TRAIN_ARGS=(
    --model        "Qwen/Qwen2.5-14B-Instruct"
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

for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    if [[ -z "${!var:-}" ]]; then
        unset "${var}"
    fi
done

echo "=== AthenaBench-aligned v7 (combined v5 broad coverage + v7 RMS addendum) full SFT [Qwen2.5-14B] ==="
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
echo "  method       : full-parameter SFT (DeepSpeed ZeRO-3)"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
