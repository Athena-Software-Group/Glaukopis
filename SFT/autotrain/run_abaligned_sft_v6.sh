#!/bin/bash

# Launch full-parameter SFT of Llama-3.1-8B-Instruct on the v6 dataset
# (ift_data_2026_04_25_abaligned_v6, 15,658 rows: RCM 25.5% / VSP 25.5% /
# ATE 23.4% / RMS 25.5%) via LLaMA-Factory + DeepSpeed ZeRO-3.
#
# Why this script exists:
#   The v0 (base Llama-3.1-8B-Instruct) AthenaBench RMS f1 was 5.88%
#   (plausible_f1 5.97%, combined 5.93%). Diagnosis pointed at two
#   structural gaps: (1) catalog hallucination -- the model anchors on
#   M1037-style identifiers regardless of the technique; (2) cardinality
#   gap -- v3/v4/v5 trained on fixed-N=2 mitigation lists while the
#   benchmark asks for N=1..8. The v6 template slate
#   (tmpl_gen/templates/04252026/Sophia-CTI-Templates-AthenaBench-abaligned-v6.txt)
#   adds 6 RMS templates (RMS.3a/b/c variable-N, RMS.4/RMS.5 ID<->name
#   flashcards, RMS.6 negative-example discrimination) on top of the
#   existing RMS.1/RMS.2, with M10xx subscripts to filter legacy Txxxx
#   COA leakage.
#
# What stays fixed vs run_abaligned_sft_v5.sh:
#   - Base model: meta-llama/Llama-3.1-8B-Instruct
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - lr 1e-5, cutoff_len 2048
#   - DeepSpeed ZeRO-3 (no offload on >=2 GPUs)
#   - per-device batch 2, grad_accum 4 -> effective batch 16 on 2 GPUs
#   - packing on, save_only_model=True
#
# What changes vs run_abaligned_sft_v5.sh:
#   - Dataset: ift_data_2026_04_25_abaligned_v6 (15,658 rows, was 170,500
#     in v5). With packing on (cutoff 2048) this is ~700-800 optimizer
#     steps/epoch, so the run lands in ~2-3 h on a dual-H100 80GB host
#     rather than v5's ~12 h.
#   - eval_steps + save_steps = 500 (were 1500): the smaller corpus
#     means 1500-step intervals would yield only 1 intermediate
#     checkpoint; dropping to 500 gives ~4 checkpoints across the run
#     and matches the original run_abaligned_sft.sh cadence.
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v6
#
# Usage:
#   ./run_abaligned_sft_v6.sh [--repo-id USER/NAME] [--output-dir DIR]
#                             [--report-to wandb|none]
#                             [--epochs N] [--lr FLOAT]
#                             [--offload | --no-offload]
#                             [--dry-run] [--extra "..."]

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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v6"
fi

DATASET_NAME="ift_data_2026_04_25_abaligned_v6"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       This file is gitignored (~46 MB). Transfer it to this host" >&2
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

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 2 ]]; then
        OFFLOAD="on"
        echo "[info] detected ${GPU_COUNT} GPU(s); auto-enabling ZeRO-3 CPU offload."
        echo "       Pass --no-offload to force the on-GPU config (will OOM on <2 x 80GB)."
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

if [[ "${GPU_COUNT}" -ge 4 ]]; then
    BATCH_DEFAULT="4"
    GRAD_ACCUM_DEFAULT="1"
else
    BATCH_DEFAULT="2"
    GRAD_ACCUM_DEFAULT="4"
fi
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * (GPU_COUNT > 0 ? GPU_COUNT : 1) ))

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 10 --save_only_model True"

if [[ -n "${EXTRA_USER}" ]]; then
    EXTRA_ALL="${EXTRA_DEFAULT} ${EXTRA_USER}"
else
    EXTRA_ALL="${EXTRA_DEFAULT}"
fi

RUN_TRAIN_ARGS=(
    --model        "meta-llama/Llama-3.1-8B-Instruct"
    --dataset      "${DATASET_NAME},alpaca_en_demo"
    --template     "llama3"
    --finetuning   "full"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "2048"
    --save-steps   "500"
    --eval-steps   "500"
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

echo "=== AthenaBench-aligned v6 (04-25 RMS-expanded slate) full SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset file : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective batch ~= ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  packing      : true  (cutoff_len=2048)"
echo "  eval / save  : every 500 steps"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}"
echo "  cpu offload  : ${OFFLOAD}"
echo "  method       : full-parameter SFT (DeepSpeed ZeRO-3)"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
