#!/bin/bash

# v8.1 single-pass full-parameter SFT of Qwen2.5-14B-Instruct on the
# consolidated v8.1 corpus (ift_data_2026_04_30_v81.json, target ~38-42K
# rows). Replaces the v8 Phase A -> Phase B chain that produced the
# Risk Mitigation Strategy catalog-collapse regression in v8-large at 14B.
#
# Why this script exists:
#   v8-large (run_abaligned_sft_qwen25_14b_v8.sh) trained Qwen-14B in two
#   phases (broad re-anchor on the v7 corpus, then JSON+long-context
#   specialization). Diagnostic on the v8-large checkpoint showed RMS
#   catalog drills (AB.RMS.{4,5}) had been dropped from Phase B and
#   produced a single-mitigation-cluster collapse on athena-rms (top-5
#   accuracy F1=8). v8.1 sidesteps the chain entirely: a single SFT pass
#   over a consolidated mixture authored from one template file
#   (tmpl_gen/templates/04302026/Sophia-CTI-Templates-v8_1.txt) that
#   carries explicit Count: floors on the catalog drills.
#
# Corpus shape (single source):
#   - SFT/data/ift_data_2026_04_30_v81.json  (41,208 rows, stratified subsample)
#     Generated end-to-end by:
#       bash tmpl_gen/data_generation/make_dataset.sh \
#         tmpl_gen/templates/04302026/Sophia-CTI-Templates-v8_1.txt \
#         _v81_build/triples \
#         SFT/data/ift_data_2026_04_30_v81.raw.json \
#         10 1500
#       tmpl_gen/venv/bin/python tmpl_gen/scripts/stratified_subsample.py \
#         --in  SFT/data/ift_data_2026_04_30_v81.raw.json \
#         --out SFT/data/ift_data_2026_04_30_v81.json --cap 170
#       tmpl_gen/venv/bin/python tmpl_gen/scripts/dedup_against_evals.py \
#         --input SFT/data/ift_data_2026_04_30_v81.json \
#         --eval-dir SFT/eval/benchmark_data --n 13 --hit-threshold 1 \
#         --report SFT/data/ift_data_2026_04_30_v81.dedup.json
#     The cap=170 subsample preserves AB.RMS.* / JS.RMS.* in full to protect
#     RMS catalog recall (regression seen in v8-large 14B).
#   - tulu_3_sft_mixture + alpaca_en_demo mixed at load (catastrophic-
#     forgetting guard, ~7K rows/epoch)
#   - --max-samples 50000 caps total visible rows per epoch.
#
# Training shape (single phase):
#   - Base model     : Qwen/Qwen2.5-14B-Instruct
#   - Template       : qwen
#   - 2 epochs, cosine, 5% warmup, bf16
#   - lr 1e-5
#   - cutoff_len 8192, packing on
#   - effective batch 16 across visible GPUs
#   - DeepSpeed ZeRO-3 (auto offload on <4 GPUs given 14B weight footprint)
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v81
#
# Usage:
#   ./run_abaligned_sft_qwen25_14b_v81.sh [--repo-id USER/NAME]
#                                         [--output-dir DIR]
#                                         [--report-to wandb|none]
#                                         [--epochs N] [--lr FLOAT]
#                                         [--offload | --no-offload]
#                                         [--dry-run] [--extra "..."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
EXTRA_USER=""
EPOCHS="2"
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
        -h|--help) sed -n '3,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v81"
fi

DATASET_NAME="ift_data_2026_04_30_v81"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       Generate locally with:" >&2
    echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
    echo "           tmpl_gen/templates/04302026/Sophia-CTI-Templates-v8_1.txt \\" >&2
    echo "           _v81_build/triples \\" >&2
    echo "           ${DATASET_FILE} \\" >&2
    echo "           10 1500" >&2
    echo "       Or rsync from the build host:" >&2
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
    if [[ "${GPU_COUNT}" -lt 4 ]]; then
        OFFLOAD="on"
        echo "[info] detected ${GPU_COUNT} GPU(s); auto-enabling ZeRO-3 CPU offload (14B + cutoff 8192)."
    else
        OFFLOAD="off"
    fi
fi

DS_CONFIG="examples/deepspeed/ds_z3_config.json"
[[ "${OFFLOAD}" == "on" ]] && DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
BATCH_DEFAULT=1
GRAD_ACCUM_DEFAULT=$(( 16 / (BATCH_DEFAULT * EFFECTIVE_GPUS) ))
[[ ${GRAD_ACCUM_DEFAULT} -lt 1 ]] && GRAD_ACCUM_DEFAULT=1
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * EFFECTIVE_GPUS ))

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 5 --save_only_model True --enable_liger_kernel True"
EXTRA_ALL="${EXTRA_DEFAULT}${EXTRA_USER:+ ${EXTRA_USER}}"

RUN_TRAIN_ARGS=(
    --model        "Qwen/Qwen2.5-14B-Instruct"
    --dataset      "${DATASET_NAME},tulu_3_sft_mixture,alpaca_en_demo"
    --template     "qwen"
    --finetuning   "full"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "8192"
    --save-steps   "200"
    --eval-steps   "200"
    --packing      "true"
    --max-samples  "50000"
    --report-to    "${REPORT_TO}"
    --push-to-hf   "${REPO_ID}"
    --extra        "${EXTRA_ALL}"
)
[[ -n "${OUTPUT_DIR}" ]] && RUN_TRAIN_ARGS+=( --output-dir "${OUTPUT_DIR}" )
[[ ${DRY_RUN} -eq 1 ]]   && RUN_TRAIN_ARGS+=( --dry-run )

export FORCE_TORCHRUN=1
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done

echo "=== AthenaBench-aligned v8.1 (Qwen2.5-14B-Instruct, single-pass) full SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset      : ${DATASET_FILE}"
echo "  mix          : ${DATASET_NAME} + tulu_3_sft_mixture + alpaca_en_demo (cap 50000)"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  cutoff_len   : 8192  (packing on)"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}  (offload=${OFFLOAD})"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
