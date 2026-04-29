#!/bin/bash

# v8-small: single-pass full-parameter SFT of Llama-3.1-8B-Instruct on a
# 60K stratified mix derived from v7/v8 corpora. Pairs with the 32B
# launcher run_abaligned_sft_qwen25_32b_v8.sh but uses a deliberately
# lighter recipe to avoid catastrophic forgetting on the 8B parameter
# count.
#
# Why "v8-small" exists:
#   The 32B v8 plan trains on ~1.18M rows across two phases. Replaying
#   the same volume on Llama-3.1-8B-Instruct would saturate the
#   8B parameter count: prior 8B runs (v3/v4) on 138K-180K rows for 3
#   epochs lifted AthenaBench but regressed CyberMetric and general
#   instruction-following. v8-small targets ~60K rows over 2 epochs at
#   cutoff 8192 to land the v8 format-strict / long-context signal
#   without overwriting Llama's instruction prior.
#
# Corpus shape (52,557 rows in the combined file + ~7.5K mixed at load):
#   ift_data_2026_04_29_combined_v8small.json :
#     - 41,857 rows : ift_data_2026_04_26_combined_v7 stratified to
#                     <=250 per shortname (185 shortnames in v7 corpus).
#     -  8,700 rows : full ift_data_2026_04_29_json_v8 (format-strict
#                     JSON-output MCQ/RCM/RMS/TAA/VSP/ATE).
#     -  2,000 rows : random subsample of ift_data_2026_04_29_longctx_v8
#                     (mean ~11K tokens; the 8K cutoff truncates the
#                     long tail, by design -- the model still sees the
#                     16K-shaped distribution within the 8K window).
#   tulu_3_sft_mixture (HF, capped via --max-samples)         5,000 rows
#   alpaca_en_demo                                            ~2,500 rows
#   ----------------------------------------------------------------
#   total seen per epoch (capped)                             ~60,000 rows
#
# Training shape (single phase):
#   - Base model     : meta-llama/Llama-3.1-8B-Instruct
#   - 2 epochs, cosine, 5% warmup, bf16
#   - lr 1e-5 (matches v3/v4)
#   - cutoff_len 8192 (long-context rows fit; cutoff 16384 would 4x
#     activation memory for negligible payload coverage)
#   - packing on (8B at cutoff 8192 is throughput-bound, not memory-bound)
#   - effective batch 16 across whatever GPUs are visible
#   - DeepSpeed ZeRO-3 (no offload on >=2 x 80GB GPUs)
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v8
#
# Usage:
#   ./run_abaligned_sft_llama31_8b_v8.sh [--repo-id USER/NAME]
#                                        [--output-dir DIR]
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
        -h|--help) sed -n '3,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v8"
fi

DATASET_NAME="ift_data_2026_04_29_combined_v8small"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       Generate locally with:" >&2
    echo "         python tmpl_gen/scripts/subsample_stratified.py \\" >&2
    echo "           --source SFT/data/ift_data_2026_04_26_combined_v7.json:250:strat \\" >&2
    echo "           --source SFT/data/ift_data_2026_04_29_json_v8.json:0 \\" >&2
    echo "           --source SFT/data/ift_data_2026_04_29_longctx_v8.json:2000:random \\" >&2
    echo "           --output ${DATASET_FILE}" >&2
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
    if [[ "${GPU_COUNT}" -lt 2 ]]; then
        OFFLOAD="on"
        echo "[info] detected ${GPU_COUNT} GPU(s); auto-enabling ZeRO-3 CPU offload."
    else
        OFFLOAD="off"
    fi
fi

DS_CONFIG="examples/deepspeed/ds_z3_config.json"
[[ "${OFFLOAD}" == "on" ]] && DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"

# At cutoff 8192 with packing on, 8B activation memory fits per-device
# batch 1 on 80GB; grad_accum lifts effective batch to 16.
EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
BATCH_DEFAULT=1
GRAD_ACCUM_DEFAULT=$(( 16 / (BATCH_DEFAULT * EFFECTIVE_GPUS) ))
[[ ${GRAD_ACCUM_DEFAULT} -lt 1 ]] && GRAD_ACCUM_DEFAULT=1
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * EFFECTIVE_GPUS ))

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 5 --save_only_model True --enable_liger_kernel True"
EXTRA_ALL="${EXTRA_DEFAULT}${EXTRA_USER:+ ${EXTRA_USER}}"

RUN_TRAIN_ARGS=(
    --model        "meta-llama/Llama-3.1-8B-Instruct"
    --dataset      "${DATASET_NAME},tulu_3_sft_mixture,alpaca_en_demo"
    --template     "llama3"
    --finetuning   "full"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "8192"
    --save-steps   "200"
    --eval-steps   "200"
    --packing      "true"
    --max-samples  "60000"
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

echo "=== AthenaBench-aligned v8-small (Llama-3.1-8B, 60K stratified mix) full SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset      : ${DATASET_FILE}"
echo "  mix          : combined_v8small + tulu_3_sft_mixture + alpaca_en_demo (cap 60000)"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  cutoff_len   : 8192  (packing on)"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}  (offload=${OFFLOAD})"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
