#!/bin/bash

# v11 single-pass full-parameter SFT of Qwen2.5-14B-Instruct on the
# unified v11 corpus (ift_data_2026_05_03_v11_train.json, ~198,644 rows
# after held-out val split). Same single-pass shape as v10; v11 ships
# the SOC.* / TAA.CANON.* / RMS-paraphrase expansions and the parser-side
# anchor-fixation fix (per_primary_grouping=true global default).
#
# Naming migration (per v11_plan.txt §0): "abaligned" suffix dropped from
# script name + HF repo id + harness alias. Every corpus from v7 onward
# has been AthenaBench-aligned by design; the suffix no longer carries
# information.
#
# Why this script exists:
#   v10 (run_abaligned_sft_qwen25_14b_v10.sh) regressed on RMS/CKT/ATE/TAA
#   versus v9 (broad-tail crowding + 44-row M-control catalog +
#   single-anchor MS/TAA collapse). v11 fixes all four:
#     - AB.RMS.{4,5} paraphrase-multiplied 10x (~440 rows per family)
#     - X.* / YN.* trimmed ~30% (RCM-axis exemptions held at 1500)
#     - F3 step-by-step Cypher emitter recovers AB.MS.* / AB.TAA.*
#     - new SOC.* family (4,884 rows, MITRE D3FEND v1.4.0 ingest)
#     - new TAA.CANON.* family (778 rows, intrusion-set alias resolution)
#     - actor cap lifted 20 -> 40 (recovers TAA.* from 1,284 to 5,594)
#     - dedup held at 50 (round-1 at 30 wiped 4,800 RMS stratification rows)
#   See tmpl_gen/templates/05032026/README.md §3 for the full v10 -> v11
#   delta list (10 items).
#
# Corpus shape (single source, deterministic build):
#   - SFT/data/ift_data_2026_05_03_v11.json  (198,994 rows, 244 MB)
#     Generated end-to-end by:
#       bash tmpl_gen/data_generation/make_dataset.sh \
#         tmpl_gen/templates/05032026/Sophia-CTI-Templates-v11.txt \
#         _v11_build/triples \
#         SFT/data/ift_data_2026_05_03_v11.raw.json \
#         10 1500
#       bash _v11_build/watcher.sh   # actor-balance + dedup post-pass
#   - SFT/data/ift_data_2026_05_03_v11_train.json  (~198,644 rows)
#     SFT/data/ift_data_2026_05_03_v11_val.json    (~500 rows; 50/axis)
#     Held-out split via _v11_build/build_val_slice.py (deterministic
#     seed=42; 50 rows per AthenaBench axis: RMS, MCQ, ATE, RCM, VSP,
#     TAA, TAA-IE, TAA-NEG, MS, TAA-CANON, SOC).
#   - tulu_3_sft_mixture + alpaca_en_demo mixed at load (catastrophic-
#     forgetting guard, unchanged from v10).
#   - --max-samples 240000 (was 200000 in v10) to absorb the larger
#     v11 corpus + mixture without truncating either source.
#
# Training shape (single phase):
#   - Base model     : Qwen/Qwen2.5-14B-Instruct
#   - Template       : qwen
#   - 1 epoch (default), cosine, 5% warmup, bf16
#   - lr 1e-5
#   - cutoff_len 8192, packing on
#   - effective batch 16 across visible GPUs
#   - DeepSpeed ZeRO-3 (auto offload on <4 GPUs given 14B weight footprint)
#   - --eval_dataset wired to ift_data_2026_05_03_v11_val (per-step eval
#     loss every 500 steps so RMS/MS regressions are visible mid-train)
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-qwen25-14b-v11
#
# Usage:
#   ./run_sft_qwen25_14b_v11.sh [--repo-id USER/NAME]
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
        -h|--help) sed -n '3,68p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v11"
fi

DATASET_NAME="ift_data_2026_05_03_v11_train"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
VAL_NAME="ift_data_2026_05_03_v11_val"
VAL_FILE="${SFT_DIR}/data/${VAL_NAME}.json"
if [[ ! -f "${DATASET_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "[FAIL] training/validation dataset not found:" >&2
    [[ ! -f "${DATASET_FILE}" ]] && echo "       ${DATASET_FILE}" >&2
    [[ ! -f "${VAL_FILE}" ]]     && echo "       ${VAL_FILE}" >&2
    echo "       Generate locally with:" >&2
    echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
    echo "           tmpl_gen/templates/05032026/Sophia-CTI-Templates-v11.txt \\" >&2
    echo "           _v11_build/triples \\" >&2
    echo "           ${SFT_DIR}/data/ift_data_2026_05_03_v11.raw.json \\" >&2
    echo "           10 1500" >&2
    echo "         bash _v11_build/watcher.sh" >&2
    echo "         python _v11_build/build_val_slice.py" >&2
    echo "       Or rsync from the build host:" >&2
    echo "         rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_05_03_v11* \\" >&2
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

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 5 --save_only_model True --enable_liger_kernel True --eval_dataset ${VAL_NAME}"
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
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done

echo "=== v11 (Qwen2.5-14B-Instruct, single-pass) full SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset      : ${DATASET_FILE}"
echo "  val          : ${VAL_FILE}"
echo "  mix          : ${DATASET_NAME} + tulu_3_sft_mixture + alpaca_en_demo (cap 240000)"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  cutoff_len   : 8192  (packing on)"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}  (offload=${OFFLOAD})"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
