#!/bin/bash

# Launch full-parameter SFT of Llama-3.1-8B-Instruct on the v5 dataset
# (ift_data_2026_04_24_v5, 170,500 rows = 138,343 04-22 abaligned core +
# 32,157 04-24 detection/exploit/PoC addendum) via LLaMA-Factory +
# DeepSpeed ZeRO-3. Throughput-optimised successor to
# run_abaligned_sft.sh, intended to land in ~12 h on a dual-H100 80GB
# host.
#
# Why this script exists (and why it's full-param, not LoRA):
#   The 04-22 LoRA replication (run_abaligned_sft_lora.sh) was
#   benchmarked at 42.1 combined on AthenaBench v1, vs the original
#   full-param 'abaligned' baseline at 52.0 on the same data -- a 9.9
#   point regression attributable entirely to LoRA r=16 being too low-
#   capacity to recover open-generation tasks (VSP -20.6, TAA -16.0,
#   ATE -14.0, RCM -7.3, RMS -6.9). MCQ (CKT) actually improved (+5.6).
#   To validate the v5 dataset on a fair footing, this run uses the
#   same full-param recipe as run_abaligned_sft.sh and only changes the
#   dataset + adds sequence packing.
#
# What stays fixed vs run_abaligned_sft.sh:
#   - Base model: meta-llama/Llama-3.1-8B-Instruct
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - lr 1e-5, cutoff_len 2048
#   - DeepSpeed ZeRO-3 (no offload on >=2 GPUs)
#   - per-device batch 2, grad_accum 4 -> effective batch 16 on 2 GPUs
#     (kept identical to the original abaligned run so the optimizer
#     trajectory stays comparable; packing changes wall time, not the
#     gradient noise scale)
#   - save_only_model=True (avoids end-of-train OOM under ZeRO-3 by
#     dropping fp32 optimizer state from checkpoints; final HF push
#     uses the last checkpoint rather than best-eval-loss because
#     transformers >=4.55 forbids the
#     DeepSpeed + save_only_model + load_best_model_at_end triple)
#
# What changes vs run_abaligned_sft.sh:
#   - Dataset: ift_data_2026_04_24_v5 (170,500 rows, was 138,343 in v3)
#   - Sequence packing on (was off): packs short Alpaca rows up to
#     cutoff_len, eliminating padding waste and cutting optimizer steps
#     by roughly 2-3x for this corpus.
#   - eval_steps + save_steps = 1500 (were 500/500): full val pass every
#     500 steps was the second-largest wall-clock contributor; tripling
#     the interval reclaims ~1-2 h without losing the eval-loss signal.
#     Save and eval intervals are kept equal so each eval is paired with
#     a recoverable checkpoint (~16 GB / 25-30 min instead of ~16 GB /
#     8-10 min, halving disk burn).
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v5
#
# Usage:
#   ./run_abaligned_sft_v5.sh [--repo-id USER/NAME] [--output-dir DIR]
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
        -h|--help) sed -n '3,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v5"
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

# Auto-pick ZeRO-3 config: full-param 8B + AdamW fp32 states needs ~96 GB
# of GPU RAM before activations, so on <2 GPUs we must offload optimizer +
# params to CPU. Mirrors the logic in run_abaligned_sft.sh.
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

# Mirror run_abaligned_sft.sh batch sizing so the optimizer trajectory is
# identical to the original 'abaligned' baseline (effective batch 16):
#   2 GPUs: batch=2, grad_accum=4 -> 2 * 4 * 2 = 16
#   4 GPUs: batch=4, grad_accum=1 -> 4 * 1 * 4 = 16
# Packing changes wall time (fewer optimizer steps per epoch), not the
# gradient noise scale, so this stays comparable to the 52.0 baseline.
if [[ "${GPU_COUNT}" -ge 4 ]]; then
    BATCH_DEFAULT="4"
    GRAD_ACCUM_DEFAULT="1"
else
    BATCH_DEFAULT="2"
    GRAD_ACCUM_DEFAULT="4"
fi
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * (GPU_COUNT > 0 ? GPU_COUNT : 1) ))

# Newer transformers (>=4.55) explicitly forbids the combination
# DeepSpeed + save_only_model=True + load_best_model_at_end=True at
# Trainer construction time. We have to drop one. Dropping
# save_only_model=True would reintroduce the end-of-train OOM
# (Trainer._load_best_model reloads ~32 GB / rank of fp32 optimizer
# state on top of the still-resident training state under ZeRO-3),
# so instead drop load_best_model_at_end. The HF push then takes
# whatever checkpoint is in output_dir at end-of-training, which is
# the last step. For a 3-epoch SFT with a cosine schedule, the final
# step's eval_loss is within noise of the best step's, so this is
# the right trade-off vs the OOM.
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

echo "=== AthenaBench-aligned v5 (04-22 core + 04-24 addendum) full SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset file : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective batch ~= ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  packing      : true  (cutoff_len=2048)"
echo "  eval / save  : every 1500 steps  (paired so each eval has a recoverable checkpoint)"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}"
echo "  cpu offload  : ${OFFLOAD}"
echo "  method       : full-parameter SFT (DeepSpeed ZeRO-3)"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
