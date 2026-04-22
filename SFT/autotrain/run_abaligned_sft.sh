#!/bin/bash

# Launch full-parameter SFT of Llama-3.1-8B-Instruct on the AthenaBench-aligned
# 2026-04-22 ift dataset via LLaMA-Factory + DeepSpeed ZeRO-3. Replaces the
# retired autotrain-advanced pipeline (autotrain-advanced is unmaintained and
# pins transformers==4.48.0, which conflicts with LLaMA-Factory's >=4.55.0).
#
# Runs in the unified `llm-sft` conda env created by SFT/utils/setup.sh. No
# second env, no autotrain CLI, no HF dataset-repo round-trip -- the trainer
# reads SFT/data/ift_data_2026_04_22.json directly via dataset_info.json.
#
# On success the merged full-weight model is pushed to
#   hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned
#
# Hyperparameters mirror the retired autotrain_llama3_8b_sft_fast_abaligned.yml
# so the run is apples-to-apples comparable with the prior baselines:
#   epochs=3, lr=1e-5 cosine, warmup=0.05, bf16, per-GPU batch=2, grad_accum=4
#   (effective batch = 2 * 4 * num_gpus -> 16 on a 2xH100 box).
#
# Usage:
#   ./run_abaligned_sft.sh [--repo-id USER/NAME] [--output-dir DIR]
#                          [--report-to wandb|none] [--offload | --no-offload]
#                          [--dry-run] [--extra "..."]
#
# Defaults:
#   --repo-id     ${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned
#                 (HF_USERNAME is read from SFT/.env or the caller's environment)
#   --report-to   wandb    (set to 'none' to skip wandb)
#   --offload     auto     (enabled on hosts with < 2 GPUs; disabled otherwise)
#                          Offload moves optimizer + params to CPU RAM so full
#                          SFT of an 8B model fits on 1 x 80 GB. Costs ~30-50%
#                          throughput from PCIe traffic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
EXTRA_USER=""
DRY_RUN=0
OFFLOAD="auto"    # auto | on | off

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)    REPO_ID="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        --report-to)  REPORT_TO="$2";   shift 2 ;;
        --extra)      EXTRA_USER="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        --offload)    OFFLOAD="on";     shift ;;
        --no-offload) OFFLOAD="off";    shift ;;
        -h|--help) sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Load HF credentials from SFT/.env or SFT/autotrain/.env without clobbering
# anything already exported. Matches the resolution order used by
# upload_to_hf.py (SFT/.env > SFT/.env.local > repo-root/.env > ...).
for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        set -a; source "${env_file}"; set +a
    fi
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned"
fi

DATASET_FILE="${SFT_DIR}/data/ift_data_2026_04_22.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       This file is gitignored (144MB, exceeds GitHub's push limit)." >&2
    echo "       Transfer it to this host before running, e.g.:" >&2
    echo "         rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_04_22.json \\" >&2
    echo "               ${SFT_DIR}/data/" >&2
    exit 2
fi

# Full-parameter SFT of an 8B bf16 model + AdamW states (fp32 m, v)
# needs ~96 GB of GPU RAM before activations, so ZeRO-3 without CPU
# offload requires at least 2 ranks to shard across. Auto-detect GPU
# count; if < 2, switch to ds_z3_offload_config.json which offloads
# optimizer + params to CPU (GPU residency drops to ~30-40 GB at the
# cost of ~30-50% throughput from PCIe traffic). The user can force
# either mode with --offload / --no-offload.
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

# --include_num_input_tokens_seen is already set by run_train.sh.
# save_total_limit=10 keeps the 10 most recent checkpoints (plus the
# best-eval one, which the HF Trainer preserves regardless of the limit
# when load_best_model_at_end=True). At ~48 GB/checkpoint this caps disk
# use around 480 GB, still comfortably under the 725 GB free on /home.
# load_best_model_at_end + metric_for_best_model=eval_loss makes the final
# OUTPUT_DIR (which is what gets pushed to HF) contain the minimum-eval
# checkpoint, not whatever the last step happened to produce.
EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 10 --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False"
if [[ -n "${EXTRA_USER}" ]]; then
    EXTRA_ALL="${EXTRA_DEFAULT} ${EXTRA_USER}"
else
    EXTRA_ALL="${EXTRA_DEFAULT}"
fi

RUN_TRAIN_ARGS=(
    --model        "meta-llama/Llama-3.1-8B-Instruct"
    --dataset      "ift_data_2026_04_22,alpaca_en_demo"
    --template     "llama3"
    --finetuning   "full"
    --epochs       "3"
    --lr           "1e-05"
    --batch        "2"
    --grad-accum   "4"
    --cutoff       "2048"
    --save-steps   "500"
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

# LLaMA-Factory refuses to start DeepSpeed training unless it is already
# running under torch.distributed. Its launcher only auto-sets
# FORCE_TORCHRUN when it detects >1 CUDA device *before* the DeepSpeed
# config is parsed, which is flaky (nvidia-smi ordering, CUDA_VISIBLE_DEVICES,
# container masking). Since this launcher is DeepSpeed-only, always
# force torchrun. Safe on single-GPU too (ZeRO-3 on 1 GPU is valid, just
# wasteful).
export FORCE_TORCHRUN=1

# Sanitize torchrun env vars: some containers (Docker ENV, k8s pod specs)
# export these as empty strings rather than leaving them unset, which
# defeats LLaMA-Factory's `os.getenv(VAR, default)` fallback and crashes
# its launcher with `invalid literal for int() with base 10: ''` on
# `int(nnodes)`. Unset them here so the defaults ("1", "0", auto-detect,
# 127.0.0.1, random port) actually take effect.
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    if [[ -z "${!var:-}" ]]; then
        unset "${var}"
    fi
done

echo "=== AthenaBench-aligned full SFT (LLaMA-Factory + DeepSpeed ZeRO-3) ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset file : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}"
echo "  cpu offload  : ${OFFLOAD}"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
