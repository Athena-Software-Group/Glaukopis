#!/bin/bash

# v21 single-phase narrow-drilling SFT of asg-ai/athena-cti-sft-qwen25-32b-v21-taa
# on the v21 CyberSOCEval-letter-set shard (ift_data_2026_05_18_v21_cse). Stage 3
# (final) of the v21 chain ported to Qwen2.5-32B-Instruct (tmpl_gen/templates/
# 05182026/README-21.md §"Scale-up to Qwen2.5-32B-Instruct"); the resulting
# checkpoint is the published 32B v21 model. The CSE shard is byte-identical
# to the 14B v21 build (template-baked, architecture-independent); only
# the base-model pointer changes (now picks up the 32B v21+TAA checkpoint).
#
# Recipe parity with run_sft_qwen25_14b_v21_final.sh:
#   - Identical cutoff, packing, lr, effective batch, max-samples, and
#     save/eval steps.
#   - 32B deltas (memory only, no recipe change):
#       * --optim adamw_8bit added to EXTRA flags (v8/v11 32B precedent;
#         mandatory at 32B ZeRO-3 no-offload).
#       * --gc default flipped from "auto" to "on" (gradient
#         checkpointing always enabled at 32B).
#
# Recipe (mirrors 14B v21 final / v18 / v16+v17.1-CSE):
#   - Base model    : asg-ai/athena-cti-sft-qwen25-32b-v21-taa
#                     (HF; overridable via --base-model)
#   - Dataset       : ift_data_2026_05_18_v21_cse  (~14-19K rows; CSE shape)
#   - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#   - Effective batch 16   (per_device 1 x grad_accum 2 x 8 GPUs)
#   - eval/save every 100 steps
#   - --max-samples 33000  (matches v16 / v17 / v17.1)
#   - Gradient checkpointing ON (32B default; see header above)
#   - adamw_8bit optimizer (v8/v11 32B precedent)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-cse
#     (the published 32B v21 chain-ship checkpoint; pass
#      --repo-id asg-ai/athena-cti-sft-qwen25-32b-v21 to drop the suffix)
#
# Estimated wall-time (32B is ~2.3x heavier than 14B; corpus matches v17.1):
#   8xH100 80GB SXM        : ~9-13 h.
#   8xRTX PRO 6000 96GB    : ~13-18 h. ~1.3-1.5x PCIe Gen5 vs NVLink
#     overhead for ZeRO-3 collectives at 32B.
#   4xH100 80GB SXM        : NOT RECOMMENDED -- 32B ZeRO-3 weight shard
#     doubles to ~16 GB/rank; activation footprint at cutoff=4096
#     packing=on tips over the 80 GB budget even with GC on. Use 8x.
#
# Full v21 chain (run sequentially after each push completes on HF):
#   1. ./run_sft_qwen25_32b_v21_core.sh        # broad + axis  -> v21-core
#   2. ./run_sft_qwen25_32b_v21_plus_taa.sh    # TAA Classic   -> v21-taa
#   3. ./run_sft_qwen25_32b_v21_final.sh       # CSE drill     -> v21-cse (final)
#
# Estimated total wall-clock (Core+TAA+CSE on 8xH100 80GB SXM):
#   ~48-60 h sequential (Core ~26-30 h, TAA ~13-17 h, CSE ~9-13 h).
# Add Recalibrate (~3-5 h on 8xH100 SXM) via run_sft_qwen25_32b_v21_recalibrate.sh
# if v21-cse sign-off shows VSP erosion (the 14B v21 chain trade-off).
#
# Usage:
#   ./run_sft_qwen25_32b_v21_final.sh [--repo-id USER/NAME]
#                                       [--base-model HF_REPO|LOCAL_DIR]
#                                       [--output-dir DIR]
#                                       [--report-to wandb|none]
#                                       [--offload | --no-offload]
#                                       [--skip-eval] [--resume]
#                                       [--dry-run]
#
# --skip-eval / --resume mirror the v21_core launcher; see that script's
# header for the in-training eval OOM rationale.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
DRY_RUN=0
OFFLOAD="auto"
SKIP_EVAL=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --base-model)   BASE_MODEL="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        --skip-eval)    SKIP_EVAL=1;       shift ;;
        --resume)       RESUME=1;          shift ;;
        -h|--help) sed -n '3,53p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-cse"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-32b-v21-taa"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-32B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_cse_${TIMESTAMP}"

DATASET="ift_data_2026_05_18_v21_cse"
VAL_NAME="ift_data_2026_05_18_v21_cse_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21-CSE dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build the v21 CSE shard from the byte-identical v17.1/v18.1 template." >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_cse.txt \\" >&2
        echo "           _v21_cse_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_18_v21_cse.raw.json \\" >&2
        echo "           10 3500" >&2
        echo "         echo \"PID=\$!\" > _v21_cse_build/build.pid" >&2
        echo "         nohup bash _v21_cse_build/watcher.sh > _v21_cse_build/watcher.log 2>&1 &" >&2
        exit 2
    fi
done

if [[ -n "${GPU_COUNT_OVERRIDE:-}" ]]; then
    GPU_COUNT="${GPU_COUNT_OVERRIDE}"
    GPU_PROBE_SOURCE="override"
else
    GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
    GPU_PROBE_SOURCE="torch"
    if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" == "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        NVIDIA_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ -n "${NVIDIA_COUNT}" && "${NVIDIA_COUNT}" != "0" ]]; then
            GPU_COUNT="${NVIDIA_COUNT}"
            GPU_PROBE_SOURCE="nvidia-smi (torch probe returned 0)"
        fi
    fi
    GPU_COUNT="${GPU_COUNT:-0}"
fi
if [[ "${GPU_COUNT}" == "0" && ${DRY_RUN} -eq 0 ]]; then
    echo "[FAIL] GPU probe returned 0 (source: ${GPU_PROBE_SOURCE}). See run_sft_qwen25_32b_v21_core.sh for diagnostic steps." >&2
    echo "       Override: GPU_COUNT_OVERRIDE=N ./run_sft_qwen25_32b_v21_final.sh ..." >&2
    exit 3
fi

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 4 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))

D_BATCH=1; D_GA=$(( 16 / (D_BATCH * EFFECTIVE_GPUS) )); [[ ${D_GA} -lt 1 ]] && D_GA=1

# 32B keeps gradient checkpointing ON unconditionally (no 14B-style
# disable-on-8x branch). 20-25% throughput tax is the right trade-off
# at 32B vs a step-N OOM that would burn the chain mid-point.
GC_FLAG=""

# --optim adamw_8bit carries v8 32B / v11 32B precedent; required to fit
# 32B ZeRO-3 no-offload at the 80 GB/rank budget.
EXTRA_BASE="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit ${GC_FLAG}"
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    EXTRA_COMMON="${EXTRA_BASE} --eval_strategy no"
else
    EXTRA_COMMON="${EXTRA_BASE} --per_device_eval_batch_size 1 --eval_dataset ${VAL_NAME} --val_size 0"
fi

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] expected 8 GPUs (8xH100); detected ${GPU_COUNT}. Recipe was sized for 8x; effective batch will reflect detected count: eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )
RESUME_FLAG=(); [[ ${RESUME} -eq 1 ]] && RESUME_FLAG=( --resume )

run_v21() {
    echo "=== v21 (Qwen2.5-32B): CyberSOCEval-letter-set narrow drill from v21-taa (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe; CHAINED] ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASET}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${D_BATCH} --grad-accum ${D_GA} \
        --cutoff 4096 --save-steps 100 --eval-steps 100 --packing true \
        --max-samples 33000 --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${D_BATCH} grad_accum=${D_GA} -> eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS )) (target 16)"
echo "  base model   : ${BASE_MODEL}"
echo "  dataset      : ${DATASET}  (eval: ${VAL_NAME})"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  skip-eval    : $([[ ${SKIP_EVAL} -eq 1 ]] && echo on || echo off)"
echo "  resume       : $([[ ${RESUME} -eq 1 ]] && echo on || echo off)"
echo

run_v21
