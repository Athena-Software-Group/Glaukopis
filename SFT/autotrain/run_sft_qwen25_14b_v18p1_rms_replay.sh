#!/bin/bash

# v18.1+RMS-replay touch-up of asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse on
# the Phase B catalog shard (ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm)
# at low LR. Stage 4 of the v18.1 chain -- replays the original RMS/ATE/VSP/RCM
# installation shard at 1/5th the Phase B learning rate to reverse the
# CSE-stage erosion observed on AthenaBench.
#
# Why a low-LR Phase B replay (and not a full Phase B re-run):
#   The v18.1-cse benchmark on 2026-05-13 isolated the regression to the
#   final stage on a single axis: RMS combined_f1 dropped 11.6 pp between
#   v18.1-taa (57.91) and v18.1-cse (46.34), while TAA stage left it at
#   +0.22. ATE and VSP also lose ~7 pp end-to-end. The Phase B shard is
#   the only place these axes were ever installed (TAA / CSE shards do
#   not touch them), so a low-LR replay of that exact shard is the
#   minimum-perturbation fix: nudges Phase-B circuits back toward their
#   post-Phase-B optimum without re-installing them at the cost of the
#   CSE letter-pattern circuits.
#
# Recipe (Phase B geometry; LR reduced 5x; row cap reduced ~14x):
#   - Base model    : asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse
#                     (HF; overridable via --base-model)
#   - Dataset       : ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm
#                     (same shard used at Core Phase B; ~70K rows total,
#                      capped here at 5000 for a touch-up)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 4   (per_device 1 x grad_accum 1 x 4 GPUs)
#   - eval/save every 200 steps
#   - --max-samples 5000  (~1250 optimizer steps; ~80-100 min on 4xH100)
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-1-cse-rms
#
# Estimated wall-time on 4xH100: ~1.5 h.
#
# Full v18.1 chain with replay touch-up:
#   1. ./run_sft_qwen25_14b_v18p1_core.sh         # broad + Phase B  -> v18-1-core
#   2. ./run_sft_qwen25_14b_v18p1_plus_taa.sh     # TAA Classic      -> v18-1-taa
#   3. ./run_sft_qwen25_14b_v18p1_final.sh        # CSE drill        -> v18-1-cse
#   4. ./run_sft_qwen25_14b_v18p1_rms_replay.sh   # Phase B replay   -> v18-1-cse-rms
#
# Usage:
#   ./run_sft_qwen25_14b_v18p1_rms_replay.sh [--repo-id USER/NAME]
#                                            [--base-model HF_REPO|LOCAL_DIR]
#                                            [--output-dir DIR]
#                                            [--report-to wandb|none]
#                                            [--max-samples N]
#                                            [--lr LR]
#                                            [--offload | --no-offload]
#                                            [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
MAX_SAMPLES=5000
LR="1e-06"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --base-model)   BASE_MODEL="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --max-samples)  MAX_SAMPLES="$2";  shift 2 ;;
        --lr)           LR="$2";           shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        -h|--help) sed -n '3,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-1-cse-rms"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18p1_rms_replay_${TIMESTAMP}"

DATASET="ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm"
VAL_NAME="ift_data_2026_05_11_v18p1_core_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v18.1 RMS-replay dataset missing: SFT/data/${ds}.json" >&2
        echo "       This shard is the same one used by Core Phase B; rebuild via" >&2
        echo "       run_sft_qwen25_14b_v18p1_core.sh's data preflight or copy from" >&2
        echo "       the Core training host." >&2
        exit 2
    fi
done

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

# Phase B geometry (cutoff 16384, packing off) is memory-heavy: default to
# offload ON for any non-8x configuration to avoid OOM at this cutoff.
# Phase B itself was sized for 8xH100 + offload OFF; on 4xH100 we keep
# offload ON unless explicitly disabled.
if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 8 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# Target eff_bs = 4 (half of Phase B's 8 -- gentler updates for a touch-up).
R_BATCH=1; R_GA=$(( 4 / (R_BATCH * EFFECTIVE_GPUS) )); [[ ${R_GA} -lt 1 ]] && R_GA=1

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --eval_dataset ${VAL_NAME} --val_size 0"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 4 ]]; then
    echo "[warn] expected 4 GPUs (4xH100); detected ${GPU_COUNT}. Recipe was sized for 4x; effective batch will reflect detected count: eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_v18p1_rms_replay() {
    echo "=== v18.1+RMS-replay (Qwen2.5-14B): Phase B replay from v18.1-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASET}" --template qwen --finetuning full \
        --epochs 1 --lr "${LR}" --batch ${R_BATCH} --grad-accum ${R_GA} \
        --cutoff 16384 --save-steps 200 --eval-steps 200 --packing false \
        --max-samples "${MAX_SAMPLES}" --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${R_BATCH} grad_accum=${R_GA} -> eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (target 4)"
echo "  base model   : ${BASE_MODEL}"
echo "  dataset      : ${DATASET}  (eval: ${VAL_NAME})  max_samples=${MAX_SAMPLES}"
echo "  learning rate: ${LR}  (Phase B was 5e-06; touch-up is 1/5th)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v18p1_rms_replay
