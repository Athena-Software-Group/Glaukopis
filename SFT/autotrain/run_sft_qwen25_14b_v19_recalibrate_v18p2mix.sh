#!/bin/bash

# v19 multi-shard Recalibrate touch-up of asg-ai/athena-cti-sft-qwen25-14b-v19-cse
# using the v18.2 prob mix (0.25,0.40,0.35). Stage 5 sibling of
# run_sft_qwen25_14b_v19_recalibrate.sh; the SOLE training-recipe delta vs
# the equal-weight v19 recalibrate is --probs 0.33,0.33,0.34 -> 0.25,0.40,0.35
# and the corresponding push target v19-recalibrate -> v19-recalibrate-v18p2mix.
#
# Purpose (isolation experiment): the v19 equal-weight Stage 5 (v19-recalibrate)
# regressed v18.2 on the §5.4 headline gate, most sharply on CSE-TI (-9.2 pp)
# and ATE (-6.2 pp), while winning RMS (+4.2 pp) and CKT (+6.0 pp). The two
# pipelines differ on two knobs simultaneously: (1) base checkpoint
# (v18-1-cse vs v19-cse) and (2) Stage 5 prob mix (0.25/0.40/0.35 vs
# 0.33/0.33/0.34). This launcher pins (2) to the v18.2 value so the bench
# delta vs v19-recalibrate isolates the prob-mix contribution; the delta vs
# v18-2 isolates the base-checkpoint contribution.
#
# Step count matches v18.2 exactly: --max-samples 2400 with probs P where
# max(P) = 0.40 -> 2400/0.40 = 6000 interleaved rows -> ~1500 optimizer
# steps at eff_bs 4 (byte-identical to v18.2's wallclock; ~17% fewer steps
# than the equal-weight v19-recalibrate's ~1765).
#
# Geometry preserved from v19-recalibrate / v18.2 / Phase B (cutoff 16384,
# packing off): catalog drill is the most fragile axis to install.
#
# Recipe (Phase B geometry; LR matches v18.2; v18.2 prob mix on v19 chain):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-14b-v19-cse
#                       (HF; overridable via --base-model)
#   - Datasets (mix)  : ift_data_2026_05_15_v19_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.25)
#                       ift_data_2026_05_15_v19_core_b_rms_ate_vsp_rcm         (0.40)
#                       ift_data_2026_05_15_v19_taa                            (0.35)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 4   (per_device 1 x grad_accum 1 x 4 GPUs)
#   - save every 200 steps; intra-training eval DISABLED (see body comment)
#   - --max-samples 2400 (per dataset; --max-samples is per-shard in
#     LlamaFactory, applied BEFORE interleaving)
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v19-recalibrate-v18p2mix
#
# Estimated wall-time on 4xH100: ~80-100 min (matches v18.2 envelope).
#
# Usage:
#   ./run_sft_qwen25_14b_v19_recalibrate_v18p2mix.sh [--repo-id USER/NAME]
#                                                    [--base-model HF_REPO|LOCAL_DIR]
#                                                    [--output-dir DIR]
#                                                    [--report-to wandb|none]
#                                                    [--max-samples N]
#                                                    [--lr LR]
#                                                    [--probs P_A,P_B,P_TAA]
#                                                    [--offload | --no-offload]
#                                                    [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
# --max-samples is applied PER DATASET in LlamaFactory (each shard subsampled
# to N rows BEFORE interleaving). With interleave_under + probs P, the
# resulting interleaved dataset has size N / max(P). Default 2400 with v18.2's
# probs 0.25/0.40/0.35 -> 2400/0.40 = 6000 final training samples (~1500
# optimizer steps at eff_bs 4, ~80-100 min on 4xH100; byte-identical step
# count to v18.2 so the bench delta isolates the v19-cse base contribution).
MAX_SAMPLES=2400
LR="1e-06"
PROBS="0.25,0.40,0.35"
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
        --probs)        PROBS="$2";        shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        -h|--help) sed -n '3,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v19-recalibrate-v18p2mix"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v19-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v19_recalibrate_v18p2mix_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_15_v19_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_15_v19_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_15_v19_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"

# Intra-training eval is DISABLED for this multi-shard touch-up; rationale
# identical to v19-recalibrate / v18.2 (LlamaFactory's loader keys datasets
# by name so the 3:3 shard:prob alignment requirement cannot be satisfied
# with the available val shards without a fresh data build). To fully
# disable eval we override BOTH --do_eval AND --eval_strategy because
# transformers' TrainingArguments __post_init__ auto-flips do_eval=True
# whenever eval_strategy != "no". See run_sft_qwen25_14b_v19_recalibrate.sh
# for the long-form derivation.

for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v19 recalibrate dataset missing: SFT/data/${ds}.json" >&2
        echo "       These shards are reused verbatim from the v19 build" >&2
        echo "       (Phase A / Phase B / standalone TAA); rebuild via" >&2
        echo "       run_sft_qwen25_14b_v19_core.sh / _taa.sh data" >&2
        echo "       preflights or copy from the Core training host." >&2
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
if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 8 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
R_BATCH=1; R_GA=$(( 4 / (R_BATCH * EFFECTIVE_GPUS) )); [[ ${R_GA} -lt 1 ]] && R_GA=1

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --do_eval False --eval_strategy no --val_size 0 --mix_strategy interleave_under --interleave_probs ${PROBS}"

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

run_v19_recalibrate_v18p2mix() {
    echo "=== v19 recalibrate v18p2mix (Qwen2.5-14B): v18.2 prob-mix 3-shard interleave from v19-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr "${LR}" --batch ${R_BATCH} --grad-accum ${R_GA} \
        --cutoff 16384 --save-steps 200 --eval-steps 200 --packing false \
        --max-samples "${MAX_SAMPLES}" --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${R_BATCH} grad_accum=${R_GA} -> eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (target 4)"
echo "  base model   : ${BASE_MODEL}"
echo "  datasets     : ${DATASETS}"
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA; v18.2 mix on v19 chain)"
echo "  max samples  : ${MAX_SAMPLES}/dataset -> ~6000 interleaved rows at probs=0.25/0.40/0.35 (eval disabled)"
echo "  learning rate: ${LR}  (matches v18.2; v19 Phase B was 5e-06; touch-up is 1/5th)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v19_recalibrate_v18p2mix
