#!/bin/bash

# v20 multi-shard Recalibrate touch-up of
# asg-ai/athena-cti-sft-qwen25-14b-v20-cse. Stage 5 of the v20 chain
# (tmpl_gen/templates/05162026/v20_plan.txt §4.5 / §1.2). Forked from
# run_sft_qwen25_14b_v19_recalibrate.sh; sole training-recipe delta vs
# v19 is the --probs default 0.33,0.33,0.34 -> 0.25,0.40,0.35 (reverts
# the equal-weight v19 experiment to the v18.2 production mix that
# delivered the strongest Stage 5 gains on the prior chain). Dataset
# names move 2026_05_15 v19 -> 2026_05_16 v20; HF base/push targets
# move v19-cse / v19-recalibrate -> v20-cse / v20-recalibrate.
#
# Why v18.2 mix on the v20 chain (see v20_plan.txt §1.2 / §2.3):
#   v19's equal-weight 0.33/0.33/0.34 experiment was justified by v18.2's
#   MCQ plateau at 62.33 (gate 70.0), but the v19 chain's regression
#   spans the AthenaBench catalog axes (ATE -9.8, RCM -8.7 vs v18.1-core)
#   rather than MCQ -- the structural problem the equal-weight mix was
#   designed to address never materialised in v19, and equal-weighting
#   actively starved Phase B of recalibrate exposure. v20 reverts to the
#   v18.2 production mix (0.25 Phase A, 0.40 Phase B, 0.35 TAA), which
#   weights Phase B highest -- exactly the recalibrate emphasis the v20
#   chain needs given v19's Phase B-axis regression.
#
# interleave_under stops at min_source_size / max(P). With v20's
# max(P) = 0.40, --max-samples 2400 yields 2400/0.40 = 6000 interleaved
# rows (~1500 optimizer steps; ~80-100 min on 4xH100, matching v18.2's
# wall-time exactly).
#
# Geometry preserved from v18.2 / Phase B (cutoff 16384, packing off):
# the catalog drill is the most fragile axis to install, so the run
# takes Phase B's long-context unpacked geometry. Phase A and TAA shards
# are short enough that they fit trivially under this cutoff.
#
# Recipe (Phase B geometry; LR matches v18.2; v18.2-mix 3-shard interleave):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-14b-v20-cse
#                       (HF; overridable via --base-model)
#   - Datasets (mix)  : ift_data_2026_05_16_v20_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.25)
#                       ift_data_2026_05_16_v20_core_b_rms_ate_vsp_rcm         (0.40)
#                       ift_data_2026_05_16_v20_taa                            (0.35)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 4   (per_device 1 x grad_accum 1 x 4 GPUs)
#   - save every 200 steps; intra-training eval DISABLED (see body comment)
#   - --max-samples 2400 (per dataset; --max-samples is per-shard in
#     LlamaFactory, applied BEFORE interleaving)
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-recalibrate
#
# Estimated wall-time on 4xH100: ~95-115 min.
#
# Full v20 chain:
#   1. ./run_sft_qwen25_14b_v20_core.sh          # broad + Phase B  -> v20-core
#   2. ./run_sft_qwen25_14b_v20_taa.sh           # TAA Classic      -> v20-taa
#   3. ./run_sft_qwen25_14b_v20_cse.sh           # CSE drill        -> v20-cse
#   4. ./run_sft_qwen25_14b_v20_recalibrate.sh   # 3-shard replay   -> v20-recalibrate
#
# Usage:
#   ./run_sft_qwen25_14b_v20_recalibrate.sh [--repo-id USER/NAME]
#                                           [--base-model HF_REPO|LOCAL_DIR]
#                                           [--output-dir DIR]
#                                           [--report-to wandb|none]
#                                           [--max-samples N]
#                                           [--lr LR]
#                                           [--probs P_A,P_B,P_TAA]
#                                           [--offload | --no-offload]
#                                           [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
# --max-samples is applied PER DATASET in LlamaFactory (each shard subsampled
# to N rows BEFORE interleaving). With interleave_under + probs P, the
# resulting interleaved dataset has size N / max(P) (the highest-weighted
# source determines when interleaving stops). Default 2400 with v20's v18.2
# mix 0.25/0.40/0.35 -> 2400/0.40 = 6000 final training samples (~1500
# optimizer steps at eff_bs 4, ~80-100 min on 4xH100; identical to v18.2's
# Stage 5 wall-time per v20_plan.txt §1.2).
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
        -h|--help) sed -n '3,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-recalibrate"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v20-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v20_recalibrate_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_16_v20_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_16_v20_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_16_v20_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"

# Intra-training eval is DISABLED for this multi-shard touch-up. LlamaFactory
# requires len(eval_dataset) == len(interleave_probs) when interleaving, and
# its loader keys datasets by name -- so listing core_val twice (the natural
# eval for Phase A and Phase B which share the unified core_val) silently
# dedupes to 2 unique entries against 3 probs, raising:
#   numpy.random.Generator.choice: a and p must have same size
# Producing three truly distinct eval shards would require a data build
# (phase_a_val / phase_b_val split out of core_val); not worth it for a
# touch-up where sign-off is via the AthenaBench/CSE/CM bench suites and
# eval loss is monitoring-only. The trainer still logs per-step train loss
# at logging_steps=5.
#
# To fully disable eval we must override BOTH --do_eval AND --eval_strategy.
# transformers.TrainingArguments.__post_init__ auto-flips do_eval back to
# True whenever eval_strategy != "no", so --do_eval False alone is silently
# undone and the parser validator at hparams/parser.py:344-347 still fires
# ("Please make sure eval_dataset be provided or val_size >1e-6"). The base
# run_train.sh sets --eval_strategy steps for normal training, so the
# override below (--eval_strategy no) is mandatory for this run.

for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v20 recalibrate dataset missing: SFT/data/${ds}.json" >&2
        echo "       These shards are reused verbatim from the v20 build" >&2
        echo "       (Phase A / Phase B / standalone TAA); rebuild via" >&2
        echo "       run_sft_qwen25_14b_v20_core.sh / _taa.sh data" >&2
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

run_v20_recalibrate() {
    echo "=== v20 recalibrate (Qwen2.5-14B): v18.2-mix 3-shard interleave from v20-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
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
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA; v18.2 production mix)"
echo "  max samples  : ${MAX_SAMPLES}/dataset -> ~6000 interleaved rows at probs=0.25/0.40/0.35 (eval disabled)"
echo "  learning rate: ${LR}  (Phase B was 5e-06; touch-up is 1/5th, same as v18.2)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v20_recalibrate
