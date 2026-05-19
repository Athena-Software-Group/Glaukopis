#!/bin/bash

# v21 multi-shard Recalibrate touch-up of
# asg-ai/athena-cti-sft-qwen25-14b-v21-cse. Off-plan extension of the v21
# chain (v21_plan.txt §3 defines only Core/TAA/CSE; this launcher mirrors
# the v19/v20 Recalibrate stage that re-exposes the chained checkpoint to
# the Core Phase A/B + TAA shards under Phase B long-context geometry so a
# single ship candidate is produced from the v21 chain). Forked from
# run_sft_qwen25_14b_v20_recalibrate.sh; sole training-recipe deltas vs
# v20 are dataset names (2026_05_16 v20 -> 2026_05_18 v21) and HF
# base/push targets (v20-cse / v20-recalibrate -> v21-cse / v21-recalibrate).
# --probs default 0.25,0.40,0.35 (v18.2 production mix) is preserved.
#
# Why a Recalibrate touch-up on the v21 chain (see v20_plan.txt §1.2 /
# §2.3 -- the rationale carries to v21 unchanged):
#   The v21 chain re-runs v18.1's Core->TAA->CSE topology to test data-
#   build reproducibility; the published v18.1 model is the chained CSE
#   checkpoint. v18.1 itself did NOT ship a Recalibrate stage, so v21
#   Recalibrate is OPTIONAL and only relevant if the v21-cse sign-off
#   exposes the same Phase B / catalog erosion v20-cse showed against
#   v18.2. Run only after v21-cse benches and a regression is identified.
#
# interleave_under stops at min_source_size / max(P). With v21's
# max(P) = 0.40, --max-samples 2400 yields 2400/0.40 = 6000 interleaved
# rows (~1500 optimizer steps; ~80-100 min on 4xH100; ~110-140 min on
# 8xRTX PRO 6000 96GB at the doubled eff_bs noted below).
#
# Geometry preserved from v18.2 / Phase B (cutoff 16384, packing off):
# the catalog drill is the most fragile axis to install, so the run
# takes Phase B's long-context unpacked geometry. Phase A and TAA shards
# are short enough that they fit trivially under this cutoff.
#
# Recipe (Phase B geometry; LR matches v18.2; v18.2-mix 3-shard interleave):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-14b-v21-cse
#                       (HF; overridable via --base-model)
#   - Datasets (mix)  : ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.25)
#                       ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm         (0.40)
#                       ift_data_2026_05_18_v21_taa                            (0.35)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 4 on 4xH100 (per_device 1 x grad_accum 1 x 4 GPUs).
#     On 8xRTX PRO 6000 (this v21 host) eff_bs auto-doubles to 8 because
#     R_GA floors to 1 when EFFECTIVE_GPUS > 4; pass --batch-target 4 to
#     force the v20 sizing (yields R_GA=0 -> 1, identical to the auto-
#     detected 8-GPU value -- this flag is informational, the math is
#     pinned by the GPU count). Phase B touch-up is robust at eff_bs 8.
#   - save every 200 steps; intra-training eval DISABLED (see body comment)
#   - --max-samples 2400 (per dataset; --max-samples is per-shard in
#     LlamaFactory, applied BEFORE interleaving)
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v21-recalibrate
#
# Estimated wall-time:
#   4xH100 80GB SXM      : ~95-115 min.
#   8xRTX PRO 6000 96GB  : ~110-140 min (PCIe Gen5 vs NVLink overhead on
#                          ZeRO-3 collectives; eff_bs 8 vs 4 partially
#                          compensates by halving optimizer-step count).
#
# Full v21 chain (run sequentially; chain wrapper available):
#   1. ./run_sft_qwen25_14b_v21_core.sh           # Stage 1 broad + Phase B -> v21-core
#   2. ./run_sft_qwen25_14b_v21_plus_taa.sh       # Stage 2 TAA Classic     -> v21-taa
#   3. ./run_sft_qwen25_14b_v21_final.sh          # Stage 3 CSE drill       -> v21-cse
#   4. ./run_sft_qwen25_14b_v21_recalibrate.sh    # Stage 4 (optional)      -> v21-recalibrate
# Or via chain wrapper:
#   ./run_sft_qwen25_14b_v21_chain.sh             # taa -> cse -> recalibrate
#
# Usage:
#   ./run_sft_qwen25_14b_v21_recalibrate.sh [--repo-id USER/NAME]
#                                            [--base-model HF_REPO|LOCAL_DIR]
#                                            [--output-dir DIR]
#                                            [--report-to wandb|none]
#                                            [--max-samples N]
#                                            [--lr LR]
#                                            [--probs P_A,P_B,P_TAA]
#                                            [--offload | --no-offload]
#                                            [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
# See header for the per-shard / interleave_under semantics. Default 2400
# preserves the v18.2 / v19 / v20 Stage 5 wall-time and step count.
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
        -h|--help) sed -n '3,76p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v21-recalibrate"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v21-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_recalibrate_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_18_v21_taa"
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
        echo "[FAIL] v21 recalibrate dataset missing: SFT/data/${ds}.json" >&2
        echo "       These shards are reused verbatim from the v21 build" >&2
        echo "       (Phase A / Phase B / standalone TAA); rebuild via" >&2
        echo "       run_sft_qwen25_14b_v21_core.sh / _plus_taa.sh data" >&2
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
if [[ "${GPU_COUNT}" -ne 4 && "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] expected 4 GPUs (4xH100) or 8 GPUs (8xRTX PRO 6000); detected ${GPU_COUNT}. Recipe was sized for 4x; effective batch will reflect detected count: eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS ))." >&2
elif [[ "${GPU_COUNT}" -eq 8 ]]; then
    echo "[info] detected 8 GPUs; eff_bs auto-doubles to $(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (v20 sized for 4xH100 eff_bs=4; Phase B touch-up is robust at eff_bs=8)." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_v21_recalibrate() {
    echo "=== v21 recalibrate (Qwen2.5-14B): v18.2-mix 3-shard interleave from v21-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
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
echo "  batch math   : per_device=${R_BATCH} grad_accum=${R_GA} -> eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (target 4 on 4xH100, 8 on 8x)"
echo "  base model   : ${BASE_MODEL}"
echo "  datasets     : ${DATASETS}"
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA; v18.2 production mix)"
echo "  max samples  : ${MAX_SAMPLES}/dataset -> ~6000 interleaved rows at probs=0.25/0.40/0.35 (eval disabled)"
echo "  learning rate: ${LR}  (Phase B was 5e-06; touch-up is 1/5th, same as v18.2)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v21_recalibrate
