#!/bin/bash

# v18.2.1 multi-shard replay touch-up of asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse.
# Stage 4 of the v18.1 chain -- iteration of the v18.2 multi-shard recipe after
# the v18.2 bench surfaced two failures against the §5 sign-off bar:
#   - MCQ 62.33 (target >=70.0; -7.67 pp). Worse than v18.1-cse (72.03) and
#     worse than the cse-rms single-shard touch-up (~68); the multi-shard
#     anti-forgetting was insufficient to protect the MCQ axis.
#   - RMS combined_f1 54.72 (target >=55.0; -0.28 pp hairline miss).
#     Phase B's effective share (0.40) yielded fewer install-shard steps
#     than the cse-rms 1.0-share run (~57.6), undercutting the recovery.
# Other axes maxed: VSP 83.87, ATE 63.20, TAA combined 47.50, RCM 72.55,
# CSE-TI 41.25 (+5.2 pp vs v18.1-cse), CM-2K 88.95, CM-10K 83.94.
#
# Recipe delta vs v18.2 (this is the only difference; geometry/LR unchanged):
#   - Phase A prob 0.25 -> 0.35  (+0.10; restore MCQ via more MCQ-bearing rows)
#   - Phase B prob 0.40 -> 0.45  (+0.05; close the 0.28 pp RMS hairline)
#   - TAA prob     0.35 -> 0.20  (-0.15; TAA combined was already PASSING and
#                                  the standalone TAA shard's short-form
#                                  structured pattern likely competes with
#                                  MCQ's letter-decoder; cse-rms had no TAA
#                                  exposure and regressed MCQ by only -4 pp
#                                  vs v18.2's -9.7 pp, supporting this theory)
#   - max-samples  2400 -> 3000  (per dataset; ~6667 interleaved total at the
#                                  new max-prob 0.45; ~1667 steps; +22% vs v18.2)
#   Mix strategy interleave_under (UNCHANGED), lr 1e-6 (UNCHANGED), cutoff
#   16384 (UNCHANGED), packing off (UNCHANGED), eff_bs 4 (UNCHANGED).
#
# Geometry preserved from Phase B (cutoff 16384, packing off): the catalog
# drill is the most fragile axis to install, so the run takes Phase B's
# long-context unpacked geometry. Phase A and TAA shards are short enough
# that they fit trivially under this cutoff (just less efficient).
#
# Recipe (Phase B geometry; LR matches v18.2; multi-shard interleave):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse
#                       (HF; overridable via --base-model)
#                       NOTE: base is v18.1-cse, NOT v18.2 -- this is a
#                       fresh Stage 4 with a rebalanced mix, not a touch-up
#                       on top of v18.2.
#   - Datasets (mix)  : ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.35)
#                       ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm         (0.45)
#                       ift_data_2026_05_11_v18p1_taa                            (0.20)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 4   (per_device 1 x grad_accum 1 x 4 GPUs)
#   - save every 200 steps; intra-training eval DISABLED (see body comment)
#   - --max-samples 3000 (per dataset; --max-samples is per-shard in
#     LlamaFactory, applied BEFORE interleaving). interleave_under +
#     probs P stops at min_source_size / max(P) = 3000/0.45 = 6667 final
#     training samples (~1667 optimizer steps; ~95-115 min on 4xH100).
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-2-1
#                  (NEW repo; v18-2 retained for regression comparison)
#
# Estimated wall-time on 4xH100: ~95-115 min.
#
# Full v18.1 chain with v18.2.1 multi-shard touch-up:
#   1. ./run_sft_qwen25_14b_v18p1_core.sh             # broad + Phase B  -> v18-1-core
#   2. ./run_sft_qwen25_14b_v18p1_plus_taa.sh         # TAA Classic      -> v18-1-taa
#   3. ./run_sft_qwen25_14b_v18p1_final.sh            # CSE drill        -> v18-1-cse
#   4. ./run_sft_qwen25_14b_v18p2p1_multi_replay.sh   # 3-shard replay   -> v18-2-1
#                                                     # (rebalanced mix of v18.2)
#
# Usage:
#   ./run_sft_qwen25_14b_v18p2p1_multi_replay.sh [--repo-id USER/NAME]
#                                                [--base-model HF_REPO|LOCAL_DIR]
#                                                [--output-dir DIR]
#                                                [--report-to wandb|none]
#                                                [--max-samples N]
#                                                [--lr LR]
#                                                [--probs P_A,P_B,P_TAA]
#                                                [--offload | --no-offload]
#                                                [--dry-run]

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
# source determines when interleaving stops). Default 3000 with probs
# 0.35/0.45/0.20 -> 3000/0.45 = 6667 final training samples (~1667 optimizer
# steps at eff_bs 4, ~95-115 min on 4xH100 -- ~22% more steps than v18.2).
MAX_SAMPLES=3000
LR="1e-06"
PROBS="0.35,0.45,0.20"
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
        -h|--help) sed -n '3,72p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-2-1"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18p2p1_multi_replay_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_11_v18p1_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"

# Intra-training eval is DISABLED for this multi-shard touch-up; rationale
# identical to v18.2 (LlamaFactory's loader keys datasets by name so the
# 3:3 shard:prob alignment requirement cannot be satisfied with the available
# val shards without a fresh data build). To fully disable eval we override
# BOTH --do_eval AND --eval_strategy because transformers' TrainingArguments
# __post_init__ auto-flips do_eval=True whenever eval_strategy != "no".
# See run_sft_qwen25_14b_v18p2_multi_replay.sh for the long-form derivation.


for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v18.2.1 multi-replay dataset missing: SFT/data/${ds}.json" >&2
        echo "       These shards are reused verbatim from the v18.1 build" >&2
        echo "       (Phase A / Phase B / standalone TAA); rebuild via" >&2
        echo "       run_sft_qwen25_14b_v18p1_core.sh / _plus_taa.sh data" >&2
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

run_v18p2p1_multi_replay() {
    echo "=== v18.2.1 multi-replay (Qwen2.5-14B): 3-shard interleave from v18.1-cse with rebalanced probs (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr "${LR}" --batch ${R_BATCH} --grad-accum ${R_GA} \
        --cutoff 16384 --save-steps 200 --eval-steps 200 --packing false \
        --max-samples "${MAX_SAMPLES}" --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

P_B="$(echo "${PROBS}" | cut -d, -f2)"
INTERLEAVED_TOTAL="$(python -c "print(int(${MAX_SAMPLES} / ${P_B}))" 2>/dev/null || echo "?")"

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${R_BATCH} grad_accum=${R_GA} -> eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (target 4)"
echo "  base model   : ${BASE_MODEL}"
echo "  datasets     : ${DATASETS}"
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA)"
echo "  max samples  : ${MAX_SAMPLES}/dataset -> ~${INTERLEAVED_TOTAL} total interleaved (eval disabled)"
echo "  learning rate: ${LR}  (matches v18.2; Phase B install was 5e-06)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v18p2p1_multi_replay
