#!/bin/bash

# v18.2.2 multi-shard replay touch-up of asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse.
# Stage 4 of the v18.1 chain -- iteration of the v18.2 multi-shard recipe after
# the v18.2.1 bench surfaced a strict regression of the §7.4 gate package
# (4 gates failing vs v18.2's 2):
#   - MCQ 63.17 (target >=70.0; -6.83 pp; +0.84 vs v18.2's 62.33; the
#     v18.2 -> v18.2.1 prob bump Phase A 0.25 -> 0.35 was within noise)
#   - RMS combined_f1 50.37 (target >=55.0; -4.63 pp; -4.35 vs v18.2;
#     the Phase B 0.40 -> 0.45 bump INVERTED -- raising Phase B's prob
#     bumps a 4-task shard uniformly so RMS-specific signal ratio did
#     not improve, then the +22% step bump over-trained the dilution)
#   - ATE 62.40 (target >=63.0; -0.60 pp; previously PASS at 63.20)
#   - RCM 66.80 (target >=67.5; -0.70 pp; previously PASS at 72.55)
# Other axes held: VSP 82.65, TAA combined 47.00, CSE-TI 41.79, CSE-Mal 23.48,
# CM-2K 89.35, CM-10K 84.17. Trade-ratio analysis (plan §8.2.3): v18.2.1's
# MCQ-for-RMS exchange (|dRMS/dMCQ| = 0.45) is half as efficient as v18.2's
# (0.86), implying v18.2 was on the better side of the diminishing-returns
# curve and the v18.2.1 rebalance pushed past the optimum on both probs and
# step count.
#
# Recipe delta vs v18.2 (this is the only difference; geometry/LR/cutoff/
# packing/eff_bs all UNCHANGED from v18.2.1):
#   - Phase A prob 0.35 -> 0.25  (REVERT to v18.2)
#   - Phase B prob 0.45 -> 0.40  (REVERT to v18.2)
#   - TAA prob     0.20 -> 0.35  (REVERT to v18.2)
#   - max-samples  3000 -> 1500  (per dataset; -50% vs v18.2.1 / -38% vs v18.2;
#                                  ~3750 interleaved total at max-prob 0.40;
#                                  ~937 vs 1667 (v18.2.1) / 1500 (v18.2) steps)
#   Mix strategy interleave_under (UNCHANGED), lr 1e-6 (UNCHANGED), cutoff
#   16384 (UNCHANGED), packing off (UNCHANGED), eff_bs 4 (UNCHANGED).
#
# Hypothesis (plan §8.3): the v18.2 prob mix is correct and the regression is
# caused by Stage 4 being TOO LONG (over-exposure of the catalog shard erodes
# the Phase A and CSE drill circuits). Reducing the step count without
# changing probs should preserve the RMS gain while reducing MCQ damage and
# protect ATE/RCM from sliding below their floors.
#
# Geometry preserved from Phase B (cutoff 16384, packing off): the catalog
# drill is the most fragile axis to install, so the run takes Phase B's
# long-context unpacked geometry. Phase A and TAA shards are short enough
# that they fit trivially under this cutoff (just less efficient).
#
# Recipe (Phase B geometry; LR matches v18.2; multi-shard interleave):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse
#                       (HF; overridable via --base-model)
#                       NOTE: base is v18.1-cse, NOT v18.2 or v18.2.1 -- this
#                       is a fresh Stage 4 with the v18.2 prob mix at half
#                       the step count, comparable to v18.2 and v18.2.1 on
#                       every axis.
#   - Datasets (mix)  : ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.25)
#                       ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm         (0.40)
#                       ift_data_2026_05_11_v18p1_taa                            (0.35)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 4   (per_device 1 x grad_accum 1 x 4 GPUs)
#   - save every 200 steps; intra-training eval DISABLED (see body comment)
#   - --max-samples 1500 (per dataset; --max-samples is per-shard in
#     LlamaFactory, applied BEFORE interleaving). interleave_under +
#     probs P stops at min_source_size / max(P) = 1500/0.40 = 3750 final
#     training samples (~937 optimizer steps; ~50-65 min on 4xH100).
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-2-2
#                  (NEW repo; v18-2 and v18-2-1 retained for regression
#                   comparison)
#
# Estimated wall-time on 4xH100: ~50-65 min.
#
# Full v18.1 chain with v18.2.2 multi-shard touch-up:
#   1. ./run_sft_qwen25_14b_v18p1_core.sh             # broad + Phase B  -> v18-1-core
#   2. ./run_sft_qwen25_14b_v18p1_plus_taa.sh         # TAA Classic      -> v18-1-taa
#   3. ./run_sft_qwen25_14b_v18p1_final.sh            # CSE drill        -> v18-1-cse
#   4. ./run_sft_qwen25_14b_v18p2p2_multi_replay.sh   # 3-shard replay   -> v18-2-2
#                                                     # (v18.2 mix at half steps)
#
# Usage:
#   ./run_sft_qwen25_14b_v18p2p2_multi_replay.sh [--repo-id USER/NAME]
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
# source determines when interleaving stops). Default 1500 with probs
# 0.25/0.40/0.35 -> 1500/0.40 = 3750 final training samples (~937 optimizer
# steps at eff_bs 4, ~50-65 min on 4xH100 -- ~50% fewer steps than v18.2.1).
MAX_SAMPLES=1500
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
        -h|--help) sed -n '3,89p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-2-2"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18p2p2_multi_replay_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_11_v18p1_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"


# Intra-training eval is DISABLED for this multi-shard touch-up; rationale
# identical to v18.2 / v18.2.1 (LlamaFactory's loader keys datasets by name
# so the 3:3 shard:prob alignment requirement cannot be satisfied with the
# available val shards without a fresh data build). To fully disable eval we
# override BOTH --do_eval AND --eval_strategy because transformers'
# TrainingArguments __post_init__ auto-flips do_eval=True whenever
# eval_strategy != "no". See run_sft_qwen25_14b_v18p2_multi_replay.sh for
# the long-form derivation.


for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v18.2.2 multi-replay dataset missing: SFT/data/${ds}.json" >&2
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

run_v18p2p2_multi_replay() {
    echo "=== v18.2.2 multi-replay (Qwen2.5-14B): 3-shard interleave from v18.1-cse with v18.2 prob mix at half steps (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
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

run_v18p2p2_multi_replay
