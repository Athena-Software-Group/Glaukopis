#!/bin/bash

# v21 multi-shard Recalibrate touch-up of
# asg-ai/athena-cti-sft-qwen25-32b-v21-cse (Qwen2.5-32B port).
# Off-plan extension of the v21 chain (v21_plan.txt §3 defines only
# Core/TAA/CSE; this launcher mirrors the 14B v21 Recalibrate stage that
# re-exposes the chained checkpoint to the Core Phase A/B + TAA shards
# under Phase B long-context geometry so a single ship candidate is
# produced from the 32B chain).
#
# Recipe parity with run_sft_qwen25_14b_v21_recalibrate.sh:
#   - Identical datasets (template-baked, architecture-independent),
#     interleave probs (0.25/0.40/0.35), lr (1e-6), cutoff (16384),
#     packing (off), max-samples (2400), and eval-disable behaviour.
#   - 32B deltas (memory only, no recipe change):
#       * --optim adamw_8bit added to EXTRA flags (v8/v11 32B precedent;
#         mandatory at 32B ZeRO-3 no-offload).
#       * Offload default flipped from "auto" (off on >=8 GPUs) to "on"
#         (offload always enabled). 32B at cutoff=16384 packing=off
#         even with adamw_8bit + GC + Liger leaves no margin for the
#         3-shard interleave's variable sequence-length spikes; offload
#         is cheap on this 1.5 h touch-up and rules out the OOM risk.
#         Pass --no-offload if you have FA2 confirmed loaded on H100
#         SXM and want the ~25% throughput win.
#
# Why a Recalibrate touch-up on the 32B v21 chain (carries the 14B v21
# rationale unchanged):
#   The 14B v21 chain showed Stage 3 (CSE drill) erodes VSP by ~10pp
#   even on a clean Core base; the off-plan Recalibrate touch-up
#   recovered VSP (83.1 vs 72.9 at CSE) without undoing the CSE gains
#   and produced the 14B ship candidate at 62.3 Total. The 32B port is
#   expected to show the same Stage 3 erosion shape; Recalibrate is on
#   the default chain path for parity with the 14B sign-off.
#
# interleave_under stops at min_source_size / max(P). With v21's
# max(P) = 0.40, --max-samples 2400 yields 2400/0.40 = 6000 interleaved
# rows (~1500 optimizer steps; ~3-4 h on 8xH100 80GB SXM at 32B with
# offload on, adamw_8bit, GC on).
#
# Geometry preserved from v18.2 / Phase B (cutoff 16384, packing off):
# the catalog drill is the most fragile axis to install, so the run
# takes Phase B's long-context unpacked geometry. Phase A and TAA shards
# are short enough that they fit trivially under this cutoff.
#
# Recipe (Phase B geometry; LR matches v18.2; v18.2-mix 3-shard interleave):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-32b-v21-cse
#                       (HF; overridable via --base-model)
#   - Datasets (mix)  : ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.25)
#                       ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm         (0.40)
#                       ift_data_2026_05_18_v21_taa                            (0.35)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 8 on 8xH100 (per_device 1 x grad_accum 1 x 8 GPUs);
#     32B chain host is assumed 8xH100 SXM (the 4xH100 4-GPU sizing the
#     14B Recalibrate used does not fit 32B at cutoff=16384 packing=off
#     even with offload on -- ZeRO-3 weight shard doubles to ~16 GB/rank
#     and tips the 80 GB budget).
#   - save every 200 steps; intra-training eval DISABLED (see body comment)
#   - --max-samples 2400 (per dataset; --max-samples is per-shard in
#     LlamaFactory, applied BEFORE interleaving)
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - adamw_8bit optimizer (v8/v11 32B precedent)
#   - Offload ON by default (see header above)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-recalibrate
#
# Estimated wall-time:
#   8xH100 80GB SXM      : ~3-4 h with offload on (offload roughly doubles
#                          the touch-up wall vs the 14B 4xH100 path; 32B
#                          throughput is the dominant factor).
#   8xRTX PRO 6000 96GB  : ~4-6 h (PCIe Gen5 vs NVLink + offload tax).
#
# Full v21 chain (run sequentially; chain wrapper available):
#   1. ./run_sft_qwen25_32b_v21_core.sh           # Stage 1 broad + Phase B -> v21-core
#   2. ./run_sft_qwen25_32b_v21_plus_taa.sh       # Stage 2 TAA Classic     -> v21-taa
#   3. ./run_sft_qwen25_32b_v21_final.sh          # Stage 3 CSE drill       -> v21-cse
#   4. ./run_sft_qwen25_32b_v21_recalibrate.sh    # Stage 4 (optional)      -> v21-recalibrate
# Or via chain wrapper:
#   ./run_sft_qwen25_32b_v21_chain.sh             # taa -> cse -> recalibrate
#
# Usage:
#   ./run_sft_qwen25_32b_v21_recalibrate.sh [--repo-id USER/NAME]
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-recalibrate"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-32b-v21-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-32B-Instruct"
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
        echo "       run_sft_qwen25_32b_v21_core.sh / _plus_taa.sh data" >&2
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

# 32B at cutoff=16384 packing=off with a 3-shard interleave leaves no
# margin for the variable sequence-length spikes even on 8xH100 80GB with
# adamw_8bit + Liger + GC; default to offload ON unconditionally and let
# the caller pass --no-offload when they have FA2 confirmed loaded and
# want the ~25% throughput win.
if [[ "${OFFLOAD}" == "auto" ]]; then
    OFFLOAD="on"
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# 32B chain host is assumed 8xH100 SXM. eff_bs target is 8
# (per_device 1 x grad_accum 1 x 8 GPUs); the integer floor in R_GA
# preserves eff_bs=8 at GPU_COUNT=8 and clamps at 1 otherwise.
R_BATCH=1; R_GA=$(( 8 / (R_BATCH * EFFECTIVE_GPUS) )); [[ ${R_GA} -lt 1 ]] && R_GA=1

# --optim adamw_8bit carries v8 32B / v11 32B precedent; required to fit
# 32B ZeRO-3 even with offload on at the 80 GB/rank budget when the
# 3-shard interleave's longer sequences hit Phase B cutoff geometry.
EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit --do_eval False --eval_strategy no --val_size 0 --mix_strategy interleave_under --interleave_probs ${PROBS}"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] expected 8 GPUs (8xH100 SXM); detected ${GPU_COUNT}. 32B Recalibrate at cutoff=16384 packing=off does not fit at GPU_COUNT<8 (ZeRO-3 weight shard doubles to ~16 GB/rank); effective batch will reflect detected count: eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_v21_recalibrate() {
    echo "=== v21 recalibrate (Qwen2.5-32B): v18.2-mix 3-shard interleave from v21-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
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
echo "  batch math   : per_device=${R_BATCH} grad_accum=${R_GA} -> eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (target 8 on 8xH100 SXM)"
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
