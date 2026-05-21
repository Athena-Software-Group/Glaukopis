#!/bin/bash

# v21 recal-32b: 32B-recipe variant of the off-plan Stage 4 Recalibrate
# touch-up. Same topology as run_sft_qwen25_32b_v21_recalibrate.sh
# (Phase A + Phase B + TAA three-shard interleave_under touch-up of
# asg-ai/athena-cti-sft-qwen25-32b-v21-cse), but with the recipe re-
# tuned for the 32B scale after the strict 14B-recipe port (the existing
# qwen25-32b-v21-recalibrate repo, which uses the 14B recal recipe
# verbatim) failed to recover VSP on Qwen2.5-32B-Instruct.
#
# Naming: both Stage-4 variants are parallel branches off v21-cse, not
# stacked. The split is by RECIPE PROVENANCE, not chain position:
#   v21-recalibrate    <- 14B recipe (lr 1e-6, mix 0.25/0.40/0.35, ms 2400)
#   v21-recal-32b      <- 32B recipe (lr 3e-6, mix 0.15/0.60/0.25, ms 3600)
#
# Empirical motivation (see Findings block in
# tmpl_gen/templates/05182026/README-21.md once updated):
#   On the 14B v21 chain, Stage 4 lifts VSP from 72.9 (post-CSE) back to
#   83.1 (above Core's 82.5). On the 32B port using the byte-identical
#   recipe, VSP drifts the wrong way: 78.9 (post-CSE) -> 75.7 (post-
#   recalibrate). Other axes (CKT, CSE-TI) move correctly; only the
#   VSP/RMS/ATE re-anchor signal is missing. The most parsimonious
#   explanation is that the 14B-tuned 1e-6 LR + 0.40 Phase-B share
#   produces enough optimizer signal at 14B but sits at the noise floor
#   on 32B + adamw_8bit, so the Phase B catalog re-exposure cannot
#   overpower the post-CSE residual at this scale.
#
# Three coupled deltas vs run_sft_qwen25_32b_v21_recalibrate.sh, chosen
# to hold step count and wall-time constant so the only A/B variable is
# the catalog-recovery recipe itself:
#
#   - LR        1e-6 -> 3e-6     (3x bump; rough 32B/14B param ratio)
#   - Probs     0.25/0.40/0.35   -> 0.15/0.60/0.25
#                                 (heavier Phase B share = more VSP/RMS
#                                  catalog exposure per interleaved row;
#                                  Phase A and TAA reduced because neither
#                                  is the bottleneck -- CKT/TAA Classic are
#                                  already in-band on v21-cse)
#   - Max-samp  2400 -> 3600     (interleave_under cap = max_samples /
#                                 max(P). New max(P)=0.60 -> 6000
#                                 interleaved rows, same as the original
#                                 2400/0.40=6000. Step count and wall-time
#                                 preserved; only composition shifts.)
#
# Everything else (cutoff 16384, packing off, eff_bs 8, --optim
# adamw_8bit, Liger, GC on, offload default-on at 32B, ZeRO-3) is held
# identical to run_sft_qwen25_32b_v21_recalibrate.sh.
#
# Base checkpoint: v21-cse (not v21-recalibrate). Both the original recal
# and this tuned variant branch off the same CSE-stage parent so the
# downstream bench comparison isolates the recipe change.
#
# Status: diagnostic / off-plan. The v21 reproducibility result for the
# 32B port is the v21-cse checkpoint (Total 65.8 / Weighted 64.9 in the
# 2026-05-20 bench sweep). This tuned variant is a follow-up experiment
# to test whether the off-plan Recalibrate stage can be re-tuned for 32B
# to lift VSP without sacrificing the cse-stage gains. If it succeeds it
# becomes the 32B ship candidate; if not, v21-cse stays the headline.
#
# Recipe (Phase B geometry; LR tuned for 32B; Phase-B-heavy 3-shard mix):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-32b-v21-cse
#                       (HF; overridable via --base-model)
#   - Datasets (mix)  : ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.15)
#                       ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm         (0.60)
#                       ift_data_2026_05_18_v21_taa                            (0.25)
#   - 1 epoch, lr 3e-6, cutoff 16384, packing OFF
#   - Effective batch 8 on 8xH100 (per_device 1 x grad_accum 1 x 8 GPUs)
#   - save every 200 steps; intra-training eval DISABLED (see body comment)
#   - --max-samples 3600 (per dataset; 3600/0.60 = 6000 interleaved rows,
#     ~1500 optimizer steps -- same step count as the original recal)
#   - Gradient checkpointing ON, adamw_8bit, offload ON by default
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-recal-32b
#
# Estimated wall-time (matches original 32B recal; only composition changed):
#   8xH100 80GB SXM      : ~3-4 h with offload on.
#   8xRTX PRO 6000 96GB  : ~4-6 h (PCIe Gen5 vs NVLink + offload tax).
#
# Usage:
#   ./run_sft_qwen25_32b_v21_recal_32b.sh [--repo-id USER/NAME]
#                                         [--base-model HF_REPO|LOCAL_DIR]
#                                         [--output-dir DIR]
#                                         [--report-to wandb|none]
#                                         [--max-samples N]
#                                         [--lr LR]
#                                                  [--probs P_A,P_B,P_TAA]
#                                                  [--offload | --no-offload]
#                                                  [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
# Per-shard cap; interleave_under stops at min_source_size / max(P). With
# the tuned mix max(P)=0.60, 3600 -> 6000 interleaved rows -> ~1500
# optimizer steps at eff_bs=8 (same step count as the original 2400 /
# 0.40 = 6000 recal recipe). Holding step count constant keeps the only
# A/B variable the recipe composition + LR.
MAX_SAMPLES=3600
LR="3e-06"
PROBS="0.15,0.60,0.25"
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
        -h|--help) sed -n '3,84p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-recal-32b"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-32b-v21-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-32B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_recal_32b_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_18_v21_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"

# Intra-training eval is DISABLED for this multi-shard touch-up. Same
# rationale as run_sft_qwen25_32b_v21_recalibrate.sh: LlamaFactory
# requires len(eval_dataset) == len(interleave_probs) when interleaving,
# and listing core_val once silently dedupes to 1 unique entry against
# 3 probs ("a and p must have same size"). Producing three distinct eval
# shards is not worth it for a touch-up where sign-off is via the
# AthenaBench/CSE/CM bench suites. Both --do_eval False and
# --eval_strategy no are required to overcome TrainingArguments'
# __post_init__ auto-flip and the parser validator at
# hparams/parser.py:344-347. The trainer still logs per-step train loss.

for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21 recal-32b dataset missing: SFT/data/${ds}.json" >&2
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
# want the ~25% throughput win. Memory envelope is identical to the
# original recal -- only LR / probs / max-samples differ.
if [[ "${OFFLOAD}" == "auto" ]]; then
    OFFLOAD="on"
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# 32B chain host is assumed 8xH100 SXM. eff_bs target is 8
# (per_device 1 x grad_accum 1 x 8 GPUs); identical to original recal.
R_BATCH=1; R_GA=$(( 8 / (R_BATCH * EFFECTIVE_GPUS) )); [[ ${R_GA} -lt 1 ]] && R_GA=1

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
    echo "[warn] expected 8 GPUs (8xH100 SXM); detected ${GPU_COUNT}. 32B recal-32b at cutoff=16384 packing=off does not fit at GPU_COUNT<8 (ZeRO-3 weight shard doubles to ~16 GB/rank); effective batch will reflect detected count: eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_v21_recal_32b() {
    echo "=== v21 recal-32b (Qwen2.5-32B): Phase-B-heavy 3-shard interleave from v21-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
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
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA; Phase-B-heavy 32B recipe)"
echo "  max samples  : ${MAX_SAMPLES}/dataset -> ~6000 interleaved rows at max(P)=0.60 (~1500 steps; eval disabled)"
echo "  learning rate: ${LR}  (3x the original recal's 1e-6 to clear the 32B+adamw_8bit optimizer noise floor)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v21_recal_32b
