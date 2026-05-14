#!/bin/bash

# v18.2 multi-shard replay touch-up of asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse.
# Stage 4 of the v18.1 chain -- supersedes the single-shard cse-rms (v18.1-cse-rms)
# experiment by interleaving three replay shards at low LR to protect MCQ and
# TAA Classic alongside the RMS/ATE/VSP recovery the cse-rms run delivered.
#
# Why a multi-shard replay (and not the single Phase B shard from cse-rms):
#   The cse-rms touch-up (run_sft_qwen25_14b_v18p1_rms_replay.sh) recovered
#   RMS (+11.3 pp combined_f1) and ATE (+5.8 pp) but introduced new
#   regressions: MCQ -4.0 pp and TAA Classic -12.0 pp, because the Phase B
#   shard contains no MCQ or TAA Classic rows. v18.2 fixes this by
#   interleaving three shards in one Stage 4 pass:
#     - core_a_kb_mcq_taa_soc_cm_ms_yn  (MCQ + KB + SOC + CM + MS + YN
#                                        coverage; protects MCQ axis)
#     - core_b_rms_ate_vsp_rcm          (Phase B catalog drill; the
#                                        only place RMS/ATE/VSP/RCM
#                                        are ever installed)
#     - taa                             (TAA Classic shard; protects
#                                        the +TAA stage's contribution)
#   Mix strategy interleave_under with probs 0.25 / 0.40 / 0.35; total
#   capped at --max-samples (default 6000) so wallclock stays in the
#   ~80-100 min envelope of cse-rms.
#
# Geometry preserved from Phase B (cutoff 16384, packing off): the catalog
# drill is the most fragile axis to install, so the run takes Phase B's
# long-context unpacked geometry. Phase A and TAA shards are short enough
# that they fit trivially under this cutoff (just less efficient).
#
# Recipe (Phase B geometry; LR matches cse-rms; multi-shard interleave):
#   - Base model      : asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse
#                       (HF; overridable via --base-model)
#   - Datasets (mix)  : ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.25)
#                       ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm         (0.40)
#                       ift_data_2026_05_11_v18p1_taa                            (0.35)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 4   (per_device 1 x grad_accum 1 x 4 GPUs)
#   - eval/save every 200 steps
#   - --max-samples 6000  (~1500 optimizer steps; ~100-120 min on 4xH100)
#   - Gradient checkpointing ON (LlamaFactory default; required at this cutoff)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-2
#                  (NEW repo; cse-rms repo retained for regression comparison)
#
# Estimated wall-time on 4xH100: ~1.5-2 h.
#
# Full v18.1 chain with multi-shard touch-up:
#   1. ./run_sft_qwen25_14b_v18p1_core.sh             # broad + Phase B  -> v18-1-core
#   2. ./run_sft_qwen25_14b_v18p1_plus_taa.sh         # TAA Classic      -> v18-1-taa
#   3. ./run_sft_qwen25_14b_v18p1_final.sh            # CSE drill        -> v18-1-cse
#   4. ./run_sft_qwen25_14b_v18p2_multi_replay.sh     # 3-shard replay   -> v18-2
#
# Usage:
#   ./run_sft_qwen25_14b_v18p2_multi_replay.sh [--repo-id USER/NAME]
#                                              [--base-model HF_REPO|LOCAL_DIR]
#                                              [--output-dir DIR]
#                                              [--report-to wandb|none]
#                                              [--max-samples N]
#                                              [--lr LR]
#                                              [--probs P_A,P_B,P_TAA]
#                                              [--offload | --no-offload]
#                                              [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
MAX_SAMPLES=6000
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-2"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18p2_multi_replay_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_11_v18p1_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"

# LlamaFactory's data_args validator requires len(eval_dataset) ==
# len(interleave_probs) when interleaving (see hparams/data_args.py:169).
# We therefore align one eval shard per train shard: core_val covers Phase A
# and Phase B (it is the unified Core validator built from both phases),
# and taa_val covers the standalone TAA shard.
VAL_PHASE_A="ift_data_2026_05_11_v18p1_core_val"
VAL_PHASE_B="ift_data_2026_05_11_v18p1_core_val"
VAL_TAA="ift_data_2026_05_11_v18p1_taa_val"
VAL_DATASETS="${VAL_PHASE_A},${VAL_PHASE_B},${VAL_TAA}"

for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}" "${VAL_PHASE_A}" "${VAL_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v18.2 multi-replay dataset missing: SFT/data/${ds}.json" >&2
        echo "       These shards are reused verbatim from the v18.1 build" >&2
        echo "       (Phase A / Phase B / standalone TAA + matching val sets);" >&2
        echo "       rebuild via run_sft_qwen25_14b_v18p1_core.sh / _plus_taa.sh" >&2
        echo "       data preflights or copy from the Core training host." >&2
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

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --eval_dataset ${VAL_DATASETS} --val_size 0 --mix_strategy interleave_under --interleave_probs ${PROBS}"

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

run_v18p2_multi_replay() {
    echo "=== v18.2 multi-replay (Qwen2.5-14B): 3-shard interleave from v18.1-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}) ==="
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
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA)"
echo "  eval datasets: ${VAL_DATASETS}  max_samples=${MAX_SAMPLES}"
echo "  learning rate: ${LR}  (Phase B was 5e-06; touch-up is 1/5th, same as cse-rms)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v18p2_multi_replay
