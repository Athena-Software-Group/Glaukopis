#!/bin/bash

# v21 multi-shard Recalibrate touch-up of
# asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
# (Qwen3-30B-A3B-Thinking-2507 port). Off-plan extension of the v21
# chain (v21_plan.txt §3 defines only Core/TAA/CSE); this launcher
# mirrors the 32B v21 Recalibrate stage that re-exposes the chained
# checkpoint to the Core Phase A/B + TAA shards under Phase B long-
# context geometry so a single ship candidate is produced from the
# Qwen3-MoE chain.
#
# NOTE: this launcher is OFF-CHAIN on the Qwen3-MoE v21 port. The
# default chain (run_sft_qwen3_30b_a3b_thinking_v21_chain.sh) now ships
# the 32B-tuned recal-32b recipe at Stage 4 (lr 3e-6, probs 0.15/0.60/
# 0.25, max-samples 3600) because the dense Qwen2.5-32B port confirmed
# that the 14B-recipe Recalibrate (lr 1e-6, probs 0.25/0.40/0.35, max-
# samples 2400) drifts VSP the wrong way at 32B+ scale under adamw_8bit
# (78.9 -> 75.7 vs the 14B 72.9 -> 83.1 recovery shape). The Qwen3-MoE
# parent is peer-scale to dense 32B, so this 14B-recipe variant is
# retained only as a manual A/B against the on-chain recal-32b stage.
# Both Stage-4 variants share v21-cse as their parent checkpoint;
# naming reflects RECIPE PROVENANCE, not chain position.
#
# Recipe parity with run_sft_qwen25_32b_v21_recalibrate.sh:
#   - Identical datasets, interleave probs (0.25/0.40/0.35), lr (1e-6),
#     cutoff (16384), packing (off), max-samples (2400), eval-disable.
#   - --optim adamw_8bit retained; Liger ON.
#
# Qwen3-MoE deltas vs Qwen2.5-32B v21-recalibrate (B300 / template / sparse):
#   - --template qwen3 (was qwen).
#   - --enable_thinking True (run_train.sh default; the qwen3 reasoning
#     template injects <think>\n\n</think> into loss/response_ids on
#     samples without a <think> block; model learns to autonomously
#     emit the empty 6-token thought + answer for CTI prompts -- see
#     run_sft_qwen3_30b_a3b_thinking_v21_core.sh header for mechanism).
#   - OFFLOAD default off (was on). 8xB300 = 288 GB HBM3e per GPU;
#     30.5B MoE ZeRO-3 shard ~15 GB/rank with adamw_8bit at 8 ranks,
#     and even at cutoff=16384 packing=off with the 3-shard interleave's
#     variable sequence-length spikes the headroom is comfortable.
#     Pass --offload to re-enable for smaller HBM hosts.
#   - --base-model defaults to the Qwen3-MoE v21-cse HF push target.
#
# Why this 14B-recipe variant exists alongside the on-chain recal-32b:
#   The 14B v21 chain shipped from this recipe (62.3 Total, VSP 72.9 ->
#   83.1 recovery). At 14B scale the lr 1e-6 + 0.40 Phase-B share
#   produces enough optimizer signal to recover VSP without undoing
#   CSE gains. At 30.5B / 3.3B-active MoE scale under adamw_8bit the
#   prior is that this recipe sits at the optimizer noise floor (the
#   dense 32B port shows this directly), but the Qwen3-MoE active-path
#   is sparser than dense 32B and the recipe could in principle behave
#   differently. This launcher exists so that hypothesis can be tested
#   off-chain after the recal-32b ship; it is NOT invoked by the
#   chain orchestrator.
#
# interleave_under stops at min_source_size / max(P). With max(P)=0.40,
# --max-samples 2400 yields 2400/0.40 = 6000 interleaved rows
# (~1500 optimizer steps; ~1.5-2 h on 8xB300 at 30.5B MoE).
#
# Recipe (Phase B geometry; LR matches 14B/32B recal; v18.2-mix interleave):
#   - Base model      : asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
#                       (HF; overridable via --base-model)
#   - Datasets (mix)  : ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.25)
#                       ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm         (0.40)
#                       ift_data_2026_05_18_v21_taa                            (0.35)
#   - 1 epoch, lr 1e-6, cutoff 16384, packing OFF
#   - Effective batch 8 on 8xB300 (per_device 1 x grad_accum 1 x 8 GPUs)
#   - save every 200 steps; intra-training eval DISABLED (see body)
#   - --max-samples 2400 per dataset (applied before interleaving)
#   - Gradient checkpointing ON (LF default; required at this cutoff)
#   - adamw_8bit optimizer; Liger kernel ON
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recalibrate
#
# Estimated wall-time:
#   8xB300 288GB SXM     : ~1.5-2 h (no offload).
#   8xH200 141GB SXM     : ~2.5-3.5 h (no offload).
#   8xH100 80GB SXM      : ~3-4 h with --offload on (matches 32B recal).
#
# Usage:
#   ./run_sft_qwen3_30b_a3b_thinking_v21_recalibrate.sh
#       [--repo-id USER/NAME] [--base-model HF_REPO|LOCAL_DIR]
#       [--output-dir DIR] [--report-to wandb|none]
#       [--max-samples N] [--lr LR] [--probs P_A,P_B,P_TAA]
#       [--offload | --no-offload] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
MAX_SAMPLES=2400
LR="1e-06"
PROBS="0.25,0.40,0.35"
DRY_RUN=0
OFFLOAD="off"

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
        -h|--help) sed -n '3,80p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recalibrate"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen3-30B-A3B-Thinking-2507"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_recalibrate_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_18_v21_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"

# Intra-training eval DISABLED: LlamaFactory requires
# len(eval_dataset) == len(interleave_probs) when interleaving and dedupes
# by name; listing core_val twice silently dedupes to 2 vs 3 probs and
# raises in numpy.random.Generator.choice. See run_sft_qwen25_32b_v21_
# recalibrate.sh header for the full rationale. Both --do_eval False AND
# --eval_strategy no are needed because TrainingArguments.__post_init__
# auto-flips do_eval back to True whenever eval_strategy != "no".

for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21 recalibrate dataset missing: SFT/data/${ds}.json" >&2
        echo "       Rebuild via run_sft_qwen3_30b_a3b_thinking_v21_core.sh /" >&2
        echo "       _plus_taa.sh data preflights." >&2
        exit 2
    fi
done


GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# 8xB300 target. eff_bs target is 8 (per_device 1 x grad_accum 1 x 8 GPUs);
# the integer floor in R_GA preserves eff_bs=8 at GPU_COUNT=8 and clamps
# at 1 otherwise.
R_BATCH=1; R_GA=$(( 8 / (R_BATCH * EFFECTIVE_GPUS) )); [[ ${R_GA} -lt 1 ]] && R_GA=1

# --optim adamw_8bit carries v8/v11 32B precedent; on MoE it operates on
# flat optimizer state and is indifferent to expert routing.
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
    echo "[warn] recipe sized for 8 GPUs (8xB300 target); detected ${GPU_COUNT}. Effective batch will reflect detected count: eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_v21_recalibrate() {
    echo "=== v21 recalibrate (Qwen3-30B-A3B-Thinking-2507): v18.2-mix 3-shard interleave from v21-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}, enable_thinking=True) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASETS}" --template qwen3 --finetuning full \
        --epochs 1 --lr "${LR}" --batch ${R_BATCH} --grad-accum ${R_GA} \
        --cutoff 16384 --save-steps 200 --eval-steps 200 --packing false \
        --max-samples "${MAX_SAMPLES}" --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${R_BATCH} grad_accum=${R_GA} -> eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (target 8 on 8xB300)"
echo "  base model   : ${BASE_MODEL}  (template=qwen3; --enable_thinking True default)"
echo "  datasets     : ${DATASETS}"
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA; v18.2 production mix)"
echo "  max samples  : ${MAX_SAMPLES}/dataset -> ~6000 interleaved rows at probs=0.25/0.40/0.35 (eval disabled)"
echo "  learning rate: ${LR}  (Phase B was 5e-06; touch-up is 1/5th, same as v18.2)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v21_recalibrate
