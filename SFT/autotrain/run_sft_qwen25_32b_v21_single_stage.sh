#!/bin/bash

# v21 matched single-stage control: one full-parameter SFT pass of
# Qwen2.5-32B-Instruct over the UNION of the v21 chain's training shards.
# Exists to answer the reviewer / committee question "does the four-stage
# schedule beat a single pass over the same data with comparable
# hyperparameters?" (C&S review COSE-D-26-05507, Reviewer 2 point 1;
# doc/v1.8 (Journal of Systems and Software - Elsevier)/SINGLE_STAGE_BASELINE.md).
#
# What is matched to the v21 32B chain (run_sft_qwen25_32b_v21_{core,
# plus_taa,final,recal_32b}.sh):
#   - Base model, template, framework, optimizer stack: Qwen/Qwen2.5-32B-
#     Instruct, qwen template, LlamaFactory full SFT, DeepSpeed ZeRO-3,
#     adamw_8bit, Liger, bf16, cosine schedule, 5% warmup, GC on.
#   - Training rows: exactly the files the chain's Stages 1-3 consumed, in
#     one epoch:
#       ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn  168,859
#       tulu_3_sft_mixture (capped at 240,000, as in Stage 1A)  240,000
#       alpaca_en_demo                                              999
#       ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm           49,581
#       ift_data_2026_05_18_v21_taa                              20,359
#       ift_data_2026_05_18_v21_cse                              17,320
#     --max-samples 240000 applies per dataset, so it caps Tulu-3 exactly
#     as Stage 1A did and leaves every CTI shard whole. The Stage-4 replay
#     (~6,000 rows resampled from Core-A/Core-B/TAA) is not repeated: those
#     rows are already in the union.
#   - Stage-1A hyperparameters (the chain's largest pass, ~82% of its
#     rows): lr 1e-5, cutoff 8192, packing on, effective batch 16, 1 epoch.
#
# What necessarily differs (state it in the paper):
#   - Order: one shuffled pass instead of Core -> TAA -> CSE -> Recal.
#   - The narrow shards see lr 1e-5 instead of 5e-6 (Stages 1B-3) / 3e-6
#     (Stage 4). Run the --lr 5e-06 sensitivity to bracket this.
#   - Core-B is packed at 8,192 (chain: unpacked at 16,384, eff. batch 8),
#     so its tokens are the same but it takes fewer optimizer steps. By a
#     character-length estimate only ~8 Core-B rows exceed 8,192 tokens.
#   - TAA/CSE at 8,192 instead of 4,096 (no row in either exceeds 4,096).
#
# Selection rule (fix BEFORE looking at benchmark scores): if both LRs are
# run, report both; designate the one with the lower mean eval loss over
# the three held-out validation shards (core_val, taa_val, cse_val) as the
# primary control. Do not pick by AthenaBench score.
#
# Wall time (planning estimate, 8xH100 80GB SXM, --gc on): the Stage-1A
# rows plus ~40M tokens of Core-B/TAA/CSE packed at 8,192 -- roughly the
# Stage-1A figure (~16-18 h) plus 10-25%; bounded above by the chain's own
# Stages 1-3 budget. Evaluate with
#   SFT/eval/utils/serve_and_bench_qwen25_32b_v21_single_stage.sh
#
# Usage:
#   ./run_sft_qwen25_32b_v21_single_stage.sh [--repo-id USER/NAME]
#                                            [--output-dir DIR]
#                                            [--lr 1e-05|5e-06]  # default 1e-05
#                                            [--report-to wandb|none]
#                                            [--offload | --no-offload]
#                                            [--gc auto|on|off]  # default: on
#                                            [--skip-eval]       # disables in-training eval
#                                            [--resume]          # resume from latest ckpt
#                                            [--dry-run]
#
# Default push target: ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-single-stage,
# with "-lr<LR>" appended when --lr is not 1e-05 so the sensitivity run
# never overwrites the primary one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
LR="1e-05"
REPORT_TO="wandb"
DRY_RUN=0
OFFLOAD="auto"
SKIP_EVAL=0
RESUME=0
GC_OVERRIDE="on"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --lr)           LR="$2";           shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        --gc)           GC_OVERRIDE="$2";  shift 2 ;;
        --skip-eval)    SKIP_EVAL=1;       shift ;;
        --resume)       RESUME=1;          shift ;;
        -h|--help) sed -n '3,64p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${GC_OVERRIDE}" in auto|on|off) ;; *) echo "--gc must be auto|on|off" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

LR_TAG=""
[[ "${LR}" != "1e-05" ]] && LR_TAG="-lr${LR}"
if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-single-stage${LR_TAG}"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-32B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_single_stage${LR_TAG}_${TIMESTAMP}"

CTI_SHARDS=( ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn
             ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm
             ift_data_2026_05_18_v21_taa
             ift_data_2026_05_18_v21_cse )
VAL_SHARDS=( ift_data_2026_05_18_v21_core_val
             ift_data_2026_05_18_v21_taa_val
             ift_data_2026_05_18_v21_cse_val )
DATASETS="${CTI_SHARDS[0]},tulu_3_sft_mixture,alpaca_en_demo,${CTI_SHARDS[1]},${CTI_SHARDS[2]},${CTI_SHARDS[3]}"
VAL_NAMES="$(IFS=,; echo "${VAL_SHARDS[*]}")"

for ds in "${CTI_SHARDS[@]}" "${VAL_SHARDS[@]}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21 shard missing: SFT/data/${ds}.json" >&2
        echo "       These are the same files the v21 chain trained on; copy them from" >&2
        echo "       the build host (rsync -avP workstation:Glaukopis/SFT/data/${ds}.json SFT/data/)." >&2
        exit 2
    fi
done

if [[ -n "${GPU_COUNT_OVERRIDE:-}" ]]; then
    GPU_COUNT="${GPU_COUNT_OVERRIDE}"
else
    GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
    if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" == "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    fi
    GPU_COUNT="${GPU_COUNT:-0}"
fi
if [[ "${GPU_COUNT}" == "0" && ${DRY_RUN} -eq 0 ]]; then
    echo "[FAIL] GPU probe returned 0. Fix the env (conda activate llm-sft) or set GPU_COUNT_OVERRIDE=N." >&2
    exit 3
fi

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 4 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

# Same sizing as v21-Core Phase A: bs=1 at cutoff 8192 packed fits ~72 GB/rank
# on 8xH100 SXM; grad-accum holds the effective batch at 16.
EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
BATCH=1; GA=$(( 16 / (BATCH * EFFECTIVE_GPUS) )); [[ ${GA} -lt 1 ]] && GA=1

case "${GC_OVERRIDE}" in
    on)   GC_FLAG="" ;;
    off)  GC_FLAG="--disable_gradient_checkpointing True" ;;
    auto) GC_FLAG="--disable_gradient_checkpointing True"
          [[ "${GPU_COUNT}" -lt 8 ]] && GC_FLAG=""
          ;;
esac

EXTRA_BASE="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit ${GC_FLAG}"
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    EXTRA_COMMON="${EXTRA_BASE} --eval_strategy no"
else
    EXTRA_COMMON="${EXTRA_BASE} --per_device_eval_batch_size 1 --eval_dataset ${VAL_NAMES} --val_size 0"
fi

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
[[ "${GPU_COUNT}" -gt 0 ]] && export NPROC_PER_NODE="${GPU_COUNT}"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] recipe sized for 8 GPUs (8xH100); detected ${GPU_COUNT}. Effective batch preserved: $(( BATCH * GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )
RESUME_FLAG=(); [[ ${RESUME} -eq 1 ]] && RESUME_FLAG=( --resume )

echo "=== v21 matched single-stage control (Qwen2.5-32B): union of Stage 1-3 shards, one epoch ==="
echo "  datasets     : ${DATASETS}"
echo "  eval sets    : ${VAL_NAMES}$([[ ${SKIP_EVAL} -eq 1 ]] && echo '  (disabled by --skip-eval)')"
echo "  lr / epochs  : ${LR} / 1   cutoff 8192, packing on, eff. batch $(( BATCH * GA * EFFECTIVE_GPUS ))"
echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}  grad-ckpt: $([[ -z "${GC_FLAG}" ]] && echo on || echo off)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo

bash "${SFT_DIR}/utils/run_train.sh" \
    --model "Qwen/Qwen2.5-32B-Instruct" \
    --dataset "${DATASETS}" --template qwen --finetuning full \
    --epochs 1 --lr "${LR}" --batch ${BATCH} --grad-accum ${GA} \
    --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
    --max-samples 240000 --report-to "${REPORT_TO}" \
    --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
    --extra "${EXTRA_COMMON}" \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
