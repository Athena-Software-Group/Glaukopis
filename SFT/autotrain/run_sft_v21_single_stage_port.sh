#!/bin/bash

# v21 matched single-pass control for the four other v21 ports: one
# full-parameter SFT pass over the UNION of the v21 chain's training shards.
# Extends run_sft_qwen25_32b_v21_single_stage.sh, same data and same recipe,
# to test whether a single pass matches the layered chain beyond Qwen2.5-32B.
#
# Models (--model-key), each compared against the layered checkpoint the paper
# reports for it:
#   foundation_8b        Foundation-Sec-8B-Instruct      vs v21-recalibrate
#   llama31_8b           Llama-3.1-8B-Instruct           vs v21-recalibrate
#   qwen25_14b           Qwen2.5-14B-Instruct            vs v21-recalibrate
#   qwen3_30b_thinking   Qwen3-30B-A3B-Thinking-2507     vs v21-cse
#
# Training rows are the files the chain's Stages 1-3 consumed, in one epoch:
#   ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn  168,859
#   tulu_3_sft_mixture (capped at 240,000, as in Stage 1A)  240,000
#   alpaca_en_demo                                              999
#   ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm           49,581
#   ift_data_2026_05_18_v21_taa                              20,359
#   ift_data_2026_05_18_v21_cse                              17,320
# --max-samples 240000 applies per dataset, so it caps Tulu-3 exactly as Stage
# 1A did and leaves every CTI shard whole. The Stage-4 replay is not repeated:
# those rows are already in the union.
#
# Hyperparameters are each model's own v21 Stage 1A values (run_sft_*_v21_core.sh
# Phase A): lr 1e-5, cutoff 8192, packing on, effective batch 16, 1 epoch, plus
# its template, per-device batch and optimizer. Only the optimizer and
# per-device batch vary by model:
#   8B and 14B: per-device batch 2, default optimizer
#   Qwen3-30B:  per-device batch 1, adamw_8bit
# Qwen3 trains with the launcher default enable_thinking=True, as its chain did.
#
# What necessarily differs from the chain (same list as the 32B control):
#   - Order: one shuffled pass instead of Core -> TAA -> CSE -> Recalibrate.
#   - The narrow shards see lr 1e-5 instead of the chain's lower later rates.
#   - Core-B is packed at 8,192 (chain: unpacked at 16,384, effective batch 8).
#   - TAA/CSE at 8,192 instead of 4,096.
#
# Memory: the 8B models fit on 2xH200 141GB with --no-offload. Use 4xH200 for
# Qwen2.5-14B (default optimizer state) and Qwen3-30B.
#
# Evaluate with SFT/eval/utils/serve_and_bench_v21_single_stage_port.sh.
#
# Usage:
#   ./run_sft_v21_single_stage_port.sh --model-key KEY
#                                      [--repo-id USER/NAME]
#                                      [--output-dir DIR]
#                                      [--report-to wandb|none]
#                                      [--offload | --no-offload]
#                                      [--gc on|off]   # default: on
#                                      [--skip-eval]   # disables in-training eval
#                                      [--resume]      # resume from latest ckpt
#                                      [--resumable]   # full ZeRO checkpoints so a crash can --resume
#                                      [--dry-run]
#
# Default push target: ${HF_USERNAME}/athena-cti-sft-<model>-v21-single-stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_KEY=""
REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
DRY_RUN=0
OFFLOAD="auto"
SKIP_EVAL=0
RESUME=0
RESUMABLE=0
GC_OVERRIDE="on"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-key)    MODEL_KEY="$2";    shift 2 ;;
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        --gc)           GC_OVERRIDE="$2";  shift 2 ;;
        --skip-eval)    SKIP_EVAL=1;       shift ;;
        --resume)       RESUME=1;          shift ;;
        --resumable)    RESUMABLE=1;       shift ;;
        -h|--help) sed -n '3,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${GC_OVERRIDE}" in on|off) ;; *) echo "--gc must be on|off" >&2; exit 1 ;; esac

OPTIM_FLAG=""
case "${MODEL_KEY}" in
    foundation_8b)
        BASE_MODEL="fdtn-ai/Foundation-Sec-8B-Instruct"; TEMPLATE="llama3"; BATCH=2
        SLUG="foundation-8b" ;;
    llama31_8b)
        BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"; TEMPLATE="llama3"; BATCH=2
        SLUG="llama31-8b" ;;
    qwen25_14b)
        BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"; TEMPLATE="qwen"; BATCH=2
        SLUG="qwen25-14b" ;;
    qwen3_30b_thinking)
        BASE_MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"; TEMPLATE="qwen3"; BATCH=1
        OPTIM_FLAG="--optim adamw_8bit"
        SLUG="qwen3-30b-a3b-thinking-2507" ;;
    *) echo "--model-key must be foundation_8b, llama31_8b, qwen25_14b or qwen3_30b_thinking" >&2; exit 1 ;;
esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-${SLUG}-v21-single-stage"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="${BASE_MODEL//\//_}"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_single_stage_${TIMESTAMP}"

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
        echo "       These must be the original files the v21 chain trained on. Copy them; do not rebuild them." >&2
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

# Grad-accum holds the effective batch at 16 whatever the GPU count.
EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
GA=$(( 16 / (BATCH * EFFECTIVE_GPUS) )); [[ ${GA} -lt 1 ]] && GA=1

GC_FLAG=""
[[ "${GC_OVERRIDE}" == "off" ]] && GC_FLAG="--disable_gradient_checkpointing True"

# --save_only_model True keeps checkpoints small but they cannot be resumed.
# --resumable writes the full ZeRO state and keeps one, so the disk peak is two
# checkpoints while the next is written. --resume implies it.
[[ ${RESUME} -eq 1 ]] && RESUMABLE=1
SAVE_ARGS="--save_total_limit 2 --save_only_model True"
[[ ${RESUMABLE} -eq 1 ]] && SAVE_ARGS="--save_total_limit 1 --save_only_model False"
EXTRA_BASE="--deepspeed ${DS_CONFIG} ${SAVE_ARGS} --enable_liger_kernel True ${OPTIM_FLAG} ${GC_FLAG}"
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

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )
RESUME_FLAG=(); [[ ${RESUME} -eq 1 ]] && RESUME_FLAG=( --resume )

echo "=== v21 matched single-pass control (${BASE_MODEL}): union of Stage 1-3 shards, one epoch ==="
echo "  datasets     : ${DATASETS}"
echo "  eval sets    : ${VAL_NAMES}$([[ ${SKIP_EVAL} -eq 1 ]] && echo '  (disabled by --skip-eval)')"
echo "  template     : ${TEMPLATE}   optimizer: ${OPTIM_FLAG:-default}"
echo "  lr / epochs  : 1e-05 / 1   cutoff 8192, packing on, batch ${BATCH} x accum ${GA} x ${EFFECTIVE_GPUS} GPUs = eff. batch $(( BATCH * GA * EFFECTIVE_GPUS ))"
echo "  cpu offload  : ${OFFLOAD}   grad-ckpt: $([[ -z "${GC_FLAG}" ]] && echo on || echo off)   resumable: ${RESUMABLE}"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo

bash "${SFT_DIR}/utils/run_train.sh" \
    --model "${BASE_MODEL}" \
    --dataset "${DATASETS}" --template "${TEMPLATE}" --finetuning full \
    --epochs 1 --lr 1e-05 --batch ${BATCH} --grad-accum ${GA} \
    --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
    --max-samples 240000 --report-to "${REPORT_TO}" \
    --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
    --extra "${EXTRA_COMMON}" \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
