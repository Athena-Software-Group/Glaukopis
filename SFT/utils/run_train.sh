#!/bin/bash

# Train an SFT model from scratch (no resume) and persist the result to a
# dedicated output directory.
#
# Wraps llamafactory-cli with the conventions used by the ift_training_*.sh
# scripts at the SFT repo root: LoRA by default, bf16, cosine schedule,
# wandb reporting optional. The key difference is that this launcher:
#   - always writes to a fresh output dir (timestamped, or --output-dir),
#     and refuses to overwrite an existing one unless --overwrite is given;
#   - snapshots the effective flags + git sha to <output_dir>/train_config.json;
#   - tees stdout+stderr to <output_dir>/train.log.
#
# Usage:
#   ./run_train.sh [--model ID] [--dataset NAME] [--template NAME]
#                  [--finetuning lora|full] [--epochs N] [--lr FLOAT]
#                  [--batch N] [--grad-accum N] [--cutoff N]
#                  [--save-steps N] [--max-samples N]
#                  [--output-dir DIR] [--report-to wandb|none|tensorboard]
#                  [--extra "--flag value --flag2 value2"]
#                  [--overwrite] [--dry-run]
#
# Defaults (tuned for a single H100 80GB; override as needed):
#   --model        meta-llama/Llama-3.1-8B-Instruct
#   --dataset      ift_data_2026_04_20
#   --template     llama3
#   --finetuning   lora
#   --epochs       2
#   --lr           1e-04
#   --batch        16
#   --grad-accum   2
#   --cutoff       2048
#   --save-steps   500
#   --max-samples  150000
#   --report-to    none
#
# Examples:
#   ./run_train.sh                                  # defaults, LoRA, fresh dir
#   ./run_train.sh --model Qwen/Qwen2.5-14B-Instruct --template qwen --batch 8 --grad-accum 4
#   ./run_train.sh --finetuning full --batch 1 --grad-accum 2 --lr 1e-05
#   ./run_train.sh --dry-run                        # print the CLI, don't run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="ift_data_2026_04_20"
TEMPLATE="llama3"
FINETUNING="lora"
EPOCHS="2"
LR="1e-04"
BATCH="16"
GRAD_ACCUM="2"
CUTOFF="2048"
SAVE_STEPS="500"
MAX_SAMPLES="150000"
REPORT_TO="none"
OUTPUT_DIR=""
EXTRA_ARGS=""
OVERWRITE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --dataset)     DATASET="$2";     shift 2 ;;
        --template)    TEMPLATE="$2";    shift 2 ;;
        --finetuning)  FINETUNING="$2";  shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --lr)          LR="$2";          shift 2 ;;
        --batch)       BATCH="$2";       shift 2 ;;
        --grad-accum)  GRAD_ACCUM="$2";  shift 2 ;;
        --cutoff)      CUTOFF="$2";      shift 2 ;;
        --save-steps)  SAVE_STEPS="$2";  shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --report-to)   REPORT_TO="$2";   shift 2 ;;
        --extra)       EXTRA_ARGS="$2";  shift 2 ;;
        --overwrite)   OVERWRITE=1;      shift ;;
        --dry-run)     DRY_RUN=1;        shift ;;
        -h|--help)
            sed -n '3,41p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${FINETUNING}" != "lora" && "${FINETUNING}" != "full" ]]; then
    echo "--finetuning must be 'lora' or 'full' (got '${FINETUNING}')" >&2
    exit 1
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="${MODEL//\//_}"

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/${FINETUNING}/train_${TIMESTAMP}"
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    if [[ ${OVERWRITE} -ne 1 ]]; then
        echo "Output dir already exists: ${OUTPUT_DIR}" >&2
        echo "Pass --overwrite to remove it, or choose a different --output-dir." >&2
        exit 2
    fi
    echo "[overwrite] removing existing ${OUTPUT_DIR}"
    rm -rf -- "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"
CONFIG_FILE="${OUTPUT_DIR}/train_config.json"

# LoRA vs full-parameter flags. LoRA adds four extra switches; full adds none.
LORA_ARGS=()
if [[ "${FINETUNING}" == "lora" ]]; then
    LORA_ARGS=(
        --lora_rank 64
        --lora_alpha 128
        --lora_dropout 0.05
        --lora_target all
    )
fi

# Snapshot the effective configuration so the run is reproducible without
# having to grep the log.
GIT_SHA="$(git -C "${SFT_DIR}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "${SFT_DIR}" diff --quiet 2>/dev/null && echo clean || echo dirty)"
python - "${CONFIG_FILE}" <<PY
import json, os, sys
cfg = {
    "model_name_or_path": "${MODEL}",
    "dataset": "${DATASET}",
    "template": "${TEMPLATE}",
    "finetuning_type": "${FINETUNING}",
    "num_train_epochs": float("${EPOCHS}"),
    "learning_rate": float("${LR}"),
    "per_device_train_batch_size": int("${BATCH}"),
    "gradient_accumulation_steps": int("${GRAD_ACCUM}"),
    "cutoff_len": int("${CUTOFF}"),
    "save_steps": int("${SAVE_STEPS}"),
    "max_samples": int("${MAX_SAMPLES}"),
    "report_to": "${REPORT_TO}",
    "output_dir": "${OUTPUT_DIR}",
    "extra_args": "${EXTRA_ARGS}",
    "git_sha": "${GIT_SHA}",
    "git_status": "${GIT_DIRTY}",
    "started": "${TIMESTAMP}",
}
with open(sys.argv[1], "w") as f:
    json.dump(cfg, f, indent=2)
PY

BASE_ARGS=(
    --stage sft
    --do_train True
    --do_eval True
    --model_name_or_path "${MODEL}"
    --preprocessing_num_workers 16
    --finetuning_type "${FINETUNING}"
    --template "${TEMPLATE}"
    --flash_attn auto
    --dataset_dir "${SFT_DIR}/data"
    --dataset "${DATASET}"
    --cutoff_len "${CUTOFF}"
    --learning_rate "${LR}"
    --num_train_epochs "${EPOCHS}"
    --max_samples "${MAX_SAMPLES}"
    --per_device_train_batch_size "${BATCH}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --logging_steps 5
    --save_steps "${SAVE_STEPS}"
    --warmup_ratio 0.05
    --packing False
    --enable_thinking False
    --report_to "${REPORT_TO}"
    --output_dir "${OUTPUT_DIR}"
    --bf16 True
    --plot_loss True
    --trust_remote_code True
    --ddp_timeout 18000
    --include_num_input_tokens_seen True
    --optim adamw_torch
    --val_size 0.1
    --eval_strategy steps
    --eval_steps "${SAVE_STEPS}"
    --per_device_eval_batch_size "${BATCH}"
    --overwrite_output_dir False
    --save_only_model False
)

# shellcheck disable=SC2206
EXTRA_ARR=( ${EXTRA_ARGS} )

print_banner() {
    echo "=== SFT training run ==="
    echo "  model      : ${MODEL}"
    echo "  dataset    : ${DATASET} (template=${TEMPLATE})"
    echo "  finetuning : ${FINETUNING}"
    echo "  epochs/lr  : ${EPOCHS} / ${LR}"
    echo "  batch/accum: ${BATCH} / ${GRAD_ACCUM} (cutoff=${CUTOFF})"
    echo "  output dir : ${OUTPUT_DIR}"
    echo "  log file   : ${LOG_FILE}"
    echo "  config     : ${CONFIG_FILE}"
    echo "  git sha    : ${GIT_SHA} (${GIT_DIRTY})"
    echo "  started    : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo
}

if [[ ${DRY_RUN} -eq 1 ]]; then
    print_banner
    echo "[dry-run] command:"
    printf '  llamafactory-cli train'
    for a in "${BASE_ARGS[@]}" ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}; do
        printf ' %q' "${a}"
    done
    echo
    exit 0
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli not found on PATH. Run utils/setup.sh first and activate the env." >&2
    exit 127
fi

{
    print_banner
    cd "${SFT_DIR}"
    set +e
    llamafactory-cli train "${BASE_ARGS[@]}" \
        ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
        ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}
    status=$?
    set -e
    echo
    echo "=== Training finished ==="
    echo "  finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "  exit    : ${status}"
    echo "  model   : ${OUTPUT_DIR}"
    exit ${status}
} 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
