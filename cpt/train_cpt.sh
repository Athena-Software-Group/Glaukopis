#!/bin/bash

# Launch Continued Pre-Training of Llama-3.1-8B (base, not Instruct) on a
# curated CTI corpus registered in SFT/data/dataset_info.json. Wraps
# LLaMA-Factory with --stage pt semantics: no chat template, packed raw
# text, 1 epoch by default.
#
# This is a parallel to SFT/autotrain/run_abaligned_sft_v4.sh but stays
# in the cpt/ directory because the corpus, hyperparameters, and post-
# training expectations differ from the SFT path.
#
# Usage:
#   ./cpt/train_cpt.sh --dataset cti_corpus_v1
#                      [--model meta-llama/Llama-3.1-8B]      # base by default
#                      [--finetuning lora|full]
#                      [--epochs 1] [--lr 1e-4]
#                      [--batch N] [--grad-accum N] [--cutoff 4096]
#                      [--repo-id USER/NAME] [--report-to wandb|none]
#                      [--dry-run] [--extra "..."]
#
# Hyperparameter rationale (see cpt/README.md):
#   - --stage pt  : LlamaFactory pretraining loop; no chat template applied
#   - --packing on: dense token coverage (CPT wants this)
#   - 1 epoch     : CPT overfits quickly on small curated corpora
#   - LoRA r=32   : higher than the v4 SFT r=16 because CPT updates a
#                   wider distribution than narrow instruction SFT
#   - LR 1e-4     : LoRA default; drop to 2e-5 for full-parameter
#
# After this run, optionally follow with a lightweight chat SFT
# (~1-2k rows) to restore instruction-following for the AthenaBench
# evaluators:
#   SFT/autotrain/run_abaligned_sft_v5.sh  # (to be written)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SFT_DIR="${REPO_ROOT}/SFT"

MODEL="meta-llama/Llama-3.1-8B"
DATASET=""
FINETUNING="lora"
EPOCHS="1"
LR="1e-4"
BATCH="4"
GRAD_ACCUM="8"
CUTOFF="4096"
SAVE_STEPS="200"
REPORT_TO="wandb"
REPO_ID=""
OUTPUT_DIR=""
EXTRA_USER=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --dataset)     DATASET="$2";     shift 2 ;;
        --finetuning)  FINETUNING="$2";  shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --lr)          LR="$2";          shift 2 ;;
        --batch)       BATCH="$2";       shift 2 ;;
        --grad-accum)  GRAD_ACCUM="$2";  shift 2 ;;
        --cutoff)      CUTOFF="$2";      shift 2 ;;
        --save-steps)  SAVE_STEPS="$2";  shift 2 ;;
        --report-to)   REPORT_TO="$2";   shift 2 ;;
        --repo-id)     REPO_ID="$2";     shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --extra)       EXTRA_USER="$2";  shift 2 ;;
        --dry-run)     DRY_RUN=1;        shift ;;
        -h|--help) sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${DATASET}" ]]; then
    echo "--dataset is required (e.g. cti_corpus_v1, registered via cpt/register_dataset.py)" >&2
    exit 1
fi

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" && -n "${HF_USERNAME:-}" ]]; then
    short="${MODEL##*/}"
    REPO_ID="${HF_USERNAME}/athena-cti-cpt-${short,,}-v1"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="${MODEL//\//_}"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/cpt-${FINETUNING}/train_${TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"

# CPT-specific LLaMA-Factory arguments. Key differences vs SFT/utils/run_train.sh:
#   --stage pt  (not sft)
#   --packing True  (SFT leaves this off; CPT wants it on)
#   no --template  (pretraining has no chat template)
#   no --do_eval / eval_strategy (validation on raw text is noisy)
BASE_ARGS=(
    --stage pt
    --do_train True
    --model_name_or_path "${MODEL}"
    --preprocessing_num_workers 16
    --finetuning_type "${FINETUNING}"
    --flash_attn auto
    --dataset_dir "${SFT_DIR}/data"
    --dataset "${DATASET}"
    --cutoff_len "${CUTOFF}"
    --learning_rate "${LR}"
    --num_train_epochs "${EPOCHS}"
    --per_device_train_batch_size "${BATCH}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --max_grad_norm 1.0
    --logging_steps 10
    --save_steps "${SAVE_STEPS}"
    --packing True
    --report_to "${REPORT_TO}"
    --output_dir "${OUTPUT_DIR}"
    --bf16 True
    --plot_loss True
    --trust_remote_code True
    --include_num_input_tokens_seen True
    --optim adamw_torch
    --save_only_model True
)

LORA_ARGS=()
if [[ "${FINETUNING}" == "lora" ]]; then
    LORA_ARGS=(--lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --lora_target all)
fi

# shellcheck disable=SC2206
EXTRA_ARR=( ${EXTRA_USER} )

echo "=== CTI CPT (LLaMA-Factory --stage pt) ==="
echo "  env        : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  model      : ${MODEL}   (expected: base, not -Instruct)"
echo "  dataset    : ${DATASET}"
echo "  finetuning : ${FINETUNING}"
echo "  epochs/lr  : ${EPOCHS} / ${LR}"
echo "  batch/accum: ${BATCH} / ${GRAD_ACCUM}  cutoff=${CUTOFF} packing=on"
echo "  output dir : ${OUTPUT_DIR}"
echo "  hf repo    : ${REPO_ID:-<none>}"
echo

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] command:"
    printf '  llamafactory-cli train'
    for a in "${BASE_ARGS[@]}" ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}; do
        printf ' %q' "${a}"
    done
    echo
    exit 0
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli not found. Run SFT/utils/setup.sh first and activate llm-sft." >&2
    exit 127
fi

export FORCE_TORCHRUN=1
{
    cd "${SFT_DIR}"
    set +e
    llamafactory-cli train "${BASE_ARGS[@]}" \
        ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
        ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}
    status=$?
    set -e
    echo
    echo "=== CPT finished ==="
    echo "  finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "  exit    : ${status}"
    echo "  model   : ${OUTPUT_DIR}"

    if [[ ${status} -eq 0 && -n "${REPO_ID}" ]]; then
        echo
        echo "=== HF push ==="
        PUSH_ARGS=(--adapter-dir "${OUTPUT_DIR}" --base-model "${MODEL}" --repo-id "${REPO_ID}")
        [[ "${FINETUNING}" == "full" ]] && PUSH_ARGS=(--merged-dir "${OUTPUT_DIR}" --repo-id "${REPO_ID}")
        python "${SFT_DIR}/upload_to_hf.py" "${PUSH_ARGS[@]}" || status=$?
    fi
    exit ${status}
} 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
