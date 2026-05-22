#!/bin/bash

# Train an SFT model and persist the result to a dedicated output directory.
#
# Wraps llamafactory-cli with the conventions used by the ift_training_*.sh
# scripts at the SFT repo root: LoRA by default, bf16, cosine schedule,
# wandb reporting optional. The key difference is that this launcher:
#   - always writes to a fresh output dir (timestamped, or --output-dir),
#     and refuses to overwrite an existing one unless --overwrite or
#     --resume is given (--resume keeps the dir and asks Trainer to pick
#     up from the latest checkpoint-N subdir);
#   - snapshots the effective flags + git sha to <output_dir>/train_config.json;
#   - tees stdout+stderr to <output_dir>/train.log.
#
# Usage:
#   ./run_train.sh [--model ID] [--dataset NAME] [--template NAME]
#                  [--finetuning lora|full] [--epochs N] [--lr FLOAT]
#                  [--batch N] [--grad-accum N] [--cutoff N]
#                  [--save-steps N] [--eval-steps N] [--max-samples N]
#                  [--packing true|false]
#                  [--output-dir DIR] [--report-to wandb|none|tensorboard]
#                  [--run-name NAME] [--wandb-project NAME]
#                  [--push-to-hf org/repo] [--hf-public] [--hf-export-dir DIR]
#                  [--extra "--flag value --flag2 value2"]
#                  [--overwrite] [--resume] [--dry-run]
#
# Defaults (tuned for a single H100 80GB; override as needed):
#   --model         meta-llama/Llama-3.1-8B-Instruct
#   --dataset       ift_data_2026_04_20
#   --template      llama3
#   --finetuning    lora
#   --epochs        2
#   --lr            1e-04
#   --batch         16
#   --grad-accum    2
#   --cutoff        2048
#   --save-steps    500
#   --eval-steps    <save-steps>
#   --packing       false
#   --max-samples   150000
#   --report-to     none
#   --wandb-project athena-cti-sft
#   --run-name      <auto: model-short_ft-tag_epN_lrX_bsY_TS>
#
# Post-training HF push (optional):
#   When --push-to-hf org/repo is set and training exits 0, the LoRA
#   adapter is merged with the base model via llamafactory-cli export
#   and uploaded to the specified repo by SFT/scripts/upload_to_hf.py. Requires
#   $HF_TOKEN or $HUGGINGFACE_TOKEN. Private by default; --hf-public
#   creates a public repo. Merged output lands in --hf-export-dir
#   (default: SFT/merged/<repo-basename>).
#
# Examples:
#   ./run_train.sh                                  # defaults, LoRA, fresh dir
#   ./run_train.sh --model Qwen/Qwen2.5-14B-Instruct --template qwen --batch 8 --grad-accum 4
#   ./run_train.sh --finetuning full --batch 1 --grad-accum 2 --lr 1e-05
#   ./run_train.sh --dry-run                        # print the CLI, don't run
#   ./run_train.sh --epochs 1 --lr 5e-5 --report-to wandb \
#                  --extra "--lora_rank 16 --lora_alpha 32 --lora_dropout 0.1" \
#                  --push-to-hf asg-ai/athena-cti-sft-llama31-8b

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
EVAL_STEPS=""
PACKING="false"
MAX_SAMPLES="150000"
REPORT_TO="none"
OUTPUT_DIR=""
EXTRA_ARGS=""
OVERWRITE=0
RESUME=0
DRY_RUN=0
RUN_NAME=""
WANDB_PROJECT_ARG=""
PUSH_TO_HF=""
HF_PUBLIC=0
HF_EXPORT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";             shift 2 ;;
        --dataset)       DATASET="$2";           shift 2 ;;
        --template)      TEMPLATE="$2";          shift 2 ;;
        --finetuning)    FINETUNING="$2";        shift 2 ;;
        --epochs)        EPOCHS="$2";            shift 2 ;;
        --lr)            LR="$2";                shift 2 ;;
        --batch)         BATCH="$2";             shift 2 ;;
        --grad-accum)    GRAD_ACCUM="$2";        shift 2 ;;
        --cutoff)        CUTOFF="$2";            shift 2 ;;
        --save-steps)    SAVE_STEPS="$2";        shift 2 ;;
        --eval-steps)    EVAL_STEPS="$2";        shift 2 ;;
        --packing)       PACKING="$2";           shift 2 ;;
        --max-samples)   MAX_SAMPLES="$2";       shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";        shift 2 ;;
        --report-to)     REPORT_TO="$2";         shift 2 ;;
        --run-name)      RUN_NAME="$2";          shift 2 ;;
        --wandb-project) WANDB_PROJECT_ARG="$2"; shift 2 ;;
        --push-to-hf)    PUSH_TO_HF="$2";        shift 2 ;;
        --hf-public)     HF_PUBLIC=1;            shift ;;
        --hf-export-dir) HF_EXPORT_DIR="$2";     shift 2 ;;
        --extra)         EXTRA_ARGS="$2";        shift 2 ;;
        --overwrite)     OVERWRITE=1;            shift ;;
        --resume)        RESUME=1;               shift ;;
        --dry-run)       DRY_RUN=1;              shift ;;
        -h|--help)
            sed -n '3,56p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${FINETUNING}" != "lora" && "${FINETUNING}" != "full" ]]; then
    echo "--finetuning must be 'lora' or 'full' (got '${FINETUNING}')" >&2
    exit 1
fi

PACKING_LC="$(printf '%s' "${PACKING}" | tr '[:upper:]' '[:lower:]')"
case "${PACKING_LC}" in
    true)  PACKING="True" ;;
    false) PACKING="False" ;;
    *) echo "--packing must be 'true' or 'false' (got '${PACKING}')" >&2; exit 1 ;;
esac

if [[ -z "${EVAL_STEPS}" ]]; then
    EVAL_STEPS="${SAVE_STEPS}"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="${MODEL//\//_}"

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/${FINETUNING}/train_${TIMESTAMP}"
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    if [[ ${RESUME} -eq 1 ]]; then
        echo "[resume] keeping existing ${OUTPUT_DIR} (will resume from latest checkpoint-* subdir)"
    elif [[ ${OVERWRITE} -ne 1 ]]; then
        echo "Output dir already exists: ${OUTPUT_DIR}" >&2
        echo "Pass --overwrite to remove it, --resume to continue from the latest checkpoint, or choose a different --output-dir." >&2
        exit 2
    else
        echo "[overwrite] removing existing ${OUTPUT_DIR}"
        rm -rf -- "${OUTPUT_DIR}"
    fi
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"
CONFIG_FILE="${OUTPUT_DIR}/train_config.json"

# LoRA vs full-parameter flags. LoRA adds four extra switches; full adds none.
# The four LoRA hyperparameters are env-var overridable so higher-level
# launchers (e.g. autotrain/run_abaligned_sft_v4.sh) can set them without
# duplicating the switches via --extra (which would leave both the default
# and the override on the llamafactory-cli command line and in
# train_config.json, making the effective value unclear to a later reader).
LORA_ARGS=()
LORA_RANK_DEFAULT="${LORA_RANK_DEFAULT:-64}"
LORA_ALPHA_DEFAULT="${LORA_ALPHA_DEFAULT:-128}"
LORA_DROPOUT_DEFAULT="${LORA_DROPOUT_DEFAULT:-0.05}"
LORA_TARGET_DEFAULT="${LORA_TARGET_DEFAULT:-all}"
if [[ "${FINETUNING}" == "lora" ]]; then
    LORA_ARGS=(
        --lora_rank "${LORA_RANK_DEFAULT}"
        --lora_alpha "${LORA_ALPHA_DEFAULT}"
        --lora_dropout "${LORA_DROPOUT_DEFAULT}"
        --lora_target "${LORA_TARGET_DEFAULT}"
    )
fi

# Build a wandb/trainer run name that encodes the headline hyperparameters,
# so runs are readable at a glance in the dashboard. User --run-name wins.
# Any --lora_rank / --lora_alpha override in --extra is reflected here too.
SHORT_MODEL="$(basename "${MODEL}")"
SHORT_MODEL="${SHORT_MODEL%-Instruct}"
SHORT_MODEL="${SHORT_MODEL%-instruct}"
SHORT_MODEL="${SHORT_MODEL%-Chat}"
EFFECTIVE_BATCH=$(( BATCH * GRAD_ACCUM ))
FT_TAG="${FINETUNING}"
if [[ "${FINETUNING}" == "lora" ]]; then
    EFF_RANK="${LORA_RANK_DEFAULT}"
    EFF_ALPHA="${LORA_ALPHA_DEFAULT}"
    if [[ "${EXTRA_ARGS}" =~ --lora_rank[[:space:]]+([0-9]+) ]]; then
        EFF_RANK="${BASH_REMATCH[1]}"
    fi
    if [[ "${EXTRA_ARGS}" =~ --lora_alpha[[:space:]]+([0-9]+) ]]; then
        EFF_ALPHA="${BASH_REMATCH[1]}"
    fi
    FT_TAG="lora-r${EFF_RANK}a${EFF_ALPHA}"
fi
AUTO_RUN_NAME="${SHORT_MODEL}_${FT_TAG}_ep${EPOCHS}_lr${LR}_bs${EFFECTIVE_BATCH}_${TIMESTAMP}"
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="${AUTO_RUN_NAME}"
fi

# wandb project: explicit --wandb-project > pre-set $WANDB_PROJECT > default
if [[ "${REPORT_TO}" == "wandb" ]]; then
    if [[ -n "${WANDB_PROJECT_ARG}" ]]; then
        export WANDB_PROJECT="${WANDB_PROJECT_ARG}"
    else
        export WANDB_PROJECT="${WANDB_PROJECT:-athena-cti-sft}"
    fi
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
    "eval_steps": int("${EVAL_STEPS}"),
    "packing": "${PACKING}".lower() == "true",
    "max_samples": int("${MAX_SAMPLES}"),
    "report_to": "${REPORT_TO}",
    "output_dir": "${OUTPUT_DIR}",
    "run_name": "${RUN_NAME}",
    "wandb_project": "${WANDB_PROJECT:-}",
    "push_to_hf": "${PUSH_TO_HF}",
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
    --packing "${PACKING}"
    --enable_thinking False
    --report_to "${REPORT_TO}"
    --run_name "${RUN_NAME}"
    --output_dir "${OUTPUT_DIR}"
    --bf16 True
    --plot_loss True
    --trust_remote_code True
    --ddp_timeout 18000
    --include_num_input_tokens_seen True
    --optim adamw_torch
    --val_size 0.1
    --eval_strategy steps
    --eval_steps "${EVAL_STEPS}"
    --per_device_eval_batch_size "${BATCH}"
    --overwrite_output_dir False
    --save_only_model False
)

# When --resume is set, resolve the newest checkpoint-N subdir under
# --output_dir and pass its absolute path to Transformers' Trainer.
# TrainingArguments.resume_from_checkpoint is typed Optional[str], so the
# CLI value is always taken as a literal path; passing "True" here is
# interpreted as the path "True/" and crashes with FileNotFoundError on
# trainer_state.json. Optimizer/scheduler state, RNG, dataloader position,
# and global_step are all restored from the resolved checkpoint dir.
if [[ ${RESUME} -eq 1 ]]; then
    latest_ckpt_n="$(ls -1d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
        | sed 's|.*/checkpoint-||' | sort -n | tail -1)"
    if [[ -z "${latest_ckpt_n}" ]]; then
        echo "[resume] no checkpoint-* subdir under ${OUTPUT_DIR}" >&2
        exit 3
    fi
    RESUME_DIR="${OUTPUT_DIR}/checkpoint-${latest_ckpt_n}"
    echo "[resume] resuming from ${RESUME_DIR}"
    BASE_ARGS+=( --resume_from_checkpoint "${RESUME_DIR}" )
fi

# shellcheck disable=SC2206
EXTRA_ARR=( ${EXTRA_ARGS} )

print_banner() {
    echo "=== SFT training run ==="
    echo "  model      : ${MODEL}"
    echo "  dataset    : ${DATASET} (template=${TEMPLATE})"
    echo "  finetuning : ${FINETUNING}"
    echo "  epochs/lr  : ${EPOCHS} / ${LR}"
    echo "  batch/accum: ${BATCH} / ${GRAD_ACCUM} (cutoff=${CUTOFF}, eff_batch=${EFFECTIVE_BATCH}, packing=${PACKING})"
    echo "  save/eval  : every ${SAVE_STEPS} / every ${EVAL_STEPS} steps"
    echo "  run name   : ${RUN_NAME}"
    echo "  wandb proj : ${WANDB_PROJECT:-<not set>}"
    echo "  push to hf : ${PUSH_TO_HF:-<none>}"
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
    if [[ -n "${PUSH_TO_HF}" ]]; then
        echo
        echo "[dry-run] post-training HF push:"
        if [[ "${FINETUNING}" == "lora" ]]; then
            echo "  python ${SFT_DIR}/scripts/upload_to_hf.py --adapter-dir ${OUTPUT_DIR} \\"
            echo "      --base-model ${MODEL} --template ${TEMPLATE} \\"
            echo "      --repo-id ${PUSH_TO_HF}$([[ ${HF_PUBLIC} -eq 1 ]] && echo ' --public')"
        else
            echo "  python ${SFT_DIR}/scripts/upload_to_hf.py --merged-dir ${OUTPUT_DIR} \\"
            echo "      --repo-id ${PUSH_TO_HF}$([[ ${HF_PUBLIC} -eq 1 ]] && echo ' --public')"
        fi
    fi
    exit 0
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli not found on PATH. Run utils/setup.sh first and activate the env." >&2
    exit 127
fi

# Blackwell Ultra (B300 / GB300, compute capability 10.3) emits PTX with the
# sm_103a target. CUDA 12.8 ptxas does not recognise it (added in 12.9), and
# torch 2.8+cu128 bundles Triton 3.4 which in turn bundles a CUDA 12.8 ptxas.
# Result: the first Triton JIT call (Liger RMSNorm, fused CE, etc.) dies with
#   ptxas fatal : Value 'sm_103a' is not defined for option 'gpu-name'
# Upstream tracking: triton-lang/triton#7964, pytorch/pytorch#163801.
# Fix is to point Triton at a CUDA 12.9+ ptxas via TRITON_PTXAS_PATH; the
# cubin produced loads fine via the B300 driver. No-op on every other arch
# (the cc check below short-circuits, so H100/H200/B200 are untouched).
maybe_export_blackwell_ptxas() {
    [[ -n "${TRITON_PTXAS_PATH:-}" ]] && return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d ' ')"
    [[ "${cc}" != "10.3" ]] && return 0
    echo "=== Blackwell B300 detected (cc=${cc}); locating CUDA 12.9+ ptxas for Triton ==="
    local triton_dir nvcc_pkg_dir
    triton_dir="$(python -c 'import os, triton; print(os.path.dirname(triton.__file__))' 2>/dev/null || true)"
    nvcc_pkg_dir="$(python -c 'import os, nvidia.cuda_nvcc as m; print(os.path.dirname(m.__file__))' 2>/dev/null || true)"
    local candidates=(
        "${nvcc_pkg_dir:+${nvcc_pkg_dir}/bin/ptxas}"
        "${triton_dir:+${triton_dir}/backends/nvidia/bin/ptxas-blackwell}"
        "/usr/local/cuda-13.0/bin/ptxas"
        "/usr/local/cuda-12.9/bin/ptxas"
        "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/ptxas}"
    )
    local p ver_major ver_minor
    for p in "${candidates[@]}"; do
        [[ -z "${p}" || ! -x "${p}" ]] && continue
        read -r ver_major ver_minor < <("${p}" --version 2>/dev/null \
            | grep -oE 'release [0-9]+\.[0-9]+' \
            | head -1 \
            | sed -E 's/release ([0-9]+)\.([0-9]+)/\1 \2/')
        [[ -z "${ver_major}" ]] && continue
        if (( ver_major > 12 )) || (( ver_major == 12 && ver_minor >= 9 )); then
            export TRITON_PTXAS_PATH="${p}"
            echo "  using TRITON_PTXAS_PATH=${p} (release ${ver_major}.${ver_minor})"
            return 0
        fi
    done
    echo "[FAIL] no CUDA 12.9+ ptxas found on this host. Triton JIT will die" >&2
    echo "       with 'sm_103a not defined' on the first Liger / fused kernel." >&2
    echo "       Fix (in this env, no system changes):" >&2
    echo "         pip install nvidia-cuda-nvcc-cu12" >&2
    echo "       then re-run; this script will auto-pick the new ptxas." >&2
    return 1
}

{
    print_banner
    cd "${SFT_DIR}"
    maybe_export_blackwell_ptxas || exit 1
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

    # Optional post-training step: merge LoRA + push to HF Hub. Runs only on
    # a clean training exit; a non-zero train status short-circuits the push.
    if [[ ${status} -eq 0 && -n "${PUSH_TO_HF}" ]]; then
        echo
        echo "=== HF push ==="
        echo "  repo     : ${PUSH_TO_HF} ($([[ ${HF_PUBLIC} -eq 1 ]] && echo public || echo private))"
        if [[ "${FINETUNING}" == "lora" ]]; then
            PUSH_ARGS=(
                --adapter-dir "${OUTPUT_DIR}"
                --base-model  "${MODEL}"
                --template    "${TEMPLATE}"
                --repo-id     "${PUSH_TO_HF}"
            )
            if [[ -n "${HF_EXPORT_DIR}" ]]; then
                PUSH_ARGS+=( --export-dir "${HF_EXPORT_DIR}" )
            fi
        else
            # Full-parameter SFT: the output dir is already a merged model
            # (no adapter), so skip llamafactory-cli export and upload the
            # directory as-is. --export-dir / --base-model / --template are
            # not applicable in this path.
            PUSH_ARGS=(
                --merged-dir "${OUTPUT_DIR}"
                --repo-id    "${PUSH_TO_HF}"
            )
        fi
        if [[ ${HF_PUBLIC} -eq 1 ]]; then
            PUSH_ARGS+=( --public )
        fi
        set +e
        python "${SFT_DIR}/scripts/upload_to_hf.py" "${PUSH_ARGS[@]}"
        push_status=$?
        set -e
        echo "  finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "  exit    : ${push_status}"
        if [[ ${push_status} -ne 0 ]]; then
            status=${push_status}
        fi
    elif [[ ${status} -ne 0 && -n "${PUSH_TO_HF}" ]]; then
        echo
        echo "[push-to-hf] skipped: training exited ${status}"
    fi

    exit ${status}
} 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
