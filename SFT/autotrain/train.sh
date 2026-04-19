#!/bin/bash

# Launch an AutoTrain Advanced training run.
#
# Reads the chosen YAML config, verifies the env + credentials, and runs
# `autotrain --config <yaml>` in the foreground (or detached with --nohup),
# with stdout/stderr teed to <project_name>_<timestamp>.log in this directory.
#
# On success AutoTrain pushes the merged full-weight model to
# huggingface.co/<HF_USERNAME>/<project_name> automatically (via the YAML
# hub: section).
#
# Usage:
#   ./train.sh [--config PATH] [--cuda-devices LIST] [--nohup]
#              [--min-vram-gb N] [--skip-vram-check]
#
# Defaults:
#   --config          autotrain_llama3_8b_sft.yml  (full-parameter SFT)
#   --cuda-devices    (unset -> all visible GPUs)
#   --min-vram-gb     72    (only enforced for full-SFT configs; see below)
#
# Full SFT of an 8B model needs ~80 GB of aggregate bf16 VRAM. The pre-flight
# check refuses to launch if the selected GPUs fall below --min-vram-gb for
# any YAML with peft: false (override with --skip-vram-check if you know
# what you're doing). LoRA configs (peft: true) skip the check entirely.
#
# Required env vars (loaded automatically from ./.env if present):
#   HF_TOKEN, HF_USERNAME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/autotrain_llama3_8b_sft.yml"
CUDA_DEVICES=""
DETACH=0
MIN_VRAM_GB=72
SKIP_VRAM_CHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG="$2"; shift 2 ;;
        --cuda-devices)     CUDA_DEVICES="$2"; shift 2 ;;
        --nohup)            DETACH=1; shift ;;
        --min-vram-gb)      MIN_VRAM_GB="$2"; shift 2 ;;
        --skip-vram-check)  SKIP_VRAM_CHECK=1; shift ;;
        -h|--help) sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
fi

: "${HF_TOKEN:?Set HF_TOKEN in SFT/autotrain/.env (see .env.example)}"
: "${HF_USERNAME:?Set HF_USERNAME in SFT/autotrain/.env (see .env.example)}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[FAIL] config not found: ${CONFIG}" >&2
    exit 1
fi

if ! command -v autotrain >/dev/null 2>&1; then
    echo "[FAIL] autotrain CLI not on PATH. Activate the env first:" >&2
    echo "       conda activate autotrain" >&2
    exit 127
fi

# Extract project_name and training hyperparams for the summary block so the
# user can eyeball effective batch size and detect silent scaling mistakes.
read -r PROJECT_NAME BATCH_SIZE GRAD_ACCUM GRAD_CKPT PEFT_FLAG < <(python - "${CONFIG}" <<'PY'
import yaml, sys
with open(sys.argv[1]) as f: c = yaml.safe_load(f) or {}
p = c.get("params", {}) or {}
print(
    c.get("project_name", "autotrain-run"),
    int(p.get("batch_size", 1)),
    int(p.get("gradient_accumulation", 1)),
    bool(p.get("gradient_checkpointing", False)),
    bool(p.get("peft", False)),
)
PY
)

NUM_VISIBLE_GPUS=1
if [[ -n "${CUDA_DEVICES}" ]]; then
    NUM_VISIBLE_GPUS=$(awk -F',' '{print NF}' <<<"${CUDA_DEVICES}")
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_VISIBLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_VISIBLE_GPUS))

TIMESTAMP="$(date -u +"%Y%m%d-%H%M%SZ")"
LOG_FILE="${SCRIPT_DIR}/${PROJECT_NAME}_${TIMESTAMP}.log"

echo "=== AutoTrain run ==="
echo "  config         : ${CONFIG}"
echo "  project_name   : ${PROJECT_NAME}"
echo "  hub target     : ${HF_USERNAME}/${PROJECT_NAME}"
echo "  mode           : $([[ "${PEFT_FLAG}" == "True" ]] && echo 'LoRA (peft)' || echo 'full SFT')"
echo "  per-GPU batch  : ${BATCH_SIZE}"
echo "  grad accum     : ${GRAD_ACCUM}"
echo "  visible GPUs   : ${NUM_VISIBLE_GPUS}"
echo "  effective batch: ${EFFECTIVE_BATCH}   (= ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_VISIBLE_GPUS})"
echo "  grad ckpt      : ${GRAD_CKPT}"
echo "  log file       : ${LOG_FILE}"
echo "  cuda devices   : ${CUDA_DEVICES:-<all visible>}"
echo "  detach         : $([[ ${DETACH} -eq 1 ]] && echo yes || echo no)"
echo "  started        : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo

# Brief GPU snapshot (non-fatal if nvidia-smi missing)
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
        --format=csv,noheader | sed 's/^/  /'
    echo
fi

if [[ -n "${CUDA_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

# Pre-flight VRAM check: full-parameter SFT of 8B needs ~80 GB aggregate.
# Refuse to launch if the box is obviously too small, so the user finds out
# in one second instead of after minutes of tokenizer download + init only
# to crash with a cryptic CUDA OOM at step 0.
if [[ ${SKIP_VRAM_CHECK} -eq 0 && "${PEFT_FLAG}" == "False" ]] \
        && command -v nvidia-smi >/dev/null 2>&1; then
    TOTAL_VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
        | awk '{s+=$1} END {printf "%.0f", s/1024}')"
    if [[ -n "${TOTAL_VRAM_GB}" && "${TOTAL_VRAM_GB}" -lt "${MIN_VRAM_GB}" ]]; then
        echo "[FAIL] full-parameter SFT requires >= ${MIN_VRAM_GB} GB aggregate VRAM;" >&2
        echo "       detected only ${TOTAL_VRAM_GB} GB across visible GPUs." >&2
        echo >&2
        echo "Options:" >&2
        echo "  * move to a bigger box (A100-80G, H100-80G, or 2x A100-40G)" >&2
        echo "  * run the LoRA + int4 variant instead:" >&2
        echo "      ./train.sh --config ${SCRIPT_DIR}/autotrain_llama3_8b_lora.yml" >&2
        echo "  * override this check (you will OOM at step 0):" >&2
        echo "      ./train.sh --skip-vram-check" >&2
        exit 2
    fi
    echo "  vram check     : ${TOTAL_VRAM_GB} GB >= ${MIN_VRAM_GB} GB required for full SFT -> OK"
    echo
fi

CMD=(autotrain --config "${CONFIG}")

if [[ ${DETACH} -eq 1 ]]; then
    echo "=== Launching detached (nohup) ==="
    nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    pid=$!
    echo "  pid  : ${pid}"
    echo "  tail : tail -f ${LOG_FILE}"
    exit 0
fi

{
    "${CMD[@]}"
    status=$?
    echo
    echo "=== AutoTrain finished ==="
    echo "  finished : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "  exit     : ${status}"
    if [[ ${status} -eq 0 ]]; then
        echo "  model    : https://huggingface.co/${HF_USERNAME}/${PROJECT_NAME}"
        echo
        echo "Next: benchmark the pushed model with"
        echo "  ${SCRIPT_DIR}/run_athenabench.sh \\"
        echo "      --repo-id ${HF_USERNAME}/${PROJECT_NAME}"
    fi
    exit ${status}
} 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
