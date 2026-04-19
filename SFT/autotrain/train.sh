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
#
# Defaults:
#   --config        autotrain_llama3_8b_sft.yml  (next to this script)
#   --cuda-devices  (unset -> all visible GPUs)
#
# Required env vars (same as prepare_dataset.sh):
#   HF_TOKEN, HF_USERNAME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/autotrain_llama3_8b_sft.yml"
CUDA_DEVICES=""
DETACH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2"; shift 2 ;;
        --cuda-devices)  CUDA_DEVICES="$2"; shift 2 ;;
        --nohup)         DETACH=1; shift ;;
        -h|--help) sed -n '3,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

PROJECT_NAME="$(python -c "
import yaml, sys
with open(sys.argv[1]) as f: print(yaml.safe_load(f).get('project_name','autotrain-run'))
" "${CONFIG}")"

TIMESTAMP="$(date -u +"%Y%m%d-%H%M%SZ")"
LOG_FILE="${SCRIPT_DIR}/${PROJECT_NAME}_${TIMESTAMP}.log"

echo "=== AutoTrain run ==="
echo "  config       : ${CONFIG}"
echo "  project_name : ${PROJECT_NAME}"
echo "  hub target   : ${HF_USERNAME}/${PROJECT_NAME}"
echo "  log file     : ${LOG_FILE}"
echo "  cuda devices : ${CUDA_DEVICES:-<all visible>}"
echo "  detach       : $([[ ${DETACH} -eq 1 ]] && echo yes || echo no)"
echo "  started      : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
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
