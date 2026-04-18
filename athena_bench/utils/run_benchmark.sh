#!/bin/bash

# Run the full Athena CTI benchmark sweep for a single model.
#
# Executes all six Athena tasks (mcq, rcm, vsp, ate, taa, rms) using each
# benchmark class's default --data_path (i.e. the full-size files under
# benchmark_data/athena_bench/), with --cleanup between tasks so a single
# GPU can handle the whole sweep sequentially.
#
# All stdout/stderr is tee'd to <model-name>.log in this directory.
#
# Usage:
#   ./run_benchmark.sh <model-name> [--version N] [--rows N] [--tasks "mcq rcm vsp"]
#
# Examples:
#   ./run_benchmark.sh deephat-7b
#   ./run_benchmark.sh deephat-7b --version 2
#   ./run_benchmark.sh deephat-7b --rows 100 --tasks "athena-mcq athena-rcm"

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

MODEL_NAME="$1"; shift

VERSION=1
ROWS=""
TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-rms"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --rows)    ROWS="$2"; shift 2 ;;
        --tasks)   TASKS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Sanitize model name for use as a filename (e.g. "meta-llama/Llama-3" -> "meta-llama_Llama-3")
SAFE_NAME="${MODEL_NAME//\//_}"
LOG_FILE="${SCRIPT_DIR}/${SAFE_NAME}.log"

extra_args=(--version "${VERSION}" --cleanup)
if [[ -n "${ROWS}" ]]; then
    extra_args+=(--rows "${ROWS}")
fi

# Run everything inside a single block so we can tee both stdout and stderr
# to the log in one shot.
{
    echo "=== Athena benchmark sweep ==="
    echo "  model     : ${MODEL_NAME}"
    echo "  safe name : ${SAFE_NAME}"
    echo "  bench dir : ${BENCH_DIR}"
    echo "  log file  : ${LOG_FILE}"
    echo "  python    : $(command -v python || echo '(none)')"
    echo "  env       : ${CONDA_DEFAULT_ENV:-<none>}"
    echo "  version   : ${VERSION}"
    echo "  rows      : ${ROWS:-all}"
    echo "  tasks     : ${TASKS}"
    echo "  started   : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo

    cd "${BENCH_DIR}" || { echo "[FAIL] cannot cd to ${BENCH_DIR}"; exit 1; }

    overall_status=0
    for task in ${TASKS}; do
        echo
        echo "----- task: ${task} -----"
        echo "  started : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        task_start=$(date +%s)

        set +e
        python inference.py "${task}" "${MODEL_NAME}" "${extra_args[@]}"
        task_status=$?
        set -e

        task_end=$(date +%s)
        elapsed=$(( task_end - task_start ))
        echo "  finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ") (elapsed ${elapsed}s, exit ${task_status})"

        if [[ ${task_status} -ne 0 ]]; then
            overall_status=${task_status}
            echo "  [WARN] task '${task}' exited non-zero; continuing with remaining tasks"
        fi
    done

    echo
    echo "=== Sweep complete ==="
    echo "  finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "  exit    : ${overall_status}"
    exit ${overall_status}
} 2>&1 | tee "${LOG_FILE}"

# Propagate the sweep's exit code (tee always exits 0 otherwise)
exit "${PIPESTATUS[0]}"
