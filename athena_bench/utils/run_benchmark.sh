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
#                                   [--overwrite] [--yes]
#
# Flags:
#   --overwrite   Delete existing response files for the selected (tasks,
#                 rows, version, model) tuple before running, forcing a
#                 fresh run instead of resume-from-checkpoint.
#   --yes / -y    Skip the interactive confirmation prompt when --overwrite
#                 is set (required for nohup / non-interactive runs).
#
# Examples:
#   ./run_benchmark.sh deephat-7b
#   ./run_benchmark.sh deephat-7b --version 2
#   ./run_benchmark.sh deephat-7b --rows 100 --tasks "athena-mcq athena-rcm"
#   ./run_benchmark.sh deephat-7b --overwrite                 # interactive
#   ./run_benchmark.sh deephat-7b --overwrite --yes           # no prompt

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

MODEL_NAME="$1"; shift

VERSION=1
ROWS=""
TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-rms"
OVERWRITE=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)   VERSION="$2"; shift 2 ;;
        --rows)      ROWS="$2"; shift 2 ;;
        --tasks)     TASKS="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --yes|-y)    ASSUME_YES=1; shift ;;
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

# Resolve the model's on-disk response directory name via the same mapping
# inference.py uses (model_mapping[alias].replace('/', '_')). We parse
# pipelines/models.py as an AST so we avoid importing the module (which
# pulls in torch, dotenv, HF login, etc. and can fail in surprising ways).
# Falls back to the raw name if the alias is not in the mapping.
DISPLAY_NAME="$(cd "${BENCH_DIR}" && python - "${MODEL_NAME}" <<'PY'
import ast, pathlib, sys
name = sys.argv[1]
mapping = {}
try:
    src = pathlib.Path("pipelines/models.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "model_mapping":
                    mapping = ast.literal_eval(node.value)
except Exception:
    pass
print(mapping.get(name, name).replace("/", "_"))
PY
)"
ROWS_STR="${ROWS:-all}"

# Build the list of response files that inference.py would produce for this
# sweep. inference.py writes:
#   responses/<display_name>/<task>/<task>_<rows_str>_v<version>_<display_name>_response.jsonl
# We use this both for the --overwrite deletion list and for reporting.
declare -a TARGET_FILES=()
for task in ${TASKS}; do
    TARGET_FILES+=("${BENCH_DIR}/responses/${DISPLAY_NAME}/${task}/${task}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.jsonl")
done

# --overwrite: prompt the user (unless --yes), then delete any existing
# response files. Done *before* the tee block so the prompt reaches the real
# terminal; the resulting deletions are re-echoed inside the tee'd block so
# they appear in the log.
declare -a DELETED_FILES=()
declare -a SKIPPED_FILES=()
if [[ ${OVERWRITE} -eq 1 ]]; then
    existing=()
    for f in "${TARGET_FILES[@]}"; do
        [[ -e "$f" ]] && existing+=("$f")
    done

    if [[ ${#existing[@]} -eq 0 ]]; then
        echo "[overwrite] no pre-existing response files match this run; nothing to delete."
    else
        echo "[overwrite] the following response files will be DELETED before the sweep:"
        for f in "${existing[@]}"; do echo "  - ${f#${BENCH_DIR}/}"; done

        if [[ ${ASSUME_YES} -eq 1 ]]; then
            reply="y"
            echo "[overwrite] --yes given; proceeding without prompt."
        else
            # Require an interactive stdin. Under nohup/pipes/CI the caller
            # must pass --yes explicitly; otherwise we fail closed.
            if [[ ! -t 0 ]]; then
                echo "[overwrite] ERROR: --overwrite requires an interactive terminal or --yes." >&2
                exit 2
            fi
            printf "[overwrite] Proceed with deletion? [y/N] " >&2
            read -r reply || reply=""
        fi

        case "${reply}" in
            y|Y|yes|YES)
                for f in "${existing[@]}"; do
                    if rm -f -- "$f"; then
                        DELETED_FILES+=("$f")
                    else
                        SKIPPED_FILES+=("$f")
                    fi
                done
                ;;
            *)
                echo "[overwrite] aborted by user; exiting without running the sweep."
                exit 1
                ;;
        esac
    fi
fi

# Run everything inside a single block so we can tee both stdout and stderr
# to the log in one shot.
{
    echo "=== Athena benchmark sweep ==="
    echo "  model       : ${MODEL_NAME}"
    echo "  display name: ${DISPLAY_NAME}"
    echo "  safe name   : ${SAFE_NAME}"
    echo "  bench dir   : ${BENCH_DIR}"
    echo "  log file    : ${LOG_FILE}"
    echo "  python      : $(command -v python || echo '(none)')"
    echo "  env         : ${CONDA_DEFAULT_ENV:-<none>}"
    echo "  version     : ${VERSION}"
    echo "  rows        : ${ROWS_STR}"
    echo "  tasks       : ${TASKS}"
    echo "  overwrite   : $([[ ${OVERWRITE} -eq 1 ]] && echo yes || echo no)"
    echo "  started     : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo

    if [[ ${OVERWRITE} -eq 1 ]]; then
        if [[ ${#DELETED_FILES[@]} -gt 0 ]]; then
            echo "[overwrite] deleted ${#DELETED_FILES[@]} existing response file(s):"
            for f in "${DELETED_FILES[@]}"; do echo "  - ${f#${BENCH_DIR}/}"; done
        fi
        if [[ ${#SKIPPED_FILES[@]} -gt 0 ]]; then
            echo "[overwrite] failed to delete ${#SKIPPED_FILES[@]} file(s):"
            for f in "${SKIPPED_FILES[@]}"; do echo "  - ${f#${BENCH_DIR}/}"; done
        fi
        echo
    fi

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
