#!/bin/bash

# Run a benchmark sweep for a single model across one or more suites.
#
# Supported suites (selected via --suite, default = athena):
#   athena      athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-rms
#   ctibench    mcq rcm vsp ate taa           (CTI-Bench, .tsv responses)
#   cybermetric cybermetric                   (CyberMetric MCQ, .csv responses;
#                                              size selected via --cybermetric-size)
#   all         athena U ctibench U cybermetric
#
# --tasks still works and overrides the suite-derived task list. Each task
# is launched as its own inference.py subprocess so VRAM is freed at process
# exit before the next task starts.
#
# NOTE: inference.py's --cleanup flag is NOT passed here. Despite the name,
# --cleanup evicts the HuggingFace model from VRAM after *every single row*
# and forces a full reload on the next row (~8-10s of wasted disk I/O per
# question). It is only useful on severely memory-starved RunPod setups
# where the model cannot stay resident between rows. For the sweep, we
# rely on per-task subprocess exit to free VRAM.
#
# All stdout/stderr is tee'd to <model-name>.log in this directory.
#
# Usage:
#   ./run_benchmark.sh <model-name> [--suite athena|ctibench|cybermetric|all]
#                                   [--version N] [--rows N]
#                                   [--tasks "mcq rcm vsp"]
#                                   [--cybermetric-size 80|500|2000|10000]
#                                   [--batch N] [--overwrite] [--yes]
#
# Flags:
#   --suite NAME  Preset task list. Default: athena. Ignored when --tasks set.
#   --cybermetric-size N
#                 Which CyberMetric-<N>-v1.json to evaluate on (default 80).
#   --batch N     Run N concurrent requests per task. Only supported for
#                 GPT/Gemini and HF Inference ('*-hf') models. Use 16-64
#                 for hosted-API runs to get real throughput.
#   --overwrite   Delete existing response files for the selected (tasks,
#                 rows, version, model) tuple before running, forcing a
#                 fresh run instead of resume-from-checkpoint.
#   --yes / -y    Skip the interactive confirmation prompt when --overwrite
#                 is set (required for nohup / non-interactive runs).
#   --single-gpu [IDX]
#                 Pin inference to a single CUDA device (default idx=0) by
#                 exporting CUDA_VISIBLE_DEVICES=IDX before launching each
#                 task. For an 8B model this removes cross-GPU PCIe hops
#                 that device_map="auto" introduces when multiple GPUs are
#                 visible, typically 1.5-2x faster than a 2-GPU split.
#
# Examples:
#   ./run_benchmark.sh deephat-7b                              # athena suite
#   ./run_benchmark.sh deephat-7b --suite ctibench
#   ./run_benchmark.sh deephat-7b --suite cybermetric --cybermetric-size 500
#   ./run_benchmark.sh deephat-7b --suite all --version 2
#   ./run_benchmark.sh deephat-7b --rows 100 --tasks "athena-mcq athena-rcm"
#   ./run_benchmark.sh deephat-7b --overwrite                 # interactive
#   ./run_benchmark.sh deephat-7b --overwrite --yes           # no prompt
#   ./run_benchmark.sh deepseek-r1-14b-hf --batch 32          # hosted + parallel

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
BATCH=""
SUITE="athena"
USER_TASKS=""
CYBERMETRIC_SIZE="80"
OVERWRITE=0
ASSUME_YES=0
SINGLE_GPU=0
SINGLE_GPU_IDX="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)   VERSION="$2"; shift 2 ;;
        --rows)      ROWS="$2"; shift 2 ;;
        --batch)     BATCH="$2"; shift 2 ;;
        --suite)     SUITE="$2"; shift 2 ;;
        --tasks)     USER_TASKS="$2"; shift 2 ;;
        --cybermetric-size) CYBERMETRIC_SIZE="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --yes|-y)    ASSUME_YES=1; shift ;;
        --single-gpu)
            SINGLE_GPU=1
            # Optional numeric index follows --single-gpu. Accept 0-9 only;
            # anything else is treated as the next flag.
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                SINGLE_GPU_IDX="$2"; shift 2
            else
                shift
            fi
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Suite -> task-list preset. --tasks always wins. 'all' is the concatenation
# of the three research-facing suites; MMLU/GLUE/SuperGLUE/URLHAUS/CVE stay
# out of the sweep because they're not the CTI research target.
case "${SUITE}" in
    athena)      SUITE_TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-rms" ;;
    ctibench)    SUITE_TASKS="mcq rcm vsp ate taa" ;;
    cybermetric) SUITE_TASKS="cybermetric" ;;
    all)         SUITE_TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-rms mcq rcm vsp ate taa cybermetric" ;;
    *) echo "Unknown --suite: ${SUITE} (expected athena|ctibench|cybermetric|all)" >&2; exit 1 ;;
esac
if [[ -n "${USER_TASKS}" ]]; then
    TASKS="${USER_TASKS}"
else
    TASKS="${SUITE_TASKS}"
fi

CYBERMETRIC_STEM="CyberMetric-${CYBERMETRIC_SIZE}-v1"
CYBERMETRIC_DATA_PATH="benchmark_data/cybermetricdataset/${CYBERMETRIC_STEM}.json"
if [[ "${TASKS}" == *"cybermetric"* ]]; then
    if [[ ! -f "${BENCH_DIR}/${CYBERMETRIC_DATA_PATH}" ]]; then
        echo "CyberMetric data file not found: ${CYBERMETRIC_DATA_PATH}" >&2
        echo "Available sizes under benchmark_data/cybermetricdataset/:" >&2
        ls "${BENCH_DIR}/benchmark_data/cybermetricdataset/" 2>/dev/null >&2 || true
        exit 1
    fi
fi

# Sanitize model name for use as a filename (e.g. "meta-llama/Llama-3" -> "meta-llama_Llama-3")
SAFE_NAME="${MODEL_NAME//\//_}"
LOG_FILE="${SCRIPT_DIR}/${SAFE_NAME}.log"

# NOTE: intentionally NOT passing --cleanup here (see header comment).
extra_args=(--version "${VERSION}")
if [[ -n "${ROWS}" ]]; then
    extra_args+=(--rows "${ROWS}")
fi
if [[ -n "${BATCH}" ]]; then
    extra_args+=(--batch "${BATCH}")
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

# Build the list of response files inference.py would produce. The filename
# pattern and extension vary by task family:
#   athena-*    -> .jsonl
#   CTI-Bench   -> .tsv  (mcq, rcm, vsp, ate, taa)
#   cybermetric -> .csv  (includes the CyberMetric-<N>-v1 stem in the name)
# resolve_resp_file echoes the expected absolute path for a given task (or
# the empty string for tasks with no fixed pattern, e.g. glue/superglue).
resolve_resp_file() {
    local task="$1"
    local base="${BENCH_DIR}/responses/${DISPLAY_NAME}/${task}"
    case "${task}" in
        athena-*)
            echo "${base}/${task}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.jsonl" ;;
        mcq|rcm|vsp|ate|taa)
            echo "${base}/${task}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.tsv" ;;
        cybermetric)
            echo "${base}/${task}_${CYBERMETRIC_STEM}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.csv" ;;
        *)
            echo "" ;;
    esac
}

# Count rows in a response file, excluding a header line for tsv/csv.
# Echoes 0 when the file is missing or empty.
count_resp_rows() {
    local f="$1"
    [[ -f "$f" ]] || { echo 0; return; }
    local total
    total=$(wc -l < "$f" | tr -d ' ')
    case "$f" in
        *.jsonl) echo "${total}" ;;
        *.tsv|*.csv)
            if [[ "${total}" -gt 0 ]]; then echo $(( total - 1 )); else echo 0; fi ;;
        *) echo "${total}" ;;
    esac
}

declare -a TARGET_FILES=()
for task in ${TASKS}; do
    rf="$(resolve_resp_file "${task}")"
    [[ -n "${rf}" ]] && TARGET_FILES+=("${rf}")
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
    echo "  suite       : ${SUITE}"
    echo "  version     : ${VERSION}"
    echo "  rows        : ${ROWS_STR}"
    echo "  batch       : ${BATCH:-<none>}"
    echo "  tasks       : ${TASKS}"
    if [[ "${TASKS}" == *"cybermetric"* ]]; then
        echo "  cybermetric : ${CYBERMETRIC_STEM} (${CYBERMETRIC_DATA_PATH})"
    fi
    if [[ ${SINGLE_GPU} -eq 1 ]]; then
        echo "  single-gpu  : yes (CUDA_VISIBLE_DEVICES=${SINGLE_GPU_IDX})"
    else
        echo "  single-gpu  : no (inherits CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
    fi
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

    # Per-task results collected during the sweep, emitted as a table + JSON
    # summary at the end.
    declare -a RES_TASKS=()
    declare -a RES_ELAPSED=()
    declare -a RES_EXIT=()
    declare -a RES_METRICS=()
    declare -a RES_ROWS=()
    declare -a RES_STARTED=()
    declare -a RES_FINISHED=()

    sweep_start_epoch=$(date +%s)
    sweep_start_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    overall_status=0
    for task in ${TASKS}; do
        echo
        echo "----- task: ${task} -----"
        task_started_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "  started : ${task_started_iso}"
        task_start=$(date +%s)

        # Task-specific extras (mainly cybermetric which needs an explicit
        # --data_path when the user picks a non-default size).
        task_extra=()
        if [[ "${task}" == "cybermetric" ]]; then
            task_extra+=(--data_path "${CYBERMETRIC_DATA_PATH}")
        fi

        # Capture the task's stdout+stderr in a temp file so we can parse the
        # "Evaluation result for ... : {...}" line while still streaming output
        # through the outer tee unchanged.
        task_out_file="$(mktemp -t athena_task.XXXXXX)"
        set +e
        set -o pipefail
        # Pin to a single GPU if --single-gpu was requested. Scoped to this
        # subshell so the parent script's environment is untouched.
        if [[ ${SINGLE_GPU} -eq 1 ]]; then
            (
                export CUDA_VISIBLE_DEVICES="${SINGLE_GPU_IDX}"
                python inference.py "${task}" "${MODEL_NAME}" "${extra_args[@]}" "${task_extra[@]}" 2>&1
            ) | tee "${task_out_file}"
        else
            python inference.py "${task}" "${MODEL_NAME}" "${extra_args[@]}" "${task_extra[@]}" 2>&1 \
                | tee "${task_out_file}"
        fi
        task_status=${PIPESTATUS[0]}
        set +o pipefail
        set -e

        task_end=$(date +%s)
        elapsed=$(( task_end - task_start ))
        task_finished_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "  finished: ${task_finished_iso} (elapsed ${elapsed}s, exit ${task_status})"

        # Extract the metrics dict printed by inference.py. Line looks like:
        #   Evaluation result for athena-mcq with deepseek-v3.2-exp-hf: {'accuracy': '78.42%'}
        metrics_raw="$(grep -E "^Evaluation result for ${task} with " "${task_out_file}" | tail -1 | sed -E "s/^Evaluation result for ${task} with [^:]+: //" || true)"
        rm -f "${task_out_file}"

        # Count rows actually written (evaluator's authoritative input file).
        resp_file="$(resolve_resp_file "${task}")"
        row_count="$(count_resp_rows "${resp_file}")"

        RES_TASKS+=("${task}")
        RES_ELAPSED+=("${elapsed}")
        RES_EXIT+=("${task_status}")
        RES_METRICS+=("${metrics_raw}")
        RES_ROWS+=("${row_count}")
        RES_STARTED+=("${task_started_iso}")
        RES_FINISHED+=("${task_finished_iso}")

        if [[ ${task_status} -ne 0 ]]; then
            overall_status=${task_status}
            echo "  [WARN] task '${task}' exited non-zero; continuing with remaining tasks"
        fi
    done

    sweep_end_epoch=$(date +%s)
    sweep_end_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    sweep_elapsed=$(( sweep_end_epoch - sweep_start_epoch ))

    echo
    echo "=== Sweep complete ==="
    echo "  finished: ${sweep_end_iso}"
    echo "  exit    : ${overall_status}"

    # Summary artifacts: pretty table to stdout (tee'd to log) + JSON/MD
    # dropped next to the response files. Handed off to Python so we can
    # literal-eval the metrics dicts and format percentages consistently.
    # Summary filename is namespaced by suite so running multiple suites
    # against the same model keeps one artifact per suite.
    summary_dir="${BENCH_DIR}/responses/${DISPLAY_NAME}"
    mkdir -p "${summary_dir}"
    summary_json="${summary_dir}/summary_${SUITE}_${ROWS_STR}_v${VERSION}.json"
    summary_md="${summary_dir}/summary_${SUITE}_${ROWS_STR}_v${VERSION}.md"

    # Hand data to Python via environment (bash arrays -> newline-joined strings).
    export RB_MODEL="${MODEL_NAME}"
    export RB_DISPLAY="${DISPLAY_NAME}"
    export RB_SUITE="${SUITE}"
    export RB_VERSION="${VERSION}"
    export RB_ROWS_STR="${ROWS_STR}"
    export RB_BATCH="${BATCH:-}"
    export RB_TASKS_REQUESTED="${TASKS}"
    export RB_CYBERMETRIC_STEM="${CYBERMETRIC_STEM}"
    export RB_STARTED="${sweep_start_iso}"
    export RB_FINISHED="${sweep_end_iso}"
    export RB_ELAPSED="${sweep_elapsed}"
    export RB_OVERALL_EXIT="${overall_status}"
    export RB_SUMMARY_JSON="${summary_json}"
    export RB_SUMMARY_MD="${summary_md}"
    export RB_LOG_FILE="${LOG_FILE}"
    export RB_ENV_NAME="${CONDA_DEFAULT_ENV:-}"
    # Join arrays with '\x1f' (ASCII unit separator) to avoid collisions with
    # quotes / braces inside metrics dicts.
    _join_us() { local IFS=$'\x1f'; echo -n "$*"; }
    export RB_RES_TASKS="$(_join_us "${RES_TASKS[@]:-}")"
    export RB_RES_ELAPSED="$(_join_us "${RES_ELAPSED[@]:-}")"
    export RB_RES_EXIT="$(_join_us "${RES_EXIT[@]:-}")"
    export RB_RES_METRICS="$(_join_us "${RES_METRICS[@]:-}")"
    export RB_RES_ROWS="$(_join_us "${RES_ROWS[@]:-}")"
    export RB_RES_STARTED="$(_join_us "${RES_STARTED[@]:-}")"
    export RB_RES_FINISHED="$(_join_us "${RES_FINISHED[@]:-}")"

    echo
    python "${SCRIPT_DIR}/_print_sweep_summary.py" || echo "[WARN] summary generation failed (non-fatal)"

    exit ${overall_status}
} 2>&1 | tee "${LOG_FILE}"

# Propagate the sweep's exit code (tee always exits 0 otherwise)
exit "${PIPESTATUS[0]}"
