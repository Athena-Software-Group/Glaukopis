#!/bin/bash

# Run a benchmark sweep for a single model across one or more suites.
#
# Supported suites (selected via --suite, default = athena):
#   athena       athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-taa-canonical athena-rms
#   cybermetric  cybermetric                   (CyberMetric MCQ, .csv responses;
#                                               size selected via --cybermetric-size)
#   cybersoceval cybersoceval-malware cybersoceval-ti
#                                              (CrowdStrike+Meta CyberSOCEval, .jsonl
#                                               responses; data fetched once via
#                                               utils/fetch_cybersoceval_data.py)
#   mmlu-pro     mmlu-pro                      (TIGER-Lab MMLU-Pro 12K, reasoning
#                                               benchmark; not in --suite all because
#                                               not the CTI research target. Opt in
#                                               explicitly via --suite mmlu-pro.)
#   all          athena U cybermetric U cybersoceval
#
#   ctibench     mcq rcm vsp ate taa           (CTI-Bench, .tsv responses)
#                DEPRECATED: superseded by AthenaBench (athena-*). Still
#                runnable explicitly via --suite ctibench for legacy
#                reproductions; excluded from --suite all and from every
#                sweep wrapper (run_foundation_8b_baselines.sh,
#                run_api_baselines.sh) to avoid double-counting against
#                the AthenaBench successor tasks.
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
#   ./run_benchmark.sh <model-name> [--suite athena|cybermetric|cybersoceval|mmlu-pro|all|ctibench]
#                                   [--version N] [--rows N]
#                                   [--tasks "mcq rcm vsp"]
#                                   [--cybermetric-size 80|500|2000|10000|N1,N2,...]
#                                   [--batch N] [--overwrite] [--yes]
#                                   [--reasoning-effort none|low|medium|high|xhigh]
#
# Flags:
#   --suite NAME  Preset task list. Default: athena. Ignored when --tasks set.
#   --cybermetric-size N[,N...]
#                 Which CyberMetric-<N>-v1.json files to evaluate on (default 80).
#                 Accepts a comma-separated list to run multiple sizes against
#                 the same served model in one sweep, e.g. --cybermetric-size 2000,10000
#                 expands the 'cybermetric' task into back-to-back runs whose
#                 results are recorded under labels 'cybermetric-2000' and
#                 'cybermetric-10000' in the summary.
#   --batch N     Run N concurrent requests per task. Supported for
#                 GPT/Gemini, HF Inference ('*-hf'), and local vLLM
#                 ('*-vllm') models. Use 16-64 for hosted-API runs and
#                 32-128 for a local vLLM server on a single H100.
#   --overwrite   Delete existing response files for the selected (tasks,
#                 rows, version, model) tuple before running, forcing a
#                 fresh run instead of resume-from-checkpoint.
#   --retry-errors
#                 Resume mode: keep existing rows but scrub any row whose
#                 response is an error sentinel ("Error", "Error: ...",
#                 or empty raw_response for cybermetric) so the per-bench
#                 resume logic re-processes only those rows on the next
#                 run. Mutually exclusive with --overwrite.
#   --yes / -y    Skip the interactive confirmation prompt when --overwrite
#                 or --retry-errors is set (required for nohup /
#                 non-interactive runs).
#   --reasoning-effort EFFORT
#                 Pass --reasoning_effort EFFORT to inference.py. Honored by
#                 the OpenAI responses-API reasoning family (gpt5.2, gpt5.5,
#                 gpt5.5-pro); inference.py rewrites the response folder to
#                 '<display>-<effort>' when set, so we mirror that suffix in
#                 DISPLAY_NAME below to keep --overwrite, resume, and summary
#                 paths consistent.
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
#   ./run_benchmark.sh athena-cti-cpt-llama31-8b-v1-vllm --batch 64
#                                                             # local vLLM + parallel

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
RETRY_ERRORS=0
ASSUME_YES=0
SINGLE_GPU=0
SINGLE_GPU_IDX="0"
REASONING_EFFORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)   VERSION="$2"; shift 2 ;;
        --rows)      ROWS="$2"; shift 2 ;;
        --batch)     BATCH="$2"; shift 2 ;;
        --suite)     SUITE="$2"; shift 2 ;;
        --tasks)     USER_TASKS="$2"; shift 2 ;;
        --cybermetric-size) CYBERMETRIC_SIZE="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --retry-errors) RETRY_ERRORS=1; shift ;;
        --yes|-y)    ASSUME_YES=1; shift ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
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

if [[ ${OVERWRITE} -eq 1 && ${RETRY_ERRORS} -eq 1 ]]; then
    echo "ERROR: --overwrite and --retry-errors are mutually exclusive." >&2
    exit 2
fi

# Suite -> task-list preset. --tasks always wins. 'all' is the concatenation
# of the three research-facing suites that are not deprecated; CTI-Bench is
# excluded because its tasks (mcq/rcm/vsp/ate/taa) have been superseded by
# the AthenaBench equivalents (athena-*) and double-counting them inflates
# headline-metric averages with strongly correlated scores.
# MMLU/GLUE/SuperGLUE/URLHAUS/CVE also stay out: not the CTI research target.
case "${SUITE}" in
    athena)       SUITE_TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-taa-canonical athena-rms" ;;
    ctibench)
        # Deprecated; still runnable for legacy reproductions.
        echo "[deprecated] --suite ctibench: superseded by --suite athena (AthenaBench)." >&2
        echo "             Running anyway for legacy reproduction; not included in --suite all." >&2
        SUITE_TASKS="mcq rcm vsp ate taa"
        ;;
    cybermetric)  SUITE_TASKS="cybermetric" ;;
    cybersoceval) SUITE_TASKS="cybersoceval-malware cybersoceval-ti" ;;
    mmlu-pro)     SUITE_TASKS="mmlu-pro" ;;
    all)          SUITE_TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-taa-canonical athena-rms cybermetric cybersoceval-malware cybersoceval-ti" ;;
    *) echo "Unknown --suite: ${SUITE} (expected athena|cybermetric|cybersoceval|mmlu-pro|all|ctibench[deprecated])" >&2; exit 1 ;;
esac
if [[ -n "${USER_TASKS}" ]]; then
    TASKS="${USER_TASKS}"
else
    TASKS="${SUITE_TASKS}"
fi

# --cybermetric-size accepts a comma-separated list (e.g. "2000,10000") so
# a single sweep can score the model on multiple CyberMetric splits without
# re-launching vllm. The arrays below are kept index-aligned: CYBERMETRIC_SIZES[i]
# <-> CYBERMETRIC_STEMS[i] <-> CYBERMETRIC_DATA_PATHS[i]. CYBERMETRIC_STEM /
# CYBERMETRIC_DATA_PATH retain their pre-multi-size meaning (the first size in
# the list) for callers that read them directly (the banner, the legacy summary
# stem path).
IFS=',' read -ra CYBERMETRIC_SIZES <<< "${CYBERMETRIC_SIZE}"
declare -a CYBERMETRIC_STEMS=()
declare -a CYBERMETRIC_DATA_PATHS=()
for _sz in "${CYBERMETRIC_SIZES[@]}"; do
    CYBERMETRIC_STEMS+=("CyberMetric-${_sz}-v1")
    CYBERMETRIC_DATA_PATHS+=("benchmark_data/cybermetricdataset/CyberMetric-${_sz}-v1.json")
done
unset _sz
CYBERMETRIC_STEM="${CYBERMETRIC_STEMS[0]}"
CYBERMETRIC_DATA_PATH="${CYBERMETRIC_DATA_PATHS[0]}"
if [[ "${TASKS}" == *"cybermetric"* ]]; then
    for _path in "${CYBERMETRIC_DATA_PATHS[@]}"; do
        if [[ ! -f "${BENCH_DIR}/${_path}" ]]; then
            echo "CyberMetric data file not found: ${_path}" >&2
            echo "Available sizes under benchmark_data/cybermetricdataset/:" >&2
            ls "${BENCH_DIR}/benchmark_data/cybermetricdataset/" 2>/dev/null >&2 || true
            exit 1
        fi
    done
    unset _path
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
if [[ -n "${REASONING_EFFORT}" ]]; then
    extra_args+=(--reasoning_effort "${REASONING_EFFORT}")
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
# inference.py rewrites the response folder to '<base>-<effort>' when a
# reasoning effort is set on a model that supports it (the OpenAI responses-API
# reasoning family: gpt5.2, gpt5.5, gpt5.5-pro). Mirror the same suffix here so
# resolve_resp_file/--overwrite/summary paths line up with what inference.py
# actually writes.
case "${MODEL_NAME}" in
    gpt5.2|gpt5.5|gpt5.5-pro)
        if [[ -n "${REASONING_EFFORT}" ]]; then
            DISPLAY_NAME="${DISPLAY_NAME}-${REASONING_EFFORT}"
        fi
        ;;
esac
ROWS_STR="${ROWS:-all}"

# Build the list of response files inference.py would produce. The filename
# pattern and extension vary by task family:
#   athena-*    -> .jsonl
#   CTI-Bench   -> .tsv  (mcq, rcm, vsp, ate, taa)
#   cybermetric -> .csv  (includes the CyberMetric-<N>-v1 stem in the name)
#   mmlu-pro    -> .csv  (keyed by SAFE_NAME, NOT DISPLAY_NAME -- see below)
# resolve_resp_file echoes the expected absolute path for a given task (or
# the empty string for tasks with no fixed pattern, e.g. glue/superglue).
resolve_resp_file() {
    local task="$1"
    # Optional 2nd arg: cybermetric stem (e.g. CyberMetric-2000-v1). Defaults
    # to CYBERMETRIC_STEM (the first --cybermetric-size value) so single-size
    # callers don't have to thread it through.
    local cm_stem="${2:-${CYBERMETRIC_STEM}}"
    local base="${BENCH_DIR}/responses/${DISPLAY_NAME}/${task}"
    case "${task}" in
        athena-*)
            echo "${base}/${task}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.jsonl" ;;
        mcq|rcm|vsp|ate|taa)
            echo "${base}/${task}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.tsv" ;;
        cybermetric)
            echo "${base}/${task}_${cm_stem}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.csv" ;;
        cybersoceval-*)
            echo "${base}/${task}_${ROWS_STR}_v${VERSION}_${DISPLAY_NAME}_response.jsonl" ;;
        mmlu-pro)
            # MMLU-Pro indexes its cache by the alias (SAFE_NAME) instead of
            # the HF repo id (DISPLAY_NAME) so different aliases pointing to
            # the same HF repo get separate caches. Matches the path
            # convention in benchmarks/mmlu_pro.py; both must move together.
            local mmlu_base="${BENCH_DIR}/responses/${SAFE_NAME}/${task}"
            echo "${mmlu_base}/${task}_${ROWS_STR}_v${VERSION}_${SAFE_NAME}_response.csv" ;;
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
    if [[ "${task}" == "cybermetric" ]]; then
        # One target file per configured CyberMetric size (so --overwrite
        # / --retry-errors covers all sizes the sweep is about to write).
        for _stem in "${CYBERMETRIC_STEMS[@]}"; do
            rf="$(resolve_resp_file "${task}" "${_stem}")"
            [[ -n "${rf}" ]] && TARGET_FILES+=("${rf}")
        done
        unset _stem
    else
        rf="$(resolve_resp_file "${task}")"
        [[ -n "${rf}" ]] && TARGET_FILES+=("${rf}")
    fi
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

# --retry-errors: scrub error rows from existing response files in-place
# so the per-bench resume logic re-processes only those rows. Same prompt
# / --yes contract as --overwrite because the operation mutates files.
declare -a SCRUBBED_FILES=()
if [[ ${RETRY_ERRORS} -eq 1 ]]; then
    existing=()
    for f in "${TARGET_FILES[@]}"; do
        [[ -e "$f" ]] && existing+=("$f")
    done
    if [[ ${#existing[@]} -eq 0 ]]; then
        echo "[retry-errors] no pre-existing response files match this run; nothing to scrub."
    else
        echo "[retry-errors] the following response files will be SCRUBBED in place"
        echo "                (rows with error sentinels removed; survivors kept):"
        for f in "${existing[@]}"; do echo "  - ${f#${BENCH_DIR}/}"; done
        if [[ ${ASSUME_YES} -eq 1 ]]; then
            reply="y"
            echo "[retry-errors] --yes given; proceeding without prompt."
        else
            if [[ ! -t 0 ]]; then
                echo "[retry-errors] ERROR: --retry-errors requires an interactive terminal or --yes." >&2
                exit 2
            fi
            printf "[retry-errors] Proceed with in-place scrub? [y/N] " >&2
            read -r reply || reply=""
        fi
        case "${reply}" in
            y|Y|yes|YES)
                python "${SCRIPT_DIR}/_scrub_response_errors.py" "${existing[@]}" \
                    || { echo "[retry-errors] scrub helper failed" >&2; exit 2; }
                SCRUBBED_FILES=("${existing[@]}")
                ;;
            *)
                echo "[retry-errors] aborted by user; exiting without running the sweep."
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
    echo "  reasoning   : ${REASONING_EFFORT:-<none>}"
    echo "  tasks       : ${TASKS}"
    if [[ "${TASKS}" == *"cybermetric"* ]]; then
        if [[ ${#CYBERMETRIC_SIZES[@]} -gt 1 ]]; then
            echo "  cybermetric : ${#CYBERMETRIC_SIZES[@]} sizes (one back-to-back run per size)"
            for _i in "${!CYBERMETRIC_SIZES[@]}"; do
                echo "                  - ${CYBERMETRIC_STEMS[$_i]} (${CYBERMETRIC_DATA_PATHS[$_i]})"
            done
            unset _i
        else
            echo "  cybermetric : ${CYBERMETRIC_STEM} (${CYBERMETRIC_DATA_PATH})"
        fi
    fi
    if [[ ${SINGLE_GPU} -eq 1 ]]; then
        echo "  single-gpu  : yes (CUDA_VISIBLE_DEVICES=${SINGLE_GPU_IDX})"
    else
        echo "  single-gpu  : no (inherits CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
    fi
    echo "  overwrite   : $([[ ${OVERWRITE} -eq 1 ]] && echo yes || echo no)"
    echo "  retry-errs  : $([[ ${RETRY_ERRORS} -eq 1 ]] && echo yes || echo no)"
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
    if [[ ${RETRY_ERRORS} -eq 1 && ${#SCRUBBED_FILES[@]} -gt 0 ]]; then
        echo "[retry-errors] scrubbed ${#SCRUBBED_FILES[@]} response file(s) in place:"
        for f in "${SCRUBBED_FILES[@]}"; do echo "  - ${f#${BENCH_DIR}/}"; done
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
        # cybermetric expands to one inference.py run per configured size
        # (back-to-back against the same served model). Every other task
        # runs exactly once. iter_sizes carries the list to iterate; the
        # empty-string sentinel for non-cybermetric tasks keeps the loop
        # body uniform without per-task conditionals later.
        declare -a iter_sizes=()
        if [[ "${task}" == "cybermetric" ]]; then
            iter_sizes=("${CYBERMETRIC_SIZES[@]}")
        else
            iter_sizes=("")
        fi

        for iter_size in "${iter_sizes[@]}"; do
            # Build per-iteration label / data path. The label is what
            # appears in the summary table (e.g. cybermetric-2000); the
            # underlying inference.py task name stays 'cybermetric' so the
            # benchmark + evaluator code paths are unchanged.
            iter_label="${task}"
            iter_stem=""
            iter_data_path=""
            if [[ "${task}" == "cybermetric" ]]; then
                # Only suffix the label with the size when more than one size
                # is being run in this sweep. Single-size invocations keep the
                # legacy 'cybermetric' label so downstream consumers that grep
                # for ^cybermetric$ in the summary table don't break.
                if [[ ${#CYBERMETRIC_SIZES[@]} -gt 1 ]]; then
                    iter_label="cybermetric-${iter_size}"
                fi
                iter_stem="CyberMetric-${iter_size}-v1"
                iter_data_path="benchmark_data/cybermetricdataset/${iter_stem}.json"
            fi

            echo
            echo "----- task: ${iter_label} -----"
            task_started_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
            echo "  started : ${task_started_iso}"
            task_start=$(date +%s)

            # Task-specific extras (cybermetric needs an explicit --data_path
            # for the size selected this iteration).
            task_extra=()
            if [[ "${task}" == "cybermetric" ]]; then
                task_extra+=(--data_path "${iter_data_path}")
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
            # inference.py emits the underlying task name (cybermetric, not the
            # size-tagged label), so we match against ${task} here.
            metrics_raw="$(grep -E "^Evaluation result for ${task} with " "${task_out_file}" | tail -1 | sed -E "s/^Evaluation result for ${task} with [^:]+: //" || true)"
            rm -f "${task_out_file}"

            # Count rows actually written (evaluator's authoritative input file).
            # cybermetric: pass the iteration's stem so resolve_resp_file picks
            # the right per-size response file.
            if [[ "${task}" == "cybermetric" ]]; then
                resp_file="$(resolve_resp_file "${task}" "${iter_stem}")"
            else
                resp_file="$(resolve_resp_file "${task}")"
            fi
            row_count="$(count_resp_rows "${resp_file}")"

            RES_TASKS+=("${iter_label}")
            RES_ELAPSED+=("${elapsed}")
            RES_EXIT+=("${task_status}")
            RES_METRICS+=("${metrics_raw}")
            RES_ROWS+=("${row_count}")
            RES_STARTED+=("${task_started_iso}")
            RES_FINISHED+=("${task_finished_iso}")

            if [[ ${task_status} -ne 0 ]]; then
                overall_status=${task_status}
                echo "  [WARN] task '${iter_label}' exited non-zero; continuing with remaining tasks"
            fi
        done
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
    # against the same model keeps one artifact per suite. CyberMetric
    # additionally namespaces by size so running e.g. --cybermetric-size
    # 2000 then 10000 produces two distinct summary files instead of the
    # second clobbering the first.
    summary_dir="${BENCH_DIR}/responses/${DISPLAY_NAME}"
    mkdir -p "${summary_dir}"
    summary_stem="${SUITE}"
    if [[ "${SUITE}" == "cybermetric" ]]; then
        # Multi-size: join with '_' (e.g. cybermetric_2000_10000) so the
        # summary file name reflects every CyberMetric split that ran.
        _join_underscore() { local IFS=_; echo -n "$*"; }
        summary_stem="${SUITE}_$(_join_underscore "${CYBERMETRIC_SIZES[@]}")"
    fi
    summary_json="${summary_dir}/summary_${summary_stem}_${ROWS_STR}_v${VERSION}.json"
    summary_md="${summary_dir}/summary_${summary_stem}_${ROWS_STR}_v${VERSION}.md"

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
