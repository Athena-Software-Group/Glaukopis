#!/bin/bash

# Run the Athena benchmark sweep for one model with **data-parallel sharding
# across multiple GPUs**. For each task we:
#   1. Split the task's input JSONL into N roughly-equal shards (N = --gpus,
#      or auto-detected via nvidia-smi).
#   2. Launch N inference.py subprocesses in parallel, each pinned to one
#      GPU via CUDA_VISIBLE_DEVICES, each writing to a shard-specific
#      response file (distinguished by --version suffix).
#   3. Concatenate the shard response JSONLs into the canonical response
#      file (as if a single sequential run had produced it).
#   4. Re-run evaluation on the merged file via tasks_evaluation.py so the
#      reported metric is computed on ALL rows, not per-shard.
#
# Usage:
#   ./run_benchmark_parallel.sh <model-name> [--gpus N] [--version N]
#                               [--tasks "athena-mcq ..."] [--overwrite] [--yes]
#
# While shards run, an aggregate progress line is printed to stdout (and log)
# every PROGRESS_INTERVAL seconds (default 15), e.g.:
#   [progress  45s] s0:12/750 s1:11/750 s2:12/750 s3:11/750 | total 46/3000 (1%) eta 48m20s
#
# Log:     SFT/test/utils/<safe-model-name>_parallel.log
#          SFT/test/utils/<safe-model-name>_parallel.<task>.shard<i>.log
# Summary: SFT/test/utils/<safe-model-name>_parallel.summary.json
#          (aggregate per-task metrics + timings, written at end of sweep)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

MODEL_NAME="$1"; shift
GPUS=""
VERSION=1
TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-taa-canonical athena-rms"
OVERWRITE=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)      GPUS="$2"; shift 2 ;;
        --version)   VERSION="$2"; shift 2 ;;
        --tasks)     TASKS="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --yes|-y)    ASSUME_YES=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SAFE_NAME="${MODEL_NAME//\//_}"
LOG_FILE="${SCRIPT_DIR}/${SAFE_NAME}_parallel.log"

# Auto-detect GPU count if not specified
if [[ -z "${GPUS}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    fi
    GPUS="${GPUS:-1}"
fi
if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || [[ ${GPUS} -lt 1 ]]; then
    echo "ERROR: --gpus must be a positive integer (got '${GPUS}')" >&2
    exit 1
fi

# Resolve display name (AST-parse pipelines/models.py, no torch import
# needed). DISPLAY_NAME is only used for human-readable echoing in the
# sweep header below; all on-disk paths key off SAFE_NAME (the alias)
# so two aliases pointing to the same HF repo get isolated caches
# (see pipelines/models.alias_to_safe_name for the rationale).
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

# Mirrors the defaults hard-coded in benchmarks/athena_*.py (self.data_file).
# Kept here in the shell so we know where to split BEFORE launching Python.
data_path_for_task() {
    case "$1" in
        athena-mcq) echo "benchmark_data/athena_bench/athena-cti-mcq-3k.jsonl" ;;
        athena-rcm) echo "benchmark_data/athena_bench/athena-cti-rcm.jsonl" ;;
        athena-vsp) echo "benchmark_data/athena_bench/athena-cti-vsp.jsonl" ;;
        athena-ate) echo "benchmark_data/athena_bench/athena-cti-ate.jsonl" ;;
        athena-taa) echo "benchmark_data/athena_bench/athena_taa/athena-cti-taa.jsonl" ;;
        athena-taa-canonical) echo "benchmark_data/athena_bench/athena_taa_canonical/athena-cti-taa-canonical.jsonl" ;;
        athena-rms) echo "benchmark_data/athena_bench/athena-cti-rms.jsonl" ;;
        *) return 1 ;;
    esac
}

# Shard version naming: base_version * 1000 + shard_index. Keeps shard output
# files distinct from the canonical single-version run and from each other.
shard_version() { printf '%d%03d' "${VERSION}" "$1"; }

final_response_path() {
    local task="$1"
    echo "${BENCH_DIR}/responses/${SAFE_NAME}/${task}/${task}_all_v${VERSION}_${SAFE_NAME}_response.jsonl"
}
shard_response_path() {
    local task="$1" i="$2"
    local sv; sv=$(shard_version "$i")
    echo "${BENCH_DIR}/responses/${SAFE_NAME}/${task}/${task}_all_v${sv}_${SAFE_NAME}_response.jsonl"
}

# --- Overwrite handling (mirrors run_benchmark.sh) -------------------------
declare -a TARGETS=()
for task in ${TASKS}; do
    TARGETS+=("$(final_response_path "$task")")
    for (( i=0; i<GPUS; i++ )); do
        TARGETS+=("$(shard_response_path "$task" "$i")")
    done
done

if [[ ${OVERWRITE} -eq 1 ]]; then
    existing=()
    for f in "${TARGETS[@]}"; do [[ -e "$f" ]] && existing+=("$f"); done
    if [[ ${#existing[@]} -gt 0 ]]; then
        echo "[overwrite] the following files will be DELETED:"
        for f in "${existing[@]}"; do echo "  - ${f#${BENCH_DIR}/}"; done
        if [[ ${ASSUME_YES} -ne 1 ]]; then
            if [[ ! -t 0 ]]; then
                echo "[overwrite] ERROR: --overwrite requires a tty or --yes." >&2
                exit 2
            fi
            printf "[overwrite] Proceed? [y/N] " >&2
            read -r reply || reply=""
            case "${reply}" in y|Y|yes|YES) : ;; *) echo "[overwrite] aborted."; exit 1 ;; esac
        fi
        for f in "${existing[@]}"; do rm -f -- "$f"; done
    fi
fi


# --- Main driver (tee'd to log) -------------------------------------------
SHARD_DIR="$(mktemp -d -t athena_parallel.XXXXXX)"
SUMMARY_TMP="$(mktemp -t athena_summary.XXXXXX)"
SUMMARY_FILE="${SCRIPT_DIR}/${SAFE_NAME}_parallel.summary.json"
trap 'rm -rf "${SHARD_DIR}"; rm -f "${SUMMARY_TMP}"' EXIT

{
    echo "=== Athena parallel benchmark sweep ==="
    echo "  model        : ${MODEL_NAME}"
    echo "  display name : ${DISPLAY_NAME}"
    echo "  bench dir    : ${BENCH_DIR}"
    echo "  shard tmpdir : ${SHARD_DIR}"
    echo "  log file     : ${LOG_FILE}"
    echo "  python       : $(command -v python || echo '(none)')"
    echo "  env          : ${CONDA_DEFAULT_ENV:-<none>}"
    echo "  gpus         : ${GPUS}"
    echo "  version      : ${VERSION}"
    echo "  tasks        : ${TASKS}"
    echo "  started      : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo

    cd "${BENCH_DIR}" || { echo "[FAIL] cannot cd to ${BENCH_DIR}"; exit 1; }

    overall_status=0
    for task in ${TASKS}; do
        echo
        echo "============================================================"
        echo "task: ${task}"
        echo "============================================================"
        task_start=$(date +%s)

        input_rel="$(data_path_for_task "${task}")"
        input_abs="${BENCH_DIR}/${input_rel}"
        if [[ ! -s "${input_abs}" ]]; then
            echo "  [SKIP] input not found or empty: ${input_rel}"
            overall_status=1; continue
        fi
        total_rows=$(wc -l < "${input_abs}")
        echo "  input   : ${input_rel} (${total_rows} rows)"

        # Split into GPUS interleaved (round-robin) shards: row j -> shard
        # (j mod n). This averages out per-row cost variation that depends
        # on input position (e.g. inputs ordered by difficulty / length),
        # which a contiguous chunked split would otherwise concentrate on
        # one GPU. Shard sizes still differ by at most 1 row.
        # Done in Python for portability (BSD split lacks -n / -d / etc.).
        shard_prefix="${SHARD_DIR}/${task}_shard_"
        python - "${input_abs}" "${shard_prefix}" "${GPUS}" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
n = int(sys.argv[3])
lines = src.read_text().splitlines(keepends=True)
for i in range(n):
    out = pathlib.Path(f"{prefix}{i:02d}.jsonl")
    out.write_text("".join(lines[i::n]))
PY
        mapfile -t SHARD_FILES < <(ls "${shard_prefix}"*.jsonl | sort)
        if [[ ${#SHARD_FILES[@]} -ne ${GPUS} ]]; then
            echo "  [FAIL] expected ${GPUS} shards, got ${#SHARD_FILES[@]}"
            overall_status=1; continue
        fi

        # Launch one inference.py per GPU, pinned via CUDA_VISIBLE_DEVICES.
        pids=(); rcs=(); shard_logs=(); shard_targets=(); shard_sizes=()
        for (( i=0; i<GPUS; i++ )); do
            sv=$(shard_version "$i")
            shard_log="${SCRIPT_DIR}/${SAFE_NAME}_parallel.${task}.shard${i}.log"
            shard_logs+=("${shard_log}")
            shard_targets+=("$(shard_response_path "${task}" "$i")")
            shard_sizes+=("$(wc -l < "${SHARD_FILES[$i]}")")
            echo "  launching shard ${i}: CUDA_VISIBLE_DEVICES=${i} --version ${sv} (${shard_sizes[$i]} rows) -> ${shard_log##*/}"
            CUDA_VISIBLE_DEVICES="${i}" \
                python inference.py "${task}" "${MODEL_NAME}" \
                    --data_path "${SHARD_FILES[$i]}" \
                    --version "${sv}" \
                    >"${shard_log}" 2>&1 &
            pids+=($!)
        done

        # Background progress monitor: polls each shard's response file and
        # prints an aggregate status line every PROGRESS_INTERVAL seconds.
        # Counting lines works because inference.py appends one JSON line per
        # completed row. Runs as a subshell; killed once all shards finish.
        PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-15}"
        mon_start=$(date +%s)
        (
            while :; do
                sleep "${PROGRESS_INTERVAL}"
                now=$(date +%s); elapsed=$(( now - mon_start ))
                done_total=0; parts=""
                for (( j=0; j<GPUS; j++ )); do
                    n=0
                    if [[ -s "${shard_targets[$j]}" ]]; then
                        n=$(wc -l < "${shard_targets[$j]}" 2>/dev/null || echo 0)
                        n=$(( n + 0 ))  # strip whitespace padding (BSD wc)
                    fi
                    done_total=$(( done_total + n ))
                    parts="${parts}s${j}:${n}/${shard_sizes[$j]} "
                done
                pct=0
                if [[ ${total_rows} -gt 0 ]]; then
                    pct=$(( done_total * 100 / total_rows ))
                fi
                eta="--"
                if [[ ${done_total} -gt 0 && ${done_total} -lt ${total_rows} ]]; then
                    rate_num=${done_total}; rate_den=${elapsed}
                    [[ ${rate_den} -lt 1 ]] && rate_den=1
                    remaining=$(( total_rows - done_total ))
                    eta_s=$(( remaining * rate_den / rate_num ))
                    eta=$(printf '%dm%02ds' $(( eta_s / 60 )) $(( eta_s % 60 )))
                fi
                printf '  [progress %3ds] %s| total %d/%d (%d%%) eta %s\n' \
                    "${elapsed}" "${parts}" "${done_total}" "${total_rows}" "${pct}" "${eta}"
            done
        ) &
        mon_pid=$!

        # Wait for all shards; record exit codes individually.
        for (( i=0; i<GPUS; i++ )); do
            if wait "${pids[$i]}"; then rcs+=(0); else rcs+=($?); fi
            echo "  shard ${i} finished (exit ${rcs[$i]})"
        done

        # Stop the progress monitor.
        kill "${mon_pid}" 2>/dev/null || true
        wait "${mon_pid}" 2>/dev/null || true

        # Concatenate shard response JSONLs -> canonical response file.
        final="$(final_response_path "${task}")"
        mkdir -p "$(dirname "${final}")"
        : > "${final}"
        missing=0
        shard_files_to_clean=()
        for (( i=0; i<GPUS; i++ )); do
            sf="$(shard_response_path "${task}" "$i")"
            if [[ -s "${sf}" ]]; then
                cat "${sf}" >> "${final}"
                shard_files_to_clean+=("${sf}")
                # Also clean up the sibling _scored.jsonl if the eval ran per-shard.
                scored="${sf%_response.jsonl}_scored.jsonl"
                [[ -e "${scored}" ]] && shard_files_to_clean+=("${scored}")
            else
                echo "  [WARN] shard ${i} produced no response file (${sf##*/})"
                missing=1
            fi
        done
        merged_rows=$(wc -l < "${final}")
        echo "  merged  : ${merged_rows}/${total_rows} rows -> ${final#${BENCH_DIR}/}"

        # Remove per-shard response/scored files now that the merge succeeded.
        if [[ ${missing} -eq 0 && ${#shard_files_to_clean[@]} -gt 0 ]]; then
            rm -f "${shard_files_to_clean[@]}"
        fi

        # Re-run evaluation on the merged file (per-shard metrics in the shard
        # logs are partial; this is the authoritative one). Capture eval
        # stdout to a file so we can both display it and parse the
        # `ATHENA-XXX Metrics: {...}` line for the summary JSON.
        echo "  evaluating merged file..."
        eval_out_file="$(mktemp -t athena_eval.XXXXXX)"
        set +e
        python tasks_evaluation.py --task "${task}" --model "${MODEL_NAME}" \
            --response_file "${final}" >"${eval_out_file}" 2>&1
        eval_status=$?
        set -e
        sed 's/^/    /' "${eval_out_file}"
        metrics_line="$(grep -E '^ATHENA-[A-Z]+ Metrics:' "${eval_out_file}" | tail -1 || true)"
        rm -f "${eval_out_file}"

        task_end=$(date +%s)
        elapsed=$(( task_end - task_start ))
        echo "  elapsed : ${elapsed}s"
        if [[ ${missing} -ne 0 || ${eval_status} -ne 0 ]]; then
            overall_status=1
        fi
        for rc in "${rcs[@]}"; do
            [[ "${rc}" -ne 0 ]] && overall_status=1
        done

        # Append a per-task record (NDJSON) to the summary tmp file.
        python - "${SUMMARY_TMP}" "${task}" "${metrics_line}" "${elapsed}" \
                "${merged_rows}" "${total_rows}" "${eval_status}" \
                "${rcs[*]}" "${final}" <<'PY'
import sys, json, ast, re
tmp, task, metrics_line, elapsed, merged_rows, total_rows, eval_status, rcs_str, final = sys.argv[1:10]
metrics = None
m = re.match(r"^ATHENA-[A-Z]+ Metrics:\s*(\{.*\})\s*$", metrics_line.strip())
if m:
    try:
        metrics = ast.literal_eval(m.group(1))
    except Exception:
        metrics = {"raw": metrics_line.strip()}
elif metrics_line.strip():
    metrics = {"raw": metrics_line.strip()}
record = {
    "task": task,
    "metrics": metrics,
    "elapsed_s": int(elapsed),
    "merged_rows": int(merged_rows),
    "total_rows": int(total_rows),
    "eval_status": int(eval_status),
    "shard_exit_codes": [int(x) for x in rcs_str.split() if x],
    "response_file": final,
}
with open(tmp, "a") as f:
    f.write(json.dumps(record) + "\n")
PY
    done

    echo
    echo "=== Sweep complete ==="
    echo "  finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "  exit    : ${overall_status}"

    # Aggregate per-task records -> summary JSON next to the log.
    python - "${SUMMARY_TMP}" "${SUMMARY_FILE}" "${MODEL_NAME}" \
            "${DISPLAY_NAME}" "${VERSION}" "${GPUS}" "${overall_status}" <<'PY'
import sys, json, pathlib
from datetime import datetime, timezone
tmp, out, model, display, version, gpus, overall = sys.argv[1:8]
records = []
p = pathlib.Path(tmp)
if p.exists():
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
summary = {
    "model": model,
    "display_name": display,
    "version": int(version),
    "gpus": int(gpus),
    "overall_status": int(overall),
    "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "tasks": records,
}
pathlib.Path(out).write_text(json.dumps(summary, indent=2) + "\n")
PY
    echo "  summary : ${SUMMARY_FILE#${BENCH_DIR}/}"
    exit ${overall_status}
} 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
