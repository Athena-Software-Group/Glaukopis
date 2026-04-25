#!/bin/bash

# Run the AthenaBench suite against the latest hosted-API frontier models
# (OpenAI gpt-5.5 / gpt-5.5-pro + Google gemini-3.1-pro-preview) for
# v0/baseline comparison alongside the v6 SFT training run. Pure HTTP work,
# no GPU/CUDA touched, so this is safe to run in a separate terminal on the
# same host as a concurrent SFT training job (zero VRAM, ~few hundred MB RSS).
#
# Models exercised (in order, sequential to avoid shared-quota contention):
#   1. gpt5.5                            -> display gpt-5.5
#   2. gpt5.5 --reasoning-effort high    -> display gpt-5.5-high
#   3. gpt5.5-pro                        -> display gpt-5.5-pro
#   4. gemini-3.1-pro                    -> display gemini-3.1-pro-preview
#
# Each run:
#   - --suite athena       (athena-mcq, -rcm, -vsp, -ate, -taa, -rms)
#   - --version 1          (matches the versioned naming convention used
#                           by the v5/v6 SFT response files)
#   - --batch 32           (in-model concurrency; well under typical
#                           OpenAI Tier-2 / Gemini quotas)
#   - --overwrite --yes    (refreshes any pre-existing response files so
#                           they conform to the current naming scheme)
#
# Usage:
#   ./run_api_baselines.sh [--rows N] [--batch N] [--no-overwrite]
#                          [--models "gpt5.5 gpt5.5-pro gemini-3.1-pro ..."]
#                          [--suite athena|ctibench|all]
#                          [--reasoning-effort low|medium|high|xhigh]
#                          [--dry-run]
#
# Environment:
#   OPENAI_API_KEY    required for the gpt5.5 / gpt5.5-pro runs
#   GEMINI_API_KEY    required for the gemini-3.1-pro run
#   Conda env         expected: ctibench (the bench-side env, NOT llm-sft)
#
# Logs:
#   SFT/test/utils/api_baselines_<UTC_TIMESTAMP>.log
#   per-model summary JSON/MD under SFT/test/responses/<display>/

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_BENCH="${SCRIPT_DIR}/run_benchmark.sh"

ROWS=""
BATCH="32"
SUITE="athena"
OVERWRITE=1
REASONING_EFFORT_OVERRIDE=""
DRY_RUN=0
MODELS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows)              ROWS="$2"; shift 2 ;;
        --batch)             BATCH="$2"; shift 2 ;;
        --suite)             SUITE="$2"; shift 2 ;;
        --no-overwrite)      OVERWRITE=0; shift ;;
        --reasoning-effort)  REASONING_EFFORT_OVERRIDE="$2"; shift 2 ;;
        --models)            MODELS_OVERRIDE="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help)           sed -n '3,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -x "${RUN_BENCH}" ]]; then
    echo "[FAIL] expected ${RUN_BENCH} to be executable" >&2
    exit 2
fi

# Pre-flight: required API keys. .env is auto-loaded by pipelines/models.py
# (python-dotenv) but a missing key surfaces as a stack trace mid-run, so
# fail fast here.
missing=()
[[ -z "${OPENAI_API_KEY:-}" ]] && missing+=("OPENAI_API_KEY")
[[ -z "${GEMINI_API_KEY:-}" ]] && missing+=("GEMINI_API_KEY")
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[FAIL] missing env var(s): ${missing[*]}" >&2
    echo "       Either export them in this shell or add them to SFT/.env." >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "ctibench" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<unset>}'; expected 'ctibench'."
    echo "       If imports fail, run: conda activate ctibench"
fi

# Per-row recipe: "<alias>|<display-suffix>|<reasoning-effort or empty>"
# Sequential execution avoids hammering OpenAI's per-org quota with the
# gpt5.5 variants in parallel; gemini-3.1-pro is its own quota pool but
# kept sequential for log clarity.
declare -a RUNS=(
    "gpt5.5||"
    "gpt5.5|-high|high"
    "gpt5.5-pro||"
    "gemini-3.1-pro||"
)
if [[ -n "${MODELS_OVERRIDE}" ]]; then
    RUNS=()
    for m in ${MODELS_OVERRIDE}; do
        RUNS+=("${m}||")
    done
fi

stamp="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
LOG_FILE="${SCRIPT_DIR}/api_baselines_${stamp}.log"

echo "=== AthenaBench API baselines (${stamp}) ==="
echo "  bench dir : ${BENCH_DIR}"
echo "  log file  : ${LOG_FILE}"
echo "  models    : ${RUNS[*]}"
echo "  suite/rows: ${SUITE} / ${ROWS:-all}"
echo "  batch     : ${BATCH}"
echo "  overwrite : ${OVERWRITE}"
echo "  dry-run   : ${DRY_RUN}"
echo

cd "${BENCH_DIR}" || { echo "[FAIL] cannot cd to ${BENCH_DIR}" >&2; exit 2; }

overall_status=0
for spec in "${RUNS[@]}"; do
    IFS='|' read -r alias _suffix effort <<< "${spec}"
    label="${alias}${_suffix:+ (${_suffix#-})}"

    args=(--suite "${SUITE}" --version 1 --batch "${BATCH}")
    [[ -n "${ROWS}" ]] && args+=(--rows "${ROWS}")
    [[ ${OVERWRITE} -eq 1 ]] && args+=(--overwrite --yes)
    if [[ -n "${REASONING_EFFORT_OVERRIDE}" ]]; then
        args+=(--reasoning-effort "${REASONING_EFFORT_OVERRIDE}")
    elif [[ -n "${effort}" ]]; then
        args+=(--reasoning-effort "${effort}")
    fi

    echo "----- ${label} -----" | tee -a "${LOG_FILE}"
    echo "  cmd: bash ${RUN_BENCH} ${alias} ${args[*]}" | tee -a "${LOG_FILE}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] not invoking" | tee -a "${LOG_FILE}"
        continue
    fi

    set +e
    bash "${RUN_BENCH}" "${alias}" "${args[@]}" 2>&1 | tee -a "${LOG_FILE}"
    rc=${PIPESTATUS[0]}
    set -e

    echo "  exit: ${rc}" | tee -a "${LOG_FILE}"
    [[ ${rc} -ne 0 ]] && overall_status=${rc}
    echo | tee -a "${LOG_FILE}"
done

echo "=== sweep complete (overall exit ${overall_status}) ===" | tee -a "${LOG_FILE}"
exit ${overall_status}
