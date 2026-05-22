#!/bin/bash

# RMS (athena-rms) sweep against Google gemini-2.5-flash. Pure HTTP; no GPU.
# Single-task, single-model carve-out for re-benching just the RMS row on
# gemini-2.5-flash without touching any other suite or model. SFT/.env is
# auto-sourced so the GEMINI_API_KEY pre-flight passes from the canonical
# location used by the rest of the bench scripts.
#
# Resolves to:
#   alias    : gemini-2.5-flash
#   model    : gemini-2.5-flash
#   display  : gemini-2.5-flash
#
# Wallclock: ~3-5 min (single task, ~500 rows).
#
# Usage:
#   conda activate ctibench
#   bash run_rms_gemini_2_5_flash.sh [--rows N] [--batch N]
#                                    [--no-overwrite] [--dry-run]
#
# Environment:
#   GEMINI_API_KEY  required (loaded from SFT/.env by _load_dotenv.sh
#                   if not already exported).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BENCH="${SCRIPT_DIR}/run_benchmark.sh"

# shellcheck source=_load_dotenv.sh
source "${SCRIPT_DIR}/_load_dotenv.sh"

ROWS=""
BATCH="16"
OVERWRITE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows)         ROWS="$2"; shift 2 ;;
        --batch)        BATCH="$2"; shift 2 ;;
        --no-overwrite) OVERWRITE=0; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${GEMINI_API_KEY:-}" ]] && \
    { echo "[FAIL] GEMINI_API_KEY required for gemini-2.5-flash (export it or add it to SFT/.env)" >&2; exit 2; }

MODE_FLAGS=()
[[ ${OVERWRITE} -eq 1 ]] && MODE_FLAGS+=(--overwrite --yes)
ROWS_FLAGS=()
[[ -n "${ROWS}" ]] && ROWS_FLAGS+=(--rows "${ROWS}")

echo "=================================================================="
echo "  RMS / gemini-2.5-flash  (alias=gemini-2.5-flash)"
echo "=================================================================="

if [[ ${DRY_RUN} -eq 1 ]]; then
    _mode="${MODE_FLAGS[*]:-}"
    _rows="${ROWS_FLAGS[*]:-}"
    echo "[dry-run] ${RUN_BENCH} gemini-2.5-flash --tasks athena-rms --version 1 --batch ${BATCH} ${_mode} ${_rows}"
    exit 0
fi

exec bash "${RUN_BENCH}" gemini-2.5-flash \
    --tasks "athena-rms" --version 1 \
    --batch "${BATCH}" "${MODE_FLAGS[@]}" "${ROWS_FLAGS[@]}"
