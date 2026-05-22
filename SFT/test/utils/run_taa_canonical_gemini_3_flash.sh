#!/bin/bash

# TAA Canonical (athena-taa-canonical) sweep against Google gemini-3-flash
# (preview). Pure HTTP; no GPU. Per-model carve-out of the six-model
# run_taa_canonical_baselines.sh orchestrator -- runnable independently
# when only the gemini-3-flash row needs a re-bench.
#
# Resolves to:
#   alias    : gemini-3-flash
#   model    : gemini-3-flash-preview
#   display  : gemini-3-flash-preview
#
# Wallclock: ~1-2 min (single task, 100 rows).
#
# Usage:
#   conda activate ctibench
#   bash run_taa_canonical_gemini_3_flash.sh [--rows N] [--batch N]
#                                            [--no-overwrite] [--dry-run]
#
# Environment:
#   GEMINI_API_KEY  required (loaded from SFT/.env by pipelines/models.py
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
    { echo "[FAIL] GEMINI_API_KEY required for gemini-3-flash (export it or add it to SFT/.env)" >&2; exit 2; }

MODE_FLAGS=()
[[ ${OVERWRITE} -eq 1 ]] && MODE_FLAGS+=(--overwrite --yes)
ROWS_FLAGS=()
[[ -n "${ROWS}" ]] && ROWS_FLAGS+=(--rows "${ROWS}")

echo "=================================================================="
echo "  TAA Canonical / gemini-3-flash-preview  (alias=gemini-3-flash)"
echo "=================================================================="

if [[ ${DRY_RUN} -eq 1 ]]; then
    _mode="${MODE_FLAGS[*]:-}"
    _rows="${ROWS_FLAGS[*]:-}"
    echo "[dry-run] ${RUN_BENCH} gemini-3-flash --tasks athena-taa-canonical --version 1 --batch ${BATCH} ${_mode} ${_rows}"
    exit 0
fi

exec bash "${RUN_BENCH}" gemini-3-flash \
    --tasks "athena-taa-canonical" --version 1 \
    --batch "${BATCH}" "${MODE_FLAGS[@]}" "${ROWS_FLAGS[@]}"
