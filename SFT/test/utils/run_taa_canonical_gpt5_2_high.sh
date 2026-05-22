#!/bin/bash

# TAA Canonical (athena-taa-canonical) sweep against OpenAI gpt-5.2 with
# reasoning_effort=high. Pure HTTP; no GPU. Per-model carve-out of the
# six-model run_taa_canonical_baselines.sh orchestrator -- runnable
# independently when only the OpenAI row needs a re-bench.
#
# Resolves to:
#   alias    : gpt5.2
#   model    : gpt-5.2
#   display  : gpt-5.2-high  (run_benchmark.sh mirrors inference.py's
#                             reasoning-effort folder rewrite for the
#                             OpenAI responses-API family)
#
# Wallclock: ~3-5 min (single task, 100 rows, reasoning traces).
#
# Usage:
#   conda activate ctibench
#   bash run_taa_canonical_gpt5_2_high.sh [--rows N] [--batch N]
#                                         [--no-overwrite]
#                                         [--reasoning-effort low|medium|high|xhigh]
#                                         [--dry-run]
#
# Environment:
#   OPENAI_API_KEY  required (loaded from SFT/.env by pipelines/models.py
#                   if not already exported).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BENCH="${SCRIPT_DIR}/run_benchmark.sh"

ROWS=""
BATCH="16"
OVERWRITE=1
REASONING_EFFORT="high"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows)             ROWS="$2"; shift 2 ;;
        --batch)            BATCH="$2"; shift 2 ;;
        --no-overwrite)     OVERWRITE=0; shift ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${OPENAI_API_KEY:-}" ]] && \
    { echo "[FAIL] OPENAI_API_KEY required for gpt5.2" >&2; exit 2; }

MODE_FLAGS=()
[[ ${OVERWRITE} -eq 1 ]] && MODE_FLAGS+=(--overwrite --yes)
ROWS_FLAGS=()
[[ -n "${ROWS}" ]] && ROWS_FLAGS+=(--rows "${ROWS}")

echo "=================================================================="
echo "  TAA Canonical / gpt-5.2-high  (alias=gpt5.2, effort=${REASONING_EFFORT})"
echo "=================================================================="

if [[ ${DRY_RUN} -eq 1 ]]; then
    _mode="${MODE_FLAGS[*]:-}"
    _rows="${ROWS_FLAGS[*]:-}"
    echo "[dry-run] ${RUN_BENCH} gpt5.2 --tasks athena-taa-canonical --version 1 --batch ${BATCH} ${_mode} ${_rows} --reasoning-effort ${REASONING_EFFORT}"
    exit 0
fi

exec bash "${RUN_BENCH}" gpt5.2 \
    --tasks "athena-taa-canonical" --version 1 \
    --batch "${BATCH}" "${MODE_FLAGS[@]}" "${ROWS_FLAGS[@]}" \
    --reasoning-effort "${REASONING_EFFORT}"
