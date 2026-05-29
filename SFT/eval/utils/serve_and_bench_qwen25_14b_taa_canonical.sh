#!/bin/bash

# TAA Canonical (athena-taa-canonical) sweep against Qwen2.5-14B-Instruct
# served on a local vLLM session (--tp 2; sized for 2xH100). Per-model
# carve-out of the six-model run_taa_canonical_baselines.sh orchestrator --
# runnable independently when only the Qwen2.5-14B row needs a re-bench.
#
# Resolves to:
#   alias    : qwen2.5-14b-vllm
#   model    : Qwen/Qwen2.5-14B-Instruct
#   display  : Qwen_Qwen2.5-14B-Instruct
#
# Wraps serve_and_bench.sh so the vLLM server is launched, the canonical
# TAA sweep runs against it, and the server is torn down on exit.
# --max-len 32768 matches the model's native ctx; --max-num-seqs 32
# matches BATCH=64 with comfortable headroom (see serve_vllm.sh header
# table for the per-family sizing rationale).
#
# Wallclock: ~4-6 min (vLLM warmup ~3 min + bench ~1-2 min).
#
# Usage:
#   conda activate vllm   # or whatever env has vllm installed
#   bash serve_and_bench_qwen25_14b_taa_canonical.sh [--rows N] [--batch N]
#                                                    [--no-overwrite] [--dry-run]
#
# Environment:
#   BENCH_CONDA_ENV  conda env that has the bench Python deps installed
#                    (default: ctibench). serve_and_bench.sh activates
#                    this for the run_benchmark.sh leg while keeping vLLM
#                    in the current (vllm) env.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_AND_BENCH="${SCRIPT_DIR}/serve_and_bench.sh"

ROWS=""
BATCH="64"
OVERWRITE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows)         ROWS="$2"; shift 2 ;;
        --batch)        BATCH="$2"; shift 2 ;;
        --no-overwrite) OVERWRITE=0; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

MODE_FLAGS=()
[[ ${OVERWRITE} -eq 1 ]] && MODE_FLAGS+=(--overwrite --yes)
ROWS_FLAGS=()
[[ -n "${ROWS}" ]] && ROWS_FLAGS+=(--rows "${ROWS}")

echo "=================================================================="
echo "  TAA Canonical / Qwen2.5-14B-Instruct  (alias=qwen2.5-14b-vllm)"
echo "=================================================================="

if [[ ${DRY_RUN} -eq 1 ]]; then
    _mode="${MODE_FLAGS[*]:-}"
    _rows="${ROWS_FLAGS[*]:-}"
    echo "[dry-run] serve_and_bench.sh qwen2.5-14b-vllm --tp 2 --max-len 32768 \\"
    echo "          --extra '--gpu-memory-utilization 0.92 --max-num-seqs 32' \\"
    echo "          -- --tasks athena-taa-canonical --version 1 --batch ${BATCH} ${_mode} ${_rows}"
    exit 0
fi

exec env BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}" \
    bash "${SERVE_AND_BENCH}" qwen2.5-14b-vllm \
    --tp 2 --max-len 32768 \
    --extra "--gpu-memory-utilization 0.92 --max-num-seqs 32" \
    -- --tasks "athena-taa-canonical" --version 1 \
       --batch "${BATCH}" "${MODE_FLAGS[@]}" "${ROWS_FLAGS[@]}"
