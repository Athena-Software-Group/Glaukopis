#!/bin/bash

# Run AthenaBench, CyberMetric and CyberSOCEval back-to-back against a
# single vLLM-served model so cell 2 of the v8 SFT matrix
# (fdtn-ai/Foundation-Sec-8B + v8-small) has clean pre-SFT baselines to
# compare against.
#
# Default target is the Cisco Foundation-Sec-8B-Instruct model
# (alias `foundation-8b-instruct-vllm`, HF id fdtn-ai/Foundation-Sec-8B-Instruct).
# That is the Cisco-shipped instruction-tuned variant (SFT+RLHF on top of
# the Foundation-Sec-8B CPT base, custom '<|system|>/<|user|>/<|assistant|>'
# chat template baked in). It is the right pre-SFT comparison point for
# cell 2 of the v8 matrix because Athena's v8 SFT replaces this exact
# stack with our own SFT recipe.
#
# To bench other variants instead:
#   --model foundation-8b-vllm                  # CPT base, no chat template
#   --model foundation-8b-reasoning-vllm --reasoning   # appends
#       `--reasoning-parser minimax_m2 --trust-remote-code` to vllm extras
#
# Each suite re-serves vLLM at the right --max-len for that suite; the
# alternative (single serve at the largest cutoff) wastes KV cache on the
# short-context Athena/CyberMetric runs and costs more wall-clock overall.
# Three serves x ~3 min cold-load = ~9 min overhead, vs ~30-60 min wasted
# on KV cache budget mismatch when servicing the short suites at 32K.
#
# Suite shapes:
#   1. AthenaBench           : --max-len 8192,  --batch 64    (~30-45 min)
#   2. CyberMetric (size N)  : --max-len 8192,  --batch 64    (~15-20 min for 2K)
#   3. CyberSOCEval          : --max-len 32768, --batch 32    (~2-3 h, TI rows are slow)
#                              with --gpu-memory-utilization 0.90 --max-num-seqs 64
#
# Usage:
#   ./run_foundation_8b_baselines.sh [--model ALIAS] [--tp N]
#                                    [--cybermetric-size 80|500|2000|10000]
#                                    [--reasoning]
#                                    [--skip-athena] [--skip-cybermetric] [--skip-cybersoceval]
#                                    [--rows N]                  # pass-through to run_benchmark.sh
#                                    [--dry-run]
#
# Environment:
#   BENCH_CONDA_ENV   conda env for the bench client (default: ctibench).
#                     Required when this script is launched from the
#                     isolated `vllm` env.
#   READY_TIMEOUT     vLLM /v1/models readiness budget (default 1800s).
#
# Examples:
#   # Foundation-Sec-8B-Instruct, full sweep, CyberMetric-2000 (default)
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh
#
#   # CPT-only base (no chat template; expect lower scores -- "before-CPT" floor)
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh \
#       --model foundation-8b-vllm
#
#   # Reasoning variant re-bench (already done earlier; here for completeness)
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh \
#       --model foundation-8b-reasoning-vllm --reasoning

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_AND_BENCH="${SCRIPT_DIR}/serve_and_bench.sh"

MODEL_ALIAS="foundation-8b-instruct-vllm"
TP="1"
CYBERMETRIC_SIZE="2000"
SKIP_ATHENA=0
SKIP_CYBERMETRIC=0
SKIP_CYBERSOCEVAL=0
REASONING=0
ROWS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL_ALIAS="$2"; shift 2 ;;
        --tp)                TP="$2"; shift 2 ;;
        --cybermetric-size)  CYBERMETRIC_SIZE="$2"; shift 2 ;;
        --reasoning)         REASONING=1; shift ;;
        --skip-athena)       SKIP_ATHENA=1; shift ;;
        --skip-cybermetric)  SKIP_CYBERMETRIC=1; shift ;;
        --skip-cybersoceval) SKIP_CYBERSOCEVAL=1; shift ;;
        --rows)              ROWS="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help) sed -n '3,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -x "${SERVE_AND_BENCH}" ]]; then
    echo "[FAIL] serve_and_bench.sh not found or not executable at ${SERVE_AND_BENCH}" >&2
    exit 2
fi

REASONING_EXTRA=""
if [[ ${REASONING} -eq 1 ]]; then
    REASONING_EXTRA=" --reasoning-parser minimax_m2 --trust-remote-code"
fi

ROWS_ARG=()
[[ -n "${ROWS}" ]] && ROWS_ARG=( --rows "${ROWS}" )
DRY_ARG=()
[[ ${DRY_RUN} -eq 1 ]] && DRY_ARG=( --dry-run )

UTC="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG="${SCRIPT_DIR}/foundation_8b_baselines_${UTC}.log"
echo "[info] log: ${LOG}"

run_suite() {
    local label="$1"; shift
    echo
    echo "=================================================================="
    echo "  ${label}"
    echo "=================================================================="
    "$@" 2>&1 | tee -a "${LOG}"
    local rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then
        echo "[WARN] ${label} exited rc=${rc}; continuing with the rest of the sweep." | tee -a "${LOG}"
    fi
}

if [[ ${SKIP_ATHENA} -eq 0 ]]; then
    run_suite "AthenaBench / ${MODEL_ALIAS}" \
        bash "${SERVE_AND_BENCH}" "${MODEL_ALIAS}" --tp "${TP}" --max-len 8192 \
            ${REASONING_EXTRA:+--extra "${REASONING_EXTRA# }"} \
            -- --suite athena --version 1 --batch 64 --overwrite --yes \
            "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

if [[ ${SKIP_CYBERMETRIC} -eq 0 ]]; then
    run_suite "CyberMetric-${CYBERMETRIC_SIZE} / ${MODEL_ALIAS}" \
        bash "${SERVE_AND_BENCH}" "${MODEL_ALIAS}" --tp "${TP}" --max-len 8192 \
            ${REASONING_EXTRA:+--extra "${REASONING_EXTRA# }"} \
            -- --suite cybermetric --cybermetric-size "${CYBERMETRIC_SIZE}" \
               --version 1 --batch 64 --overwrite --yes \
               "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

if [[ ${SKIP_CYBERSOCEVAL} -eq 0 ]]; then
    SOC_EXTRA="--gpu-memory-utilization 0.90 --max-num-seqs 64${REASONING_EXTRA}"
    run_suite "CyberSOCEval (malware + TI) / ${MODEL_ALIAS}" \
        bash "${SERVE_AND_BENCH}" "${MODEL_ALIAS}" --tp "${TP}" --max-len 32768 \
            --extra "${SOC_EXTRA}" \
            -- --suite cybersoceval --version 1 --batch 32 --overwrite --yes \
               "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

echo
echo "[done] foundation-8b baselines complete; log=${LOG}"
