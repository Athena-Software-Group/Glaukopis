#!/bin/bash

# Run AthenaBench + CyberMetric (2K + 10K) + CyberSOCEval (malware + TI)
# back-to-back against an HF Inference Router-routed model alias. Pure HTTP
# work; no GPU/CUDA touched, so this is safe to run alongside concurrent
# vLLM benches or SFT training on the same host.
#
# Default target is deepseek-v3.1-terminus-hf (deepseek-ai/DeepSeek-V3.1-
# Terminus, 685B MoE / 37B active). Routes through router.huggingface.co/v1
# to whichever provider currently serves the repo (Together / Fireworks /
# Novita / etc., resolved server-side at request time when provider=auto).
#
# This is the HF-router-side equivalent of run_foundation_8b_baselines.sh
# (which only handles vLLM-served '-vllm' aliases). Same suite shape and
# same per-suite summary file layout, so the resulting summary_athena.md /
# summary_cybermetric_2000_10000.md / summary_cybersoceval.md drop into
# responses/<display>/ alongside the foundation baselines for direct diff.
#
# Suite wallclock estimates against HF Router at --batch 32 (rough; varies
# by provider load and model size):
#   AthenaBench    ~8233 rows -> 60-90 min for a 685B-class MoE.
#   CyberMetric 2K -> 30-45 min.
#   CyberMetric 10K -> 2.0-3.0 h.
#   CyberSOCEval   ~1197 rows -> 90-180 min (TI rows hit 25-32K input tokens).
#   Total (default sweep)    ~ 5-9 h on a typical provider.
#
# Usage:
#   conda activate ctibench
#   bash bench_hf_router.sh [--model ALIAS]
#                           [--batch N]
#                           [--cybermetric-size N[,N...]]
#                           [--mode resume|overwrite|retry-errors]
#                           [--skip-athena|--skip-cybermetric|--skip-cybersoceval]
#                           [--rows N] [--dry-run]
#
# Environment:
#   HUGGINGFACE_TOKEN (or HF_TOKEN)  Required. Loaded from SFT/.env (or
#                                    SFT/test/.env as a legacy fallback) if
#                                    not already exported. Token must have
#                                    'Inference Providers' scope.
#   HF_INFERENCE_ENDPOINT_URL        Optional. When set, bypasses provider
#                                    auto-routing and targets a dedicated
#                                    Inference Endpoint URL instead.
#
# Examples:
#   # Default: full DeepSeek V3.1 Terminus sweep, fresh baseline.
#   bash bench_hf_router.sh
#
#   # Resume after a partial run (keep what completed, only redo missing rows):
#   bash bench_hf_router.sh --mode resume
#
#   # Just AthenaBench (skip the slow CyberSOCEval suite):
#   bash bench_hf_router.sh --skip-cybermetric --skip-cybersoceval
#
#   # Different HF-router model with the same suite shape:
#   bash bench_hf_router.sh --model deepseek-v4-pro-hf
#   bash bench_hf_router.sh --model kimi-k2.6-hf

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BENCH="${SCRIPT_DIR}/run_benchmark.sh"

MODEL_ALIAS="deepseek-v3.1-terminus-hf"
BATCH="32"
CYBERMETRIC_SIZE="2000,10000"
MODE="overwrite"
SKIP_ATHENA=0
SKIP_CYBERMETRIC=0
SKIP_CYBERSOCEVAL=0
ROWS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL_ALIAS="$2"; shift 2 ;;
        --batch)             BATCH="$2"; shift 2 ;;
        --cybermetric-size)  CYBERMETRIC_SIZE="$2"; shift 2 ;;
        --mode)              MODE="$2"; shift 2 ;;
        --skip-athena)       SKIP_ATHENA=1; shift ;;
        --skip-cybermetric)  SKIP_CYBERMETRIC=1; shift ;;
        --skip-cybersoceval) SKIP_CYBERSOCEVAL=1; shift ;;
        --rows)              ROWS="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help) sed -n '3,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${MODE}" in
    resume|overwrite|retry-errors) ;;
    *) echo "Unknown --mode: ${MODE} (expected resume|overwrite|retry-errors)" >&2; exit 2 ;;
esac

MODE_ARGS=()
case "${MODE}" in
    overwrite)    MODE_ARGS=( --overwrite --yes ) ;;
    retry-errors) MODE_ARGS=( --retry-errors --yes ) ;;
esac

# Pre-flight: HF token. pipelines/models.py calls load_dotenv() with no
# explicit path, which walks parent directories from cwd; from
# SFT/test/ that resolves SFT/.env correctly. We mirror that lookup
# here so a missing token fails fast (clear message) rather than
# surfacing mid-run as a 401 from the router. Canonical location is
# SFT/.env (one level above SFT/test/); SFT/test/.env retained as a
# legacy fallback. Skipped under --dry-run so the script is testable
# on dev boxes without HF credentials.
if [[ ${DRY_RUN} -eq 0 ]]; then
    if [[ -z "${HUGGINGFACE_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
        for env_path in "${SCRIPT_DIR}/../../.env" "${SCRIPT_DIR}/../.env"; do
            if [[ -f "${env_path}" ]]; then
                set -a
                # shellcheck disable=SC1091
                source "${env_path}"
                set +a
                break
            fi
        done
    fi
    if [[ -z "${HUGGINGFACE_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
        echo "[FAIL] HUGGINGFACE_TOKEN (or HF_TOKEN) is required for HF Router routing." >&2
        echo "       Either export it in this shell or add it to SFT/.env." >&2
        exit 2
    fi
fi

ROWS_ARG=()
[[ -n "${ROWS}" ]] && ROWS_ARG=( --rows "${ROWS}" )

UTC="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
SAFE_ALIAS="${MODEL_ALIAS//\//_}"
LOG="${SCRIPT_DIR}/${SAFE_ALIAS}_baseline_${UTC}.log"

echo "=== HF Router bench sweep (${UTC}) ==="
echo "  alias       : ${MODEL_ALIAS}"
echo "  batch       : ${BATCH}"
echo "  cm sizes    : ${CYBERMETRIC_SIZE}"
echo "  mode        : ${MODE}"
echo "  rows        : ${ROWS:-all}"
echo "  log file    : ${LOG}"
echo

run_suite() {
    local suite="$1"; shift
    echo
    echo "=================================================================="
    echo "  Suite: ${suite}"
    echo "=================================================================="
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] bash ${RUN_BENCH} ${MODEL_ALIAS} --suite ${suite} --batch ${BATCH} ${MODE_ARGS[*]} $* ${ROWS_ARG[*]}"
        return 0
    fi
    bash "${RUN_BENCH}" "${MODEL_ALIAS}" \
        --suite "${suite}" \
        --batch "${BATCH}" \
        "${MODE_ARGS[@]}" \
        "$@" \
        "${ROWS_ARG[@]}"
}

overall=0
if [[ ${SKIP_ATHENA} -eq 0 ]]; then
    run_suite athena 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}; [[ ${rc} -ne 0 ]] && overall=${rc}
fi
if [[ ${SKIP_CYBERMETRIC} -eq 0 ]]; then
    run_suite cybermetric --cybermetric-size "${CYBERMETRIC_SIZE}" 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}; [[ ${rc} -ne 0 ]] && overall=${rc}
fi
if [[ ${SKIP_CYBERSOCEVAL} -eq 0 ]]; then
    run_suite cybersoceval 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}; [[ ${rc} -ne 0 ]] && overall=${rc}
fi

echo
echo "=== sweep complete (overall exit ${overall}) ===" | tee -a "${LOG}"
exit ${overall}
