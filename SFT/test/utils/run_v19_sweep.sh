#!/bin/bash

# Multi-model sweep: runs the AthenaBench + CyberMetric (2K + 10K) +
# CyberSOCEval suite back-to-back against each of the v19 Qwen2.5-14B
# checkpoints (core, taa, cse, recalibrate) under one warm vLLM session per
# model.
#
# Thin wrapper around run_foundation_8b_baselines.sh (which is itself a
# generic single-model orchestrator -- the name is historical). For each
# alias passed in --models, this script:
#   1. Probes huggingface.co for the model artifact; skips if not yet pushed.
#   2. Invokes run_foundation_8b_baselines.sh with Qwen2.5-14B-on-2xH100
#      defaults: --tp 2 --max-len 32768 (Qwen2.5 native ctx).
#   3. Lets the orchestrator launch vllm once, run all selected suites,
#      and tear vllm down on exit.
#
# Sized for a 2xH100 box (160 GB combined HBM): Qwen2.5-14B in bf16 plus
# KV cache for cybersoceval-ti at 32K context, --max-num-seqs 32, fits
# comfortably with --gpu-memory-utilization 0.90.
#
# Headline gate (v19_plan.txt §5.4): v19-recalibrate MUST PASS RMS >= 54.0,
# MCQ >= 62.0, TAA Classic >= 40.0, CSE-TI >= 34.0, CSE-Malware >= 20.0,
# ATE >= 62.0, RCM >= 67.5, VSP >= 80.0, CM-2K >= 85.5, CM-10K >= 81.0.
# The core/taa/cse intermediates are benched for regression-diagnostics
# (so we can see which axis each stage installed vs eroded); only
# v19-recalibrate is the ship candidate.
#
# Usage:
#   ./run_v19_sweep.sh [--models alias1,alias2,...] [--tp N]
#                      [--cybermetric-size N[,N...]]
#                      [--max-len N] [--mode resume|overwrite|retry-errors]
#                      [--rows N] [--skip-athena] [--skip-cybermetric]
#                      [--skip-cybersoceval] [--dry-run]
#
# Defaults:
#   --models           athena-cti-sft-qwen25-14b-v19-core-vllm,
#                      athena-cti-sft-qwen25-14b-v19-taa-vllm,
#                      athena-cti-sft-qwen25-14b-v19-cse-vllm,
#                      athena-cti-sft-qwen25-14b-v19-recalibrate-vllm
#   --tp               2          (2xH100 tensor parallel)
#   --cybermetric-size 2000,10000 (both splits in one warm session)
#   --max-len          32768      (Qwen2.5-14B native ctx; do NOT raise --
#                                  RoPE produces NaN past max_position_embeddings)
#   --mode             resume     (skips already-completed rows; pass
#                                  --mode overwrite for clean re-bench)
#
# Environment:
#   BENCH_CONDA_ENV   conda env for the bench client (default: ctibench).
#                     Required when this script is launched from the
#                     isolated `vllm` env.
#   READY_TIMEOUT     vLLM /v1/models readiness budget per model (default 1800s).
#
# Examples:
#   # Full sweep (all 4 v19 stages, all 3 suites, both CyberMetric splits):
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_v19_sweep.sh
#
#   # Headline-only (just v19-recalibrate, full suite):
#   bash SFT/test/utils/run_v19_sweep.sh \
#       --models athena-cti-sft-qwen25-14b-v19-recalibrate-vllm
#
#   # Skip the long CyberSOCEval suite (Athena + CyberMetric only, ~2-3h/model):
#   bash SFT/test/utils/run_v19_sweep.sh --skip-cybersoceval

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${SCRIPT_DIR}/run_foundation_8b_baselines.sh"
if [[ ! -f "${ORCH}" ]]; then
    echo "[FAIL] orchestrator not found at ${ORCH}" >&2
    exit 2
fi

# Bench client needs pandas/openai/transformers/tqdm, which live in 'ctibench',
# not in the 'vllm' env that typically launches this sweep. Default
# BENCH_CONDA_ENV to ctibench so the orchestrator switches envs for the bench
# loop. Honour any pre-set value (e.g. BENCH_CONDA_ENV=llm-sft for combined envs).
export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

DEFAULT_MODELS="athena-cti-sft-qwen25-14b-v19-core-vllm,athena-cti-sft-qwen25-14b-v19-taa-vllm,athena-cti-sft-qwen25-14b-v19-cse-vllm,athena-cti-sft-qwen25-14b-v19-recalibrate-vllm"
MODELS_CSV="${DEFAULT_MODELS}"
TP="2"
CYBERMETRIC_SIZE="2000,10000"
MAX_LEN="32768"
MODE="resume"
ROWS=""
DRY_RUN=0
PASS_ARGS=()  # forwarded verbatim to the orchestrator (--skip-* etc.)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)             MODELS_CSV="$2"; shift 2 ;;
        --tp)                 TP="$2"; shift 2 ;;
        --cybermetric-size)   CYBERMETRIC_SIZE="$2"; shift 2 ;;
        --max-len)            MAX_LEN="$2"; shift 2 ;;
        --mode)               MODE="$2"; shift 2 ;;
        --rows)               ROWS="$2"; shift 2 ;;
        --skip-athena|--skip-cybermetric|--skip-cybersoceval)
                              PASS_ARGS+=("$1"); shift ;;
        --dry-run)            DRY_RUN=1; PASS_ARGS+=("$1"); shift ;;
        -h|--help) sed -n '3,62p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${ROWS}" ]] && PASS_ARGS+=( --rows "${ROWS}" )

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"

# HF-availability probe. Resolves alias -> HF repo id via the same AST parse
# that serve_and_bench.sh uses, then HEADs the model API. Skips models that
# are not yet pushed (e.g. when v19-recalibrate is still training on the SFT box).
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
resolve_repo_id() {
    python - "${BENCH_DIR}/pipelines/models.py" "$1" <<'PY'
import ast, sys
path, alias = sys.argv[1], sys.argv[2]
mapping = None
for node in ast.walk(ast.parse(open(path).read())):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "model_mapping":
                mapping = ast.literal_eval(node.value); break
        if mapping is not None: break
print(mapping.get(alias, "") if mapping else "")
PY
}

UTC="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
SWEEP_LOG="${SCRIPT_DIR}/v19_sweep_${UTC}.log"
echo "[info] sweep log : ${SWEEP_LOG}"
echo "[info] models    : ${MODELS_CSV}"
echo "[info] config    : --tp ${TP} --max-len ${MAX_LEN} --cybermetric-size ${CYBERMETRIC_SIZE} --mode ${MODE}"
echo

for alias in "${MODELS[@]}"; do
    alias="${alias// /}"
    [[ -z "${alias}" ]] && continue
    repo_id="$(resolve_repo_id "${alias}")"
    if [[ -z "${repo_id}" ]]; then
        echo "[skip] ${alias}: not found in pipelines/models.py" | tee -a "${SWEEP_LOG}"
        continue
    fi
    if ! curl -fsS -o /dev/null "https://huggingface.co/api/models/${repo_id}"; then
        echo "[skip] ${alias} (${repo_id}): not yet on HF (training may still be running)" | tee -a "${SWEEP_LOG}"
        continue
    fi
    echo | tee -a "${SWEEP_LOG}"
    echo "==================================================================" | tee -a "${SWEEP_LOG}"
    echo "  v19 sweep -> ${alias}  (${repo_id})" | tee -a "${SWEEP_LOG}"
    echo "==================================================================" | tee -a "${SWEEP_LOG}"
    bash "${ORCH}" --model "${alias}" --tp "${TP}" \
        --max-len "${MAX_LEN}" --cybermetric-size "${CYBERMETRIC_SIZE}" \
        --mode "${MODE}" "${PASS_ARGS[@]}" 2>&1 | tee -a "${SWEEP_LOG}"
    echo "[done] ${alias}" | tee -a "${SWEEP_LOG}"
done

echo
echo "[done] v19 sweep complete; log=${SWEEP_LOG}"
