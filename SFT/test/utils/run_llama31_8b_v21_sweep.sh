#!/bin/bash

# Multi-model sweep: runs the AthenaBench + CyberMetric (2K + 10K) +
# CyberSOCEval suite back-to-back against each of the v21 Llama-3.1-8B
# checkpoints (core, taa, cse, recalibrate) under one warm vLLM session
# per model.
#
# Thin wrapper around run_foundation_8b_baselines.sh (the generic single-
# model orchestrator). For each alias passed in --models, this script:
#   1. Probes huggingface.co for the model artifact; skips if not yet
#      pushed (lets you re-run the same sweep as each chain stage lands
#      on HF without editing flags).
#   2. Invokes run_foundation_8b_baselines.sh with Llama-3.1-8B-on-2xH100
#      defaults: --tp 2 --max-len 32768.
#   3. Lets the orchestrator launch vllm once, run all selected suites,
#      and tear vllm down on exit.
#
# Sized for a 2xH100 box (160 GB combined HBM): Llama-3.1-8B in bf16
# (~16 GB weights, ~8 GB per GPU) leaves ample headroom for KV cache for
# cybersoceval-ti at 32K context, --max-num-seqs 32, --gpu-memory-
# utilization 0.90. Llama-3.1's native ctx is 131072; 32768 is chosen for
# parity with the Qwen 14B v21 sweep and to keep --max-num-seqs healthy
# (raise only if a future bench needs longer prompts).
#
# Sign-off question (cross-architecture probe of the v21 recipe; see
# tmpl_gen/templates/05182026/v21_plan.txt §7.5): does the Qwen 14B v21
# Stage-3-CSE-erodes-VSP / Stage-4-Recalibrate-recovers signature
# reproduce on the smaller Llama-3.1-8B base? If yes, the Recalibrate
# stage is locked into the default ship topology for the 8B port; if no
# (CSE does not erode VSP at 8B), the chain stops at CSE for 8B and
# Recalibrate is dropped from the 8B SOP.
#
# Usage:
#   ./run_llama31_8b_v21_sweep.sh [--models alias1,alias2,...] [--tp N]
#                                 [--cybermetric-size N[,N...]]
#                                 [--max-len N] [--mode resume|overwrite|retry-errors]
#                                 [--rows N] [--skip-athena] [--skip-cybermetric]
#                                 [--skip-cybersoceval] [--dry-run]
#
# Defaults:
#   --models           athena-cti-sft-llama31-8b-v21-core-vllm,
#                      athena-cti-sft-llama31-8b-v21-taa-vllm,
#                      athena-cti-sft-llama31-8b-v21-cse-vllm,
#                      athena-cti-sft-llama31-8b-v21-recalibrate-vllm
#   --tp               2          (2xH100 tensor parallel)
#   --cybermetric-size 2000,10000 (both splits in one warm session)
#   --max-len          32768      (Llama-3.1-8B; native 131072 but 32768
#                                  matches Qwen v21 sweep and keeps KV
#                                  headroom for high --max-num-seqs)
#   --mode             resume     (skips already-completed rows; pass
#                                  --mode overwrite for clean re-bench)
#
# Environment:
#   BENCH_CONDA_ENV   conda env for the bench client (default: ctibench).
#                     Required when this script is launched from the
#                     isolated `vllm` env.
#   READY_TIMEOUT     vLLM /v1/models readiness budget per model (default 1800s).
#
# Wall-time estimate (2xH100, Llama-3.1-8B is ~half Qwen2.5-14B weights):
#   Athena (~25-35 min) + CM-2K (~15-20 min) + CM-10K (~60-90 min) +
#   CSE (~1.5-2.5 h) ~ 3-4.5 h per model; full 4-stage sweep ~12-18 h.
#
# Examples:
#   # Core only (start here as soon as v21-core lands on HF; the other
#   # three will auto-skip until they are pushed):
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_llama31_8b_v21_sweep.sh \
#       --models athena-cti-sft-llama31-8b-v21-core-vllm
#
#   # Full sweep (re-run as each successive stage lands; --mode resume
#   # skips already-completed rows and the HF probe skips not-yet-pushed
#   # stages, so the same command is safe to re-issue):
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_llama31_8b_v21_sweep.sh
#
#   # 3-stage v18.1-equivalent ship topology (skip off-plan recalibrate):
#   bash SFT/test/utils/run_llama31_8b_v21_sweep.sh \
#       --models athena-cti-sft-llama31-8b-v21-core-vllm,athena-cti-sft-llama31-8b-v21-taa-vllm,athena-cti-sft-llama31-8b-v21-cse-vllm
#
#   # Skip the long CyberSOCEval suite (Athena + CyberMetric only):
#   bash SFT/test/utils/run_llama31_8b_v21_sweep.sh --skip-cybersoceval

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

DEFAULT_MODELS="athena-cti-sft-llama31-8b-v21-core-vllm,athena-cti-sft-llama31-8b-v21-taa-vllm,athena-cti-sft-llama31-8b-v21-cse-vllm,athena-cti-sft-llama31-8b-v21-recalibrate-vllm"
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
        -h|--help) sed -n '3,80p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${ROWS}" ]] && PASS_ARGS+=( --rows "${ROWS}" )

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"

# HF-availability probe. Resolves alias -> HF repo id via the same AST parse
# that serve_and_bench.sh uses, then HEADs the model API. Skips on 404 only
# (true "not yet pushed"); 200 = public/cached, 401/403 = private + we are
# not authenticated on the probe but vllm-serve will use HF_TOKEN /
# ~/.cache/huggingface/token to download, so proceed. asg-ai org repos are
# private, so this distinction is required to stop the probe from skipping
# every v21 stage as "missing".
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_PROBE_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "${HF_PROBE_TOKEN}" && -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_PROBE_TOKEN="$(tr -d '\n\r' < "${HOME}/.cache/huggingface/token" 2>/dev/null || true)"
fi
hf_probe_status() {
    local url="https://huggingface.co/api/models/$1"
    if [[ -n "${HF_PROBE_TOKEN}" ]]; then
        curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${HF_PROBE_TOKEN}" "${url}"
    else
        curl -s -o /dev/null -w "%{http_code}" "${url}"
    fi
}
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
SWEEP_LOG="${SCRIPT_DIR}/v21_llama31_8b_sweep_${UTC}.log"
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
    http_code="$(hf_probe_status "${repo_id}")"
    case "${http_code}" in
        200|401|403)
            : ;;  # exists; private repos return 401/403 to anon and that's fine
        404)
            echo "[skip] ${alias} (${repo_id}): not yet on HF (HTTP 404 -- training may still be running)" | tee -a "${SWEEP_LOG}"
            continue ;;
        *)
            echo "[warn] ${alias} (${repo_id}): HF probe returned HTTP ${http_code}; attempting bench anyway" | tee -a "${SWEEP_LOG}" ;;
    esac
    echo | tee -a "${SWEEP_LOG}"
    echo "==================================================================" | tee -a "${SWEEP_LOG}"
    echo "  v21 sweep (Llama-3.1-8B) -> ${alias}  (${repo_id})" | tee -a "${SWEEP_LOG}"
    echo "==================================================================" | tee -a "${SWEEP_LOG}"
    bash "${ORCH}" --model "${alias}" --tp "${TP}" \
        --max-len "${MAX_LEN}" --cybermetric-size "${CYBERMETRIC_SIZE}" \
        --mode "${MODE}" "${PASS_ARGS[@]}" 2>&1 | tee -a "${SWEEP_LOG}"
    echo "[done] ${alias}" | tee -a "${SWEEP_LOG}"
done

echo
echo "[done] v21 sweep (Llama-3.1-8B) complete; log=${SWEEP_LOG}"
