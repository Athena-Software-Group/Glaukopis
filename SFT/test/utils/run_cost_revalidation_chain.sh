#!/bin/bash

# Cost-revalidation sweep: runs AthenaBench + CyberMetric (2K + 10K) +
# CyberSOCEval (malware + TI) + MMLU-Pro back-to-back against the six
# models that appear (or should appear) as the v21 + base-model
# baseline block in responses/cost_summary.tsv, so the wallclock-derived
# GPU cost column is recomputed from a single clean 2xH100 session per
# model on the same hardware footprint that build_cost_summary.py bills
# against.
#
# Thin wrapper around run_foundation_8b_baselines.sh. For each alias
# this script:
#   1. Probes huggingface.co for the model artifact; skips if missing.
#   2. Sets per-model serve flags (max-len floor, reasoning parser).
#   3. Invokes the orchestrator with --tp 2 --include-mmlu-pro.
#   4. Lets the orchestrator launch vllm once, run all four suites, and
#      tear vllm down on exit before the next model starts.
#
# Mode default is `overwrite` because the entry point for this chain is
# cost-revalidation: stale partial runs (Foundation-Sec-8B-Instruct at
# 5-6s/task in the prior snapshot) must be discarded so summary_*.json
# wallclocks reflect a single clean serve. Pass --mode resume to keep
# pre-existing rows and only fill gaps.
#
# Per-model knobs baked in:
#   athena-cti-sft-qwen25-14b-v21-cse-vllm  --max-len 32768  (Qwen2.5 ctx cap)
#   foundation-8b-instruct-vllm             --max-len 32768  (matches existing per-model wrapper)
#   llama-3-8b-vllm                         --max-len 32768  (Llama-3.1 ships 131K; we cap to keep
#                                                              KV-cache budget aligned with the rest)
#   qwen2.5-14b-vllm                        --max-len 32768  (Qwen2.5 ctx cap)
#   qwen2.5-32b-vllm                        --max-len 32768  (Qwen2.5 ctx cap)
#   qwen3-30b-a3b-thinking-2507-vllm        --max-len 32768  + EXTRA_SERVE_FLAGS=
#                                                              "--reasoning-parser qwen3"
#                                                              (thinking-on; 8192-token
#                                                              client-side floor applied by
#                                                              VLLMModel via the 'thinking'
#                                                              substring in the alias)
#
# Usage:
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_cost_revalidation_chain.sh
#
#   # Resume mode (keep existing rows; only fill gaps):
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_cost_revalidation_chain.sh --mode resume
#
#   # Run a subset of the chain:
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_cost_revalidation_chain.sh \
#       --models foundation-8b-instruct-vllm,qwen2.5-14b-vllm
#
#   # Smoke-test against first 8 rows of every suite:
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_cost_revalidation_chain.sh --rows 8
#
# Wallclock estimate (2xH100, sequential, --mode overwrite):
#   foundation-8b-instruct-vllm               ~4-5 h
#   llama-3-8b-vllm                           ~4-5 h
#   qwen2.5-14b-vllm                          ~5-6 h
#   athena-cti-sft-qwen25-14b-v21-cse-vllm    ~5-6 h
#   qwen2.5-32b-vllm                          ~7-9 h
#   qwen3-30b-a3b-thinking-2507-vllm          ~8-10 h  (thinking trace tax)
#   total                                     ~33-41 h
#
# Environment:
#   BENCH_CONDA_ENV   conda env for the bench client (default: ctibench).
#   READY_TIMEOUT     vLLM /v1/models readiness budget per model (default 1800s).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${SCRIPT_DIR}/run_foundation_8b_baselines.sh"
if [[ ! -f "${ORCH}" ]]; then
    echo "[FAIL] orchestrator not found at ${ORCH}" >&2
    exit 2
fi

export BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-ctibench}"

DEFAULT_MODELS="athena-cti-sft-qwen25-14b-v21-cse-vllm,foundation-8b-instruct-vllm,llama-3-8b-vllm,qwen2.5-14b-vllm,qwen2.5-32b-vllm,qwen3-30b-a3b-thinking-2507-vllm"
MODELS_CSV="${DEFAULT_MODELS}"
TP="2"
CYBERMETRIC_SIZE="2000,10000"
MAX_LEN="32768"
MODE="overwrite"
ROWS=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)             MODELS_CSV="$2"; shift 2 ;;
        --tp)                 TP="$2"; shift 2 ;;
        --cybermetric-size)   CYBERMETRIC_SIZE="$2"; shift 2 ;;
        --max-len)            MAX_LEN="$2"; shift 2 ;;
        --mode)               MODE="$2"; shift 2 ;;
        --rows)               ROWS="$2"; shift 2 ;;
        --skip-athena|--skip-cybermetric|--skip-cybersoceval|--dry-run)
                              PASS_ARGS+=("$1"); shift ;;
        -h|--help) sed -n '3,68p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${ROWS}" ]] && PASS_ARGS+=( --rows "${ROWS}" )

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"

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
SWEEP_LOG="${SCRIPT_DIR}/cost_revalidation_chain_${UTC}.log"
echo "[info] sweep log : ${SWEEP_LOG}"
echo "[info] models    : ${MODELS_CSV}"
echo "[info] config    : --tp ${TP} --max-len ${MAX_LEN} --cybermetric-size ${CYBERMETRIC_SIZE} --mode ${MODE} +mmlu-pro"
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
        echo "[skip] ${alias} (${repo_id}): not reachable on HF" | tee -a "${SWEEP_LOG}"
        continue
    fi
    echo | tee -a "${SWEEP_LOG}"
    echo "==================================================================" | tee -a "${SWEEP_LOG}"
    echo "  cost-revalidation -> ${alias}  (${repo_id})" | tee -a "${SWEEP_LOG}"
    echo "==================================================================" | tee -a "${SWEEP_LOG}"
    # Per-model serve overrides. The Qwen3-thinking model wants the qwen3
    # reasoning parser so the <think> trace lands in reasoning_content and
    # the bench-visible content field stays clean; all other models
    # inherit the orchestrator default (empty EXTRA_SERVE_FLAGS).
    extra_serve_flags=""
    case "${alias}" in
        qwen3-30b-a3b-thinking-2507-vllm)
            extra_serve_flags="--reasoning-parser qwen3" ;;
    esac
    EXTRA_SERVE_FLAGS="${extra_serve_flags}" \
        bash "${ORCH}" --model "${alias}" --tp "${TP}" \
            --max-len "${MAX_LEN}" --cybermetric-size "${CYBERMETRIC_SIZE}" \
            --mode "${MODE}" --include-mmlu-pro \
            "${PASS_ARGS[@]}" 2>&1 | tee -a "${SWEEP_LOG}"
    echo "[done] ${alias}" | tee -a "${SWEEP_LOG}"
done

echo
echo "[done] cost-revalidation chain complete; log=${SWEEP_LOG}"
