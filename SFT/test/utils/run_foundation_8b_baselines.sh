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
#   2. CyberMetric (size N)  : --max-len 8192,  --batch 64    (~15-20 min for 2K,
#                              ~60-90 min for 10K). --cybermetric-size accepts
#                              a comma-separated list (e.g. 2000,10000) and runs
#                              each size against the same warm vLLM session.
#   3. CyberSOCEval          : --max-len 49152, --batch 32    (~2-3 h, TI rows are slow)
#                              with --gpu-memory-utilization 0.90 --max-num-seqs 32.
#                              The 49152 cap (vs the 32K natural ceiling on
#                              Foundation-Sec-8B) leaves headroom over the TI
#                              report prompts that sit at 32K-32.7K tokens and
#                              previously triggered cascading 400 BadRequestError
#                              on the borderline rows. Foundation-Sec-8B inherits
#                              Llama-3.1-8B's 131K trained context, so 49152 is
#                              well within range; the only cost is a tighter
#                              KV-cache budget, hence --max-num-seqs 32.
#
# Usage:
#   ./run_foundation_8b_baselines.sh [--model ALIAS] [--tp N]
#                                    [--cybermetric-size N[,N...]]   # 80|500|2000|10000
#                                    [--reasoning]
#                                    [--skip-athena] [--skip-cybermetric] [--skip-cybersoceval]
#                                    [--rows N]                  # pass-through to run_benchmark.sh
#                                    [--mode resume|overwrite|retry-errors]
#                                    [--dry-run]
#
# --mode controls what each sub-suite does with pre-existing response files:
#   resume         (default) keep existing rows, only run rows that have
#                  never been processed. Errored rows from a previous run
#                  count as "processed" and stay errored.
#   overwrite      delete pre-existing response files before each suite
#                  runs, forcing a clean run. Equivalent to the old
#                  hardcoded --overwrite --yes behaviour.
#   retry-errors   keep existing rows but scrub any row whose response is
#                  an error sentinel ("Error", "Error: ...", or empty
#                  raw_response for cybermetric). The bench's resume
#                  logic then re-processes only those rows. Useful after
#                  a partial run where some rows failed (e.g. transient
#                  vLLM 400s on the borderline-context CyberSOCEval-TI
#                  rows that we now have headroom for at --max-len 49152).
#
# Environment:
#   BENCH_CONDA_ENV   conda env for the bench client (default: ctibench).
#                     Required when this script is launched from the
#                     isolated `vllm` env.
#   READY_TIMEOUT     vLLM /v1/models readiness budget (default 1800s).
#
# Examples:
#   # Foundation-Sec-8B-Instruct, full sweep, CyberMetric-2000 (default).
#   # Resume mode: skips any task whose response file is complete.
#   conda activate vllm
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh
#
#   # Re-bench from scratch (the old default before --mode was added):
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh \
#       --mode overwrite
#
#   # Retry only the rows that errored on a previous run (typical use
#   # case after fixing a context-overflow bug or a transient API blip):
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh \
#       --mode retry-errors
#
#   # CPT-only base (no chat template; expect lower scores -- "before-CPT" floor)
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh \
#       --model foundation-8b-vllm
#
#   # Reasoning variant re-bench (already done earlier; here for completeness)
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh \
#       --model foundation-8b-reasoning-vllm --reasoning
#
#   # CyberMetric-2000 + CyberMetric-10000 in a single warm-vLLM session,
#   # skipping the other suites:
#   BENCH_CONDA_ENV=ctibench bash SFT/test/utils/run_foundation_8b_baselines.sh \
#       --skip-athena --skip-cybersoceval --cybermetric-size 2000,10000

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
MODE="resume"

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
        --mode)              MODE="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help) sed -n '3,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${MODE}" in
    resume|overwrite|retry-errors) ;;
    *) echo "Unknown --mode: ${MODE} (expected resume|overwrite|retry-errors)" >&2; exit 2 ;;
esac

# Mode -> per-suite flags forwarded to run_benchmark.sh. resume just
# omits both --overwrite and --retry-errors so the per-bench resume
# logic kicks in unchanged.
MODE_ARGS=()
case "${MODE}" in
    overwrite)    MODE_ARGS=( --overwrite --yes ) ;;
    retry-errors) MODE_ARGS=( --retry-errors --yes ) ;;
esac

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
            -- --suite athena --version 1 --batch 64 \
               "${MODE_ARGS[@]}" "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

if [[ ${SKIP_CYBERMETRIC} -eq 0 ]]; then
    # Comma-separated sizes -> one run per size, all against the same warm
    # vLLM server (serve_and_bench keeps the server up for the duration of
    # one invocation; we pay one cold-load per size). Order matters when
    # two sizes overlap (e.g. 10000 contains 2000) but the bench handles
    # that internally via --cybermetric-size selecting a fixed slice.
    IFS=',' read -r -a CYBERMETRIC_SIZES <<< "${CYBERMETRIC_SIZE}"
    for cm_size in "${CYBERMETRIC_SIZES[@]}"; do
        cm_size_trimmed="${cm_size// /}"
        [[ -z "${cm_size_trimmed}" ]] && continue
        run_suite "CyberMetric-${cm_size_trimmed} / ${MODEL_ALIAS}" \
            bash "${SERVE_AND_BENCH}" "${MODEL_ALIAS}" --tp "${TP}" --max-len 8192 \
                ${REASONING_EXTRA:+--extra "${REASONING_EXTRA# }"} \
                -- --suite cybermetric --cybermetric-size "${cm_size_trimmed}" \
                   --version 1 --batch 64 \
                   "${MODE_ARGS[@]}" "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
    done
fi

if [[ ${SKIP_CYBERSOCEVAL} -eq 0 ]]; then
    # 49152 (vs the 32K natural ceiling): adds ~16K headroom over the worst
    # CyberSOCEval-TI report prompts (observed at 32.5K tokens) so the row
    # is no longer one chat-template re-wrap away from a 400. Foundation-Sec-8B
    # is a Llama-3.1-8B derivative trained at 131K context; 49152 is well
    # within range. KV-cache budget tightens, hence --max-num-seqs 32 (down
    # from 64) -- still saturates the H100 80GB at --batch 32.
    SOC_EXTRA="--gpu-memory-utilization 0.90 --max-num-seqs 32${REASONING_EXTRA}"
    run_suite "CyberSOCEval (malware + TI) / ${MODEL_ALIAS}" \
        bash "${SERVE_AND_BENCH}" "${MODEL_ALIAS}" --tp "${TP}" --max-len 49152 \
            --extra "${SOC_EXTRA}" \
            -- --suite cybersoceval --version 1 --batch 32 \
               "${MODE_ARGS[@]}" "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

# Aggregate per-suite summary_*.json files into a single model-wide
# table. Resolve the model's on-disk display directory via the same AST
# parse run_benchmark.sh uses (avoids importing pipelines.models, which
# pulls in torch/dotenv/HF login). Falls back to MODEL_ALIAS verbatim
# when the alias is not in the mapping.
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DISPLAY_NAME="$(cd "${BENCH_DIR}" && python - "${MODEL_ALIAS}" <<'PY'
import ast, pathlib, sys
name = sys.argv[1]
mapping = {}
try:
    src = pathlib.Path("pipelines/models.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "model_mapping":
                    mapping = ast.literal_eval(node.value)
except Exception:
    pass
print(mapping.get(name, name).replace("/", "_"))
PY
)"

echo
echo "=================================================================="
echo "  Model-wide summary / ${MODEL_ALIAS}"
echo "=================================================================="
( cd "${BENCH_DIR}" && python "${SCRIPT_DIR}/_print_model_summary.py" "${DISPLAY_NAME}" ) \
    2>&1 | tee -a "${LOG}" \
    || echo "[WARN] model-wide summary failed (non-fatal); per-suite summaries are still on disk." | tee -a "${LOG}"

echo
echo "[done] foundation-8b baselines complete; log=${LOG}"
