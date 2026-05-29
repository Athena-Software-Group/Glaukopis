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
# vLLM lifecycle: a single serve_vllm.sh process is launched at the start
# of the sweep and stays up for every selected suite, then is torn down
# once on exit (cleanup trap covers normal exit, Ctrl-C and SIGTERM). This
# trades a small amount of KV-cache headroom on the short suites for ~6 min
# of saved cold-load + cudagraph-capture overhead per extra serve that the
# old per-suite design paid.
#
# Serve sizing is picked from the union of selected suites:
#   any cybersoceval selected -> --max-len 49152 --max-num-seqs 32 + --batch 32
#   otherwise                 -> --max-len 16384 --max-num-seqs 64 + --batch 64
# 49152 (vs the 32K natural ceiling on Foundation-Sec-8B) leaves headroom
# over the TI report prompts that sit at 32K-32.7K tokens and previously
# triggered cascading 400 BadRequestError on the borderline rows.
# 16384 (vs the prior 8192) is the new non-cybersoceval floor: AthenaBench
# rows in athena-vsp/-rcm/-rms can carry CVE descriptions that push the
# wrapped prompt to ~7900 tokens, leaving zero generation budget under
# 8192 and triggering 400s that no client-side max_tokens shrink can
# recover from (the vLLM error reports a derived lower bound, not the
# real prompt size). Doubling to 16384 gives ~8K of generation headroom
# on the worst row and is comfortably within H100 KV-cache budget at
# --max-num-seqs 64. Foundation-Sec-8B and Llama-3.1-8B both have 131K
# trained context, so the cap is purely a serve-side memory choice.
#
# Suite shapes (wall-clock estimates on 1xH100, 8B model):
#   1. AthenaBench           : ~30-45 min
#   2. CyberMetric size N    : ~15-20 min for 2K, ~60-90 min for 10K.
#                              --cybermetric-size accepts a comma-separated
#                              list (default '2000,10000') and runs each
#                              size as its own task against the warm server.
#   3. CyberSOCEval          : ~2-3 h (TI rows are slow).
#   4. MMLU-Pro              : ~5-10 min on 14B/2xH100 (12K rows; opt-in via
#                              --include-mmlu-pro, off by default since
#                              MMLU-Pro is a reasoning benchmark, not a CTI
#                              one, and the standing wrappers shouldn't
#                              silently add it to their wall-clock budget).
#
# Usage:
#   ./run_foundation_8b_baselines.sh [--model ALIAS] [--tp N]
#                                    [--cybermetric-size N[,N...]]   # 80|500|2000|10000
#                                    [--reasoning]
#                                    [--max-len N]               # override serve --max-len
#                                    [--skip-athena] [--skip-cybermetric] [--skip-cybersoceval]
#                                    [--include-mmlu-pro]        # opt in; default off
#                                    [--rows N]                  # pass-through to run_benchmark.sh
#                                    [--mode resume|overwrite|retry-errors]
#                                    [--dry-run]
#
# --max-len overrides the auto-pick (49152 with cybersoceval, 16384 without).
# Required for model families whose max_position_embeddings is below the
# auto-pick: Qwen2.5-* tops out at 32768, and vLLM rejects --max-len above
# the model's native ctx (RoPE produces NaN past it). Drop to 32768 for
# Qwen2.5-14B/32B full sweep -- cybersoceval-TI rows that exceed 32K are
# caught by the client-side ctx-overflow path in pipelines/models.py and
# bail with a one-line notice instead of crashing the suite.
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
#   BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/run_foundation_8b_baselines.sh
#
#   # Re-bench from scratch (the old default before --mode was added):
#   BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/run_foundation_8b_baselines.sh \
#       --mode overwrite
#
#   # Retry only the rows that errored on a previous run (typical use
#   # case after fixing a context-overflow bug or a transient API blip):
#   BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/run_foundation_8b_baselines.sh \
#       --mode retry-errors
#
#   # CPT-only base (no chat template; expect lower scores -- "before-CPT" floor)
#   BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/run_foundation_8b_baselines.sh \
#       --model foundation-8b-vllm
#
#   # Reasoning variant re-bench (already done earlier; here for completeness)
#   BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/run_foundation_8b_baselines.sh \
#       --model foundation-8b-reasoning-vllm --reasoning
#
#   # CyberMetric-2000 + CyberMetric-10000 in a single warm-vLLM session,
#   # skipping the other suites:
#   BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/run_foundation_8b_baselines.sh \
#       --skip-athena --skip-cybersoceval --cybermetric-size 2000,10000

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ALIAS="foundation-8b-instruct-vllm"
TP="1"
CYBERMETRIC_SIZE="2000,10000"
SKIP_ATHENA=0
SKIP_CYBERMETRIC=0
SKIP_CYBERSOCEVAL=0
RUN_MMLU_PRO=0
REASONING=0
ROWS=""
DRY_RUN=0
MODE="resume"
MAX_LEN_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL_ALIAS="$2"; shift 2 ;;
        --tp)                TP="$2"; shift 2 ;;
        --cybermetric-size)  CYBERMETRIC_SIZE="$2"; shift 2 ;;
        --reasoning)         REASONING=1; shift ;;
        --max-len)           MAX_LEN_OVERRIDE="$2"; shift 2 ;;
        --skip-athena)       SKIP_ATHENA=1; shift ;;
        --skip-cybermetric)  SKIP_CYBERMETRIC=1; shift ;;
        --skip-cybersoceval) SKIP_CYBERSOCEVAL=1; shift ;;
        --include-mmlu-pro)  RUN_MMLU_PRO=1; shift ;;
        --rows)              ROWS="$2"; shift 2 ;;
        --mode)              MODE="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help) sed -n '3,72p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

SERVE_VLLM="${SCRIPT_DIR}/serve_vllm.sh"
if [[ ! -f "${SERVE_VLLM}" ]]; then
    echo "[FAIL] serve_vllm.sh not found at ${SERVE_VLLM}" >&2
    exit 2
fi

BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve <alias> -> HF repo id via the same ast parse serve_and_bench.sh
# uses (avoids importing pipelines.models which pulls in torch/dotenv/HF
# login). Required because serve_vllm.sh takes the HF repo id, not the
# bench alias.
REPO_ID="$(python - "${BENCH_DIR}/pipelines/models.py" "${MODEL_ALIAS}" <<'PY'
import ast, sys
path, alias = sys.argv[1], sys.argv[2]
tree = ast.parse(open(path).read())
mapping = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "model_mapping":
                mapping = ast.literal_eval(node.value)
                break
        if mapping is not None:
            break
if mapping is None or alias not in mapping:
    sys.stderr.write(f"unknown alias: {alias}\n"); sys.exit(2)
if not alias.endswith("-vllm"):
    sys.stderr.write(f"alias must end with '-vllm': {alias}\n"); sys.exit(3)
print(mapping[alias])
PY
)"
if [[ -z "${REPO_ID}" ]]; then
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

# Pick the union-max serve config: if any suite needs the long-context
# headroom (cybersoceval-ti rows hover around 32K tokens), serve at
# 49152 with --max-num-seqs 32 and run the bench client at --batch 32 for
# every suite so client-side concurrency matches server-side. Otherwise
# stay at 16384 (was 8192; bumped to fit the worst AthenaBench row at
# ~7900 prompt tokens with non-zero generation budget) with the wider
# batch.
if [[ ${SKIP_CYBERSOCEVAL} -eq 0 ]]; then
    SERVE_MAX_LEN=49152
    SERVE_MAX_SEQS=32
    BENCH_BATCH=32
else
    SERVE_MAX_LEN=16384
    SERVE_MAX_SEQS=64
    BENCH_BATCH=64
fi
if [[ -n "${MAX_LEN_OVERRIDE}" ]]; then
    if ! [[ "${MAX_LEN_OVERRIDE}" =~ ^[0-9]+$ ]] || [[ "${MAX_LEN_OVERRIDE}" -lt 1024 ]]; then
        echo "[FAIL] --max-len must be a positive integer >= 1024 (got: ${MAX_LEN_OVERRIDE})" >&2
        exit 2
    fi
    echo "[info] --max-len override: ${SERVE_MAX_LEN} -> ${MAX_LEN_OVERRIDE}"
    SERVE_MAX_LEN="${MAX_LEN_OVERRIDE}"
fi
SERVE_EXTRA="--gpu-memory-utilization 0.90 --max-num-seqs ${SERVE_MAX_SEQS}${REASONING_EXTRA}"
# Optional env-var passthrough for ad-hoc vllm serve flags (e.g.
# `--limit-mm-per-prompt image=0` for multimodal-capable models served
# text-only here, like Gemma 4 31B IT). Whitespace-trimmed and appended
# verbatim to the existing SERVE_EXTRA template.
if [[ -n "${EXTRA_SERVE_FLAGS:-}" ]]; then
    SERVE_EXTRA="${SERVE_EXTRA} ${EXTRA_SERVE_FLAGS}"
fi

UTC="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG="${SCRIPT_DIR}/foundation_8b_baselines_${UTC}.log"
SAFE_ALIAS="${MODEL_ALIAS//\//_}"
SERVE_LOG="${SCRIPT_DIR}/${SAFE_ALIAS}_serve_${UTC}.log"
PORT=8000
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
READY_POLL="${READY_POLL:-5}"
echo "[info] sweep log : ${LOG}"
echo "[info] serve log : ${SERVE_LOG}"
echo "[info] serve cfg : --max-len ${SERVE_MAX_LEN} --max-num-seqs ${SERVE_MAX_SEQS} --tp ${TP}"
echo "[info] bench batch: ${BENCH_BATCH}"

# Launch vLLM once for the whole sweep. setsid puts it in its own process
# group so cleanup can signal the entire tree (vllm spawns per-TP worker
# procs that a plain pid kill would orphan).
echo
echo "=================================================================="
echo "  Launching vLLM (single session for the entire sweep)"
echo "=================================================================="
setsid bash "${SERVE_VLLM}" --model "${REPO_ID}" --tp "${TP}" \
    --max-len "${SERVE_MAX_LEN}" --port "${PORT}" \
    --extra "${SERVE_EXTRA# }" \
    >"${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
SERVE_PGID="$(ps -o pgid= "${SERVE_PID}" | tr -d ' ')"

cleanup() {
    local rc=$?
    echo
    echo "=== tearing down vllm server (pgid=${SERVE_PGID}) ===" | tee -a "${LOG}"
    kill -TERM "-${SERVE_PGID}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "${SERVE_PID}" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "-${SERVE_PGID}" 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT INT TERM

echo "=== waiting for http://localhost:${PORT}/v1/models (timeout ${READY_TIMEOUT}s) ===" | tee -a "${LOG}"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "  ready." | tee -a "${LOG}"
        break
    fi
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "[FAIL] vllm serve exited before becoming ready. Tail of serve log:" | tee -a "${LOG}"
        tail -40 "${SERVE_LOG}" | tee -a "${LOG}"
        exit 3
    fi
    if [[ $(date +%s) -ge ${deadline} ]]; then
        echo "[FAIL] vllm did not become ready within ${READY_TIMEOUT}s." | tee -a "${LOG}"
        tail -40 "${SERVE_LOG}" | tee -a "${LOG}"
        exit 4
    fi
    sleep "${READY_POLL}"
done

# Bench-client invoker: wraps run_benchmark.sh with `conda run` when
# BENCH_CONDA_ENV is set so the bench picks up pandas/transformers/openai
# from a different env than the one serving vllm (typical case: this
# wrapper is launched from the isolated `vllm` env).
BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-}"
run_bench() {
    if [[ -n "${BENCH_CONDA_ENV}" ]]; then
        conda run --no-capture-output -n "${BENCH_CONDA_ENV}" \
            bash "${SCRIPT_DIR}/run_benchmark.sh" "${MODEL_ALIAS}" "$@"
    else
        bash "${SCRIPT_DIR}/run_benchmark.sh" "${MODEL_ALIAS}" "$@"
    fi
}

run_suite() {
    local label="$1"; shift
    echo
    echo "==================================================================" | tee -a "${LOG}"
    echo "  ${label}" | tee -a "${LOG}"
    echo "==================================================================" | tee -a "${LOG}"
    run_bench "$@" 2>&1 | tee -a "${LOG}"
    local rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then
        echo "[WARN] ${label} exited rc=${rc}; continuing with the rest of the sweep." | tee -a "${LOG}"
    fi
}

if [[ ${SKIP_ATHENA} -eq 0 ]]; then
    run_suite "AthenaBench / ${MODEL_ALIAS}" \
        --suite athena --version 1 --batch "${BENCH_BATCH}" \
        "${MODE_ARGS[@]}" "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

if [[ ${SKIP_CYBERMETRIC} -eq 0 ]]; then
    # Comma-separated sizes -> one run per size, all against the same warm
    # vLLM server. Order matters when two sizes overlap (e.g. 10000 contains
    # 2000) but the bench handles that internally via --cybermetric-size
    # selecting a fixed slice.
    IFS=',' read -r -a CYBERMETRIC_SIZES <<< "${CYBERMETRIC_SIZE}"
    for cm_size in "${CYBERMETRIC_SIZES[@]}"; do
        cm_size_trimmed="${cm_size// /}"
        [[ -z "${cm_size_trimmed}" ]] && continue
        run_suite "CyberMetric-${cm_size_trimmed} / ${MODEL_ALIAS}" \
            --suite cybermetric --cybermetric-size "${cm_size_trimmed}" \
            --version 1 --batch "${BENCH_BATCH}" \
            "${MODE_ARGS[@]}" "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
    done
fi

if [[ ${SKIP_CYBERSOCEVAL} -eq 0 ]]; then
    run_suite "CyberSOCEval (malware + TI) / ${MODEL_ALIAS}" \
        --suite cybersoceval --version 1 --batch "${BENCH_BATCH}" \
        "${MODE_ARGS[@]}" "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

# MMLU-Pro: opt-in, off by default. Reasoning benchmark (TIGER-Lab,
# 12K rows, up to 10 options per question); excluded from --suite all
# in run_benchmark.sh for the same reason. Prompts are small (~1K
# input, 1024 generation cap per TASK_MAX_NEW_TOKENS) so the non-
# cybersoceval serve sizing (16384/64) is comfortable; no separate
# union-max adjustment needed.
if [[ ${RUN_MMLU_PRO} -eq 1 ]]; then
    run_suite "MMLU-Pro / ${MODEL_ALIAS}" \
        --suite mmlu-pro --version 1 --batch "${BENCH_BATCH}" \
        "${MODE_ARGS[@]}" "${ROWS_ARG[@]}" "${DRY_ARG[@]}"
fi

# Aggregate per-suite summary_*.json files into a single model-wide
# table. The on-disk directory is keyed by SAFE_ALIAS (the sanitized
# alias, NOT the HF repo id) so two aliases pointing to the same HF
# repo aggregate independently; see pipelines/models.alias_to_safe_name
# for the rationale and run_benchmark.sh's summary_dir/SAFE_NAME for
# the matching write-side convention.

echo
echo "=================================================================="
echo "  Model-wide summary / ${MODEL_ALIAS}"
echo "=================================================================="
( cd "${BENCH_DIR}" && python "${SCRIPT_DIR}/_print_model_summary.py" "${SAFE_ALIAS}" ) \
    2>&1 | tee -a "${LOG}" \
    || echo "[WARN] model-wide summary failed (non-fatal); per-suite summaries are still on disk." | tee -a "${LOG}"

echo
echo "[done] foundation-8b baselines complete; log=${LOG}"
