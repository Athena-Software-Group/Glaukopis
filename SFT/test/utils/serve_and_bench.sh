#!/bin/bash

# One-shot: launch a local vLLM server, wait for it to come up, run an
# AthenaBench sweep against it, and tear the server down on exit.
#
# Wraps serve_vllm.sh + run_benchmark.sh so a full baseline can be
# kicked off from a single terminal (no two-terminal choreography, no
# orphaned vllm workers on Ctrl-C).
#
# Usage:
#   ./serve_and_bench.sh <model-alias> [serve-flags...] -- [bench-flags...]
#
# The alias must be a '-vllm' entry in pipelines/models.py; this script
# resolves it to the HF repo id and passes that to vllm serve. Flags
# before '--' go to serve_vllm.sh; flags after '--' go to run_benchmark.sh.
#
# Defaults if you omit the separator:
#   serve : --tp 2 --max-len 4096 --port 8000
#   bench : --suite athena --version 2 --batch 64
#
# Examples:
#   ./serve_and_bench.sh phi-4-vllm
#   ./serve_and_bench.sh phi-4-vllm --tp 2 -- --suite athena --version 2 --batch 64
#   ./serve_and_bench.sh llama-3-8b-vllm --tp 1 --max-len 8192 -- --batch 128
#
# CyberSOCEval long-context serve sizing (see also serve_vllm.sh header):
#   The TI rows embed full extracted PDF text in the prompt (CrowdStrike /
#   CISA / NSA reports, frequently 5K-25K tokens). Serving at the default
#   --max-len 4096 collapses to >85% parse errors as vLLM's retry-shrink
#   path drops max_tokens to a floor that can't fit the JSON answer block.
#   Pick --max-len from the table below based on the served model's native
#   trained context (vllm fail-closes if --max-len exceeds the model's
#   max_position_embeddings, since RoPE positions past the trained max
#   produce NaN). Match --batch to --max-num-seqs so client concurrency
#   does not queue beyond what the engine can hold.
#
#     Model family             Native ctx   --max-len   --max-num-seqs   --batch
#     Qwen2.5-14B-Instruct     32768        32768       32               32
#     Qwen2.5-32B-Instruct     32768        32768       16               16
#     Llama-3.1-8B-Instruct    131072       65536       32-64            32-64
#     Foundation-Sec-8B*       131072       49152       32               32  (Llama-3.1 base)
#
#   Qwen2.5-14B-Instruct on 2xH100, native 32K (validated v15 W1):
#   ./serve_and_bench.sh athena-cti-sft-qwen25-14b-v12-plus-taa-vllm \
#       --tp 2 --max-len 32768 \
#       --extra "--gpu-memory-utilization 0.92 --max-num-seqs 32" \
#       -- --suite cybersoceval --batch 32 --version 1 --overwrite --yes
#
#   Llama-3.1-8B baseline on 2xH100, 65K ctx:
#   ./serve_and_bench.sh llama-3-8b-vllm --tp 2 --max-len 65536 \
#       --extra "--gpu-memory-utilization 0.92 --max-num-seqs 64" \
#       -- --suite cybersoceval --batch 64 --version 1 --overwrite --yes
#
#   Qwen2.5 with YaRN to 65K (only if 32K leaves a tail of overflows on
#   the longest TI rows; YaRN slightly degrades short-prompt quality, so
#   use sparingly). Note --hf-overrides not --rope-scaling -- the latter
#   is not a vllm CLI flag:
#   ./serve_and_bench.sh athena-cti-sft-qwen25-14b-v12-plus-taa-vllm \
#       --tp 2 --max-len 65536 \
#       --extra "--gpu-memory-utilization 0.92 --max-num-seqs 16 \
#                --hf-overrides '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}}'" \
#       -- --suite cybersoceval --batch 16 --version 1 --overwrite --yes
#
# Env vars:
#   READY_TIMEOUT     seconds to wait for /v1/models (default 1800)
#                     First-time cold-cache runs can easily exceed 15 min
#                     once HF download + torch compile + cudagraph capture
#                     are added up (Gemma-2 captures ~50 sizes x TP ranks).
#   READY_POLL        poll interval in seconds (default 5)
#   BENCH_CONDA_ENV   conda env to use for the bench client. When set, the
#                     run_benchmark.sh invocation is wrapped with
#                     `conda run -n $BENCH_CONDA_ENV` so the bench picks up
#                     pandas/transformers/openai from a different env than
#                     the one serving vllm. When unset, this script auto-
#                     detects: it first tries the currently active shell,
#                     then probes 'ctibench' and 'llm-sft' (the names
#                     created by SFT/utils/setup.sh), and uses the first
#                     env that can import the critical deps (pandas,
#                     openai, transformers, tqdm). If none qualify, the
#                     script aborts BEFORE starting vllm so the user does
#                     not waste a 10-minute warm-up on a doomed bench step.
#                     Pass BENCH_CONDA_ENV=<name> explicitly to override
#                     the auto-detection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

READY_TIMEOUT="${READY_TIMEOUT:-1800}"
READY_POLL="${READY_POLL:-5}"
BENCH_CONDA_ENV="${BENCH_CONDA_ENV:-}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '3,83p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

# --- Bench env resolution -------------------------------------------------
# The bench client needs pandas / openai / transformers / tqdm. When this
# script is launched from the isolated 'vllm' env (which omits those by
# design), running run_benchmark.sh directly fails on the first import
# AFTER vllm has already finished its 5-15 min warm-up -- a costly silent
# trap. Resolution rules (in order):
#   1. If BENCH_CONDA_ENV is set: validate it exists AND has the deps;
#      fail closed if either check fails.
#   2. If BENCH_CONDA_ENV is unset: probe the current shell first, then
#      'ctibench', then 'llm-sft' (the env names produced by setup.sh).
#      First candidate that imports the deps wins. If none qualify,
#      abort with a clear actionable error.
# All probes happen BEFORE serve_vllm.sh is launched so a doomed run dies
# in <2 s, not >15 min.

_BENCH_PROBE_IMPORTS='import pandas, openai, transformers, tqdm'

# Probe a single env (or the current shell when name is empty). Echoes
# nothing on success; emits a one-line failure diagnostic to stderr on
# failure. Returns 0 iff all critical imports succeed.
_env_has_bench_deps() {
    local env_name="$1"
    if [[ -z "${env_name}" ]]; then
        python -c "${_BENCH_PROBE_IMPORTS}" >/dev/null 2>&1
        return $?
    fi
    if ! command -v conda >/dev/null 2>&1; then
        return 1
    fi
    # `conda run` returns the inner command's exit code; suppress its own
    # stderr noise (which prepends 'CondaError:' on missing-env failures
    # we already handle separately) so the caller sees only the verdict.
    conda run -n "${env_name}" python -c "${_BENCH_PROBE_IMPORTS}" \
        >/dev/null 2>&1
}

# Convenience: does the named env exist at all? Used to give better error
# messages (missing env vs. env exists but missing deps).
_conda_env_exists() {
    local env_name="$1"
    command -v conda >/dev/null 2>&1 || return 1
    conda env list 2>/dev/null | awk '{print $1}' | grep -qx "${env_name}"
}

if [[ -n "${BENCH_CONDA_ENV}" ]]; then
    # User-supplied: must exist AND import the deps. No fallback -- an
    # explicit override should never silently pick a different env.
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: BENCH_CONDA_ENV='${BENCH_CONDA_ENV}' set but 'conda' not on PATH." >&2
        exit 5
    fi
    if ! _conda_env_exists "${BENCH_CONDA_ENV}"; then
        echo "ERROR: conda env '${BENCH_CONDA_ENV}' not found. Available envs:" >&2
        conda env list >&2 || true
        exit 6
    fi
    if ! _env_has_bench_deps "${BENCH_CONDA_ENV}"; then
        echo "ERROR: conda env '${BENCH_CONDA_ENV}' is missing one or more bench-client deps." >&2
        echo "       Required: pandas openai transformers tqdm" >&2
        echo "       Reproduce: conda run -n ${BENCH_CONDA_ENV} python -c '${_BENCH_PROBE_IMPORTS}'" >&2
        echo "       Fix     : conda run -n ${BENCH_CONDA_ENV} pip install -r ${TEST_DIR}/requirements.txt" >&2
        exit 7
    fi
    BENCH_ENV_RESOLUTION="explicit (BENCH_CONDA_ENV=${BENCH_CONDA_ENV})"
else
    # Auto-detect. Try the inheriting path first so users with a single
    # combined env (--mode all into one --env-name) keep the existing
    # zero-config behaviour. Then probe the canonical setup.sh env names.
    BENCH_ENV_RESOLUTION=""
    if _env_has_bench_deps ""; then
        BENCH_CONDA_ENV=""
        BENCH_ENV_RESOLUTION="auto: inherit current shell (${CONDA_DEFAULT_ENV:-<no conda env active>}) -- imports OK"
    else
        for _candidate in ctibench llm-sft; do
            if _conda_env_exists "${_candidate}" \
                && _env_has_bench_deps "${_candidate}"; then
                BENCH_CONDA_ENV="${_candidate}"
                BENCH_ENV_RESOLUTION="auto: '${_candidate}' (current shell '${CONDA_DEFAULT_ENV:-<none>}' lacked deps)"
                break
            fi
        done
    fi
    if [[ -z "${BENCH_ENV_RESOLUTION}" ]]; then
        echo "ERROR: could not find a conda env (or current shell) with the bench-client deps." >&2
        echo "       Probed: current shell '${CONDA_DEFAULT_ENV:-<none>}', then 'ctibench', then 'llm-sft'." >&2
        echo "       Required: pandas openai transformers tqdm" >&2
        echo "       Fix options:" >&2
        echo "         a) bash ${SFT_DIR}/utils/setup.sh --mode test         # creates 'ctibench'" >&2
        echo "         b) BENCH_CONDA_ENV=<your-env> $(basename "${BASH_SOURCE[0]}") ..." >&2
        exit 8
    fi
fi
# --------------------------------------------------------------------------

ALIAS="$1"; shift

serve_args=()
bench_args=()
seen_sep=0
for arg in "$@"; do
    if [[ ${seen_sep} -eq 0 && "${arg}" == "--" ]]; then
        seen_sep=1
        continue
    fi
    if [[ ${seen_sep} -eq 0 ]]; then
        serve_args+=("${arg}")
    else
        bench_args+=("${arg}")
    fi
done

# Resolve <alias> -> HF repo id. Parse pipelines/models.py via ast rather
# than importing it: the module imports torch/transformers at top level,
# which is heavyweight and unavailable on non-GPU hosts. ast-parsing keeps
# this wrapper usable from any shell while staying in sync with the one
# authoritative model_mapping dict.
REPO_ID="$(python - "${TEST_DIR}/pipelines/models.py" "${ALIAS}" <<'PY'
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
if mapping is None:
    sys.stderr.write("model_mapping not found in pipelines/models.py\n")
    sys.exit(1)
if alias not in mapping:
    sys.stderr.write(f"unknown alias: {alias}\n")
    sys.exit(2)
if not alias.endswith("-vllm"):
    sys.stderr.write(f"alias must end with '-vllm' for serve_and_bench: {alias}\n")
    sys.exit(3)
print(mapping[alias])
PY
)"

# Pull --port out of serve_args (if present) so we can poll the right URL;
# default 8000 otherwise. Keep the flag itself in serve_args so it reaches
# serve_vllm.sh unchanged.
PORT=8000
for i in "${!serve_args[@]}"; do
    if [[ "${serve_args[$i]}" == "--port" && $((i+1)) -lt ${#serve_args[@]} ]]; then
        PORT="${serve_args[$((i+1))]}"
    fi
done

if [[ ${#serve_args[@]} -eq 0 ]]; then
    serve_args=(--tp 2 --max-len 4096)
fi
if [[ ${#bench_args[@]} -eq 0 ]]; then
    bench_args=(--suite athena --version 2 --batch 64)
fi

TS="$(date +%Y%m%d_%H%M%S)"
SAFE_ALIAS="${ALIAS//\//_}"
SERVE_LOG="${TEST_DIR}/${SAFE_ALIAS}_serve_${TS}.log"
BENCH_LOG="${TEST_DIR}/${SAFE_ALIAS}_bench_${TS}.log"

echo "=== serve_and_bench.sh ==="
echo "  alias      : ${ALIAS}"
echo "  repo id    : ${REPO_ID}"
echo "  port       : ${PORT}"
echo "  serve args : ${serve_args[*]}"
echo "  bench args : ${bench_args[*]}"
echo "  bench env  : ${BENCH_CONDA_ENV:-<inherit current shell>}"
echo "  bench resv : ${BENCH_ENV_RESOLUTION}"
echo "  serve log  : ${SERVE_LOG}"
echo "  bench log  : ${BENCH_LOG}"
echo

# Launch serve_vllm.sh in its own process group so we can signal the whole
# tree on teardown (vllm spawns per-TP worker procs that a plain pid kill
# would orphan).
setsid bash "${SCRIPT_DIR}/serve_vllm.sh" --model "${REPO_ID}" "${serve_args[@]}" \
    >"${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
SERVE_PGID="$(ps -o pgid= "${SERVE_PID}" | tr -d ' ')"

cleanup() {
    local rc=$?
    echo
    echo "=== tearing down vllm server (pgid=${SERVE_PGID}) ==="
    kill -TERM "-${SERVE_PGID}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "${SERVE_PID}" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "-${SERVE_PGID}" 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT INT TERM

echo "=== waiting for http://localhost:${PORT}/v1/models (timeout ${READY_TIMEOUT}s) ==="
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "  ready."
        break
    fi
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "ERROR: vllm serve exited before becoming ready. Tail of log:" >&2
        tail -40 "${SERVE_LOG}" >&2 || true
        exit 3
    fi
    if [[ $(date +%s) -ge ${deadline} ]]; then
        echo "ERROR: vllm did not become ready within ${READY_TIMEOUT}s." >&2
        tail -40 "${SERVE_LOG}" >&2 || true
        exit 4
    fi
    sleep "${READY_POLL}"
done

echo
echo "=== launching benchmark ==="
# When BENCH_CONDA_ENV is set, wrap the bench client with `conda run` so it
# picks up pandas/transformers/openai from a separate env than the one
# serving vllm. --no-capture-output keeps stdout/stderr streaming live to
# the tee'd log instead of being buffered until the subprocess exits.
if [[ -n "${BENCH_CONDA_ENV}" ]]; then
    conda run --no-capture-output -n "${BENCH_CONDA_ENV}" \
        bash "${SCRIPT_DIR}/run_benchmark.sh" "${ALIAS}" "${bench_args[@]}" \
        2>&1 | tee "${BENCH_LOG}"
else
    bash "${SCRIPT_DIR}/run_benchmark.sh" "${ALIAS}" "${bench_args[@]}" \
        2>&1 | tee "${BENCH_LOG}"
fi
