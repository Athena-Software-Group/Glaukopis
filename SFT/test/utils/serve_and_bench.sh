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
# Env vars:
#   READY_TIMEOUT   seconds to wait for /v1/models (default 900)
#   READY_POLL      poll interval in seconds (default 5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

READY_TIMEOUT="${READY_TIMEOUT:-900}"
READY_POLL="${READY_POLL:-5}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

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
bash "${SCRIPT_DIR}/run_benchmark.sh" "${ALIAS}" "${bench_args[@]}" \
    2>&1 | tee "${BENCH_LOG}"
