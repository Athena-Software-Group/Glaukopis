#!/bin/bash

# Register a trained+pushed HF model in athena_bench/pipelines/models.py
# (idempotent) and run athenabench against it.
#
# Run this AFTER train.sh has pushed the model to
# https://huggingface.co/<HF_USERNAME>/<project_name>.
#
# Usage:
#   ./run_athenabench.sh [--repo-id USER/NAME] [--alias NAME] [--env-name NAME]
#                        [--smoke-only] [--rows N] [--batch N]
#                        [--tasks "athena-mcq athena-rcm ..."]
#
# Defaults:
#   --repo-id       ${HF_USERNAME}/llama3.1-8b-athena-ift
#   --alias         llama3.1-8b-athena-ift
#   --env-name      ctibench
#   --rows          2    (smoke test size)
#   --tasks         "athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-rms"
#
# Flow:
#   1. Verify the HF repo exists.
#   2. Patch athena_bench/pipelines/models.py to add the alias if missing.
#   3. Activate the ctibench env.
#   4. Run a 2-row smoke test on athena-mcq.
#   5. If not --smoke-only and the smoke test passes, run the full sweep
#      via athena_bench/utils/run_benchmark.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_PY="${REPO_ROOT}/athena_bench/pipelines/models.py"
BENCH_SCRIPT="${REPO_ROOT}/athena_bench/utils/run_benchmark.sh"

REPO_ID=""
ALIAS="llama3.1-8b-athena-ift"
ENV_NAME="ctibench"
ROWS=2
BATCH=""
TASKS="athena-mcq athena-rcm athena-vsp athena-ate athena-taa athena-rms"
SMOKE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)    REPO_ID="$2"; shift 2 ;;
        --alias)      ALIAS="$2"; shift 2 ;;
        --env-name)   ENV_NAME="$2"; shift 2 ;;
        --rows)       ROWS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --tasks)      TASKS="$2"; shift 2 ;;
        --smoke-only) SMOKE_ONLY=1; shift ;;
        -h|--help) sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME or pass --repo-id}"
    REPO_ID="${HF_USERNAME}/llama3.1-8b-athena-ift"
fi

echo "=== athenabench run ==="
echo "  repo_id  : ${REPO_ID}"
echo "  alias    : ${ALIAS}"
echo "  env      : ${ENV_NAME}"
echo "  rows     : ${ROWS} (smoke)"
echo "  tasks    : ${TASKS}"
echo "  mode     : $([[ ${SMOKE_ONLY} -eq 1 ]] && echo 'smoke only' || echo 'smoke + full sweep')"
echo

# 1. HF repo sanity check ------------------------------------------------------
echo "=== Verifying HF model repo ==="
python - "${REPO_ID}" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id = sys.argv[1]
try:
    info = HfApi().model_info(repo_id)
    files = {s.rfilename for s in (info.siblings or [])}
    required = {"config.json"}
    missing = required - files
    if missing:
        sys.exit(f"[FAIL] repo {repo_id} is missing required files: {missing}")
    print(f"  OK: {repo_id} ({len(files)} files)")
except Exception as e:
    sys.exit(f"[FAIL] cannot read repo {repo_id}: {e}")
PY

# 2. Idempotent registry patch -------------------------------------------------
echo "=== Registering '${ALIAS}' in athena_bench/pipelines/models.py ==="
python - "${MODELS_PY}" "${ALIAS}" "${REPO_ID}" <<'PY'
import ast, sys
from pathlib import Path

path, alias, repo_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
src = path.read_text()
tree = ast.parse(src)

mapping_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "model_mapping":
                mapping_node = node.value
if mapping_node is None:
    sys.exit("[FAIL] could not locate model_mapping dict in models.py")

existing = ast.literal_eval(mapping_node)
if alias in existing:
    if existing[alias] == repo_id:
        print(f"  already registered: '{alias}' -> '{repo_id}' (no-op)")
        sys.exit(0)
    sys.exit(f"[FAIL] '{alias}' already maps to '{existing[alias]}', not '{repo_id}'")

lines = src.splitlines(keepends=True)
close_line = mapping_node.end_lineno - 1   # 0-indexed
new_entry = f"    '{alias}': '{repo_id}',\n"
lines.insert(close_line, new_entry)
path.write_text("".join(lines))
print(f"  inserted: '{alias}' -> '{repo_id}' at line {close_line + 1}")
PY

# 3. Activate ctibench --------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[FAIL] conda not found" >&2; exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 4. Smoke test ---------------------------------------------------------------
echo
echo "=== Smoke test (${ROWS} rows, task=athena-mcq) ==="
cd "${REPO_ROOT}/athena_bench"
SMOKE_ARGS=(athena-mcq "${ALIAS}" --rows "${ROWS}" --version 99)
[[ -n "${BATCH}" ]] && SMOKE_ARGS+=(--batch "${BATCH}")
python inference.py "${SMOKE_ARGS[@]}"

if [[ ${SMOKE_ONLY} -eq 1 ]]; then
    echo
    echo "=== Smoke test complete (--smoke-only given; skipping full sweep) ==="
    exit 0
fi

# 5. Full sweep ---------------------------------------------------------------
echo
echo "=== Full sweep ==="
FULL_ARGS=("${ALIAS}" --tasks "${TASKS}")
[[ -n "${BATCH}" ]] && FULL_ARGS+=(--batch "${BATCH}")
"${BENCH_SCRIPT}" "${FULL_ARGS[@]}"
