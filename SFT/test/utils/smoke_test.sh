#!/bin/bash

# Smoke test for an SFT/test install.
#
# Verifies — without needing API keys or downloading any HF model weights — that:
#   1. python / pip are on PATH and point at the expected env
#   2. PyTorch is importable and (optionally) has CUDA
#   3. Core SFT/test Python packages import cleanly
#   4. The inference.py CLI parses its arguments (--help exits 0)
#   5. Benchmark data files exist (LFS pulled) for the standard Athena tasks
#
# Usage:
#   conda activate ctibench         # or your custom env
#   ./smoke_test.sh [--require-cuda] [--env-name NAME]
#
# Flags:
#   --require-cuda   Fail if torch.cuda.is_available() is False
#   --env-name NAME  Warn if the active conda env name does not match

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REQUIRE_CUDA=0
EXPECTED_ENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --require-cuda) REQUIRE_CUDA=1; shift ;;
        --env-name)     EXPECTED_ENV="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

pass() { echo "  [ OK ] $1"; }
fail() { echo "  [FAIL] $1" >&2; exit 1; }
warn() { echo "  [WARN] $1"; }

echo "=== SFT/test smoke test ==="
echo "  bench dir : ${BENCH_DIR}"
echo "  python    : $(command -v python || echo '(none)')"
echo "  env       : ${CONDA_DEFAULT_ENV:-<none>}"
echo

# 1. Basic interpreter --------------------------------------------------------
command -v python >/dev/null 2>&1 || fail "python not found on PATH"
pass "python on PATH ($(python --version 2>&1))"

if [[ -n "${EXPECTED_ENV}" && "${CONDA_DEFAULT_ENV}" != "${EXPECTED_ENV}" ]]; then
    warn "active conda env '${CONDA_DEFAULT_ENV:-<none>}' != expected '${EXPECTED_ENV}'"
fi

# 2. Torch / CUDA -------------------------------------------------------------
TORCH_OUT="$(python - <<'PY'
import json, sys
try:
    import torch
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
    sys.exit(0)
info = {
    "ok": True,
    "version": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if info["cuda_available"]:
    info["device"] = torch.cuda.get_device_name(0)
    info["bf16"] = torch.cuda.is_bf16_supported()
print(json.dumps(info))
PY
)"
echo "  torch info: ${TORCH_OUT}"

case "${TORCH_OUT}" in
    *'"ok": true'*) pass "torch importable" ;;
    *) fail "torch import failed: ${TORCH_OUT}" ;;
esac

case "${TORCH_OUT}" in
    *'"cuda_available": true'*)  pass "CUDA available" ;;
    *) if [[ ${REQUIRE_CUDA} -eq 1 ]]; then fail "CUDA not available"; else warn "CUDA not available"; fi ;;
esac

# 3. Package imports ----------------------------------------------------------
(cd "${BENCH_DIR}" && python - <<'PY'
import importlib
targets = [
    "benchmarks",
    "benchmarks.athena_mcq",
    "benchmarks.cti_mcq",
    "pipelines.models",
    "pipelines.data_loader",
]
for name in targets:
    importlib.import_module(name)
print("imports ok")
PY
) >/dev/null 2>&1 && pass "SFT/test modules import" \
                  || fail "SFT/test module import failed (run manually from ${BENCH_DIR} to see trace)"

# 4. inference.py CLI ---------------------------------------------------------
(cd "${BENCH_DIR}" && python inference.py --help >/dev/null 2>&1) \
    && pass "inference.py --help runs" \
    || fail "inference.py --help failed"

# 5. Benchmark data files (LFS pulled) ---------------------------------------
REQUIRED_FILES=(
    "benchmark_data/athena_bench/athena-cti-mcq-3k.jsonl"
    "benchmark_data/athena_bench/athena-cti-rcm.jsonl"
    "benchmark_data/athena_bench/athena-cti-vsp.jsonl"
    "benchmark_data/athena_bench/athena-cti-ate.jsonl"
    "benchmark_data/athena_bench/athena-cti-rms.jsonl"
)

missing=0
for rel in "${REQUIRED_FILES[@]}"; do
    path="${BENCH_DIR}/${rel}"
    if [[ ! -s "${path}" ]]; then
        warn "missing or empty: ${rel} (run 'git lfs pull' in ${BENCH_DIR})"
        missing=1
    else
        # LFS pointer files start with 'version https://git-lfs...'
        if head -n1 "${path}" | grep -q "^version https://git-lfs"; then
            warn "LFS pointer (not fetched): ${rel} (run 'git lfs pull')"
            missing=1
        fi
    fi
done

if [[ ${missing} -eq 0 ]]; then
    pass "benchmark data files present"
fi

echo
echo "=== Smoke test complete ==="
echo "Next: run a tiny real benchmark, e.g."
echo "    python inference.py athena-mcq gemini-2.5-flash --rows 2 --batch 2 \\"
echo "        --data_path benchmark_data_mini/athena-cti-mcq.jsonl --version 99"
