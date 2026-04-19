#!/bin/bash

# One-time environment setup for running HuggingFace AutoTrain Advanced
# locally against a GPU box. AutoTrain pins a number of packages
# aggressively (transformers==4.48.0, huggingface-hub==0.27.0, etc.), so
# we deliberately isolate it from the main llm-sft conda env used by
# LlamaFactory.
#
# Usage:
#   ./setup.sh [--env-name NAME] [--python VERSION] [--autotrain-version SPEC]
#              [--recreate] [--no-conda-init]
#
# Defaults:
#   --env-name           autotrain
#   --python             3.11
#   --autotrain-version  0.8.36      (latest on PyPI as of 2025-04)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="autotrain"
PYTHON_VERSION="3.11"
AUTOTRAIN_VERSION="0.8.36"
RECREATE=0
RUN_CONDA_INIT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)           ENV_NAME="$2"; shift 2 ;;
        --python)             PYTHON_VERSION="$2"; shift 2 ;;
        --autotrain-version)  AUTOTRAIN_VERSION="$2"; shift 2 ;;
        --recreate)           RECREATE=1; shift ;;
        --no-conda-init)      RUN_CONDA_INIT=0; shift ;;
        -h|--help) sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=== AutoTrain setup ==="
echo "  env name          : ${ENV_NAME}"
echo "  python            : ${PYTHON_VERSION}"
echo "  autotrain version : ${AUTOTRAIN_VERSION}"
echo

if ! command -v conda >/dev/null 2>&1; then
    echo "[FAIL] conda not found. Run SFT/utils/setup.sh first or install Miniconda." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ ${RECREATE} -eq 1 ]]; then
        echo "=== Removing existing conda env: ${ENV_NAME} ==="
        conda deactivate 2>/dev/null || true
        conda env remove -n "${ENV_NAME}" -y
        echo "=== Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION}) ==="
        conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    else
        echo "=== Reusing existing conda env: ${ENV_NAME} (use --recreate to rebuild) ==="
    fi
else
    echo "=== Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION}) ==="
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel setuptools
# autotrain-advanced pins its entire dependency tree exactly
# (transformers==4.48.0, huggingface-hub==0.27.0, accelerate==1.2.1, ...).
# Installing anything else afterwards risks breaking that tree; let pip
# resolve the closure in a single pass.
python -m pip install "autotrain-advanced==${AUTOTRAIN_VERSION}"

echo
echo "=== Verification ==="
autotrain --version || { echo "[FAIL] autotrain CLI not on PATH" >&2; exit 1; }
python - <<'PY'
import importlib.metadata as md
def _v(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return "not installed"
print("autotrain         :", _v("autotrain-advanced"))
print("transformers      :", _v("transformers"))
print("huggingface_hub   :", _v("huggingface-hub"))
print("accelerate        :", _v("accelerate"))
print("peft              :", _v("peft"))
print("trl               :", _v("trl"))
try:
    from autotrain.cli.autotrain import main  # noqa: F401
    print("autotrain import  : ok")
except Exception as e:
    print("autotrain import  : FAIL ->", e)
    raise SystemExit(1)
import torch
print("torch             :", torch.__version__)
print("cuda available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device            :", torch.cuda.get_device_name(0))
PY

if [[ ${RUN_CONDA_INIT} -eq 1 ]]; then
    target_shell="$(basename "${SHELL:-/bin/bash}")"
    case "${target_shell}" in
        bash|zsh|fish) conda init "${target_shell}" || true ;;
    esac
fi

echo
echo "=== Next steps ==="
echo "1. Activate the env:"
echo "     conda activate ${ENV_NAME}"
echo "2. Export HF credentials (needed by prepare_dataset.sh and train.sh):"
echo "     export HF_TOKEN=hf_xxx...   # write-scope token"
echo "     export HF_USERNAME=<your-hf-user>"
echo "3. Prepare the dataset:"
echo "     ${SCRIPT_DIR}/prepare_dataset.sh"
echo "4. Launch training:"
echo "     ${SCRIPT_DIR}/train.sh"
echo "5. After training pushes the model, run the benchmark:"
echo "     ${SCRIPT_DIR}/run_athenabench.sh"
