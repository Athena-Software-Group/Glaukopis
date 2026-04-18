#!/bin/bash

# End-to-end setup script for the SFT (LlamaFactory) pipeline on a Linux CUDA
# machine.
#
# Installs (as needed):
#   1. Miniconda (if `conda` is not on PATH)
#   2. A conda environment with Python 3.11
#   3. PyTorch matched to the requested CUDA version
#   4. LlamaFactory (editable install of this directory)
#   5. Optional extras: metrics, deepspeed, vllm (opt-in via --extras)
#   6. wandb + huggingface_hub
#   7. flash-attn (optional, non-fatal; skipped with --no-flash-attn)
#
# Usage:
#   ./setup.sh [--cuda cu124|cu121|cu118|cpu] [--env-name NAME] [--python VERSION]
#              [--extras "metrics deepspeed"] [--no-flash-attn] [--no-conda-init]
#
# Defaults:
#   --cuda cu124
#   --env-name llm-sft
#   --python 3.11
#   --extras "metrics deepspeed"
#   (conda init runs by default for your shell; use --no-conda-init to skip)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_TAG="cu124"
ENV_NAME="llm-sft"
PYTHON_VERSION="3.11"
EXTRAS="metrics deepspeed"
INSTALL_FLASH_ATTN=1
RUN_CONDA_INIT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)           CUDA_TAG="$2"; shift 2 ;;
        --env-name)       ENV_NAME="$2"; shift 2 ;;
        --python)         PYTHON_VERSION="$2"; shift 2 ;;
        --extras)         EXTRAS="$2"; shift 2 ;;
        --no-flash-attn)  INSTALL_FLASH_ATTN=0; shift ;;
        --no-conda-init)  RUN_CONDA_INIT=0; shift ;;
        -h|--help)
            sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "${CUDA_TAG}" in
    cu124|cu121|cu118) TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}" ;;
    cpu)               TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
    *) echo "Unsupported --cuda value: ${CUDA_TAG} (expected cu124|cu121|cu118|cpu)"; exit 1 ;;
esac

echo "=== SFT (LlamaFactory) setup ==="
echo "  sft dir   : ${SFT_DIR}"
echo "  env name  : ${ENV_NAME}"
echo "  python    : ${PYTHON_VERSION}"
echo "  cuda tag  : ${CUDA_TAG}"
echo "  extras    : ${EXTRAS:-<none>}"
echo "  flash-attn: $([[ ${INSTALL_FLASH_ATTN} -eq 1 ]] && echo yes || echo no)"
echo

# 1. Miniconda bootstrap ------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "=== conda not found — installing Miniconda to \$HOME/miniconda3 ==="
    MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    curl -L -o "${MINICONDA_INSTALLER}" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "${MINICONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
    rm -f "${MINICONDA_INSTALLER}"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2. Conda environment --------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "=== Reusing existing conda env: ${ENV_NAME} ==="
else
    echo "=== Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION}) ==="
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel setuptools

# 3. PyTorch ------------------------------------------------------------------
echo "=== Installing PyTorch (${CUDA_TAG}) ==="
pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio

# 4. LlamaFactory (editable) --------------------------------------------------
echo "=== Installing LlamaFactory in editable mode ==="
pip install -e "${SFT_DIR}"

# 5. Optional extras ----------------------------------------------------------
for extra in ${EXTRAS}; do
    req_file="${SFT_DIR}/requirements/${extra}.txt"
    if [[ -f "${req_file}" ]]; then
        echo "=== Installing extras: ${extra} ==="
        pip install -r "${req_file}"
    else
        echo "  [WARN] requirements/${extra}.txt not found — skipping"
    fi
done

# 6. Training/experiment tooling ---------------------------------------------
echo "=== Installing wandb + huggingface_hub ==="
pip install wandb huggingface_hub

# 7. flash-attn (optional) ----------------------------------------------------
# flash-attn is installed opportunistically: training configs can use it when
# available, but failures are non-fatal because the prebuilt wheels are pinned
# to a specific torch x cuda x python combo and often ABI-mismatch against the
# version pip resolves. flash-attn's setup.py also has an EXDEV bug when
# $CONDA_PREFIX and $PIP_CACHE_DIR live on different filesystems (e.g. RunPod:
# /root vs /home), which is why we try a prebuilt wheel first.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

install_flash_attn() {
    local info
    info="$(python - <<'PY'
import torch, sys
tv = torch.__version__.split("+")[0]
tv_mm = ".".join(tv.split(".")[:2])
cu = (torch.version.cuda or "").replace(".", "")
cu_major = cu[:2] if cu else ""
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"{tv_mm} {cu_major} {py_tag}")
PY
)"
    read -r TORCH_MM CU_MAJOR PY_TAG <<< "${info}"

    if [[ -z "${CU_MAJOR}" ]]; then
        echo "  [WARN] torch has no CUDA build; skipping flash-attn"
        return 0
    fi

    local wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu${CU_MAJOR}torch${TORCH_MM}cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    echo "  Trying prebuilt wheel: ${wheel_url}"
    if pip install --no-build-isolation "${wheel_url}"; then
        return 0
    fi

    echo "  Prebuilt wheel failed; retrying standard install with pinned TMPDIR/PIP_CACHE_DIR..."
    export TMPDIR="${HOME}/tmp" PIP_CACHE_DIR="${HOME}/.cache/pip"
    mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
    pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
}

if [[ ${INSTALL_FLASH_ATTN} -eq 1 && "${CUDA_TAG}" != "cpu" ]]; then
    echo "=== Installing flash-attn (v${FLASH_ATTN_VERSION}) — optional, non-fatal ==="
    set +e
    install_flash_attn
    fa_status=$?
    set -e
    if [[ ${fa_status} -ne 0 ]]; then
        echo "  [WARN] flash-attn install failed (exit ${fa_status}); continuing without it."
        echo "         Trainer configs that request flash-attn must be adjusted or"
        echo "         this install must be repaired to match the local torch ABI."
    fi
elif [[ "${CUDA_TAG}" == "cpu" ]]; then
    echo "=== Skipping flash-attn (CPU build) ==="
fi

# 8. Verification -------------------------------------------------------------
echo
echo "=== Verifying PyTorch / CUDA / LlamaFactory ==="
python - <<'PY'
import torch
print("torch version     :", torch.__version__)
print("torch cuda build  :", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device            :", torch.cuda.get_device_name(0))
    print("bf16 supported    :", torch.cuda.is_bf16_supported())
try:
    import llamafactory
    print("llamafactory      :", getattr(llamafactory, "__version__", "installed"))
except Exception as e:
    print("llamafactory import failed:", e)
PY

if command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli  : $(command -v llamafactory-cli)"
else
    echo "  [WARN] llamafactory-cli not on PATH after install"
fi

# 9. Shell integration --------------------------------------------------------
# In a fresh interactive shell, `conda activate` fails with
#   "Run 'conda init' before 'conda activate'"
# unless the conda shell hook has been installed into the user's rc file.
# `conda init` is idempotent: it will not duplicate the block on reruns.
if [[ ${RUN_CONDA_INIT} -eq 1 ]]; then
    target_shell="$(basename "${SHELL:-/bin/bash}")"
    case "${target_shell}" in
        bash|zsh|fish)
            echo
            echo "=== Running 'conda init ${target_shell}' ==="
            conda init "${target_shell}" || true
            ;;
        *)
            echo "  [WARN] Unsupported shell '${target_shell}' for conda init; skipping."
            target_shell=""
            ;;
    esac
fi

echo
echo "=== Setup complete ==="
if [[ ${RUN_CONDA_INIT} -eq 1 && -n "${target_shell:-}" ]]; then
    echo "Start a new shell (or run 'exec ${target_shell}') to pick up the conda hook,"
    echo "then activate the environment with:"
else
    echo "Activate the environment with:"
fi
echo "    conda activate ${ENV_NAME}"
echo "Then launch training, e.g.:"
echo "    cd ${SFT_DIR}"
echo "    bash ift_training_qwen_2.5_14b.sh"
echo "Remember to 'wandb login' and 'huggingface-cli login' before training if needed."
