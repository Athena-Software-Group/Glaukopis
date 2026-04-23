#!/bin/bash

# End-to-end setup script for athena_bench on a Linux CUDA machine.
#
# Installs (as needed):
#   1. Miniconda (if `conda` is not on PATH)
#   2. A conda environment with Python 3.11
#   3. PyTorch matched to the requested CUDA version
#   4. athena_bench Python dependencies from requirements.txt
#   5. flash-attn (optional, non-fatal; the runtime defaults to SDPA)
#   6. Git LFS, and runs `git lfs pull` to fetch the large files under data/
#
# Usage:
#   ./setup.sh [--cuda cu124|cu121|cu118|cpu] [--env-name NAME] [--python VERSION]
#              [--no-flash-attn] [--lfs-pull] [--no-conda-init]
#
# Defaults:
#   --cuda cu124
#   --env-name ctibench
#   --python 3.11
#   (conda init runs by default for your shell; use --no-conda-init to skip)
#
# Note on Git LFS:
#   'git lfs pull' is NOT run by default. The benchmark data under
#   benchmark_data/ is tracked as regular git files and does not need LFS.
#   The only LFS-tracked content lives under data/ (scrape output used for
#   dataset *generation*, not for running benchmarks) and a number of those
#   objects are missing from the LFS server. Pass --lfs-pull to attempt a
#   pull anyway; failures will be reported as a warning but will not abort
#   the setup.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_TAG="cu124"
ENV_NAME="ctibench"
PYTHON_VERSION="3.11"
INSTALL_FLASH_ATTN=1
RUN_LFS_PULL=0
RUN_CONDA_INIT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)           CUDA_TAG="$2"; shift 2 ;;
        --env-name)       ENV_NAME="$2"; shift 2 ;;
        --python)         PYTHON_VERSION="$2"; shift 2 ;;
        --no-flash-attn)  INSTALL_FLASH_ATTN=0; shift ;;
        --lfs-pull)       RUN_LFS_PULL=1; shift ;;
        --no-lfs-pull)    RUN_LFS_PULL=0; shift ;;    # kept for backwards compat
        --no-conda-init)  RUN_CONDA_INIT=0; shift ;;
        -h|--help)
            sed -n '3,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

echo "=== athena_bench setup ==="
echo "  bench dir : ${BENCH_DIR}"
echo "  env name  : ${ENV_NAME}"
echo "  python    : ${PYTHON_VERSION}"
echo "  cuda tag  : ${CUDA_TAG}"
echo "  flash-attn: $([[ ${INSTALL_FLASH_ATTN} -eq 1 ]] && echo yes || echo no)"
echo "  git lfs   : $([[ ${RUN_LFS_PULL} -eq 1 ]] && echo yes || echo no)"
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

# Make `conda activate` work inside this non-interactive shell.
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

# 4. athena_bench requirements ------------------------------------------------
echo "=== Installing athena_bench requirements ==="
pip install -r "${BENCH_DIR}/requirements.txt"

# 5. flash-attn (optional) ----------------------------------------------------
# flash-attn is no longer required: the benchmark runner defaults to PyTorch's
# SDPA attention (ATHENA_ATTN_IMPL=sdpa), which dispatches to flash / memory-
# efficient kernels under the hood and avoids the transformers x flash-attn
# version-mismatch bugs that surface on Qwen2-based models.
#
# We still attempt to install flash-attn 2.x here for users who want to opt
# back into it via ATHENA_ATTN_IMPL=flash_attention_2, but failures are
# *non-fatal*: setup completes and the runtime falls back to SDPA.
#
# flash-attn's setup.py also has a known EXDEV bug on hosts where
# $CONDA_PREFIX and $PIP_CACHE_DIR live on different filesystems (e.g. RunPod:
# /root vs /home), which is why we try a prebuilt wheel first.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

install_flash_attn() {
    local info
    info="$(python - <<'PY'
import torch, sys
tv = torch.__version__.split("+")[0]          # "2.6.0"
tv_mm = ".".join(tv.split(".")[:2])           # "2.6"
cu = (torch.version.cuda or "").replace(".", "")  # "124"
cu_major = cu[:2] if cu else ""               # "12"
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
        echo "  [WARN] flash-attn install failed (exit ${fa_status})."
        echo "         This is non-fatal: the runtime uses SDPA by default."
        echo "         Set ATHENA_ATTN_IMPL=flash_attention_2 only if you fix this install."
    fi
elif [[ "${CUDA_TAG}" == "cpu" ]]; then
    echo "=== Skipping flash-attn (CPU build) ==="
fi

# 6. Git LFS ------------------------------------------------------------------
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "=== Installing git-lfs via conda-forge ==="
    conda install -n "${ENV_NAME}" -c conda-forge git-lfs -y
fi
git lfs install

if [[ ${RUN_LFS_PULL} -eq 1 ]]; then
    echo "=== Running 'git lfs pull' (requested via --lfs-pull) ==="
    if ! (cd "${BENCH_DIR}" && git lfs pull); then
        echo "  [WARN] 'git lfs pull' reported errors."
        echo "         This is expected: many LFS objects under data/ are missing"
        echo "         from the server. Benchmark runs do not depend on data/."
    fi
else
    echo "=== Skipping 'git lfs pull' (default; pass --lfs-pull to opt in) ==="
    echo "    benchmark_data/ is plain git-tracked and does not need LFS."
fi

# 7. Verification -------------------------------------------------------------
echo
echo "=== Verifying PyTorch / CUDA ==="
python - <<'PY'
import torch
print("torch version     :", torch.__version__)
print("torch cuda build  :", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device            :", torch.cuda.get_device_name(0))
    print("bf16 supported    :", torch.cuda.is_bf16_supported())
PY

# 8. Shell integration --------------------------------------------------------
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
echo "Then run a benchmark, e.g.:"
echo "    cd ${BENCH_DIR}"
echo "    python inference.py athena-mcq <model_name> --batch 5 --version 1 \\"
echo "        --data_path benchmark_data/athena_bench/athena-mcq.tsv"
