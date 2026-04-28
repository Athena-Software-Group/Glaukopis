#!/bin/bash

# End-to-end setup script for the SFT pipeline on a Linux CUDA machine.
# By default (--mode all) installs both the LlamaFactory training stack into
# `llm-sft` and the SFT/test benchmarking stack into `ctibench` (the two-env
# layout the rest of the repo + docs assume). Pass --env-name NAME together
# with --mode all to collapse both stacks into a single named env instead.
# Use --mode train|test|vllm to install only one side. --split-envs is kept
# as an explicit alias of the default split behavior.
#
# Installs (as needed):
#   1. Miniconda (if `conda` is not on PATH)
#   2. Conda env(s) with Python 3.11
#   3. PyTorch matched to the requested CUDA version
#   4. [train] LlamaFactory (editable install of SFT/) + training extras
#              (metrics, deepspeed) + wandb + huggingface_hub + python-dotenv
#   5. [test]  SFT/test requirements + git-lfs
#   6. [vllm]  vllm + openai client into an isolated env (default: vllm)
#   7. flash-attn (optional, non-fatal; skipped with --no-flash-attn)
#   8. Bootstraps SFT/.env from SFT/.env.example on first run
#
# Usage:
#   ./setup.sh [--mode all|train|test|vllm]
#              [--cuda cu124|cu121|cu118|cpu]
#              [--env-name NAME] [--python VERSION]
#              [--extras "metrics deepspeed"] [--no-flash-attn]
#              [--lfs-pull] [--split-envs] [--no-conda-init]
#
# Defaults:
#   --mode all           (creates llm-sft + ctibench; pass --env-name to
#                         force everything into a single named env instead)
#   --cuda cu124
#   --env-name <unset>   (train -> llm-sft, test -> ctibench, vllm -> vllm,
#                         all -> llm-sft + ctibench)
#   --python 3.11
#   --extras "metrics deepspeed"
#   (conda init runs by default for your shell; use --no-conda-init to skip)
#
# Dependency note:
#   LlamaFactory pins datasets<=4.0.0 and transformers<=5.2.0; SFT/test asks
#   for datasets>=4.0.0 and transformers>=4.56.2. The joint solution is
#   datasets==4.0.0 and transformers 4.56.2..5.2.0, which pip resolves cleanly
#   when LlamaFactory is installed first (the order used below).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_DIR="${SFT_DIR}/test"

MODE="all"
CUDA_TAG="cu124"
ENV_NAME=""
ENV_NAME_EXPLICIT=0
PYTHON_VERSION="3.11"
EXTRAS="metrics deepspeed"
INSTALL_FLASH_ATTN=1
RUN_LFS_PULL=0
SPLIT_ENVS=0
RUN_CONDA_INIT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)           MODE="$2"; shift 2 ;;
        --cuda)           CUDA_TAG="$2"; shift 2 ;;
        --env-name)       ENV_NAME="$2"; ENV_NAME_EXPLICIT=1; shift 2 ;;
        --python)         PYTHON_VERSION="$2"; shift 2 ;;
        --extras)         EXTRAS="$2"; shift 2 ;;
        --no-flash-attn)  INSTALL_FLASH_ATTN=0; shift ;;
        --lfs-pull)       RUN_LFS_PULL=1; shift ;;
        --split-envs)     SPLIT_ENVS=1; shift ;;
        --no-conda-init)  RUN_CONDA_INIT=0; shift ;;
        -h|--help)
            sed -n '3,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "${MODE}" in
    all|train|test|vllm) ;;
    *) echo "Unsupported --mode value: ${MODE} (expected all|train|test|vllm)"; exit 1 ;;
esac

# `--mode all` without an explicit `--env-name` is treated as the two-env
# layout (llm-sft for training, ctibench for testing). Earlier this script
# silently put both stacks into a single `llm-sft` env, which surprised every
# caller who then looked for `ctibench` to run benchmarks. If the user did
# pass `--env-name FOO` together with `--mode all`, that's a clear opt-in to
# the single-env collapse, so we honor it.
if [[ "${MODE}" == "all" && ${ENV_NAME_EXPLICIT} -eq 0 && ${SPLIT_ENVS} -eq 0 ]]; then
    SPLIT_ENVS=1
fi

# Default env names. --split-envs overrides ENV_NAME entirely (it runs two
# passes, one per stack) so it's checked in the dispatch block further down.
if [[ -z "${ENV_NAME}" ]]; then
    case "${MODE}" in
        test) ENV_NAME="ctibench" ;;
        vllm) ENV_NAME="vllm" ;;
        *)    ENV_NAME="llm-sft" ;;
    esac
fi

case "${CUDA_TAG}" in
    cu124|cu121|cu118) TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}" ;;
    cpu)               TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
    *) echo "Unsupported --cuda value: ${CUDA_TAG} (expected cu124|cu121|cu118|cpu)"; exit 1 ;;
esac

echo "=== SFT unified setup ==="
echo "  sft dir   : ${SFT_DIR}"
echo "  test dir  : ${TEST_DIR}"
echo "  mode      : ${MODE}$([[ ${SPLIT_ENVS} -eq 1 ]] && echo ' (split-envs)')"
echo "  env name  : ${ENV_NAME}$([[ ${SPLIT_ENVS} -eq 1 ]] && echo ' (ignored: split-envs uses llm-sft + ctibench)')"
echo "  python    : ${PYTHON_VERSION}"
echo "  cuda tag  : ${CUDA_TAG}"
echo "  extras    : ${EXTRAS:-<none>}"
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

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Install one stack ({train|test|all}) into a named conda env. Caller is
# responsible for passing a valid stack string; env is created on first use
# and reused on subsequent runs. Ordered so LlamaFactory's explicit version
# pins are resolved first, then relaxed (>=) test requirements fit on top.
install_stack() {
    local env="$1"
    local stack="$2"

    if conda env list | awk '{print $1}' | grep -qx "${env}"; then
        echo "=== Reusing existing conda env: ${env} ==="
    else
        echo "=== Creating conda env: ${env} (python=${PYTHON_VERSION}) ==="
        conda create -n "${env}" "python=${PYTHON_VERSION}" -y
    fi
    conda activate "${env}"

    python -m pip install --upgrade pip wheel setuptools

    # vllm ships its own torch pin; installing torch first causes pip to
    # downgrade/upgrade it when `pip install vllm` runs, which wastes ~2min
    # and sometimes leaves a broken env. Skip the explicit torch install
    # in vllm mode and let vllm resolve its own dependency tree.
    if [[ "${stack}" != "vllm" ]]; then
        echo "=== Installing PyTorch (${CUDA_TAG}) into ${env} ==="
        pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
    fi

    if [[ "${stack}" == "vllm" ]]; then
        echo "=== Installing vllm + openai client into ${env} ==="
        pip install vllm openai python-dotenv huggingface_hub

        # DeepGEMM: required by vllm's kernel warmup path on Hopper/Blackwell.
        # vllm imports it unconditionally during engine init when DeepGEMM is
        # enabled by default (PR vllm-project/vllm#24462), and the probe
        # raises a hard RuntimeError if the package is missing or outdated --
        # even for bf16 models that don't actually use FP8 kernels. Pin to
        # v2.1.1.post3 which is the version recommended in the vllm issue
        # thread (vllm-project/vllm#29946) for current vllm releases.
        # Non-fatal: the install needs nvcc + CUDA toolkit and may fail in
        # CPU-only smoke tests; in that case the user can either rerun
        # later with toolkit available, or skip the warmup at runtime via
        # `export VLLM_DEEP_GEMM_WARMUP=skip VLLM_USE_DEEP_GEMM=0`.
        if [[ "${CUDA_TAG}" != "cpu" ]]; then
            echo "=== Installing deep_gemm (vllm warmup dep) into ${env} ==="
            set +e
            pip install --no-build-isolation \
                "git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3"
            dg_status=$?
            set -e
            if [[ ${dg_status} -ne 0 ]]; then
                echo "  [WARN] deep_gemm install failed (exit ${dg_status}). vllm"
                echo "         will crash at engine init unless you set"
                echo "         VLLM_USE_DEEP_GEMM=0 and VLLM_DEEP_GEMM_WARMUP=skip"
                echo "         before launching, or rerun this install once nvcc"
                echo "         + CUDA toolkit are available."
            fi
        fi
    fi

    if [[ "${stack}" == "train" || "${stack}" == "all" ]]; then
        echo "=== Installing LlamaFactory in editable mode into ${env} ==="
        pip install -e "${SFT_DIR}"

        for extra in ${EXTRAS}; do
            req_file="${SFT_DIR}/requirements/${extra}.txt"
            if [[ -f "${req_file}" ]]; then
                echo "=== Installing extras: ${extra} ==="
                pip install -r "${req_file}"
            else
                echo "  [WARN] requirements/${extra}.txt not found — skipping"
            fi
        done

        # python-dotenv: upload_to_hf.py reads HF_TOKEN / HUGGINGFACE_TOKEN
        # from .env files at SFT/ (and repo root) as one of its credential
        # sources. ninja + packaging: accelerate deepspeed's first-run op
        # compilation (parallel builds) and are required by deepspeed setup.
        echo "=== Installing wandb + huggingface_hub + python-dotenv + ninja ==="
        pip install wandb huggingface_hub python-dotenv packaging ninja
    fi

    if [[ "${stack}" == "test" || "${stack}" == "all" ]]; then
        echo "=== Installing SFT/test requirements into ${env} ==="
        pip install -r "${TEST_DIR}/requirements.txt"
    fi

    # flash-attn is only meaningful for the training / transformers-based
    # inference paths. vllm bundles its own attention kernels internally.
    if [[ "${stack}" != "vllm" && ${INSTALL_FLASH_ATTN} -eq 1 && "${CUDA_TAG}" != "cpu" ]]; then
        echo "=== Installing flash-attn (v${FLASH_ATTN_VERSION}) — optional, non-fatal ==="
        set +e
        install_flash_attn
        local fa_status=$?
        set -e
        if [[ ${fa_status} -ne 0 ]]; then
            echo "  [WARN] flash-attn install failed (exit ${fa_status}); continuing without it."
            echo "         The benchmark runner defaults to SDPA; trainer configs that"
            echo "         request flash-attn must be adjusted or this install repaired."
        fi
    elif [[ "${CUDA_TAG}" == "cpu" ]]; then
        echo "=== Skipping flash-attn (CPU build) ==="
    fi

    if [[ "${stack}" == "test" || "${stack}" == "all" ]]; then
        if ! command -v git-lfs >/dev/null 2>&1; then
            echo "=== Installing git-lfs via conda-forge into ${env} ==="
            conda install -n "${env}" -c conda-forge git-lfs -y
        fi
        git lfs install

        if [[ ${RUN_LFS_PULL} -eq 1 ]]; then
            echo "=== Running 'git lfs pull' in ${TEST_DIR} (requested via --lfs-pull) ==="
            if ! (cd "${TEST_DIR}" && git lfs pull); then
                echo "  [WARN] 'git lfs pull' reported errors."
                echo "         Many LFS objects under data/ are missing from the server;"
                echo "         benchmark runs do not depend on data/."
            fi
        else
            echo "=== Skipping 'git lfs pull' (default; pass --lfs-pull to opt in) ==="
        fi
    fi

    local verify_label="PyTorch / CUDA"
    if [[ "${stack}" == "train" || "${stack}" == "all" ]]; then
        verify_label="${verify_label} / LlamaFactory"
    elif [[ "${stack}" == "vllm" ]]; then
        verify_label="${verify_label} / vLLM"
    fi
    echo
    echo "=== Verifying ${verify_label} in ${env} ==="
    python - "${stack}" <<'PY'
import sys
stack = sys.argv[1]
import torch
print("torch version     :", torch.__version__)
print("torch cuda build  :", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device            :", torch.cuda.get_device_name(0))
    print("bf16 supported    :", torch.cuda.is_bf16_supported())
if stack in ("train", "all"):
    try:
        import llamafactory
        print("llamafactory      :", getattr(llamafactory, "__version__", "installed"))
    except Exception as e:
        print("llamafactory import failed:", e)
if stack == "vllm":
    try:
        import vllm
        print("vllm              :", getattr(vllm, "__version__", "installed"))
    except Exception as e:
        print("vllm import failed:", e)
    try:
        import deep_gemm
        print("deep_gemm         :", getattr(deep_gemm, "__version__", "installed"))
    except Exception as e:
        print("deep_gemm import failed:", e)
PY

    if [[ "${stack}" == "train" || "${stack}" == "all" ]]; then
        if command -v llamafactory-cli >/dev/null 2>&1; then
            echo "llamafactory-cli  : $(command -v llamafactory-cli)"
        else
            echo "  [WARN] llamafactory-cli not on PATH after install"
        fi
    fi
    if [[ "${stack}" == "vllm" ]]; then
        if command -v vllm >/dev/null 2>&1; then
            echo "vllm cli          : $(command -v vllm)"
        else
            echo "  [WARN] vllm cli not on PATH after install"
        fi
    fi
}

# flash-attn helper ----------------------------------------------------------
# flash-attn is installed opportunistically: training configs can use it when
# available, but failures are non-fatal because the prebuilt wheels are pinned
# to a specific torch x cuda x python combo and often ABI-mismatch against the
# version pip resolves. flash-attn's setup.py also has an EXDEV bug when
# $CONDA_PREFIX and $PIP_CACHE_DIR live on different filesystems (e.g. RunPod:
# /root vs /home), which is why we try a prebuilt wheel first. The runtime
# defaults to SDPA when flash-attn is absent, so this is best-effort.
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

# Dispatch: run install_stack for each requested env/stack pair ---------------
# --split-envs ignores MODE and always runs both stacks in two separate envs
# (llm-sft for training, ctibench for testing), preserving the legacy layout.
INSTALLED_ENVS=()
if [[ ${SPLIT_ENVS} -eq 1 ]]; then
    install_stack "llm-sft"  "train"
    install_stack "ctibench" "test"
    INSTALLED_ENVS=("llm-sft" "ctibench")
else
    install_stack "${ENV_NAME}" "${MODE}"
    INSTALLED_ENVS=("${ENV_NAME}")
fi

# Shell integration -----------------------------------------------------------
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

# .env bootstrap --------------------------------------------------------------
# Copy SFT/.env.example -> SFT/.env on first run so the user has a single
# place to drop HF / wandb credentials. upload_to_hf.py, run_train.sh, and
# autotrain/run_abaligned_sft.sh all auto-source this file. Skipped in test-
# only mode since those tooling hooks aren't exercised there.
NEEDS_ENV_EDIT=0
if [[ "${MODE}" != "test" || ${SPLIT_ENVS} -eq 1 ]]; then
    ENV_FILE="${SFT_DIR}/.env"
    ENV_EXAMPLE="${SFT_DIR}/.env.example"
    if [[ ! -f "${ENV_FILE}" && -f "${ENV_EXAMPLE}" ]]; then
        cp "${ENV_EXAMPLE}" "${ENV_FILE}"
        chmod 600 "${ENV_FILE}" 2>/dev/null || true
        NEEDS_ENV_EDIT=1
        echo
        echo "=== Bootstrapped ${ENV_FILE} from .env.example ==="
    elif [[ -f "${ENV_FILE}" ]] && grep -q 'hf_xxx_replace_me\|your-hf-username' "${ENV_FILE}"; then
        NEEDS_ENV_EDIT=1
    fi
fi

echo
echo "=== Setup complete ==="
if [[ ${RUN_CONDA_INIT} -eq 1 && -n "${target_shell:-}" ]]; then
    echo "Start a new shell (or run 'exec ${target_shell}') to pick up the conda hook,"
    echo "then activate one of the environments:"
else
    echo "Activate one of the environments:"
fi
for env in "${INSTALLED_ENVS[@]}"; do
    echo "    conda activate ${env}"
done
echo
if [[ ${NEEDS_ENV_EDIT} -eq 1 ]]; then
    echo "Fill in HF / wandb credentials (placeholders still present):"
    echo "    \$EDITOR ${ENV_FILE}"
    echo "    # set HF_TOKEN (write-scope) and HF_USERNAME at minimum"
elif [[ "${MODE}" != "test" || ${SPLIT_ENVS} -eq 1 ]]; then
    echo "Credentials file: ${ENV_FILE}  (already populated)"
fi
echo
echo "Alternative to editing .env: run the interactive CLIs once:"
echo "    hf auth login            # REQUIRED (Llama-3.1-8B-Instruct is a gated model)"
echo "    wandb login              # optional, only needed if passing --report-to wandb"
echo
if [[ "${MODE}" == "train" || "${MODE}" == "all" || ${SPLIT_ENVS} -eq 1 ]]; then
    echo "run_train.sh, upload_to_hf.py, and autotrain/run_abaligned_sft.sh all"
    echo "auto-source ${SFT_DIR}/.env -- no manual 'export' needed."
    echo
    echo "Launch training (defaults: Llama-3.1-8B-Instruct LoRA on ift_data_2026_04_20):"
    echo "    cd ${SFT_DIR}"
    echo "    bash utils/run_train.sh --dry-run   # inspect the command, no training"
    echo "    bash utils/run_train.sh             # kick off the real run"
    echo
fi
if [[ "${MODE}" == "test" || "${MODE}" == "all" || ${SPLIT_ENVS} -eq 1 ]]; then
    echo "Launch a benchmark (e.g. Athena MCQ):"
    echo "    cd ${TEST_DIR}"
    echo "    python inference.py athena-mcq <model_name> --batch 5 --version 1 \\"
    echo "        --data_path benchmark_data/athena_bench/athena-mcq.tsv"
fi
if [[ "${MODE}" == "vllm" ]]; then
    echo "Launch a vLLM server (terminal 1):"
    echo "    conda activate ${ENV_NAME}"
    echo "    bash ${TEST_DIR}/utils/serve_vllm.sh --model <hf-repo-id> --tp 2"
    echo
    echo "Run a benchmark against it (terminal 2, in the ctibench/llm-sft env):"
    echo "    cd ${TEST_DIR}/utils"
    echo "    ./run_benchmark.sh <alias>-vllm --batch 64"
fi
