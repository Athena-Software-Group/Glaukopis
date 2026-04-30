#!/bin/bash

# End-to-end setup script for the SFT pipeline on a Linux CUDA machine.
# By default (--mode all) installs both the LlamaFactory training stack into
# `llm-sft` and the SFT/test benchmarking stack into `ctibench` (the two-env
# layout the rest of the repo + docs assume). Pass --env-name NAME together
# with --mode all to collapse both stacks into a single named env instead.
# Use --mode train|test|vllm to install only one side. --split-envs is kept
# as an explicit alias of the default split behavior.
#
# --mode vllm additionally creates the ctibench bench-client env by default,
# because a vllm server with no client to drive it has no use case in this
# repo (serve_and_bench.sh requires BENCH_CONDA_ENV=ctibench to run the
# bench loop while vllm holds the GPU). Pass --no-bench-env to suppress
# the ctibench co-install (e.g. dedicated serving-only nodes).
#
# Installs (as needed):
#   1. Miniconda (if `conda` is not on PATH)
#   2. Conda env(s) with Python 3.11
#   3. PyTorch matched to the requested CUDA version
#   4. [train] LlamaFactory (editable install of SFT/) + training extras
#              (metrics, deepspeed) + wandb + huggingface_hub + python-dotenv
#   5. [test]  SFT/test requirements + git-lfs + CyberSOCEval data fetch
#              (Athena/CTI/CyberMetric data are committed regular files; only
#              CyberSOCEval requires post-checkout downloads, which run via
#              SFT/test/utils/fetch_cybersoceval_data.py unless --skip-cybersoceval)
#   6. [vllm]  vllm + openai client into an isolated env (default: vllm),
#              plus the [test] stack into ctibench unless --no-bench-env
#   7. flash-attn (optional, non-fatal; skipped with --no-flash-attn)
#   8. Bootstraps SFT/.env from SFT/.env.example on first run
#   9. Configures global git identity when --git-user-name/--git-user-email
#      (or GIT_USER_NAME/GIT_USER_EMAIL env vars) are provided; otherwise
#      warns at the end if no global identity is set on this box
#  10. Reclaims disk after install: prunes pip cache, conda package tarballs,
#      apt cache, stale serve/bench logs (>7 days), /tmp scratch dirs, and
#      vacuums journald to 200 MB. HF model caches under ~/.cache/huggingface
#      are NEVER touched (use --no-disk-cleanup to skip entirely; the post-
#      cleanup summary lists the largest HF cache entries so manual pruning
#      can be done deliberately).
#
# Usage:
#   ./setup.sh [--mode all|train|test|vllm]
#              [--cuda cu124|cu121|cu118|cpu]
#              [--env-name NAME] [--python VERSION]
#              [--extras "metrics deepspeed"] [--no-flash-attn]
#              [--lfs-pull] [--split-envs] [--no-conda-init]
#              [--skip-cybersoceval] [--no-bench-env]
#              [--git-user-name "Your Name"] [--git-user-email you@example.com]
#              [--no-disk-cleanup]
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
#   (vllm mode also installs ctibench by default; --no-bench-env opts out)
#   (git identity left untouched unless both --git-user-name and
#    --git-user-email are passed, or GIT_USER_NAME/GIT_USER_EMAIL exported)
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
FETCH_CYBERSOCEVAL=1
INSTALL_BENCH_WITH_VLLM=1
RUN_DISK_CLEANUP=1
# Git identity: empty default. Picks up GIT_USER_NAME / GIT_USER_EMAIL
# from the environment (so .env or shell exports work) and is overridden
# by the --git-user-name / --git-user-email flags. Only applied when both
# values are present; otherwise we just warn at the end so a fresh box
# doesn't get a surprise empty author on its first commit.
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)               MODE="$2"; shift 2 ;;
        --cuda)               CUDA_TAG="$2"; shift 2 ;;
        --env-name)           ENV_NAME="$2"; ENV_NAME_EXPLICIT=1; shift 2 ;;
        --python)             PYTHON_VERSION="$2"; shift 2 ;;
        --extras)             EXTRAS="$2"; shift 2 ;;
        --no-flash-attn)      INSTALL_FLASH_ATTN=0; shift ;;
        --lfs-pull)           RUN_LFS_PULL=1; shift ;;
        --split-envs)         SPLIT_ENVS=1; shift ;;
        --no-conda-init)      RUN_CONDA_INIT=0; shift ;;
        --skip-cybersoceval)  FETCH_CYBERSOCEVAL=0; shift ;;
        --no-bench-env)       INSTALL_BENCH_WITH_VLLM=0; shift ;;
        --no-disk-cleanup)    RUN_DISK_CLEANUP=0; shift ;;
        --git-user-name)      GIT_USER_NAME="$2"; shift 2 ;;
        --git-user-email)     GIT_USER_EMAIL="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
echo "  cybersoce : $([[ ${FETCH_CYBERSOCEVAL} -eq 1 ]] && echo yes || echo no)"
if [[ "${MODE}" == "vllm" ]]; then
    echo "  bench env : $([[ ${INSTALL_BENCH_WITH_VLLM} -eq 1 ]] && echo 'yes (ctibench co-install)' || echo no)"
fi
echo

# Git identity ----------------------------------------------------------------
# Fresh boxes (containers, EC2 spot, Modal, RunPod, etc.) ship with no
# global git identity, which causes any subsequent `git commit` to fail
# loudly and any `git pull --rebase` to refuse implicit merges. Set the
# global config when the operator passed both --git-user-name and
# --git-user-email (or exported GIT_USER_NAME / GIT_USER_EMAIL, including
# via SFT/.env); otherwise leave any existing config alone and warn at the
# end if it's still empty.
#
# Source SFT/.env early so GIT_USER_NAME / GIT_USER_EMAIL persisted there
# are visible here. Done before the bootstrap further down because that
# bootstrap only creates the file from .env.example (no real values in it
# yet) and because the git identity logic must run before any later step
# that might invoke `git`. CLI flags (--git-user-name / --git-user-email)
# and pre-existing shell exports take precedence over .env values; we
# snapshot whatever was set before sourcing and restore afterwards.
if [[ -f "${SFT_DIR}/.env" ]]; then
    _git_user_name_pre="${GIT_USER_NAME:-}"
    _git_user_email_pre="${GIT_USER_EMAIL:-}"
    set -a
    # shellcheck source=/dev/null
    source "${SFT_DIR}/.env"
    set +a
    [[ -n "${_git_user_name_pre}"  ]] && GIT_USER_NAME="${_git_user_name_pre}"
    [[ -n "${_git_user_email_pre}" ]] && GIT_USER_EMAIL="${_git_user_email_pre}"
    unset _git_user_name_pre _git_user_email_pre
fi
GIT_IDENTITY_WAS_SET=0
GIT_IDENTITY_MISSING=0
if command -v git >/dev/null 2>&1; then
    if [[ -n "${GIT_USER_NAME}" && -n "${GIT_USER_EMAIL}" ]]; then
        echo "=== Setting global git identity ==="
        echo "  user.name : ${GIT_USER_NAME}"
        echo "  user.email: ${GIT_USER_EMAIL}"
        git config --global user.name  "${GIT_USER_NAME}"
        git config --global user.email "${GIT_USER_EMAIL}"
        GIT_IDENTITY_WAS_SET=1
    elif [[ -z "$(git config --global --get user.name 2>/dev/null)" \
         || -z "$(git config --global --get user.email 2>/dev/null)" ]]; then
        GIT_IDENTITY_MISSING=1
    fi
    echo
fi

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
        # pypdf: utils/fetch_cybersoceval_data.py extracts text from the
        # CyberSOCEval threat-intel PDFs and is the only opt-in step that
        # may run from within this env (the data fetch is otherwise driven
        # from the ctibench env). Cheap dep, no torch interaction, so we
        # bake it in unconditionally rather than gate behind --no-bench-env.
        pip install vllm openai python-dotenv huggingface_hub "pypdf>=5.0.0"

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

        # liger-kernel: fused linear cross-entropy + RMSNorm/RoPE kernels
        # that LlamaFactory enables via --enable_liger_kernel. Required for
        # full SFT of Qwen2.5-32B (152K vocab) on 8x80GB without OOM at the
        # CE step; also a small throughput win on Llama / Qwen2.5 14B.
        # Optional at runtime (skip the flag if uninstalled), but baked in
        # here so the trainer's default extras work out of the box.
        echo "=== Installing liger-kernel ==="
        pip install liger-kernel

        # bitsandbytes: 8-bit AdamW (--optim adamw_8bit) cuts the fp32 Adam
        # momentum + variance (~32 GB/rank for 32B under ZeRO-3) down to
        # ~8 GB, freeing the headroom needed for the LM-head grad_weight
        # accumulation on 8x80GB. Required by run_abaligned_sft_qwen25_32b_v7.sh.
        echo "=== Installing bitsandbytes ==="
        pip install bitsandbytes
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

        # CyberSOCEval data fetch ------------------------------------------
        # AthenaBench / CTI-Bench / CyberMetric files are committed regular
        # git files and need no post-checkout step. CyberSOCEval is the one
        # benchmark whose corpus must be pulled at install time: the
        # malware-analysis questions + hybrid-analysis sandbox JSONs come
        # from CrowdStrike/CyberSOCEval_data, the threat-intel question set
        # comes from meta-llama/PurpleLlama, and per-question PDFs are
        # downloaded from the upstream vendors then converted to text via
        # pypdf (already pinned in SFT/test/requirements.txt). Idempotent
        # on rerun. Skipped with --skip-cybersoceval (e.g. air-gapped hosts
        # or training-only nodes that won't run benchmarks).
        if [[ ${FETCH_CYBERSOCEVAL} -eq 1 ]]; then
            echo "=== Fetching CyberSOCEval data (CrowdStrike + Meta PurpleLlama) ==="
            set +e
            (cd "${TEST_DIR}" && python utils/fetch_cybersoceval_data.py \
                --out-dir "${TEST_DIR}/benchmark_data/cybersoceval" \
                --cache-dir "${TEST_DIR}/benchmark_data/cybersoceval/_cyberSOCEval_data")
            local cse_status=$?
            set -e
            if [[ ${cse_status} -ne 0 ]]; then
                echo "  [WARN] CyberSOCEval fetch exited with status ${cse_status}."
                echo "         The cybersoceval-malware / cybersoceval-ti benchmarks"
                echo "         will fail until you rerun:"
                echo "             cd ${TEST_DIR} && python utils/fetch_cybersoceval_data.py"
                echo "         Other benchmark suites (athena, ctibench, cybermetric)"
                echo "         are unaffected."
            fi
        else
            echo "=== Skipping CyberSOCEval data fetch (--skip-cybersoceval) ==="
            echo "         To populate later:"
            echo "             cd ${TEST_DIR} && python utils/fetch_cybersoceval_data.py"
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
    try:
        import liger_kernel
        print("liger_kernel      :", getattr(liger_kernel, "__version__", "installed"))
    except Exception as e:
        print("liger_kernel import failed:", e)
    try:
        import bitsandbytes as bnb
        print("bitsandbytes      :", getattr(bnb, "__version__", "installed"))
    except Exception as e:
        print("bitsandbytes import failed:", e)
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
# --mode vllm additionally installs the test stack into ctibench so the bench
# client (run_benchmark.sh / inference.py) has a runnable env to drive the
# server from. Suppressed by --no-bench-env.
INSTALLED_ENVS=()
if [[ ${SPLIT_ENVS} -eq 1 ]]; then
    install_stack "llm-sft"  "train"
    install_stack "ctibench" "test"
    INSTALLED_ENVS=("llm-sft" "ctibench")
else
    install_stack "${ENV_NAME}" "${MODE}"
    INSTALLED_ENVS=("${ENV_NAME}")
    if [[ "${MODE}" == "vllm" && ${INSTALL_BENCH_WITH_VLLM} -eq 1 ]]; then
        if [[ "${ENV_NAME}" == "ctibench" ]]; then
            echo "  [WARN] --env-name ctibench collides with bench co-install; skipping."
        else
            install_stack "ctibench" "test"
            INSTALLED_ENVS+=("ctibench")
        fi
    fi
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

# Persist HF + wandb auth at the box level -----------------------------------
# Sourcing SFT/.env in every wrapper script is fragile (each new tool has to
# remember to do it; multi-process spawns sometimes lose the env). Persist
# the credentials globally instead:
#   * HF_TOKEN  -> ~/.cache/huggingface/token   (read by huggingface_hub
#                  for *every* Python process on the box, no env required)
#   * WANDB_API_KEY -> ~/.netrc                  (same idea for wandb)
# Skipped silently when the value is the .env.example placeholder, when
# .env doesn't exist, or when the relevant CLI isn't installed (e.g. test
# mode without the train stack).
if [[ -f "${ENV_FILE:-}" ]] && [[ ${NEEDS_ENV_EDIT} -eq 0 ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a

    # HF: prefer the new `hf` CLI (huggingface_hub >=0.27); fall back to
    # huggingface-cli for older installs. Either path writes the same token
    # file, so downstream code paths don't care.
    if [[ -n "${HF_TOKEN:-}" && "${HF_TOKEN}" != "hf_xxx_replace_me" ]]; then
        for stack_env in "${INSTALLED_ENVS[@]}"; do
            if conda run --no-capture-output -n "${stack_env}" \
                    python -c "import huggingface_hub" 2>/dev/null; then
                echo "=== Persisting HF_TOKEN to ~/.cache/huggingface/token via '${stack_env}' env ==="
                conda run --no-capture-output -n "${stack_env}" \
                    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)" \
                    >/dev/null 2>&1 \
                    && echo "  ok." \
                    || echo "  [WARN] hf login failed; export HF_TOKEN manually if downloads fail."
                break
            fi
        done
    fi

    # wandb: only persist if explicitly set (and not the placeholder). wandb
    # login writes to ~/.netrc which any wandb client picks up.
    if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != "wandb_xxx_replace_me" ]]; then
        for stack_env in "${INSTALLED_ENVS[@]}"; do
            if conda run --no-capture-output -n "${stack_env}" \
                    python -c "import wandb" 2>/dev/null; then
                echo "=== Persisting WANDB_API_KEY to ~/.netrc via '${stack_env}' env ==="
                conda run --no-capture-output -n "${stack_env}" \
                    wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 \
                    && echo "  ok." \
                    || echo "  [WARN] wandb login failed; export WANDB_API_KEY manually if needed."
                break
            fi
        done
    fi
fi

# Disk cleanup ---------------------------------------------------------------
# Conda + pip leave behind several GB of redundant tarballs/wheels after a
# fresh env build, and prior bench/serve sessions accumulate /tmp scratch,
# stale logs, and journald spool. None of these are needed at runtime; pip
# and conda redownload on demand. HF model caches under ~/.cache/huggingface
# are intentionally left alone — those are large but they are exactly the
# artefacts the bench client needs to load. The summary at the end lists
# the top 10 HF cache entries so the operator can prune deliberately.
# Skipped with --no-disk-cleanup.
cleanup_disk() {
    local label="$1"
    if command -v df >/dev/null 2>&1; then
        echo "  [${label}] $(df -h /home / 2>/dev/null | awk 'NR==1 || /\/home$|\/$/' | tr '\n' ' | ')"
    fi
}

if [[ ${RUN_DISK_CLEANUP} -eq 1 ]]; then
    echo
    echo "=== Disk cleanup (safe targets only; HF model cache untouched) ==="
    cleanup_disk "before"

    # pip wheel cache. Frees 1-5 GB after a fresh vllm/torch install.
    if command -v pip >/dev/null 2>&1; then
        pip cache purge >/dev/null 2>&1 \
            && echo "  ok: pip cache purged" \
            || echo "  skip: pip cache purge failed (non-fatal)"
    fi

    # conda's redundant package tarballs (already extracted into envs/).
    # Frees 2-10 GB after building llm-sft + ctibench + vllm envs.
    if command -v conda >/dev/null 2>&1; then
        conda clean --all --yes >/dev/null 2>&1 \
            && echo "  ok: conda package tarballs cleaned" \
            || echo "  skip: conda clean failed (non-fatal)"
    fi

    # apt cache (Debian/Ubuntu only, root only). Frees 100 MB - 1 GB.
    if command -v apt-get >/dev/null 2>&1 && [[ $(id -u) -eq 0 ]]; then
        apt-get clean >/dev/null 2>&1 \
            && echo "  ok: apt cache cleaned" \
            || echo "  skip: apt-get clean failed (non-fatal)"
    fi

    # Old serve/bench logs (>7 days) under SFT/test. Each run produces two
    # multi-MB logs; on a long-lived bench host these add up.
    if [[ -d "${TEST_DIR}" ]]; then
        REMOVED_LOGS=0
        REMOVED_LOGS=$(find "${TEST_DIR}" -maxdepth 1 -name '*_serve_*.log' -mtime +7 -print -delete 2>/dev/null | wc -l)
        REMOVED_LOGS=$((REMOVED_LOGS + $(find "${TEST_DIR}" -maxdepth 1 -name '*_bench_*.log' -mtime +7 -print -delete 2>/dev/null | wc -l)))
        echo "  ok: pruned ${REMOVED_LOGS} stale serve/bench log(s) older than 7 days"
    fi

    # Stale /tmp scratch from prior runs. torchinductor caches per-shape
    # kernels here; vllm's CUDA_VISIBLE_DEVICES probes leave _v81_probe etc.
    rm -rf /tmp/torchinductor_* /tmp/_v81_probe /tmp/v81_train.log 2>/dev/null \
        && echo "  ok: pruned /tmp scratch (torchinductor, probes)" \
        || true

    # journald spool (root only on systems with journalctl).
    if command -v journalctl >/dev/null 2>&1 && [[ $(id -u) -eq 0 ]]; then
        journalctl --vacuum-size=200M >/dev/null 2>&1 \
            && echo "  ok: journald vacuumed to 200M" \
            || echo "  skip: journalctl vacuum failed (non-fatal)"
    fi

    cleanup_disk "after "

    # Surface the HF cache footprint so the operator can decide whether to
    # prune individual model dirs by hand. We don't auto-delete here because
    # the operator may legitimately have multiple bench targets cached.
    HF_HUB_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
    if [[ -d "${HF_HUB_DIR}" ]]; then
        echo
        echo "  HF model cache footprint (largest 10 entries; prune manually if needed):"
        du -sh "${HF_HUB_DIR}"/* 2>/dev/null | sort -rh | head -10 | sed 's/^/    /'
        echo "  Total: $(du -sh "${HF_HUB_DIR}" 2>/dev/null | awk '{print $1}')"
        echo "  To remove a single model: rm -rf ${HF_HUB_DIR}/models--<org>--<name>"
    fi
else
    echo
    echo "=== Disk cleanup skipped (--no-disk-cleanup) ==="
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
if [[ ${GIT_IDENTITY_MISSING} -eq 1 ]]; then
    echo "Git identity is not configured globally. Any 'git commit' on this box"
    echo "will fail until you set it. Either rerun setup with the flags:"
    echo "    bash SFT/utils/setup.sh --git-user-name 'Your Name' --git-user-email you@example.com ..."
    echo "or configure manually now:"
    echo "    git config --global user.name  'Your Name'"
    echo "    git config --global user.email you@example.com"
    echo
elif [[ ${GIT_IDENTITY_WAS_SET} -eq 1 ]]; then
    echo "Git identity configured: ${GIT_USER_NAME} <${GIT_USER_EMAIL}>"
    echo
fi
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
    if [[ ${INSTALL_BENCH_WITH_VLLM} -eq 1 ]]; then
        echo "One-shot serve+bench (vllm holds the GPU, ctibench drives the client):"
        echo "    cd ${TEST_DIR}"
        echo "    BENCH_CONDA_ENV=ctibench ./utils/serve_and_bench.sh <alias>-vllm \\"
        echo "        --tp 1 --max-len 8192 \\"
        echo "        -- --suite all --version 1 --batch 64 --overwrite --yes"
        echo
        echo "Or split it across two terminals:"
        echo "    # terminal 1 (vllm env)"
        echo "    conda activate ${ENV_NAME}"
        echo "    bash ${TEST_DIR}/utils/serve_vllm.sh --model <hf-repo-id> --tp 1"
        echo "    # terminal 2 (ctibench env)"
        echo "    conda activate ctibench"
        echo "    cd ${TEST_DIR}/utils"
        echo "    ./run_benchmark.sh <alias>-vllm --batch 64"
    else
        echo "Launch a vLLM server (terminal 1):"
        echo "    conda activate ${ENV_NAME}"
        echo "    bash ${TEST_DIR}/utils/serve_vllm.sh --model <hf-repo-id> --tp 2"
        echo
        echo "Run a benchmark against it (terminal 2, in the ctibench/llm-sft env):"
        echo "    cd ${TEST_DIR}/utils"
        echo "    ./run_benchmark.sh <alias>-vllm --batch 64"
    fi
fi
