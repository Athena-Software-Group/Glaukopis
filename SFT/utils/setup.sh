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
#              SFT/test/utils/fetch_cybersoceval_data.py. The fetch is gated
#              to --mode test and --mode vllm only -- it is intentionally
#              skipped under --mode all and --mode train so training-only
#              boxes don't burn time pulling PurpleLlama/CrowdStrike PDFs
#              they will never read. --skip-cybersoceval still forces it
#              off in the test/vllm modes; populate later from any host with
#              `cd SFT/test && python utils/fetch_cybersoceval_data.py`.)
#   6. [vllm]  vllm + openai client into an isolated env (default: vllm),
#              plus the [test] stack into ctibench unless --no-bench-env
#   7. flash-attn (only for the train stack; optional, non-fatal; skipped
#      with --no-flash-attn). The vllm stack uses its own attention kernels
#      and the test/bench stack drives vllm over HTTP, so neither imports
#      flash-attn -- avoids a 30-60 min source build with no runtime impact.
#   8. Bootstraps SFT/.env from SFT/.env.example on first run
#   9. Configures global git identity when --git-user-name/--git-user-email
#      (or GIT_USER_NAME/GIT_USER_EMAIL env vars) are provided; otherwise
#      warns at the end if no global identity is set on this box. Also wires
#      up `credential.helper store` + ~/.git-credentials when --git-token
#      (or GITHUB_TOKEN env / .env) is provided so HTTPS pulls/pushes against
#      the private repo work without per-invocation prompts.
#  10. Reclaims disk after install: prunes pip cache, conda package tarballs,
#      apt cache, stale serve/bench logs (>7 days), any *.log files anywhere
#      under SFT/ (>7 days), stale full-SFT checkpoint trees under SFT/saves/
#      (>14 days), /tmp scratch dirs, and vacuums journald to 200 MB. HF model
#      caches under ~/.cache/huggingface are NEVER touched (use --no-disk-
#      cleanup to skip entirely; the post-cleanup summary lists the largest
#      HF cache entries and the largest saves/ entries so manual pruning can
#      be done deliberately).
#
# Usage:
#   ./setup.sh [--mode all|train|test|vllm]
#              [--cuda cu130|cu128|cu126|cu124|cu121|cu118|cpu|auto]
#              [--env-name NAME] [--python VERSION]
#              [--extras "metrics deepspeed"] [--no-flash-attn]
#              [--lfs-pull] [--split-envs] [--no-conda-init]
#              [--skip-cybersoceval] [--no-bench-env]
#              [--git-user-name "Your Name"] [--git-user-email you@example.com]
#              [--git-token ghp_xxx]
#              [--no-disk-cleanup]
#
# Defaults:
#   --mode all           (creates llm-sft + ctibench; pass --env-name to
#                         force everything into a single named env instead)
#   --cuda cu124       (pass --cuda auto to pick the tag matching system nvcc;
#                       cu126/cu128/cu130 are also accepted for newer toolkits)
#   --env-name <unset>   (train -> llm-sft, test -> ctibench, vllm -> vllm,
#                         all -> llm-sft + ctibench)
#   --python 3.11
#   --extras "metrics deepspeed"
#   (conda init runs by default for your shell; use --no-conda-init to skip)
#   (vllm mode also installs ctibench by default; --no-bench-env opts out)
#   (git identity left untouched unless both --git-user-name and
#    --git-user-email are passed, or GIT_USER_NAME/GIT_USER_EMAIL exported)
#   (train stack caps torch at FA_MAX_TORCH_MINOR=2.8 by default so the
#    prebuilt flash-attn 2.8.3 wheel exists; export FA_MAX_TORCH_MINOR=2.x
#    to widen the cap, or pass --no-flash-attn to drop it entirely)
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
CUDA_TAG_EXPLICIT=0
ENV_NAME=""
ENV_NAME_EXPLICIT=0
PYTHON_VERSION="3.11"
EXTRAS="metrics deepspeed"
INSTALL_FLASH_ATTN=1
# Train stack only: cap the torch wheel that the cu sub-index install pulls
# to a minor that flash-attn's current stable line (FA2 v2.8.3, see
# install_flash_attn helper) ships prebuilt wheels for. The cu128 sub-index
# has rolled past torch 2.8 (today: max torch 2.11.0+cu128), but FA 2.8.3
# only publishes wheels for torch 2.4 - 2.8; if torch overshoots the cap,
# the wheel URL 404s, the source-build fallback fails against torch's newer
# API, and llamafactory's --flash_attn auto silently falls back to torch
# SDPA. On Hopper / Blackwell the SDPA cuDNN backend then trips
# "cudnn_frontend Error: No valid execution plans built" for Qwen2-class
# attention shapes under gradient checkpointing -- the exact failure mode
# observed on the 8xH200 v21 32B SFT run. Default 2.8 keeps both the CVE
# floor (test/requirements.txt torch>=2.8.0) and the FA wheel satisfied.
# Override with FA_MAX_TORCH_MINOR=2.x or skip via --no-flash-attn.
FA_MAX_TORCH_MINOR="${FA_MAX_TORCH_MINOR:-2.8}"
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
# GitHub PAT: empty default. Picks up GITHUB_TOKEN from the environment
# (so .env or shell exports work) and is overridden by --git-token. When
# present (and not the .env.example placeholder), enables credential.helper
# store and writes ~/.git-credentials so private-repo HTTPS pulls don't
# fail with "Password authentication is not supported" on fresh boxes.
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)               MODE="$2"; shift 2 ;;
        --cuda)               CUDA_TAG="$2"; CUDA_TAG_EXPLICIT=1; shift 2 ;;
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
        --git-token)          GITHUB_TOKEN="$2"; shift 2 ;;
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

# Auto-detect from nvcc when --cuda was not explicitly passed (or set to
# "auto"). Falls back to the prior cu124 default when nvcc is missing or
# reports a CUDA whose tag isn't supported by PyTorch's wheel index. This
# is the fix for the torchvision/torchaudio mismatch we keep hitting on
# fresh boxes: the default cu124 index ships torch<=2.6, but
# SFT/test/requirements.txt pins torch>=2.8.0 (CVE), which forces pip to
# yank a +cu130 torch from PyPI later -- orphaning torchvision/torchaudio
# at the older cu124 wheels and producing the canonical
# "RuntimeError: operator torchvision::nms does not exist" trip.
if [[ "${CUDA_TAG}" == "auto" || ${CUDA_TAG_EXPLICIT} -eq 0 ]]; then
    detected_cu=""
    if command -v nvcc >/dev/null 2>&1; then
        detected_cu="$(nvcc --version 2>/dev/null \
            | grep -oE 'release [0-9]+\.[0-9]+' \
            | awk '{print $2}' \
            | head -1 \
            | tr -d .)"
    fi
    if [[ -n "${detected_cu}" ]]; then
        candidate="cu${detected_cu}"
        # PyTorch ecosystem cap: as of 2026-05, the whl/cu130 sub-index
        # publishes torch and torchvision but NOT torchaudio, so an
        # auto-detected cu130 host installs torch+cu130 from the index
        # and silently pulls torchaudio (plain, cu128-compiled) from the
        # PyPI fallback, producing a hard runtime trip on first import:
        #   RuntimeError: Detected that PyTorch and TorchAudio were
        #   compiled with different CUDA versions. PyTorch has CUDA
        #   version 13.0 whereas TorchAudio has CUDA version 12.8.
        # Hopper / Blackwell are fully supported by cu128, so cap the
        # auto path there until the cu130 torchaudio wheel set lands.
        # An explicit --cuda cu130 still works (opt-in; user accepts
        # they may need to source-build torchaudio themselves).
        if [[ "${candidate}" == "cu130" ]]; then
            echo "  [auto-detect] system nvcc reports CUDA ${detected_cu}; capping --cuda at cu128 (whl/cu130 lacks torchaudio; pass --cuda cu130 to override)"
            candidate="cu128"
        fi
        case "${candidate}" in
            cu130|cu128|cu126|cu124|cu121|cu118)
                if [[ "${CUDA_TAG}" == "auto" || "${candidate}" != "${CUDA_TAG}" ]]; then
                    echo "  [auto-detect] system nvcc reports CUDA ${detected_cu}; using --cuda ${candidate}"
                fi
                CUDA_TAG="${candidate}"
                ;;
            *)
                echo "  [auto-detect] system nvcc reports CUDA ${detected_cu} (no matching PyTorch wheel index); keeping --cuda ${CUDA_TAG}"
                ;;
        esac
    elif [[ "${CUDA_TAG}" == "auto" ]]; then
        echo "  [auto-detect] nvcc not on PATH; falling back to --cuda cu124"
        CUDA_TAG="cu124"
    fi
fi
case "${CUDA_TAG}" in
    cu130|cu128|cu126|cu124|cu121|cu118)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}" ;;
    cpu)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
    *)
        echo "Unsupported --cuda value: ${CUDA_TAG} (expected cu118|cu121|cu124|cu126|cu128|cu130|cpu|auto)"
        exit 1 ;;
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
# CyberSOCEval fetch is mode-gated: only test and vllm pull the corpus,
# even when FETCH_CYBERSOCEVAL=1 (the default). Surfacing the effective
# decision here avoids the "I asked for --mode all, why didn't the data
# land?" question downstream.
if [[ ${FETCH_CYBERSOCEVAL} -eq 1 && ( "${MODE}" == "test" || "${MODE}" == "vllm" ) ]]; then
    CSE_SUMMARY="yes"
elif [[ ${FETCH_CYBERSOCEVAL} -eq 0 ]]; then
    CSE_SUMMARY="no (--skip-cybersoceval)"
else
    CSE_SUMMARY="no (mode=${MODE}; only test/vllm fetch)"
fi
echo "  cybersoce : ${CSE_SUMMARY}"
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
    _github_token_pre="${GITHUB_TOKEN:-}"
    set -a
    # shellcheck source=/dev/null
    source "${SFT_DIR}/.env"
    set +a
    [[ -n "${_git_user_name_pre}"  ]] && GIT_USER_NAME="${_git_user_name_pre}"
    [[ -n "${_git_user_email_pre}" ]] && GIT_USER_EMAIL="${_git_user_email_pre}"
    [[ -n "${_github_token_pre}"   ]] && GITHUB_TOKEN="${_github_token_pre}"
    unset _git_user_name_pre _git_user_email_pre _github_token_pre
fi
GIT_IDENTITY_WAS_SET=0
GIT_IDENTITY_MISSING=0
GIT_CREDENTIAL_WAS_SET=0
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

    # GitHub HTTPS auth: write ~/.git-credentials so the private repo can be
    # pulled/pushed without per-invocation prompts. GitHub disabled password
    # auth in 2021, so HTTPS clones of private repos require either a PAT
    # or SSH keys. We use the canonical 'x-access-token' username (any
    # non-empty string actually works, but this is what GitHub recommends).
    # Skipped silently when the value is the .env.example placeholder so a
    # fresh `cp .env.example .env` doesn't write a useless credential.
    if [[ -n "${GITHUB_TOKEN}" && "${GITHUB_TOKEN}" != "ghp_xxx_replace_me" ]]; then
        echo "=== Configuring GitHub HTTPS credential helper ==="
        git config --global credential.helper store
        GIT_CREDENTIALS_FILE="${HOME}/.git-credentials"
        # Strip any pre-existing github.com line so re-runs replace stale tokens.
        if [[ -f "${GIT_CREDENTIALS_FILE}" ]]; then
            grep -v '@github.com' "${GIT_CREDENTIALS_FILE}" \
                > "${GIT_CREDENTIALS_FILE}.tmp" 2>/dev/null || true
            mv "${GIT_CREDENTIALS_FILE}.tmp" "${GIT_CREDENTIALS_FILE}"
        fi
        ( umask 077 && \
          printf 'https://x-access-token:%s@github.com\n' "${GITHUB_TOKEN}" \
              >> "${GIT_CREDENTIALS_FILE}" )
        chmod 600 "${GIT_CREDENTIALS_FILE}" 2>/dev/null || true
        GIT_CREDENTIAL_WAS_SET=1
        echo "  ok (helper=store, file=${GIT_CREDENTIALS_FILE})."
    fi
    echo
fi

# 1. Miniconda bootstrap ------------------------------------------------------
# Verda cloud RTX Pro 6000 base image (and other bare CUDA images: lambdalabs,
# vast.ai, plain nvidia/cuda:*-base) ship without conda *and* sometimes without
# curl/bzip2/ca-certificates either, which the Miniconda installer needs to
# fetch + unpack itself. Install those prerequisites via apt first when running
# as root on a Debian/Ubuntu base, then drop the installer into $HOME so the
# script still works for unprivileged callers on shared boxes (RunPod, etc.).
if ! command -v conda >/dev/null 2>&1; then
    # Recover from "conda installed but not on PATH" before reinstalling.
    # This happens when a prior setup.sh run installed Miniconda to
    # $HOME/miniconda3 but the shell session predates that install (or the
    # PATH export was never persisted via `conda init`). In that case the
    # binary is already on disk and reusing it is correct; reinstalling
    # would either trip the installer's "directory already exists" guard
    # or wipe perfectly good environments.
    if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
        echo "=== conda not on PATH but \$HOME/miniconda3 exists — reusing it ==="
        export PATH="${HOME}/miniconda3/bin:${PATH}"
    else
        echo "=== conda not found — installing Miniconda to \$HOME/miniconda3 ==="
        if command -v apt-get >/dev/null 2>&1 && [[ $(id -u) -eq 0 ]]; then
            MINICONDA_APT_DEPS=()
            command -v curl  >/dev/null 2>&1 || MINICONDA_APT_DEPS+=("curl")
            command -v bzip2 >/dev/null 2>&1 || MINICONDA_APT_DEPS+=("bzip2")
            [[ -f /etc/ssl/certs/ca-certificates.crt ]] || MINICONDA_APT_DEPS+=("ca-certificates")
            if [[ ${#MINICONDA_APT_DEPS[@]} -gt 0 ]]; then
                echo "  installing apt prereqs: ${MINICONDA_APT_DEPS[*]}"
                DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null
                DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                    "${MINICONDA_APT_DEPS[@]}" >/dev/null
            fi
        fi
        MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL -o "${MINICONDA_INSTALLER}" \
                https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO "${MINICONDA_INSTALLER}" \
                https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        else
            echo "  [FATAL] neither curl nor wget on PATH; cannot fetch Miniconda installer." >&2
            echo "          Install one of them (apt-get install -y curl) and rerun." >&2
            exit 1
        fi
        # -b batch (no prompts), -p prefix. If $HOME/miniconda3 exists but
        # lacks bin/conda (broken half-install from an aborted prior run),
        # the installer refuses with "File or directory already exists".
        # Detect that ahead of time and pass -u (update in place) so reruns
        # heal a broken tree instead of bailing out.
        MINICONDA_INSTALL_FLAGS=(-b -p "${HOME}/miniconda3")
        if [[ -d "${HOME}/miniconda3" ]]; then
            echo "  [recovery] \$HOME/miniconda3 exists but bin/conda is missing; running installer with -u"
            MINICONDA_INSTALL_FLAGS=(-b -u -p "${HOME}/miniconda3")
        fi
        bash "${MINICONDA_INSTALLER}" "${MINICONDA_INSTALL_FLAGS[@]}"
        rm -f "${MINICONDA_INSTALLER}"
        export PATH="${HOME}/miniconda3/bin:${PATH}"
    fi
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Anaconda channel Terms of Service acceptance ------------------------------
# Since conda 24.7 the default channels (repo.anaconda.com/pkgs/main and
# /pkgs/r) require an explicit one-time ToS acceptance; otherwise every
# `conda create` / `conda install` aborts with "CondaToSNonInteractiveError:
# Terms of Service have not been accepted". Acceptance is recorded under
# ~/.conda/ and idempotent, so we run it unconditionally on every setup
# invocation -- cheap, and protects fresh boxes that haven't been bootstrapped
# before. The `conda tos` subcommand only landed in conda 24.7; older conda
# builds error out, which we treat as non-fatal (those installs don't enforce
# ToS anyway).
if conda tos --help >/dev/null 2>&1; then
    echo "=== Accepting Anaconda channel ToS (one-time; idempotent) ==="
    for ch in https://repo.anaconda.com/pkgs/main \
              https://repo.anaconda.com/pkgs/r; do
        conda tos accept --override-channels --channel "${ch}" >/dev/null 2>&1 \
            && echo "  ok: ${ch}" \
            || echo "  [WARN] could not accept ToS for ${ch} (non-fatal)"
    done
fi

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

    # Clear any PIP_CONSTRAINT / PIP_EXTRA_INDEX_URL exported by a previous
    # install_stack invocation in this shell (--split-envs / MODE=all calls
    # us twice: train first, then test). The train stack sets PIP_CONSTRAINT
    # to torch==<FA_MAX_TORCH_MINOR>.x+cu128 so flash-attn's prebuilt wheel
    # is satisfied; without this reset the test stack inherits that pin,
    # then tries to install the cu sub-index's current max torch (typically
    # one or two minors newer) and dies with ResolutionImpossible:
    #   "The user requested torch==2.11.0+cu128
    #    The user requested (constraint) torch==2.8.0+cu128"
    # The constraint is recomputed below per-stack so the train cap is
    # reapplied for train and dropped for test/vllm (where flash-attn is
    # not installed).
    unset PIP_CONSTRAINT PIP_EXTRA_INDEX_URL

    python -m pip install --upgrade pip wheel setuptools

    # vllm ships its own torch pin; installing torch first causes pip to
    # downgrade/upgrade it when `pip install vllm` runs, which wastes ~2min
    # and sometimes leaves a broken env. Skip the explicit torch install
    # in vllm mode and let vllm resolve its own dependency tree.
    if [[ "${stack}" != "vllm" ]]; then
        echo "=== Installing PyTorch (${CUDA_TAG}) into ${env} ==="

        # Pin to the latest torch/torchvision/torchaudio that the cu-tag
        # wheel index actually publishes, instead of letting pip resolve
        # an unbounded `torch torchvision torchaudio`. Two regressions
        # this guards against, both observed during the 8xH200 bring-up:
        #
        # 1. PEP 440 upstream-version dominance. The cu128 sub-index
        #    typically caps a release or two behind PyPI (today: cu128
        #    has torch 2.11.0+cu128 max; PyPI has plain torch 2.12.0
        #    compiled against cu13). With `pip install torch
        #    --index-url cu128 --extra-index-url pypi`, pip compares
        #    upstream versions first and only uses the +cu128 local
        #    segment to break ties on the SAME upstream version, so
        #    plain `2.12.0` from PyPI beats `2.11.0+cu128` from the
        #    sub-index. The result is a cu13-compiled torch landing
        #    alongside the +cu128 torchaudio that pip pulls correctly
        #    (since the sub-index serves the only torchaudio 2.12 wheel
        #    via the +cu128 build), producing the canonical:
        #      RuntimeError: Detected that PyTorch and TorchAudio were
        #      compiled with different CUDA versions. PyTorch has CUDA
        #      version 13.0 whereas TorchAudio has CUDA version 12.8.
        #    -- on first import.
        #
        # 2. Downstream pip installs (e.g. SFT/test/requirements.txt's
        #    `torch>=2.8.0` CVE pin, line ~550 below) re-evaluate torch
        #    against PyPI and, finding a higher upstream version there,
        #    silently "upgrade" the env into the same broken state.
        #
        # Fix is two-part: (a) discover the highest +cu${CUDA_TAG#cu}
        # torch wheel actually on the sub-index and pin to that exact
        # version (so PyPI's higher upstream wheel can never beat it,
        # because the strict pin excludes it from consideration); and
        # (b) export PIP_CONSTRAINT for the rest of the script so any
        # downstream `pip install` that touches torch is held to the
        # same pin. The constraint file is keyed on the exact +cu
        # local-version segment, so `torch>=2.8.0` style transitive
        # pins still validate against it while preventing the silent
        # upgrade past what the sub-index publishes.
        TORCH_CU_VER="$(curl -fsSL "${TORCH_INDEX_URL}/torch/" \
            | grep -oE "torch-[0-9]+\.[0-9]+\.[0-9]+\+${CUDA_TAG}" \
            | sed -E "s/torch-//" \
            | sort -uV \
            | tail -1)"
        TV_CU_VER="$(curl -fsSL "${TORCH_INDEX_URL}/torchvision/" \
            | grep -oE "torchvision-[0-9]+\.[0-9]+\.[0-9]+\+${CUDA_TAG}" \
            | sed -E "s/torchvision-//" \
            | sort -uV \
            | tail -1)"
        TA_CU_VER="$(curl -fsSL "${TORCH_INDEX_URL}/torchaudio/" \
            | grep -oE "torchaudio-[0-9]+\.[0-9]+\.[0-9]+\+${CUDA_TAG}" \
            | sed -E "s/torchaudio-//" \
            | sort -uV \
            | tail -1)"
        # Train-stack cap (see FA_MAX_TORCH_MINOR at top of file): re-pick the
        # highest +cu wheels whose torch minor is <= FA_MAX_TORCH_MINOR so the
        # prebuilt flash-attn wheel installed by install_flash_attn (below)
        # actually exists. Pairing rule observed on the cu128 sub-index:
        # torchaudio shares torch's X.Y; torchvision is 0.(Y+15) when torch
        # major == 2 (torch 2.7 -> tv 0.22, 2.8 -> 0.23, 2.9 -> 0.24, ...).
        # No-op for stack != train, and no-op when the discovered torch is
        # already at or below the cap.
        if [[ "${stack}" == "train" && ${INSTALL_FLASH_ATTN} -eq 1 && -n "${TORCH_CU_VER}" ]]; then
            CAP_MM_ESC="${FA_MAX_TORCH_MINOR//./\\.}"
            CAP_TORCH="$(curl -fsSL "${TORCH_INDEX_URL}/torch/" \
                | grep -oE "torch-${CAP_MM_ESC}\.[0-9]+\+${CUDA_TAG}" \
                | sed -E "s/torch-//" \
                | sort -uV \
                | tail -1)"
            if [[ -n "${CAP_TORCH}" && "${CAP_TORCH}" != "${TORCH_CU_VER}" ]]; then
                TORCH_MINOR="$(echo "${CAP_TORCH%%+*}" | cut -d. -f2)"
                TV_CAP_MM="0.$(( TORCH_MINOR + 15 ))"
                TA_CAP_MM="${FA_MAX_TORCH_MINOR}"
                CAP_TV="$(curl -fsSL "${TORCH_INDEX_URL}/torchvision/" \
                    | grep -oE "torchvision-${TV_CAP_MM//./\\.}\.[0-9]+\+${CUDA_TAG}" \
                    | sed -E "s/torchvision-//" \
                    | sort -uV \
                    | tail -1)"
                CAP_TA="$(curl -fsSL "${TORCH_INDEX_URL}/torchaudio/" \
                    | grep -oE "torchaudio-${TA_CAP_MM//./\\.}\.[0-9]+\+${CUDA_TAG}" \
                    | sed -E "s/torchaudio-//" \
                    | sort -uV \
                    | tail -1)"
                if [[ -n "${CAP_TV}" && -n "${CAP_TA}" ]]; then
                    echo "  [fa-cap] capping train torch ${TORCH_CU_VER} -> ${CAP_TORCH}"
                    echo "  [fa-cap]              torchvision ${TV_CU_VER} -> ${CAP_TV}"
                    echo "  [fa-cap]              torchaudio  ${TA_CU_VER} -> ${CAP_TA}"
                    echo "  [fa-cap] reason: flash-attn 2.8.3 has no prebuilt wheel for"
                    echo "  [fa-cap]         torch>${FA_MAX_TORCH_MINOR}; without the cap"
                    echo "  [fa-cap]         SFT falls back to SDPA -> cuDNN frontend"
                    echo "  [fa-cap]         failure on H100/H200 + Qwen2 + GC."
                    echo "  [fa-cap] override: FA_MAX_TORCH_MINOR=2.x or --no-flash-attn."
                    TORCH_CU_VER="${CAP_TORCH}"
                    TV_CU_VER="${CAP_TV}"
                    TA_CU_VER="${CAP_TA}"
                else
                    echo "  [fa-cap] WARN: torch cap ${CAP_TORCH} found but matching"
                    echo "  [fa-cap]       torchvision ${TV_CAP_MM}.x / torchaudio"
                    echo "  [fa-cap]       ${TA_CAP_MM}.x missing on ${TORCH_INDEX_URL};"
                    echo "  [fa-cap]       leaving torch at ${TORCH_CU_VER}. flash-attn"
                    echo "  [fa-cap]       install will most likely fail -- training on"
                    echo "  [fa-cap]       H100/H200 with Qwen2 will then break."
                fi
            fi
        fi
        if [[ -z "${TORCH_CU_VER}" || -z "${TV_CU_VER}" || -z "${TA_CU_VER}" ]]; then
            echo "  [WARN] could not discover torch/torchvision/torchaudio versions on"
            echo "         ${TORCH_INDEX_URL} (torch='${TORCH_CU_VER}' tv='${TV_CU_VER}'"
            echo "         ta='${TA_CU_VER}'); falling back to unpinned install -- the"
            echo "         resulting env may have a cu-version mismatch on first import."
            pip install --index-url "${TORCH_INDEX_URL}" \
                        --extra-index-url https://pypi.org/simple \
                        torch torchvision torchaudio
        else
            echo "  [pin] cu wheel index publishes max torch=${TORCH_CU_VER}"
            echo "  [pin]                       torchvision=${TV_CU_VER}"
            echo "  [pin]                       torchaudio=${TA_CU_VER}"
            pip install --index-url "${TORCH_INDEX_URL}" \
                        --extra-index-url https://pypi.org/simple \
                        "torch==${TORCH_CU_VER}" \
                        "torchvision==${TV_CU_VER}" \
                        "torchaudio==${TA_CU_VER}"
            # Hold the pin for the rest of this env build via PIP_CONSTRAINT
            # so the test stack's torch>=2.8.0 line cannot silently regress
            # the env to a PyPI cu13 wheel.
            TORCH_PIN_FILE="$(mktemp -t torch-pin.XXXXXX.txt)"
            cat > "${TORCH_PIN_FILE}" <<EOF
torch==${TORCH_CU_VER}
torchvision==${TV_CU_VER}
torchaudio==${TA_CU_VER}
EOF
            export PIP_CONSTRAINT="${TORCH_PIN_FILE}"
            export PIP_EXTRA_INDEX_URL="${TORCH_INDEX_URL}"
        fi
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

        # SFT/test/requirements.txt pins torch>=2.8.0 (CVE), which is newer
        # than what the cu124 PyTorch wheel index ships. When that pin
        # forces pip to upgrade torch from PyPI (typically to a +cu130
        # wheel), torchvision/torchaudio installed earlier from the cu-tag
        # index get left behind and ABI-mismatch the new torch -- producing
        # the canonical "operator torchvision::nms does not exist" RuntimeError
        # on `from transformers import pipeline`. align_torch_family detects
        # the mismatch and re-anchors the two satellites to torch's actual
        # cu build with --no-deps so this doesn't ripple back into torch.
        echo "=== Aligning torchvision/torchaudio with installed torch in ${env} ==="
        align_torch_family
    fi

    # flash-attn is only meaningful for the training stack (LlamaFactory SFT).
    # vllm bundles its own attention kernels internally, and the bench client
    # (test stack) just drives the vllm server over HTTP -- it never imports
    # flash-attn. Skipping it for non-train stacks avoids a 30-60 min source
    # build that has no runtime impact on benchmarking.
    if [[ ( "${stack}" == "train" || "${stack}" == "all" ) && ${INSTALL_FLASH_ATTN} -eq 1 && "${CUDA_TAG}" != "cpu" ]]; then
        echo "=== Installing flash-attn (v${FLASH_ATTN_VERSION}) — optional, non-fatal ==="
        set +e
        install_flash_attn
        local fa_status=$?
        set -e
        # install_flash_attn returns 0 even when the prebuilt wheel 404s and
        # the source-build fallback is skipped (CUDA mismatch / no nvcc), so
        # the exit code alone is not a reliable signal. Verify the import
        # actually succeeds and escalate to a loud banner if it doesn't --
        # the silent failure path is exactly what put the 8xH200 v21 32B
        # SFT run into the cuDNN-frontend "No valid execution plans built"
        # crash, because llamafactory's --flash_attn auto falls back to
        # SDPA -> cuDNN backend, which can't plan Qwen2 attention shapes
        # under gradient checkpointing on Hopper / Blackwell.
        if [[ ${fa_status} -eq 0 ]] && python -c "import flash_attn" 2>/dev/null; then
            local fa_installed_ver
            fa_installed_ver="$(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo unknown)"
            echo "  flash-attn import OK (version ${fa_installed_ver})."
        else
            echo
            echo "  ############################################################"
            echo "  # [WARN] flash-attn NOT available in ${env}."
            echo "  #        install_flash_attn exit=${fa_status}; import flash_attn failed."
            echo "  #"
            echo "  #        Training on Hopper / Blackwell (H100 / H200 / B200)"
            echo "  #        with Qwen2-class models + gradient checkpointing will"
            echo "  #        fall back to SDPA, then to the cuDNN SDPA backend,"
            echo "  #        which trips:"
            echo "  #          RuntimeError: cuDNN Frontend error:"
            echo "  #          [cudnn_frontend] Error: No valid execution plans built."
            echo "  #"
            echo "  #        Fix options (pick one before launching SFT):"
            echo "  #          1. Re-run setup.sh after confirming the torch cap"
            echo "  #             (FA_MAX_TORCH_MINOR=${FA_MAX_TORCH_MINOR}, default 2.8)"
            echo "  #             actually held -- check the [fa-cap] lines above."
            echo "  #          2. Drop the cap (FA_MAX_TORCH_MINOR=2.X) if a newer"
            echo "  #             flash-attn release lands with matching wheels."
            echo "  #          3. Pin --flash_attn sdpa in the launcher EXTRA_COMMON"
            echo "  #             and export TORCH_CUDNN_SDPA_ENABLED=0 to keep"
            echo "  #             SDPA off the cuDNN backend (~25% slower than FA2)."
            echo "  ############################################################"
            echo
        fi
    elif [[ "${CUDA_TAG}" == "cpu" ]]; then
        echo "=== Skipping flash-attn (CPU build) ==="
    elif [[ "${stack}" != "train" && "${stack}" != "all" ]]; then
        echo "=== Skipping flash-attn (stack=${stack}; only installed for train/all) ==="
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
        # pypdf (already pinned in SFT/test/requirements.txt). The fetch
        # is wall-time non-trivial (~5-15 min of PDFs/JSONs, gated by the
        # ic3.gov / web.archive.org rate limits) and pulls ~200MB to disk,
        # so it's gated to operator intent: only --mode test and --mode
        # vllm trigger the download. --mode all and --mode train skip it
        # silently because those paths are normally used on training
        # boxes that will never read the corpus. --skip-cybersoceval still
        # forces it off in test/vllm modes (e.g. air-gapped reruns).
        # Idempotent: rerunning fetch_cybersoceval_data.py from any host
        # populates it on demand.
        if [[ ${FETCH_CYBERSOCEVAL} -eq 1 && ( "${MODE}" == "test" || "${MODE}" == "vllm" ) ]]; then
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
        elif [[ ${FETCH_CYBERSOCEVAL} -eq 0 ]]; then
            echo "=== Skipping CyberSOCEval data fetch (--skip-cybersoceval) ==="
            echo "         To populate later:"
            echo "             cd ${TEST_DIR} && python utils/fetch_cybersoceval_data.py"
        else
            echo "=== Skipping CyberSOCEval data fetch (mode=${MODE}; only test/vllm fetch) ==="
            echo "         To populate later (any host, no torch needed):"
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
if stack in ("test", "all"):
    # Smoke-test the two imports that recur as bench-client failure modes:
    #   - torchvision.ops.nms: trips the torch/torchvision ABI mismatch
    #     ("operator torchvision::nms does not exist") at the dispatcher
    #     registration step before any model code runs.
    #   - transformers.pipeline: pulls torchvision transitively via
    #     image_processing_utils, so a broken torchvision shows up as a
    #     ModuleNotFoundError("Could not import module 'pipeline'").
    # Both must succeed for run_benchmark.sh to reach the first request.
    # Hard-exit on either failure so a broken env is caught at install
    # time, not 4 seconds into the first benchmark task.
    try:
        from torchvision.ops import nms  # noqa: F401
        import torchvision
        print("torchvision OK    :", torchvision.__version__)
    except Exception as e:
        print("torchvision FAIL  :", repr(e))
        print("                    -> torch/torchvision ABI mismatch; rerun setup.sh")
        print("                       or: pip install --upgrade --force-reinstall --no-deps \\")
        print("                              torchvision torchaudio --index-url <torch-cu-index>")
        sys.exit(1)
    try:
        from transformers import pipeline  # noqa: F401
        print("transformers OK   : from transformers import pipeline")
    except Exception as e:
        print("transformers FAIL :", repr(e))
        sys.exit(1)
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

# torchvision/torchaudio alignment helper ------------------------------------
# SFT/test/requirements.txt pins torch>=2.8.0 (CVE), but the default cu124
# PyTorch wheel index ships at most torch 2.6. When pip resolves the pin it
# yanks a newer torch (typically +cu130) from PyPI, leaving torchvision and
# torchaudio at the original cu-tag wheels installed earlier. The two then
# fail to register their C++ ops against the new torch dispatcher and any
# `from torchvision.ops import nms` (or `from transformers import pipeline`,
# which imports torchvision transitively) explodes with:
#     RuntimeError: operator torchvision::nms does not exist
# Re-anchoring with --no-deps to torch's actual cu build resolves it without
# perturbing torch itself.
align_torch_family() {
    local info
    info="$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.__version__, torch.version.cuda or "")
except Exception:
    print("", "")
PY
)"
    local torch_full torch_cuda
    read -r torch_full torch_cuda <<< "${info}"

    if [[ -z "${torch_full}" ]]; then
        echo "  [WARN] torch not importable in this env; skipping torchvision/torchaudio alignment"
        return 0
    fi
    if [[ -z "${torch_cuda}" ]]; then
        echo "  [info] torch is CPU-only (${torch_full}); skipping cu alignment"
        return 0
    fi

    local cu_tag="cu${torch_cuda//./}"
    local target_index="https://download.pytorch.org/whl/${cu_tag}"

    # If both satellites import cleanly the env is already aligned -- skip
    # the reinstall to keep idempotent reruns fast.
    if python -c "from torchvision.ops import nms" 2>/dev/null \
       && python -c "import torchaudio" 2>/dev/null; then
        echo "  [info] torch/torchvision/torchaudio already aligned (torch ${torch_full}, ${cu_tag})"
        return 0
    fi

    echo "  [repair] torchvision/torchaudio mismatched against torch ${torch_full}; reinstalling from ${target_index}"
    set +e
    pip install --upgrade --force-reinstall --no-deps \
        torchvision torchaudio \
        --index-url "${target_index}"
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
        echo "  [WARN] torchvision/torchaudio reinstall from ${target_index} failed (exit ${rc})."
        echo "         The PyTorch ${cu_tag} wheel index may not have wheels matching this torch yet."
        echo "         Fallbacks (in order of preference):"
        echo "           1) pip install --upgrade --force-reinstall --no-deps \\"
        echo "                  torchvision torchaudio   # try latest from PyPI default"
        echo "           2) pip uninstall -y torchvision torchaudio   # transformers handles"
        echo "                  # their absence for text-only models (>=4.45)"
        return 0
    fi

    if python -c "from torchvision.ops import nms" 2>/dev/null; then
        local tv_ver
        tv_ver="$(python -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null)"
        echo "  [repair] alignment ok (torchvision ${tv_ver})"
    else
        echo "  [WARN] torchvision still failing after reinstall."
        echo "         Consider: pip uninstall -y torchvision   (text-only bench code"
        echo "         doesn't use torchvision; transformers >= 4.45 imports it lazily)."
    fi
}

# flash-attn helper ----------------------------------------------------------
# flash-attn is installed opportunistically: training configs can use it when
# available, but failures are non-fatal because the prebuilt wheels are pinned
# to a specific torch x cuda x python combo and often ABI-mismatch against the
# version pip resolves. flash-attn's setup.py also has an EXDEV bug when
# $CONDA_PREFIX and $PIP_CACHE_DIR live on different filesystems (e.g. RunPod:
# /root vs /home), which is why we try a prebuilt wheel first.
#
# Source-build fallback is gated on torch's compiled CUDA matching the system
# nvcc CUDA. torch's cpp_extension._check_cuda_version aborts the build with
# the verbose 5-page "RuntimeError: ('The detected CUDA version mismatches...')"
# traceback when these differ, so attempting the build under a known mismatch
# is pure noise. The runtime defaults to SDPA when flash-attn is absent, so
# this is best-effort.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

install_flash_attn() {
    local info
    info="$(python - <<'PY'
import torch, sys
tv = torch.__version__.split("+")[0]
tv_mm = ".".join(tv.split(".")[:2])
cu = (torch.version.cuda or "")
cu_compact = cu.replace(".", "")
cu_major = cu_compact[:2] if cu_compact else ""
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"{tv_mm} {cu_major} {py_tag} {cu}")
PY
)"
    local TORCH_MM CU_MAJOR PY_TAG TORCH_CUDA
    read -r TORCH_MM CU_MAJOR PY_TAG TORCH_CUDA <<< "${info}"

    if [[ -z "${CU_MAJOR}" ]]; then
        echo "  [WARN] torch has no CUDA build; skipping flash-attn"
        return 0
    fi

    # Try both ABI variants of the prebuilt wheel. Modern torch wheels
    # (>=2.4 on most cu sub-indices) default to cxx11abi=TRUE; older builds
    # ship abi=FALSE. Picking the wrong one installs cleanly but fails at
    # `import flash_attn` with a libstdc++ ABI mismatch, so try TRUE first
    # and fall back to FALSE before declaring the wheel unavailable.
    local wheel_base="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu${CU_MAJOR}torch${TORCH_MM}"
    local abi
    for abi in TRUE FALSE; do
        local wheel_url="${wheel_base}cxx11abi${abi}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
        echo "  Trying prebuilt wheel (abi=${abi}): ${wheel_url}"
        if pip install --no-build-isolation "${wheel_url}" 2>&1 | tail -20; then
            # pip exits 0 even on 404 when fed a direct URL it can't fetch;
            # confirm flash_attn actually imports before declaring success.
            if python -c "import flash_attn" 2>/dev/null; then
                echo "  flash-attn ${FLASH_ATTN_VERSION} installed (abi=${abi})."
                return 0
            fi
            echo "  [WARN] wheel installed but import failed (likely ABI mismatch); trying next variant."
            pip uninstall -y flash-attn >/dev/null 2>&1 || true
        fi
    done
    echo "  [WARN] Prebuilt wheel unavailable for torch ${TORCH_MM}+cu${CU_MAJOR} / ${PY_TAG}."

    # Gate the source-build fallback on a CUDA version match. nvcc's reported
    # "release X.Y" must match torch's compile-time CUDA major.minor or torch's
    # cpp_extension will abort with a multi-page RuntimeError.
    local nvcc_cuda=""
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_cuda="$(nvcc --version 2>/dev/null \
            | grep -oE 'release [0-9]+\.[0-9]+' \
            | awk '{print $2}' \
            | head -1)"
    fi
    if [[ -z "${nvcc_cuda}" ]]; then
        echo "  [WARN] nvcc not found on PATH; skipping source-build fallback."
        echo "         flash-attn requires a system CUDA toolkit matching torch ${TORCH_CUDA}."
        return 0
    fi
    if [[ "${nvcc_cuda}" != "${TORCH_CUDA}" ]]; then
        echo "  [WARN] CUDA mismatch detected (system nvcc=${nvcc_cuda}, torch=${TORCH_CUDA});"
        echo "         skipping source-build fallback because torch.utils.cpp_extension"
        echo "         would abort with a verbose RuntimeError. To enable flash-attn here,"
        echo "         either install a torch wheel matching CUDA ${nvcc_cuda} or upgrade"
        echo "         the system CUDA toolkit to ${TORCH_CUDA}. flash-attn is optional;"
        echo "         vllm bundles its own attention kernels and the benchmark client"
        echo "         doesn't import flash-attn at all."
        return 0
    fi

    echo "  CUDA versions agree (${nvcc_cuda}); attempting source build..."
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

    # Any other stray *.log files anywhere under SFT/ (>7 days). Catches
    # autotrain launcher logs (_v*_train.log), api_baselines_*.log dumps from
    # benchmark debugging, ad-hoc tee'd output, etc. that the test-specific
    # rule above misses. mtime gate keeps the active run's log untouched.
    if [[ -d "${SFT_DIR}" ]]; then
        REMOVED_STRAY_LOGS=0
        REMOVED_STRAY_LOGS=$(find "${SFT_DIR}" -name '*.log' -type f -mtime +7 -print -delete 2>/dev/null | wc -l)
        echo "  ok: pruned ${REMOVED_STRAY_LOGS} stale *.log file(s) anywhere under SFT/ (>7 days)"
    fi

    # Stale full-SFT checkpoint trees under SFT/saves/<model>/<ft_type>/<run>/.
    # A single Qwen2.5-14B bf16 checkpoint is ~29 GB; a multi-phase run leaves
    # several of those plus optimizer states (when save_only_model is off)
    # behind. On a long-lived training host these accumulate to hundreds of
    # GB and crash the next run's first checkpoint write with the canonical
    # "safetensors_rust.SafetensorError: No space left on device" trip --
    # cf. SFT v14 Phase A 2026-05-07 incident, where ~700 GB of v8/v12/v13
    # save trees crowded /home enough that the first save at step 1000
    # failed mid-write. mtime gate of 14 days is well outside any plausible
    # interactive workflow (active runs touch their checkpoint dir every few
    # minutes); operators who want to keep something past 14d should push it
    # to HF (upload_to_hf.py) or relocate it outside saves/.
    SAVES_DIR="${SFT_DIR}/saves"
    if [[ -d "${SAVES_DIR}" ]]; then
        REMOVED_SAVES=0
        TOTAL_FREED_KB=0
        # mindepth/maxdepth=3 targets exactly <model>/<ft_type>/<run_dir>/
        # so we never recurse into checkpoint shards or delete the model or
        # finetuning-type wrappers themselves.
        while IFS= read -r -d '' stale_dir; do
            dir_kb=$(du -sk "${stale_dir}" 2>/dev/null | awk '{print $1}')
            rm -rf "${stale_dir}"
            REMOVED_SAVES=$((REMOVED_SAVES + 1))
            TOTAL_FREED_KB=$((TOTAL_FREED_KB + ${dir_kb:-0}))
        done < <(find "${SAVES_DIR}" -mindepth 3 -maxdepth 3 -type d -mtime +14 -print0 2>/dev/null)
        if [[ ${REMOVED_SAVES} -gt 0 ]]; then
            freed_human=$(awk -v kb="${TOTAL_FREED_KB}" \
                'BEGIN { for (s="KMGT"; kb>=1024 && length(s)>1; s=substr(s,2)) kb/=1024; printf "%.1f%sB", kb, substr(s,1,1) }')
            echo "  ok: pruned ${REMOVED_SAVES} stale checkpoint dir(s) under saves/ (>14 days, freed ~${freed_human})"
        else
            echo "  ok: no stale checkpoint dirs under saves/ (>14 days)"
        fi
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

    # Surface remaining checkpoint trees (anything <14d that was preserved)
    # so the operator can decide whether to push to HF + delete by hand.
    if [[ -d "${SAVES_DIR:-${SFT_DIR}/saves}" ]]; then
        SAVES_LIST_DIR="${SAVES_DIR:-${SFT_DIR}/saves}"
        # Same depth as the cleanup pass: <model>/<ft_type>/<run_dir>/.
        readarray -t saves_entries < <(
            find "${SAVES_LIST_DIR}" -mindepth 3 -maxdepth 3 -type d 2>/dev/null
        )
        if [[ ${#saves_entries[@]} -gt 0 ]]; then
            echo
            echo "  SFT checkpoint footprint under saves/ (largest 10 entries; push to HF + delete to reclaim):"
            du -sh "${saves_entries[@]}" 2>/dev/null | sort -rh | head -10 | sed 's/^/    /'
            echo "  Total: $(du -sh "${SAVES_LIST_DIR}" 2>/dev/null | awk '{print $1}')"
            echo "  To remove a single run: rm -rf ${SAVES_LIST_DIR}/<model>/<ft_type>/<run_dir>"
        fi
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
if [[ ${GIT_CREDENTIAL_WAS_SET} -eq 1 ]]; then
    echo "GitHub HTTPS credential helper active: 'git pull/push' against the"
    echo "private repo will use ${GIT_CREDENTIALS_FILE} (perms 0600)."
    echo
elif command -v git >/dev/null 2>&1 && [[ -d "${SFT_DIR}/../.git" ]]; then
    if ! git -C "${SFT_DIR}/.." config --get-all credential.helper >/dev/null 2>&1 \
       && [[ ! -f "${HOME}/.git-credentials" ]]; then
        echo "GitHub HTTPS auth not configured. Private-repo 'git pull' will"
        echo "fail with 'Password authentication is not supported'. Either"
        echo "rerun setup with --git-token ghp_xxx (or set GITHUB_TOKEN in"
        echo "SFT/.env) or configure manually:"
        echo "    git config --global credential.helper store"
        echo "    echo 'https://x-access-token:ghp_xxx@github.com' >> ~/.git-credentials"
        echo "    chmod 600 ~/.git-credentials"
        echo
    fi
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
