#!/bin/bash

# Parallel SFT setup that installs an *upstream* LlamaFactory (pinned to a
# commit that includes Gemma 4 support) and a transformers release that
# carries the text-only Gemma 3/4 `(mm_)token_type_ids` fix, into a NEW
# conda env (default: `llm-sft-gemma`). The original `llm-sft` env produced
# by `setup.sh` is left untouched so all pre-v21 chains (Qwen2.5-14B v18.1,
# v18.2, v21; Llama-3.1-8B v21; etc.) keep running off the vendored
# `SFT/src/llamafactory/` tree.
#
# Background:
#   The vendored LlamaFactory in `SFT/src/llamafactory/` predates Gemma 4
#   support. Upstream PRs #10346 (template + Gemma4Plugin + model groups),
#   #10359 (mm_token_type_ids padding in collator), #10378 (set_mm_projectors
#   patcher for text-only training), #10381/#10382 (missing projector key
#   handling) collectively land Gemma 4 SFT, merged through 2026-04-12.
#   Separately, transformers PR #45222 (merged 2026-04-15) defaults
#   `(mm_)token_type_ids` to zeros for text-only Gemma 3/4 training instead
#   of raising `ValueError` -- first shipped in transformers v5.7.0 (2026-04-28).
#
# This script does NOT mutate the vendored tree. It does NOT touch the
# existing `llm-sft` env. Run it alongside the original setup. Activate
# `llm-sft-gemma` to launch the Gemma 4 v21 chain; activate `llm-sft` for
# every other architecture.
#
# Usage:
#   ./setup_gemma.sh [--cuda cu130|cu128|cu126|cu124|cu121|cu118|cpu|auto]
#                    [--env-name NAME]            # default: llm-sft-gemma
#                    [--python VERSION]           # default: 3.11
#                    [--lf-ref REF]               # default: 436d26bc... (post #10382)
#                    [--transformers-spec SPEC]   # default: transformers>=5.7.0,<5.9
#                    [--no-flash-attn] [--no-conda-init]
#                    [--git-token ghp_xxx]
#
# Notes on Gemma 4 SFT on top of the new env:
#   - Use `--template gemma4` (registered upstream; the withdrawn
#     `gemma4_text` template from PR #10362 is NOT needed because
#     #10378 fixes text-only at the model-patching layer).
#   - Gemma 4's head_dim=512 is not supported by stock FlashAttention 2;
#     the Gemma 4 launchers already pin `--flash_attn sdpa`. flash-attn
#     is still installed opportunistically because dropping it here
#     would also break Qwen/Llama experiments that happen to be run
#     out of this env.
#   - DO NOT enable DeepSpeed ZeRO-3 for Gemma 4 E2B/E4B per the LF
#     PR #10346 release note. ZeRO-3 is fine for the 31B dense variant
#     used by the v21 chain (which is what this env is built for).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pinned to commit `436d26bc28b7c6422c89b63064c5a87e258ed73e` (2026-04-12),
# the last of the Gemma 4 fix-up commits (PR #10382, closing #10381).
# Bump this when LlamaFactory cuts a tag that supersedes it (v0.9.5+).
LF_DEFAULT_REF="436d26bc28b7c6422c89b63064c5a87e258ed73e"
LF_REPO_URL="https://github.com/hiyouga/LLaMA-Factory.git"

# transformers v5.7.0 (2026-04-28) is the first stable release containing
# PR #45222 (Gemma 3/4 text-only token_type_ids fix). Cap below v5.9 to
# stay inside the API surface validated by LF post-#10382.
TRANSFORMERS_DEFAULT_SPEC="transformers>=5.7.0,<5.9"

ENV_NAME="llm-sft-gemma"
PYTHON_VERSION="3.11"
CUDA_TAG="cu128"
CUDA_TAG_EXPLICIT=0
LF_REF="${LF_DEFAULT_REF}"
TRANSFORMERS_SPEC="${TRANSFORMERS_DEFAULT_SPEC}"
INSTALL_FLASH_ATTN=1
RUN_CONDA_INIT=1
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)               CUDA_TAG="$2"; CUDA_TAG_EXPLICIT=1; shift 2 ;;
        --env-name)           ENV_NAME="$2"; shift 2 ;;
        --python)             PYTHON_VERSION="$2"; shift 2 ;;
        --lf-ref)             LF_REF="$2"; shift 2 ;;
        --transformers-spec)  TRANSFORMERS_SPEC="$2"; shift 2 ;;
        --no-flash-attn)      INSTALL_FLASH_ATTN=0; shift ;;
        --no-conda-init)      RUN_CONDA_INIT=0; shift ;;
        --git-user-name)      GIT_USER_NAME="$2"; shift 2 ;;
        --git-user-email)     GIT_USER_EMAIL="$2"; shift 2 ;;
        --git-token)          GITHUB_TOKEN="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# CUDA tag auto-detect (mirrors setup.sh; default cu128 here because the
# B300/Blackwell hosts this script targets ship with CUDA 12.8 toolkits).
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
        echo "  [auto-detect] nvcc not on PATH; falling back to --cuda cu128"
        CUDA_TAG="cu128"
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

echo "=== SFT gemma-side setup ==="
echo "  sft dir         : ${SFT_DIR}"
echo "  env name        : ${ENV_NAME}"
echo "  python          : ${PYTHON_VERSION}"
echo "  cuda tag        : ${CUDA_TAG}"
echo "  llamafactory    : ${LF_REPO_URL}@${LF_REF}"
echo "  transformers    : ${TRANSFORMERS_SPEC}"
echo "  flash-attn      : $([[ ${INSTALL_FLASH_ATTN} -eq 1 ]] && echo yes || echo no)"
echo

# 1. Source SFT/.env so HF/wandb/git creds are visible. CLI overrides win.
if [[ -f "${SFT_DIR}/.env" ]]; then
    _gun_pre="${GIT_USER_NAME:-}"
    _gue_pre="${GIT_USER_EMAIL:-}"
    _ght_pre="${GITHUB_TOKEN:-}"
    set -a
    # shellcheck source=/dev/null
    source "${SFT_DIR}/.env"
    set +a
    [[ -n "${_gun_pre}" ]] && GIT_USER_NAME="${_gun_pre}"
    [[ -n "${_gue_pre}" ]] && GIT_USER_EMAIL="${_gue_pre}"
    [[ -n "${_ght_pre}" ]] && GITHUB_TOKEN="${_ght_pre}"
    unset _gun_pre _gue_pre _ght_pre
fi

# 2. Miniconda bootstrap (mirrors setup.sh; only runs when conda missing).
if ! command -v conda >/dev/null 2>&1; then
    if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
        echo "=== conda not on PATH but \$HOME/miniconda3 exists -- reusing it ==="
        export PATH="${HOME}/miniconda3/bin:${PATH}"
    else
        echo "=== conda not found -- installing Miniconda to \$HOME/miniconda3 ==="
        if command -v apt-get >/dev/null 2>&1 && [[ $(id -u) -eq 0 ]]; then
            DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                curl bzip2 ca-certificates >/dev/null
        fi
        MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL -o "${MINICONDA_INSTALLER}" \
                https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        else
            wget -qO "${MINICONDA_INSTALLER}" \
                https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        fi
        MINICONDA_FLAGS=(-b -p "${HOME}/miniconda3")
        [[ -d "${HOME}/miniconda3" ]] && MINICONDA_FLAGS=(-b -u -p "${HOME}/miniconda3")
        bash "${MINICONDA_INSTALLER}" "${MINICONDA_FLAGS[@]}"
        rm -f "${MINICONDA_INSTALLER}"
        export PATH="${HOME}/miniconda3/bin:${PATH}"
    fi
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda tos --help >/dev/null 2>&1; then
    for ch in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
        conda tos accept --override-channels --channel "${ch}" >/dev/null 2>&1 || true
    done
fi

# 3. Create / reuse the gemma env.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "=== Reusing existing conda env: ${ENV_NAME} ==="
else
    echo "=== Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION}) ==="
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel setuptools

# 4. Install PyTorch matched to the requested CUDA tag.
echo "=== Installing PyTorch (${CUDA_TAG}) into ${ENV_NAME} ==="
pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio

# 5. Install upstream LlamaFactory at the pinned ref. This drops the
# upstream `llamafactory` package into site-packages of this env only;
# the vendored copy under SFT/src/llamafactory remains the source of
# truth for the original `llm-sft` env.
echo "=== Installing upstream LlamaFactory @ ${LF_REF} into ${ENV_NAME} ==="
pip install "llamafactory @ git+${LF_REPO_URL}@${LF_REF}"

# 6. Force the transformers spec we want. The upstream pyproject at the
# pinned commit may still cap transformers below 5.7; `--upgrade` with
# an explicit spec overrides that cap. pip emits a dependency-conflict
# warning if so, which is expected and benign for this use case (Gemma
# 4 text-only SFT was validated end-to-end on transformers 5.7+ per
# PR #45454).
echo "=== Pinning transformers to '${TRANSFORMERS_SPEC}' (Gemma 4 text-only fix; PR #45222) ==="
pip install --upgrade "${TRANSFORMERS_SPEC}"

# 7. Training extras (matches setup.sh `train` stack minus the editable
# install of SFT/). deepspeed + metrics live in SFT/requirements/.
for extra in deepspeed metrics; do
    req_file="${SFT_DIR}/requirements/${extra}.txt"
    if [[ -f "${req_file}" ]]; then
        echo "=== Installing extras: ${extra} ==="
        pip install -r "${req_file}"
    fi
done

echo "=== Installing wandb + huggingface_hub + python-dotenv + ninja ==="
pip install wandb huggingface_hub python-dotenv packaging ninja

echo "=== Installing liger-kernel ==="
pip install liger-kernel

echo "=== Installing bitsandbytes ==="
pip install bitsandbytes

# 8. flash-attn (opportunistic; non-fatal). Gemma 4's head_dim=512 isn't
# supported by stock FlashAttention 2, so the Gemma 4 v21 launchers
# already pin --flash_attn sdpa. flash-attn is still installed because
# this env can also be used for non-Gemma experiments. Helper mirrors
# the prebuilt-wheel-first strategy from setup.sh.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
install_flash_attn_local() {
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
    [[ -z "${CU_MAJOR}" ]] && { echo "  [WARN] torch has no CUDA build; skipping flash-attn"; return 0; }
    local wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu${CU_MAJOR}torch${TORCH_MM}cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    echo "  Trying prebuilt wheel: ${wheel_url}"
    if pip install --no-build-isolation "${wheel_url}" 2>&1 | tail -10; then
        python -c "import flash_attn" 2>/dev/null && return 0
    fi
    echo "  [WARN] Prebuilt wheel unavailable; skipping source build (Gemma 4 uses --flash_attn sdpa anyway)."
    return 0
}
if [[ ${INSTALL_FLASH_ATTN} -eq 1 && "${CUDA_TAG}" != "cpu" ]]; then
    echo "=== Installing flash-attn (v${FLASH_ATTN_VERSION}) -- optional, non-fatal ==="
    set +e
    install_flash_attn_local
    set -e
fi

# 9. Verify imports.
echo
echo "=== Verifying PyTorch / LlamaFactory / transformers in ${ENV_NAME} ==="
python - <<'PY'
import torch
print("torch version     :", torch.__version__)
print("torch cuda build  :", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device            :", torch.cuda.get_device_name(0))
    print("bf16 supported    :", torch.cuda.is_bf16_supported())
try:
    import transformers
    print("transformers      :", transformers.__version__)
except Exception as e:
    print("transformers import failed:", e)
try:
    import llamafactory
    print("llamafactory      :", getattr(llamafactory, "__version__", "installed"))
except Exception as e:
    print("llamafactory import failed:", e)
try:
    from llamafactory.data.template import TEMPLATES
    print("gemma4 template   :", "yes" if "gemma4" in TEMPLATES else "MISSING")
except Exception as e:
    print("template probe failed:", e)
for mod in ("liger_kernel", "bitsandbytes", "deepspeed"):
    try:
        m = __import__(mod)
        print(f"{mod:18}:", getattr(m, "__version__", "installed"))
    except Exception as e:
        print(f"{mod:18}: import failed: {e}")
PY

if command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli  : $(command -v llamafactory-cli)"
fi

# 10. Persist HF / wandb credentials at the box level (matches setup.sh).
if [[ -f "${SFT_DIR}/.env" ]] && ! grep -q 'hf_xxx_replace_me\|your-hf-username' "${SFT_DIR}/.env" 2>/dev/null; then
    set -a
    # shellcheck source=/dev/null
    source "${SFT_DIR}/.env"
    set +a
    if [[ -n "${HF_TOKEN:-}" && "${HF_TOKEN}" != "hf_xxx_replace_me" ]]; then
        echo "=== Persisting HF_TOKEN to ~/.cache/huggingface/token ==="
        python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)" \
            >/dev/null 2>&1 && echo "  ok." || echo "  [WARN] hf login failed."
    fi
    if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != "wandb_xxx_replace_me" ]]; then
        echo "=== Persisting WANDB_API_KEY to ~/.netrc ==="
        wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 \
            && echo "  ok." || echo "  [WARN] wandb login failed."
    fi
fi

# 11. Git identity + GitHub HTTPS credential helper (matches setup.sh).
if command -v git >/dev/null 2>&1; then
    if [[ -n "${GIT_USER_NAME}" && -n "${GIT_USER_EMAIL}" ]]; then
        git config --global user.name  "${GIT_USER_NAME}"
        git config --global user.email "${GIT_USER_EMAIL}"
        echo "=== Git identity set: ${GIT_USER_NAME} <${GIT_USER_EMAIL}> ==="
    fi
    if [[ -n "${GITHUB_TOKEN}" && "${GITHUB_TOKEN}" != "ghp_xxx_replace_me" ]]; then
        git config --global credential.helper store
        GCF="${HOME}/.git-credentials"
        [[ -f "${GCF}" ]] && grep -v '@github.com' "${GCF}" > "${GCF}.tmp" 2>/dev/null && mv "${GCF}.tmp" "${GCF}"
        ( umask 077 && printf 'https://x-access-token:%s@github.com\n' "${GITHUB_TOKEN}" >> "${GCF}" )
        chmod 600 "${GCF}" 2>/dev/null || true
        echo "=== GitHub HTTPS credential helper configured ==="
    fi
fi

# 12. Shell integration.
if [[ ${RUN_CONDA_INIT} -eq 1 ]]; then
    target_shell="$(basename "${SHELL:-/bin/bash}")"
    case "${target_shell}" in
        bash|zsh|fish) conda init "${target_shell}" >/dev/null 2>&1 || true ;;
    esac
fi

echo
echo "=== Gemma-side setup complete ==="
echo "Activate the new env to launch the Gemma 4 v21 chain:"
echo "    conda activate ${ENV_NAME}"
echo
echo "Then from SFT/autotrain/:"
echo "    ./run_sft_gemma4_31b_v21_core.sh --report-to wandb"
echo "    ./run_sft_gemma4_31b_v21_plus_taa.sh --report-to wandb"
echo "    ./run_sft_gemma4_31b_v21_final.sh --report-to wandb         # CSE"
echo "    ./run_sft_gemma4_31b_v21_recalibrate.sh --report-to wandb"
echo
echo "The original 'llm-sft' env is untouched -- use it for Qwen/Llama runs:"
echo "    conda activate llm-sft"

