#!/bin/bash

# Launch a local vLLM OpenAI-compatible server for benchmark inference.
#
# vLLM loads the model once and accepts concurrent HTTP requests, so the
# benchmark harness can replace per-row HuggingFace generate() calls with
# N in-flight /v1/chat/completions requests. See pipelines/models.py
# (VLLMModel) and inference.py for the client side; --batch N in
# run_benchmark.sh maps directly to N concurrent workers here.
#
# Runs in the foreground. Ctrl-C tears down the server cleanly.
#
# Usage:
#   ./serve_vllm.sh --model <hf-repo-id> [--port 8000] [--tp 2]
#                    [--max-len 4096] [--dtype bfloat16]
#                    [--chat-template PATH | --no-auto-template]
#                    [--env-name vllm] [--extra "--any --vllm --flag"]
#
# Chat-template handling:
#   Models whose HF repo already carries a chat_template (most Instruct and
#   fine-tuned models) need nothing extra. For base models that do not ship
#   one (e.g. meta-llama/Llama-3.1-8B), this script probes the repo via
#   transformers.AutoTokenizer and, when no template is found, auto-applies
#   a bundled jinja from ./chat_templates/ based on the repo name family:
#     llama-3 / llama3   -> chat_templates/llama3.jinja
#   Override with --chat-template <path>, or disable the auto-apply with
#   --no-auto-template (vllm will then error on /v1/chat/completions).
#
# Examples:
#   # CPT model on 2xH100 (chat template ships on the repo):
#   ./serve_vllm.sh --model asg-ai/athena-cti-cpt-llama31-8b-v1 --tp 2
#
#   # Base Llama-3.1-8B (no chat template): auto-applied from bundle.
#   ./serve_vllm.sh --model meta-llama/Llama-3.1-8B --tp 2
#
#   # Explicit override path:
#   ./serve_vllm.sh --model meta-llama/Llama-3.1-8B \
#       --chat-template /path/to/custom_llama3.jinja
#
# Env vars consumed:
#   HF_TOKEN / HUGGINGFACE_TOKEN   passed through to vllm serve
#   VLLM_CONDA_ENV                 override --env-name (default: vllm)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL=""
PORT=8000
TP=1
MAX_LEN=4096
DTYPE="bfloat16"
CHAT_TEMPLATE=""
AUTO_TEMPLATE=1
ENV_NAME="${VLLM_CONDA_ENV:-vllm}"
EXTRA=""

TEMPLATES_DIR="${SCRIPT_DIR}/chat_templates"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --tp)                TP="$2"; shift 2 ;;
        --max-len)           MAX_LEN="$2"; shift 2 ;;
        --dtype)             DTYPE="$2"; shift 2 ;;
        --chat-template)     CHAT_TEMPLATE="$2"; shift 2 ;;
        --no-auto-template)  AUTO_TEMPLATE=0; shift ;;
        --env-name)          ENV_NAME="$2"; shift 2 ;;
        --extra)             EXTRA="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: --model <hf-repo-id> is required" >&2
    exit 1
fi

# Source HF / wandb / etc. credentials from SFT/.env if present so gated
# repos (Llama-3.x) can download without re-exporting the token manually.
if [[ -f "${SFT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${SFT_DIR}/.env"
    set +a
fi

# Activate the vllm conda env if available. setup.sh --mode vllm creates
# this env with a standalone vllm install so its torch pin stays isolated
# from the llamafactory training stack.
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "=== Activating conda env: ${ENV_NAME} ==="
        conda activate "${ENV_NAME}"
    else
        echo "  [WARN] conda env '${ENV_NAME}' not found; using current env."
        echo "         Create it via: bash SFT/utils/setup.sh --mode vllm"
    fi
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: 'vllm' CLI not found on PATH." >&2
    echo "  Install into the ${ENV_NAME} conda env:" >&2
    echo "    bash SFT/utils/setup.sh --mode vllm" >&2
    exit 2
fi

# Chat-template auto-detect. Base models (e.g. meta-llama/Llama-3.1-8B) do
# not ship a chat_template on the HF repo, which makes vllm's
# /v1/chat/completions endpoint return 400. Probe the tokenizer and, if no
# template is set, map the repo name to a bundled jinja. Skipped entirely
# when the user passed --chat-template explicitly or --no-auto-template.
if [[ -z "${CHAT_TEMPLATE}" && ${AUTO_TEMPLATE} -eq 1 ]]; then
    echo "=== Probing ${MODEL} for a chat_template ==="
    probe_family="$(python - "${MODEL}" <<'PY' 2>/dev/null || true
import sys
model = sys.argv[1]
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if getattr(tok, "chat_template", None):
        print("__present__")
        sys.exit(0)
except Exception as e:
    sys.stderr.write(f"probe: tokenizer load failed: {e}\n")
ml = model.lower()
if "llama-3" in ml or "llama3" in ml:
    print("llama3")
else:
    print("")
PY
)"
    case "${probe_family}" in
        __present__)
            echo "  chat_template present on repo; no override needed."
            ;;
        "")
            echo "  [WARN] no chat_template on repo and no family match; "
            echo "         /v1/chat/completions will likely 400. Pass --chat-template PATH"
            echo "         or --no-auto-template to silence this warning."
            ;;
        *)
            auto_path="${TEMPLATES_DIR}/${probe_family}.jinja"
            if [[ -f "${auto_path}" ]]; then
                CHAT_TEMPLATE="${auto_path}"
                echo "  no chat_template on repo; auto-applying bundled: ${auto_path}"
            else
                echo "  [WARN] family='${probe_family}' detected but bundle missing at ${auto_path}"
            fi
            ;;
    esac
fi

echo "=== vllm serve ==="
echo "  model         : ${MODEL}"
echo "  port          : ${PORT}"
echo "  tensor-parallel: ${TP}"
echo "  max-model-len : ${MAX_LEN}"
echo "  dtype         : ${DTYPE}"
[[ -n "${CHAT_TEMPLATE}" ]] && echo "  chat-template : ${CHAT_TEMPLATE}"
[[ -n "${EXTRA}" ]]         && echo "  extra args    : ${EXTRA}"
echo "  env           : ${CONDA_DEFAULT_ENV:-<none>}"
echo

# Assemble and exec. 'exec' so Ctrl-C / SIGTERM propagate straight to
# vllm and we don't leave an orphaned server on shell exit.
args=(serve "${MODEL}"
      --port "${PORT}"
      --tensor-parallel-size "${TP}"
      --max-model-len "${MAX_LEN}"
      --dtype "${DTYPE}")

if [[ -n "${CHAT_TEMPLATE}" ]]; then
    args+=(--chat-template "${CHAT_TEMPLATE}")
fi

if [[ -n "${EXTRA}" ]]; then
    # shellcheck disable=SC2206
    extra_arr=(${EXTRA})
    args+=("${extra_arr[@]}")
fi

echo "Running: vllm ${args[*]}"
echo
exec vllm "${args[@]}"
