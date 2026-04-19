#!/bin/bash

# Launch an AutoTrain Advanced training run.
#
# Reads the chosen YAML config, verifies the env + credentials, and runs
# `autotrain --config <yaml>` in the foreground (or detached with --nohup),
# with stdout/stderr teed to <project_name>_<timestamp>.log in this directory.
#
# On success AutoTrain pushes the merged full-weight model to
# huggingface.co/<HF_USERNAME>/<project_name> automatically (via the YAML
# hub: section).
#
# Usage:
#   ./train.sh [--config PATH] [--cuda-devices LIST] [--nohup] [--no-follow]
#              [--min-vram-gb N] [--skip-vram-check]
#
# Defaults:
#   --config          autotrain_llama3_8b_sft.yml  (full-parameter SFT)
#   --cuda-devices    (unset -> all visible GPUs)
#   --min-vram-gb     140   (only enforced for full-SFT configs; see below)
#
# Output:
#   Progress is always duplicated to both stdout and <project>_<ts>.log.
#   In foreground mode this is done via `tee`; in --nohup mode the script
#   launches training detached and then attaches `tail -F` on the log so
#   you see live progress in the current shell. Press Ctrl-C to detach
#   the tail -- the nohup'd training keeps running. Use --no-follow to
#   skip the tail and return to the shell immediately.
#
# Full SFT of an 8B model needs ~80 GB of aggregate bf16 VRAM. The pre-flight
# check refuses to launch if the selected GPUs fall below --min-vram-gb for
# any YAML with peft: false (override with --skip-vram-check if you know
# what you're doing). LoRA configs (peft: true) skip the check entirely.
#
# Required env vars (loaded automatically from ./.env if present):
#   HF_TOKEN, HF_USERNAME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/autotrain_llama3_8b_sft.yml"
CUDA_DEVICES=""
DETACH=0
FOLLOW=1
# Aggregate VRAM floor for full-parameter 8B SFT. AutoTrain 0.8.36 loads the
# model in fp32 (autocast-only mixed precision), so even with ZeRO-3 sharding
# across 2 GPUs the working set is ~70 GB/GPU => ~140 GB aggregate minimum.
MIN_VRAM_GB=140
SKIP_VRAM_CHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG="$2"; shift 2 ;;
        --cuda-devices)     CUDA_DEVICES="$2"; shift 2 ;;
        --nohup)            DETACH=1; shift ;;
        --no-follow)        FOLLOW=0; shift ;;
        --min-vram-gb)      MIN_VRAM_GB="$2"; shift 2 ;;
        --skip-vram-check)  SKIP_VRAM_CHECK=1; shift ;;
        -h|--help) sed -n '3,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
fi

: "${HF_TOKEN:?Set HF_TOKEN in SFT/autotrain/.env (see .env.example)}"
: "${HF_USERNAME:?Set HF_USERNAME in SFT/autotrain/.env (see .env.example)}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[FAIL] config not found: ${CONFIG}" >&2
    exit 1
fi

if ! command -v autotrain >/dev/null 2>&1; then
    echo "[FAIL] autotrain CLI not on PATH. Activate the env first:" >&2
    echo "       conda activate autotrain" >&2
    exit 127
fi

# Extract project_name and training hyperparams for the summary block so the
# user can eyeball effective batch size and detect silent scaling mistakes.
read -r PROJECT_NAME BATCH_SIZE GRAD_ACCUM GRAD_CKPT PEFT_FLAG DIST_BACKEND FA2_FLAG < <(python - "${CONFIG}" <<'PY'
import yaml, sys
with open(sys.argv[1]) as f: c = yaml.safe_load(f) or {}
p = c.get("params", {}) or {}
# autotrain stores `disable_gradient_checkpointing` (inverted) in params;
# translate back to the more readable "grad_ckpt ON/OFF" flag for the banner.
grad_ckpt_on = not bool(p.get("disable_gradient_checkpointing", False))
print(
    c.get("project_name", "autotrain-run"),
    int(p.get("batch_size", 1)),
    int(p.get("gradient_accumulation", 1)),
    bool(grad_ckpt_on),
    bool(p.get("peft", False)),
    c.get("distributed_backend") or "ddp",
    bool(p.get("use_flash_attention_2", False)),
)
PY
)

NUM_VISIBLE_GPUS=1
if [[ -n "${CUDA_DEVICES}" ]]; then
    NUM_VISIBLE_GPUS=$(awk -F',' '{print NF}' <<<"${CUDA_DEVICES}")
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_VISIBLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_VISIBLE_GPUS))

TIMESTAMP="$(date -u +"%Y%m%d-%H%M%SZ")"
LOG_FILE="${SCRIPT_DIR}/${PROJECT_NAME}_${TIMESTAMP}.log"

echo "=== AutoTrain run ==="
echo "  config         : ${CONFIG}"
echo "  project_name   : ${PROJECT_NAME}"
echo "  hub target     : ${HF_USERNAME}/${PROJECT_NAME}"
echo "  mode           : $([[ "${PEFT_FLAG}" == "True" ]] && echo 'LoRA (peft)' || echo 'full SFT')"
echo "  per-GPU batch  : ${BATCH_SIZE}"
echo "  grad accum     : ${GRAD_ACCUM}"
echo "  visible GPUs   : ${NUM_VISIBLE_GPUS}"
echo "  effective batch: ${EFFECTIVE_BATCH}   (= ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_VISIBLE_GPUS})"
echo "  grad ckpt      : ${GRAD_CKPT}"
echo "  flash attn 2   : ${FA2_FLAG}"
echo "  dist backend   : ${DIST_BACKEND}"
echo "  log file       : ${LOG_FILE}"
echo "  cuda devices   : ${CUDA_DEVICES:-<all visible>}"
echo "  detach         : $([[ ${DETACH} -eq 1 ]] && echo yes || echo no)"
echo "  started        : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo

# Brief GPU snapshot (non-fatal if nvidia-smi missing)
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
        --format=csv,noheader | sed 's/^/  /'
    echo
fi

if [[ -n "${CUDA_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

# Pre-flight VRAM check: full-parameter SFT of 8B needs ~80 GB aggregate.
# Refuse to launch if the box is obviously too small, so the user finds out
# in one second instead of after minutes of tokenizer download + init only
# to crash with a cryptic CUDA OOM at step 0.
if [[ ${SKIP_VRAM_CHECK} -eq 0 && "${PEFT_FLAG}" == "False" ]] \
        && command -v nvidia-smi >/dev/null 2>&1; then
    TOTAL_VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
        | awk '{s+=$1} END {printf "%.0f", s/1024}')"
    if [[ -n "${TOTAL_VRAM_GB}" && "${TOTAL_VRAM_GB}" -lt "${MIN_VRAM_GB}" ]]; then
        echo "[FAIL] full-parameter SFT requires >= ${MIN_VRAM_GB} GB aggregate VRAM;" >&2
        echo "       detected only ${TOTAL_VRAM_GB} GB across visible GPUs." >&2
        echo >&2
        echo "Options:" >&2
        echo "  * move to a bigger box (A100-80G, H100-80G, or 2x A100-40G)" >&2
        echo "  * run the LoRA + int4 variant instead:" >&2
        echo "      ./train.sh --config ${SCRIPT_DIR}/autotrain_llama3_8b_lora.yml" >&2
        echo "  * override this check (you will OOM at step 0):" >&2
        echo "      ./train.sh --skip-vram-check" >&2
        exit 2
    fi
    echo "  vram check     : ${TOTAL_VRAM_GB} GB >= ${MIN_VRAM_GB} GB required for full SFT -> OK"
    echo
fi

# --- Env-var expansion in the YAML -------------------------------------------
# AutoTrain only substitutes ${HF_USERNAME}/${HF_TOKEN} inside its 'hub:'
# section; any '${VAR}' reference elsewhere (e.g. data.path) is passed to
# the downstream datasets/huggingface_hub loaders as a literal string and
# blows up with an HFValidationError. Render the YAML ourselves so every
# field is a fully resolved literal before autotrain sees it.
RENDERED_CONFIG="${SCRIPT_DIR}/.rendered_${PROJECT_NAME}_$(date -u +%s).yml"
trap 'rm -f "${RENDERED_CONFIG}"' EXIT
python - "${CONFIG}" "${RENDERED_CONFIG}" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f: text = f.read()
with open(dst, "w") as f: f.write(os.path.expandvars(text))
PY
echo "  rendered yaml  : ${RENDERED_CONFIG}"

# --- Distributed backend (deepspeed) -----------------------------------------
# The full-SFT YAMLs request ZeRO-3 via `distributed_backend: deepspeed`.
# If this env was built before deepspeed was added to setup.sh, install it
# on demand rather than failing with an opaque autotrain error.
if [[ "${DIST_BACKEND}" == "deepspeed" ]]; then
    if ! python -c "import deepspeed" 2>/dev/null; then
        echo "  deepspeed      : installing (missing from env) ..."
        python -m pip install --quiet "deepspeed>=0.15,<0.17"
    fi
fi

# --- Logging backend (wandb) -------------------------------------------------
# The YAMLs set `log: wandb`. Ensure the `wandb` python package is present
# and that WANDB_API_KEY is exported, otherwise autotrain's WandbCallback
# silently falls back to offline mode.
LOG_BACKEND="$(python - "${RENDERED_CONFIG}" <<'PY'
import yaml, sys
with open(sys.argv[1]) as f: print(yaml.safe_load(f).get("log", "none"))
PY
)"
if [[ "${LOG_BACKEND}" == "wandb" ]]; then
    if ! python -c "import wandb" 2>/dev/null; then
        echo "  wandb          : installing (missing from env) ..."
        python -m pip install --quiet wandb
    fi
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "[WARN] log: wandb but WANDB_API_KEY is not set; wandb will run in offline mode." >&2
        echo "       Add WANDB_API_KEY=... to SFT/autotrain/.env (see .env.example)." >&2
    else
        echo "  wandb project  : ${WANDB_PROJECT:-athena-cti-sft}"
        export WANDB_PROJECT="${WANDB_PROJECT:-athena-cti-sft}"
        export WANDB_NAME="${PROJECT_NAME}_${TIMESTAMP}"
        export WANDB_WATCH="false"     # skip grad/param logging to reduce overhead
    fi
fi
echo

# Line-buffer python stdout so `tail -f` on the log shows Trainer progress
# lines in real time instead of in fsync-flush chunks.
export PYTHONUNBUFFERED=1

# Expandable segments materially reduce CUDA OOM-by-fragmentation during the
# first optimizer step when Adam states materialize (the PyTorch error message
# at step 0 explicitly suggests this as a remediation).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Line-buffer wrapper: without this, piping autotrain through `tee` (fg) or
# redirecting to a file (nohup) switches python's stdout/stderr to block-
# buffered mode, so progress only appears in 4-8 KB chunks. stdbuf is part
# of GNU coreutils -- available on every Linux distro, absent on bare macOS.
if command -v stdbuf >/dev/null 2>&1; then
    BUF=(stdbuf -oL -eL)
else
    BUF=()
fi

CMD=(autotrain --config "${RENDERED_CONFIG}")

if [[ ${DETACH} -eq 1 ]]; then
    echo "=== Launching detached (nohup) ==="
    # Create the log up front so tail -F can attach instantly without the
    # brief "waiting for file" pause that would otherwise show up between
    # the nohup fork and the first write.
    : > "${LOG_FILE}"
    # Line-buffered so the log (and `tail -F` below) receive progress lines
    # as soon as they are emitted rather than in fsync-flush chunks.
    nohup "${BUF[@]}" "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    pid=$!
    # Don't let the EXIT trap delete the rendered config while the child
    # is still using it; reset the trap and let the log file reference it.
    trap - EXIT
    echo "  pid          : ${pid}"
    echo "  log file     : ${LOG_FILE}"
    echo "  gpu live     : watch -n 2 nvidia-smi"
    [[ "${LOG_BACKEND}" == "wandb" && -n "${WANDB_API_KEY:-}" ]] && \
        echo "  wandb        : https://wandb.ai/${WANDB_ENTITY:-$(whoami)}/${WANDB_PROJECT}/runs/${WANDB_NAME}"

    if [[ ${FOLLOW} -eq 1 ]]; then
        echo
        echo "=== Following ${LOG_FILE} (Ctrl-C detaches tail; training keeps running) ==="
        # tail -F retries across log rotation. -n +1 starts from the top
        # so the user sees initialization output, not just new lines.
        # Exit cleanly on Ctrl-C without signalling the nohup'd child.
        trap 'echo; echo "[detached] training still running as pid ${pid}."; echo "[detached] reattach with: tail -F ${LOG_FILE}"; exit 0' INT
        tail -n +1 -F "${LOG_FILE}"
    else
        echo "  follow log   : tail -F ${LOG_FILE}"
    fi
    exit 0
fi

echo "=== Streaming to stdout + ${LOG_FILE} ==="
echo
# Line-buffered so progress flows through the `tee` pipe in real time;
# without BUF, piping to `tee` would switch autotrain's stdout to block
# buffering (4-8 KB) and progress lines would appear in chunks.
{
    "${BUF[@]}" "${CMD[@]}"
    status=$?
    echo
    echo "=== AutoTrain finished ==="
    echo "  finished : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "  exit     : ${status}"
    if [[ ${status} -eq 0 ]]; then
        echo "  model    : https://huggingface.co/${HF_USERNAME}/${PROJECT_NAME}"
        echo
        echo "Next: benchmark the pushed model with"
        echo "  ${SCRIPT_DIR}/run_athenabench.sh \\"
        echo "      --repo-id ${HF_USERNAME}/${PROJECT_NAME}"
    fi
    exit ${status}
} 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
