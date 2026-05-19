#!/bin/bash

# v21 single-phase narrow-drilling SFT of
# asg-ai/athena-cti-sft-gemma4-31b-v21-taa on the v21 CyberSOCEval
# letter-set shard (ift_data_2026_05_18_v21_cse). Stage 3 of the v21
# chain on the Gemma 4 31B architecture (v21_plan.txt §7.5); the
# resulting checkpoint is the in-plan ship candidate (v21_plan.txt §3
# stops at CSE for v18.1 parity). Recipe is verbatim mirror of
# run_sft_qwen25_14b_v21_final.sh -- only the base model, template
# (qwen -> gemma4), SAFE_MODEL path component, HF push targets, and
# attention impl (auto -> sdpa) change.
#
# Why v21 reuses the v21 (=v18.1) CSE shard verbatim:
#   v21 only repaired the Core stage on Qwen2.5-14B; the v17.1 chained
#   CSE recipe with Shuffle: mcq_multi already balanced the letter
#   distribution across A-H and posted the desired CSE-TI / CSE-MAL
#   gains in the v18 chain; no template changes are required for v21
#   (v21 rebuilds the CSE shard from the byte-identical v17.1/v18.1
#   template for end-to-end reproducibility, not because the recipe
#   changed).
#
# Gemma 4 SFT specifics: see run_sft_gemma4_31b_v21_core.sh header for
# the LlamaFactory template (`gemma4`), the head_dim=512 / FlashAttention
# constraint that pins --flash_attn sdpa, and the multimodal-weights
# footprint note.
#
# Recipe (verbatim mirror of run_sft_qwen25_14b_v21_final.sh / v16+v17.1-CSE):
#   - Base model    : asg-ai/athena-cti-sft-gemma4-31b-v21-taa
#                     (HF; overridable via --base-model)
#   - Dataset       : ift_data_2026_05_18_v21_cse  (~14-19K rows; CSE shape)
#   - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#   - Effective batch 16   (per_device 1 x grad_accum 2 x 8 GPUs)
#   - eval/save every 100 steps
#   - --max-samples 33000
#   - Gradient checkpointing OFF on GPU_COUNT==8 (auto)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-gemma4-31b-v21-cse
#     (the in-plan ship candidate; an optional Recalibrate touch-up
#      stage is available via run_sft_gemma4_31b_v21_recalibrate.sh and
#      is on the default chain wrapper path because the Qwen 14B v21
#      Recalibrate variant outperformed v21-cse by ~1.5 Total points
#      via VSP recovery -- see tmpl_gen/templates/05182026/README-21.md
#      §"Findings" and v21_plan.txt §7.2 for the recurring CSE-VSP
#      erosion / Recalibrate-recovery pattern.)
#
# Estimated wall-time (31B is ~2.2x heavier than Qwen2.5-14B; B300
# compute ~2-3x H100; SDPA ~1.5-2x slower than FA at attention):
#   8xB300 (288GB/GPU)    : ~4-6 h (corpus size matches v17.1).
#   8xH100 80GB SXM       : ~6-8 h.
#   4x H100 / smaller     : not recommended; per-rank weight shard
#                           doubles and exceeds 80GB even with GC on.
#
# Full v21 chain on Gemma 4 31B (run sequentially after each push):
#   1. ./run_sft_gemma4_31b_v21_core.sh        # broad + axis   -> v21-core
#   2. ./run_sft_gemma4_31b_v21_plus_taa.sh    # TAA Classic    -> v21-taa
#   3. ./run_sft_gemma4_31b_v21_final.sh       # CSE drill      -> v21-cse
#   4. ./run_sft_gemma4_31b_v21_recalibrate.sh # touch-up       -> v21-recalibrate
# Or via wrapper: ./run_sft_gemma4_31b_v21_chain.sh (TAA -> CSE -> Recalibrate).
#
# Estimated total wall-clock (Core + chain wrapper):
#   8xB300 (288GB/GPU)    : ~23-33 h.
#   8xH100 80GB SXM       : ~35-48 h.
#
# Usage:
#   ./run_sft_gemma4_31b_v21_final.sh [--repo-id USER/NAME]
#                                      [--base-model HF_REPO|LOCAL_DIR]
#                                      [--output-dir DIR]
#                                      [--report-to wandb|none]
#                                      [--offload | --no-offload]
#                                      [--skip-eval] [--resume]
#                                      [--dry-run]
#
# --skip-eval / --resume mirror the v21_core launcher; see that script's
# header for the in-training eval OOM rationale.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
DRY_RUN=0
OFFLOAD="auto"
SKIP_EVAL=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --base-model)   BASE_MODEL="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        --skip-eval)    SKIP_EVAL=1;       shift ;;
        --resume)       RESUME=1;          shift ;;
        -h|--help) sed -n '3,73p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-gemma4-31b-v21-cse"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-gemma4-31b-v21-taa"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="google_gemma-4-31B-it"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_cse_${TIMESTAMP}"

DATASET="ift_data_2026_05_18_v21_cse"
VAL_NAME="ift_data_2026_05_18_v21_cse_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21-CSE dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build the v21 CSE shard from the byte-identical v17.1/v18.1 template." >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_cse.txt \\" >&2
        echo "           _v21_cse_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_18_v21_cse.raw.json \\" >&2
        echo "           10 3500" >&2
        echo "         echo \"PID=\$!\" > _v21_cse_build/build.pid" >&2
        echo "         nohup bash _v21_cse_build/watcher.sh > _v21_cse_build/watcher.log 2>&1 &" >&2
        exit 2
    fi
done


if [[ -n "${GPU_COUNT_OVERRIDE:-}" ]]; then
    GPU_COUNT="${GPU_COUNT_OVERRIDE}"
    GPU_PROBE_SOURCE="override"
else
    GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
    GPU_PROBE_SOURCE="torch"
    if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" == "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        NVIDIA_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ -n "${NVIDIA_COUNT}" && "${NVIDIA_COUNT}" != "0" ]]; then
            GPU_COUNT="${NVIDIA_COUNT}"
            GPU_PROBE_SOURCE="nvidia-smi (torch probe returned 0)"
        fi
    fi
    GPU_COUNT="${GPU_COUNT:-0}"
fi
if [[ "${GPU_COUNT}" == "0" && ${DRY_RUN} -eq 0 ]]; then
    echo "[FAIL] GPU probe returned 0 (source: ${GPU_PROBE_SOURCE}). See run_sft_gemma4_31b_v21_core.sh for diagnostic steps." >&2
    echo "       Override: GPU_COUNT_OVERRIDE=N ./run_sft_gemma4_31b_v21_final.sh ..." >&2
    exit 3
fi

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 8 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))

D_BATCH=1; D_GA=$(( 16 / (D_BATCH * EFFECTIVE_GPUS) )); [[ ${D_GA} -lt 1 ]] && D_GA=1

# 31B at cutoff=4096 packing=on bs=1 on 8xB300 (288GB/GPU) is light
# enough that GC-off is comfortable. The auto policy is kept in sync
# with the Qwen 14B / Llama 8B v21 final launchers: GC disabled when
# GPU_COUNT==8, GC enabled when <8.
GC_FLAG="--disable_gradient_checkpointing True"
[[ "${GPU_COUNT}" -lt 8 ]] && GC_FLAG=""

# --flash_attn sdpa override required for Gemma 4 head_dim=512
# (Dao-AILab/flash-attention#2427). See core launcher for full note.
EXTRA_BASE="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --flash_attn sdpa ${GC_FLAG}"
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    EXTRA_COMMON="${EXTRA_BASE} --eval_strategy no"
else
    EXTRA_COMMON="${EXTRA_BASE} --per_device_eval_batch_size 1 --eval_dataset ${VAL_NAME} --val_size 0"
fi

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# See run_sft_gemma4_31b_v21_core.sh header for why this override is
# required for every Gemma 4 launcher (LF check_version pins
# transformers<=5.2.0 but Gemma 4 text-only needs >=5.7.0).
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] expected 8 GPUs (8xB300 / 8xH100); detected ${GPU_COUNT}. Recipe was sized for 8x; effective batch will reflect detected count: eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )
RESUME_FLAG=(); [[ ${RESUME} -eq 1 ]] && RESUME_FLAG=( --resume )

run_v21() {
    echo "=== v21 (Gemma 4 31B): CyberSOCEval-letter-set narrow drill from v21-taa (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe; CHAINED] ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASET}" --template gemma4 --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${D_BATCH} --grad-accum ${D_GA} \
        --cutoff 4096 --save-steps 100 --eval-steps 100 --packing true \
        --max-samples 33000 --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${D_BATCH} grad_accum=${D_GA} -> eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS )) (target 16)"
echo "  base model   : ${BASE_MODEL}"
echo "  dataset      : ${DATASET}  (eval: ${VAL_NAME})"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  flash-attn   : sdpa  (head_dim=512; FA #2427 pending)"
echo "  skip-eval    : $([[ ${SKIP_EVAL} -eq 1 ]] && echo on || echo off)"
echo "  resume       : $([[ ${RESUME} -eq 1 ]] && echo on || echo off)"
echo

run_v21
