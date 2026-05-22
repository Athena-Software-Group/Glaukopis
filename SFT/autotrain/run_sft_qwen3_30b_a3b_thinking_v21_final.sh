#!/bin/bash

# v21 single-phase narrow-drilling SFT of
# asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa on the v21
# CyberSOCEval-letter-set shard (ift_data_2026_05_18_v21_cse). Stage 3
# (final) of the v21 chain ported to Qwen3-30B-A3B-Thinking-2507; the
# resulting checkpoint is the Qwen3-MoE v21 ship-equivalent (mirrors the
# v18.1 three-stage ship topology). The CSE shard is byte-identical to
# the 14B/32B v21 build (template-baked, architecture-independent); only
# the base-model pointer + template change vs the 32B reference.
#
# Recipe parity with run_sft_qwen25_32b_v21_final.sh:
#   - Identical cutoff (4096), packing (on), lr (5e-6), effective batch
#     (16), max-samples (33000), save/eval steps (100).
#   - --optim adamw_8bit retained; Liger ON; --gc ON.
#
# Qwen3-MoE deltas vs Qwen2.5-32B v21-final (B300 / template / sparse):
#   - --template qwen3 (was qwen).
#   - --enable_thinking True (run_train.sh default; matches LF default;
#     model learns to autonomously emit empty <think>\n\n</think> in
#     response -- see run_sft_qwen3_30b_a3b_thinking_v21_core.sh header
#     for the full mechanism).
#   - OFFLOAD default off (was auto). 8xB300 = 288 GB HBM3e per GPU;
#     30.5B MoE ZeRO-3 shard ~15 GB/rank with adamw_8bit, leaving
#     >250 GB headroom for activations + KV at cutoff=4096 packing=on.
#   - --base-model defaults to the Qwen3-MoE v21-taa HF push target.
#
# Recipe (mirrors 32B v21-final / v18.1 three-stage ship):
#   - Base model    : asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa
#                     (HF; overridable via --base-model)
#   - Dataset       : ift_data_2026_05_18_v21_cse  (~14-19K rows; CSE shape)
#   - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#   - Effective batch 16   (per_device 1 x grad_accum 2 x 8 GPUs)
#   - eval/save every 100 steps
#   - --max-samples 33000  (matches 14B/32B v21-final)
#   - Gradient checkpointing ON
#   - adamw_8bit optimizer, Liger kernel ON
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
#
# Estimated wall-time (Qwen3-MoE 3.3B active per token vs dense 32B;
# ZeRO-3 collective traffic is on the FULL 30.5B param shard so the
# speedup over 32B at the same hardware tier is ~1.5-2x at this shape):
#   8xB300 288GB SXM       : ~5-7 h.
#   8xH200 141GB SXM       : ~7-10 h.
#   8xH100 80GB SXM        : ~9-13 h (matches 32B v21-final; sparse fwd
#                                     win is partially eaten by routing).
#
# Full v21 chain (Qwen3-MoE):
#   1. ./run_sft_qwen3_30b_a3b_thinking_v21_core.sh           # -> v21-core
#   2. ./run_sft_qwen3_30b_a3b_thinking_v21_plus_taa.sh       # -> v21-taa
#   3. ./run_sft_qwen3_30b_a3b_thinking_v21_final.sh          # -> v21-cse (final)
#   4. ./run_sft_qwen3_30b_a3b_thinking_v21_recalibrate.sh    # -> v21-recalibrate (off-plan)
# Or via chain wrapper:
#   ./run_sft_qwen3_30b_a3b_thinking_v21_chain.sh
#
# Usage:
#   ./run_sft_qwen3_30b_a3b_thinking_v21_final.sh
#       [--repo-id USER/NAME] [--base-model HF_REPO|LOCAL_DIR]
#       [--output-dir DIR] [--report-to wandb|none]
#       [--offload | --no-offload]
#       [--skip-eval] [--resume] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
DRY_RUN=0
OFFLOAD="off"
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
        -h|--help) sed -n '3,62p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen3-30B-A3B-Thinking-2507"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_cse_${TIMESTAMP}"

DATASET="ift_data_2026_05_18_v21_cse"
VAL_NAME="ift_data_2026_05_18_v21_cse_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21-CSE dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_cse.txt \\" >&2
        echo "           _v21_cse_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_18_v21_cse.raw.json \\" >&2
        echo "           10 3500" >&2
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
    echo "[FAIL] GPU probe returned 0 (source: ${GPU_PROBE_SOURCE})." >&2
    echo "       Override: GPU_COUNT_OVERRIDE=N ./run_sft_qwen3_30b_a3b_thinking_v21_final.sh ..." >&2
    exit 3
fi

DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
D_BATCH=1; D_GA=$(( 16 / (D_BATCH * EFFECTIVE_GPUS) )); [[ ${D_GA} -lt 1 ]] && D_GA=1

# Gradient checkpointing kept ON for parity with the 32B recipe; MoE
# per-expert activation spikes from dropless routing benefit from GC
# even on B300 headroom.
GC_FLAG=""

EXTRA_BASE="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit ${GC_FLAG}"
if [[ ${SKIP_EVAL} -eq 1 ]]; then
    EXTRA_COMMON="${EXTRA_BASE} --eval_strategy no"
else
    EXTRA_COMMON="${EXTRA_BASE} --per_device_eval_batch_size 1 --eval_dataset ${VAL_NAME} --val_size 0"
fi

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] recipe sized for 8 GPUs (8xB300 target); detected ${GPU_COUNT}. Effective batch preserved via grad-accum auto-scaling: eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS )) (target 16)." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )
RESUME_FLAG=(); [[ ${RESUME} -eq 1 ]] && RESUME_FLAG=( --resume )

run_v21() {
    echo "=== v21 (Qwen3-30B-A3B-Thinking-2507): CyberSOCEval-letter-set narrow drill from v21-taa (cutoff=4096, packing=on, lr=5e-6, eff_bs=16, enable_thinking=True) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASET}" --template qwen3 --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${D_BATCH} --grad-accum ${D_GA} \
        --cutoff 4096 --save-steps 100 --eval-steps 100 --packing true \
        --max-samples 33000 --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${D_BATCH} grad_accum=${D_GA} -> eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS )) (target 16)"
echo "  base model   : ${BASE_MODEL}  (template=qwen3; --enable_thinking True default)"
echo "  dataset      : ${DATASET}  (eval: ${VAL_NAME})"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  skip-eval    : $([[ ${SKIP_EVAL} -eq 1 ]] && echo on || echo off)"
echo "  resume       : $([[ ${RESUME} -eq 1 ]] && echo on || echo off)"
echo

run_v21
