#!/bin/bash

# v18.1 single-phase narrow-drilling SFT of asg-ai/athena-cti-sft-qwen25-14b-v18-1-taa
# on the v18 CyberSOCEval-letter-set shard (ift_data_2026_05_13_v18_cse). Stage 3
# (final) of the v18.1 chain (tmpl_gen/templates/05112026/v18_1_plan.txt §"Re-chain plan");
# the resulting checkpoint is the published v18.1 model. The CSE dataset and
# recipe are byte-identical to the v18 chain; only the base-model pointer
# changes (now picks up the v18.1+TAA checkpoint).
#
# Why v18.1 reuses the v18 CSE shard verbatim:
#   v18.1 only repaired the Core stage (CKT/RMS/VSP regressions). The v17.1
#   chained CSE recipe with Shuffle: mcq_multi already balanced the letter
#   distribution across A-H and posted the desired CSE-TI / CSE-MAL gains in
#   the v18 chain; no template changes are required for v18.1.
#
# Recipe (verbatim mirror of v18 / v16+v17.1-CSE):
#   - Base model    : asg-ai/athena-cti-sft-qwen25-14b-v18-1-taa
#                     (HF; overridable via --base-model)
#   - Dataset       : ift_data_2026_05_13_v18_cse  (~14-19K rows; CSE shape)
#   - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#   - Effective batch 16   (per_device 1 x grad_accum 2 x 8 GPUs)
#   - eval/save every 100 steps
#   - --max-samples 33000  (matches v16 / v17 / v17.1)
#   - Gradient checkpointing OFF (v14.1 hot-fix carried forward)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-1-cse
#     (the published v18.1 model; rename to '...-v18-1' (no suffix) at
#      sign-off if you want the canonical short name -- pass
#      --repo-id asg-ai/athena-cti-sft-qwen25-14b-v18-1 to override here)
#
# Estimated wall-time on 8xH100: ~4-6 h (corpus size matches v17.1).
#
# Full v18.1 chain (run sequentially after each push completes on HF):
#   1. ./run_sft_qwen25_14b_v18p1_core.sh        # broad + axis  -> v18-1-core
#   2. ./run_sft_qwen25_14b_v18p1_plus_taa.sh    # TAA Classic   -> v18-1-taa
#   3. ./run_sft_qwen25_14b_v18p1_final.sh       # CSE drill     -> v18-1-cse (final)
#
# Estimated total wall-clock on 8xH100 80GB: ~24 h.
#
# Usage:
#   ./run_sft_qwen25_14b_v18p1_final.sh [--repo-id USER/NAME]
#                                       [--base-model HF_REPO|LOCAL_DIR]
#                                       [--output-dir DIR]
#                                       [--report-to wandb|none]
#                                       [--offload | --no-offload]
#                                       [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --base-model)   BASE_MODEL="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        -h|--help) sed -n '3,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-1-cse"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v18-1-taa"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18p1_cse_${TIMESTAMP}"

DATASET="ift_data_2026_05_13_v18_cse"
VAL_NAME="ift_data_2026_05_13_v18_cse_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v18-CSE dataset missing: SFT/data/${ds}.json" >&2
        echo "       Reuse the v18 CSE shard (recipe is unchanged for v18.1)." >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05102026/Sophia-CTI-Templates-v17.1.txt \\" >&2
        echo "           _v18_cse_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_13_v18_cse.raw.json \\" >&2
        echo "           10 3500" >&2
        echo "         echo \"PID=\$!\" > _v18_cse_build/build.pid" >&2
        echo "         nohup bash _v18_cse_build/watcher.sh > _v18_cse_build/watcher.log 2>&1 &" >&2
        exit 2
    fi
done

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 4 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))

D_BATCH=1; D_GA=$(( 16 / (D_BATCH * EFFECTIVE_GPUS) )); [[ ${D_GA} -lt 1 ]] && D_GA=1

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --disable_gradient_checkpointing True --eval_dataset ${VAL_NAME} --val_size 0"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] expected 8 GPUs (8xH100); detected ${GPU_COUNT}. Recipe was sized for 8x; effective batch will reflect detected count: eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_v18p1() {
    echo "=== v18.1 (Qwen2.5-14B): CyberSOCEval-letter-set narrow drill from v18.1-taa (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe; CHAINED] ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASET}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${D_BATCH} --grad-accum ${D_GA} \
        --cutoff 4096 --save-steps 100 --eval-steps 100 --packing true \
        --max-samples 33000 --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${D_BATCH} grad_accum=${D_GA} -> eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS )) (target 16)"
echo "  base model   : ${BASE_MODEL}"
echo "  dataset      : ${DATASET}  (eval: ${VAL_NAME})"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_v18p1
