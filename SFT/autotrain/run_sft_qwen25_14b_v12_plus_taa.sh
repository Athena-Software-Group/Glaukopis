#!/bin/bash

# v12+TAA single-phase narrow-drilling SFT of asg-ai/athena-cti-sft-qwen25-14b-v12
# on the v14 TAA shard (ift_data_2026_05_08_v14_taa, 32,783 rows). First W1
# experiment of the v15 architecture (parallel-branching specialists off a
# frozen v12 baseline; see tmpl_gen/templates/05082026/v15_plan.txt).
#
# Why v12+TAA exists:
#   v14.1 demonstrated that chained five-pass SFT regressed below v12 on
#   the weighted total (49.8 vs 57.3) and below the Qwen base on TAA
#   Classic. The v15 hypothesis is that narrow per-axis SFT applied
#   independently to a healthy v12 baseline (rather than stacked into one
#   long chain) will preserve v12's other-axis capability while moving
#   only the targeted axis. v12+TAA is the simplest possible test of
#   this hypothesis: train one narrow specialist (TAA) off v12 and bench
#   to see whether (a) TAA improves vs v12 and (b) other axes regress.
#
# Recipe (verbatim mirror of v14.1 Phase D-TAA, with v12 as the base):
#   - Base model    : asg-ai/athena-cti-sft-qwen25-14b-v12  (HF; overridable via --base-model)
#   - Dataset       : ift_data_2026_05_08_v14_taa           (32,783 rows; CANON excluded)
#   - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#   - Effective batch 16   (per_device 1 x grad_accum 2 x 8 GPUs)
#   - eval/save every 100 steps
#   - --max-samples 33000  (matches v14.1 Phase D-TAA)
#   - Gradient checkpointing OFF (v14.1 hot-fix carried forward)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v12-plus-taa
#
# Decision criteria (post-bench; see v15_plan.txt §4):
#   1. TAA up AND other axes >= v12 - 2pp  -> ship as standalone specialist
#   2. TAA up BUT other axes regressed     -> escalate to merge sweep (mergekit)
#   3. TAA <= v12                          -> data/recipe issue; halt and re-examine
#
# Estimated wall-time on 8xH100: ~6-8 h (matches v14.1 Phase D-TAA).
#
# Usage:
#   ./run_sft_qwen25_14b_v12_plus_taa.sh [--repo-id USER/NAME]
#                                        [--base-model HF_REPO|LOCAL_DIR]
#                                        [--output-dir DIR]
#                                        [--report-to wandb|none]
#                                        [--offload | --no-offload]
#                                        [--dry-run]

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
        -h|--help) sed -n '3,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v12-plus-taa"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v12"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v12_plus_taa_${TIMESTAMP}"

DATASET="ift_data_2026_05_08_v14_taa"
VAL_NAME="ift_data_2026_05_08_v14_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v14 dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05082026/Sophia-CTI-Templates-v14.txt \\" >&2
        echo "           _v14_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_08_v14.raw.json \\" >&2
        echo "           10 2000" >&2
        echo "         bash _v14_build/watcher.sh   # all 9 phases (incl. four-shard split)" >&2
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

# v9 narrow recipe: per-device batch 1, eff batch 16. On 8xH100 this is
# per_device 1 x grad_accum 2 x 8 GPUs = 16. Verbatim from v14.1 Phase D-TAA.
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

run_v12_plus_taa() {
    echo "=== v12+TAA (Qwen2.5-14B): TAA narrow drill from v12 (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe] ==="
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

run_v12_plus_taa
