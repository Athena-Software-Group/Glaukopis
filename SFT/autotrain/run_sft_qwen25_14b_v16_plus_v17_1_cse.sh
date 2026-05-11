#!/bin/bash

# v16+v17.1-CSE chained narrow-drilling SFT of asg-ai/athena-cti-sft-qwen25-14b-v16
# on the v17.1 CyberSOCEval-letter-set shard (ift_data_2026_05_12_v17_1_cse).
# v17.1 is the data-fix recovery of v17: the chained CSE specialist that
# regressed all axes when first trained off the v17 corpus. v17 hard-coded
# the answer letters into the Answer block (4 unique correct_answers tuples
# across 16,548 rows; structural mode-collapse), and the model learnt
# "emit A,B regardless of input". v17.1 keeps the manifest body byte-
# identical to v17 except every template now declares Shuffle: mcq_multi,
# which the new tmpl_parser._shuffle_mcq_options_multi engine path uses to
# permute the option block A-H and rewrite the correct_answers list to
# the new positions.
# See tmpl_gen/templates/05102026/v17_1_plan.txt.
#
# Why v17.1 exists:
#   v17 was Outcome D (net regression). Forensic analysis of the corpus
#   isolated the cause to a single defect in the manifest -- not a flaw
#   in the chained-SFT mechanism. v17.1 changes ONE variable (corpus
#   quality) and holds everything else constant so the resulting bench
#   tells us, unambiguously, whether the v17 regression was data-driven
#   (CSE up vs v16 -> chained SFT validated) or architectural (CSE flat
#   or net regression -> chained SFT competes with v16 head, mergekit or
#   parallel-branch architecture indicated for v18).
#
# Recipe (verbatim mirror of v17 -- which mirrored v16):
#   - Base model    : asg-ai/athena-cti-sft-qwen25-14b-v16  (UNCHANGED from v17)
#   - Dataset       : ift_data_2026_05_12_v17_1_cse         (~14-16K rows; CSE shape)
#   - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#   - Effective batch 16   (per_device 1 x grad_accum 2 x 8 GPUs)
#   - eval/save every 100 steps
#   - --max-samples 33000  (matches v16 / v17)
#   - Gradient checkpointing OFF (v14.1 hot-fix carried forward)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v17-1
#
# Decision criteria (post-bench; see v17_1_plan.txt §4):
#   A. CSE-Malware AND CSE-TI up by >=10pp AND TAA Classic >= v16 - 2pp
#                                                  -> v17.1 ships; v17
#                                                     deprecated on HF
#   B. CSE up BUT TAA regressed                    -> mergekit alpha sweep
#                                                     against v16 weights
#   C. CSE flat                                    -> data fix necessary
#                                                     but not sufficient;
#                                                     architectural change
#                                                     for v18 (parallel-
#                                                     branch or DPO replay)
#   D. Net regression                              -> deeper diagnosis;
#                                                     do not deploy
#
# Estimated wall-time on 8xH100: ~4-6 h (corpus size matches v17).
#
# Usage:
#   ./run_sft_qwen25_14b_v16_plus_v17_1_cse.sh [--repo-id USER/NAME]
#                                              [--base-model HF_REPO|LOCAL_DIR]
#                                              [--output-dir DIR]
#                                              [--report-to wandb|none]
#                                              [--offload | --no-offload]
#                                              [--dry-run]

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
        -h|--help) sed -n '3,57p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v17-1"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v16"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v16_plus_v17_1_cse_${TIMESTAMP}"

DATASET="ift_data_2026_05_12_v17_1_cse"
VAL_NAME="ift_data_2026_05_12_v17_1_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v17.1 dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05102026/Sophia-CTI-Templates-v17.1.txt \\" >&2
        echo "           _v17_1_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_12_v17_1.raw.json \\" >&2
        echo "           10 3500" >&2
        echo "         echo \"PID=\$!\" > _v17_1_build/build.pid" >&2
        echo "         nohup bash _v17_1_build/watcher.sh > _v17_1_build/watcher.log 2>&1 &" >&2
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

run_v16_plus_v17_1_cse() {
    echo "=== v16+v17.1-CSE (Qwen2.5-14B): CyberSOCEval-letter-set narrow drill from v16 (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe; CHAINED; corpus-fix recovery of v17] ==="
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

run_v16_plus_v17_1_cse
