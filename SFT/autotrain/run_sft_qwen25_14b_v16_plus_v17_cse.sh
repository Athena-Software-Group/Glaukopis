#!/bin/bash

# v16+v17-CSE chained narrow-drilling SFT of asg-ai/athena-cti-sft-qwen25-14b-v16
# on the v17 CyberSOCEval-letter-set shard (ift_data_2026_05_11_v17_cse). FIRST
# chained build in the project; every prior vintage trained off the frozen v12
# baseline. v17 layers a single-shape JSON-letter-set head on top of the v16 TAA
# specialist to lift the CyberSOCEval-Malware (10.69%) and CyberSOCEval-TI
# (30.63%) accuracy axes that v16 left on the table.
# See tmpl_gen/templates/05112026/v17_plan.txt.
#
# Why v17 exists:
#   v16 (asg-ai/athena-cti-sft-qwen25-14b-v16) lifted CSE-TI from 3.74% (v12)
#   to 30.63% accuracy and CSE-Malware from 7.06% to 10.69%, but neither
#   axis cleared the ~50% bar implied by their avg_score figures (58.54% TI,
#   45.15% Malware). The accuracy/avg_score gap is consistent with the v16
#   model producing the right SEMANTIC content but the wrong OUTPUT SHAPE:
#   the formal CSE evaluator scores Jaccard similarity on a JSON object
#   {"correct_answers": [...]} and v16 was trained on prose+letter JS.TAA.*
#   templates, not multi-select letter-set JSON.
#
#   v17 is a chained narrow SFT that introduces only the missing output
#   shape across two benchmark-mirrored families:
#     JS.CSE.TI.{GRP,MAL,CMP,NEG}.*  -> <json_object>{"correct_answers":...}</json_object>
#     JS.CSE.MAL.{RPT,TAC,TGT,NEG}.* -> bare {"correct_answers":...}
#   Both prefixes are synthetic (Neo4j-derived); no contamination by
#   construction (zero rows from the CrowdStrike CSE corpus).
#
# Recipe (verbatim mirror of v16; only the base-model and dataset pointers change):
#   - Base model    : asg-ai/athena-cti-sft-qwen25-14b-v16  (CHAINED off v16,
#                     not v12; this is the only architectural delta from v16)
#   - Dataset       : ift_data_2026_05_11_v17_cse           (~14-16K rows; CSE shape)
#   - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#   - Effective batch 16   (per_device 1 x grad_accum 2 x 8 GPUs)
#   - eval/save every 100 steps
#   - --max-samples 33000  (matches v16 / v15 W1 / v14.1 Phase D-TAA)
#   - Gradient checkpointing OFF (v14.1 hot-fix carried forward)
#   - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v17
#
# Decision criteria (post-bench; see v17_plan.txt §4):
#   A. CSE-Malware AND CSE-TI up by >=10pp AND TAA Classic >= v16 - 2pp
#                                                  -> v17 ships as chained head
#   B. CSE up BUT TAA regressed                    -> mergekit alpha sweep
#                                                     against v16 weights
#   C. CSE flat                                    -> output-shape was not the
#                                                     bottleneck; author v18
#                                                     with deeper CSE templates
#   D. Net regression                              -> diagnose; do not deploy
#
# Estimated wall-time on 8xH100: ~4-6 h (smaller corpus than v16).
#
# Usage:
#   ./run_sft_qwen25_14b_v16_plus_v17_cse.sh [--repo-id USER/NAME]
#                                            [--base-model HF_REPO|LOCAL_DIR]
#                                            [--output-dir DIR]
#                                            [--report-to wandb|none]
#                                            [--offload | --no-offload]
#                                            [--dry-run]

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
        -h|--help) sed -n '3,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v17"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen25-14b-v16"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v16_plus_v17_cse_${TIMESTAMP}"

DATASET="ift_data_2026_05_11_v17_cse"
VAL_NAME="ift_data_2026_05_11_v17_val"

for ds in "${DATASET}" "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v17 dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05112026/Sophia-CTI-Templates-v17.txt \\" >&2
        echo "           _v17_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_11_v17.raw.json \\" >&2
        echo "           10 3500" >&2
        echo "         echo \"PID=\$!\" > _v17_build/build.pid" >&2
        echo "         nohup bash _v17_build/watcher.sh > _v17_build/watcher.log 2>&1 &" >&2
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
# per_device 1 x grad_accum 2 x 8 GPUs = 16. Verbatim from v16.
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

run_v16_plus_v17_cse() {
    echo "=== v16+v17-CSE (Qwen2.5-14B): CyberSOCEval-letter-set narrow drill from v16 (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe; CHAINED] ==="
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

run_v16_plus_v17_cse
