#!/bin/bash

# v13 two-phase full-parameter SFT of Qwen2.5-14B-Instruct on the v13
# corpus (tmpl_gen/templates/05072026/v13_plan.txt §6). Branched from
# the v12 three-phase launcher (run_sft_qwen25_14b_v12.sh); Phase C
# dropped, both phases reverted to the v9 hyperparameter shape.
#
# Why v9-shape revert (vs v12's three-phase):
#   v12 ran Phase B at cutoff=16384 / packing=OFF / eff_batch=8; the
#   confounded knob change vs v9 (8192/on/16) coincided with RMS
#   regressing 65.8 -> 61.85. v12's Phase C (15-step TAA.CANON
#   memorisation) bumped TAA-strict +6 pp but destroyed TAA-plausible
#   by -32 pp -- net negative. v13 isolates: revert Phase B knobs to
#   v9 values, drop Phase C entirely, ride TAA.CANON supervision via
#   inclusion in the Phase A broad+canon shard.
#
# Phase shape:
#   Phase A -- broad knowledge anchor + TAA.CANON memorisation
#     - Datasets   : ift_data_2026_05_07_v13_broad_plus_canon,
#                    tulu_3_sft_mixture, alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 8192, packing ON
#     - Effective batch 16
#     - --max-samples 260000
#
#   Phase B -- RMS+ATE+VSP+RCM+SOC catalog drills (v9-shape REVERT)
#     - Datasets   : ift_data_2026_05_07_v13_axis
#     - 1 epoch, lr 5e-6, cutoff 8192, packing ON   (REVERT vs v12)
#     - Effective batch 16                          (REVERT vs v12)
#     - eval/save every 400 steps
#     - --model points at Phase A's output dir
#     - --max-samples 55000
#     - Final merged model pushed to HF
#
# SOC dual-shard supervision: SOC.* + SOC.GEN.* appear in BOTH the
# broad+canon shard AND the axis shard (per §6.3 split_corpus
# --two-phase). v9's SOC retention shape proved this is not
# over-training; v12 dropping SOC from Phase B caused the -9.4 pp
# SOC regression that v13 fixes.
#
# Only Phase B's final merged model is pushed to HF.
#
# Usage:
#   ./run_sft_qwen25_14b_v13.sh [--repo-id USER/NAME]
#                               [--phase-a-dir DIR] [--phase-b-dir DIR]
#                               [--report-to wandb|none]
#                               [--phase a|b|ab]   # default: ab
#                               [--offload | --no-offload]
#                               [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
PHASE_A_DIR=""
PHASE_B_DIR=""
REPORT_TO="wandb"
PHASE="ab"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --phase-a-dir)  PHASE_A_DIR="$2";  shift 2 ;;
        --phase-b-dir)  PHASE_B_DIR="$2";  shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --phase)        PHASE="$2";        shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        -h|--help) sed -n '3,47p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|ab) ;; *) echo "--phase must be a|b|ab" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v13"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${PHASE_A_DIR}" ]] && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v13_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]] && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v13_phase_b_${TIMESTAMP}"

PHASE_A_DATASETS="ift_data_2026_05_07_v13_broad_plus_canon,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_05_07_v13_axis"
VAL_NAME="ift_data_2026_05_07_v13_val"

for ds in ift_data_2026_05_07_v13_broad_plus_canon ift_data_2026_05_07_v13_axis "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v13 dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05072026/Sophia-CTI-Templates-v13.txt \\" >&2
        echo "           _v13_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_07_v13.raw.json \\" >&2
        echo "           10 2000" >&2
        echo "         bash _v13_build/watcher.sh   # all 9 phases" >&2
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
A_BATCH=1; A_GA=$(( 16 / (A_BATCH * EFFECTIVE_GPUS) )); [[ ${A_GA} -lt 1 ]] && A_GA=1
B_BATCH=1; B_GA=$(( 16 / (B_BATCH * EFFECTIVE_GPUS) )); [[ ${B_GA} -lt 1 ]] && B_GA=1

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --eval_dataset ${VAL_NAME} --val_size 0"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_phase_a() {
    echo "=== v13 Phase A (Qwen2.5-14B): broad knowledge + TAA.CANON anchor (cutoff=8192, packing=on, lr=1e-5) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen2.5-14B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
        --max-samples 260000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_b() {
    echo "=== v13 Phase B (Qwen2.5-14B): RMS+ATE+VSP+RCM+SOC axis drill (cutoff=8192, packing=on, lr=5e-6) [v9 REVERT] ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${B_BATCH} --grad-accum ${B_GA} \
        --cutoff 8192 --save-steps 400 --eval-steps 400 --packing true \
        --max-samples 55000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_B_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}"
echo "  phase A dir  : ${PHASE_A_DIR}"
echo "  phase B dir  : ${PHASE_B_DIR}"
echo "  hf repo      : ${REPO_ID}  (only Phase B is pushed)"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

case "${PHASE}" in
    a)  run_phase_a ;;
    b)  run_phase_b ;;
    ab) run_phase_a; run_phase_b ;;
esac
