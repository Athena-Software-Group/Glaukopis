#!/bin/bash

# v18 three-phase full-parameter SFT of Qwen2.5-14B-Instruct on the v18
# corpus (tmpl_gen/templates/05132026/v18_plan.txt §6). Verbatim fork of
# run_sft_qwen25_14b_v12.sh: same three-phase recipe (Broad / Axis / TAA-
# CANON), identical hyperparameters per phase, only the dataset names and
# default HF push target are retargeted to the v18 shards.
#
# Why v18 mirrors v12 (vs the v17.1 chained variant):
#   v17.1 chained a CSE drill on top of v16 (which was v12 + a TAA refresher)
#   and recovered SOC/CKT slightly but plateaued on AthenaBench MCQ (CKT) and
#   ATE. v18 returns to the v12 three-phase scratch recipe and instead lifts
#   the upstream MCQ.EXT / ATE template families (incl. the new GLOSS family
#   sourced from NIST/MITRE/CISA/ENISA/ISO glossaries) so the additional
#   gradient signal for those two axes is in the corpus rather than in a
#   fourth chained pass.
#
# Phase shape (identical to v12):
#   Phase A -- broad knowledge re-anchor (carries MCQ; v18 ships the
#              new MCQ.EXT.GLOSS family + lifted MCQ.EXT.{MITRE,SEC}
#              row counts here, per the v12 splitter convention that
#              keeps MCQ.* in the broad shard)
#     - Datasets   : ift_data_2026_05_13_v18_broad,
#                    tulu_3_sft_mixture, alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 8192, packing on
#     - Effective batch 16
#     - --max-samples 240000 (matches v12)
#
#   Phase B -- AthenaBench catalog recovery (v9-shape recipe; v18 lifts
#              the ATE template families here via the broader Sigma /
#              malware / multi-fact intrusion-set traversals)
#     - Datasets   : ift_data_2026_05_13_v18_rms_ate_vsp_rcm
#     - 1 epoch, lr 5e-6, cutoff 16384, packing OFF
#     - Effective batch 8 (cutoff doubled => half the effective batch)
#     - eval/save every 400 steps
#     - --model points at Phase A's output dir
#
#   Phase C -- TAA.CANON alias memorization
#     - Datasets   : ift_data_2026_05_13_v18_taa_canon
#     - 1 epoch, lr 3e-6, cutoff 8192, packing on
#     - Effective batch 16
#     - --model points at Phase B's output dir
#     - Final merged model pushed to HF
#
# Only Phase C's final merged model is pushed to HF.
#
# Usage:
#   ./run_sft_qwen25_14b_v18.sh [--repo-id USER/NAME]
#                               [--phase-a-dir DIR] [--phase-b-dir DIR] [--phase-c-dir DIR]
#                               [--report-to wandb|none]
#                               [--phase a|b|c|ab|bc|all]   # default: all
#                               [--offload | --no-offload]
#                               [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
PHASE_A_DIR=""
PHASE_B_DIR=""
PHASE_C_DIR=""
REPORT_TO="wandb"
PHASE="all"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --phase-a-dir)  PHASE_A_DIR="$2";  shift 2 ;;
        --phase-b-dir)  PHASE_B_DIR="$2";  shift 2 ;;
        --phase-c-dir)  PHASE_C_DIR="$2";  shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --phase)        PHASE="$2";        shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        -h|--help) sed -n '3,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|c|ab|bc|all) ;; *) echo "--phase must be a|b|c|ab|bc|all" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${PHASE_A_DIR}" ]] && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]] && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18_phase_b_${TIMESTAMP}"
[[ -z "${PHASE_C_DIR}" ]] && PHASE_C_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18_phase_c_${TIMESTAMP}"

PHASE_A_DATASETS="ift_data_2026_05_13_v18_broad,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_05_13_v18_rms_ate_vsp_rcm"
PHASE_C_DATASETS="ift_data_2026_05_13_v18_taa_canon"
VAL_NAME="ift_data_2026_05_13_v18_val"

for ds in ift_data_2026_05_13_v18_broad ift_data_2026_05_13_v18_rms_ate_vsp_rcm ift_data_2026_05_13_v18_taa_canon "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v18 dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05132026/Sophia-CTI-Templates-v18.txt \\" >&2
        echo "           _v18_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_13_v18.raw.json \\" >&2
        echo "           10 1500" >&2
        echo "         bash _v18_build/watcher.sh   # all 9 phases" >&2
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
B_BATCH=1; B_GA=$(( 8  / (B_BATCH * EFFECTIVE_GPUS) )); [[ ${B_GA} -lt 1 ]] && B_GA=1
C_BATCH=1; C_GA=$(( 16 / (C_BATCH * EFFECTIVE_GPUS) )); [[ ${C_GA} -lt 1 ]] && C_GA=1


EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --eval_dataset ${VAL_NAME} --val_size 0"
EXTRA_PHASE_B="${EXTRA_COMMON}"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_phase_a() {
    echo "=== v18 Phase A (Qwen2.5-14B): broad knowledge re-anchor (cutoff=8192, packing=on, lr=1e-5) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen2.5-14B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
        --max-samples 240000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_b() {
    echo "=== v18 Phase B (Qwen2.5-14B): RMS+ATE+VSP+RCM+MCQ.EXT+MCQ.GLOSS catalog recovery (cutoff=16384, packing=off, lr=5e-6) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${B_BATCH} --grad-accum ${B_GA} \
        --cutoff 16384 --save-steps 400 --eval-steps 400 --packing false \
        --max-samples 70000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_B_DIR}" \
        --extra "${EXTRA_PHASE_B}" "${DRY_FLAG[@]}"
}

run_phase_c() {
    echo "=== v18 Phase C (Qwen2.5-14B): TAA.CANON alias memorization (cutoff=8192, packing=on, lr=3e-6) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_B_DIR}" \
        --dataset "${PHASE_C_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 3e-06 --batch ${C_BATCH} --grad-accum ${C_GA} \
        --cutoff 8192 --save-steps 250 --eval-steps 250 --packing true \
        --max-samples 12000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_C_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}"
echo "  phase A dir  : ${PHASE_A_DIR}"
echo "  phase B dir  : ${PHASE_B_DIR}"
echo "  phase C dir  : ${PHASE_C_DIR}"
echo "  hf repo      : ${REPO_ID}  (only Phase C is pushed)"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

case "${PHASE}" in
    a)   run_phase_a ;;
    b)   run_phase_b ;;
    c)   run_phase_c ;;
    ab)  run_phase_a; run_phase_b ;;
    bc)  run_phase_b; run_phase_c ;;
    all) run_phase_a; run_phase_b; run_phase_c ;;
esac
