#!/bin/bash

# v18-Core two-phase full-parameter SFT of Qwen2.5-14B-Instruct on the v18
# core corpus (broad + axis shards). Stage 1 of the v17.1-pattern chained
# v18 architecture (tmpl_gen/templates/05132026/v18_plan.txt §"v17.1 chained
# architecture"); the resulting checkpoint is the base for the v18-plus-taa
# refresher (run_sft_qwen25_14b_v18_plus_taa.sh).
#
# Why v18-core mirrors the v12 two-phase shape (vs the dropped v18 monolithic
# 3-phase recipe):
#   The earlier v18 monolithic recipe baked TAA.CANON alias-resolution into
#   Phase C, which the v15 W1 post-mortem isolated as the wrong TAA flavour
#   for the AthenaBench TAA Classic axis. v18 now follows the v17.1 chained
#   pattern: the core (broad + axis) ships standalone as a v12-shape base
#   model, then a TAA-Classic refresher (v16 manifest) and a CSE drill
#   (v17.1 manifest) chain on top. TAA.CANON is dropped from the v18
#   lineage entirely.
#
# Phase shape (identical to v12 Phases A and B):
#   Phase A -- broad knowledge re-anchor (carries MCQ; v18 ships the
#              new MCQ.EXT.GLOSS family + lifted MCQ.EXT.{MITRE,SEC}
#              row counts here, plus the v16-shape TAA Classic baseline
#              that already lives in the Phase A shard)
#     - Datasets   : ift_data_2026_05_13_v18_core_a_kb_mcq_taa_soc_cm_ms_yn,
#                    tulu_3_sft_mixture, alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 8192, packing on
#     - Effective batch 16
#     - --max-samples 240000 (matches v12)
#
#   Phase B -- AthenaBench catalog recovery (v9-shape recipe; v18 lifts
#              the ATE template families here via the broader Sigma /
#              malware / multi-fact intrusion-set traversals)
#     - Datasets   : ift_data_2026_05_13_v18_core_b_rms_ate_vsp_rcm
#     - 1 epoch, lr 5e-6, cutoff 16384, packing OFF
#     - Effective batch 8 (cutoff doubled => half the effective batch)
#     - eval/save every 400 steps
#     - --model points at Phase A's output dir
#
# Only Phase B's final merged model is pushed to HF.
# Default push target: ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-core.
#
# Estimated wall-time on 8xH100 80GB: ~13 h (Phase A 8 h, Phase B 5 h).
#
# Usage:
#   ./run_sft_qwen25_14b_v18_core.sh [--repo-id USER/NAME]
#                                    [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                    [--report-to wandb|none]
#                                    [--phase a|b|ab]   # default: ab
#                                    [--offload | --no-offload]
#                                    [--dry-run]

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
        -h|--help) sed -n '3,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|ab) ;; *) echo "--phase must be a|b|ab" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-core"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${PHASE_A_DIR}" ]] && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18_core_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]] && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v18_core_phase_b_${TIMESTAMP}"

PHASE_A_DATASETS="ift_data_2026_05_13_v18_core_a_kb_mcq_taa_soc_cm_ms_yn,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_05_13_v18_core_b_rms_ate_vsp_rcm"
VAL_NAME="ift_data_2026_05_13_v18_core_val"

for ds in ift_data_2026_05_13_v18_core_a_kb_mcq_taa_soc_cm_ms_yn \
          ift_data_2026_05_13_v18_core_b_rms_ate_vsp_rcm \
          "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v18-core dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05132026/Sophia-CTI-Templates-v18.txt \\" >&2
        echo "           _v18_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_13_v18_core.raw.json \\" >&2
        echo "           10 1500" >&2
        echo "         bash _v18_build/watcher.sh   # all phases" >&2
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

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --eval_dataset ${VAL_NAME} --val_size 0"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_phase_a() {
    echo "=== v18-Core Phase A (Qwen2.5-14B): broad knowledge re-anchor (cutoff=8192, packing=on, lr=1e-5) ==="
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
    echo "=== v18-Core Phase B (Qwen2.5-14B): RMS+ATE+VSP+RCM catalog recovery (cutoff=16384, packing=off, lr=5e-6) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${B_BATCH} --grad-accum ${B_GA} \
        --cutoff 16384 --save-steps 400 --eval-steps 400 --packing false \
        --max-samples 70000 --report-to "${REPORT_TO}" \
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
