#!/bin/bash

# Two-phase full-parameter SFT of Qwen2.5-14B-Instruct for the v9 corpus.
# v9 restores the v8 broad-knowledge baseline and grafts the v8.1 RMS
# catalog drills onto Phase B.
#
# Why v9 exists (v8.1 broad regression diagnosis):
#   v8.1 (run_abaligned_sft_qwen25_14b_v81.sh) recovered RMS (+8.9 pp F1
#   over v8) but regressed CKT -14.4, ATE -17.4, RCM -10.5, CyberMetric
#   -5.1 vs the v8 14B checkpoint. Root cause traced to the cap=170
#   stratified subsample of the v8.1 corpus: the rule preserved AB.RMS.*
#   and JS.RMS.* at 100% retention but capped every other family at 170
#   rows/shortname, cutting V/W/X/S/P/M from 9-35K rows down to ~1-4K rows
#   each (0.09-0.16x of v7). With Tulu/Alpaca dilution, v8.1 saw ~85K CTI
#   example-passes vs v7's ~540K and v8's ~262K -- a ~3-6x compute deficit
#   on the broad-knowledge surface that drives CKT/ATE/RCM/CyberMetric.
#
# Phase shape (Phase A identical to v8; Phase B adds v9 RMS slice):
#   Phase A -- broad knowledge re-anchor.
#     - Datasets   : ift_data_2026_04_26_combined_v7,tulu_3_sft_mixture,alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 4096, packing on
#     - Effective batch 16
#   Phase B -- format + long-context + RMS catalog drills.
#     - Datasets   : ift_data_2026_04_29_json_v8,ift_data_2026_04_29_longctx_v8,
#                    ift_data_2026_04_30_v9_rms
#     - 1 epoch, lr 5e-6, cutoff 16384, packing OFF
#     - Effective batch 8 (cutoff 4x => half the effective batch)
#     - eval/save every 400 steps, group_by_length on
#     - --model points at Phase A's output dir
#
#   The Phase B RMS slice is built first-class from its own template
#   manifest (the AB.RMS.* / JS.RMS.* templates lifted verbatim from
#   v8.1 into a self-contained file) and run through the standard
#   tmpl_gen pipeline so the v9 build does not depend on any v8.1
#   output artefact:
#
#     python tmpl_gen/scripts/tmpl_docx2json.py \
#         -i tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt \
#         -o tmpl_gen/data_generation/Sophia-CTI-Templates-v9_rms.json \
#         --count_limit 1500
#     bash tmpl_gen/data_generation/make_dataset.sh \
#         tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt \
#         _v9_rms_build/triples \
#         SFT/data/ift_data_2026_04_30_v9_rms.raw.json \
#         10 1500
#     python tmpl_gen/scripts/stratified_subsample.py \
#         --in  SFT/data/ift_data_2026_04_30_v9_rms.raw.json \
#         --out SFT/data/ift_data_2026_04_30_v9_rms.json \
#         --cap 170
#
#   stratified_subsample.py is mostly inert here because every
#   shortname in the manifest is in PRESERVE_FULL_PREFIXES; the step
#   is run for parity with v8.1 and to keep the post-processing
#   pipeline byte-for-byte identical between the two corpora. Final
#   dataset is ~12,158 rows (10,433 AB.RMS.* + 1,725 JS.RMS.*).
#
# Only Phase B's final merged model is pushed to HF.
#
# Usage:
#   ./run_abaligned_sft_qwen25_14b_v9.sh [--repo-id USER/NAME]
#                                        [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                        [--report-to wandb|none]
#                                        [--phase a|b|both]   # default: both
#                                        [--offload | --no-offload]
#                                        [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
PHASE_A_DIR=""
PHASE_B_DIR=""
REPORT_TO="wandb"
PHASE="both"
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
        -h|--help) sed -n '3,65p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|both) ;; *) echo "--phase must be a|b|both" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v9"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${PHASE_A_DIR}" ]] && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v9_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]] && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v9_phase_b_${TIMESTAMP}"

PHASE_A_DATASETS="ift_data_2026_04_26_combined_v7,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_04_29_json_v8,ift_data_2026_04_29_longctx_v8,ift_data_2026_04_30_v9_rms"

if [[ "${PHASE}" != "a" ]]; then
    for ds in ift_data_2026_04_29_json_v8 ift_data_2026_04_29_longctx_v8 ift_data_2026_04_30_v9_rms; do
        if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
            echo "[FAIL] Phase B dataset missing: SFT/data/${ds}.json" >&2
            echo "       json_v8 / longctx_v8 are produced by tmpl_gen + stitch_long_context.py;" >&2
            echo "       v9_rms is built from tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt" >&2
            echo "       (see header comment for the docx2json -> make_dataset.sh -> stratified_subsample chain)." >&2
            exit 2
        fi
    done
fi

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

A_BATCH=1
A_GA=$(( 16 / (A_BATCH * EFFECTIVE_GPUS) )); [[ ${A_GA} -lt 1 ]] && A_GA=1
B_BATCH=1
B_GA=$(( 8 / (B_BATCH * EFFECTIVE_GPUS) ));  [[ ${B_GA} -lt 1 ]] && B_GA=1

# --save_only_model True keeps checkpoints to model weights only (no
# optimizer state) so a 14B run fits inside save_total_limit 2 without
# eating ~110 GB per checkpoint. The Phase B handoff only consumes model
# weights from PHASE_A_DIR, so dropping optimizer state is safe.
#
# We deliberately do NOT pass --load_best_model_at_end: current
# transformers refuses that flag together with --save_only_model under
# DeepSpeed (full optimizer state required to restore the best checkpoint),
# and at 1 epoch with cosine LR 1e-5 / 5e-6 the final-step weights are
# effectively the best eval-loss weights anyway.
EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit"
EXTRA_PHASE_B="${EXTRA_COMMON} --group_by_length True"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_phase_a() {
    echo "=== v9 Phase A (Qwen2.5-14B): broad knowledge re-anchor (cutoff=4096, packing=on) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen2.5-14B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 4096 --save-steps 200 --eval-steps 200 --packing true \
        --max-samples 250000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_b() {
    echo "=== v9 Phase B (Qwen2.5-14B): format + long-context + RMS drills (cutoff=16384, packing=off) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${B_BATCH} --grad-accum ${B_GA} \
        --cutoff 16384 --save-steps 400 --eval-steps 400 --packing false \
        --max-samples 50000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_B_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_PHASE_B}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}"
echo "  phase A dir  : ${PHASE_A_DIR}"
echo "  phase B dir  : ${PHASE_B_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

[[ "${PHASE}" == "a" || "${PHASE}" == "both" ]] && run_phase_a
[[ "${PHASE}" == "b" || "${PHASE}" == "both" ]] && run_phase_b
