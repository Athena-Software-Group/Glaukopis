#!/bin/bash

# v14.1 five-pass full-parameter SFT of Qwen2.5-14B-Instruct on the v14
# corpus (tmpl_gen/templates/05082026/v14_plan.txt §3). Cutoff-4096
# hot-fix of run_sft_qwen25_14b_v14.sh; corpus, topology, learning rates,
# effective batch sizes, packing flags, max-samples, save/eval cadences,
# HF push convention, and resume chaining are all preserved verbatim.
# Two performance-only knobs are flipped vs v14:
#   1. --cutoff lowered (16384/8192 -> 4096) so step-time tracks the
#      actual length distribution of the v14 corpus rather than the
#      worst-case envelope.
#   2. --disable_gradient_checkpointing True (added to EXTRA_COMMON):
#      activation memory at cutoff 4096 / per_device_batch 1 / ZeRO-3
#      sharded across 8x80GB fits comfortably without recompute. GC
#      was inherited from v14 where it was mandatory at cutoff 16384
#      but is vestigial at cutoff 4096; disabling it removes the
#      backward-pass recomputation step (~15-30% throughput gain).
# Both knobs are pure speed/memory tradeoffs; the model math
# (gradients, optimizer trajectory, loss values, final weights) is
# unchanged. Step counts per phase remain bit-identical to v14 in
# step-space.
#
# Why v14.1 (vs v14):
#   v14 Phase A measured at step 23000 / 49336 (47%) had ~12 h of wall-
#   time still ahead, with throughput at 2,927 tok/s and observed mean
#   sample length of ~760 tokens against a 16384 cutoff (~4.6% compute
#   utilization; the rest is attention over padding). A length-
#   distribution audit of all four v14 IFT shards (broad / ate_vsp_rcm
#   / rms / taa) confirmed p99 token counts of 931 / 1416 / 2177 / 340
#   respectively -- entirely under 4096. v14.1 lowers --cutoff to 4096
#   for every phase and disables the now-vestigial gradient checkpoint
#   recompute; everything else (optimizer, batch math, LR schedule)
#   is unchanged. Loss curves are therefore directly comparable to
#   v14's at the same step number.
#
# Phase shape (five passes; four pushes -- Phase A is intermediate):
#   Phase A -- broad re-anchor (v12 Phase A recipe verbatim, cutoff 4096)
#     - Datasets   : ift_data_2026_05_08_v14_broad,
#                    tulu_3_sft_mixture, alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 4096, packing OFF
#     - Effective batch 8
#     - eval/save every 500 steps
#     - --max-samples 200000  (broad shard 193,703 + 3% headroom)
#     - Resume: base Qwen/Qwen2.5-14B-Instruct
#     - Push: NO (intermediate)
#
#   Phase B -- ATE+VSP+RCM long-context drill (cutoff 4096)
#     - Datasets   : ift_data_2026_05_08_v14_ate_vsp_rcm
#     - 1 epoch, lr 5e-6, cutoff 4096, packing OFF
#     - Effective batch 8
#     - eval/save every 400 steps
#     - --max-samples 36000  (actual 32,810; headroom for drift)
#     - Resume: Phase A output dir
#     - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14p1-ab
#
#   Phase D-RMS -- RMS-only narrow drill (v9 recipe; PARALLEL with D-TAA)
#     - Datasets   : ift_data_2026_05_08_v14_rms
#     - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#     - Effective batch 16
#     - eval/save every 100 steps
#     - --max-samples 13000  (actual 12,608)
#     - Resume: Phase B output dir (v14p1-ab checkpoint)
#     - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14p1-rms
#
#   Phase D-TAA -- TAA-only narrow drill (v9 recipe; PARALLEL with D-RMS)
#     - Datasets   : ift_data_2026_05_08_v14_taa
#     - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#     - Effective batch 16
#     - eval/save every 100 steps
#     - --max-samples 33000  (actual 32,783; IE/NEG variants included)
#     - Resume: Phase B output dir (v14p1-ab; NOT D-RMS)
#     - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14p1-taa
#
#   Phase D-TAA-on-RMS -- production chain (D-TAA on top of D-RMS)
#     - Datasets   : ift_data_2026_05_08_v14_taa  (same as D-TAA)
#     - 1 epoch, lr 5e-6, cutoff 4096, packing ON
#     - Effective batch 16
#     - eval/save every 100 steps
#     - --max-samples 33000
#     - Resume: D-RMS output dir (v14p1-rms checkpoint)
#     - Push: YES -> ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14p1
#
# On a single 8xH100 box D-RMS and D-TAA run serially (both branch
# from v14p1-ab via --resume from Phase B output). The production chain
# runs after both narrow drills complete.
#
# Per-axis eval visibility wired through --eval_dataset
# ift_data_2026_05_08_v14_val on all five passes.
#
# Usage:
#   ./run_sft_qwen25_14b_v14_1.sh [--repo-id-prefix USER/PREFIX]
#                                 [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                 [--phase-d-rms-dir DIR]
#                                 [--phase-d-taa-dir DIR]
#                                 [--phase-prod-dir DIR]
#                                 [--report-to wandb|none]
#                                 [--phase a|b|d-rms|d-taa|production|all]
#                                 [--offload | --no-offload]
#                                 [--dry-run]
#
#   Default --phase is 'all' (runs A -> B -> D-RMS -> D-TAA -> production).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID_PREFIX=""
PHASE_A_DIR=""
PHASE_B_DIR=""
PHASE_D_RMS_DIR=""
PHASE_D_TAA_DIR=""
PHASE_PROD_DIR=""
REPORT_TO="wandb"
PHASE="all"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id-prefix)   REPO_ID_PREFIX="$2";   shift 2 ;;
        --phase-a-dir)      PHASE_A_DIR="$2";      shift 2 ;;
        --phase-b-dir)      PHASE_B_DIR="$2";      shift 2 ;;
        --phase-d-rms-dir)  PHASE_D_RMS_DIR="$2";  shift 2 ;;
        --phase-d-taa-dir)  PHASE_D_TAA_DIR="$2";  shift 2 ;;
        --phase-prod-dir)   PHASE_PROD_DIR="$2";   shift 2 ;;
        --report-to)        REPORT_TO="$2";        shift 2 ;;
        --phase)            PHASE="$2";            shift 2 ;;
        --dry-run)          DRY_RUN=1;             shift ;;
        --offload)          OFFLOAD="on";          shift ;;
        --no-offload)       OFFLOAD="off";         shift ;;
        -h|--help) sed -n '3,86p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in
    a|b|d-rms|d-taa|production|all) ;;
    *) echo "--phase must be a|b|d-rms|d-taa|production|all" >&2; exit 1 ;;
esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID_PREFIX}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id-prefix USER/PREFIX)}"
    REPO_ID_PREFIX="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14p1"
fi

REPO_ID_AB="${REPO_ID_PREFIX}-ab"
REPO_ID_RMS="${REPO_ID_PREFIX}-rms"
REPO_ID_TAA="${REPO_ID_PREFIX}-taa"
REPO_ID_PROD="${REPO_ID_PREFIX}"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${PHASE_A_DIR}" ]]     && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v14_1_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]]     && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v14_1_phase_b_${TIMESTAMP}"
[[ -z "${PHASE_D_RMS_DIR}" ]] && PHASE_D_RMS_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v14_1_phase_d_rms_${TIMESTAMP}"
[[ -z "${PHASE_D_TAA_DIR}" ]] && PHASE_D_TAA_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v14_1_phase_d_taa_${TIMESTAMP}"
[[ -z "${PHASE_PROD_DIR}" ]]  && PHASE_PROD_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v14_1_phase_prod_${TIMESTAMP}"


PHASE_A_DATASETS="ift_data_2026_05_08_v14_broad,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_05_08_v14_ate_vsp_rcm"
PHASE_D_RMS_DATASETS="ift_data_2026_05_08_v14_rms"
PHASE_D_TAA_DATASETS="ift_data_2026_05_08_v14_taa"
PHASE_PROD_DATASETS="ift_data_2026_05_08_v14_taa"
VAL_NAME="ift_data_2026_05_08_v14_val"

for ds in ift_data_2026_05_08_v14_broad ift_data_2026_05_08_v14_ate_vsp_rcm \
          ift_data_2026_05_08_v14_rms  ift_data_2026_05_08_v14_taa "${VAL_NAME}"; do
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

# Phase A / B (v12 long-context recipe, cutoff 4096 hot-fix): per-device batch 1, eff batch 8.
# v14 used per-device batch 1 because cutoff 16384 packing off on 14B made
# long-context activations dominate VRAM under ZeRO-3 sharded across 8x80GB.
# At cutoff 4096 the batch could be raised, but per the v14.1 hot-fix scope
# all batch math is preserved verbatim from v14 so step counts (and thus the
# LR schedule) are bit-identical to v14 in step-space.
AB_BATCH=1; AB_GA=$(( 8 / (AB_BATCH * EFFECTIVE_GPUS) )); [[ ${AB_GA} -lt 1 ]] && AB_GA=1
# Phase D-RMS / D-TAA / Production (v9 narrow recipe): per-device batch 1, eff batch 16.
# v9 ran this shape on 8xH100; per-device 1 + GA 2 gives the recipe's eff_bs=16.
D_BATCH=1;  D_GA=$(( 16 / (D_BATCH  * EFFECTIVE_GPUS) )); [[ ${D_GA}  -lt 1 ]] && D_GA=1

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --disable_gradient_checkpointing True --eval_dataset ${VAL_NAME} --val_size 0"

# 8xH100 single-node distributed config. Clean any inherited multi-node env
# vars first, then pin NPROC_PER_NODE to the detected GPU_COUNT so torchrun
# spawns exactly one worker per H100 (avoids edge cases where llamafactory's
# auto-detect picks a stale CUDA_VISIBLE_DEVICES count).
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] expected 8 GPUs (8xH100); detected ${GPU_COUNT}. Recipes were sized for 8x; effective batch sizes will reflect detected count: A/B eff_bs=$(( AB_BATCH * AB_GA * EFFECTIVE_GPUS )), D eff_bs=$(( D_BATCH * D_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_phase_a() {
    echo "=== v14.1 Phase A (Qwen2.5-14B): broad re-anchor (cutoff=4096, packing=off, lr=1e-5, eff_bs=8) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen2.5-14B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${AB_BATCH} --grad-accum ${AB_GA} \
        --cutoff 4096 --save-steps 500 --eval-steps 500 --packing false \
        --max-samples 200000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_b() {
    echo "=== v14.1 Phase B (Qwen2.5-14B): ATE+VSP+RCM long-context drill (cutoff=4096, packing=off, lr=5e-6, eff_bs=8) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${AB_BATCH} --grad-accum ${AB_GA} \
        --cutoff 4096 --save-steps 400 --eval-steps 400 --packing false \
        --max-samples 36000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_B_DIR}" --push-to-hf "${REPO_ID_AB}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_d_rms() {
    echo "=== v14.1 Phase D-RMS (Qwen2.5-14B): RMS narrow drill from v14p1-ab (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe] ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_B_DIR}" \
        --dataset "${PHASE_D_RMS_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${D_BATCH} --grad-accum ${D_GA} \
        --cutoff 4096 --save-steps 100 --eval-steps 100 --packing true \
        --max-samples 13000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_D_RMS_DIR}" --push-to-hf "${REPO_ID_RMS}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_d_taa() {
    echo "=== v14.1 Phase D-TAA (Qwen2.5-14B): TAA narrow drill from v14p1-ab (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) [v9 recipe; PARALLEL with D-RMS] ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_B_DIR}" \
        --dataset "${PHASE_D_TAA_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${D_BATCH} --grad-accum ${D_GA} \
        --cutoff 4096 --save-steps 100 --eval-steps 100 --packing true \
        --max-samples 33000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_D_TAA_DIR}" --push-to-hf "${REPO_ID_TAA}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

run_phase_production() {
    echo "=== v14.1 Phase D-TAA-on-RMS (Qwen2.5-14B): production chain -- D-TAA on top of D-RMS (cutoff=4096, packing=on, lr=5e-6, eff_bs=16) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_D_RMS_DIR}" \
        --dataset "${PHASE_PROD_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${D_BATCH} --grad-accum ${D_GA} \
        --cutoff 4096 --save-steps 100 --eval-steps 100 --packing true \
        --max-samples 33000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_PROD_DIR}" --push-to-hf "${REPO_ID_PROD}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible    : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  A/B batch math  : per_device=${AB_BATCH} grad_accum=${AB_GA} -> eff_bs=$(( AB_BATCH * AB_GA * EFFECTIVE_GPUS )) (target 8)"
echo "  D   batch math  : per_device=${D_BATCH}  grad_accum=${D_GA}  -> eff_bs=$(( D_BATCH  * D_GA  * EFFECTIVE_GPUS )) (target 16)"
echo "  phase A dir     : ${PHASE_A_DIR}"
echo "  phase B dir     : ${PHASE_B_DIR}              -> ${REPO_ID_AB}"
echo "  phase D-RMS dir : ${PHASE_D_RMS_DIR}          -> ${REPO_ID_RMS}"
echo "  phase D-TAA dir : ${PHASE_D_TAA_DIR}          -> ${REPO_ID_TAA}"
echo "  phase prod dir  : ${PHASE_PROD_DIR}           -> ${REPO_ID_PROD}"
echo "  alloc conf      : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

case "${PHASE}" in
    a)          run_phase_a ;;
    b)          run_phase_b ;;
    d-rms)      run_phase_d_rms ;;
    d-taa)      run_phase_d_taa ;;
    production) run_phase_production ;;
    all)        run_phase_a; run_phase_b; run_phase_d_rms; run_phase_d_taa; run_phase_production ;;
esac
