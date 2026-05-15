#!/bin/bash

# v19-Core two-phase full-parameter SFT of Qwen2.5-14B-Instruct on the
# v19 core corpus (broad + axis shards). Stages 1+2 of the v19
# reproducibility-first chain (tmpl_gen/templates/05152026/v19_plan.txt
# §4.1-§4.2). Recipe is byte-identical to run_sft_qwen25_14b_v18p1_core.sh;
# only dataset names (05_11 -> 05_15, v18p1 -> v19), the HF push target
# (v18-1-core -> v19-core), and the build-dir / plan references change.
#
# Phase shape (verbatim from v18p1-core / v18-Core):
#   Phase A -- broad knowledge re-anchor
#     - Datasets   : ift_data_2026_05_15_v19_core_a_kb_mcq_taa_soc_cm_ms_yn,
#                    tulu_3_sft_mixture, alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 8192, packing on
#     - Effective batch 16
#     - --max-samples 240000
#
#   Phase B -- AthenaBench catalog recovery
#     - Datasets   : ift_data_2026_05_15_v19_core_b_rms_ate_vsp_rcm
#     - 1 epoch, lr 5e-6, cutoff 16384, packing OFF
#     - Effective batch 8 (cutoff doubled => half the effective batch)
#     - eval/save every 400 steps
#     - --model points at Phase A's output dir
#
# Only Phase B's final merged model is pushed to HF.
# Default push target: ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v19-core.
#
# Estimated wall-time on 8xH100 80GB: ~13 h (Phase A 8 h, Phase B 5 h).
# On 4xH100 expect ~2x; Phase B (cutoff=16384, packing=off) is memory-tight
# at 14B/4 ranks -- pass --offload to force ZeRO-3 CPU offload on if Phase B
# OOMs. The auto-heuristic only flips offload on at <4 GPUs.
#
# Usage:
#   ./run_sft_qwen25_14b_v19_core.sh [--repo-id USER/NAME]
#                                    [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                    [--report-to wandb|none]
#                                    [--phase a|b|ab]   # default: ab
#                                    [--offload | --no-offload]
#                                    [--skip-eval]      # disables in-training eval
#                                    [--dry-run]
#
# --skip-eval is the targeted fix for Phase B OOM at cutoff=16384: the eval
# cross-entropy logits ([1, 16384, 152064] fp32 = ~10 GiB) crash GPU 1 at the
# first eval boundary even with batch=1, because no-offload training already
# occupies ~70 GiB per rank. Eval-during-training is monitoring-only (no
# load_best_model_at_end), so disabling it changes zero training-state
# weights -- final benchmark suite remains the authoritative quality signal.

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
SKIP_EVAL=0

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
        --skip-eval)    SKIP_EVAL=1;       shift ;;
        -h|--help) sed -n '3,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|ab) ;; *) echo "--phase must be a|b|ab" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v19-core"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
[[ -z "${PHASE_A_DIR}" ]] && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v19_core_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]] && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v19_core_phase_b_${TIMESTAMP}"

PHASE_A_DATASETS="ift_data_2026_05_15_v19_core_a_kb_mcq_taa_soc_cm_ms_yn,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_05_15_v19_core_b_rms_ate_vsp_rcm"
VAL_NAME="ift_data_2026_05_15_v19_core_val"

for ds in ift_data_2026_05_15_v19_core_a_kb_mcq_taa_soc_cm_ms_yn \
          ift_data_2026_05_15_v19_core_b_rms_ate_vsp_rcm \
          "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v19-core dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         python _v19_core_build/_neo4j_check.py" >&2
        echo "         mkdir -p _v19_core_build/triples" >&2
        echo "         nohup bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05152026/Sophia-CTI-Templates-v19_core.txt \\" >&2
        echo "           _v19_core_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_15_v19_core.raw.json \\" >&2
        echo "           2500 3500 > _v19_core_build/build.log 2>&1 &" >&2
        echo "         echo \"PID=\$!\" > _v19_core_build/build.pid" >&2
        echo "         nohup bash _v19_core_build/watcher.sh > _v19_core_build/watcher.log 2>&1 &" >&2
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
# Phase A runs cutoff=8192 with packing on; H100 80GB fits batch=2 at 14B
# under ZeRO-3 (no offload), halving micro-batches per optimizer step at
# the same effective batch of 16. Phase B stays at batch=1 because
# cutoff=16384 with packing off is memory-tight.
A_BATCH=2; A_GA=$(( 16 / (A_BATCH * EFFECTIVE_GPUS) )); [[ ${A_GA} -lt 1 ]] && A_GA=1
B_BATCH=1; B_GA=$(( 8  / (B_BATCH * EFFECTIVE_GPUS) )); [[ ${B_GA} -lt 1 ]] && B_GA=1

# --per_device_eval_batch_size 1 is a 4xH100 hardening vs the v18p1 8xH100
# baseline: HF Trainer defaults eval batch to 8 (NOT inherited from train batch),
# so the eval-time cross-entropy logits would peak at
# eval_batch * cutoff * vocab * 4 = 8 * 8192 * 152064 * 4B ~= 39.9 GiB at
# Phase A and 79.7 GiB at Phase B. With ZeRO-3 across only 4 ranks (vs 8x),
# per-rank training state already occupies ~76 GiB, leaving no headroom for
# the eval logits -- the run OOMs at the first eval_steps=500 hit. Lowering
# eval batch to 1 cuts the logits to ~5 GiB / ~10 GiB respectively.
#
# Even at eval batch=1 the Phase B logits ([1, 16384, 152064] fp32 ~= 10 GiB)
# crash at the first eval boundary on 4xH100 no-offload because Liger's fused
# linear-cross-entropy is bypassed by Trainer.prediction_step (the eval path
# falls through to nn.functional.cross_entropy on materialized logits). When
# --skip-eval is passed we replace the eval block with --eval_strategy no,
# which sets do_eval=False and prevents _maybe_log_save_evaluate from ever
# entering the prediction loop. Eval-time arguments (eval_dataset/val_size/
# per_device_eval_batch_size) are dropped because they're irrelevant when
# eval_strategy=no, and run_train.sh's BASE_ARGS hardcodes --val_size 0.1
# which we'd otherwise need to override anyway.
EXTRA_BASE="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True"
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

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_phase_a() {
    echo "=== v19-Core Phase A (Qwen2.5-14B): broad knowledge re-anchor (cutoff=8192, packing=on, lr=1e-5) ==="
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
    echo "=== v19-Core Phase B (Qwen2.5-14B): RMS+ATE+VSP+RCM catalog recovery (cutoff=16384, packing=off, lr=5e-6) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${B_BATCH} --grad-accum ${B_GA} \
        --cutoff 16384 --save-steps 400 --eval-steps 400 --packing false \
        --max-samples 70000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_B_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}  skip-eval: ${SKIP_EVAL}"
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
