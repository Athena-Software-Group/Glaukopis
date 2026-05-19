#!/bin/bash

# v21-Core two-phase full-parameter SFT of Llama-3.1-8B-Instruct on the
# v21 core corpus (broad + axis shards). Stage 1 of the v21 chain
# applied to the 8B architecture (tmpl_gen/templates/05182026/v21_plan.txt
# §7.5); the resulting checkpoint is the base for the v21 TAA + CSE +
# Recalibrate chain on Llama-3.1-8B.
#
# Why a Llama-3.1-8B v21 chain exists:
#   The v21 chain on Qwen2.5-14B shipped at 62.3 Total (v21-recalibrate),
#   above v18.1-core (58.9) and v21-core (60.8). v21 also exposed
#   per-axis non-determinism in the data-build layer (v21_plan.txt §7.4)
#   that is invariant to model architecture. Re-running the same recipe
#   on Llama-3.1-8B-Instruct tests whether the Core->TAA->CSE->Recalibrate
#   chain shape (and the VSP erosion -> Recalibrate recovery pattern)
#   generalises off the Qwen2.5 family, or is Qwen-specific.
#
# Recipe parity with run_sft_qwen25_14b_v21_core.sh:
#   - Identical Phase A / Phase B geometry (cutoff, packing, lr,
#     effective batch, max-samples, save/eval steps).
#   - Identical dataset names; the v21 shards are template-baked and
#     architecture-independent.
#   - Only the base model, LlamaFactory --template (qwen -> llama3),
#     SAFE_MODEL path component, and HF push targets change.
#
# Phase shape (identical to v21-Core / v18.1-Core):
#   Phase A -- broad knowledge re-anchor
#     - Datasets   : ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn,
#                    tulu_3_sft_mixture, alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 8192, packing on
#     - Effective batch 16
#     - --max-samples 240000
#
#   Phase B -- AthenaBench catalog recovery
#     - Datasets   : ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm
#     - 1 epoch, lr 5e-6, cutoff 16384, packing OFF
#     - Effective batch 8 (cutoff doubled => half the effective batch)
#     - eval/save every 400 steps
#     - --model points at Phase A's output dir
#
# Only Phase B's final merged model is pushed to HF.
# Default push target: ${HF_USERNAME}/athena-cti-sft-llama31-8b-v21-core.
#
# Estimated wall-time (8B is ~1.75x lighter than Qwen2.5-14B; weight,
# activation, and optimizer-state footprints all scale roughly linearly):
#   8xH100 80GB SXM        : ~7-9 h (Phase A ~4-5 h, Phase B ~3-4 h).
#     Default --gc auto path: GC disabled on GPU_COUNT==8. 8B at
#     cutoff=8192 packing=on bs=2 fits comfortably with ~40-50GB
#     headroom/rank under ZeRO-3.
#   8xRTX PRO 6000 96GB    : ~11-15 h with --gc auto. 8B's lighter
#     activation footprint typically clears the OOM mode the 14B
#     recipe hit on this hardware, so --gc on is usually NOT required
#     (pass it only if Phase A step 0 OOMs). ZeRO-3 all-gather /
#     reduce-scatter still pays the PCIe Gen5 (~64 GB/s) vs NVLink
#     (~900 GB/s) ~1.3-1.5x communication overhead.
#   4xH100 80GB SXM        : ~14-18 h. GPU-count auto-detect halves
#     micro-batches per optimizer step (A_GA 1->2, B_GA 1->2) so
#     effective batch is preserved; gradient checkpointing is
#     auto-enabled for <8 GPUs to fit Phase B's cutoff=16384
#     packing=off pass within 80GB once the ZeRO-3 weight shard
#     doubles.
#
# Usage:
#   ./run_sft_llama31_8b_v21_core.sh [--repo-id USER/NAME]
#                                     [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                     [--report-to wandb|none]
#                                     [--phase a|b|ab]   # default: ab
#                                     [--offload | --no-offload]
#                                     [--gc auto|on|off] # default: auto
#                                     [--skip-eval]      # disables in-training eval
#                                     [--resume]         # resume from latest ckpt in phase dir
#                                     [--dry-run]
#
# --gc / --skip-eval / --resume semantics mirror the Qwen 14B v21 Core
# launcher; see that script's header (run_sft_qwen25_14b_v21_core.sh)
# for the full rationale on the GPU_COUNT>=8 GC-disable default, the
# in-training eval OOM workaround, and the resume-from-checkpoint path.

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
RESUME=0
GC_OVERRIDE="auto"

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
        --gc)           GC_OVERRIDE="$2";  shift 2 ;;
        --skip-eval)    SKIP_EVAL=1;       shift ;;
        --resume)       RESUME=1;          shift ;;
        -h|--help) sed -n '3,76p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|ab) ;; *) echo "--phase must be a|b|ab" >&2; exit 1 ;; esac
case "${GC_OVERRIDE}" in auto|on|off) ;; *) echo "--gc must be auto|on|off" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-v21-core"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="meta-llama_Llama-3.1-8B-Instruct"
[[ -z "${PHASE_A_DIR}" ]] && PHASE_A_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_core_phase_a_${TIMESTAMP}"
[[ -z "${PHASE_B_DIR}" ]] && PHASE_B_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_core_phase_b_${TIMESTAMP}"

PHASE_A_DATASETS="ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn,tulu_3_sft_mixture,alpaca_en_demo"
PHASE_B_DATASETS="ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm"
VAL_NAME="ift_data_2026_05_18_v21_core_val"

for ds in ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn \
          ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm \
          "${VAL_NAME}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21-core dataset missing: SFT/data/${ds}.json" >&2
        echo "       Build via:" >&2
        echo "         bash tmpl_gen/data_generation/make_dataset.sh \\" >&2
        echo "           tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21.txt \\" >&2
        echo "           _v21_core_build/triples \\" >&2
        echo "           ${SFT_DIR}/data/ift_data_2026_05_18_v21_core.raw.json \\" >&2
        echo "           10 1500" >&2
        echo "         bash _v21_core_build/watcher.sh   # all phases" >&2
        exit 2
    fi
done

if [[ -n "${GPU_COUNT_OVERRIDE:-}" ]]; then
    GPU_COUNT="${GPU_COUNT_OVERRIDE}"
    GPU_PROBE_SOURCE="override"
else
    GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
    GPU_PROBE_SOURCE="torch"
    if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" == "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        NVIDIA_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ -n "${NVIDIA_COUNT}" && "${NVIDIA_COUNT}" != "0" ]]; then
            GPU_COUNT="${NVIDIA_COUNT}"
            GPU_PROBE_SOURCE="nvidia-smi (torch probe returned 0 -- check 'python -c \"import torch; torch.cuda.is_available()\"' in this shell)"
        fi
    fi
    GPU_COUNT="${GPU_COUNT:-0}"
fi
# Fail-fast guard mirrors run_sft_qwen25_14b_v21_core.sh; see that
# launcher's header for the nproc_per_node=0 / torchrun AssertionError
# rationale and the GPU_COUNT_OVERRIDE escape hatch.
if [[ "${GPU_COUNT}" == "0" && ${DRY_RUN} -eq 0 ]]; then
    echo "[FAIL] GPU probe returned 0 (source: ${GPU_PROBE_SOURCE})." >&2
    echo "  Verify in this shell:" >&2
    echo "    python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'" >&2
    echo "    nvidia-smi -L" >&2
    echo "    echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    echo "  Then either fix the env (conda activate llm-sft; unset CUDA_VISIBLE_DEVICES)" >&2
    echo "  or re-run with an explicit override:" >&2
    echo "    GPU_COUNT_OVERRIDE=4 ./run_sft_llama31_8b_v21_core.sh ..." >&2
    exit 3
fi

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 4 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# Phase A runs cutoff=8192 with packing on; 8B fits batch=2 trivially
# on >=40GB GPUs under ZeRO-3 (no offload). Phase B stays at batch=1
# because cutoff=16384 with packing off is memory-tight even at 8B.
A_BATCH=2; A_GA=$(( 16 / (A_BATCH * EFFECTIVE_GPUS) )); [[ ${A_GA} -lt 1 ]] && A_GA=1
B_BATCH=1; B_GA=$(( 8  / (B_BATCH * EFFECTIVE_GPUS) )); [[ ${B_GA} -lt 1 ]] && B_GA=1

# v14.1 hot-fix (--disable_gradient_checkpointing True) was originally
# sized for 8xH100 80GB SXM running Qwen2.5-14B where ZeRO-3 shards each
# parameter across 8 ranks. On the 8B architecture the activation
# footprint is materially lighter, so GC-off is comfortable on both
# 8xH100 80GB SXM and 8xRTX PRO 6000 96GB at the default Phase B
# geometry. The same auto policy is kept for parity: GC disabled when
# GPU_COUNT>=8, GC enabled when <8. --gc on remains the escape hatch
# if a future hardware/library combination tightens the per-rank budget.
case "${GC_OVERRIDE}" in
    on)   GC_FLAG="" ;;
    off)  GC_FLAG="--disable_gradient_checkpointing True" ;;
    auto) GC_FLAG="--disable_gradient_checkpointing True"
          [[ "${GPU_COUNT}" -lt 8 ]] && GC_FLAG=""
          ;;
esac

EXTRA_BASE="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True ${GC_FLAG}"
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
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] recipe sized for 8 GPUs (8xH100); detected ${GPU_COUNT}. Effective batch preserved via grad-accum auto-scaling: phase_A eff_bs=$(( A_BATCH * A_GA * EFFECTIVE_GPUS )) phase_B eff_bs=$(( B_BATCH * B_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )
RESUME_FLAG=(); [[ ${RESUME} -eq 1 ]] && RESUME_FLAG=( --resume )

run_phase_a() {
    echo "=== v21-Core Phase A (Llama-3.1-8B): broad knowledge re-anchor (cutoff=8192, packing=on, lr=1e-5) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template llama3 --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
        --max-samples 240000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

run_phase_b() {
    echo "=== v21-Core Phase B (Llama-3.1-8B): RMS+ATE+VSP+RCM catalog recovery (cutoff=16384, packing=off, lr=5e-6) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template llama3 --finetuning full \
        --epochs 1 --lr 5e-06 --batch ${B_BATCH} --grad-accum ${B_GA} \
        --cutoff 16384 --save-steps 400 --eval-steps 400 --packing false \
        --max-samples 70000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_B_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

echo "  gpus visible : ${GPU_COUNT}  cpu offload: ${OFFLOAD}"
echo "  phase A dir  : ${PHASE_A_DIR}"
echo "  phase B dir  : ${PHASE_B_DIR}"
echo "  hf repo      : ${REPO_ID}  (only Phase B is pushed)"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  grad-ckpt    : $([[ -z "${GC_FLAG}" ]] && echo on || echo off)  (--gc ${GC_OVERRIDE})"
echo "  skip-eval    : $([[ ${SKIP_EVAL} -eq 1 ]] && echo on || echo off)"
echo "  resume       : $([[ ${RESUME} -eq 1 ]] && echo on || echo off)"
echo

case "${PHASE}" in
    a)  run_phase_a ;;
    b)  run_phase_b ;;
    ab) run_phase_a; run_phase_b ;;
esac
