#!/bin/bash

# v21-Core two-phase full-parameter SFT of Qwen2.5-14B-Instruct on the
# v21 core corpus (broad + axis shards). Stage 1 of the v21 Core-only
# redo (tmpl_gen/templates/05182026/v21_plan.txt); the resulting
# checkpoint is the new base for the unchanged v18 TAA + CSE chained
# stages, which are renamed on HF rather than re-trained.
#
# Why v21-Core exists (v18 Core regression):
#   The v18 chain Stage 1 (Core) regressed against three historical
#   AthenaBench peaks:
#     CKT 62.6 vs v8small 77.6   (-15.0 pp)
#     RMS 55.6 vs v9_rms  65.8   (-10.2 pp)
#     VSP 76.8 vs v10     86.7   ( -9.9 pp)
#   Diagnosis: the v18 Core MCQ shard was 61% AB.MCQ.EXT.* KB
#   flashcards (not the eval scenario shape), AB.RMS.{4,5} were
#   fragmented across 10 paraphrases each at Count: 50, and
#   AB.VSP.{1..4} + V.CPE.{1..4} ran without explicit Counts so the
#   build overshot v10's 12K shape by 2.25x. v21 reverts each axis
#   to the historical-peak recipe (MCQ -> v8small, RMS -> v7/v9_rms,
#   VSP -> v10) without changing the v18-Core training shape.
#
# Phase shape (identical to v18-Core; only datasets and HF target change):
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
# Default push target: ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v21-core.
#
# Estimated wall-time:
#   8xH100 80GB SXM        : ~13 h (Phase A 8 h, Phase B 5 h). Default
#     --gc auto path: GC disabled on GPU_COUNT==8, activations fit in
#     ~78GB/rank with 2GB headroom.
#   8xRTX PRO 6000 96GB    : ~21-25 h with --gc on (REQUIRED on this
#     hardware; observed OOM at Phase A step 0 with the default GC-off
#     path -- activations at cutoff=8192 packing=on bs=2 consume the full
#     94GB rank capacity in pytorch>=2.5 / transformers>=4.50, leaving no
#     margin for the o_proj backward temp buffer. --gc on adds ~25%
#     wall-clock vs the H100 SXM path via activation recompute but
#     preserves identical gradients / optimizer steps / effective batch,
#     well inside the v21 §5.1 ±1.5pp reproducibility band.) ZeRO-3
#     all-gather/reduce-scatter runs over PCIe Gen5 (~64 GB/s) instead
#     of NVLink (~900 GB/s) which adds another ~1.3-1.5x communication
#     overhead on top.
#   4xH100 80GB SXM        : ~26 h (Phase A 16 h, Phase B 10 h). GPU-count
#     auto-detect halves micro-batches per optimizer step (A_GA 2->4,
#     B_GA 1->2) so effective batch is preserved; gradient checkpointing
#     is re-enabled for <8 GPUs to keep Phase B's cutoff=16384 packing=off
#     pass within 80GB once the ZeRO-3 weight shard doubles.
#
# Usage:
#   ./run_sft_qwen25_14b_v21_core.sh [--repo-id USER/NAME]
#                                      [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                      [--report-to wandb|none]
#                                      [--phase a|b|ab]   # default: ab
#                                      [--offload | --no-offload]
#                                      [--gc auto|on|off] # default: auto
#                                      [--skip-eval]      # disables in-training eval
#                                      [--resume]         # resume from latest ckpt in phase dir
#                                      [--dry-run]
#
# --gc controls gradient checkpointing. auto (default) disables GC for
# GPU_COUNT>=8 and enables for <8, matching the original v18.1 8xH100 SXM
# tuning. Pass --gc on to force-enable (REQUIRED on 8xRTX PRO 6000 96GB
# to avoid the Phase A step-0 OOM; recipe-equivalent, just adds ~25%
# wall-clock via activation recompute). --gc off is the escape hatch for
# hardware with even more headroom than 80GB H100 SXM if/when that
# materializes -- no current use case.
#
# --skip-eval is the targeted fix for in-training eval OOM (observed on
# 4xH100 at Phase A step 500 cutoff=8192: ForCausalLMLoss materialises a
# [batch, 8192, 152064] fp32 logits tensor for cross-entropy because
# Liger's fused linear-cross-entropy is bypassed by Trainer.prediction_step --
# ~9.3 GiB on top of the resident ZeRO-3 weight + activation footprint
# tips the 80GB budget over). When --skip-eval is passed we replace the
# eval block with --eval_strategy no, which sets do_eval=False and
# prevents _maybe_log_save_evaluate from ever entering the prediction
# loop. Eval-time arguments (eval_dataset/val_size) are dropped because
# they're irrelevant when eval_strategy=no. Bench evaluation (AthenaBench
# etc.) runs out-of-band via SFT/eval/utils/serve_and_bench_v21_*.sh and
# is unaffected.
#
# --resume keeps the existing --phase-{a,b}-dir and asks Trainer to pick
# up from the newest checkpoint-N subdir (optimizer/scheduler state, RNG,
# dataloader position, global_step all restored). Useful for OOM-recovery
# without burning the wall-clock spent before the crash.

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
        -h|--help) sed -n '3,79p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-14b-v21-core"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
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
# Fail-fast guard: nproc_per_node=0 produces a cryptic torchrun AssertionError
# 30 seconds into the run. Better to refuse to launch and surface the probe
# diagnostic immediately. Override with GPU_COUNT_OVERRIDE=N when running on
# a CPU-only host for inspection (e.g. --dry-run).
if [[ "${GPU_COUNT}" == "0" && ${DRY_RUN} -eq 0 ]]; then
    echo "[FAIL] GPU probe returned 0 (source: ${GPU_PROBE_SOURCE})." >&2
    echo "  Verify in this shell:" >&2
    echo "    python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'" >&2
    echo "    nvidia-smi -L" >&2
    echo "    echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    echo "  Then either fix the env (conda activate llm-sft; unset CUDA_VISIBLE_DEVICES)" >&2
    echo "  or re-run with an explicit override:" >&2
    echo "    GPU_COUNT_OVERRIDE=4 ./run_sft_qwen25_14b_v21_core.sh ..." >&2
    exit 3
fi

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

# v14.1 hot-fix (--disable_gradient_checkpointing True) was sized for 8xH100
# 80GB SXM where ZeRO-3 shards each parameter across 8 ranks; on <8 GPUs the
# per-rank weight + activation footprint OOMs (Phase B at cutoff=16384
# packing=off is the danger case), so re-enable gradient checkpointing
# (LlamaFactory default) for those configurations. --gc on additionally
# forces GC back on regardless of GPU count -- required on 8xRTX PRO 6000
# 96GB (Verda) where the 8xH100-sized GC-off recipe OOMs at Phase A step 0
# despite the +16GB/rank nominal headroom; see the §"Estimated wall-time"
# header block for the full diagnosis.
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
    echo "=== v21-Core Phase A (Qwen2.5-14B): broad knowledge re-anchor (cutoff=8192, packing=on, lr=1e-5) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen2.5-14B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
        --max-samples 240000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

run_phase_b() {
    echo "=== v21-Core Phase B (Qwen2.5-14B): RMS+ATE+VSP+RCM catalog recovery (cutoff=16384, packing=off, lr=5e-6) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen --finetuning full \
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
