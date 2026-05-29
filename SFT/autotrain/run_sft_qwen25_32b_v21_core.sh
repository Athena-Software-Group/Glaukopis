#!/bin/bash

# v21-Core two-phase full-parameter SFT of Qwen2.5-32B-Instruct on the
# v21 core corpus (broad + axis shards). Stage 1 of the v21 chain applied
# to the dense 32B Qwen2.5 architecture (tmpl_gen/templates/05182026/
# README-21.md §"Scale-up to Qwen2.5-32B-Instruct"); the resulting
# checkpoint is the base for the v21 TAA + CSE + Recalibrate chain at
# the 32B scale.
#
# Why a Qwen2.5-32B v21 chain exists:
#   The Qwen2.5-14B v21 chain shipped at 62.3 Total (v21-recalibrate),
#   above v18.1-core (58.9) and v21-core (60.8). Qwen2.5-32B-Instruct
#   baseline benchmarked at 59.0 avg with the same AthenaBench collapse
#   signature the v21 recipe was originally designed to repair on the
#   14B base (CKT high, ATE/RMS/TAA suppressed). Porting the v21
#   Core->TAA->CSE->Recalibrate chain to the larger dense base tests
#   whether the repair scales without LR / shape retuning; v8 32B
#   (Phase A/B at 1e-5 / 5e-6) and v11 32B (single-pass at 1e-5) have
#   already validated these LRs on the 32B architecture under
#   adamw_8bit + Liger so the v21 hyperparameters carry verbatim.
#
# Recipe parity with run_sft_qwen25_14b_v21_core.sh:
#   - Identical Phase A / Phase B cutoff, packing, lr, max-samples,
#     save/eval steps, and dataset names (v21 shards are template-baked
#     and architecture-independent).
#   - 32B deltas (memory only, no recipe change):
#       * Phase A per_device_batch_size 2 -> 1 (32B does not fit at
#         bs=2 cutoff=8192 packing=on on H100 80GB); A_GA scales 1 -> 2
#         on 8 GPUs so the effective batch stays at 16.
#       * --optim adamw_8bit added to EXTRA flags (carries v8 32B and
#         v11 32B precedent; ~12 GB/rank saved on optimizer state vs
#         the default adamw_torch -- mandatory at 32B ZeRO-3 no-offload).
#       * --gc default flipped from "auto" to "on" (gradient
#         checkpointing always enabled). The 14B v21 GC-off-on-8xH100
#         path leaves no headroom for the 32B o_proj backward temp
#         buffer at Phase B cutoff=16384; the throughput tax (~20-25%)
#         is preferable to recurring OOMs at the chain mid-point.
#
# Phase shape (identical to 14B v21-Core; only batch sizing changes):
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
# Default push target: ${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-core.
#
# Estimated wall-time (32B is ~2.3x heavier than Qwen2.5-14B at the same
# cutoff; throughput scales sub-linearly because the ZeRO-3 all-gather is
# bandwidth-bound and adamw_8bit cuts the optimizer-state traffic ~4x):
#   8xH100 80GB SXM        : ~26-30 h with --gc on default (Phase A
#     ~16-18 h, Phase B ~10-12 h). adamw_8bit + Liger + ZeRO-3 no-offload
#     fits in ~72 GB/rank at Phase A (bs=1 cutoff=8192 packing=on) and
#     ~76 GB/rank at Phase B (bs=1 cutoff=16384 packing=off). Pass
#     --no-offload explicitly if --offload auto resolved to "on".
#   8xRTX PRO 6000 96GB    : ~36-42 h with --gc on default. ZeRO-3
#     all-gather / reduce-scatter pays the PCIe Gen5 (~64 GB/s) vs
#     NVLink (~900 GB/s) penalty (~1.3-1.5x communication tax on top
#     of the FLOPs ratio).
#   4xH100 80GB SXM        : NOT RECOMMENDED -- Phase B at cutoff=16384
#     packing=off OOMs even with offload on (per-rank ZeRO-3 weight
#     shard ~16 GB plus activation footprint exceeds 80 GB once shards
#     halve). Use 8xH100 SXM or 8xRTX PRO 6000 96GB.
#
# Usage:
#   ./run_sft_qwen25_32b_v21_core.sh [--repo-id USER/NAME]
#                                      [--phase-a-dir DIR] [--phase-b-dir DIR]
#                                      [--report-to wandb|none]
#                                      [--phase a|b|ab]   # default: ab
#                                      [--offload | --no-offload]
#                                      [--gc auto|on|off] # default: on
#                                      [--skip-eval]      # disables in-training eval
#                                      [--resume]         # resume from latest ckpt in phase dir
#                                      [--dry-run]
#
# --gc controls gradient checkpointing. on (default for 32B) keeps GC
# enabled at all GPU counts; the activation-recompute tax (~20-25%) is
# absorbed in exchange for leaving the o_proj backward temp buffer
# headroom intact at Phase B. Pass --gc off only on hardware with
# >=120 GB HBM/rank when FA2 is confirmed loaded. --gc auto retains the
# 14B v21 behaviour (off on GPU_COUNT>=8) for callers who explicitly
# want the riskier-but-faster path on 8xH100 SXM.
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
GC_OVERRIDE="on"

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
        -h|--help) sed -n '3,91p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen25-32b-v21-core"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen2.5-32B-Instruct"
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
    echo "    GPU_COUNT_OVERRIDE=4 ./run_sft_qwen25_32b_v21_core.sh ..." >&2
    exit 3
fi

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 4 ]]; then OFFLOAD="on"; else OFFLOAD="off"; fi
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# Phase A runs cutoff=8192 with packing on; at 32B under ZeRO-3 +
# adamw_8bit + Liger + GC the per-rank footprint maxes ~72 GB at bs=1
# on 8xH100 SXM (no headroom for bs=2 like the 14B recipe). A_GA scales
# to keep the effective batch at 16. Phase B stays at bs=1 because
# cutoff=16384 with packing off is the memory-tight pass.
A_BATCH=1; A_GA=$(( 16 / (A_BATCH * EFFECTIVE_GPUS) )); [[ ${A_GA} -lt 1 ]] && A_GA=1
B_BATCH=1; B_GA=$(( 8  / (B_BATCH * EFFECTIVE_GPUS) )); [[ ${B_GA} -lt 1 ]] && B_GA=1

# 32B default keeps gradient checkpointing ON regardless of GPU count
# (--gc on); the 14B v21 "auto disable GC on 8xH100" path leaves no
# margin for the o_proj backward temp buffer at the doubled weight
# footprint. --gc auto reproduces the 14B branch (off when GPU_COUNT>=8)
# for callers who want to probe the riskier path on hardware with FA2
# confirmed loaded.
case "${GC_OVERRIDE}" in
    on)   GC_FLAG="" ;;
    off)  GC_FLAG="--disable_gradient_checkpointing True" ;;
    auto) GC_FLAG="--disable_gradient_checkpointing True"
          [[ "${GPU_COUNT}" -lt 8 ]] && GC_FLAG=""
          ;;
esac

# --optim adamw_8bit carries v8 32B / v11 32B precedent; saves ~12 GB/rank
# vs the default adamw_torch optimizer state at 32B, which is what keeps
# the no-offload ZeRO-3 path viable on 8xH100 80GB SXM.
EXTRA_BASE="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit ${GC_FLAG}"
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
    echo "=== v21-Core Phase A (Qwen2.5-32B): broad knowledge re-anchor (cutoff=8192, packing=on, lr=1e-5) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen2.5-32B-Instruct" \
        --dataset "${PHASE_A_DATASETS}" --template qwen --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
        --max-samples 240000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

run_phase_b() {
    echo "=== v21-Core Phase B (Qwen2.5-32B): RMS+ATE+VSP+RCM catalog recovery (cutoff=16384, packing=off, lr=5e-6) ==="
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
