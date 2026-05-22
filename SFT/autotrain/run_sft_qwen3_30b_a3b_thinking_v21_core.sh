#!/bin/bash

# v21-Core two-phase full-parameter SFT of Qwen/Qwen3-30B-A3B-Thinking-2507
# on the v21 core corpus (broad + axis shards). Stage 1 of the v21 chain
# ported to the Qwen3 MoE thinking-2507 architecture (30.5B total /
# 3.3B active per token); the resulting checkpoint is the base for the
# v21 TAA + CSE + Recalibrate chain at the Qwen3-MoE scale.
#
# Why a Qwen3-MoE v21 chain exists:
#   The Qwen2.5-32B v21 chain shipped at 65.8 Total (v21-cse) and 66.3
#   (v21-recal-32b). The Qwen3-30B-A3B-Thinking-2507 base provides
#   comparable param scale (30.5B vs 32B) with sparse compute (3.3B
#   active) and native thinking template. This chain runs the full
#   Core->TAA->CSE->Recalibrate stack to test whether the sparse
#   architecture absorbs the v21 catalog under the same recipe shape
#   that produced the 32B headline; the 32B-tuned recal-32b recipe is
#   then run off v21-cse as a parallel Stage-4 branch via
#   run_sft_qwen3_30b_a3b_thinking_v21_recal_32b.sh (matching the
#   qwen25-32b chain layout). README-21.md §"Scale-up to Qwen2.5-32B-
#   Instruct" applies verbatim to template/dataset/LR; only the base
#   + memory math change.
#
# Recipe parity with run_sft_qwen25_32b_v21_core.sh:
#   - Identical Phase A / Phase B cutoff, packing, lr, max-samples,
#     save/eval steps, and dataset names.
#   - Effective batches preserved: Phase A eff_bs=16, Phase B eff_bs=8.
#   - --optim adamw_8bit kept for parity with the 32B chain (validated
#     on MoE; bitsandbytes 8-bit adamw operates on flat optimizer state
#     and is indifferent to expert routing).
#   - Liger kernel on; --gc on (see below).
#
# Qwen3-MoE deltas vs Qwen2.5-32B v21-core (B300 / template / sparse):
#   - --template qwen3 (the reasoning template; was qwen).
#   - --enable_thinking True (run_train.sh default; matches LF default).
#     The reasoning template injects <think>\n\n</think> into the
#     loss/response_ids on every sample without a <think> block (i.e.
#     all our CTI rows). Net: the model learns to autonomously emit a
#     6-token empty thought + answer for CTI prompts; the thinking
#     apparatus is preserved for OOD prompts (see header in
#     run_sft_qwen3_30b_a3b_thinking_v21_recal_32b.sh for the full
#     mechanism). --enable_thinking False would prefill the empty think
#     into the prompt instead, never exposing the loss to think tokens
#     and attenuating the reasoning generation path -- not preferred.
#   - OFFLOAD default off (was auto, which resolved to off at GPU_COUNT
#     >= 4 on the 32B host). 8xB300 = 288 GB HBM3e per GPU; ZeRO-3
#     weight+grad+optim shard for a 30.5B-total MoE comes out to
#     ~15 GB/rank with adamw_8bit at 8 ranks, leaving >250 GB headroom
#     for activations + KV. CPU offload would only burn host PCIe.
#   - --gc kept on (matches 32B default). MoE's per-expert activation
#     spikes during dropless routing are awkward to budget without GC;
#     the ~20-25% throughput tax is the right trade-off for chain
#     stability. Caller can pass --gc off if HBM headroom on B300 +
#     FA2 are both confirmed and the throughput win matters.
#
# Phase shape:
#   Phase A -- broad knowledge re-anchor
#     - Datasets   : ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn,
#                    tulu_3_sft_mixture, alpaca_en_demo
#     - 1 epoch, lr 1e-5, cutoff 8192, packing on
#     - Effective batch 16 (per_device 1 x grad_accum 2 x 8 GPUs)
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
# Default push target: ${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core.
#
# Estimated wall-time (Qwen3-MoE 3.3B active vs dense 32B reduces forward
# FLOPs ~10x at the same shape, but ZeRO-3 all-gather is bandwidth-bound
# and the all-reduce traffic is on the FULL 30.5B param shard, so end-
# to-end speedup over 32B at the same hardware tier is closer to 1.5-2x):
#   8xB300 288GB SXM       : ~14-18 h (Phase A ~9-11 h, Phase B ~5-7 h).
#     No offload, FA2 on Blackwell, adamw_8bit + Liger + ZeRO-3.
#   8xH200 141GB SXM       : ~20-26 h. HBM3e + NVLink Gen4 still the
#     dominant constraint; sparse MoE forward win is partially eaten
#     by the heavier per-token routing overhead.
#   8xH100 80GB SXM        : ~26-32 h with --offload on (the 30.5B
#     weight shard ~ 15 GB/rank at 8 ranks fits without offload, but
#     Phase B cutoff=16384 packing=off needs the headroom).
#
# Usage:
#   ./run_sft_qwen3_30b_a3b_thinking_v21_core.sh
#       [--repo-id USER/NAME] [--phase-a-dir DIR] [--phase-b-dir DIR]
#       [--report-to wandb|none] [--phase a|b|ab]   # default: ab
#       [--offload | --no-offload]
#       [--gc on|off]      # default: on
#       [--skip-eval]      # disables in-training eval
#       [--resume]         # resume from latest ckpt in phase dir
#       [--dry-run]
#
# See run_sft_qwen25_32b_v21_core.sh for the --skip-eval / --resume
# rationale (in-training eval OOM at Phase A cutoff=8192 on 4xH100; not
# a concern on 8xB300 but the flag is preserved for parity).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
PHASE_A_DIR=""
PHASE_B_DIR=""
REPORT_TO="wandb"
PHASE="ab"
DRY_RUN=0
OFFLOAD="off"
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
        -h|--help) sed -n '3,95p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${PHASE}" in a|b|ab) ;; *) echo "--phase must be a|b|ab" >&2; exit 1 ;; esac
case "${GC_OVERRIDE}" in on|off) ;; *) echo "--gc must be on|off" >&2; exit 1 ;; esac

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core"
fi

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen3-30B-A3B-Thinking-2507"
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
            GPU_PROBE_SOURCE="nvidia-smi (torch probe returned 0)"
        fi
    fi
    GPU_COUNT="${GPU_COUNT:-0}"
fi
if [[ "${GPU_COUNT}" == "0" && ${DRY_RUN} -eq 0 ]]; then
    echo "[FAIL] GPU probe returned 0 (source: ${GPU_PROBE_SOURCE}). See run_sft_qwen25_32b_v21_core.sh for diagnostic steps." >&2
    echo "       Override: GPU_COUNT_OVERRIDE=N ./run_sft_qwen3_30b_a3b_thinking_v21_core.sh ..." >&2
    exit 3
fi

DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# Same batch sizing as the 32B recipe (bs=1, eff_bs=16 Phase A / 8 Phase B).
# Qwen3-MoE on B300 has memory for bs=2 Phase A, but holding eff_bs constant
# across chain ports keeps gradient noise + optimizer dynamics aligned with
# the 32B reference; bs vs grad-accum is interchangeable here.
A_BATCH=1; A_GA=$(( 16 / (A_BATCH * EFFECTIVE_GPUS) )); [[ ${A_GA} -lt 1 ]] && A_GA=1
B_BATCH=1; B_GA=$(( 8  / (B_BATCH * EFFECTIVE_GPUS) )); [[ ${B_GA} -lt 1 ]] && B_GA=1

case "${GC_OVERRIDE}" in
    on)  GC_FLAG="" ;;
    off) GC_FLAG="--disable_gradient_checkpointing True" ;;
esac

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
    echo "[warn] recipe sized for 8 GPUs (8xB300 target); detected ${GPU_COUNT}. Effective batch preserved via grad-accum auto-scaling: phase_A eff_bs=$(( A_BATCH * A_GA * EFFECTIVE_GPUS )) phase_B eff_bs=$(( B_BATCH * B_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )
RESUME_FLAG=(); [[ ${RESUME} -eq 1 ]] && RESUME_FLAG=( --resume )

run_phase_a() {
    echo "=== v21-Core Phase A (Qwen3-30B-A3B-Thinking-2507): broad knowledge re-anchor (cutoff=8192, packing=on, lr=1e-5, enable_thinking=True) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "Qwen/Qwen3-30B-A3B-Thinking-2507" \
        --dataset "${PHASE_A_DATASETS}" --template qwen3 --finetuning full \
        --epochs 1 --lr 1e-05 --batch ${A_BATCH} --grad-accum ${A_GA} \
        --cutoff 8192 --save-steps 500 --eval-steps 500 --packing true \
        --max-samples 240000 --report-to "${REPORT_TO}" \
        --output-dir "${PHASE_A_DIR}" \
        --extra "${EXTRA_COMMON}" \
        ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

run_phase_b() {
    echo "=== v21-Core Phase B (Qwen3-30B-A3B-Thinking-2507): RMS+ATE+VSP+RCM catalog recovery (cutoff=16384, packing=off, lr=5e-6, enable_thinking=True) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${PHASE_A_DIR}" \
        --dataset "${PHASE_B_DATASETS}" --template qwen3 --finetuning full \
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
