#!/bin/bash

# v21 recal-32b: 32B-tuned Stage 4 Recalibrate touch-up of the Qwen3-MoE
# v21 chain. Default on-chain Stage 4 (invoked by
# run_sft_qwen3_30b_a3b_thinking_v21_chain.sh); chains off
# asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse. Held byte-
# identical to run_sft_qwen25_32b_v21_recal_32b.sh so the Qwen3-MoE
# outcome can be compared directly against the dense 32B sibling
# (athena-cti-sft-qwen25-32b-v21-recal-32b: Total 66.3 / Weighted 65.3).
#
# Why the 32B-tuned recipe is the canonical Stage 4 on Qwen3-MoE (and
# the 14B-recipe recalibrate is off-chain):
#   At dense 32B scale the 14B recal recipe (lr 1e-6, mix 0.25/0.40/
#   0.35, max-samples 2400) drifted VSP the wrong way after Stage 3
#   CSE drilling (78.9 -> 75.7) rather than recovering it the way it
#   did at 14B (72.9 -> 83.1). The 32B-tuned recipe (3x LR, Phase-B-
#   heavy mix 0.15/0.60/0.25, max-samples 3600) was developed to lift
#   the optimizer signal above the adamw_8bit noise floor at 32B+
#   scale. The Qwen3-MoE parent is peer-scale (30.5B total / 3.3B
#   active per token) and uses the same adamw_8bit + Liger + ZeRO-3
#   footprint, so the on-chain default at Stage 4 is this 32B-tuned
#   recipe. The 14B-recipe variant (run_sft_qwen3_30b_a3b_thinking_
#   v21_recalibrate.sh) remains available as an off-chain A/B against
#   this on-chain stage.
#
# Architectural notes (vs Qwen2.5-32B-Instruct):
#   - 30.5B total params, 3.3B active per token (128 experts, top-8).
#   - Full-SFT memory footprint under ZeRO-3 is comparable to dense 32B
#     (the optimiser shards ALL params, not just active), ~120 GB for
#     params+grads+optim before activations. Comfortably inside 8xB300.
#   - Pure-thinking post-training: the base ALWAYS emits a
#     <think>...</think> trace. Training data here has no <think> blocks
#     and run_train.sh now defaults --enable_thinking to True (matching
#     LlamaFactory's own default). The qwen3 reasoning template detects
#     the missing <think> in the response and injects <think>\n\n</think>
#     into the loss/response_ids (ReasoningTemplate.encode_oneturn, see
#     SFT/src/llamafactory/data/template.py:420-432). Net effect: the SFT
#     teaches the model to autonomously emit an empty <think>\n\n</think>
#     block (~6 tokens) followed directly by the answer for CTI prompts.
#     The thinking apparatus stays alive as a generation path -- on OOD
#     (non-CTI) prompts the base's real-thinking behaviour can still
#     resurface, unlike the --enable_thinking False training path which
#     never exposes the loss to think tokens and attenuates the reasoning
#     generation. The matching bench wrapper still uses the '-no-think'
#     alias suffix at serve time: it suppresses VLLMModel's '-thinking'
#     8192-token decode floor (per-task caps MCQ=128, RCM/RMS/TAA=256 then
#     apply correctly) and the chat_template_kwargs.enable_thinking=False
#     override is a belt-and-suspenders that prefills <think>\n\n</think>
#     in case a checkpoint drifts off the empty-thought pattern.
#
# Recipe (held byte-identical to run_sft_qwen25_32b_v21_recal_32b.sh):
#   - Datasets (mix) : ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn  (0.15)
#                      ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm         (0.60)
#                      ift_data_2026_05_18_v21_taa                            (0.25)
#   - 1 epoch, lr 3e-6, cutoff 16384, packing OFF
#   - Effective batch 8 on 8xB300 (per_device 1 x grad_accum 1 x 8 GPUs)
#   - save every 200 steps; intra-training eval DISABLED (same shard-count
#     vs eval-shard mismatch as the 32B recal; see body comment in
#     run_sft_qwen25_32b_v21_recal_32b.sh)
#   - --max-samples 3600 -> ~6000 interleaved rows at max(P)=0.60 ->
#     ~1500 optimiser steps (same step count as the 32B recal)
#   - --optim adamw_8bit, Liger kernel ON, mix_strategy interleave_under
#   - Template: qwen3 (native), --enable_thinking True (run_train.sh
#     default; matches LlamaFactory default; see header above for the
#     "learn empty <think>\n\n</think>" semantics)
#
# Hardware deltas vs 32B recal-32b script (recipe constant, infra tuned
# for B300; per the user direction "leave the SFT recipe the same"):
#   - OFFLOAD default OFF (was ON). 8xB300 = 288 GB HBM3e per GPU; the
#     ZeRO-3 shard footprint (~15 GB per rank for params+grads+optim at
#     30.5B total / 8 ranks) leaves >250 GB headroom for activations and
#     KV. CPU offload would burn host PCIe bandwidth for no memory win.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (kept; helps with
#     MoE routing churn even on B300).
#   - Flash-attn auto (resolves to FA2 on Blackwell with FA>=2.7 + cu128
#     wheel; falls back to sdpa if the wheel is missing -- log a warning).
#   - Gradient checkpointing left ON for parity with the 32B recipe;
#     can be disabled via --extra "--gradient_checkpointing False" for a
#     ~20-30% throughput win on B300 if HBM headroom is confirmed.
#
# Base checkpoint: asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse
# (HF; overridable via --base-model -- e.g. Qwen/Qwen3-30B-A3B-Thinking-2507
# for the standalone-off-bare-base diagnostic that pre-dated the Qwen3
# chain). Pushes to athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b
# under HF_USERNAME, matching the qwen25-32b-v21-recal-32b naming.
#
# Status: off-plan Stage 4 (v21_plan.txt §3 defines only Core/TAA/CSE);
# default on-chain Stage 4 on the Qwen3-MoE port (invoked by
# run_sft_qwen3_30b_a3b_thinking_v21_chain.sh). Diverges from the
# qwen25-32b chain layout, which keeps the 14B-recipe recalibrate on
# the chain for cross-architecture parity; the Qwen3-MoE chain ships
# the 32B-tuned recipe instead because the dense 32B port confirmed
# the 14B recipe fails VSP recovery at 32B+ scale under adamw_8bit
# (see header above and README-21.md §"Qwen3-30B-A3B-Thinking-2507
# MoE port").
#
# Estimated wall-time on 8xB300 (no offload, FA2 on Blackwell):
#   ~1.5-2 h for the 3-shard ~1500-step run. 8xB300's HBM bandwidth
#   (8 TB/s/GPU vs H200's 4.8 TB/s) gives a ~1.5x throughput edge over
#   8xH200 SXM at this shape; the MoE active-path (3.3B vs dense 32B)
#   shaves another factor on the forward FLOPs that ZeRO-3 all-gather
#   dominates anyway. CPU offload removal saves another ~20-30%.
#
# Usage:
#   ./run_sft_qwen3_30b_a3b_thinking_v21_recal_32b.sh
#       [--repo-id USER/NAME] [--base-model HF_REPO|LOCAL_DIR]
#       [--output-dir DIR] [--report-to wandb|none] [--max-samples N]
#       [--lr LR] [--probs P_A,P_B,P_TAA]
#       [--offload | --no-offload] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
BASE_MODEL=""
OUTPUT_DIR=""
REPORT_TO="wandb"
MAX_SAMPLES=3600
LR="3e-06"
PROBS="0.15,0.60,0.25"
DRY_RUN=0
OFFLOAD="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)      REPO_ID="$2";      shift 2 ;;
        --base-model)   BASE_MODEL="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --report-to)    REPORT_TO="$2";    shift 2 ;;
        --max-samples)  MAX_SAMPLES="$2";  shift 2 ;;
        --lr)           LR="$2";           shift 2 ;;
        --probs)        PROBS="$2";        shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        --offload)      OFFLOAD="on";      shift ;;
        --no-offload)   OFFLOAD="off";     shift ;;
        -h|--help) sed -n '3,86p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b"
fi

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse"

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
SAFE_MODEL="Qwen_Qwen3-30B-A3B-Thinking-2507"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full/v21_recal_32b_${TIMESTAMP}"

DS_PHASE_A="ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn"
DS_PHASE_B="ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm"
DS_TAA="ift_data_2026_05_18_v21_taa"
DATASETS="${DS_PHASE_A},${DS_PHASE_B},${DS_TAA}"

# Intra-training eval is DISABLED for this multi-shard touch-up. Same
# rationale as run_sft_qwen25_32b_v21_recal_32b.sh: LlamaFactory requires
# len(eval_dataset) == len(interleave_probs) when interleaving, and
# listing core_val once silently dedupes to 1 unique entry against 3
# probs ("a and p must have same size"). Both --do_eval False and
# --eval_strategy no are required to overcome TrainingArguments'
# __post_init__ auto-flip. Per-step train loss is still logged.

for ds in "${DS_PHASE_A}" "${DS_PHASE_B}" "${DS_TAA}"; do
    if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
        echo "[FAIL] v21 recal-32b dataset missing: SFT/data/${ds}.json" >&2
        echo "       These shards are reused verbatim from the v21 build" >&2
        echo "       (Phase A / Phase B / standalone TAA); rebuild via" >&2
        echo "       run_sft_qwen25_32b_v21_core.sh / _plus_taa.sh data" >&2
        echo "       preflights or copy from the Core training host." >&2
        exit 2
    fi
done

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

# B300 has 288 GB HBM3e per GPU; the ZeRO-3 weight+grad+optim shard for a
# 30.5B-total MoE comes out to ~15 GB/rank at 8 ranks, leaving >250 GB
# for activations + KV. CPU offload would only burn host PCIe bandwidth
# for no memory win. Default OFFLOAD=off on Blackwell-class hardware; the
# original 32B recal script defaults to on for the 8xH200 chain-host
# parity. Pass --offload to opt back into CPU offload (e.g. when running
# this script on the 8xH100 80GB diagnostic host).
if [[ "${OFFLOAD}" == "auto" ]]; then
    OFFLOAD="off"
fi
DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
[[ "${OFFLOAD}" == "off" ]] && DS_CONFIG="examples/deepspeed/ds_z3_config.json"

EFFECTIVE_GPUS=$(( GPU_COUNT > 0 ? GPU_COUNT : 1 ))
# eff_bs target 8 (per_device 1 x grad_accum 1 x 8 GPUs); identical to
# the 32B recal recipe. eff_bs is a RECIPE parameter and is held constant
# across architectures/hardware; only the DS config + offload toggle vary.
R_BATCH=1; R_GA=$(( 8 / (R_BATCH * EFFECTIVE_GPUS) )); [[ ${R_GA} -lt 1 ]] && R_GA=1

EXTRA_COMMON="--deepspeed ${DS_CONFIG} --save_total_limit 2 --save_only_model True --enable_liger_kernel True --optim adamw_8bit --do_eval False --eval_strategy no --val_size 0 --mix_strategy interleave_under --interleave_probs ${PROBS}"

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    [[ -z "${!var:-}" ]] && unset "${var}"
done
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export NPROC_PER_NODE="${GPU_COUNT}"
fi
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "[warn] expected 8 GPUs (8xB300 target host); detected ${GPU_COUNT}. Qwen3-MoE v21-recal-32b at cutoff=16384 packing=off is sized for GPU_COUNT=8; at <8 the ZeRO-3 weight shard grows per rank and the rank-0 CPU gather peak at save-time can OOM the host; effective batch will reflect detected count: eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS ))." >&2
fi

DRY_FLAG=(); [[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

run_qwen3_v21_recal_32b() {
    echo "=== v21 recal-32b (Qwen3-30B-A3B-Thinking-2507): 3-shard interleave touch-up off v21-cse (cutoff=16384, packing=off, lr=${LR}, eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )), max-samples=${MAX_SAMPLES}, probs=${PROBS}, enable_thinking=True) ==="
    bash "${SFT_DIR}/utils/run_train.sh" \
        --model "${BASE_MODEL}" \
        --dataset "${DATASETS}" --template qwen3 --finetuning full \
        --epochs 1 --lr "${LR}" --batch ${R_BATCH} --grad-accum ${R_GA} \
        --cutoff 16384 --save-steps 200 --eval-steps 200 --packing false \
        --max-samples "${MAX_SAMPLES}" --report-to "${REPORT_TO}" \
        --output-dir "${OUTPUT_DIR}" --push-to-hf "${REPO_ID}" \
        --extra "${EXTRA_COMMON}" "${DRY_FLAG[@]}"
}

echo "  gpus visible : ${GPU_COUNT}  nproc: ${NPROC_PER_NODE:-auto}  cpu offload: ${OFFLOAD}  ds: ${DS_CONFIG}"
echo "  batch math   : per_device=${R_BATCH} grad_accum=${R_GA} -> eff_bs=$(( R_BATCH * R_GA * EFFECTIVE_GPUS )) (target 8 on 8xB300)"
echo "  base model   : ${BASE_MODEL}  (template=qwen3; --enable_thinking True default -> model learns empty <think>\\n\\n</think> in response)"
echo "  datasets     : ${DATASETS}"
echo "  mix strategy : interleave_under  probs=${PROBS}  (Phase A / Phase B / TAA; Phase-B-heavy 32B recipe)"
echo "  max samples  : ${MAX_SAMPLES}/dataset -> ~6000 interleaved rows at max(P)=0.60 (~1500 steps; eval disabled)"
echo "  learning rate: ${LR}  (recal-32b recipe held constant; 3x the 14B recal's 1e-6)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  hf repo      : ${REPO_ID}"
echo "  alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

run_qwen3_v21_recal_32b
