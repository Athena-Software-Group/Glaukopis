#!/bin/bash

# Launch full-parameter SFT of Llama-3.1-8B-Instruct on the consolidated v7
# dataset via LLaMA-Factory + DeepSpeed ZeRO-3. Trains on
# ift_data_2026_04_26_combined_v7 (180,533 rows: v5 broad SFT coverage +
# v7 RMS-only addendum, pre-merged into one file) plus alpaca_en_demo
# (instruction-following baseline). Total ~181.5k rows.
#
# Why this script exists (v6 -> v7 RMS recovery):
#   The v6 SFT (run_abaligned_sft_v6.sh) regressed athena-rms from the v0
#   baseline of 5.88% f1 to 0.00% f1. A v6 post-mortem identified three
#   structural bugs in the v6 RMS-addendum templates / launcher:
#     (1) Output truncation. v6 inlined the full {coa.description} per
#         mitigation in the Answer body. At N=4-5 the output reached
#         ~400-700 tokens. With the v6 launcher's --cutoff 2048 and the
#         input attack-pattern descriptions (~300-1000 tokens), ~80%+ of
#         training rows were right-truncated mid-explanation, never
#         reaching the terminal sentence.
#     (2) Wrong terminator. The AthenaBench RMS prompt mandates a final
#         line beginning with "Answer:" containing only the mitigation
#         IDs. v6 outputs ended with "Therefore, the recommended ..." and
#         never emitted an "Answer:" line; the trained model inherited
#         this format and emitted zero compliant responses.
#     (3) Cardinality coverage gap. v6 trained on N=3,4,5 only. The
#         benchmark distribution peaks at N=1 (39%) and N=2 (24%); the
#         trained model collapsed to emitting exactly one M-ID in 98.4%
#         of rows.
#
#   The v7 RMS-addendum template slate
#   (tmpl_gen/templates/04262026/Sophia-CTI-Templates-AthenaBench-abaligned-v7.txt)
#   addresses all three:
#     - RMS.3a..3h : variable-N at N=1..8 matching benchmark mass
#     - Per-mitigation clauses reduced to "{coa.mitre_id} ({coa.name})"
#       (no inline {coa.description}); est. output <600 chars at N=8.
#     - Every variable-N template (and RMS.6) terminates with a literal
#       "Answer: M####, M####, ..." final-line directive (requires the
#       multi-line Answer parser support in tmpl_docx2json.py and the
#       lazy-quantifier fix in to_alpaca.py).
#     - Instruction text brought verbatim into alignment with the
#       benchmark prompt ("Return exactly N mitigation IDs ...").
#
# What stays fixed vs run_abaligned_sft_v6.sh:
#   - Base model: meta-llama/Llama-3.1-8B-Instruct
#   - 3 epochs, cosine schedule, 5% warmup, bf16
#   - lr 1e-5
#   - DeepSpeed ZeRO-3 (no offload on >=2 GPUs)
#   - Effective batch 16, packing on, save_only_model=True
#
# What changes vs run_abaligned_sft_v6.sh:
#   - Dataset: ift_data_2026_04_26_combined_v7 (single pre-merged file
#     containing v5 broad coverage + v7 RMS addendum; was the v5 +
#     v6 RMS-addendum split).
#   - cutoff_len 4096 (was 2048). The v6 truncation diagnosis showed
#     ~80%+ of RMS rows were cut mid-explanation; doubling cutoff
#     fits the longest input + terse-justification output.
#   - per-device batch halved (2->1 on <4 GPUs, 4->2 on >=4 GPUs)
#     and grad_accum doubled to keep effective batch at 16, since
#     activation memory grows roughly linearly with packed sequence
#     length on FFN-dominant models like Llama-3.1-8B.
#   - eval_steps + save_steps = 200 (was 400). With cutoff_len=4096
#     packing roughly halves the packed-sequence count vs 2048,
#     yielding ~220 opt steps/epoch and ~660 total over 3 epochs.
#     save_steps=200 gives 3 intermediates (200/400/600) + final.
#   - Final merged model pushed to
#     hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v7
#
# Usage:
#   ./run_abaligned_sft_v7.sh [--repo-id USER/NAME] [--output-dir DIR]
#                             [--report-to wandb|none]
#                             [--epochs N] [--lr FLOAT]
#                             [--offload | --no-offload]
#                             [--dry-run] [--extra "..."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID=""
OUTPUT_DIR=""
REPORT_TO="wandb"
EXTRA_USER=""
EPOCHS="3"
LR="1e-05"
DRY_RUN=0
OFFLOAD="auto"    # auto | on | off

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)    REPO_ID="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        --report-to)  REPORT_TO="$2";   shift 2 ;;
        --epochs)     EPOCHS="$2";      shift 2 ;;
        --lr)         LR="$2";          shift 2 ;;
        --extra)      EXTRA_USER="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        --offload)    OFFLOAD="on";     shift ;;
        --no-offload) OFFLOAD="off";    shift ;;
        -h|--help) sed -n '3,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        set -a; source "${env_file}"; set +a
    fi
done

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v7"
fi

DATASET_NAME="ift_data_2026_04_26_combined_v7"
DATASET_FILE="${SFT_DIR}/data/${DATASET_NAME}.json"
if [[ ! -f "${DATASET_FILE}" ]]; then
    echo "[FAIL] training dataset not found: ${DATASET_FILE}" >&2
    echo "       The combined v7 file (~193 MB) is gitignored. Either" >&2
    echo "       regenerate locally with tmpl_gen (Section A of" >&2
    echo "       tmpl_gen/templates/04262026/Sophia-CTI-Templates-Combined-v7.txt" >&2
    echo "       documents the build pipeline) or transfer from another host:" >&2
    echo "         rsync -avP workstation:Glaukopis/SFT/data/$(basename "${DATASET_FILE}") \\" >&2
    echo "               ${SFT_DIR}/data/" >&2
    exit 2
fi

GPU_COUNT="$(python - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
GPU_COUNT="${GPU_COUNT:-0}"

if [[ "${OFFLOAD}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -lt 2 ]]; then
        OFFLOAD="on"
        echo "[info] detected ${GPU_COUNT} GPU(s); auto-enabling ZeRO-3 CPU offload."
        echo "       Pass --no-offload to force the on-GPU config (will OOM on <2 x 80GB)."
    else
        OFFLOAD="off"
    fi
fi

if [[ "${OFFLOAD}" == "on" ]]; then
    DS_CONFIG="examples/deepspeed/ds_z3_offload_config.json"
else
    DS_CONFIG="examples/deepspeed/ds_z3_config.json"
fi
if [[ ! -f "${SFT_DIR}/${DS_CONFIG}" ]]; then
    echo "[FAIL] deepspeed config missing: ${SFT_DIR}/${DS_CONFIG}" >&2
    exit 2
fi

# At cutoff_len=4096 (2x v6) activation memory rises by roughly the same
# factor on Llama-3.1-8B; halve per-device batch and double grad_accum
# vs v6 to keep effective batch at 16 without OOMing 80 GB GPUs.
if [[ "${GPU_COUNT}" -ge 4 ]]; then
    BATCH_DEFAULT="2"
    GRAD_ACCUM_DEFAULT="2"
else
    BATCH_DEFAULT="1"
    GRAD_ACCUM_DEFAULT="8"
fi
EFFECTIVE_BATCH=$(( BATCH_DEFAULT * GRAD_ACCUM_DEFAULT * (GPU_COUNT > 0 ? GPU_COUNT : 1) ))

EXTRA_DEFAULT="--deepspeed ${DS_CONFIG} --save_total_limit 10 --save_only_model True"

if [[ -n "${EXTRA_USER}" ]]; then
    EXTRA_ALL="${EXTRA_DEFAULT} ${EXTRA_USER}"
else
    EXTRA_ALL="${EXTRA_DEFAULT}"
fi

RUN_TRAIN_ARGS=(
    --model        "meta-llama/Llama-3.1-8B-Instruct"
    --dataset      "${DATASET_NAME},alpaca_en_demo"
    --template     "llama3"
    --finetuning   "full"
    --epochs       "${EPOCHS}"
    --lr           "${LR}"
    --batch        "${BATCH_DEFAULT}"
    --grad-accum   "${GRAD_ACCUM_DEFAULT}"
    --cutoff       "4096"
    --save-steps   "200"
    --eval-steps   "200"
    --packing      "true"
    --max-samples  "200000"
    --report-to    "${REPORT_TO}"
    --push-to-hf   "${REPO_ID}"
    --extra        "${EXTRA_ALL}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
    RUN_TRAIN_ARGS+=( --output-dir "${OUTPUT_DIR}" )
fi
if [[ ${DRY_RUN} -eq 1 ]]; then
    RUN_TRAIN_ARGS+=( --dry-run )
fi

export FORCE_TORCHRUN=1

for var in NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT RDZV_ID MIN_NNODES MAX_NNODES; do
    if [[ -z "${!var:-}" ]]; then
        unset "${var}"
    fi
done

echo "=== AthenaBench-aligned v7 (combined v5 broad coverage + v7 RMS addendum) full SFT ==="
echo "  env          : ${CONDA_DEFAULT_ENV:-<unset>}  (expected: llm-sft)"
echo "  dataset      : ${DATASET_FILE}"
echo "  hf repo      : ${REPO_ID}"
echo "  gpus visible : ${GPU_COUNT}"
echo "  per-gpu batch: ${BATCH_DEFAULT}  grad_accum: ${GRAD_ACCUM_DEFAULT}  (effective batch ~= ${EFFECTIVE_BATCH})"
echo "  epochs / lr  : ${EPOCHS} / ${LR}"
echo "  packing      : true  (cutoff_len=4096)"
echo "  eval / save  : every 200 steps"
echo "  deepspeed    : ${SFT_DIR}/${DS_CONFIG}"
echo "  cpu offload  : ${OFFLOAD}"
echo "  method       : full-parameter SFT (DeepSpeed ZeRO-3)"
echo "  launcher     : ${SFT_DIR}/utils/run_train.sh"
echo "  torchrun     : forced (FORCE_TORCHRUN=1)"
echo

exec bash "${SFT_DIR}/utils/run_train.sh" "${RUN_TRAIN_ARGS[@]}"
