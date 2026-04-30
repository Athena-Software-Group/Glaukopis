#!/bin/bash

# Retry-only HF push for the v8-large 2-phase SFT recipe (Phase B merged
# checkpoint). Used when the in-line push at the tail of run_train.sh
# fails or is skipped, e.g. the validate-yaml 400 BadRequest caused by
# LLaMA-Factory writing the local Phase A path into README.md's
# 'base_model:' YAML field. SFT/upload_to_hf.py now sanitizes that
# field; this script is the small wrapper that picks the right Phase B
# dir + repo + base-model triple per supported model size.
#
# Supported sizes:
#   qwen25-14b -> Qwen/Qwen2.5-14B-Instruct -> asg-ai/athena-cti-sft-qwen25-14b-abaligned-v8
#   qwen25-32b -> Qwen/Qwen2.5-32B-Instruct -> asg-ai/athena-cti-sft-qwen25-32b-abaligned-v8
#
# Phase B dir auto-discovery: picks the most recent v8_phase_b_* under
#   SFT/saves/<safe-model>/full/  (lexicographic sort = chronological
#   because the timestamp is YYYY-MM-DD-HH-MM-SS).  Override with
#   --phase-b-dir DIR to target a specific run.
#
# Usage:
#   ./push_v8_phase_b.sh [--size qwen25-14b|qwen25-32b]
#                        [--phase-b-dir DIR]
#                        [--repo-id USER/NAME]
#                        [--base-model HF_ID]
#                        [--public]
#                        [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIZE="qwen25-14b"
PHASE_B_DIR=""
REPO_ID=""
BASE_MODEL=""
PUBLIC=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)         SIZE="$2";          shift 2 ;;
        --phase-b-dir)  PHASE_B_DIR="$2";   shift 2 ;;
        --repo-id)      REPO_ID="$2";       shift 2 ;;
        --base-model)   BASE_MODEL="$2";    shift 2 ;;
        --public)       PUBLIC=1;           shift ;;
        --dry-run)      DRY_RUN=1;          shift ;;
        -h|--help) sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for env_file in "${SFT_DIR}/.env" "${SFT_DIR}/.env.local" "${SCRIPT_DIR}/.env"; do
    [[ -f "${env_file}" ]] && { set -a; source "${env_file}"; set +a; }
done

case "${SIZE}" in
    qwen25-14b)
        SAFE_MODEL="Qwen_Qwen2.5-14B-Instruct"
        DEFAULT_BASE="Qwen/Qwen2.5-14B-Instruct"
        DEFAULT_REPO_NAME="athena-cti-sft-qwen25-14b-abaligned-v8"
        ;;
    qwen25-32b)
        SAFE_MODEL="Qwen_Qwen2.5-32B-Instruct"
        DEFAULT_BASE="Qwen/Qwen2.5-32B-Instruct"
        DEFAULT_REPO_NAME="athena-cti-sft-qwen25-32b-abaligned-v8"
        ;;
    *)
        echo "[FAIL] --size must be qwen25-14b or qwen25-32b (got: ${SIZE})" >&2
        exit 1 ;;
esac

[[ -z "${BASE_MODEL}" ]] && BASE_MODEL="${DEFAULT_BASE}"

if [[ -z "${REPO_ID}" ]]; then
    : "${HF_USERNAME:?Set HF_USERNAME in SFT/.env (or pass --repo-id USER/NAME)}"
    REPO_ID="${HF_USERNAME}/${DEFAULT_REPO_NAME}"
fi

if [[ -z "${PHASE_B_DIR}" ]]; then
    SAVES_DIR="${SFT_DIR}/saves/${SAFE_MODEL}/full"
    if [[ ! -d "${SAVES_DIR}" ]]; then
        echo "[FAIL] no saves dir: ${SAVES_DIR}" >&2
        echo "       train Phase B first via run_abaligned_sft_${SIZE//-/_}_v8.sh" >&2
        exit 2
    fi
    # Lexicographic sort matches chronological because of the YYYY-MM-DD-HH-MM-SS suffix.
    PHASE_B_DIR="$(ls -1d "${SAVES_DIR}"/v8_phase_b_* 2>/dev/null | sort | tail -1 || true)"
    if [[ -z "${PHASE_B_DIR}" || ! -d "${PHASE_B_DIR}" ]]; then
        echo "[FAIL] no v8_phase_b_* dir found under ${SAVES_DIR}" >&2
        exit 2
    fi
fi

if [[ ! -f "${PHASE_B_DIR}/config.json" ]]; then
    echo "[FAIL] ${PHASE_B_DIR} doesn't look like a merged HF model dir (no config.json)" >&2
    exit 2
fi

PUBLIC_FLAG=()
[[ ${PUBLIC} -eq 1 ]] && PUBLIC_FLAG=( --public )

DRY_FLAG=()
[[ ${DRY_RUN} -eq 1 ]] && DRY_FLAG=( --dry-run )

echo "=== HF push (v8 Phase B retry) ==="
echo "  size      : ${SIZE}"
echo "  phase-b   : ${PHASE_B_DIR}"
echo "  repo      : ${REPO_ID} ($([[ ${PUBLIC} -eq 1 ]] && echo public || echo private))"
echo "  base-model: ${BASE_MODEL}  (-> README.md base_model:)"
echo "  dry-run   : ${DRY_RUN}"
echo

python "${SFT_DIR}/upload_to_hf.py" \
    --merged-dir "${PHASE_B_DIR}" \
    --repo-id "${REPO_ID}" \
    --readme-base-model "${BASE_MODEL}" \
    "${PUBLIC_FLAG[@]}" "${DRY_FLAG[@]}"
