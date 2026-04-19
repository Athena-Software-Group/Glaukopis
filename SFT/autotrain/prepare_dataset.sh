#!/bin/bash

# Convert SFT/data/ift_data.json (instruction/input/output format) into an
# AutoTrain-ready JSONL with a single pre-templated `text` column, then
# create/update an HF dataset repo and upload it so AutoTrain can consume
# it via `data.path: <user>/<repo>`.
#
# Usage:
#   ./prepare_dataset.sh [--src PATH] [--base-model HF_ID]
#                        [--dataset-repo NAME] [--split-name train]
#                        [--private] [--overwrite]
#
# Defaults:
#   --src           SFT/data/ift_data.json
#   --base-model    meta-llama/Llama-3.1-8B-Instruct
#   --dataset-repo  ${HF_USERNAME}/athena-ift
#   --split-name    train   (produces <split>.jsonl in the repo)
#
# Requires these env vars (set by you before running):
#   HF_TOKEN       write-scope HF token
#   HF_USERNAME    target namespace for the dataset repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC="${SFT_DIR}/data/ift_data.json"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET_REPO=""
SPLIT_NAME="train"
PRIVATE=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)          SRC="$2"; shift 2 ;;
        --base-model)   BASE_MODEL="$2"; shift 2 ;;
        --dataset-repo) DATASET_REPO="$2"; shift 2 ;;
        --split-name)   SPLIT_NAME="$2"; shift 2 ;;
        --private)      PRIVATE=1; shift ;;
        --overwrite)    OVERWRITE=1; shift ;;
        -h|--help) sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

: "${HF_TOKEN:?Set HF_TOKEN before running (write-scope token)}"
: "${HF_USERNAME:?Set HF_USERNAME before running}"

if [[ -z "${DATASET_REPO}" ]]; then
    DATASET_REPO="${HF_USERNAME}/athena-ift"
fi

if [[ ! -f "${SRC}" ]]; then
    echo "[FAIL] source dataset not found: ${SRC}" >&2
    exit 1
fi

OUT_DIR="${SCRIPT_DIR}/_dataset_staging"
mkdir -p "${OUT_DIR}"
OUT_JSONL="${OUT_DIR}/${SPLIT_NAME}.jsonl"

if [[ -f "${OUT_JSONL}" && ${OVERWRITE} -ne 1 ]]; then
    echo "[skip] ${OUT_JSONL} already exists (use --overwrite to rebuild)"
else
    echo "=== Converting ${SRC} -> ${OUT_JSONL} ==="
    echo "  applying chat template from : ${BASE_MODEL}"
    python - "${SRC}" "${OUT_JSONL}" "${BASE_MODEL}" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

src, dst, base_model = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
rows = json.loads(Path(src).read_text())

def to_messages(ex):
    msgs = []
    sys_prompt = (ex.get("instruction") or "").strip()
    user = (ex.get("input") or "").strip()
    asst = (ex.get("output") or "").strip()
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    if user:
        msgs.append({"role": "user", "content": user})
    if asst:
        msgs.append({"role": "assistant", "content": asst})
    return msgs

n_kept = 0
with Path(dst).open("w", encoding="utf-8") as f:
    for ex in rows:
        msgs = to_messages(ex)
        if len(msgs) < 2:
            continue
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        n_kept += 1
print(f"  wrote {n_kept}/{len(rows)} rows to {dst}")
PY
fi

echo
echo "=== Uploading to HF dataset repo: ${DATASET_REPO} ==="

PRIVATE_FLAG=""
[[ ${PRIVATE} -eq 1 ]] && PRIVATE_FLAG="--private"

python - "${DATASET_REPO}" "${HF_TOKEN}" "${PRIVATE}" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id, token, private = sys.argv[1], sys.argv[2], bool(int(sys.argv[3]))
api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
print(f"  repo ready: https://huggingface.co/datasets/{repo_id}")
PY

huggingface-cli upload "${DATASET_REPO}" \
    "${OUT_JSONL}" "${SPLIT_NAME}.jsonl" \
    --repo-type dataset \
    --token "${HF_TOKEN}"

echo
echo "=== Dataset upload complete ==="
echo "  repo     : https://huggingface.co/datasets/${DATASET_REPO}"
echo "  split    : ${SPLIT_NAME} (file: ${SPLIT_NAME}.jsonl)"
echo "  local    : ${OUT_JSONL}"
echo
echo "Next: launch training with"
echo "  ${SCRIPT_DIR}/train.sh"
