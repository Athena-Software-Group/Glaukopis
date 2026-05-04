#!/bin/bash

# Full dataset generation pipeline.
# Uses whichever Python is active in the current shell (conda env, venv, or system Python).
#
#   1. docx → JSON templates  (docx2json)  [skipped if input is already .json]
#   2. JSON templates → triples  (tmpl2triples)
#   3. triples → Alpaca dataset  (triples2alpaca)

usage() {
    echo "Usage: $0 tmpl.docx|tmpl.json results_dir alpaca_output.json [count_limit=10] [count_max=2000]" >&2
    echo >&2
    echo "  tmpl.docx|tmpl.json - Word document or JSON templates file" >&2
    echo "  results_dir         - directory for generated triples (WILL BE ERASED)" >&2
    echo "  alpaca_output.json  - final Alpaca-format dataset file" >&2
    echo "  count_limit         - max generations per template in docx2json (default: 10, ignored for .json input)" >&2
    echo "  count_max           - max triples per template in tmpl2triples (default: 2000)" >&2
    exit 1
}

if [ "$#" -lt 3 ]; then
    echo "Error: at least 3 arguments required" >&2
    usage
fi

SOURCE_INPUT="$1"
RESULTS_DIR="$2"
ALPACA_JSON="$3"
COUNT_LIMIT="${4:-10}"
COUNT_MAX="${5:-2000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_DOCX2JSON="${SCRIPT_DIR}/../scripts/tmpl_docx2json.py"
PY_IFTGEN="${SCRIPT_DIR}/../scripts/iftgen.py"
PY_ALPACA="${SCRIPT_DIR}/../scripts/to_alpaca.py"
# Default to the per-primary gencfg, which enables sample-with-replacement
# anchor diversity (per_primary_grouping=true) and null/empty/N/A property
# tolerance (allow_nullprops=true) -- both required for v11 to avoid the
# v10 anchor-fixation collapse on AB.MS.* / AB.TAA.* templates.
# Override with: GENCONF=path/to/other_gencfg.json ./make_dataset.sh ...
GENCONF="${GENCONF:-${SCRIPT_DIR}/gencfg_per_primary_neo4j.json}"
NEO4JCONF="${NEO4JCONF:-${SCRIPT_DIR}/neo4j-local-config.json}"

set -e

# ── Step 1: docx → JSON templates (skipped if input is already .json) ────────
EXT=$(echo "${SOURCE_INPUT##*.}" | tr '[:upper:]' '[:lower:]')

if [[ "${EXT}" == "json" ]]; then
    echo "=== [1/3] docx → JSON templates: SKIPPED (input is already JSON) ==="
    TMPL_JSON="${SOURCE_INPUT}"
    echo "  Using: ${TMPL_JSON}"
    echo
else
    TMPL_JSON="${SCRIPT_DIR}/$(basename "${SOURCE_INPUT%.*}").json"
    echo "=== [1/3] docx → JSON templates ==="
    echo "  Input : ${SOURCE_INPUT}"
    echo "  Output: ${TMPL_JSON}"
    echo "  count_limit=${COUNT_LIMIT}"
    echo

    python "${PY_DOCX2JSON}" \
        -i "${SOURCE_INPUT}" \
        -o "${TMPL_JSON}" \
        --count_limit "${COUNT_LIMIT}"

    echo
    echo "=== [1/3] done ==="
    echo
fi

# ── Step 2: JSON templates → triples ─────────────────────────────────────────
echo "=== [2/3] templates → triples ==="
echo "  Input     : ${TMPL_JSON}"
echo "  Output dir: ${RESULTS_DIR}  (WILL BE ERASED)"
echo "  count_max=${COUNT_MAX}"
echo

rm -vfr "${RESULTS_DIR}"

python "${PY_IFTGEN}" \
    --cmd generate \
    --genconf "${GENCONF}" \
    --dbconf "${NEO4JCONF}" \
    --tmpl "${TMPL_JSON}" \
    --results_dir "${RESULTS_DIR}" \
    --count_max "${COUNT_MAX}"

echo
echo "=== [2/3] done ==="
echo

# ── Step 3: triples → Alpaca dataset ─────────────────────────────────────────
echo "=== [3/3] triples → Alpaca dataset ==="
echo "  Input : ${RESULTS_DIR}"
echo "  Output: ${ALPACA_JSON}"
echo

# Uncomment to override the instruction field:
# OVERRIDE_INSTRUCTIONS="You are a CTI expert who gives precise and concise answers."

if [[ -z "${OVERRIDE_INSTRUCTIONS}" ]]; then
    python "${PY_ALPACA}" \
        --results_dir "${RESULTS_DIR}" \
        --output "${ALPACA_JSON}" \
        --count_max -1
else
    echo "  Overriding instruction field: ${OVERRIDE_INSTRUCTIONS}"
    python "${PY_ALPACA}" \
        --results_dir "${RESULTS_DIR}" \
        --output "${ALPACA_JSON}" \
        --instruction "${OVERRIDE_INSTRUCTIONS}" \
        --count_max -1
fi

echo
echo "=== [3/3] done ==="
echo
echo "Dataset ready: ${ALPACA_JSON}"
