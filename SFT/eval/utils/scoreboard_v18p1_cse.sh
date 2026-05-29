#!/bin/bash

# Scoreboard scraper for the v18.1 CyberSOCEval results on the test box.
# Walks SFT/eval/responses/ for the three v18.1 display names and prints,
# for each model:
#   1. Whether per-suite summary_cybersoceval{,_malware,_ti}.json exists.
#   2. The headline metrics from each summary (avg_score / correct_mc_pct /
#      response_parsing_error_count) without needing pandas / numpy.
#   3. Raw .jsonl response counts (so partial / aborted runs are still
#      visible even when no summary was written).
#   4. The most recent sweep log line mentioning 'cybersoceval' so you can
#      tell whether the suite ran, errored, or was skipped.
#
# Usage (on the test box, from repo root):
#   bash SFT/eval/utils/scoreboard_v18p1_cse.sh
#
# No deps beyond bash + python3 stdlib (the bench-client conda env is NOT
# required; this script reads JSON only).

set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
RESPONSES="responses"

DISPLAY_NAMES=(
    "asg-ai_athena-cti-sft-qwen25-14b-v18-1-core"
    "asg-ai_athena-cti-sft-qwen25-14b-v18-1-taa"
    "asg-ai_athena-cti-sft-qwen25-14b-v18-1-cse"
)

print_summary_json() {
    local path="$1"
    [[ -f "${path}" ]] || { echo "    (missing: $(basename "${path}"))"; return; }
    python3 - "${path}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text())
except Exception as e:
    print(f"    (failed to parse {p.name}: {e})")
    sys.exit(0)
print(f"    file: {p.name}  ({p.stat().st_size} bytes)")
def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if isinstance(v, (dict, list)):
                walk(v, prefix + k + ".")
            elif any(t in kl for t in ("score", "accuracy", "pct", "error", "count", "jaccard", "correct")):
                print(f"      {prefix}{k}: {v}")
walk(d)
PY
}

count_response_rows() {
    local path="$1"
    [[ -f "${path}" ]] || { echo "0 (missing)"; return; }
    case "${path}" in
        *.jsonl) wc -l < "${path}" | tr -d ' ' ;;
        *.csv)   echo "$(($(wc -l < "${path}") - 1))" ;;
        *)       echo "?" ;;
    esac
}

echo "================================================================="
echo "  v18.1 CyberSOCEval scoreboard ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo "================================================================="
for disp in "${DISPLAY_NAMES[@]}"; do
    echo
    echo "----- ${disp} -----"
    dir="${RESPONSES}/${disp}"
    if [[ ! -d "${dir}" ]]; then
        echo "  [no responses dir: ${dir}]"
        continue
    fi

    # Per-suite summaries (the canonical scoreboard artefact)
    echo "  summaries:"
    for stem in summary_cybersoceval summary_cybersoceval_malware summary_cybersoceval_ti; do
        for ext in json; do
            f="${dir}/${stem}.${ext}"
            [[ -f "${f}" ]] && print_summary_json "${f}"
        done
    done
    # Match anything else that looks cybersoc-shaped (older naming, sub-slices)
    extra_summaries="$(find "${dir}" -maxdepth 1 -name 'summary_cybersoc*.json' \
        ! -name 'summary_cybersoceval.json' \
        ! -name 'summary_cybersoceval_malware.json' \
        ! -name 'summary_cybersoceval_ti.json' 2>/dev/null)"
    if [[ -n "${extra_summaries}" ]]; then
        while IFS= read -r f; do print_summary_json "${f}"; done <<< "${extra_summaries}"
    fi

    # Raw response rows (so partial runs still surface)
    echo "  raw response rows:"
    for sub in cybersoceval-malware cybersoceval-ti; do
        if [[ -d "${dir}/${sub}" ]]; then
            for f in "${dir}/${sub}"/*.jsonl "${dir}/${sub}"/*.csv; do
                [[ -f "${f}" ]] || continue
                printf "    %-60s %s rows\n" "${sub}/$(basename "${f}")" "$(count_response_rows "${f}")"
            done
        else
            echo "    ${sub}/  (no dir)"
        fi
    done
done

echo
echo "----- recent sweep log lines mentioning cybersoceval -----"
LOG_DIR="utils"
recent_logs="$(ls -t "${LOG_DIR}"/v18p1_*.log "${LOG_DIR}"/foundation_8b_baselines_*.log 2>/dev/null | head -5)"
if [[ -z "${recent_logs}" ]]; then
    echo "  (no sweep logs in ${LOG_DIR}/)"
else
    while IFS= read -r log; do
        echo
        echo "  ${log}:"
        grep -niE "cybersoceval|avg_score|correct_mc_pct|parsing_error" "${log}" 2>/dev/null \
            | tail -20 | sed 's/^/    /'
    done <<< "${recent_logs}"
fi

echo
echo "[done] If every block above shows '[no responses dir]' or '0 rows', the"
echo "       CyberSOCEval suite has not produced any data yet. Check the most"
echo "       recent sweep log for the abort point (the pandas-import error"
echo "       documented in commit 165d787 was the last known blocker)."
