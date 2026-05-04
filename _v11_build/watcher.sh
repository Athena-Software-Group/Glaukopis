#!/bin/bash
# v11 build watcher: polls the make_dataset.sh process; on success runs the
# TAA actor-balance post-pass and the eval-set dedup, then sends a macOS
# notification. Status is appended to _v11_build/watcher.log and the
# final outcome lands in _v11_build/watcher_status.json.
#
# Same shape as _v10_build/watcher.sh (see git history at c5f992c~1).
# v11 deltas vs v10:
#   ACTOR_CAP            20 -> 40   (guardrails proven; permits ~3500 TAA
#                                    positives vs v10's 1284, dropping the
#                                    IE/NEG : positive ratio from 3.9:1 to
#                                    ~1.7:1 -- still well above mode-collapse
#                                    risk threshold)
#   DEDUP_DROP_THRESHOLD 50 -> 30   (less conservative; v10's 50 dropped
#                                    only 621 rows total, some of which were
#                                    likely legitimate diversity. Re-inspect
#                                    100 random borderline cases (30-50
#                                    shared 13-grams) before locking the v11
#                                    corpus.)
#
# Build-side defaults inherited by Phase 1 (make_dataset.sh) for v11
# (encoded in tmpl_gen/data_generation/gencfg_per_primary_neo4j.json,
# now the default GENCONF in tmpl_gen/data_generation/{make_dataset,
# tmpl2triples}.sh):
#   per_primary_grouping=true   sample-with-replacement anchor diversity
#                               (fixes v10 AB.MS.* / AB.TAA.* anchor
#                               fixation; see tmpl_parser.py process_template)
#   allow_nullprops=true        tolerate null/empty/"N/A" property values
#                               so RMS/MS templates do not lose rows when
#                               descriptions are missing on Neo4j nodes
#
# Launch with nohup so it survives terminal close:
#   nohup bash _v11_build/watcher.sh > _v11_build/watcher.log 2>&1 &

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

BUILD_PID="$(cat _v11_build/build.pid 2>/dev/null | sed 's/PID=//')"
RAW_JSON="SFT/data/ift_data_2026_05_03_v11.raw.json"
BALANCED_JSON="SFT/data/ift_data_2026_05_03_v11.balanced.json"
CLEAN_JSON="SFT/data/ift_data_2026_05_03_v11.json"
DEDUP_REPORT="_v11_build/dedup_report.json"
STATUS_JSON="_v11_build/watcher_status.json"

# Tunables (encoded for reproducibility):
#   actor cap/floor: keeps the long-tail TAA spread, trims mode-collapse heads.
#   dedup hit-threshold/drop-threshold: hit-threshold low enough to surface
#     all overlap, drop-threshold high enough to ignore incidental shared
#     CVE/MITRE description vocabulary (eval & training share the same
#     knowledge base) and only drop verbatim contamination.
ACTOR_CAP=40
ACTOR_FLOOR=100
DEDUP_HIT_THRESHOLD=1
DEDUP_DROP_THRESHOLD=30

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()   { echo "[$(stamp)] $*"; }

notify() {
    local title="$1" msg="$2"
    osascript -e "display notification \"${msg}\" with title \"${title}\"" \
        2>/dev/null || true
}

write_status() {
    local stage="$1" outcome="$2" detail="$3"
    cat > "${STATUS_JSON}" <<EOF
{
  "ended_at": "$(stamp)",
  "build_pid": ${BUILD_PID:-null},
  "stage": "${stage}",
  "outcome": "${outcome}",
  "detail": ${detail}
}
EOF
}

if [[ -z "${BUILD_PID}" ]]; then
    log "ERROR: _v11_build/build.pid not found; nothing to watch."
    write_status "init" "fail" "\"build.pid missing\""
    exit 1
fi

log "watcher starting; build_pid=${BUILD_PID}"
notify "v11 build" "watcher started; tracking PID ${BUILD_PID}"

# Phase 1: poll until the make_dataset.sh process exits
# Triple count is manifest-dependent; v10 had 216, v11 will be that plus
# the new template blocks per tmpl_gen/templates/05032026/v11_plan.txt §5.
while kill -0 "${BUILD_PID}" 2>/dev/null; do
    triples_n=$(ls _v11_build/triples/ 2>/dev/null | wc -l | tr -d ' ')
    log "build still running; triples=${triples_n}"
    sleep 60
done
log "build process ${BUILD_PID} has exited"

# Phase 2: validate that the raw json was produced
if [[ ! -f "${RAW_JSON}" ]]; then
    log "FAIL: build exited but ${RAW_JSON} was not produced"
    notify "v11 build FAILED" "raw json missing; see _v11_build/build.log"
    write_status "build" "fail" "\"raw json not produced; check _v11_build/build.log\""
    exit 1
fi
raw_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_JSON}'))))" \
           2>/dev/null || echo "0")
log "build OK; raw rows=${raw_rows}"

# Phase 3: TAA actor-balance post-pass
log "running taa_actor_balance.py (cap=${ACTOR_CAP}, floor=${ACTOR_FLOOR})..."
balance_log="_v11_build/balance.log"
if python tmpl_gen/scripts/taa_actor_balance.py \
        --input "${RAW_JSON}" --output "${BALANCED_JSON}" \
        --max-per-actor "${ACTOR_CAP}" --min-actors "${ACTOR_FLOOR}" \
        > "${balance_log}" 2>&1; then
    balanced_rows=$(python3 -c "import json; print(len(json.load(open('${BALANCED_JSON}'))))" \
                    2>/dev/null || echo "0")
    log "actor-balance OK; balanced rows=${balanced_rows}"
else
    rc=$?
    log "FAIL: taa_actor_balance.py exit ${rc}; see ${balance_log}"
    notify "v11 build FAILED" "actor floor not met (rc=${rc}); see _v11_build/balance.log"
    write_status "actor_balance" "fail" \
        "{\"raw_rows\": ${raw_rows}, \"exit_code\": ${rc}, \"log\": \"${balance_log}\"}"
    exit 1
fi

# Phase 4: dedup against eval sets (filter + report)
log "running dedup_against_evals.py (n=13, hit=${DEDUP_HIT_THRESHOLD}, drop>=${DEDUP_DROP_THRESHOLD})..."
dedup_log="_v11_build/dedup.log"
# --max-fail set deliberately high: shared CVE/MITRE description vocabulary
# between training and eval is expected since both derive from the same
# Neo4j knowledge base. The drop step is what matters for contamination.
if python tmpl_gen/scripts/dedup_against_evals.py \
        --input "${BALANCED_JSON}" \
        --filter-output "${CLEAN_JSON}" \
        --drop-threshold "${DEDUP_DROP_THRESHOLD}" \
        --hit-threshold "${DEDUP_HIT_THRESHOLD}" \
        --max-fail 999999 \
        --report "${DEDUP_REPORT}" \
        > "${dedup_log}" 2>&1; then
    clean_rows=$(python3 -c "import json; print(len(json.load(open('${CLEAN_JSON}'))))" \
                 2>/dev/null || echo "0")
    dropped=$((balanced_rows - clean_rows))
    log "dedup OK; dropped ${dropped} verbatim-contam rows; clean=${clean_rows}"
    notify "v11 build done" "raw=${raw_rows} clean=${clean_rows} (dropped ${dropped})"
    write_status "dedup" "ok" \
        "{\"raw_rows\": ${raw_rows}, \"balanced_rows\": ${balanced_rows}, \"clean_rows\": ${clean_rows}, \"dropped_contam\": ${dropped}, \"clean_path\": \"${CLEAN_JSON}\", \"balanced_path\": \"${BALANCED_JSON}\", \"dedup_report\": \"${DEDUP_REPORT}\", \"actor_cap\": ${ACTOR_CAP}, \"actor_floor\": ${ACTOR_FLOOR}, \"dedup_hit_threshold\": ${DEDUP_HIT_THRESHOLD}, \"dedup_drop_threshold\": ${DEDUP_DROP_THRESHOLD}}"
else
    rc=$?
    log "FAIL: dedup_against_evals.py exit ${rc}; see ${dedup_log}"
    notify "v11 build FAILED" "dedup error rc=${rc}; see _v11_build/dedup.log"
    write_status "dedup" "fail" \
        "{\"raw_rows\": ${raw_rows}, \"balanced_rows\": ${balanced_rows}, \"exit_code\": ${rc}, \"log\": \"${dedup_log}\", \"report\": \"${DEDUP_REPORT}\"}"
    exit 1
fi

log "watcher complete"
exit 0
