#!/bin/bash
# v21-TAA build watcher: forked from _v18_taa_build/watcher.sh as a
# strict-reproducibility replay of v18.1's TAA stage (tmpl_gen/templates/
# 05182026/v21_plan.txt). Operationally byte-identical; only paths and
# dataset names are relabelled to v21 + 2026_05_18.
#
# v21-TAA is the chained TAA-Classic refresher shard for v21, generated
# from tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_taa.txt
# (byte-identical to v16's TAA Classic manifest; TAA.CANON purged).
# Output is a single TAA shard (ift_data_2026_05_18_v21_taa.json)
# consumed by SFT/autotrain/run_sft_qwen25_14b_v21_plus_taa.sh, which
# chains off asg-ai/athena-cti-sft-qwen25-14b-v21-core.
#
# Pipeline (verbatim port of v16 watcher; only paths/labels change):
#   Phase 1  -- poll make_dataset.sh PID until exit (raw json produced)
#   Phase 2  -- validate raw json exists
#   Phase 4  -- TAA actor-balance with per-actor cap 60
#   Phase 5  -- dedup against eval sets (n=13, drop>=50)
#   Phase 6  -- row-count gate (re-uses v16 thresholds in
#               tmpl_gen/templates/05182026/v21_taa_row_count_gate.json since
#               the manifest is identical)
#   Phase 6b -- licence-allowlist gate
#   Phase 7  -- stratified shuffle
#   Phase 8  -- val/train split (build_val_slice.py with v21-TAA paths)
#
# Status appended to _v21_taa_build/watcher.log; final outcome lands in
# _v21_taa_build/watcher_status.json.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

BUILD_PID="$(cat _v21_taa_build/build.pid 2>/dev/null | sed 's/PID=//')"
RAW_JSON="SFT/data/ift_data_2026_05_18_v21_taa.raw.json"
BALANCED_JSON="SFT/data/ift_data_2026_05_18_v21_taa.balanced.json"
CLEAN_JSON="SFT/data/ift_data_2026_05_18_v21_taa.clean.json"
SHUFFLED_JSON="SFT/data/ift_data_2026_05_18_v21_taa.shuffled.json"
VAL_JSON="SFT/data/ift_data_2026_05_18_v21_taa_val.json"
TRAIN_JSON="SFT/data/ift_data_2026_05_18_v21_taa.json"
DEDUP_REPORT="_v21_taa_build/dedup_report.json"
ROW_GATE_REPORT="_v21_taa_build/row_count_gate_report.json"
LICENCE_GATE_REPORT="_v21_taa_build/licence_gate_report.json"
STATUS_JSON="_v21_taa_build/watcher_status.json"

ACTOR_CAP=60
ACTOR_FLOOR=100
DEDUP_HIT_THRESHOLD=1
DEDUP_DROP_THRESHOLD=50
ROW_GATE_PLAN="tmpl_gen/templates/05182026/v21_taa_row_count_gate.json"
SHUFFLE_SEED=42
VAL_PER_AXIS=50

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

fail() {
    local stage="$1" rc="$2" msg="$3" detail="$4"
    log "FAIL: ${msg}"
    notify "v21-TAA build FAILED" "${msg}"
    write_status "${stage}" "fail" "${detail}"
    exit "${rc}"
}

if [[ -z "${BUILD_PID}" ]]; then
    log "ERROR: _v21_taa_build/build.pid not found; nothing to watch."
    write_status "init" "fail" "\"build.pid missing\""
    exit 1
fi

log "watcher starting; build_pid=${BUILD_PID}"
notify "v21-TAA build" "watcher started; tracking PID ${BUILD_PID}"

while kill -0 "${BUILD_PID}" 2>/dev/null; do
    triples_n=$(ls _v21_taa_build/triples/ 2>/dev/null | wc -l | tr -d ' ')
    log "build still running; triples=${triples_n}"
    sleep 60
done
log "build process ${BUILD_PID} has exited"

if [[ ! -f "${RAW_JSON}" ]]; then
    fail "build" 1 "raw json ${RAW_JSON} missing; see _v21_taa_build/build.log" \
        "\"raw json not produced; check _v21_taa_build/build.log\""
fi
raw_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_JSON}'))))" \
           2>/dev/null || echo "0")
log "build OK; raw rows=${raw_rows}"

log "running taa_actor_balance.py (per-actor=${ACTOR_CAP}, floor=${ACTOR_FLOOR})..."
balance_log="_v21_taa_build/balance.log"
if python tmpl_gen/scripts/taa_actor_balance.py \
        --input "${RAW_JSON}" --output "${BALANCED_JSON}" \
        --max-per-actor "${ACTOR_CAP}" --min-actors "${ACTOR_FLOOR}" \
        > "${balance_log}" 2>&1; then
    balanced_rows=$(python3 -c "import json; print(len(json.load(open('${BALANCED_JSON}'))))" \
                    2>/dev/null || echo "0")
    log "actor-balance OK; balanced rows=${balanced_rows}"
else
    rc=$?
    fail "actor_balance" "${rc}" "actor floor not met (rc=${rc}); see ${balance_log}" \
        "{\"raw_rows\": ${raw_rows}, \"exit_code\": ${rc}, \"log\": \"${balance_log}\"}"
fi

log "running dedup_against_evals.py (n=13, hit=${DEDUP_HIT_THRESHOLD}, drop>=${DEDUP_DROP_THRESHOLD})..."
dedup_log="_v21_taa_build/dedup.log"
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
else
    rc=$?
    fail "dedup" "${rc}" "dedup error rc=${rc}; see ${dedup_log}" \
        "{\"balanced_rows\": ${balanced_rows}, \"exit_code\": ${rc}, \"log\": \"${dedup_log}\"}"
fi

log "running check_corpus_row_counts.py against ${ROW_GATE_PLAN}..."
gate_log="_v21_taa_build/row_count_gate.log"
if python tmpl_gen/scripts/check_corpus_row_counts.py \
        --input "${CLEAN_JSON}" \
        --plan "${ROW_GATE_PLAN}" \
        --report "${ROW_GATE_REPORT}" \
        > "${gate_log}" 2>&1; then
    log "row-count gate OK; all axes meet floor"
else
    rc=$?
    fail "row_count_gate" "${rc}" \
        "row-count gate failed (rc=${rc}); see ${ROW_GATE_REPORT} and ${gate_log}" \
        "{\"clean_rows\": ${clean_rows}, \"exit_code\": ${rc}, \"report\": \"${ROW_GATE_REPORT}\", \"log\": \"${gate_log}\"}"
fi


log "running check_corpus_licences.py..."
licence_log="_v21_taa_build/licence_gate.log"
if python tmpl_gen/scripts/check_corpus_licences.py \
        --input "${CLEAN_JSON}" \
        --report "${LICENCE_GATE_REPORT}" \
        > "${licence_log}" 2>&1; then
    log "licence gate OK; all rows in commercial-use allowlist"
else
    rc=$?
    fail "licence_gate" "${rc}" \
        "licence gate failed (rc=${rc}); see ${LICENCE_GATE_REPORT} and ${licence_log}" \
        "{\"clean_rows\": ${clean_rows}, \"exit_code\": ${rc}, \"report\": \"${LICENCE_GATE_REPORT}\", \"log\": \"${licence_log}\"}"
fi

log "running stratified_shuffle.py (seed=${SHUFFLE_SEED})..."
shuffle_log="_v21_taa_build/shuffle.log"
if python tmpl_gen/scripts/stratified_shuffle.py \
        --input "${CLEAN_JSON}" \
        --output "${SHUFFLED_JSON}" \
        --seed "${SHUFFLE_SEED}" \
        --validate \
        > "${shuffle_log}" 2>&1; then
    shuffled_rows=$(python3 -c "import json; print(len(json.load(open('${SHUFFLED_JSON}'))))" \
                    2>/dev/null || echo "0")
    log "stratified shuffle OK; shuffled rows=${shuffled_rows}"
else
    rc=$?
    fail "stratified_shuffle" "${rc}" \
        "stratified_shuffle.py exit ${rc}; see ${shuffle_log}" \
        "{\"clean_rows\": ${clean_rows}, \"exit_code\": ${rc}, \"log\": \"${shuffle_log}\"}"
fi

log "running build_val_slice.py (per-axis=${VAL_PER_AXIS}, seed=${SHUFFLE_SEED})..."
val_log="_v21_taa_build/val_slice.log"
if python _v21_taa_build/build_val_slice.py \
        --input "${SHUFFLED_JSON}" \
        --val-out "${VAL_JSON}" \
        --train-out "${TRAIN_JSON}" \
        --per-axis "${VAL_PER_AXIS}" \
        --seed "${SHUFFLE_SEED}" \
        > "${val_log}" 2>&1; then
    val_rows=$(python3 -c "import json; print(len(json.load(open('${VAL_JSON}'))))" \
               2>/dev/null || echo "0")
    train_rows=$(python3 -c "import json; print(len(json.load(open('${TRAIN_JSON}'))))" \
                 2>/dev/null || echo "0")
    log "val/train split OK; val=${val_rows} train=${train_rows}"
else
    rc=$?
    fail "val_slice" "${rc}" "build_val_slice.py exit ${rc}; see ${val_log}" \
        "{\"shuffled_rows\": ${shuffled_rows}, \"exit_code\": ${rc}, \"log\": \"${val_log}\"}"
fi

notify "v21-TAA build done" "raw=${raw_rows} clean=${clean_rows} train=${train_rows} val=${val_rows}"
write_status "val_slice" "ok" \
    "{\"raw_rows\": ${raw_rows}, \"balanced_rows\": ${balanced_rows}, \"clean_rows\": ${clean_rows}, \"shuffled_rows\": ${shuffled_rows}, \"val_rows\": ${val_rows}, \"train_rows\": ${train_rows}, \"clean_path\": \"${CLEAN_JSON}\", \"shuffled_path\": \"${SHUFFLED_JSON}\", \"val_path\": \"${VAL_JSON}\", \"train_path\": \"${TRAIN_JSON}\", \"actor_cap\": ${ACTOR_CAP}, \"actor_floor\": ${ACTOR_FLOOR}, \"dedup_hit_threshold\": ${DEDUP_HIT_THRESHOLD}, \"dedup_drop_threshold\": ${DEDUP_DROP_THRESHOLD}, \"shuffle_seed\": ${SHUFFLE_SEED}, \"val_per_axis\": ${VAL_PER_AXIS}, \"reports\": {\"dedup\": \"${DEDUP_REPORT}\", \"row_count_gate\": \"${ROW_GATE_REPORT}\", \"licence_gate\": \"${LICENCE_GATE_REPORT}\"}}"

log "watcher complete"
exit 0
