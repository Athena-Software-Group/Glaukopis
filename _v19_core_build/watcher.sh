#!/bin/bash
# v19-core build watcher: builds the two Core shards consumed by
# SFT/autotrain/run_sft_qwen25_14b_v19_core.sh (Phase A broad,
# Phase B catalog drill). Forked from _v19_core_build/watcher.sh per
# tmpl_gen/templates/05152026/v19_plan.txt §1.3 (v19 inherits the
# v18.1 build pipeline verbatim; only Stage 5 prob-mix changes):
#
#   * paths     : _v18p1_build -> _v19_core_build,
#                 2026_05_11_v18p1 -> 2026_05_15_v19
#   * gate plan : tmpl_gen/templates/05152026/v19_row_count_gate.json
#                 (carries v18.1 floors verbatim: MCQ 6000 / VSP 5400 /
#                  RMS 12000 / ATE 12500 / RCM 9000 / etc.)
#   * Phase 3c  : MCQ.EXT.* generator merge stays SKIPPED (v18.1 carry)
#   * Stage 5 prob-mix delta is at training time, not build time, so
#     the watcher requires no behavioural change for Stage 5.
#
# v19 TAA + CSE downstream stages are built by the sibling watchers
# _v19_taa_build/watcher.sh and _v19_cse_build/watcher.sh.
#
# Pipeline:
#
#   Phase 0a -- substrate validation (_neo4j_check.py)        (v18.1 carry)
#   Phase 0b -- seed-provenance gate                          (v18.1 carry)
#   Phase 1  -- poll make_dataset.sh PID until exit (raw json produced)
#   Phase 2  -- validate raw json exists                      (v18.1 carry)
#   Phase 3b -- CM.* generator merge                          (v18.1 carry)
#   Phase 3c -- [SKIPPED] MCQ.EXT.* generator merge           (v18.1 carry)
#   Phase 3d -- SOC.*.GEN.* generator merge                   (v18.1 carry)
#   Phase 4  -- TAA actor-balance with --max-rows-per-family-total 3500
#   Phase 5  -- dedup against eval sets (n=13, drop>=50)      (v18.1 carry)
#   Phase 6  -- row-count gate (v19 thresholds in v19_row_count_gate.json)
#   Phase 6b -- licence-allowlist gate                        (v18.1 carry)
#   Phase 7  -- stratified shuffle                            (v18.1 carry)
#   Phase 8  -- val/train split (build_val_slice.py)          (v19 paths)
#   Phase 9  -- per-phase corpus split: TWO shards (broad + axis)
#
# Status appended to _v19_core_build/watcher.log; final outcome lands in
# _v19_core_build/watcher_status.json. Each gate writes its own report json
# under _v19_core_build/.
#
# Build-side defaults inherited from v18.1 (gencfg_per_primary_neo4j.json):
#   per_primary_grouping=true, allow_nullprops=true.
#
# Launch with nohup so it survives terminal close:
#   echo "PID=$$" > _v19_core_build/build.pid     # if calling from a wrapper
#   nohup bash _v19_core_build/watcher.sh > _v19_core_build/watcher.log 2>&1 &

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

BUILD_PID="$(cat _v19_core_build/build.pid 2>/dev/null | sed 's/PID=//')"
RAW_JSON="SFT/data/ift_data_2026_05_15_v19_core.raw.json"
RAW_MERGED_JSON="SFT/data/ift_data_2026_05_15_v19_core.raw_merged.json"
BALANCED_JSON="SFT/data/ift_data_2026_05_15_v19_core.balanced.json"
CLEAN_JSON="SFT/data/ift_data_2026_05_15_v19_core.json"
SHUFFLED_JSON="SFT/data/ift_data_2026_05_15_v19_core.shuffled.json"
VAL_JSON="SFT/data/ift_data_2026_05_15_v19_core_val.json"
TRAIN_JSON="SFT/data/ift_data_2026_05_15_v19_core_train.json"
PHASE_A_JSON="SFT/data/ift_data_2026_05_15_v19_core_a_kb_mcq_taa_soc_cm_ms_yn.json"
PHASE_B_JSON="SFT/data/ift_data_2026_05_15_v19_core_b_rms_ate_vsp_rcm.json"
DEDUP_REPORT="_v19_core_build/dedup_report.json"
ROW_GATE_REPORT="_v19_core_build/row_count_gate_report.json"
LICENCE_GATE_REPORT="_v19_core_build/licence_gate_report.json"
SUBSTRATE_REPORT="_v19_core_build/substrate_report.json"
SEED_PROVENANCE_REPORT="_v19_core_build/seed_provenance_report.json"
PHASE_SPLIT_REPORT="_v19_core_build/phase_split_report.json"
CM_REPORT="_v19_core_build/cm_report.json"
SOC_REPORT="_v19_core_build/soc_report.json"
STATUS_JSON="_v19_core_build/watcher_status.json"

# Tunables (v18 defaults; carried verbatim except ROW_GATE_PLAN)
ACTOR_CAP=40
ACTOR_FLOOR=100
TAA_TOTAL_CAP=3500
DEDUP_HIT_THRESHOLD=1
DEDUP_DROP_THRESHOLD=50
ROW_GATE_PLAN="tmpl_gen/templates/05152026/v19_row_count_gate.json"
CM_TARGET_CRYPTO=1500
CM_TARGET_ACCESS=1500
CM_TARGET_COMPLIANCE=2000
CM_TARGET_GOV=1000
# v18.1 / v19: MCQ.EXT.* generator merge dropped per v19_plan.txt §3.1
SOC_TARGET_SIGMA=1500
SOC_TARGET_MAL=1500
SOC_TARGET_IR=1000
SOC_TARGET_TRIAGE=1000
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
    notify "v19 build FAILED" "${msg}"
    write_status "${stage}" "fail" "${detail}"
    exit "${rc}"
}

if [[ -z "${BUILD_PID}" ]]; then
    log "ERROR: _v19_core_build/build.pid not found; nothing to watch."
    write_status "init" "fail" "\"build.pid missing\""
    exit 1
fi

log "watcher starting; build_pid=${BUILD_PID}"
notify "v19 build" "watcher started; tracking PID ${BUILD_PID}"

# ---- Phase 0a: substrate validation (v18 carry; same Neo4j entity floors) ----
log "running _v19_core_build/_neo4j_check.py against athena-cti-db..."
substrate_log="_v19_core_build/substrate.log"
if python3 _v19_core_build/_neo4j_check.py \
        --report "${SUBSTRATE_REPORT}" \
        > "${substrate_log}" 2>&1; then
    log "substrate validation OK; all v18 entity + traversal floors met"
else
    rc=$?
    fail "substrate" "${rc}" \
        "substrate validation failed; see ${substrate_log} and ${SUBSTRATE_REPORT}" \
        "{\"exit_code\": ${rc}, \"log\": \"${substrate_log}\", \"report\": \"${SUBSTRATE_REPORT}\"}"
fi


# ---- Phase 0b: seed-provenance gate (v13 carry; v18 inherits MITRE+MISP+Athena registrations) ----
log "running check_seed_provenance.py against registered upstream seeds..."
seed_log="_v19_core_build/seed_provenance.log"
if python3 tmpl_gen/scripts/check_seed_provenance.py \
        --report "${SEED_PROVENANCE_REPORT}" \
        > "${seed_log}" 2>&1; then
    log "seed-provenance gate OK; all seeds carry PROVENANCE + matching SHA-256"
else
    rc=$?
    fail "seed_provenance" "${rc}" \
        "seed-provenance gate failed; see ${seed_log} and ${SEED_PROVENANCE_REPORT}" \
        "{\"exit_code\": ${rc}, \"log\": \"${seed_log}\", \"report\": \"${SEED_PROVENANCE_REPORT}\"}"
fi

# ---- Phase 1: poll until make_dataset.sh exits ----
while kill -0 "${BUILD_PID}" 2>/dev/null; do
    triples_n=$(ls _v19_core_build/triples/ 2>/dev/null | wc -l | tr -d ' ')
    log "build still running; triples=${triples_n}"
    sleep 60
done
log "build process ${BUILD_PID} has exited"

# ---- Phase 2: validate raw json ----
if [[ ! -f "${RAW_JSON}" ]]; then
    fail "build" 1 "raw json ${RAW_JSON} missing; see _v19_core_build/build.log" \
        "\"raw json not produced; check _v19_core_build/build.log\""
fi
raw_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_JSON}'))))" \
           2>/dev/null || echo "0")
log "build OK; raw rows=${raw_rows}"

# ---- Phase 3: [v18 chain pivot carry] TAA.CANON merge dropped; seed RAW_MERGED_JSON ----
canon_rows=0
log "[skip] TAA.CANON merge -- v18 chain pivot; copying raw -> raw_merged..."
python3 - <<EOF || fail "raw_merged_init" 1 "init failed" "{\"raw_rows\": ${raw_rows}}"
import json
raw = json.load(open("${RAW_JSON}"))
with open("${RAW_MERGED_JSON}", "w") as f:
    json.dump(raw, f, indent=2)
print(f"[init] raw_merged={len(raw):,} (no TAA.CANON merge)")
EOF
merged_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_MERGED_JSON}'))))" \
              2>/dev/null || echo "0")
log "raw_merged init OK; merged rows=${merged_rows}"

# ---- Phase 3b: CM.* generator merge (v18 carry) ----
CM_SEED="_v19_core_build/cm_seed.json"
log "running cm_generator.py (targets: crypto=${CM_TARGET_CRYPTO}, access=${CM_TARGET_ACCESS}, compliance=${CM_TARGET_COMPLIANCE}, gov=${CM_TARGET_GOV})..."
cm_log="_v19_core_build/cm.log"
if python tmpl_gen/scripts/cm_generator.py \
        --output "${CM_SEED}" \
        --report "${CM_REPORT}" \
        --target-crypto "${CM_TARGET_CRYPTO}" \
        --target-access "${CM_TARGET_ACCESS}" \
        --target-compliance "${CM_TARGET_COMPLIANCE}" \
        --target-gov "${CM_TARGET_GOV}" \
        --seed "${SHUFFLE_SEED}" \
        > "${cm_log}" 2>&1; then
    cm_rows=$(python3 -c "import json; print(len(json.load(open('${CM_SEED}'))))" \
              2>/dev/null || echo "0")
    log "cm_generator OK; generated rows=${cm_rows}"
else
    rc=$?
    fail "cm" "${rc}" "cm_generator.py exit ${rc}; see ${cm_log}" \
        "{\"merged_rows\": ${merged_rows}, \"exit_code\": ${rc}, \"log\": \"${cm_log}\"}"
fi

log "appending CM generator output to raw_merged json..."
python3 - <<EOF || fail "cm_merge" 1 "merge failed" "{\"merged_rows\": ${merged_rows}, \"cm_rows\": ${cm_rows}}"
import json
prev = json.load(open("${RAW_MERGED_JSON}"))
cm = json.load(open("${CM_SEED}"))
combined = prev + cm
with open("${RAW_MERGED_JSON}", "w") as f:
    json.dump(combined, f, indent=2)
print(f"[merge] prev={len(prev):,} + cm={len(cm):,} -> total={len(combined):,}")
EOF
merged_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_MERGED_JSON}'))))" \
              2>/dev/null || echo "0")
log "cm merge OK; merged rows=${merged_rows}"


# ---- Phase 3c: [SKIPPED v18.1/v19] MCQ.EXT.* generator merge ----
# v18 ran mcq_generator.py with --target-mitre 2000 --target-sec 2000
# --target-gloss 1500 producing 5478 rows (61% of the v18-Core MCQ axis).
# Per v18_1_plan.txt the axis-CKT regression (CKT 62.6 vs v8small 77.6) was
# traced to those KB-flashcard rows displacing the scenario MCQ shape that
# AthenaBench actually evaluates. v18.1 (carried into v19) reverts to
# scenario-only AB.MCQ.{1..6} + JS.MCQ.{1,2,5} (already present in
# raw_merged from Phase 1 via the v19_core.txt manifest); no extra merge.
mcq_rows=0
log "[skip] MCQ.EXT.* generator merge -- v18.1/v19 scenario-only MCQ recipe per v19_plan.txt"


# ---- Phase 3d: SOC.*.GEN.* generator merge (v18 carry) ----
SOC_SEED="_v19_core_build/soc_seed.json"
log "running soc_generator.py (targets: sigma=${SOC_TARGET_SIGMA}, mal=${SOC_TARGET_MAL}, ir=${SOC_TARGET_IR}, triage=${SOC_TARGET_TRIAGE})..."
soc_log="_v19_core_build/soc.log"
if python tmpl_gen/scripts/soc_generator.py \
        --output "${SOC_SEED}" \
        --report "${SOC_REPORT}" \
        --target-sigma "${SOC_TARGET_SIGMA}" \
        --target-mal "${SOC_TARGET_MAL}" \
        --target-ir "${SOC_TARGET_IR}" \
        --target-triage "${SOC_TARGET_TRIAGE}" \
        --seed "${SHUFFLE_SEED}" \
        > "${soc_log}" 2>&1; then
    soc_rows=$(python3 -c "import json; print(len(json.load(open('${SOC_SEED}'))))" \
               2>/dev/null || echo "0")
    log "soc_generator OK; generated rows=${soc_rows}"
else
    rc=$?
    fail "soc" "${rc}" "soc_generator.py exit ${rc}; see ${soc_log}" \
        "{\"merged_rows\": ${merged_rows}, \"exit_code\": ${rc}, \"log\": \"${soc_log}\"}"
fi

log "appending SOC generator output to raw_merged json..."
python3 - <<EOF || fail "soc_merge" 1 "merge failed" "{\"merged_rows\": ${merged_rows}, \"soc_rows\": ${soc_rows}}"
import json
prev = json.load(open("${RAW_MERGED_JSON}"))
soc = json.load(open("${SOC_SEED}"))
combined = prev + soc
with open("${RAW_MERGED_JSON}", "w") as f:
    json.dump(combined, f, indent=2)
print(f"[merge] prev={len(prev):,} + soc={len(soc):,} -> total={len(combined):,}")
EOF
merged_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_MERGED_JSON}'))))" \
              2>/dev/null || echo "0")
log "soc merge OK; merged rows=${merged_rows}"


# ---- Phase 4: TAA actor-balance with total cap (v18 carry) ----
log "running taa_actor_balance.py (per-actor=${ACTOR_CAP}, floor=${ACTOR_FLOOR}, family-total=${TAA_TOTAL_CAP})..."
balance_log="_v19_core_build/balance.log"
if python tmpl_gen/scripts/taa_actor_balance.py \
        --input "${RAW_MERGED_JSON}" --output "${BALANCED_JSON}" \
        --max-per-actor "${ACTOR_CAP}" --min-actors "${ACTOR_FLOOR}" \
        --max-rows-per-family-total "${TAA_TOTAL_CAP}" \
        > "${balance_log}" 2>&1; then
    balanced_rows=$(python3 -c "import json; print(len(json.load(open('${BALANCED_JSON}'))))" \
                    2>/dev/null || echo "0")
    log "actor-balance OK; balanced rows=${balanced_rows}"
else
    rc=$?
    fail "actor_balance" "${rc}" "actor floor not met (rc=${rc}); see ${balance_log}" \
        "{\"merged_rows\": ${merged_rows}, \"exit_code\": ${rc}, \"log\": \"${balance_log}\"}"
fi

# ---- Phase 5: dedup against eval sets (v18 carry) ----
log "running dedup_against_evals.py (n=13, hit=${DEDUP_HIT_THRESHOLD}, drop>=${DEDUP_DROP_THRESHOLD})..."
dedup_log="_v19_core_build/dedup.log"
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


# ---- Phase 6: row-count gate (v19 thresholds carried from v18.1; MCQ 6000 / RMS 12000 / VSP 5400) ----
log "running check_corpus_row_counts.py against ${ROW_GATE_PLAN}..."
gate_log="_v19_core_build/row_count_gate.log"
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

# ---- Phase 6b: licence-allowlist gate (v18 carry) ----
log "running check_corpus_licences.py..."
licence_log="_v19_core_build/licence_gate.log"
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

# ---- Phase 7: stratified shuffle (v18 carry) ----
log "running stratified_shuffle.py (seed=${SHUFFLE_SEED})..."
shuffle_log="_v19_core_build/shuffle.log"
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

# ---- Phase 8: val/train split (v19 paths) ----
log "running build_val_slice.py (per-axis=${VAL_PER_AXIS}, seed=${SHUFFLE_SEED})..."
val_log="_v19_core_build/val_slice.log"
if python _v19_core_build/build_val_slice.py \
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

# ---- Phase 9: per-phase corpus split: TWO shards (broad + axis) ----
log "running split_corpus_for_phases.py (two-shard mode; --out-taa-canon omitted)..."
split_log="_v19_core_build/phase_split.log"
if python tmpl_gen/scripts/split_corpus_for_phases.py \
        --input "${TRAIN_JSON}" \
        --val "${VAL_JSON}" \
        --out-broad           "${PHASE_A_JSON}" \
        --out-rms-ate-vsp-rcm "${PHASE_B_JSON}" \
        --report "${PHASE_SPLIT_REPORT}" \
        > "${split_log}" 2>&1; then
    phase_a_rows=$(python3 -c "import json; print(len(json.load(open('${PHASE_A_JSON}'))))" \
                   2>/dev/null || echo "0")
    phase_b_rows=$(python3 -c "import json; print(len(json.load(open('${PHASE_B_JSON}'))))" \
                   2>/dev/null || echo "0")
    log "phase split OK; phase_a=${phase_a_rows} phase_b=${phase_b_rows}"
else
    rc=$?
    fail "phase_split" "${rc}" "split_corpus_for_phases.py exit ${rc}; see ${split_log}" \
        "{\"train_rows\": ${train_rows}, \"exit_code\": ${rc}, \"log\": \"${split_log}\"}"
fi

# ---- Done ----
notify "v19-core build done" "raw=${raw_rows} merged=${merged_rows} clean=${clean_rows} train=${train_rows} (phase_a=${phase_a_rows} phase_b=${phase_b_rows})"
write_status "phase_split" "ok" \
    "{\"raw_rows\": ${raw_rows}, \"cm_seed_rows\": ${cm_rows:-0}, \"mcq_seed_rows\": ${mcq_rows:-0}, \"soc_seed_rows\": ${soc_rows:-0}, \"merged_rows\": ${merged_rows}, \"balanced_rows\": ${balanced_rows}, \"clean_rows\": ${clean_rows}, \"shuffled_rows\": ${shuffled_rows}, \"val_rows\": ${val_rows}, \"train_rows\": ${train_rows}, \"phase_a_rows\": ${phase_a_rows}, \"phase_b_rows\": ${phase_b_rows}, \"clean_path\": \"${CLEAN_JSON}\", \"shuffled_path\": \"${SHUFFLED_JSON}\", \"val_path\": \"${VAL_JSON}\", \"train_path\": \"${TRAIN_JSON}\", \"phase_a_path\": \"${PHASE_A_JSON}\", \"phase_b_path\": \"${PHASE_B_JSON}\", \"actor_cap\": ${ACTOR_CAP}, \"actor_floor\": ${ACTOR_FLOOR}, \"taa_total_cap\": ${TAA_TOTAL_CAP}, \"dedup_hit_threshold\": ${DEDUP_HIT_THRESHOLD}, \"dedup_drop_threshold\": ${DEDUP_DROP_THRESHOLD}, \"shuffle_seed\": ${SHUFFLE_SEED}, \"val_per_axis\": ${VAL_PER_AXIS}, \"reports\": {\"substrate\": \"${SUBSTRATE_REPORT}\", \"seed_provenance\": \"${SEED_PROVENANCE_REPORT}\", \"cm\": \"${CM_REPORT}\", \"soc\": \"${SOC_REPORT}\", \"dedup\": \"${DEDUP_REPORT}\", \"row_count_gate\": \"${ROW_GATE_REPORT}\", \"licence_gate\": \"${LICENCE_GATE_REPORT}\", \"phase_split\": \"${PHASE_SPLIT_REPORT}\"}}"

log "watcher complete"
exit 0
