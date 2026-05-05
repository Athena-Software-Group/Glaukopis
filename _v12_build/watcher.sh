#!/bin/bash
# v12 build watcher: branched from _v11_build/watcher.sh per
# tmpl_gen/templates/05052026/v12_plan.txt §4.
#
# Pipeline (v11 phases + 5 new gates):
#   Phase 1 -- poll make_dataset.sh PID until exit (raw json produced)
#   Phase 2 -- validate raw json exists                                (v11)
#   Phase 3 -- TAA.CANON generator merge into raw json                 (NEW v12)
#   Phase 3b -- CM.* generator merge into raw json                     (NEW v12)
#   Phase 3c -- MCQ.EXT.* generator merge into raw json                (NEW v12)
#   Phase 3d -- SOC.*.GEN.* generator merge into raw json              (NEW v12)
#   Phase 4 -- TAA actor-balance with --max-rows-per-family-total 3500 (v12 §4.3)
#   Phase 5 -- dedup against eval sets                                 (v11)
#   Phase 6 -- row-count gate (v12 §4.1)                               (NEW v12)
#   Phase 7 -- stratified shuffle (v12 §4.2)                           (NEW v12)
#   Phase 8 -- val/train split (build_val_slice.py)                    (v11 carry)
#   Phase 9 -- per-phase corpus split (split_corpus_for_phases.py)     (NEW v12)
#
# Status is appended to _v12_build/watcher.log; final outcome lands in
# _v12_build/watcher_status.json. Each new gate writes its own report json
# under _v12_build/.
#
# Build-side defaults inherited from v11 (gencfg_per_primary_neo4j.json):
#   per_primary_grouping=true, allow_nullprops=true.
#
# Launch with nohup so it survives terminal close:
#   nohup bash _v12_build/watcher.sh > _v12_build/watcher.log 2>&1 &

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

BUILD_PID="$(cat _v12_build/build.pid 2>/dev/null | sed 's/PID=//')"
RAW_JSON="SFT/data/ift_data_2026_05_05_v12.raw.json"
RAW_MERGED_JSON="SFT/data/ift_data_2026_05_05_v12.raw_merged.json"
BALANCED_JSON="SFT/data/ift_data_2026_05_05_v12.balanced.json"
CLEAN_JSON="SFT/data/ift_data_2026_05_05_v12.json"
SHUFFLED_JSON="SFT/data/ift_data_2026_05_05_v12.shuffled.json"
VAL_JSON="SFT/data/ift_data_2026_05_05_v12_val.json"
TRAIN_JSON="SFT/data/ift_data_2026_05_05_v12_train.json"
PHASE_BROAD_JSON="SFT/data/ift_data_2026_05_05_v12_broad.json"
PHASE_RAVR_JSON="SFT/data/ift_data_2026_05_05_v12_rms_ate_vsp_rcm.json"
PHASE_TAACANON_JSON="SFT/data/ift_data_2026_05_05_v12_taa_canon.json"
DEDUP_REPORT="_v12_build/dedup_report.json"
ROW_GATE_REPORT="_v12_build/row_count_gate_report.json"
PHASE_SPLIT_REPORT="_v12_build/phase_split_report.json"
TAA_CANON_REPORT="_v12_build/taa_canon_report.json"
CM_REPORT="_v12_build/cm_report.json"
MCQ_REPORT="_v12_build/mcq_report.json"
SOC_REPORT="_v12_build/soc_report.json"
STATUS_JSON="_v12_build/watcher_status.json"

# Tunables (v12 plan §4.3 defaults)
ACTOR_CAP=40
ACTOR_FLOOR=100
TAA_TOTAL_CAP=3500
DEDUP_HIT_THRESHOLD=1
DEDUP_DROP_THRESHOLD=50
ROW_GATE_PLAN="tmpl_gen/templates/05052026/v12_row_count_gate.json"
TAA_CANON_TARGET_1=3500
TAA_CANON_TARGET_2=3500
TAA_CANON_TARGET_3=3000
CM_TARGET_CRYPTO=1500
CM_TARGET_ACCESS=1500
CM_TARGET_COMPLIANCE=2000
CM_TARGET_GOV=1000
MCQ_TARGET_MITRE=1500
MCQ_TARGET_SEC=1500
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
    notify "v12 build FAILED" "${msg}"
    write_status "${stage}" "fail" "${detail}"
    exit "${rc}"
}

if [[ -z "${BUILD_PID}" ]]; then
    log "ERROR: _v12_build/build.pid not found; nothing to watch."
    write_status "init" "fail" "\"build.pid missing\""
    exit 1
fi

log "watcher starting; build_pid=${BUILD_PID}"
notify "v12 build" "watcher started; tracking PID ${BUILD_PID}"

# ---- Phase 1: poll until make_dataset.sh exits ----
while kill -0 "${BUILD_PID}" 2>/dev/null; do
    triples_n=$(ls _v12_build/triples/ 2>/dev/null | wc -l | tr -d ' ')
    log "build still running; triples=${triples_n}"
    sleep 60
done
log "build process ${BUILD_PID} has exited"

# ---- Phase 2: validate raw json ----
if [[ ! -f "${RAW_JSON}" ]]; then
    fail "build" 1 "raw json ${RAW_JSON} missing; see _v12_build/build.log" \
        "\"raw json not produced; check _v12_build/build.log\""
fi
raw_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_JSON}'))))" \
           2>/dev/null || echo "0")
log "build OK; raw rows=${raw_rows}"

# ---- Phase 3: TAA.CANON generator merge (NEW v12) ----
# Generates the ~10K-row TAA.CANON.{1,2,3} expansion directly from MITRE
# enterprise-attack.json + Athena aliases.csv (bypasses the v11 187-row
# Neo4j ceiling on intrusion-set anchors), then merges the rows into the
# raw json before actor-balance / dedup. The generator output is shuffled
# with the rest of the corpus by Phase 7.
TAA_CANON_SEED="_v12_build/taa_canon_seed.json"
log "running taa_canon_generator.py (targets: ${TAA_CANON_TARGET_1}/${TAA_CANON_TARGET_2}/${TAA_CANON_TARGET_3})..."
canon_log="_v12_build/taa_canon.log"
if python tmpl_gen/scripts/taa_canon_generator.py \
        --mitre cpt/cache/raw/mitre_attack_enterprise/enterprise-attack.json \
        --athena-aliases SFT/test/benchmark_data/athena_bench/athena_taa/aliases.csv \
        --output "${TAA_CANON_SEED}" \
        --report "${TAA_CANON_REPORT}" \
        --target-canon1 "${TAA_CANON_TARGET_1}" \
        --target-canon2 "${TAA_CANON_TARGET_2}" \
        --target-canon3 "${TAA_CANON_TARGET_3}" \
        --seed "${SHUFFLE_SEED}" \
        > "${canon_log}" 2>&1; then
    canon_rows=$(python3 -c "import json; print(len(json.load(open('${TAA_CANON_SEED}'))))" \
                 2>/dev/null || echo "0")
    log "taa_canon_generator OK; generated rows=${canon_rows}"
else
    rc=$?
    fail "taa_canon" "${rc}" "taa_canon_generator.py exit ${rc}; see ${canon_log}" \
        "{\"raw_rows\": ${raw_rows}, \"exit_code\": ${rc}, \"log\": \"${canon_log}\"}"
fi

# Merge generator output into raw json (concatenation; downstream dedup
# will collapse any (instruction,input,output) collisions).
log "merging TAA.CANON generator output into raw json..."
python3 - <<EOF || fail "taa_canon_merge" 1 "merge failed" "{\"raw_rows\": ${raw_rows}, \"canon_rows\": ${canon_rows}}"
import json
raw = json.load(open("${RAW_JSON}"))
canon = json.load(open("${TAA_CANON_SEED}"))
merged = raw + canon
with open("${RAW_MERGED_JSON}", "w") as f:
    json.dump(merged, f, indent=2)
print(f"[merge] raw={len(raw):,} + canon={len(canon):,} -> merged={len(merged):,}")
EOF
merged_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_MERGED_JSON}'))))" \
              2>/dev/null || echo "0")
log "merge OK; merged rows=${merged_rows}"

# ---- Phase 3b: CM.* generator merge (NEW v12) ----
# Generates the ~6K-row CM.{CRYPTO,ACCESS,COMPLIANCE,GOV} corpus from the
# curated knowledge tables in tmpl_gen/scripts/cm_data/. The CM.* manifest
# entries declare Source: external; the generator output is the actual
# source. Rows are appended to the raw_merged json before actor-balance.
CM_SEED="_v12_build/cm_seed.json"
log "running cm_generator.py (targets: crypto=${CM_TARGET_CRYPTO}, access=${CM_TARGET_ACCESS}, compliance=${CM_TARGET_COMPLIANCE}, gov=${CM_TARGET_GOV})..."
cm_log="_v12_build/cm.log"
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

# ---- Phase 3c: MCQ.EXT.* generator merge (NEW v12) ----
# Generates the ~3K-row AB.MCQ.EXT.{MITRE,SEC}.1 corpus from curated
# knowledge tables in tmpl_gen/scripts/mcq_data/. Complements the
# template-driven AB.MCQ.* / JS.MCQ.* families which saturate at ~3K
# rows of distinct MITRE-anchor combinations. Rows are appended to the
# raw_merged json before actor-balance.
MCQ_SEED="_v12_build/mcq_seed.json"
log "running mcq_generator.py (targets: mitre=${MCQ_TARGET_MITRE}, sec=${MCQ_TARGET_SEC})..."
mcq_log="_v12_build/mcq.log"
if python tmpl_gen/scripts/mcq_generator.py \
        --output "${MCQ_SEED}" \
        --report "${MCQ_REPORT}" \
        --target-mitre "${MCQ_TARGET_MITRE}" \
        --target-sec "${MCQ_TARGET_SEC}" \
        --seed "${SHUFFLE_SEED}" \
        > "${mcq_log}" 2>&1; then
    mcq_rows=$(python3 -c "import json; print(len(json.load(open('${MCQ_SEED}'))))" \
               2>/dev/null || echo "0")
    log "mcq_generator OK; generated rows=${mcq_rows}"
else
    rc=$?
    fail "mcq" "${rc}" "mcq_generator.py exit ${rc}; see ${mcq_log}" \
        "{\"merged_rows\": ${merged_rows}, \"exit_code\": ${rc}, \"log\": \"${mcq_log}\"}"
fi

log "appending MCQ generator output to raw_merged json..."
python3 - <<EOF || fail "mcq_merge" 1 "merge failed" "{\"merged_rows\": ${merged_rows}, \"mcq_rows\": ${mcq_rows}}"
import json
prev = json.load(open("${RAW_MERGED_JSON}"))
mcq = json.load(open("${MCQ_SEED}"))
combined = prev + mcq
with open("${RAW_MERGED_JSON}", "w") as f:
    json.dump(combined, f, indent=2)
print(f"[merge] prev={len(prev):,} + mcq={len(mcq):,} -> total={len(combined):,}")
EOF
merged_rows=$(python3 -c "import json; print(len(json.load(open('${RAW_MERGED_JSON}'))))" \
              2>/dev/null || echo "0")
log "mcq merge OK; merged rows=${merged_rows}"

# ---- Phase 3d: SOC.*.GEN.* generator merge (NEW v12) ----
# Generates the ~5K-row SOC.{SIGMA,MAL,IR,TRIAGE}.GEN.1 corpus from
# curated knowledge tables in tmpl_gen/scripts/soc_data/. Complements
# the template-driven SOC.IR.* / SOC.MAL.* / SOC.SIGMA.* / SOC.TRIAGE.*
# families which bind to SigmaHQ-rule and malware-family Cypher seeds
# that saturate at ~5K rows in v12. Rows are appended to the raw_merged
# json before actor-balance.
SOC_SEED="_v12_build/soc_seed.json"
log "running soc_generator.py (targets: sigma=${SOC_TARGET_SIGMA}, mal=${SOC_TARGET_MAL}, ir=${SOC_TARGET_IR}, triage=${SOC_TARGET_TRIAGE})..."
soc_log="_v12_build/soc.log"
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

# ---- Phase 4: TAA actor-balance with total cap ----
log "running taa_actor_balance.py (per-actor=${ACTOR_CAP}, floor=${ACTOR_FLOOR}, family-total=${TAA_TOTAL_CAP})..."
balance_log="_v12_build/balance.log"
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

# ---- Phase 5: dedup against eval sets ----
log "running dedup_against_evals.py (n=13, hit=${DEDUP_HIT_THRESHOLD}, drop>=${DEDUP_DROP_THRESHOLD})..."
dedup_log="_v12_build/dedup.log"
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

# ---- Phase 6: row-count gate (NEW v12) ----
log "running check_corpus_row_counts.py against ${ROW_GATE_PLAN}..."
gate_log="_v12_build/row_count_gate.log"
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

# ---- Phase 7: stratified shuffle (NEW v12) ----
# Stride-interleaves rows by shortname so any window of ~N/k_f positions
# contains every family at least once. Eliminates the v11 unshuffled-tail
# regression where the last decile was 100% TAA.* / SOC.* contiguous.
log "running stratified_shuffle.py (seed=${SHUFFLE_SEED})..."
shuffle_log="_v12_build/shuffle.log"
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

# ---- Phase 8: val/train split ----
log "running build_val_slice.py (per-axis=${VAL_PER_AXIS}, seed=${SHUFFLE_SEED})..."
val_log="_v12_build/val_slice.log"
if python _v12_build/build_val_slice.py \
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

# ---- Phase 9: per-phase corpus split (NEW v12) ----
# Splits the train corpus into three disjoint shards consumed by the
# Phase A/B/C launcher (run_sft_qwen25_14b_v12.sh).
log "running split_corpus_for_phases.py..."
split_log="_v12_build/phase_split.log"
if python tmpl_gen/scripts/split_corpus_for_phases.py \
        --input "${TRAIN_JSON}" \
        --val "${VAL_JSON}" \
        --out-broad "${PHASE_BROAD_JSON}" \
        --out-rms-ate-vsp-rcm "${PHASE_RAVR_JSON}" \
        --out-taa-canon "${PHASE_TAACANON_JSON}" \
        --report "${PHASE_SPLIT_REPORT}" \
        > "${split_log}" 2>&1; then
    broad_rows=$(python3 -c "import json; print(len(json.load(open('${PHASE_BROAD_JSON}'))))" \
                 2>/dev/null || echo "0")
    ravr_rows=$(python3 -c "import json; print(len(json.load(open('${PHASE_RAVR_JSON}'))))" \
                2>/dev/null || echo "0")
    canon_rows=$(python3 -c "import json; print(len(json.load(open('${PHASE_TAACANON_JSON}'))))" \
                 2>/dev/null || echo "0")
    log "phase split OK; broad=${broad_rows} rms_ate_vsp_rcm=${ravr_rows} taa_canon=${canon_rows}"
else
    rc=$?
    fail "phase_split" "${rc}" "split_corpus_for_phases.py exit ${rc}; see ${split_log}" \
        "{\"train_rows\": ${train_rows}, \"exit_code\": ${rc}, \"log\": \"${split_log}\"}"
fi

# ---- Done ----
notify "v12 build done" "raw=${raw_rows} merged=${merged_rows} clean=${clean_rows} train=${train_rows} (broad=${broad_rows} B=${ravr_rows} C=${canon_rows})"
write_status "phase_split" "ok" \
    "{\"raw_rows\": ${raw_rows}, \"canon_seed_rows\": ${canon_rows:-0}, \"cm_seed_rows\": ${cm_rows:-0}, \"mcq_seed_rows\": ${mcq_rows:-0}, \"soc_seed_rows\": ${soc_rows:-0}, \"merged_rows\": ${merged_rows}, \"balanced_rows\": ${balanced_rows}, \"clean_rows\": ${clean_rows}, \"shuffled_rows\": ${shuffled_rows}, \"val_rows\": ${val_rows}, \"train_rows\": ${train_rows}, \"phase_a_broad_rows\": ${broad_rows}, \"phase_b_rms_ate_vsp_rcm_rows\": ${ravr_rows}, \"phase_c_taa_canon_rows\": ${canon_rows}, \"clean_path\": \"${CLEAN_JSON}\", \"shuffled_path\": \"${SHUFFLED_JSON}\", \"val_path\": \"${VAL_JSON}\", \"train_path\": \"${TRAIN_JSON}\", \"phase_a_path\": \"${PHASE_BROAD_JSON}\", \"phase_b_path\": \"${PHASE_RAVR_JSON}\", \"phase_c_path\": \"${PHASE_TAACANON_JSON}\", \"actor_cap\": ${ACTOR_CAP}, \"actor_floor\": ${ACTOR_FLOOR}, \"taa_total_cap\": ${TAA_TOTAL_CAP}, \"dedup_hit_threshold\": ${DEDUP_HIT_THRESHOLD}, \"dedup_drop_threshold\": ${DEDUP_DROP_THRESHOLD}, \"shuffle_seed\": ${SHUFFLE_SEED}, \"val_per_axis\": ${VAL_PER_AXIS}, \"reports\": {\"taa_canon\": \"${TAA_CANON_REPORT}\", \"cm\": \"${CM_REPORT}\", \"mcq\": \"${MCQ_REPORT}\", \"soc\": \"${SOC_REPORT}\", \"dedup\": \"${DEDUP_REPORT}\", \"row_count_gate\": \"${ROW_GATE_REPORT}\", \"phase_split\": \"${PHASE_SPLIT_REPORT}\"}}"

log "watcher complete"
exit 0
