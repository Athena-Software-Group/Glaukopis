# Sophia CTI Templates — v18 (May 13, 2026 vintage)

v18 is the **standalone three-phase SFT rebuild** that follows the
v17.1 chained-SFT post-mortem. Where v17.1 chained a CSE specialist
shard onto v16 (recovering CSE-TI / CSE-Malware), v18 returns to the
v12 standalone recipe — full-axis manifest, 3-phase training off
`Qwen/Qwen2.5-14B-Instruct` — but with the corpus evolved to lift the
two AthenaBench axes that refused to move across the v12 → v17.1 line:
**CKT (MCQ)** and **ATE**. The branch produces one HF checkpoint:

| checkpoint | answers the question |
|---|---|
| `asg-ai/athena-cti-sft-qwen25-14b-v18` | Does evolving the v12 manifest with (a) lifted MCQ counts on AB.MCQ.{1,2,4,5,6}, (b) a third generator family AB.MCQ.EXT.GLOSS.1 sourced from public CTI/sec glossaries, and (c) three new ATE templates that bind to SigmaRule, malware, and intrusion-set narratives lift CKT to v8/8B-parity (≥77.6) and ATE to ≥61.0 without regressing RMS / VSP / RCM / TAA-attribution / SOC / CM? |

The vintage directory is self-contained per project convention:

```
05132026/
  Sophia-CTI-Templates-v18.txt    self-contained full-axis manifest;
                                  body byte-identical to v12 except
                                  for (a) AB.MCQ.{1,2,4,5,6} Count
                                  lifts, (b) AB.MCQ.EXT.{MITRE,SEC}.1
                                  Count lifts, and (c) two new
                                  appended sections at EOF:
                                    "v18 NEW BLOCK -- AB.MCQ.EXT.GLOSS.1"
                                    "v18 NEW BLOCK -- AB.ATE.{9..11} +
                                                       JS.ATE.{4..6}"
  v18_plan.txt                    master plan (motivation in §1, deltas
                                  in §2, row-count plan in §3, training
                                  recipe in §4, falsification + sign-off
                                  in §5/§6)
  v18_row_count_gate.json         per-axis REJECT_IF_BELOW thresholds;
                                  MCQ floor 5400 → 8100; ATE floor
                                  9000 → 13500; all other axes carry
                                  v12 floors verbatim
  README.md                       this document
```

Local build artefacts and helpers live in `_v18_build/`:

```
_v18_build/
  _neo4j_check.py        Phase 0 substrate validator (entity floors +
                         traversal floors against athena-cti-db; exits
                         non-zero so the watcher halts before
                         make_dataset.sh starts)
  _neo4j_extras.py       throwaway diagnostic; relationship-direction
                         enumeration used during template authoring
  _neo4j_probe_v2.py     throwaway diagnostic; field-population probe
                         (attack-pattern keys, malware keys, etc.)
  _neo4j_probe_v3.py     throwaway diagnostic; substrate validation
                         for AB.ATE.{9,10,11} alternative bindings
  watcher.sh             9-phase post-build pipeline (substrate gate,
                         seed-provenance gate, generator merges,
                         row-count gate, licence gate, dedup,
                         stratified shuffle, val/train split, three-
                         shard phase split)
```

## 1. Why v18 exists — the CKT/ATE plateau across v12 → v17.1

Two AthenaBench axes refused to lift across the entire v12..v17.1 line:

| axis      | v12  | v16  | v17.1 | athena_v8 (8B) | v18 target |
|-----------|------|------|-------|----------------|------------|
| CKT (MCQ) | 70.4 | 72.1 | 70.0  | **77.6**       | 77.6       |
| ATE       | 55.1 | 56.9 | 56.6  | 50.6           | **61.0**   |

The v8 LLaMA-3.1 8B abaligned model hit CKT 77.6 with ~6,000 rows of MCQ
supervision and a single training pass. The Qwen2.5-14B v12 line carries
6,100 MCQ rows but lands ~7 pp below v8. The diagnosis is two correlated
effects:

- **Distractor quality.** v12 MCQ distractors are drawn unconstrained
  from the catalog (e.g. AB.MCQ.5 negative options are "any
  attack-pattern"). The Athena MCQ eval distractors are usually siblings
  — same tactic, same technique class, etc. The 14B model rejects the
  obviously-wrong v12 negatives easily; its loss on training rows is
  small while its loss on eval rows (with closer negatives) stays high.
- **Volume.** v12 has 6.1K MCQ rows; v8 had ~6K but the underlying
  benchmark is ~6,000 questions wide, so the per-question gradient
  density was effectively higher because v8's training surface was a
  narrow corpus and v12's is a 260K-row mixed-axis corpus.

ATE has the same shape: v12 carries 10.4K rows from 8 sub-templates
(AB.ATE.{1..8} + JS.ATE.{1..3}) but the templates render most of their
volume from `attack-pattern.description` verbatim (AB.ATE.{1,2,4,5} all
bind on the same description column). The gradient is a single anchor
seen from five angles; the eval set probes ~11 distinct narrative shapes.

## 2. v12 → v18 deltas (manifest only)

1. **AB.MCQ.{1,2,5}**: `Count: 800 → 1200` (3 of 6 sub-families;
   the others bind on smaller anchor pools and would saturate)
2. **AB.MCQ.{4,6}**: `Count: 250 → 500`
3. **AB.MCQ.EXT.{MITRE,SEC}.1**: target 1500 → 2000 each
4. **NEW** `AB.MCQ.EXT.GLOSS.1` (~1500 rows; generator-only, sourced
   from public CTI/sec glossaries: NIST SP 800-150, MITRE ATT&CK
   glossary, CISA Stop-Ransomware, ENISA Threat Landscape, ISO/IEC
   27000:2018) — knowledge table at
   `tmpl_gen/scripts/mcq_data/glossary.py`
5. **NEW** `AB.ATE.{9,10,11}` + `JS.ATE.{4,5,6}` Cypher templates:
   - `.9` SigmaRule.title + .description → `sr.detects>attack-pattern`
   - `.10` malware.description → `mw.uses>attack-pattern` (anchored on
     malware, not on attack-pattern as in AB.ATE.8)
   - `.11` intrusion-set.description → `grp.uses>attack-pattern`
     (anchored on group, not on attack-pattern as in AB.ATE.3)
6. Row-count gate floors lifted: MCQ 5400 → 8100; ATE 9000 → 13500.
   All other axis floors carry v12 thresholds verbatim. See
   `v18_row_count_gate.json`.
7. Build watcher Phase 0 augmented with `_v18_build/_neo4j_check.py`
   substrate validation. Surfaces athena-cti-db drift at build time.

The training recipe is the v12 three-phase recipe verbatim (Phase A
broad re-anchor 1e-5; Phase B RMS/ATE/VSP/RCM drill 5e-6; Phase C
TAA.CANON memorisation 3e-6). See `v18_plan.txt §4` for the full
hyperparameter table and `SFT/autotrain/run_sft_qwen25_14b_v18.sh` for
the launcher.

## 3. Substrate validation (Phase 0)

`_v18_build/_neo4j_check.py` was authored against the actual
`athena-cti-db` content (probed 2026-05-13). Key findings that shaped
the manifest:

- `attack-pattern.x_mitre_procedure_examples` is **not populated** in
  this DB. The initial AB.ATE.9 draft used this property; rewritten to
  bind on the SigmaRule narrative instead.
- `x-mitre-data-component` has **no outgoing `:detects` edges** to
  attack-pattern in this DB (the edge name appears as `:requires_data`
  *to* the data-component node from `x-mitre-analytic`). The initial
  AB.ATE.11 draft used a `dc.detects>attack-pattern` traversal that
  returns 0 rows; rewritten to bind on the intrusion-set narrative.
- `tactic->achieves->technique` is reversed in this DB: the actual edge
  is `attack-pattern->achieves->x-mitre-tactic` (the v12 templates
  already use the correct direction).

All v18-critical floors pass on the current DB:

```
intrusion-set->uses->attack-pattern  4362  (floor 4000)  OK
malware->uses->attack-pattern        9836  (floor 8000)  OK
SigmaRule->detects->attack-pattern   3742  (floor 3000)  OK
intrusion-set with description >100  185   (floor  150)  OK
malware with description >80  &&     690   (floor  500)  OK
  uses>=1
SigmaRule with description >30       3116  (floor 2500)  OK
intrusion-set with aliases populated 187   (floor  150)  OK
course-of-action->mitigates->ap      1445  (floor 1000)  OK
attack-pattern->subtechnique-of->ap   477  (floor  400)  OK
attack-pattern->achieves->tactic     1089  (floor  800)  OK
```

## 4. Run-book

### 4.1 Generation

```bash
cd /Users/pietro/code/Glaukopis     # or the cluster equivalent
# Phase 0 substrate gate (runs inside watcher; included here for
# manual smoke-test before kicking off the long build)
python _v18_build/_neo4j_check.py

mkdir -p _v18_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05132026/Sophia-CTI-Templates-v18.txt \
     _v18_build/triples \
     SFT/data/ift_data_2026_05_13_v18.raw.json \
     2500 3500 > _v18_build/build.log 2>&1 &
echo "PID=$!" > _v18_build/build.pid

nohup bash _v18_build/watcher.sh > _v18_build/watcher.log 2>&1 &
```

The watcher polls the build PID, then runs Phases 1–9: substrate
gate (re-run), seed-provenance gate, generator merges (TAA.CANON,
MCQ-EXT including the new GLOSS family, SOC.GEN), actor-balance,
dedup, row-count gate against `v18_row_count_gate.json`, licence
gate, stratified shuffle, val/train split, three-shard phase split.

Final outputs:

- `SFT/data/ift_data_2026_05_13_v18.shuffled.json` — stratified, deduped, val-excluded
- `SFT/data/ift_data_2026_05_13_v18_val.json` — held-out validation slice (350 rows = 50 × 7 axes)
- `SFT/data/ift_data_2026_05_13_v18_broad.json` — Phase A shard
- `SFT/data/ift_data_2026_05_13_v18_rms_ate_vsp_rcm.json` — Phase B shard
- `SFT/data/ift_data_2026_05_13_v18_taa_canon.json` — Phase C shard
- `_v18_build/watcher_status.json` — per-phase row counts and reports

### 4.2 Dataset registration

The launcher resolves dataset names through `SFT/data/dataset_info.json`.
The v18 entries (`ift_data_2026_05_13_v18_broad`,
`ift_data_2026_05_13_v18_rms_ate_vsp_rcm`,
`ift_data_2026_05_13_v18_taa_canon`,
`ift_data_2026_05_13_v18_val`) are registered alongside the v17.1 / v17
/ v16 / v15 / v12 entries.

### 4.3 Training

```bash
bash SFT/autotrain/run_sft_qwen25_14b_v18.sh
# defaults to ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18
# base model: Qwen/Qwen2.5-14B-Instruct  (NOT chained off v16/v17.1)
# wall-time estimate: Phase A ~4 h + Phase B ~3 h + Phase C ~1 h = ~8 h on 8xH100
```

### 4.4 Bench

Standard 14B AthenaBench + CyberMetric + CyberSOCEval sweep against the
v18 HF checkpoint. Compare against v12 (the standalone baseline v18
returns to) and v17.1 (the chained-SFT branch). Decision matrix in
`v18_plan.txt §5`. If v18 hits CKT ≥ 75 and ATE ≥ 58.5 without
regressing RMS / VSP / RCM / TAA-attr / SOC / CM by more than 2 pp,
the corpus-driven recipe is validated and v18 becomes the new shared
base for any future v19+ chained specialist branches.
