# Sophia CTI Templates — v18 (May 13, 2026 vintage; consolidated into 05112026/)

> Looking for the **v18.1 Core-only redo**? See `README-18-1.md` in this
> directory. v18.1 is the post-mortem rebuild of the v18 Core stage after
> v18 regressed CKT/RMS/VSP against the v8small / v9_rms / v10 historical
> peaks; v18+TAA and v18 (CSE) stages are unchanged and reused verbatim
> on top of the new v18.1-Core base.

v18 is the **v17.1-pattern chained SFT rebuild** that follows the
v17.1 chained-SFT post-mortem and the v15 W1 TAA-flavour audit.
Where v17.1 chained a single CSE specialist shard onto v16, v18
formalises the chain shape into a self-contained three-stage recipe
off `Qwen/Qwen2.5-14B-Instruct`:

1. **v18-Core** — v12-shape full-axis base (broad re-anchor + RMS /
   ATE / VSP / RCM catalog drill) with the corpus deltas in §2 to
   lift CKT (MCQ) and ATE.
2. **v18+TAA** — v16-shape TAA Classic refresher chained off
   v18-Core. TAA.CANON / MISP.CANON alias data is dropped from the
   v18 lineage entirely (the v15 W1 audit isolated CANON as the
   wrong TAA flavour for the AthenaBench TAA Classic axis).
3. **v18** — v17.1-shape CyberSOCEval letter-set drill chained off
   v18+TAA. Final published checkpoint.

The branch produces three HF checkpoints (only the final is the
published v18 model):

| stage | checkpoint | answers the question |
|---|---|---|
| 1 | `asg-ai/athena-cti-sft-qwen25-14b-v18-core` | Does evolving the v12 manifest with (a) lifted MCQ counts on AB.MCQ.{1,2,4,5,6}, (b) a third generator family AB.MCQ.EXT.GLOSS.1 sourced from public CTI/sec glossaries, and (c) three new ATE templates that bind to SigmaRule, malware, and intrusion-set narratives lift CKT to v8/8B-parity (≥77.6) and ATE to ≥61.0 without regressing RMS / VSP / RCM / SOC / CM? |
| 2 | `asg-ai/athena-cti-sft-qwen25-14b-v18-taa` | Does the v16 chained TAA Classic recipe reproduce on top of v18-Core (TAA-attr ≥ v12 + 2pp) without regressing the stage-1 axes? |
| 3 | `asg-ai/athena-cti-sft-qwen25-14b-v18-cse` | Does the v17.1 chained CSE recipe reproduce on top of v18+TAA (CSE-TI / CSE-MAL ≥ v17.1 − 2pp) while keeping all v18-Core / v18+TAA gains? |

The vintage directory is self-contained per project convention; only
the Core (stage 1) manifest lives here, because stages 2 and 3 reuse
the v16 (`05092026/`) and v17.1 (`05102026/`) manifests verbatim:

```
05112026/                         (originally authored as 05132026/; consolidated)
  Sophia-CTI-Templates-v18.txt    Core-shard manifest; body
                                  byte-identical to v12 except
                                  for (a) AB.MCQ.{1,2,4,5,6} Count
                                  lifts, (b) AB.MCQ.EXT.{MITRE,SEC}.1
                                  Count lifts, and (c) two new
                                  appended sections at EOF:
                                    "v18 NEW BLOCK -- AB.MCQ.EXT.GLOSS.1"
                                    "v18 NEW BLOCK -- AB.ATE.{9..11} +
                                                       JS.ATE.{4..6}"
  v18_plan.txt                    master plan (motivation in §1, deltas
                                  in §2, row-count plan in §3, chained
                                  training recipe in §4, falsification
                                  + sign-off in §5/§6)
  v18_row_count_gate.json         per-axis REJECT_IF_BELOW thresholds
                                  for the Core shard; MCQ floor
                                  5400 → 8100; ATE floor 9000 → 13500;
                                  all other axes carry v12 floors
                                  verbatim. Stage-2 / stage-3 floors
                                  are sourced from the v16 / v17.1
                                  vintage gate files respectively.
  README.md                       this document
```

Local build artefacts and helpers live in three sibling build dirs,
one per chain stage:

```
_v18_build/                Core (stage 1) build artefacts
  _neo4j_check.py          Phase 0 substrate validator (entity floors
                           + traversal floors against athena-cti-db;
                           exits non-zero so the watcher halts before
                           make_dataset.sh starts)
  _neo4j_extras.py         throwaway diagnostic; relationship-
                           direction enumeration used during template
                           authoring
  _neo4j_probe_v2.py       throwaway diagnostic; field-population
                           probe (attack-pattern keys, malware keys)
  _neo4j_probe_v3.py       throwaway diagnostic; substrate validation
                           for AB.ATE.{9,10,11} alternative bindings
  watcher.sh               post-build pipeline (substrate gate,
                           seed-provenance gate, generator merges,
                           row-count gate, licence gate, dedup,
                           stratified shuffle, val/train split,
                           two-shard phase split: broad + axis)
  validate_corpus.py       cross-stage end-to-end validator
                           (JSON well-formedness, required fields,
                           source allowlist, MCQ letter balance, ATE
                           lift, per-stage train/val disjointness)

_v18_taa_build/            v18+TAA (stage 2) build artefacts
  watcher.sh               TAA Classic post-build pipeline (forked
                           from _v16_build/watcher.sh; uses v16 row-
                           count gate)
  build_val_slice.py       per-axis val/train splitter (50 rows per
                           shortname; seed=42)

_v18_cse_build/            v18 (stage 3) build artefacts
  watcher.sh               CSE post-build pipeline (forked from
                           _v17_1_build/watcher.sh; uses v17.1 row-
                           count gate and the letter-balance gate)
  letter_balance_gate.py   asserts the CSE letter-tuple distribution
                           is not collapsed (forked from _v17_1_build)
  build_val_slice.py       per-axis val/train splitter (50 rows per
                           shortname; seed=42)
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

The training recipe is a three-stage chain (v17.1 pattern):
**v18-Core** runs the v12 two-phase recipe verbatim (Phase A broad
re-anchor 1e-5, Phase B RMS/ATE/VSP/RCM drill 5e-6) and pushes
`-v18-core`; **v18+TAA** chains a v16-shape TAA Classic refresher
(lr 5e-6, cutoff 4096, packing on) and pushes `-v18-taa`;
**v18** chains a v17.1-shape CSE drill (same recipe as v18+TAA, CSE
shard) and pushes `-v18-cse`. See `v18_plan.txt §4` for the
full hyperparameter table; launchers are
`SFT/autotrain/run_sft_qwen25_14b_v18_core.sh`,
`run_sft_qwen25_14b_v18_plus_taa.sh`, and
`run_sft_qwen25_14b_v18_final.sh`.

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

Three independent shard builds, one per chain stage. The Core build
uses the v18 manifest in this directory; the TAA and CSE builds reuse
the v16 and v17.1 manifests verbatim:

```bash
cd /Users/pietro/code/Glaukopis     # or the cluster equivalent

# --- stage 1: Core shard (v18 manifest) ---
python _v18_build/_neo4j_check.py    # manual smoke-test; also runs in watcher
mkdir -p _v18_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05112026/Sophia-CTI-Templates-v18.txt \
     _v18_build/triples \
     SFT/data/ift_data_2026_05_13_v18_core.raw.json \
     2500 3500 > _v18_build/build.log 2>&1 &
echo "PID=$!" > _v18_build/build.pid
nohup bash _v18_build/watcher.sh > _v18_build/watcher.log 2>&1 &

# --- stage 2: TAA Classic shard (v16 manifest verbatim) ---
mkdir -p _v18_taa_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05092026/Sophia-CTI-Templates-v16.txt \
     _v18_taa_build/triples \
     SFT/data/ift_data_2026_05_13_v18_taa.raw.json \
     10 3500 > _v18_taa_build/build.log 2>&1 &
echo "PID=$!" > _v18_taa_build/build.pid
nohup bash _v18_taa_build/watcher.sh > _v18_taa_build/watcher.log 2>&1 &

# --- stage 3: CSE shard (v17.1 manifest verbatim) ---
mkdir -p _v18_cse_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05102026/Sophia-CTI-Templates-v17.1.txt \
     _v18_cse_build/triples \
     SFT/data/ift_data_2026_05_13_v18_cse.raw.json \
     10 3500 > _v18_cse_build/build.log 2>&1 &
echo "PID=$!" > _v18_cse_build/build.pid
nohup bash _v18_cse_build/watcher.sh > _v18_cse_build/watcher.log 2>&1 &
```

Each watcher polls its build PID, then runs the post-build pipeline
appropriate to its shard (substrate gate / seed-provenance gate /
generator merges / actor-balance / dedup / row-count gate / licence
gate / [letter-balance gate for CSE] / stratified shuffle / val-train
split). The Core watcher additionally runs the legacy two-shard
phase split (broad + axis).

Final outputs:

- `SFT/data/ift_data_2026_05_13_v18_core_a_kb_mcq_taa_soc_cm_ms_yn.json` — Core Phase A shard (broad re-anchor; ~79% KB/glossary/Wikipedia/NIST + MCQ + TAA Classic + SOC + CM + MS + YN)
- `SFT/data/ift_data_2026_05_13_v18_core_b_rms_ate_vsp_rcm.json` — Core Phase B shard (RMS+ATE+VSP+RCM catalog drill)
- `SFT/data/ift_data_2026_05_13_v18_core_val.json` — Core val slice (50 rows × N axes)
- `SFT/data/ift_data_2026_05_13_v18_taa.json` — v18+TAA train shard
- `SFT/data/ift_data_2026_05_13_v18_taa_val.json` — v18+TAA val slice
- `SFT/data/ift_data_2026_05_13_v18_cse.json` — v18 CSE train shard
- `SFT/data/ift_data_2026_05_13_v18_cse_val.json` — v18 CSE val slice
- `_v18_build/watcher_status.json`, `_v18_taa_build/watcher_status.json`,
  `_v18_cse_build/watcher_status.json` — per-stage row counts and reports

Validate end-to-end with `python _v18_build/validate_corpus.py` (asserts
schema, source allowlist, MCQ letter balance, ATE lift, and per-stage
train/val disjointness across all four train shards).

### 4.2 Dataset registration

The launchers resolve dataset names through
`SFT/data/dataset_info.json`. The v18 entries are
`ift_data_2026_05_13_v18_core_a_kb_mcq_taa_soc_cm_ms_yn`,
`ift_data_2026_05_13_v18_core_b_rms_ate_vsp_rcm`,
`ift_data_2026_05_13_v18_core_val`,
`ift_data_2026_05_13_v18_taa`,
`ift_data_2026_05_13_v18_taa_val`,
`ift_data_2026_05_13_v18_cse`, and
`ift_data_2026_05_13_v18_cse_val`, registered alongside the
v17.1 / v17 / v16 / v15 / v12 entries.

### 4.3 Training

The three launchers run sequentially; each pulls the previous stage's
pushed HF checkpoint as its base. Stage 1 must complete and push to
HF before stage 2 starts:

```bash
# stage 1 (~13 h on 8xH100): broad + axis -> v18-core
bash SFT/autotrain/run_sft_qwen25_14b_v18_core.sh
# stage 2 (~6-8 h):           TAA Classic  -> v18-taa
bash SFT/autotrain/run_sft_qwen25_14b_v18_plus_taa.sh
# stage 3 (~4-6 h):           CSE drill    -> v18-cse (final)
bash SFT/autotrain/run_sft_qwen25_14b_v18_final.sh
# defaults to ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-cse
# base of stage 1 : Qwen/Qwen2.5-14B-Instruct
# total wall-time : ~24 h on 8xH100
```

### 4.4 Bench

Standard 14B AthenaBench + CyberMetric + CyberSOCEval sweep against
each of the three pushed checkpoints; only the v18 final stage is the
published deliverable, but the v18-core and v18-plus-taa benches are
recorded for regression diagnosis if the final fails. Compare v18-core
against v12 (the standalone baseline its Phase A/B mirrors), v18-plus-taa
against v16 (the TAA Classic chain it reproduces), and v18 against
v17.1 (the CSE chain it reproduces). Decision matrix in
`v18_plan.txt §5`. If v18 hits CKT ≥ 75 and ATE ≥ 58.5 without
regressing RMS / VSP / RCM / TAA-attr / SOC / CM by more than 2 pp and
keeps CSE-TI / CSE-MAL within 2 pp of v17.1, the chained recipe is
validated and v18 becomes the new shared base for any future v19+
chained specialist branches.

## 5. Contamination posture

Inherited from v8 / v10 / v11 / v14 / v17 without modification. The
v18 / v18.1 corpus is built by `_v18_build/watcher.sh` and
`_v18p1_build/watcher.sh`, whose Phase 5 runs
`tmpl_gen/scripts/dedup_against_evals.py` against
`SFT/eval/benchmark_data/` with `n=13` word-grams,
`hit-threshold=1`, and the v10 soft-drop policy (`--drop-threshold
50`). Verbatim leakage of any AthenaBench / CTIBench / CyberMetric /
CyberSOCEval row into the training corpus is blocked at build time;
structural overlap with the public MITRE / NIST / FIRST / CISA /
D3FEND knowledge bases is accepted by design.

**Canonical reference** -- conceptual taxonomy, n=13 rationale, and
literature pointers (SecKnowledge / CyberPal.AI Levi et al. 2024
arXiv:2408.09304, CTIBench, AthenaBench technical report): see
[`../04292026/README.md` §2](../04292026/README.md#2-contamination-posture).

**Exhaustive restatement** -- per-shard enforcement, per-benchmark
structural-overlap matrix, adjacent corpus-hygiene gates, and the
falsifiability list, scoped to the three-shard Core / TAA / CSE
pipeline that v18.1 originates and v21 reproduces byte-for-byte:
see
[`../05182026/README-21.md` §Contamination posture](../05182026/README-21.md#contamination-posture).

**v18 / v18.1-specific note.** v18.1 originates the three-shard
chained training topology (Core = v12 substrate, TAA Classic = v16
TAA-shard substrate, CSE = v17.1 CSE-shard substrate). Each shard's
template file and per-axis row-count gate is built from the
substrate it inherits, and each shard's dedup pass runs
independently against the full eval-dir. v21 forks v18.1's exact
template / gate / watcher inputs (byte-identical) for a strict
reproducibility experiment, which means **the v21 contamination
audit certifies the v18.1 corpus content as well** -- the two
corpora differ only by build-date stamp.

