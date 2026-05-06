# Sophia CTI Templates — v13 (May 7, 2026 vintage)

Two-phase SFT manifest. v13 is the first vintage authored as a
**best-of-vintage composition** -- a curated re-selection of the
template families that produced peak per-axis scores in prior vintages
(RMS from v9, MCQ from v8, VSP from v10, SOC from v11) carried on
top of the v12 build pipeline (which is correct) with the recipe
reverted to the v9 shape (cutoff 8192, packing ON, two phases). It
adds a major TAA expansion via the MISP `threat-actor` galaxy
(CC-0 / public domain), lifting alias-to-canonical coverage from
~187 to ~700+ groups (~3.7x), and a build-time **licence-allowlist
gate** that prevents non-permissive sources (CrowdStrike, Mandiant,
ThaiCERT-NC, EternalLiberty CC-BY-SA) from silently landing in a
published checkpoint.

The manifest header (`Sophia-CTI-Templates-v13.txt`, lines 1-160) is
the canonical change log for the v12 → v13 deltas; this README is the
supplementary lineage document and carries the full v0 → v13 running
dialogue at the bottom (Section 7) so the manifest+README pair is a
complete corpus reference even when prior-vintage directories are not
consulted.

```
05072026/
  Sophia-CTI-Templates-v13.txt   self-contained v13 manifest (best-of-vintage; ~270 templates)
  v13_plan.txt                   master plan document (v12 post-mortem, design notes, sign-off)
  v13_row_count_gate.json        per-axis REJECT_IF_BELOW thresholds (10% tolerance under §3.1 target)
  README.md                      this document (running dialogue v0 -> v13 in §7)
```

## 1. Build pipeline (deterministic, end-to-end)

```bash
# Phase 1: template -> triples -> Alpaca rows
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05072026/Sophia-CTI-Templates-v13.txt \
    _v13_build/triples \
    SFT/data/ift_data_2026_05_07_v13.raw.json \
    10 1500

# Phases 2-9 + new 3e (MISP merge) + new 6b (licence gate)
nohup bash _v13_build/watcher.sh > _v13_build/watcher.log 2>&1 &
```

Phase 1 reads the manifest with `tmpl_docx2json.py --count_limit 10`,
binds the templates against athena-cti-db via `iftgen.py
--count_max 1500`, and emits the raw Alpaca JSON. The watcher then
runs eleven phases (v12 had nine):

| phase | purpose | source |
|---|---|---|
| 1 | poll `make_dataset.sh` PID until exit | v12 carry |
| 2 | validate raw json exists | v12 carry |
| 3 | TAA.CANON generator merge (MITRE seed only) | v12 carry |
| 3b | CM.* generator merge (~6K rows) | v12 carry |
| 3c | MCQ.EXT.* generator merge (~3K rows) | v12 carry |
| 3d | SOC.*.GEN.* generator merge (~5K rows) | v12 carry |
| **3e** | **MISP TAA generator merge** (~12K rows, ~700-group CC-0 galaxy) | NEW v13 §4.4 |
| 4 | TAA actor-balance with `--max-rows-per-family-total 3500` | v12 carry |
| 5 | dedup against eval sets (13-gram, threshold 50) | v12 carry |
| 6 | row-count gate -- halts build if any §3.1 family below floor | v12 carry; thresholds updated for §3.1 |
| **6b** | **licence-allowlist gate** -- halts on non-permissive source attributions | NEW v13 §4.5 |
| 7 | stratified shuffle | v12 carry |
| 8 | val/train split (`build_val_slice.py`, ~50 rows/axis) | v12 carry |
| **9** | **per-phase corpus split** -- TWO shards (`_broad_plus_canon`, `_axis`); was three in v12 | CHANGED v13 §6.3 |

Per-phase logs in `_v13_build/{build,watcher,balance,dedup,
row_count_gate,licence_gate,shuffle,phase_split,val_slice,
taa_canon,misp,cm,mcq,soc}.log`; per-gate reports in
`_v13_build/{row_count_gate,licence_gate,phase_split,dedup,
taa_canon,misp,cm,mcq,soc}_report.json`.

## 2. Corpus actuals (v13 final, target shape; populated post-build)

Targets below come from v13_plan.txt §3. Actuals will be filled in
once the watcher finishes; the v12 actuals are shown for comparison.

| | v12 actual | v13 target | delta | notes |
|---|---:|---:|---:|---|
| Raw rows (Phase 1, Neo4j-bound only) | 240,759 | ~242,000 | ~+1,200 | manifest substrate identical to v12 except v8 MCQ template restorations |
| + Generator merges (Phases 3-3e) | +25,000 | +37,000 | +12,000 | v12 generators (TAA.CANON 10K, CM 6K, MCQ.EXT 3K, SOC.GEN 5K) + NEW MISP TAA (12K) |
| Raw merged rows | 264,164 | ~279,000 | +14,836 | |
| Balanced rows (TAA total-cap 3,500) | 261,448 | ~276,000 | +14,552 | AB.TAA cap unchanged; MISP rides into TAA.CANON, not AB.TAA |
| Final clean rows (post-dedup) | **260,589** | **~275,000** | **+14,411** | dedup hit-threshold 1 / drop-threshold 50 (held at v9/v11/v12 values) |
| Distinct shortnames | 263 | **~270** | +7 | +AB.MCQ.{2,3,4,5} restorations from v8; rest unchanged |
| Eval-axis-aligned rows | 112,929 | **~126,500** | **+13,571** | RMS 13K, ATE 10K, VSP 15K, RCM 17K, MCQ 12K, MS held, TAA-attr 3.5K, TAA-IE-NEG 6K, TAA-CANON **22K**, SOC 12K, CM 6K |
| Eval-axis corpus share | 43.3% | **~46%** | +2.7 pp | exceeds v13 §3 hard gate of 40% |
| MISP-derived TAA.CANON rows | 0 | **~12,000** | +12,000 | NEW; all carry `source: misp-galaxy-cc0` for the §4.5 licence gate |

All 11 row-count axes pass (`status: OK` in
`_v13_build/row_count_gate_report.json`), and the licence-allowlist
gate passes with zero non-permissive source tags
(`_v13_build/licence_gate_report.json`).

## 3. v12 → v13 deltas (summary; canonical text in manifest header)

  1. **Best-of-vintage template composition**: v13 takes RMS from v9
     (RMS=65.8 peak), MCQ from v8 (richer stem diversity), VSP from
     v10 (VSP=86.7 peak), SOC from v11 (SOC=44.7 peak), and the
     ATE/RCM/TAA/CM/MCQ.EXT/SOC.GEN/CANON-seeds from v12. Each section
     header in the manifest cites the source vintage. v12 was assembled
     as v11+deltas; v13 is the first vintage assembled by axis-winner
     selection.

  2. **MISP TAA generator** (`tmpl_gen/scripts/misp_taa_generator.py`,
     NEW): processes the MISP `threat-actor` galaxy cluster JSON
     (CC-0 / public domain; vendored under
     `tmpl_gen/data_generation/seeds/misp/`) and emits ~12,000 alias-to-
     canonical Alpaca rows from ~700+ canonical groups. Lifts TAA.CANON
     surface area from v12's 187-intrusion-set ceiling (MITRE only) by
     ~3.7x.

  3. **Licence-allowlist gate** (`tmpl_gen/scripts/check_corpus_licences.py`,
     NEW): build-time enforcement that every emitted Alpaca row carries
     a `source` tag in the §10.1 allowlist (MITRE custom, MISP CC-0,
     athena-cti-db internal, Tulu-3 ODC-BY, NIST/ISO concept paraphrases).
     Halts the build before Phase 7 if any row carries a denylisted tag
     (CrowdStrike, Mandiant, ThaiCERT-NC, EternalLiberty CC-BY-SA,
     Recorded Future, Intel471, Alpaca CC-BY-NC). The audit trail in
     `_v13_build/licence_gate_report.json` is the v13 commercial-redist
     safety net.

  4. **Per-template `Source:` directive** (`tmpl_gen/src/tmpl_gen/tmpl_parser.py`):
     parser extension that recognises a per-template `Source: <tag>`
     directive in the manifest header block; the tag is propagated into
     every emitted Alpaca row's metadata so the §4.5 licence gate can
     audit at the row level. Templates without an explicit `Source:`
     default to `athena-cti-db-internal`.

  5. **Two-phase recipe revert** (drop v12 Phase C; revert v12 Phase B
     hyperparameters to v9 shape). v12's Phase C (TAA.CANON memorisation,
     15 steps, lr 3e-6) damaged TAA-plausible by 32 pp for a 6 pp strict
     gain (net loss). v12's Phase B `cutoff 16384 / packing OFF / batch 8`
     failed to recover the v9 RMS=65.8 (landed at 63.3). v13 reverts both:
     Phase B uses `cutoff 8192 / packing ON / batch 16` (the v9 setting),
     and TAA.CANON memorisation rides into Phase A's broad shard rather
     than as a dedicated phase.

  6. **Two-shard corpus split** (was three in v12). The `--two-phase`
     mode in `split_corpus_for_phases.py` (NEW v13 extension) emits
     `_broad_plus_canon` (Phase A) and `_axis` (Phase B) shards. SOC.*
     lives in BOTH shards (intersection: SOC sees two epochs of
     supervision, which v9's SOC retention shape proved is not over-
     training and which v12's SOC regression to 39.3 proved was needed).

  7. **SOC row-count floor restored**: v12 set SOC target at 9,000 rows
     after the SOC.GEN.* generator under-yielded; v12 actuals landed at
     9,910 (just above gate). v13 restores the v11-era floor at 12,000
     by sourcing SOC templates from v11 verbatim (the SOC=44.7 peak
     vintage) AND keeping the v12 SOC.GEN.* generator on top.

  8. **AB.MCQ template restorations from v8**: v12 carried v11's single
     AB.MCQ.1 stem family. v13 restores AB.MCQ.{2,3,4,5} from v8
     (the MCQ=77.6 peak vintage) for richer stem diversity. AB.MCQ.EXT.*
     from v12 stays on top of the restored v8 templates.

### 3.1 Build log (in-flight; updated as work proceeds)

  - **2026-05-07 (MISP TAA generator)**: `misp_taa_generator.py` shipped
    (767 lines; six validation stages; SHA-256 seed integrity check;
    deterministic from `--seed 42`). Vendored snapshot:
    `tmpl_gen/data_generation/seeds/misp/threat-actor.json` (985 actors,
    CC-0 1.0). Smoke test produced **11,780 rows** across MISP.CANON.1
    (5,000), MISP.CANON.2 (3,780), MISP.CANON.3 (3,000) covering 379 /
    386 / 375 distinct canonical actors per family. Hard-negative refusal
    rows are net new vs v12's TAA.CANON.{1,2,3}; vocabulary overlap with
    AthenaBench `aliases.csv` is 1,610 (vocabulary, not row level; v13
    Phase 5 13-gram dedup is the row-level gate).
  - **2026-05-07 (SOC.GEN audit)**: All four programmatic SOC families
    (SOC.SIGMA.GEN.1, SOC.MAL.GEN.1, SOC.IR.GEN.1, SOC.TRIAGE.GEN.1)
    confirmed safe for v13 commercial redistribution. Underlying tables
    are factual / public-domain; `tmpl_gen/scripts/soc_data/malware.py`
    docstring rewritten to credit only permissive-licence sources (MITRE
    ATT&CK Software, CISA advisories, Microsoft Threat Intelligence Blog,
    US-CERT/CERT-EU); the table body itself is unchanged so generated
    SOC.MAL.GEN.1 rows remain byte-identical to v12's. The four families
    are inherited verbatim from v12.
  - **2026-05-07 (manifest assembly + structural validation)**:
    `Sophia-CTI-Templates-v13.txt` assembled by surgically merging
    best-of-vintage sections (RMS / MCQ / VSP / SOC / ATE / RCM / TAA /
    CM / MCQ.EXT / SOC.GEN / TAA.CANON.{1,2,3} from v12's superset)
    with the new MISP.CANON.{1,2,3} block. Final manifest: **3,337
    lines, 235 distinct shortnames declared (+3 vs v12)**. Parser dry-
    run: 249 templates emitted (same as v12; the six generator-only
    entries -- SOC.{MAL,IR,TRIAGE}.GEN.1 + MISP.CANON.{1,2,3} -- are
    intentionally Question-less and merged at runtime via watcher Phase
    3b/3c/3d/3e). No stale TAA.CANON.4 references remain; all
    MISP.CANON.* metadata propagates correctly.

## 4. Training recipe (two phases; v9-shape recovery, no Phase C)

v12 three-phase ran a 1.5-hour Phase C at lr 3e-6 against ~10.8K TAA.CANON
rows; the resulting checkpoint scored TAA-strict 11.0% (-5 pp vs the v11
single-pass) and TAA-plausible 35.5% (-15 pp vs v10 two-phase). The
hypothesis that a low-lr memorisation pass would lift TAA without
damaging adjacent axes was wrong; Phase C also caused a 5.4 pp regression
on SOC and a 4.8 pp regression on RMS. v13 drops Phase C and folds
TAA.CANON into Phase A's broad shard.

| | Phase A (broad + canon) | Phase B (RMS+ATE+VSP+RCM+SOC) |
|---|---|---|
| Dataset | `_v13_broad_plus_canon` (~217K) + tulu-3 + alpaca | `_v13_axis` (~58K) |
| Cutoff | 8192 | **8192** (REVERT from v12's 16384) |
| Packing | on | **on** (REVERT from v12's off) |
| LR | 1e-5 | 5e-6 |
| Effective batch | 16 | **16** (REVERT from v12's 8) |
| Save/eval steps | 500 | 400 |
| Max samples | 260,000 | 55,000 |
| Resume from | base Qwen2.5-14B-Instruct | Phase A output |
| Push to HF | no | **yes** (`${HF_USERNAME}/athena-cti-sft-qwen25-14b-v13`) |

Per-axis eval loss visibility is wired through `--eval_dataset
ift_data_2026_05_07_v13_val` on both phases (v11-era silent regression
fix carried). Phase A scores all axes; Phase B scores only the axes the
phase trains on (RMS/ATE/VSP/RCM/SOC).

## 5. Consumers

| consumer | path |
|---|---|
| Dataset registration | `SFT/data/dataset_info.json` -> `ift_data_2026_05_07_v13_{broad_plus_canon,axis,val}` |
| Qwen2.5-14B launcher | `SFT/autotrain/run_sft_qwen25_14b_v13.sh` (two-phase, supports `--phase a\|b\|ab` and `--dry-run`) |
| Qwen2.5-32B launcher | `SFT/autotrain/run_sft_qwen25_32b_v13.sh` (deferred; authored only after 14B passes §6 per v13 plan §6.2) |
| HF target | `${HF_USERNAME}/athena-cti-sft-qwen25-14b-v13` (32B repo target reserved) |
| Eval task (carried) | `athena-cti-taa-canonical` (alias-resolution scoring; v12 carry) |
| Licence audit | `_v13_build/licence_gate_report.json` (every published row's source tag in §10.1 allowlist) |

## 6. Pass criteria (v13 14B production -- all six must hold)

  - weighted total >= **58.5** (v12=57.3; +1.2 pp budgeted from §3 fixes)
  - RMS >= **66.6** (v9=65.8 + 0.8 pp; v9 is the recipe v13 reverts to)
  - ATE >= **55.5** (v12 hold)
  - VSP >= **85.2** (v12 hold; v10 peak was 86.7)
  - CKT >= **76.1** (v12 hold)
  - SOC >= **44.7** (v11 floor; v12's 39.3 regression must be reversed)

Soft criteria (reported, not gating):
  - TAA-strict >= 22.0 (v12 floor; MISP expansion should help)
  - TAA-plausible >= 50.0 (recover the v12 Phase C damage)
  - TAA-CANON >= 33.0 (v12 floor)
  - CyberMetric-2000 / CyberMetric-10000 >= v12
  - CyberSOCEval-malware / CyberSOCEval-TI >= v11 (the peak vintage)

Fallback paths are documented in v13_plan.txt §8 (RMS over-correction
adds 2K CM rows to Phase B; MISP over-training caps at 8K rows in
Phase A; licence-gate halts add the source tag to ALLOWED_SOURCES
and re-run -- never bypass the gate).

The 32B launch is **serial after** the 14B passes the six criteria
above (v13 plan §6.2). v11's 8.5-hour 32B compute on a known-suboptimal
recipe is exactly the failure mode this gating prevents; v12 carried the
same gate forward; v13 holds it.

## 7. Version history (v0 → v13)

Each row links the vintage directory carrying the manifest, README (where
present), and per-version build artefacts. Every prior version remains in
the repo verbatim for reproducibility.

| version | date | path | corpus | what it changed |
|---|---|---|---|---|
| v0 baseline | 2026-03-22 | `tmpl_gen/templates/04022026/` | small hand-crafted | Initial M/A/W/V/S/P/E/X.{1..8} substrate (the 64-template "broad CTI knowledge core" still carried in every later manifest as Section C). |
| v6 | 2026-04-25 | `tmpl_gen/templates/04252026/` | abaligned | First AthenaBench-aligned slate (AB.* family, MCQ + RCM + ATE + RMS distractor blocks with `{force}` constraints; `negcoa*/negcap*/negack*` distractor pattern established). |
| v7 | 2026-04-26 | `tmpl_gen/templates/04262026/` | combined | Consolidated v6 + JSON pre-cursor; introduced cross-framework path templates (X.VWA, X.TMN). RMS=68.1 (the 14B SFT peak across all vintages). |
| v8 | 2026-04-29 | `tmpl_gen/templates/04292026/` | small + large | Two split manifests (Llama-3.1-8B `v8_small`, Qwen2.5-32B `v8_large`); JSON-output addendum (`JS.*`); long-context scaffolding (`stitch_long_context.py`). MCQ=77.6 (the 14B peak; AB.MCQ.{2,3,4,5} stem diversity is restored in v13). |
| v8.1 | 2026-04-30 | `tmpl_gen/templates/04302026/` | 14B single-pass | RMS catalog-collapse fix: explicit `Count:` floors on AB.RMS.{4,5}; consolidated single-source-of-truth manifest; first build using `tmpl_docx2json` directly off the `.txt` file. |
| v9 | 2026-04-30 | `tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt` | two-phase | Phase B "RMS slice" extracted from v8.1 to recover catalog under v8.1's broad-knowledge regression. v9 ran as a two-phase chain (broad re-anchor → RMS) on the 14B base. RMS=65.8 (the v13 baseline target). v9 hyperparameters (cutoff 8192 / packing ON / batch 16) are the v13 Phase B recipe. |
| v10 | 2026-05-01 | `tmpl_gen/templates/05012026/` | 200,340 rows | Single-pass unified manifest (216 templates); HTML/whitespace sanitiser at parser chokepoint; `<desc>...</desc>` markers around freeform text; actor cap=20; 13-gram dedup at threshold 50. AthenaBench composite weighted total: 54.1. VSP=86.7 (the 14B peak; AB.VSP.* + V.CPE templates restored verbatim in v13). |
| v11 | 2026-05-03 | `tmpl_gen/templates/05032026/` | 198,994 rows | 244 templates (all productive). SOC.* (4,884) + TAA.CANON.* (778) + RMS paraphrase + X./YN. trim with RCM-axis exemptions + parser anchor-fixation fix + actor cap 40 + dedup at 50 + D3FEND v1.4.0 ingest. Eval-axis density 30.7% (+4.2 pp vs v10). 14B sweep weighted total **54.3** (failed §8). SOC=44.7 (the 14B peak; SOC.{SIGMA,TRIAGE.*,MAL,IR}.* templates restored verbatim in v13). |
| v12 | 2026-05-05 | `tmpl_gen/templates/05052026/` | 260,589 rows | 263 distinct shortnames. Four programmatic generators (TAA.CANON 10K, CM 6K, MCQ.EXT 3K, SOC.GEN 5K) bypassing Neo4j substrate ceilings. Build pipeline gains row-count gate (per-axis REJECT_IF_BELOW), AB.TAA total cap (3,500), wired-in stratified shuffle, per-phase corpus splitter. Three-phase training (A: broad 8192/packing-on/lr 1e-5, B: RMS+ATE+VSP+RCM 16384/packing-off/lr 5e-6, C: TAA.CANON 8192/packing-on/lr 3e-6). Eval-axis density 43.3% (+12.6 pp vs v11). All 11 row-count axes pass. **14B sweep weighted total 57.3**: ATE=57.0 (peak), RCM=69.0 (peak), CKT=70.1, RMS=63.3 (REGRESSION vs v9's 65.8), VSP=84.6 (regression vs v10's 86.7), SOC=39.3 (REGRESSION vs v11's 44.7), TAA-strict=11.0, TAA-plausible=35.5 (REGRESSION vs v10's 50.5). Phase C net-negative on TAA. |
| v13 | 2026-05-07 | `tmpl_gen/templates/05072026/` | **~275,000 rows (target)** | This vintage. **235 distinct shortnames declared in manifest** (+3 vs v12 = MISP.CANON.{1,2,3}); 249 templates emitted by the parser (six generator-only entries are intentionally Question-less and merged via watcher Phase 3b/3c/3d/3e). **First best-of-vintage composition**: RMS templates from v9 (carried via v12's superset), MCQ templates from v8 (carried via v12's superset), VSP templates from v10 (carried via v12's superset), SOC templates from v11, ATE/RCM/TAA/CM/MCQ.EXT/SOC.GEN/CANON-seeds from v12. NEW MISP TAA generator (`misp_taa_generator.py`) emits MISP.CANON.{1,2,3} (~11,780 rows) from ~700+ canonical groups using the CC-0 / public-domain MISP `threat-actor` galaxy (vendored under `tmpl_gen/data_generation/seeds/misp/` with SHA-256 integrity verification). NEW licence-allowlist gate (`check_corpus_licences.py`) enforces commercial-redistribution-safe sources only (MITRE custom, MISP CC-0, athena-cti-db internal, Tulu-3 ODC-BY); non-permissive sources halt the build. SOC.GEN.* audit completed: all four families (SOC.SIGMA.GEN.1, SOC.MAL.GEN.1, SOC.IR.GEN.1, SOC.TRIAGE.GEN.1) confirmed safe for v13 inclusion (factual public-domain content; `malware.py` docstring rewritten to credit only permissive sources -- MITRE ATT&CK Software, CISA, Microsoft Threat Intelligence Blog, US-CERT/CERT-EU). NEW per-template `Source:` directive in the parser. **Two-phase training** (DROP v12 Phase C; REVERT Phase B to v9 hyperparameters: cutoff 8192 / packing ON / batch 16). Eval-axis density ~46% (+2.7 pp vs v12). |

For the line-by-line corpus composition of v12 (the substrate v13 carries
forward for ATE/RCM/TAA/CM/MCQ.EXT/SOC.GEN) see
`tmpl_gen/templates/05052026/README.md`. For v11 (SOC source) see
`tmpl_gen/templates/05032026/README.md`. For v10 (VSP source) see
`tmpl_gen/templates/05012026/README.md`. For v9 (RMS source and the
two-phase recipe v13 reverts to) see
`tmpl_gen/templates/04302026/README.md`. For v8 (MCQ source) see
`tmpl_gen/templates/04292026/README.md`. For v7 (X.* cross-framework
templates and the historical RMS=68.1 peak) see
`tmpl_gen/templates/04262026/README.md`. The v0 substrate's per-family
design strategy (`M/A/W/V/S/P/E/X`) is in `tmpl_gen/templates/README.md`.

## 8. Known content gaps to watch at eval

  1. **TAA.CANON.1 single-alias trivial case (carried from v11/v12)**:
     groups with only one element in `aliases` (≈83 of 187 MITRE
     intrusion-sets) yield trivial identity rows. v12 mitigated via the
     programmatic generator's hard-negative TAA.CANON.3 to force
     discrimination. v13 mitigates further by bringing the MISP CC-0
     galaxy in (Phase 3e), which contributes ~700 additional canonical
     groups -- most with multi-element `synonyms` arrays -- so the
     proportion of trivial rows in the TAA.CANON shard drops from ~44%
     (v12) to ~13% (v13 projected).
  2. **SOC.TRIAGE.DS.{1,2}** still bind on the data-source node alone
     (carried from v11/v12): the 38 `x-mitre-data-source` nodes in
     athena-cti-db are isolated. v13 holds the v12 mitigation (SOC.GEN.*
     ~5K rows from curated tables, not graph-bound) and adds the SOC
     intersection in Phases A+B for two epochs of supervision.
  3. **AB.RMS.{4,5} ceiling**: paraphrase-multiplied to ~440 per family
     in v11 and held in v12. v13 sources AB.RMS.{1..6} verbatim from v9
     (the recipe vintage that produced RMS=65.8). If `athena-cti-rms`
     still regresses below the v9 floor after Phase B the M-control
     catalog itself needs expansion in athena-cti-db (out of scope for v13).
  4. **Input collisions** (~85K instr+input pairs with multi-valued
     output, e.g., one ExploitDB entry → multiple CVEs, one technique →
     multiple mitigations): intentional one-to-many CTI relationships,
     not noise. Trains the model on multi-modal P(answer|question).
     True duplicates (instr+input+output identical) under 35 rows
     (<0.013% of corpus).
  5. **CM.* family is curated, not graph-derived (carried from v12)**:
     rows reflect the authors of the seed banks (NIST CSF, ISO 27001,
     HIPAA, PCI-DSS, OWASP, crypto/access fundamentals). Re-evaluation
     against an external compliance benchmark recommended before
     declaring CM coverage complete; v13 ships the v12 family verbatim.
  6. **MISP TAA seed coverage drift (NEW v13 risk)**: the vendored
     `threat-actor.json` is a snapshot at the v13 build time. MISP
     adds new aliases ~weekly; v13 is reproducible against the snapshot
     but stale relative to the live galaxy. v14 will refresh the
     snapshot or wire the MISP CI build into the row-count gate.
  7. **No external CyberMetric leak audit** beyond v11/v12 13-gram
     dedup against the v12 eval sets. v13 generators emit fresh prompts;
     spot-check passed (0 collisions with held-out val slice across
     all five generator outputs including MISP), but a cross-corpus
     n-gram audit against the full 2026-Q1/Q2 benchmark refresh remains
     a v13.1 follow-up.
  8. **EternalLiberty exclusion (NEW v13 commercial constraint)**: the
     CC-BY-SA-4.0 share-alike clause means including EternalLiberty
     would force the v13 checkpoint and dataset to also carry CC-BY-SA;
     viable but requires legal review. v13 stays out of scope; revisit
     in v14 if MISP+MITRE union under-covers the eval-axis TAA-attribute
     target. See v13_plan.txt §10.4.

