# Sophia CTI Templates — v11 (May 3, 2026 vintage)

Self-contained, single-pass SFT manifest. v11 carries v10 forward verbatim
and adds three orthogonal expansions (SOC.* operations, TAA.CANON.* alias
resolution, RMS paraphrase-multiplied catalog drills) plus a ~30% trim of
the `X.*` and `YN.*` broad-knowledge tail to rebalance the corpus toward
eval-aligned and SOC-operations density. The manifest header
(`Sophia-CTI-Templates-v11.txt`, lines 1-128) is the canonical change log
for the v10 → v11 deltas; this README is the supplementary lineage
document and carries the full v0 → v11 history at the bottom so the
manifest+README pair is a complete corpus reference even when prior-
vintage directories are not consulted.

```
05032026/
  Sophia-CTI-Templates-v11.txt   self-contained v11 manifest (244 templates)
  v11_plan.txt                   master plan document (probe outputs, design notes)
  README.md                      this document
```

## 1. Build pipeline (deterministic, end-to-end)

```bash
# Phase 1: template -> triples -> Alpaca rows
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05032026/Sophia-CTI-Templates-v11.txt \
    _v11_build/triples \
    SFT/data/ift_data_2026_05_03_v11.raw.json \
    10 1500

# Phase 2 + 3: actor-balance + eval-set dedup post-pass
bash _v11_build/watcher.sh
```

Phase 1 reads the manifest with `tmpl_docx2json.py --count_limit 10`,
binds 244 templates against athena-cti-db via `iftgen.py --count_max 1500`
(per-template `Count:` overrides take precedence; see
`tmpl_parser.process_template` priority chain), and emits the raw Alpaca
JSON. Phase 2/3 (`_v11_build/watcher.sh`) applies the actor balancer
(`ACTOR_CAP=40`, `ACTOR_FLOOR=100`) and the 13-gram dedup filter
(`DEDUP_DROP_THRESHOLD=50`, held at v10's value after the round-1 attempt
at 30 wiped 4,800 legitimate `AB.RMS.3{b..h}` stratification rows).
Final outcome lands in `_v11_build/watcher_status.json`; per-phase logs
in `_v11_build/{build,watcher}.log`.

## 2. Corpus actuals (v11 round-3, final)

| | v10 actual | v11 actual | delta | notes |
|---|---:|---:|---:|---|
| Raw rows (post-build) | 208,077 | 199,796 | -8,281 | SOC/TAA.CANON caps reset to substrate ceilings (Sigma 1100, IR 670, TAA.CANON.1 187, etc.); X./YN. trim partially offset by uncapping X.VW/YN.VW back to 1500 to defend the RCM axis |
| Balanced rows | 200,961 | 199,648 | -1,313 | actor cap lifted 20 → 40 (recovers TAA.* from 1,284 to 5,594) |
| Final clean rows | 200,340 | **198,994** | **-1,346** | dedup held at 50 (round-1 at 30 lost 4.8K legitimate RMS rows) |
| Distinct templates | 216 (208 productive) | **244 (244 productive)** | +28 (+36 productive) | +SOC.{SIGMA,TRIAGE.*,MAL,IR.{1,2}}, +TAA.CANON.{1,2}, +AB.RMS.{4,5}{a..j} |
| Eval-axis-aligned rows | 52,992 | **61,003** | **+8,011** | rms +792, mcq +2,936, taa +4,310, taa-neg full recovery; vsp/rcm/ate at parity |
| Eval-axis corpus share | 26.5% | **30.7%** | +4.2 pp | |
| New-family rows (SOC + TAA.CANON) | 0 | **5,662** | +5,662 | net new training surface |

Net: corpus is 1,346 rows below v10 in absolute terms but adds 8,011
eval-axis-aligned rows and 5,662 net new training rows on top, lifting
eval-axis density from 26.5 % to 30.7 %.

## 3. v10 → v11 deltas (summary; canonical text in manifest header)

  1. **Naming migration**: drop "abaligned" from all v11 assets.
  2. **AB.RMS.{4,5} paraphrase-multiplied**: 10 phrasing variants per
     family (`{a..j}`), Count: 50 each, catalog-capped at ~44 per variant.
     Net yield ~440 per family (vs 44 in v10).
  3. **Anchor-fixation fix at parser chokepoint**: F3 step-by-step Cypher
     form gated by `Sample:` + `per_primary_grouping=true`; default gencfg
     switched to `gencfg_per_primary_neo4j.json`. Recovers `AB.MS.*` and
     `AB.TAA.*` from v10's single-anchor collapse.
  4. **New SOC.* family (4,884 actual rows, 2.5 % of corpus)**:
     `SOC.SIGMA.1` (1,100), `SOC.TRIAGE.SIGMA.1` (1,100),
     `SOC.TRIAGE.AN.1` (1,100), `SOC.TRIAGE.DS.{1,2}` (38 each),
     `SOC.MAL.1` (819), `SOC.IR.{1,2}` (335 + 335) via the new MITRE
     D3FEND v1.4.0 ingest landed at
     `athena_cti_db/threat_framework/populate_neo4j_complete.py`
     commit `67db101`. Caps in the manifest are set to the per-primary
     substrate ceilings actually achievable in athena-cti-db (round-2
     correction; round-1 used aspirational `Count: 3500` that yielded
     identical numbers but read as misleading).
  5. **New TAA.CANON.* family (778 actual rows)**: `TAA.CANON.1` (187,
     capped to substrate ceiling: 187 distinct intrusion-sets with
     `aliases`) and `TAA.CANON.2` (591, Count: 800). Source:
     `intrusion-set.aliases` array on athena-cti-db (104/187 groups have
     multi-element aliases averaging 3.1 per group).
  6. **X.* and YN.* trimmed ~30 %, with the RCM-axis exceptions
     uncapped**: 23 X.* templates capped at `Count: 850`, 17 YN.*
     templates capped at `Count: 950`, but `X.VW.{1,2,3}` and
     `YN.VW.{P,N}.1` (the five `athena-cti-rcm` axis templates) held at
     1500 to defend the RCM eval axis (round-2 correction; round-1 had
     them trimmed and `athena-cti-rcm` regressed by ~3K rows). The 8
     `X.{1..8}` broad-knowledge templates in Section C are pre-existing
     substrate and out of scope. Net trim ~17K rows reallocated to
     (4)/(5)/(2).
  7. **Build tunables**: `ACTOR_CAP` 20 → 40 (recovers TAA.* from 1,284
     to 5,594), `DEDUP_DROP_THRESHOLD` held at v10's 50 (round-1 attempt
     at 30 destroyed 4,800 legitimate `AB.RMS.3{b..h}` stratification
     rows whose multi-M-code answers structurally match eval ground
     truth, not training contamination).
  8. **Per-template `Per_primary_grouping: false` directive
     (parser extension)**: `tmpl_docx2json.py` recognises the new
     `Per_primary_grouping: true|false` directive and
     `tmpl_parser.process_template` consults it ahead of the gencfg
     default. Used on `AB.TAA.NEG.1` and `JS.TAA.NEG.1` whose
     `{force grp != rel}` + shared `(ap1, ap2, mw)` constraint collapses
     yield to ~10 % of the cap under the per-primary CALL-subquery
     `LIMIT 1` chaining (substrate probe in
     `_v11_build/_probe_taa_neg_yield.py`). Override fully recovers
     v10 raw yield (1500/500 vs round-2's 1106/378).
  9. **Stratified shuffle** pre-train (`stratified_shuffle.py`) to defend
     per-family minimum representation under LlamaFactory's uniform
     sampler.
 10. **Held-out validation slice** `ift_data_2026_05_03_v11_val.json`.

## 4. Consumers

| consumer | path |
|---|---|
| Dataset registration | `SFT/data/dataset_info.json` -> `ift_data_2026_05_03_v11` |
| Qwen2.5-14B launcher | `SFT/autotrain/run_sft_qwen25_14b_v11.sh` (no "abaligned" suffix) |
| HF target | `${HF_USERNAME}/athena-cti-sft-qwen25-14b-v11` |

## 5. Version history (v0 → v11)

Each row links the vintage directory carrying the manifest, README (where
present), and per-version build artefacts. Every prior version remains in
the repo verbatim for reproducibility.

| version | date | path | corpus | what it changed |
|---|---|---|---|---|
| v0 baseline | 2026-03-22 | `tmpl_gen/templates/04022026/` | small hand-crafted | Initial M/A/W/V/S/P/E/X.{1..8} substrate (the 64-template "broad CTI knowledge core" still carried in every later manifest as Section C). |
| v6 | 2026-04-25 | `tmpl_gen/templates/04252026/` | abaligned | First AthenaBench-aligned slate (AB.* family, MCQ + RCM + ATE + RMS distractor blocks with `{force}` constraints; `negcoa*/negcap*/negack*` distractor pattern established). |
| v7 | 2026-04-26 | `tmpl_gen/templates/04262026/` | combined | Consolidated v6 + JSON pre-cursor; introduced cross-framework path templates (X.VWA, X.TMN). |
| v8 | 2026-04-29 | `tmpl_gen/templates/04292026/` | small + large | Two split manifests (Llama-3.1-8B `v8_small`, Qwen2.5-32B `v8_large`); JSON-output addendum (`JS.*`); long-context scaffolding (`stitch_long_context.py`). |
| v8.1 | 2026-04-30 | `tmpl_gen/templates/04302026/` | 14B single-pass | RMS catalog-collapse fix: explicit `Count:` floors on AB.RMS.{4,5}; consolidated single-source-of-truth manifest; first build using `tmpl_docx2json` directly off the `.txt` file. |
| v9 | 2026-04-30 | `tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt` | two-phase | Phase B "RMS slice" extracted from v8.1 to recover catalog under v8.1's broad-knowledge regression. v9 ran as a two-phase chain (broad re-anchor → RMS) on the 14B base. |
| v10 | 2026-05-01 | `tmpl_gen/templates/05012026/` | 200,340 rows | Single-pass unified manifest (216 templates); HTML/whitespace sanitiser at parser chokepoint; `<desc>...</desc>` markers around freeform text; actor cap=20; 13-gram dedup at threshold 50. AthenaBench composite weighted total: 54.1. |
| v11 | 2026-05-03 | `tmpl_gen/templates/05032026/` | 198,994 rows | This vintage. 244 templates (all productive). SOC.* (4,884) + TAA.CANON.* (778) + RMS paraphrase + X./YN. trim with RCM-axis exemptions + parser anchor-fixation fix (per_primary_grouping=true global) + per-template `Per_primary_grouping: false` override for TAA.NEG + actor cap 20 → 40 + dedup held at 50 + D3FEND v1.4.0 ingest in athena-cti-db (commit `67db101`). Eval-axis-aligned rows: 61,003 (+8,011 vs v10), corpus share 30.7 % (+4.2 pp). See deltas (1)-(10) above. |

For the line-by-line corpus composition of v10 (the substrate v11 carries
forward) see `tmpl_gen/templates/05012026/README.md`. For the v8.1 / v9
two-phase rationale (the chain v10 collapsed back into a single pass) see
`tmpl_gen/templates/04302026/README.md`. For the v8 split-manifest design
see `tmpl_gen/templates/04292026/README.md`. The v0 substrate's per-family
design strategy (`M/A/W/V/S/P/E/X`) is in `tmpl_gen/templates/README.md`.

## 6. Known content gaps to watch at eval

  1. **TAA.CANON.1 single-alias trivial case**: groups with only one
     element in `aliases` (≈83 of 187 intrusion-sets) yield trivial
     identity rows ("alias `Storm-0501` → canonical `Storm-0501`"). The
     template grammar's `{force ...}` is path-vs-path RELOP only and
     cannot express `size(aliases) >= 2`; multi-alias filtering would
     require a parser extension. Information value is preserved by the
     non-trivial multi-alias rows (104 groups, average 3.1 aliases each).
  2. **SOC.TRIAGE.DS.{1,2}** bind on the data-source node alone because
     the 38 `x-mitre-data-source` nodes in athena-cti-db are isolated
     (no edges to `attack-pattern`, and the `x_mitre_data_sources`
     arrays on `attack-pattern` are empty). Catalog-drill coverage only.
  3. **`AB.RMS.{4,5}` ceiling**: paraphrase-multiplied to ~440 per family
     in v11; if `athena-cti-rms` still regresses vs v8.1 the M-control
     catalog itself needs expansion in athena-cti-db.

## 7. Contamination posture

Inherited from v8 (`../04292026/README.md` §2) and v10
(`../05012026/README.md` §10) without modification. The v11 corpus
is built by `_v11_build/watcher.sh`, whose Phase 5 runs
`tmpl_gen/scripts/dedup_against_evals.py` against
`SFT/test/benchmark_data/` with `n=13` word-grams,
`hit-threshold=1`, and the v10 soft-drop policy (`--drop-threshold
50`). Verbatim leakage of any AthenaBench / CTIBench / CyberMetric /
CyberSOCEval row into the training corpus is blocked at build time;
structural overlap with the public MITRE / NIST / FIRST / CISA /
D3FEND knowledge bases is accepted by design.

**Canonical reference** -- conceptual taxonomy (verbatim vs.
structural), n=13 rationale anchored to OLMo / Pythia / Llama
decontamination conventions, and literature pointers (SecKnowledge /
CyberPal.AI Levi et al. 2024 arXiv:2408.09304, CTIBench, AthenaBench
technical report): see
[`../04292026/README.md` §2](../04292026/README.md#2-contamination-posture).

**Exhaustive restatement** -- per-shard enforcement, per-benchmark
structural-overlap matrix, adjacent corpus-hygiene gates, and the
falsifiability list: see
[`../05182026/README-21.md` §Contamination posture](../05182026/README-21.md#contamination-posture).

**v11-specific note.** v11 changes corpus composition (paraphrase
multiplication on `AB.RMS.{4,5}`, see §6) but does not change the
dedup tunables or the eval-dir scope. The `dedup_report.json` shape
and the soft-drop policy are byte-equivalent to v10's.

