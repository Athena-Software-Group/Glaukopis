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
(`ACTOR_CAP=40`, `ACTOR_FLOOR` unchanged) and the 13-gram dedup filter
(`DEDUP_DROP_THRESHOLD=30`, tightened from v10's 50). Final outcome lands
in `_v11_build/watcher_status.json`; per-phase logs in
`_v11_build/{build,balance,dedup,watcher}.log`.

## 2. Corpus targets

| | v10 actual | v11 target | source of delta |
|---|---:|---:|---|
| Raw rows (post-build) | 208,077 | ~250,000 | SOC.* (+15.7K), TAA.CANON.* (+1.4K), RMS paraphrase (~+8K), X.*/YN.* trim (-24K) |
| Balanced rows | 200,961 | ~245,000 | actor cap lifted 20 → 40 (TAA.* recovers ~3.5K from v10's 1,284) |
| Final clean rows | 200,340 | ~240,000 | dedup threshold 50 → 30 (slightly stricter) |
| Distinct templates | 216 (208 productive) | 244 | +SOC.{SIGMA,TRIAGE.*,MAL,IR.{1,2}}, +TAA.CANON.{1,2}, +AB.RMS.{4,5}{a..j} |

## 3. v10 → v11 deltas (summary; canonical text in manifest header)

  1. **Naming migration**: drop "abaligned" from all v11 assets.
  2. **AB.RMS.{4,5} paraphrase-multiplied**: 10 phrasing variants per
     family (`{a..j}`), Count: 50 each, catalog-capped at ~44 per variant.
     Net yield ~440 per family (vs 44 in v10).
  3. **Anchor-fixation fix at parser chokepoint**: F3 step-by-step Cypher
     form gated by `Sample:` + `per_primary_grouping=true`; default gencfg
     switched to `gencfg_per_primary_neo4j.json`.
  4. **New SOC.* family (~15.7K rows)**: `SOC.SIGMA.1` (3.5K),
     `SOC.TRIAGE.SIGMA.1` (3.5K), `SOC.TRIAGE.AN.1` (1.9K),
     `SOC.TRIAGE.DS.{1,2}` (200 each), `SOC.MAL.1` (1.4K),
     `SOC.IR.{1,2}` (3.5K + 1.5K via the new MITRE D3FEND v1.4.0 ingest
     landed at `athena_cti_db/threat_framework/populate_neo4j_complete.py`
     commit `67db101`).
  5. **New TAA.CANON.* family (~1.4K rows)**: `TAA.CANON.1`
     (alias-list → canonical-name, Count: 600) and `TAA.CANON.2`
     (canonical → alias-resolution card with G-code and signature
     technique, Count: 800). Source: `intrusion-set.aliases` array on
     athena-cti-db (104/187 groups have multi-element aliases averaging
     3.1 per group).
  6. **X.* and YN.* trimmed ~30%**: 23 X.* templates capped at
     `Count: 850`, 17 YN.* templates capped at `Count: 950`. Net trim
     ~24K rows reallocated to (4)/(5)/(2). The 8 X.{1..8} broad-knowledge
     templates in Section C are the pre-existing substrate and are not in
     scope of this trim.
  7. **Build tunables**: `ACTOR_CAP` 20 → 40, `DEDUP_DROP_THRESHOLD`
     50 → 30 in `_v11_build/watcher.sh`.
  8. **Stratified shuffle** pre-train (`stratified_shuffle.py`) to defend
     per-family minimum representation under LlamaFactory's uniform
     sampler.
  9. **Held-out validation slice** `ift_data_2026_05_03_v11_val.json`.

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
| v11 | 2026-05-03 | `tmpl_gen/templates/05032026/` | ~240K target | This vintage. SOC.* + TAA.CANON.* + RMS paraphrase + X./YN. trim + parser anchor-fixation fix + actor cap 20 → 40 + dedup 50 → 30 + D3FEND ingest in athena-cti-db. See deltas (1)-(9) above. |

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
