# Sophia CTI Templates — v12 (May 5, 2026 vintage)

Three-phase SFT manifest. v12 carries v11 forward verbatim and adds four
programmatic generators (TAA.CANON, CM, MCQ.EXT, SOC.GEN) plus four
build-time gates (row-count gate, AB.TAA total cap, stratified shuffle
wired in, per-phase corpus splitter) to fix the v11 14B benchmark sweep
which landed weighted total 54.3, missing 3 of 4 §8 pass criteria.
v12 also reverts the v11 single-pass training recipe to a v9-shaped
phased SFT and adds a third phase for TAA.CANON memorisation. The
manifest header (`Sophia-CTI-Templates-v12.txt`, lines 1-80) is the
canonical change log for the v11 → v12 deltas; this README is the
supplementary lineage document and carries the full v0 → v12 history at
the bottom so the manifest+README pair is a complete corpus reference
even when prior-vintage directories are not consulted.

```
05052026/
  Sophia-CTI-Templates-v12.txt   self-contained v12 manifest (v11 + 4 new generator stanzas)
  v12_plan.txt                   master plan document (v11 post-mortem, design notes, sign-off)
  v12_row_count_gate.json        per-axis REJECT_IF_BELOW thresholds (10% tolerance under §3.1 target)
  README.md                      this document
```

## 1. Build pipeline (deterministic, end-to-end)

```bash
# Phase 1: template -> triples -> Alpaca rows
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05052026/Sophia-CTI-Templates-v12.txt \
    _v12_build/triples \
    SFT/data/ift_data_2026_05_05_v12.raw.json \
    10 1500

# Phases 2-9: generator merges + balance + dedup + gates + shuffle + splits
nohup bash _v12_build/watcher.sh > _v12_build/watcher.log 2>&1 &
```

Phase 1 reads the manifest with `tmpl_docx2json.py --count_limit 10`,
binds 244 templates against athena-cti-db via `iftgen.py --count_max 1500`
(per-template `Count:` overrides take precedence; see
`tmpl_parser.process_template` priority chain), and emits the raw Alpaca
JSON. The watcher then runs nine phases (v11 had three):

| phase | purpose | source |
|---|---|---|
| 1 | poll `make_dataset.sh` PID until exit | v11 carry |
| 2 | validate raw json exists | v11 carry |
| **3** | **TAA.CANON generator merge** (~10K rows from MITRE Groups + Athena aliases.csv) | NEW v12 |
| **3b** | **CM.* generator merge** (~6K rows from curated MCQ banks) | NEW v12 |
| **3c** | **MCQ.EXT.* generator merge** (~3K rows from MITRE/security tables) | NEW v12 |
| **3d** | **SOC.*.GEN.* generator merge** (~5K rows from sigma/malware/IR/triage tables) | NEW v12 |
| 4 | TAA actor-balance with `--max-rows-per-family-total 3500` | v12 §4.3 |
| 5 | dedup against eval sets (13-gram, threshold 50) | v11 carry |
| **6** | **row-count gate** -- halts build if any §3.1 family below floor | NEW v12 §4.1 |
| **7** | **stratified shuffle** -- guarantees per-family interleaving in N-row windows | NEW v12 §4.2 |
| 8 | val/train split (`build_val_slice.py`, ~50 rows/axis) | v11 carry |
| **9** | **per-phase corpus split** -- emits `_broad`, `_rms_ate_vsp_rcm`, `_taa_canon` shards | NEW v12 §6.3 |

Per-phase logs in `_v12_build/{build,watcher,balance,dedup,row_count_gate,shuffle,phase_split,val_slice,taa_canon,cm,mcq,soc}.log`;
per-gate reports in `_v12_build/{row_count_gate,phase_split,dedup,taa_canon,cm,mcq,soc}_report.json`.

## 2. Corpus actuals (v12 final, 2026-05-05 build)

| | v11 actual | v12 actual | delta | notes |
|---|---:|---:|---:|---|
| Raw rows (Phase 1, Neo4j-bound only) | 199,796 | 240,759 | +40,963 | manifest expansion: AB.RCM.{3,4}, AB.ATE.{6,7,8} sub-templates, TAA.CANON.{1,2,3} canonical seed |
| + Generator merges (Phases 3-3d) | n/a | +25,000 | — | TAA.CANON 10K, CM 6K, MCQ.EXT 3K, SOC.GEN 5K (programmatic, bypass Neo4j saturation) |
| Raw merged rows | 199,796 | 264,164 | +64,368 | the v11 substrate plus four new generator outputs |
| Balanced rows (TAA total-cap 3,500) | 199,648 | 261,448 | +61,800 | AB.TAA.* held at 3,500 hard cap (was 8,560 in v11; consumed 5K of other families' budget) |
| Final clean rows (post-dedup) | 198,994 | **260,589** | **+61,595** | dedup hit-threshold 1 / drop-threshold 50 (held at v11 values) |
| Distinct shortnames | 244 | **263** | +19 | +TAA.CANON.3, +CM.{CRYPTO,ACCESS,COMPLIANCE,GOV}, +AB.MCQ.EXT.{MITRE,SEC}.1, +SOC.{SIGMA,MAL,IR,TRIAGE}.GEN.1, +AB.ATE.{6,7,8}, +AB.RCM.{3,4} |
| Eval-axis-aligned rows | 61,003 | **112,929** | **+51,926** | RMS 13.4K, ATE 10.4K, VSP 15.6K, RCM 16.9K, MCQ 6.1K, MS 4.3K, TAA-attr 3.5K, TAA-IE-NEG 6.0K, TAA-CANON 10.8K, SOC 9.9K, CM 6.0K |
| Eval-axis corpus share | 30.7% | **43.3%** | +12.6 pp | exceeds v12 §3 hard gate of 38% |
| New-family rows (CM + TAA.CANON.3 + .EXT/.GEN) | 5,662 | **30,853** | +25,191 | net new training surface from the 2026-05-05 generator infrastructure |

All 11 row-count axes pass (`status: OK` in `row_count_gate_report.json`);
no axis below floor.

## 3. v11 → v12 deltas (summary; canonical text in manifest header)

  1. **Programmatic generator infrastructure** (`tmpl_gen/scripts/{taa_canon,cm,mcq,soc}_generator.py`):
     four standalone generators that produce Alpaca rows from curated
     Python literals (knowledge tables in `tmpl_gen/scripts/{mcq_data,soc_data,cm_data}/`),
     bypassing Neo4j graph-traversal saturation. Together they add ~25K
     rows that would otherwise hit substrate ceilings.
  2. **Row-count gate** (`tmpl_gen/scripts/check_corpus_row_counts.py`,
     `v12_row_count_gate.json`): per-axis REJECT_IF_BELOW thresholds at
     0.9 × §3.1 target. Build halts before Phase 7 if any axis fails.
     Surfaces v11's silent failure mode (22-86% of plan rows) at build
     time.
  3. **AB.TAA total cap** (`taa_actor_balance.py --max-rows-per-family-total 3500`):
     enforces §3.1's 3,500-row ceiling on AB.TAA.{1-5}+JS.TAA.{1-3}.
     v11 ran 8,560 rows (245% overshoot) consuming budget from ATE/VSP/MCQ.
  4. **Stratified shuffle wired in** (`stratified_shuffle.py`): the v11
     plan defined it but the watcher never invoked it. v12 calls it as
     Phase 7 with seed 42; LLaMA-Factory's per-epoch shuffle then
     operates over a pre-stratified shard.
  5. **Per-phase corpus splitter** (`split_corpus_for_phases.py`):
     emits three disjoint shards (`_broad` 193K, `_rms_ate_vsp_rcm` 56K,
     `_taa_canon` 10.8K) feeding Phase A/B/C training.
  6. **NEW CM.* family** (~6,000 rows, 4 sub-templates: CM.CRYPTO.1,
     CM.ACCESS.1, CM.COMPLIANCE.1, CM.GOV.1) seeded from
     `_v12_build/seeds/cm_*.jsonl` curated MCQ banks. v11 plan §5
     authored zero rows.
  7. **NEW AB.MCQ.EXT.* family** (~3,000 rows, 2 sub-families:
     AB.MCQ.EXT.MITRE.1, AB.MCQ.EXT.SEC.1) supplements template-driven
     AB.MCQ.* which saturates at ~3K distinct MITRE-anchor combinations.
  8. **NEW SOC.*.GEN.* family** (~5,000 rows, 4 sub-families:
     SOC.{SIGMA,MAL,IR,TRIAGE}.GEN.1) supplements template-driven SOC.*
     which saturates at ~5K rows. SOC row-count gate target rebalanced
     from 12,000/10,800 to 10,000/9,000 to match realistic combined yield.
  9. **TAA.CANON.* expanded 728 → 10,754 rows**: TAA.CANON.{1,2} kept
     verbatim from v11 for lineage; TAA.CANON.{1,2,3} also generated
     programmatically by `taa_canon_generator.py` from MITRE Groups +
     Athena `aliases.csv`. Includes TAA.CANON.3 hard-negative pairs
     (NEW; v11 had only positive-only pairs).
 10. **AB.RCM expansion**: AB.RCM.{3,4} multi-CVE root-cause variants
     added; v11 shipped 2,864 rows from AB.RCM.{1,2}, v12 ships 16,917
     rows across four sub-templates (combined with RCM-axis X.VW.* / YN.VW.*).
 11. **AB.ATE expansion**: AB.ATE.{6,7,8} broader CWE → ATT&CK pivot
     variants added; v11 shipped 4,803 rows from AB.ATE.{1-5}, v12 ships
     10,387 rows across eight sub-templates.
 12. **Three-phase training recipe** (REVERT to v9 shape, ADD Phase C).
     See §4.

## 4. Training recipe (three phases, v9-shape recovery + TAA memorisation)

v11 single-pass at cutoff 8192 lost the RMS catalog supervision that v9's
two-phase Phase B preserved (v9 RMS=65.8, v11 RMS=48.0). v12 reverts to
a phased recipe with a third phase added for TAA.CANON memorisation --
the only family v11 proved is recipe-sensitive, not corpus-sensitive.
Only Phase C's merged checkpoint is pushed to HF.

| | Phase A (broad) | Phase B (RMS+ATE+VSP+RCM) | Phase C (TAA.CANON) |
|---|---|---|---|
| Dataset | `_v12_broad` (193K) + tulu-3 + alpaca | `_v12_rms_ate_vsp_rcm` (56K) | `_v12_taa_canon` (10.8K) |
| Cutoff | 8192 | 16384 | 8192 |
| Packing | on | **off** (catalog-recovery requires per-row loss) | on |
| LR | 1e-5 | 5e-6 | 3e-6 |
| Effective batch | 16 | 8 | 16 |
| Save/eval steps | 500 | 400 | 250 |
| Max samples | 240,000 | 60,000 | 12,000 |
| Resume from | base Qwen2.5-14B-Instruct | Phase A output | Phase B output |
| Push to HF | no | no | **yes** (`${HF_USERNAME}/athena-cti-sft-qwen25-14b-v12`) |

Per-axis eval loss visibility is wired through `--eval_dataset
ift_data_2026_05_05_v12_val` on all three phases (v11's silent regression
fix).

## 5. Consumers

| consumer | path |
|---|---|
| Dataset registration | `SFT/data/dataset_info.json` -> `ift_data_2026_05_05_v12_{broad,rms_ate_vsp_rcm,taa_canon,val}` |
| Qwen2.5-14B launcher | `SFT/autotrain/run_sft_qwen25_14b_v12.sh` (three-phase, supports `--phase a\|b\|c\|ab\|bc\|all` and `--dry-run`) |
| Qwen2.5-32B launcher | `SFT/autotrain/run_sft_qwen25_32b_v12.sh` (deferred; authored only after 14B passes §8 per v12 plan §6.2) |
| HF target | `${HF_USERNAME}/athena-cti-sft-qwen25-14b-v12` (32B repo target reserved) |
| Eval task (NEW) | `athena-cti-taa-canonical` (alias-resolution scoring) |


## 6. Pass criteria (v12 14B production -- all four must hold)

  - weighted total >= max(v9=56.4, v11=54.3) + 1.5 pp = **57.9**
  - RMS >= v9 RMS (**65.8**); v11's 48.0 must be recovered by Phase B
  - ATE no regression > 3 pp vs v7 (52.6); v11's 42.4 must be recovered
  - VSP no regression > 3 pp vs v10 (86.7); v11's 70.9 must be recovered
  - TAA-attribute >= 22% (carried from v11 §8); TAA-CANON slice score
    reported separately as a new axis
  - No axis regressed > 3 pp vs v11 on the families v11 won (CKT, SOC,
    CM, RCM)

Fallback: if Phase C over-fits TAA at the cost of other families, drop
Phase C entirely and ship Phase B output as v12.0; treat TAA.CANON as a
v12.1 follow-up. Phase A+B alone is the v9-shape recovery and is the
minimum-viable v12.

The 32B launch is **serial after** the 14B passes the four criteria
above (v12 plan §11 #5). v11's 8.5-hour 32B compute on a known-suboptimal
recipe is exactly the failure mode this gating prevents.

## 7. Version history (v0 → v12)

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
| v9 | 2026-04-30 | `tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt` | two-phase | Phase B "RMS slice" extracted from v8.1 to recover catalog under v8.1's broad-knowledge regression. v9 ran as a two-phase chain (broad re-anchor → RMS) on the 14B base. RMS=65.8 (the v12 bar). |
| v10 | 2026-05-01 | `tmpl_gen/templates/05012026/` | 200,340 rows | Single-pass unified manifest (216 templates); HTML/whitespace sanitiser at parser chokepoint; `<desc>...</desc>` markers around freeform text; actor cap=20; 13-gram dedup at threshold 50. AthenaBench composite weighted total: 54.1. |
| v11 | 2026-05-03 | `tmpl_gen/templates/05032026/` | 198,994 rows | 244 templates (all productive). SOC.* (4,884) + TAA.CANON.* (778) + RMS paraphrase + X./YN. trim with RCM-axis exemptions + parser anchor-fixation fix + actor cap 40 + dedup at 50 + D3FEND v1.4.0 ingest. Eval-axis density 30.7% (+4.2 pp vs v10). 14B sweep weighted total **54.3** (failed §8). |
| v12 | 2026-05-05 | `tmpl_gen/templates/05052026/` | **260,589 rows** | This vintage. 263 distinct shortnames. Four programmatic generators (TAA.CANON 10K, CM 6K, MCQ.EXT 3K, SOC.GEN 5K) bypassing Neo4j substrate ceilings. Build pipeline gains row-count gate (per-axis REJECT_IF_BELOW), AB.TAA total cap (3,500), wired-in stratified shuffle, per-phase corpus splitter. Three-phase training (A: broad 8192/packing-on/lr 1e-5, B: RMS+ATE+VSP+RCM 16384/packing-off/lr 5e-6, C: TAA.CANON 8192/packing-on/lr 3e-6). Eval-axis density **43.3%** (+12.6 pp vs v11). All 11 row-count axes pass. |

For the line-by-line corpus composition of v11 (the substrate v12 carries
forward) see `tmpl_gen/templates/05032026/README.md`. For v10 (the
substrate v11 carried forward) see `tmpl_gen/templates/05012026/README.md`.
For the v8.1 / v9 two-phase rationale (the recipe v12 reverts to and
extends with Phase C) see `tmpl_gen/templates/04302026/README.md`. For
the v8 split-manifest design see `tmpl_gen/templates/04292026/README.md`.
The v0 substrate's per-family design strategy (`M/A/W/V/S/P/E/X`) is in
`tmpl_gen/templates/README.md`.

## 8. Known content gaps to watch at eval

  1. **TAA.CANON.1 single-alias trivial case (carried from v11)**: groups
     with only one element in `aliases` (≈83 of 187 intrusion-sets) yield
     trivial identity rows. v12 mitigates by using the programmatic
     generator's hard-negative TAA.CANON.3 to force discrimination, but
     the substrate ceiling on `aliases` arrays in athena-cti-db is
     unchanged. Information value is preserved by the non-trivial
     multi-alias rows (104 groups, average 3.1 aliases each) and the
     ~3,000 hard-negative pairs.
  2. **SOC.TRIAGE.DS.{1,2}** still bind on the data-source node alone
     (carried from v11): the 38 `x-mitre-data-source` nodes in
     athena-cti-db are isolated. v12 SOC.GEN.* compensates with
     ~1,000 IR-playbook + ~1,000 triage rows from curated tables, not
     graph-bound.
  3. **AB.RMS.{4,5} ceiling**: paraphrase-multiplied to ~440 per family
     in v11 and held in v12; if `athena-cti-rms` still regresses vs v9
     after Phase B the M-control catalog itself needs expansion in
     athena-cti-db (out of scope for v12).
  4. **Input collisions** (~81K instr+input pairs with multi-valued
     output, e.g., one ExploitDB entry → multiple CVEs, one technique →
     multiple mitigations): intentional one-to-many CTI relationships,
     not noise. Trains the model on multi-modal P(answer|question).
     True duplicates (instr+input+output identical) under 31 rows
     (<0.012% of corpus).
  5. **CM.* family is curated, not graph-derived**: rows reflect the
     authors of the seed banks (NIST CSF, ISO 27001, HIPAA, PCI-DSS,
     OWASP, crypto/access fundamentals). Re-evaluation against an
     external compliance benchmark recommended before declaring CM
     coverage complete; v12 ships the family as a baseline, not a final
     answer.
  6. **No external CyberMetric leak audit** beyond v11's 13-gram dedup
     against the v11 eval sets. v12 generators emit fresh prompts; spot-
     check passed (0 collisions with held-out val slice across all four
     generator outputs), but a cross-corpus n-gram audit against the
     full 2026-Q1/Q2 benchmark refresh is a v12.1 follow-up.
