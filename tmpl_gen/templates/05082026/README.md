# Sophia CTI Templates — v14 / v14.1 (May 8, 2026 vintage)

Multi-checkpoint SFT manifest. v14 is the first vintage authored as a
**per-axis narrow-drilling experiment**: the v12 corpus and recipe are
restored as the substrate (proven 57.3 weighted total), v13's
best-of-vintage merges are abandoned, and two parallel late-stage
narrow drills (D-RMS, D-TAA) are added to test whether axis-specific
phases can recover the v9 RMS peak (65.8) and lift TAA Classic above
the **untrained Qwen2.5-14B-Instruct baseline (52.0)**, which every
SFT vintage we have shipped (v11, v12, v13) has regressed below.

The vintage produces **four HF checkpoints** from a single training
chain so the effects of each phase can be attributed cleanly:

| checkpoint | answers the question |
|---|---|
| `athena-cti-sft-qwen25-14b-v14-ab` | Did dropping RMS+TAA from Phase B fix v13's regression? |
| `athena-cti-sft-qwen25-14b-v14-rms` | Does a v9-shape RMS-only narrow drill recover RMS without damaging adjacent axes? |
| `athena-cti-sft-qwen25-14b-v14-taa` | Does a v9-shape TAA-only narrow drill lift TAA Classic above the 52.0 base-model floor? |
| `athena-cti-sft-qwen25-14b-v14` | Production candidate (D-RMS → D-TAA chained on top of AB) |

The manifest header (`Sophia-CTI-Templates-v14.txt`, lines 1-180) is
the canonical change log for the v13 → v14 deltas; this README is the
supplementary lineage document and carries the full v0 → v14 running
dialogue at the bottom (Section 10) so the manifest+README pair is a
complete corpus reference even when prior-vintage directories are not
consulted.

```
05082026/
  Sophia-CTI-Templates-v14.txt   self-contained v14 manifest (v13 manifest verbatim with AB.RMS.4{a..j}+AB.RMS.5{a..j} replaced by v9's narrow AB.RMS.4+AB.RMS.5)
  v14_plan.txt                   master plan document (v13 post-mortem, design notes, sign-off)
  v14_row_count_gate.json        per-axis REJECT_IF_BELOW thresholds
  README.md                      this document (v13 post-mortem in §2; running dialogue v0 -> v14 in §10)
```

## 1. Why v14 exists (one-paragraph overview)

v13 shipped at weighted total **54.5**, regressing -2.8 pp vs v12 and
under-shooting every prior 14B vintage including v7 (the original
Llama-3.1-8B baseline derived corpus on Qwen2.5-14B). The v13
hypothesis -- that compositing best-of-vintage templates from v7/v8/v9
on top of the v12 build pipeline would add per-axis gains while the
recipe revert to v9 shape (cutoff 8192, packing ON) would restore RMS
-- was falsified on **all five axes** the composition targeted, with
the largest regression (-7.5 pp) on RMS, the very axis the recipe
revert was designed to recover. The v13 README's central premise --
"each vintage's per-axis peak can be re-mixed in a single corpus and
held by the v9 recipe" -- is wrong: every SFT vintage we have shipped
underperforms its constituent peaks on the axis those peaks belong to,
and v13 demonstrated that this is recipe/dilution interaction, not a
template-quality problem.

v14 is the controlled experiment that disentangles these effects.
**Phase A** (broad re-anchor) and **Phase B** (3-axis long-context
drill, ATE+VSP+RCM only) restore the v12 production substrate without
RMS dilution. **Phase D-RMS** and **Phase D-TAA** are *parallel*
narrow drills (each branched from the v14-AB checkpoint) using the
verbatim v9 recipe, each on a single axis. The four resulting
checkpoints answer four falsifiable questions in a single training
cycle.

## 1.1 v14.1 hot-fix supersession (cutoff 4096; corpus/topology held)

v14 Phase A live measurement at step 23000 / 49336 (47%) showed
~12 h of wall-time still ahead at 2,927 tok/s and ~1.78 s/it. A
length-distribution audit of all four v14 IFT shards returned the
following p99 token counts:

| shard           |     n   | median |   p95 |   p99 |   max |
|-----------------|--------:|-------:|------:|------:|------:|
| `_v14_broad`    | 193,703 |    167 |   625 |   931 | 2,020 |
| `_v14_ate_vsp_rcm` | 32,810 | 339 |   992 | 1,416 | 16,706* |
| `_v14_rms`      |  12,608 |    666 | 1,823 | 2,177 | 2,667 |
| `_v14_taa`      |  32,783 |    180 |   332 |   340 |   802 |

(*single outlier; next-longest sample is well under 2,000.)

Every shard's p99 is below 4,096 tokens, meaning the v14 cutoffs
(16,384 in A/B; 8,192 in D-RMS/D-TAA/production) were spending
~95% of every step on padding rather than on real tokens. **v14.1
is a launcher-only hot-fix** that flips two performance-only knobs
and changes nothing else about the recipe:

  1. `--cutoff` lowered to 4,096 in every phase (was 16,384 in A/B,
     8,192 in D-RMS/D-TAA/production).
  2. `--disable_gradient_checkpointing True` added to `EXTRA_COMMON`
     so it applies uniformly to all five phases. At cutoff 4,096 /
     per_device_batch 1 / ZeRO-3 sharded across 8x80GB H100s,
     activation memory fits comfortably under 30 GB per GPU; GC was
     inherited from v14 (where cutoff 16,384 made it mandatory) but
     is vestigial here. Disabling it removes the backward-pass
     recomputation step (~15-30% throughput gain on top of the
     cutoff reduction). GC is a forward/backward implementation
     detail, not part of the model math: gradients, loss values,
     optimiser trajectory, and final weights are unchanged.

Corpus, topology (5-pass sequential), learning rates, effective
batch sizes (8 for A/B, 16 for D-*), packing flags (OFF in A/B,
ON in D-*), max-samples, save/eval cadences, and resume chaining
are all preserved verbatim from v14. Step counts per phase are
bit-identical to v14 in step-space, so the optimiser trajectory
and LR schedule are unchanged; only step-time shrinks. Loss curves
at any given step number are directly comparable to v14's.

  - Launcher        : `SFT/autotrain/run_sft_qwen25_14b_v14_1.sh`
  - HF push targets : `${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14p1{-ab,-rms,-taa,}`
                      (`v14p1` token replaces v14's `v14` in the
                      repo-name suffix; underscores would break HF
                      tooling, dots are flaky in some clients, so
                      `v14p1` is the chosen on-the-wire spelling)
  - Save dirs       : `SFT/saves/Qwen_Qwen2.5-14B-Instruct/full/v14_1_phase_{a,b,d_rms,d_taa,prod}_${TIMESTAMP}`
                      (the local-filesystem identifier uses `v14_1`)
  - Datasets        : `ift_data_2026_05_08_v14_{broad,ate_vsp_rcm,rms,taa,val}`
                      -- byte-identical to v14; same JSON files

Estimated total wall-time after both knobs: ~9-11 h (vs v14's ~32 h
projected; cutoff alone gave ~10-12 h, GC-off trims a further
15-30%), dominated by Phase A. Phase B / D-RMS / D-TAA / production
each land in the ~45 min - ~1.5 h band per phase.

The original v14 launcher (`run_sft_qwen25_14b_v14.sh`) and its
HF repo namespace (`...-v14{-ab,-rms,-taa}`) remain in-repo for
reproducibility; v14.1 is the production execution path. Sections
2-11 below describe v14 as designed; the hot-fix overrides only
the cutoff values quoted in §6 and the launcher/repo identifiers
quoted in §7 (annotated inline in those sections).

The narrow-drilling experiment (the questions in §1, §2, §9) is
unchanged: v14.1 still produces four HF checkpoints answering the
four falsifiable questions, just on the `v14p1` namespace and
faster.

## 2. v13 14B post-mortem (the regression analysis)

### 2.1 Headline numbers

| axis (combined) | v9 14B | v11 14B | v12 14B | **v13 14B** | v13 vs v12 | v13 vs target |
|---|---:|---:|---:|---:|---:|---:|
| **weighted total** | 56.4 | 54.3 | **57.3** | **54.5** | **-2.8** | -4.0 pp (target ≥58.5) |
| **RMS** | **65.8** | 48.0 | 63.3 | **55.8** | **-7.5** | -10.8 pp (target ≥66.6) |
| ATE | 52.6 | 42.4 | **57.0** | 56.5 | -0.5 | +1.0 pp (target ≥55.5) |
| VSP | 86.7 | 70.9 | 84.6 | **85.3** | +0.7 | +0.1 pp (target ≥85.2) |
| RCM | ~62 | ~62 | **69.0** | 67.8 | -1.2 | +5.8 pp (target ≥62.0) |
| CKT | ~70 | 63.4 | **70.1** | 69.5 | -0.6 | -6.6 pp (target ≥76.1) |
| SOC | ~32 | **44.7** | 39.3 | 38.6 | -0.7 | -6.1 pp (target ≥44.7) |
| TAA Classic strict | ~16 | 16.0 | 11.0 | 7.0 | -4.0 | n/a |
| TAA Classic plausible | ~50 | 83.0 | 60.0 | 80.0 | +20.0 | n/a |
| TAA Classic **combined** | ~33 | 49.5 | 35.5 | **43.5** | +8.0 | n/a |
| TAA Canonical combined | n/a | n/a | 38.7 | 24.8 | **-13.9** | n/a |

Five of the six §6 hard gates failed: weighted total (-4.0 pp),
RMS (-10.8 pp), CKT (-6.6 pp), SOC (-6.1 pp). Only ATE, VSP, and RCM
held. The 32B launch (gated on the 14B passing) does not happen.

### 2.2 The TAA-vs-base regression (the central indictment)

| model | TAA Classic accurate | plausible | **combined** |
|---|---:|---:|---:|
| **Qwen/Qwen2.5-14B-Instruct (base, no SFT)** | **16.0** | **88.0** | **52.0** |
| asg-ai/athena-cti-sft-qwen25-14b-v11 | 16.0 (=) | 83.0 (-5.0) | 49.5 (-2.5) |
| asg-ai/athena-cti-sft-qwen25-14b-v12 | 11.0 (-5.0) | 60.0 (-28.0) | 35.5 (-16.5) |
| asg-ai/athena-cti-sft-qwen25-14b-v13 | 7.0 (-9.0) | 80.0 (-8.0) | 43.5 (-8.5) |

**Every SFT vintage has regressed on TAA Classic combined vs the
untrained base model.** The v12 → v13 change (Phase C drop, MISP
expansion, two-phase v9-shape recipe) recovered 8.0 pp of plausible
attribution but lost 4.0 pp of strict precision; the net is still
8.5 pp below what the base model does *with no TAA-specialised
training at all*. This is the central question v14 must answer: is
there any training configuration that produces a TAA Classic
combined >= 52.0, or does the base model's broad pretraining beat
anything our 23K rows of curated TAA content can teach?

### 2.3 Falsified hypotheses (do not retry without protocol change)

  1. **"v12's RMS template superset (which contains v9 as a subset)
     carries the v9 RMS peak under the v9 recipe"** -- v13's RMS
     templates were v12's expanded RMS section (AB.RMS.{1,2,3a..3h,
     4a..4j,5a..5j,6} + JS.RMS.{1..8}), which v13's own header
     audit confirmed is a strict superset of v9's narrower set
     (v9 had AB.RMS.4 and AB.RMS.5 as single templates with
     Count=600 each; v12 split them into 10 paraphrase variants
     each with Count=50). v13 ran those v12-superset RMS templates
     under v9's training recipe (cutoff 8192 / packing ON / batch
     16) and RMS still landed at 55.8 -- 10.0 pp below v9's 65.8.
     The "strict superset of v9 templates + v9 recipe = v9 result"
     hypothesis is falsified. Either the .4{b..j} + .5{b..j}
     paraphrase variants are net-negative on RMS supervision, or
     the failure was elsewhere (Phase B axis dilution, see #2).
     v13's MCQ templates were v8 superset; CKT regressed -0.6 pp.
     v13's VSP templates were v10 (the 86.7 vintage); VSP held at
     85.3, only the unaffected baseline. The "per-axis peak
     templates carry per-axis peaks" hypothesis is falsified for
     RMS and MCQ; held only for VSP (which has a content ceiling
     on the eval set).
  2. **"v9 recipe revert recovers RMS"** -- v13 reverted Phase B to
     v9 hyperparameters (cutoff 8192, packing ON, batch 16) AND
     used v12's RMS template section (which textually contains
     v9's RMS templates plus 18 paraphrase variants) AND ran them
     in a 5-axis Phase B (RMS+ATE+VSP+RCM+SOC). RMS landed -10.0 pp
     below the v9 result. v9 ran a 1-axis Phase B (RMS only) on
     the narrower v9 template set. v13 changed two variables at
     once -- axis count (1 -> 5) and template surface (v9 narrow ->
     v12 superset) -- so the failure cannot be cleanly attributed
     to either alone. v14 disentangles these: Phase B drops to
     3 axes (no RMS, no SOC), Phase D-RMS isolates RMS to 1 axis
     AND restores v9's narrow template set (drop the .4{b..j} +
     .5{b..j} paraphrase variants).
  3. **"Adding canonical TAA content (TAA.CANON, MISP) reduces TAA
     regression vs base"** -- v12 introduced TAA.CANON (~11K rows) and
     dropped TAA Classic combined from v11's 49.5 to 35.5. v13 added
     MISP CC-0 expansion (+12K rows) and raised TAA Classic combined
     to 43.5 (still -8.5 pp vs base) BUT regressed TAA Canonical from
     v12's 38.7 to 24.8 (-13.9 pp on the very task the canonical
     content was added to support). Adding canonical content is at
     best a wash on TAA Classic and **net-negative on TAA Canonical
     itself** when the canonical slice is buried in a broad shard.
  4. **"Five-axis Phase B is equivalent to v9's one-axis Phase B at
     the same recipe"** -- v13 ran RMS+ATE+VSP+RCM+SOC together in
     Phase B at the v9 recipe; RMS regressed -10.0 pp vs the v9 result
     produced by the same templates and recipe in a 1-axis Phase B.
     Multi-axis Phase B at a narrow-recipe shape causes per-axis
     gradient dilution roughly proportional to the axis count.

### 2.4 Root cause hypotheses (ranked, with v14 mechanism)

| rank | root cause | v13 evidence | v14 mechanism |
|---|---|---|---|
| 1 | Phase B 5-axis dilution | RMS -10 pp despite v9-recipe and v12-superset (which contains v9) RMS templates | Phase B = ATE+VSP+RCM only (3 axes); RMS & TAA isolated to dedicated narrow phases |
| 2 | Long-context recipe (8192/packing ON) hurts long-context axes when applied broadly | v13 ATE -0.5 pp, RCM -1.2 pp, CKT -0.6 pp vs v12 | Phase A keeps v12 broad recipe (16384, packing OFF); narrow phases use 8192/packing ON only on their slice |
| 3 | TAA training is net-negative on Classic | All SFT vintages below base 52.0 | Phase D-TAA dedicated drill on full TAA slice; if it can't cross 52.0 the v15+ recipe drops TAA training entirely |
| 4 | Canonical content (CANON+MISP) buried in broad shard regresses on its own task | v13 TAA-CANON 24.8 vs v12 38.7 with 2x more canonical rows | Phase D-TAA includes ALL TAA content (Classic + Canonical + MISP) so the canonical slice gets dedicated supervision |
| 5 | SOC excluded from Phase B in v12 caused -5.4 pp; v13 included it in 5-axis Phase B and still lost -0.7 pp | v13 SOC 38.6 vs v12 39.3 | v14 holds v12: SOC in Phase A only; SOC narrow drill deferred to v15 |


## 3. Build pipeline (v12 substrate; v13 watcher with new shard plan)

```bash
# Phase 1: template -> triples -> Alpaca rows
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05082026/Sophia-CTI-Templates-v14.txt \
    _v14_build/triples \
    SFT/data/ift_data_2026_05_08_v14.raw.json \
    10 1500

# Phases 2-9 (carry v13 watcher; updated shard plan in §3.1 delta 2)
nohup bash _v14_build/watcher.sh > _v14_build/watcher.log 2>&1 &
```

The eleven-phase v13 watcher (`_v13_build/watcher.sh`) is carried
verbatim except for **Phase 9 (per-phase corpus split)**, which now
emits **four shards** instead of two:

| phase | purpose | source |
|---|---|---|
| 1 | poll `make_dataset.sh` PID until exit | v13 carry |
| 2 | validate raw json exists | v13 carry |
| 3 | TAA.CANON generator merge (MITRE seed) | v13 carry |
| 3b | CM.* generator merge (~6K rows) | v13 carry |
| 3c | MCQ.EXT.* generator merge (~3K rows) | v13 carry |
| 3d | SOC.*.GEN.* generator merge (~5K rows) | v13 carry |
| 3e | MISP TAA generator merge (~12K rows) | v13 carry |
| 4 | TAA actor-balance with `--max-rows-per-family-total 3500` | v13 carry; cap raised to 4500 (see §3.1 delta 3) |
| 5 | dedup against eval sets (13-gram, threshold 50) | v13 carry |
| 6 | row-count gate -- halts if any §4 family below floor | v13 carry; thresholds in `v14_row_count_gate.json` |
| 6b | licence-allowlist gate | v13 carry |
| 7 | stratified shuffle | v13 carry |
| 8 | val/train split (`build_val_slice.py`, ~50 rows/axis) | v13 carry |
| **9** | **per-phase corpus split -- FOUR shards** (`_broad`, `_ate_vsp_rcm`, `_rms`, `_taa`) | CHANGED v14 §3.1 |

Per-phase logs in `_v14_build/{build,watcher,balance,dedup,
row_count_gate,licence_gate,shuffle,phase_split,val_slice,
taa_canon,misp,cm,mcq,soc}.log`; per-gate reports in
`_v14_build/{row_count_gate,licence_gate,phase_split,dedup,
taa_canon,misp,cm,mcq,soc}_report.json`.

### 3.1 Build deltas vs v13

  1. **Manifest substrate held; v13 RMS catalog-lookup expansion
     reverted (subtractive, hybrid slice)**: v14 manifest = v13
     manifest verbatim, EXCEPT v13's expanded RMS catalog-lookup
     section (`AB.RMS.4{a..j}` + `AB.RMS.5{a..j}`, 20 paraphrase
     variants at Count=50 each) is replaced by v9's narrow
     `AB.RMS.4` + `AB.RMS.5` (2 templates at Count=600 each) pulled
     byte-identically from
     `tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt`.
     For the catalog-lookup `.4` / `.5` block, v9's RMS section is
     a strict SUBSET of v12's/v13's, so this is a subtractive
     change: drop the 18 paraphrase variants and concentrate the
     same row budget on the v9-shape templates. The remaining RMS
     templates (`AB.RMS.{1, 2, 3a..3h, 6}`) keep v13 bodies; of
     these only `.6` is byte-identical to v9, while `.{1, 2, 3a..3h}`
     differ from v9 only in v10's `<desc>...</desc>` parser-routing
     wrappers around `{ap.description}` (see §5 delta 3 for the
     hybrid-slice rationale). All other v13 content held (v12
     substrate + TAA.CANON.{1,2,3} + MISP.CANON.{1,2,3} + CM.* +
     MCQ.EXT.* + SOC.GEN.*).
  2. **Phase 9 four-shard split**: `split_corpus_for_phases.py`
     gains `--four-shard` mode that emits `_broad` (everything
     except the four narrow-drill axes), `_ate_vsp_rcm` (long-context
     drill), `_rms` (hybrid RMS slice per delta 1: v9-narrow .4 + .5,
     v13-with-`<desc>` .{1, 2, 3a..3h}, v9 == v13 .6 + JS.RMS.*),
     and `_taa` (AB.TAA.* + JS.TAA.* + TAA.CANON.* + MISP.CANON.*,
     ~23K rows total). Each row appears in EXACTLY ONE shard except
     SOC.* which lives in `_broad` only (no SOC narrow drill in v14).
  3. **TAA total cap raised to 4,500** (was 3,500 in v12/v13):
     justification is that the cap previously protected the broad
     shard from TAA over-representation, but in v14 the TAA slice
     trains in its own dedicated phase where the cap's purpose
     (preventing dilution of broad knowledge) does not apply. The
     cap remains at 4,500 to preserve actor diversity (the v12 lower
     bound was set when AB.TAA.{1-5} could compress to <100 distinct
     actors at high counts; raising the per-template ceiling to ~700
     each preserves the v12 actor-balance audit semantics).
  4. **No new generators, no new Source: tags, no licence-gate
     changes**: v14 is a recipe/sharding experiment on the v13
     content substrate. All build infrastructure carries forward
     unchanged except Phase 9's shard plan and the TAA cap.

## 4. Corpus actuals (v14 final, target shape; populated post-build)

Targets below come from v14_plan.txt §3. Actuals will be filled in
once the watcher finishes; v13 actuals are shown for comparison.

| | v13 actual | v14 target | delta | notes |
|---|---:|---:|---:|---|
| Raw rows (Phase 1, Neo4j-bound only) | 204,695 | ~204,000 | ~0 | manifest = v13 verbatim with RMS catalog-lookup section reshaped (v13: 20 templates × Count=50 = 1,000 raw rows for catalog-lookup; v14: 2 templates × Count=600 = 1,200 raw rows; net change ~+200 rows pre-dedup) |
| + Generator merges (Phases 3-3e) | +37,000 | +37,000 | 0 | unchanged from v13 |
| Raw merged rows | 241,695 | ~241,900 | ~+200 | |
| Balanced rows (TAA total-cap 4,500) | 241,448 | ~242,900 | +1,452 | TAA cap raised from 3,500 to 4,500 per §3.1 delta 3 (~+1K TAA rows admitted) |
| Final clean rows (post-dedup) | **240,759** | **~242,000** | **~+1,200** | dedup hit-threshold 1 / drop-threshold 50 (v9/v11/v12/v13 carry); v9-narrow templates have higher per-template Count so dedup-survival is comparable |
| Distinct shortnames | 263 | **245** | **-18** | v13 had AB.RMS.4{a..j}+AB.RMS.5{a..j} (20 shortnames); v14 has AB.RMS.4+AB.RMS.5 (2 shortnames). Net delta: -18. All other shortnames held verbatim from v13. |
| Eval-axis-aligned rows | 112,929 | **~114,000** | ~+1,000 | RMS row count holds ±1K (v9-narrow at higher Count vs v12-superset at lower Count); TAA cap raise contributes +1K; rest unchanged |
| Eval-axis corpus share | 43.3% | **~47%** | +3.7 pp | clears v14 §3 hard gate of 40% (driven by held v13 axis density, NOT by net RMS row growth) |

Per-shard row counts (ACTUAL after 2026-05-07 build):

| shard | actual rows | composition |
|---|---:|---|
| `_v14_broad` | 193,703 | M/A/W/V/S/P/E/X.* + JS.* (non-axis) + AB.MCQ.* + MCQ.EXT.* + SOC.* + SOC.GEN.* + CM.* (everything not in the 3 axis shards) |
| `_v14_ate_vsp_rcm` | 32,810 | AB.ATE.{1..3} + AB.VSP.{1,2} + AB.RCM.{1,2} + V.CPE + (JS counterparts) |
| `_v14_rms` | 12,608 | hybrid RMS slice (see §3.1 delta 1): AB.RMS.4 + AB.RMS.5 byte-identical to v9 (Count=600 each, but each generates only ~44 rows because the M-control catalog is the binding constraint -- identical to v9's actual catalog-lookup row count); AB.RMS.6 byte-identical to v9; AB.RMS.{1, 2, 3a..3h} v13 bodies (v10 `<desc>` wrappers); JS.RMS.{1..8} byte-identical to v9. Drops v13's AB.RMS.4{b..j}+AB.RMS.5{b..j} 18 paraphrase variants (~792 rows). |
| `_v14_taa` | 32,783 | AB.TAA.{1..5} + AB.TAA.IE.{1,2} + AB.TAA.NEG.1 + JS.TAA.{1..3} + JS.TAA.IE.1 + JS.TAA.NEG.1 + TAA.CANON.{1,2,3} + MISP.CANON.{1,2,3}. IE/NEG variants (~5,900 rows) routed to TAA shard via the `AB.TAA.*`/`JS.TAA.*` glob in splitter rule 2 -- consistent with §3.4 D-TAA "dedicated TAA training phase" framing. |

The four shards are disjoint by construction (each row appears in
exactly one shard); the `phase_split_report.json` artifact records
the row count per shard plus the pre-split clean-corpus row count to
detect leakage.

## 5. v13 → v14 deltas (summary; canonical text in manifest header)

  1. **Drop best-of-vintage composition framing**: v13's manifest
     header documented "best-of-vintage" but in practice held v12's
     template bodies verbatim (which the v13 author audited as a
     strict superset of v9/v10/v11 per-axis peaks). v13's RMS
     regression vs v9 (-10 pp despite v9-recipe and v12-superset
     RMS templates) demonstrated that strict-superset templates +
     same recipe + multi-axis Phase B do NOT recover the v9 RMS
     peak. v14 holds the v13 substrate (which is the v12 substrate
     plus v13's TAA.CANON+MISP+CM+MCQ.EXT+SOC.GEN generator
     additions) and SUBTRACTS v12's RMS catalog-paraphrase
     expansion to test whether the .4{b..j}+.5{b..j} variants are
     net-negative under narrow-phase training.

  2. **Two-phase recipe → four-phase recipe with parallel narrow
     drills**: v13 ran A (broad+canon, 8192/packing-on/lr 1e-5) →
     B (RMS+ATE+VSP+RCM+SOC, 8192/packing-on/lr 5e-6, batch 16). v14
     runs A (broad, 16384/packing-off/lr 1e-5; v12 Phase A recipe)
     → B (ATE+VSP+RCM only, 16384/packing-off/lr 5e-6; v12 Phase B
     recipe minus RMS minus SOC) → **D-RMS** (v9-recipe narrow drill
     on RMS slice only) **PARALLEL WITH** **D-TAA** (v9-recipe narrow
     drill on TAA slice only). Both narrow drills branch from the AB
     checkpoint; one production candidate chains them.

  3. **v9 narrow RMS catalog-lookup section re-bound (subtractive,
     hybrid slice)**: v13's `AB.RMS.4{a..j}` (10 paraphrase variants
     × Count=50) and `AB.RMS.5{a..j}` (10 paraphrase variants ×
     Count=50) are replaced by v9's `AB.RMS.4` (Count=600) and
     `AB.RMS.5` (Count=600), pulled byte-identically from
     `tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt`
     and re-rendered against the current Neo4j (which has more
     M-control content than the v9-era graph). v14's RMS slice is
     therefore a HYBRID: `AB.RMS.4` + `AB.RMS.5` bodies are
     byte-identical to v9; `AB.RMS.6` is also byte-identical to v9
     (no v9 -> v13 drift); `AB.RMS.{1, 2, 3a..3h}` keep v13's
     bodies, which differ from v9 only in that v10 wrapped the
     freeform `{ap:attack-pattern.description}` field in
     `<desc>...</desc>` parser-routing markers (semantic intent
     unchanged; affects parser-side text routing, not user-visible
     scenario text). `JS.RMS.{1..8}` are byte-identical between v9
     and v13/v14. The hypothesis tested by D-RMS is "v9-narrow
     `.4` + `.5` catalog-lookup templates + v9 recipe in a 1-axis
     Phase D should reproduce v9's RMS=65.8"; the `<desc>` wrappers
     on the scenario-driven `.1`/`.2`/`.3` templates are held
     constant with the rest of the v14 corpus to avoid introducing
     a second confound. If RMS still under-shoots, the regression
     is in current Neo4j contamination of the M-control catalog or
     in carry-loss from Phase A+B -- not in templates or recipe.

  4. **Full TAA slice in dedicated phase**: D-TAA includes the full
     TAA family -- AB.TAA.{1..5} + JS.TAA.{1..3} (Classic), TAA.CANON.{1,2,3}
     (Canonical from MITRE seed), MISP.CANON.{1,2,3} (MISP CC-0
     expansion). ~23K rows total. Tests whether dedicated training on
     the full TAA surface can lift TAA Classic combined above the
     untrained-base-model floor of 52.0 (something no SFT vintage has
     achieved). If it can't, v15+ permanently drops TAA training and
     all TAA-specialised content; the eval becomes "report what the
     base model does" for TAA.

  5. **Multi-checkpoint protocol** (NEW v14): the launcher saves and
     pushes FOUR HF checkpoints from a single training chain:
     `v14-ab` (after Phase A+B), `v14-rms` (after D-RMS branched
     from v14-ab), `v14-taa` (after D-TAA branched from v14-ab),
     and `v14` (production candidate: D-TAA chained on top of D-RMS,
     in that order so RMS damage from a TAA narrow drill is minimised).
     Each checkpoint runs through the full benchmark sweep so the
     contribution of each phase is measurable independently.

  6. **Drop Phase C entirely**: v12 Phase C (low-lr TAA.CANON
     memorisation) was abandoned in v13; v14 holds. The Phase D-TAA
     narrow drill at v9 recipe replaces the Phase C concept.

  7. **SOC stays in Phase A only**: v12 placed SOC in Phase A only
     and lost -5.4 pp vs v11 peak. v13 added SOC to a 5-axis Phase B
     and lost another -0.7 pp. v14 isolates the variable: SOC stays
     in Phase A only (matching v12) so any SOC change is attributable
     to the broader v12 → v14 deltas, not to a SOC-specific
     intervention. SOC narrow drilling is a v15 hypothesis.

  8. **TAA total cap raised 3,500 → 4,500**: v14's TAA is in its own
     phase where the cap's broad-shard-protection purpose does not
     apply; the cap is held at a non-zero value only to preserve actor
     diversity (per §3.1 delta 3).

## 6. Training recipe (four phases; multi-checkpoint protocol)

| | Phase A (broad) | Phase B (ATE+VSP+RCM) | Phase D-RMS (narrow) | Phase D-TAA (narrow) |
|---|---|---|---|---|
| Dataset | `_v14_broad` (193,703) + tulu-3 + alpaca | `_v14_ate_vsp_rcm` (32,810) | `_v14_rms` (12,608) | `_v14_taa` (32,783) |
| Cutoff | 16384 _(v14.1: **4096**)_ | 16384 _(v14.1: **4096**)_ | **8192** (v9 narrow) _(v14.1: **4096**)_ | **8192** (v9 narrow) _(v14.1: **4096**)_ |
| Packing | off | off | **on** (v9 narrow) | **on** (v9 narrow) |
| Gradient checkpointing | on _(v14.1: **off**)_ | on _(v14.1: **off**)_ | on _(v14.1: **off**)_ | on _(v14.1: **off**)_ |
| LR | 1e-5 | 5e-6 | 5e-6 | 5e-6 |
| Effective batch | 8 | 8 | **16** (v9 narrow) | **16** (v9 narrow) |
| Save/eval steps | 500 | 400 | 100 | 100 |
| Max samples | 200,000 | 36,000 | 13,000 | 33,000 |
| Resume from | base Qwen2.5-14B-Instruct | Phase A output | Phase A+B output | Phase A+B output (PARALLEL with D-RMS) |
| Push to HF | no | yes (`...-v14-ab`) _(v14.1: `...-v14p1-ab`)_ | yes (`...-v14-rms`) _(v14.1: `...-v14p1-rms`)_ | yes (`...-v14-taa`) _(v14.1: `...-v14p1-taa`)_ |

After D-RMS and D-TAA complete (parallel), a fifth chained pass
produces the production candidate:

| | Phase D-TAA-on-RMS (production chain) |
|---|---|
| Dataset | `_v14_taa` (32,783; same as Phase D-TAA) |
| Cutoff | 8192 _(v14.1: **4096**)_ |
| Packing | on |
| Gradient checkpointing | on _(v14.1: **off**)_ |
| LR | 5e-6 |
| Effective batch | 16 |
| Save/eval steps | 100 |
| Max samples | 23,000 |
| Resume from | **D-RMS output** (the v14-rms checkpoint; v14.1: the `v14p1-rms` checkpoint) |
| Push to HF | yes (`...-v14`) _(v14.1: `...-v14p1`)_ |

Phase ordering rationale (D-RMS → D-TAA, not the other way around):
TAA Classic alias resolution is shallower knowledge that gets
overwritten more easily than M-control catalog memorisation; running
TAA last preserves any TAA gains the chain produces. The parallel
D-TAA-only checkpoint isolates the TAA-narrow effect from any
RMS-narrow interaction.

Per-axis eval loss visibility is wired through `--eval_dataset
ift_data_2026_05_08_v14_val` on all four phases (v11-era silent
regression fix carried through v12, v13). Phase A scores all axes;
Phase B scores ATE+VSP+RCM; Phase D-RMS scores RMS; Phase D-TAA
scores TAA Classic+Canonical.

## 7. Consumers

| consumer | path |
|---|---|
| Dataset registration | `SFT/data/dataset_info.json` -> `ift_data_2026_05_08_v14_{broad,ate_vsp_rcm,rms,taa,val}` |
| Qwen2.5-14B launcher | `SFT/autotrain/run_sft_qwen25_14b_v14.sh` (four-phase, supports `--phase a\|b\|d-rms\|d-taa\|production\|all` and `--dry-run`); **v14.1 production execution: `SFT/autotrain/run_sft_qwen25_14b_v14_1.sh` (cutoff-4096 + gradient-checkpointing-off hot-fix; same flags)** |
| Qwen2.5-32B launcher | `SFT/autotrain/run_sft_qwen25_32b_v14.sh` (deferred; authored only after 14B passes §8 per v14 plan §6.2) |
| HF targets | `${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14{-ab,-rms,-taa,}` (4 checkpoints; 32B repo target reserved); **v14.1: `${HF_USERNAME}/athena-cti-sft-qwen25-14b-v14p1{-ab,-rms,-taa,}`** |
| vLLM aliases | `athena-cti-sft-qwen25-14b-v14-{ab,rms,taa}-vllm` and `athena-cti-sft-qwen25-14b-v14-vllm` (`SFT/test/pipelines/models.py`); v14.1 aliases registered as `athena-cti-sft-qwen25-14b-v14p1-{ab,rms,taa}-vllm` and `athena-cti-sft-qwen25-14b-v14p1-vllm` once the v14.1 chain completes |
| Eval task (carried) | `athena-cti-taa-canonical` (alias-resolution scoring; v12 carry) |
| Licence audit | `_v14_build/licence_gate_report.json` (every published row's source tag in §10.1 allowlist; v13 carry) |

## 8. Pass criteria (v14 14B production -- all hard gates must hold)

**HARD (production gate; all must hold for the v14 production
checkpoint -- the chained D-RMS → D-TAA -- to clear)**:

  - weighted total >= **60.0** (v12=57.3; +2.7 pp budgeted from RMS recovery)
  - RMS combined >= **60.0** (v9=65.8; budget 5.8 pp slip from carry losses)
  - ATE >= **56.0** (v12 hold, +0.5 pp carry buffer)
  - RCM >= **68.0** (v12 hold)
  - VSP >= **84.0** (v12 hold)
  - CKT >= **69.0** (v12 hold)
  - SOC >= **39.0** (v12 hold; SOC narrow drilling deferred to v15)
  - **TAA Classic combined >= 52.0 (Qwen2.5-14B-Instruct base; non-negotiable)**

**STRETCH (validates Phase D-TAA permanence; reported, not gating)**:

  - TAA Classic combined >= **55.0** (base + 3 pp; "Phase D-TAA is
    genuinely productive, keep it in the production recipe")
  - TAA Canonical combined >= **38.7** (v12 hold; v13 regressed to 24.8)
  - RMS combined >= **65.0** (v9 floor; "Phase D-RMS fully recovered")
  - CyberMetric-2000 / -10000 >= v12 actuals
  - CyberSOCEval-malware / -TI >= v11 actuals (peak vintage)

**FAILURE MODES → falsifiable consequences for v15+**:

  - If `v14-rms` does not improve RMS over `v14-ab` by >= 3 pp, the
    "per-axis narrow drilling recovers per-axis peaks" hypothesis is
    falsified for RMS. v15 stops trying to recover RMS via narrow
    phases; budget shifts to RMS template/catalog expansion.
  - If `v14-taa` does not push TAA Classic combined to >= 52.0, the
    "TAA training is net-positive" hypothesis is falsified at the
    14B base. v15+ permanently drops TAA-specialised content
    (TAA.CANON, MISP.CANON, AB.TAA.* training rows) and the eval
    becomes "report what the base model does" for TAA Classic and
    Canonical.
  - If `v14-rmstaa` (production chain) regresses on RMS or TAA vs the
    parallel single-axis checkpoints by >= 2 pp, the "narrow drills
    chain cleanly" hypothesis is falsified. v15 ships parallel
    branches as separate model variants instead of one chained
    production checkpoint.

The 32B launch is **serial after** the 14B production candidate
passes §8 (v14 plan §6.2). v11's 8.5-hour 32B compute on a
known-suboptimal recipe is exactly the failure mode this gating
prevents; v12/v13 carried the same gate; v14 holds it.


## 9. Experimental protocol (which checkpoint answers which question)

The four HF checkpoints are not redundant: each isolates a different
hypothesis. The benchmark sweep runs against ALL FOUR independently
and the comparison table below is the v14 verdict.

| comparison | hypothesis tested | outcome interpretation |
|---|---|---|
| `v14-ab` vs v12 | "Removing RMS+SOC dilution from Phase B holds v12-tier scores" | If v14-ab matches v12 ±1 pp on every axis, the v12 substrate is intact and v14 is a clean test bed for the narrow drills. If v14-ab regresses, the broad-shard re-balance (TAA cap raise, four-shard splitter) introduced unintended damage; investigate before drawing conclusions about D-RMS/D-TAA. |
| `v14-rms` vs `v14-ab` | "v9-shape RMS-only narrow drill recovers RMS without damaging adjacent axes" | RMS ↑ ≥ 3 pp AND adjacent axes (ATE, RCM, VSP, CKT) within ±1 pp = pattern works. RMS ↑ but adjacent axes ↓ = narrow drilling causes catastrophic forgetting; a ceiling is reached on this base model. RMS unchanged = v9-recipe + v9-templates do not transfer when applied as a late-stage drill on a Phase A+B base; root cause is in the base, not the recipe. |
| `v14-taa` vs `v14-ab` | "v9-shape TAA-only narrow drill lifts TAA Classic above the 52.0 base-model floor" | TAA Classic combined ≥ 52.0 = TAA narrow drilling works; keep canonical+MISP content in production recipe. TAA Classic combined < 52.0 = TAA training is net-negative on this base; drop all TAA-specialised content in v15+. TAA Canonical combined ↑ ≥ v12's 38.7 = canonical content benefits from dedicated phase; held. |
| `v14` (production) vs `v14-rms` | "RMS gain survives a chained TAA narrow drill" | RMS combined within ±2 pp of v14-rms's RMS = chain is clean; ship the production checkpoint. RMS combined ↓ ≥ 2 pp = TAA drill damages RMS; ship parallel branches as separate variants and pick per-deployment. |
| `v14` (production) vs `v14-taa` | "TAA gain survives running on top of an RMS-narrow base" | TAA Classic combined within ±2 pp of v14-taa's TAA Classic = chain is clean. TAA Classic combined ↓ ≥ 2 pp = v14-taa is a more honest test of TAA narrow drilling than v14. |

The v14 production candidate is the chained `v14` checkpoint UNLESS
the chain damages RMS or TAA by ≥ 2 pp vs the parallel single-axis
checkpoints, in which case the v14 production set is the parallel
pair (`v14-rms` for RMS-deployments, `v14-taa` for TAA-deployments)
and the unified production candidate is deferred to v15.

## 10. Version history (v0 → v14.1)

Each row links the vintage directory carrying the manifest, README
(where present), and per-version build artefacts. Every prior
version remains in the repo verbatim for reproducibility.

| version | date | path | corpus | what it changed |
|---|---|---|---|---|
| v0 baseline | 2026-03-22 | `tmpl_gen/templates/04022026/` | small hand-crafted | Initial M/A/W/V/S/P/E/X.{1..8} substrate (the 64-template "broad CTI knowledge core" still carried in every later manifest as Section C). |
| v6 | 2026-04-25 | `tmpl_gen/templates/04252026/` | abaligned | First AthenaBench-aligned slate (AB.* family, MCQ + RCM + ATE + RMS distractor blocks with `{force}` constraints; `negcoa*/negcap*/negack*` distractor pattern established). |
| v7 | 2026-04-26 | `tmpl_gen/templates/04262026/` | combined | Consolidated v6 + JSON pre-cursor; introduced cross-framework path templates (X.VWA, X.TMN). RMS=68.1 (the L31-8B SFT peak across all vintages). |
| v8 | 2026-04-29 | `tmpl_gen/templates/04292026/` | small + large | Two split manifests (Llama-3.1-8B `v8_small`, Qwen2.5-32B `v8_large`); JSON-output addendum (`JS.*`); long-context scaffolding (`stitch_long_context.py`). MCQ=77.6 (the 14B peak). |
| v8.1 | 2026-04-30 | `tmpl_gen/templates/04302026/` | 14B single-pass | RMS catalog-collapse fix: explicit `Count:` floors on AB.RMS.{4,5}; consolidated single-source-of-truth manifest; first build using `tmpl_docx2json` directly off the `.txt` file. |
| v9 | 2026-04-30 | `tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt` | two-phase | Phase B "RMS slice" extracted from v8.1 to recover catalog under v8.1's broad-knowledge regression. v9 ran as a two-phase chain (broad re-anchor → 1-axis RMS) on the 14B base. **RMS=65.8** (the v14 D-RMS target). v9 hyperparameters (cutoff 8192 / packing ON / batch 16) are the v14 D-RMS and D-TAA recipe. |
| v10 | 2026-05-01 | `tmpl_gen/templates/05012026/` | 200,340 rows | Single-pass unified manifest (216 templates); HTML/whitespace sanitiser at parser chokepoint; `<desc>...</desc>` markers around freeform text; actor cap=20; 13-gram dedup at threshold 50. AthenaBench composite weighted total: 54.1. **VSP=86.7** (the 14B peak; AB.VSP.* + V.CPE templates carried verbatim through v12 and into v14). |
| v11 | 2026-05-03 | `tmpl_gen/templates/05032026/` | 198,994 rows | 244 templates (all productive). SOC.* (4,884) + TAA.CANON.* (778) + RMS paraphrase + X./YN. trim with RCM-axis exemptions + parser anchor-fixation fix + actor cap 40 + dedup at 50 + D3FEND v1.4.0 ingest. Eval-axis density 30.7%. **14B sweep weighted total 54.3 (failed §8). SOC=44.7** (the 14B peak; SOC content carried via v12 substrate into v14). |
| v12 | 2026-05-05 | `tmpl_gen/templates/05052026/` | 260,589 rows | 263 distinct shortnames. Four programmatic generators (TAA.CANON 10K, CM 6K, MCQ.EXT 3K, SOC.GEN 5K) bypassing Neo4j substrate ceilings. Build pipeline gains row-count gate (per-axis REJECT_IF_BELOW), AB.TAA total cap (3,500), wired-in stratified shuffle, per-phase corpus splitter. Three-phase training (A: broad 8192/packing-on/lr 1e-5, B: RMS+ATE+VSP+RCM 16384/packing-off/lr 5e-6, C: TAA.CANON 8192/packing-on/lr 3e-6). Eval-axis density 43.3%. All 11 row-count axes pass. **14B sweep weighted total 57.3** (current production baseline): ATE=57.0 (peak), RCM=69.0 (peak), CKT=70.1, RMS=63.3 (regression vs v9's 65.8), VSP=84.6, SOC=39.3 (regression vs v11's 44.7), TAA Classic combined=35.5 (regression vs v11's 49.5), TAA-CANON=38.7. |
| v13 | 2026-05-07 | `tmpl_gen/templates/05072026/` | 240,759 rows | First "best-of-vintage" composition framing (in practice held v12 template bodies verbatim, which the v13 audit confirmed as a strict superset of v9/v10/v11 per-axis peaks); added MISP CC-0 TAA expansion (~12K rows) and licence-allowlist gate; reverts to two-phase v9-shape recipe (cutoff 8192, packing ON). **14B sweep weighted total 54.5 (FAILED §6, regressed -2.8 pp vs v12)**: RMS=55.8 (-7.5 vs v12; v12-superset RMS templates + v9 recipe DID NOT recover RMS in 5-axis Phase B), ATE=56.5, VSP=85.3, RCM=67.8, CKT=69.5, SOC=38.6, TAA Classic combined=43.5 (still -8.5 below base 52.0), TAA-CANON=24.8 (-13.9 vs v12). Falsified four hypotheses (see v14 README §2.3). |
| v14 | 2026-05-08 | `tmpl_gen/templates/05082026/` | ~242,000 (target) | **245 distinct shortnames declared (-18 vs v13: drops AB.RMS.4{b..j}+AB.RMS.5{b..j} paraphrase variants in favour of v9-narrow AB.RMS.4+AB.RMS.5)**; manifest = v13 verbatim with v9-narrow AB.RMS.4 + AB.RMS.5 substitution (hybrid RMS slice; AB.RMS.{1,2,3a..3h} keep v13's `<desc>`-wrapped bodies — see README §5 delta 3). **Per-axis narrow-drilling experiment** with **four-phase training** (A broad / B 3-axis ATE+VSP+RCM long-context / D-RMS narrow drill / D-TAA narrow drill PARALLEL with D-RMS, plus chained production candidate). **Four HF checkpoints** (`v14-ab`, `v14-rms`, `v14-taa`, `v14`) for clean per-phase attribution. **Hard pass criterion: TAA Classic combined ≥ 52.0** (the Qwen2.5-14B-Instruct base-model floor that every prior SFT vintage has regressed below). NO new generators, NO new manifest content beyond v9-narrow .4+.5 RMS substitution; v14 is a recipe/sharding/RMS-template experiment on the v13 content substrate. **Phase A live measurement (step 23000/49336, 47%) at 1.78 s/it / 2,927 tok/s flagged ~95% padding overhead at cutoff 16384 against an audited p99 < 4096 across all four shards — superseded by v14.1 hot-fix mid-run.** |
| **v14.1** | **2026-05-08** | `tmpl_gen/templates/05082026/` (corpus shared with v14) | **same as v14 (~242,000)** | This vintage. **Launcher-only hot-fix** of v14: `SFT/autotrain/run_sft_qwen25_14b_v14_1.sh` flips two performance-only knobs and changes nothing else. (1) `--cutoff` lowered to 4096 in every phase (16384→4096 in A/B; 8192→4096 in D-RMS, D-TAA, production). (2) `--disable_gradient_checkpointing True` added to `EXTRA_COMMON` for all five phases -- vestigial at cutoff 4096 / per_device_batch 1 / ZeRO-3 on 8x80GB; removes backward-pass recompute (~15-30% throughput gain on top of cutoff reduction). Corpus, topology, learning rates, effective batch sizes, packing flags, max-samples, save/eval cadences, and resume chaining are preserved verbatim from v14, so step counts per phase are bit-identical and loss curves are step-for-step comparable to v14's (GC is a forward/backward implementation detail, not part of the model math). **Four HF checkpoints**: `v14p1-ab`, `v14p1-rms`, `v14p1-taa`, `v14p1` (the `v14p1` token replaces v14's `v14` in the repo-name suffix; underscores break HF tooling, dots are flaky in some clients). Save dirs use `v14_1_phase_{a,b,d_rms,d_taa,prod}_TIMESTAMP`. Datasets are byte-identical to v14 (`ift_data_2026_05_08_v14_*`). Estimated total wall-time ~9-11 h vs v14's ~32 h projected. The narrow-drilling experiment and all §1/§2/§9 questions are unchanged; v14 launcher remains in-repo for reproducibility. |

For the line-by-line corpus composition of v12 (the substrate v14
carries forward) see `tmpl_gen/templates/05052026/README.md`. For
v13 (the post-mortem subject) see
`tmpl_gen/templates/05072026/README.md`. For v9 (the RMS source and
the narrow-drill recipe) see
`tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt` and
the v9 launcher in
`SFT/autotrain/run_sft_qwen25_14b_v9.sh`. For prior vintages, follow
the chain in §10 above; the v0 substrate's per-family design strategy
(`M/A/W/V/S/P/E/X`) is in `tmpl_gen/templates/README.md`.

## 11. Known content gaps to watch at eval

  1. **TAA.CANON.1 single-alias trivial case (carried from v11/v12/v13)**:
     groups with only one element in `aliases` (≈83 of 187 MITRE
     intrusion-sets) yield trivial identity rows. v13 mitigated via
     MISP CC-0 expansion (~700 additional canonical groups). v14
     carries the v13 mitigation unchanged; the dedicated D-TAA phase
     should give the discrimination signal more weight than v12/v13's
     diluted broad-shard exposure.
  2. **SOC.TRIAGE.DS.{1,2}** still bind on the data-source node alone
     (carried from v11/v12/v13): the 38 `x-mitre-data-source` nodes
     in athena-cti-db are isolated. v14 holds the v12 mitigation
     (SOC.GEN.* ~5K rows from curated tables, not graph-bound). SOC
     remains in Phase A only; no SOC narrow drill in v14.
  3. **AB.RMS.{4,5} ceiling**: paraphrase-multiplied to ~440 per
     family in v11; held in v12. v14 sources AB.RMS.{1..6} verbatim
     from v9 (the recipe vintage that produced RMS=65.8). The v9
     templates have higher per-anchor combo yields than v12's RMS
     section, projecting to ~13K RMS rows total. If `athena-cti-rms`
     still regresses below v9's 65.8 floor after Phase D-RMS, the
     M-control catalog itself in current Neo4j has drifted from v9-era
     content; investigation is out of scope for v14 and becomes a
     v15 graph-content audit.
  4. **Input collisions** (~85K instr+input pairs with multi-valued
     output, e.g., one ExploitDB entry → multiple CVEs): intentional
     one-to-many CTI relationships, not noise. Trains on multi-modal
     P(answer|question). Carried from v12/v13.
  5. **CM.* family is curated, not graph-derived (carried from v12/v13)**:
     rows reflect the authors of the seed banks. v14 ships the v12
     family verbatim (held in `_v14_broad`).
  6. **MISP TAA seed snapshot age**: the vendored `threat-actor.json`
     is the same v13 snapshot. v14 inherits the snapshot-vs-live drift
     risk; refresh deferred to v15.
  7. **TAA non-improvement risk (NEW v14 risk)**: if Phase D-TAA on
     the full TAA slice (Classic + Canonical + MISP) at the v9-shape
     recipe still fails to cross TAA Classic combined ≥ 52.0, the
     v15 corpus will permanently drop TAA training and all TAA-
     specialised content (~23K rows, ~9% of v14 corpus). The eval
     remains; the training does not.
  8. **Carry-loss in chained narrow drills (NEW v14 risk)**: the
     production chain runs D-TAA on top of D-RMS. If TAA narrow
     training damages the M-control memorisation laid down by the
     RMS narrow phase by more than 2 pp, the production candidate
     ships as parallel branches instead of a chained composite. The
     `v14-rms` and `v14-taa` parallel checkpoints are deliberately
     authored as fallback production candidates so this failure mode
     does not require a v14.1 re-run.
