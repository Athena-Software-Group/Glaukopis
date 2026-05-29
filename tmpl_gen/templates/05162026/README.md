# Sophia CTI Templates — v20 (May 16, 2026 vintage)

v20 is a **targeted axis-density rebalance** on top of the v19
reproducibility-first rebuild. The pipeline topology, per-stage
recipes, and the Core/TAA manifest bodies are carried from v19
verbatim. Two behavioural deltas ship in v20: (1) the regressed
ATE, RCM, and CSE-TI axes get a gradient-signal lift via raised
`count_max` (Core) and raised `Count:` directives (CSE-TI), and
(2) Stage 5 (Recalibrate) reverts its `interleave_probs` mix from
v19's equal-weight `0.33/0.33/0.34` back to v18.2's asymmetric
`0.25/0.40/0.35` (Phase A / Phase B / TAA).

The chain produces four cumulative HF checkpoints (all four pushed;
`v20-recalibrate` is the headline):

| stage | checkpoint | answers the question |
|---|---|---|
| 1+2 | `asg-ai/athena-cti-sft-qwen25-14b-v20-core` | Does raising `count_max` 3500 → 5500 lift ATE and RCM back toward the v18.1 / v18.2 baseline (ATE ≥ 60.0, RCM ≥ 67.5) without regressing the other Core axes by more than 2 pp? |
| 3 | `asg-ai/athena-cti-sft-qwen25-14b-v20-taa` | Does the v19-byte-identical TAA Classic narrow drill reproduce on top of `v20-core` (TAA Classic ≥ 40.0) without regressing the stage-1+2 axes by more than 2 pp? |
| 4 | `asg-ai/athena-cti-sft-qwen25-14b-v20-cse` | Does raising five JS.CSE.TI.* template Counts from 1500 to 2500 lift CSE-TI back to ≥ 34.0 and hold CSE-Malware ≥ 20.0, while keeping v20-core gains within 2 pp on MCQ + TAA + RCM? |
| 5 | `asg-ai/athena-cti-sft-qwen25-14b-v20-recalibrate` | Does reverting Stage 5's interleave mix to v18.2's `0.25/0.40/0.35` recover ATE ≥ 62.0, RCM ≥ 67.5, and RMS ≥ 54.0 while holding MCQ + TAA + CSE within 2 pp? |

The vintage directory is self-contained per project convention. v20
follows the v19 pattern of shipping every load-bearing artifact in-
repo — the three stage manifests, the three row-count gates, the
master plan, this README, and (under the sibling `_v20_*_build/`
trees) the build watchers, generators, gates, validators, and per-
stage val/train splitters.

```
05162026/
  README.md                        this document
  v20_plan.txt                     master plan (motivation in §1, deltas
                                   vs v19 in §2, row-count plan in §3,
                                   training recipe in §4, sign-off in §5)
  v20_row_count_gate.json          per-axis REJECT_IF_BELOW thresholds for
                                   the Core shard (ATE and RCM floors
                                   bumped vs v19; other axes carry v19)
  v20_taa_row_count_gate.json      per-axis REJECT_IF_BELOW thresholds for
                                   the TAA shard (carried verbatim from
                                   v19_taa_row_count_gate.json)
  v20_cse_row_count_gate.json      per-axis REJECT_IF_BELOW thresholds for
                                   the CSE shard (TI-actor and TI-other
                                   floors bumped vs v19)
  Sophia-CTI-Templates-v20_core.txt  Core-shard manifest; body byte-
                                     identical to v19_core.txt (v20 header
                                     documents the count_max bump)
  Sophia-CTI-Templates-v20_taa.txt   TAA-shard manifest; body byte-
                                     identical to v19_taa.txt (v20 header
                                     documents the unchanged-vs-v19 stance)
  Sophia-CTI-Templates-v20_cse.txt   CSE-shard manifest; 5 Count: 1500
                                     directives bumped to Count: 2500 for
                                     JS.CSE.TI.{GRP.2, GRP.3, MAL.2, MAL.3,
                                     NEG.1}; other templates verbatim from
                                     v19_cse.txt
```

Predecessor vintage directories that v20 carries content from:

| vintage | role in v20 |
|---|---|
| `05152026/` (v19) | Source of all three stage manifests, all three row-count gates, all four launchers, and all three build watchers. v20 forks the v19 substrate and applies the §2 deltas. |
| `05132026/` (v18.2) | Source of the Stage 5 `interleave_probs` distribution that v20 reverts to (0.25/0.40/0.35). |

Build infrastructure ships in three sibling build trees at the repo
root (forked from `_v19_core_build/`, `_v19_taa_build/`,
`_v19_cse_build/` with v20-named output paths, the v20 row-count
gate paths, and the v20 vintage label):

```
_v20_core_build/                   Core (Stages 1+2) build artefacts
  _neo4j_check.py                  Phase 0 substrate validator
  watcher.sh                       post-build pipeline (substrate gate /
                                   seed-provenance / generator merges /
                                   TAA actor-balance / dedup / row-count
                                   gate / licence gate / stratified
                                   shuffle / val/train split / two-shard
                                   phase split)
  build_val_slice.py               per-axis val/train splitter (50 rows
                                   / shortname; seed=42)
  validate_corpus.py               cross-stage end-to-end validator

_v20_taa_build/                    Stage 3 (+TAA) build artefacts
  watcher.sh                       TAA Classic post-build pipeline
  build_val_slice.py               per-axis val/train splitter

_v20_cse_build/                    Stage 4 (+CSE) build artefacts
  watcher.sh                       CSE post-build pipeline
  letter_balance_gate.py           CSE letter-tuple distribution gate
  build_val_slice.py               per-axis val/train splitter
```

## 1. Why v20 exists — axis-density rebalance + Stage 5 revert

See `v20_plan.txt §1` for the full motivation. In short: v19's
reproducibility-first rebuild regressed against the v18.1 / v18.2
baseline on ATE (−8.6 pp), RCM (−1.2 pp), and CSE-TI plateau
(−0.6 pp). The diagnosed cause is a basin-shift attributable to a
hardware change (8xH100 → 4xH100) plus data drift; v20 cannot
eliminate the hardware effect but raises the gradient signal on the
regressed axes via density bumps (§2). Stage 5 also reverts to
v18.2's `0.25/0.40/0.35` so the catalog-axis touch-up reapplies
Phase B at the higher weight that v18.2 used.

## 2. Run-book

### 2.1 Generation

Three independent shard builds, one per chain stage. The CTI-DB
substrate (`athena-cti-db` neo4j instance) must be reachable at
`tmpl_gen/data_generation/neo4j-local-config.json`.

```bash
cd /Users/pietro/code/Glaukopis    # or the cluster equivalent

# --- Stage 1+2: Core shard (v20 manifest; count_max bumped to 5500) ---
python _v20_core_build/_neo4j_check.py
mkdir -p _v20_core_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05162026/Sophia-CTI-Templates-v20_core.txt \
     _v20_core_build/triples \
     SFT/data/ift_data_2026_05_16_v20_core.raw.json \
     2500 5500 > _v20_core_build/build.log 2>&1 &
echo "PID=$!" > _v20_core_build/build.pid
nohup bash _v20_core_build/watcher.sh > _v20_core_build/watcher.log 2>&1 &

# --- Stage 3: TAA Classic shard (manifest verbatim from v19) ---
mkdir -p _v20_taa_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05162026/Sophia-CTI-Templates-v20_taa.txt \
     _v20_taa_build/triples \
     SFT/data/ift_data_2026_05_16_v20_taa.raw.json \
     10 3500 > _v20_taa_build/build.log 2>&1 &
echo "PID=$!" > _v20_taa_build/build.pid
nohup bash _v20_taa_build/watcher.sh > _v20_taa_build/watcher.log 2>&1 &

# --- Stage 4: CSE shard (5 TI Count: bumps applied in v20_cse.txt) ---
mkdir -p _v20_cse_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05162026/Sophia-CTI-Templates-v20_cse.txt \
     _v20_cse_build/triples \
     SFT/data/ift_data_2026_05_16_v20_cse.raw.json \
     10 3500 > _v20_cse_build/build.log 2>&1 &
echo "PID=$!" > _v20_cse_build/build.pid
nohup bash _v20_cse_build/watcher.sh > _v20_cse_build/watcher.log 2>&1 &
```

Final outputs (7 SFT JSON shards):

- `SFT/data/ift_data_2026_05_16_v20_core_a_kb_mcq_taa_soc_cm_ms_yn.json`
- `SFT/data/ift_data_2026_05_16_v20_core_b_rms_ate_vsp_rcm.json`
- `SFT/data/ift_data_2026_05_16_v20_core_val.json`
- `SFT/data/ift_data_2026_05_16_v20_taa.json`
- `SFT/data/ift_data_2026_05_16_v20_taa_val.json`
- `SFT/data/ift_data_2026_05_16_v20_cse.json`
- `SFT/data/ift_data_2026_05_16_v20_cse_val.json`

Validate end-to-end with `python _v20_core_build/validate_corpus.py`.

### 2.2 Training

Four launchers, run sequentially (each pulls the previous stage's
pushed HF checkpoint as its base):

```bash
bash SFT/autotrain/run_sft_qwen25_14b_v20_core.sh        # ~13 h on 8xH100, ~26 h on 4xH100 -> v20-core
bash SFT/autotrain/run_sft_qwen25_14b_v20_taa.sh         # ~6-8 h        -> v20-taa
bash SFT/autotrain/run_sft_qwen25_14b_v20_cse.sh         # ~4-6 h        -> v20-cse
bash SFT/autotrain/run_sft_qwen25_14b_v20_recalibrate.sh # ~80-100 min on 4xH100 -> v20-recalibrate
```

### 2.3 Bench

Standard 14B AthenaBench + CyberMetric + CyberSOCEval sweep against
each of the four pushed checkpoints. Sign-off gates per stage are
in `v20_plan.txt §5`. The `v20-recalibrate` checkpoint is graded
against the v18.2 §7.4 gate package: RMS ≥ 54.0, MCQ ≥ 62.0,
TAA ≥ 40.0, CSE-TI ≥ 34.0, CSE-Malware ≥ 20.0, ATE ≥ 62.0,
RCM ≥ 67.5, VSP ≥ 80.0, CyberMetric-2K ≥ 85.5, CyberMetric-10K ≥ 81.0.

## 3. Contamination posture

Inherited from v8 / v18.1 / v19 without modification. The v20 corpus
is built by `_v20_{core,taa,cse}_build/watcher.sh`, each of whose
Phase 5 runs `tmpl_gen/scripts/dedup_against_evals.py` against
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
falsifiability list: see
[`../05182026/README-21.md` §Contamination posture](../05182026/README-21.md#contamination-posture).

**v20-specific note.** v20 is an axis-density rebalance over the
v18.1 / v19 substrate with a Stage 5 revert (see §1). The
rebalance changes per-axis row counts in the row-count gates but
does **not** introduce any new eval-overlapping content surface, any
new dataset source, or any change to the dedup tunables. The dedup
contract and eval-dir scope are unchanged from v18.1 / v19; per-shard
`dedup_report.json` artefacts are regenerated against the
rebalanced corpus.

