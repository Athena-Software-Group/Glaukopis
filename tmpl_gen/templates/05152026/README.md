# Sophia CTI Templates — v19 (May 15, 2026 vintage)

v19 is the **reproducibility-first ground-up rebuild** of the v18.1 /
v18.2 SFT pipeline. The training recipe is preserved verbatim through
Stage 4; the only behavioural delta is at Stage 5 (Recalibrate), where
the 3-shard interleave-probs mix flips from v18.2's
`0.25 / 0.40 / 0.35` (Phase A / Phase B / TAA) to **equal-weight
`0.33 / 0.33 / 0.34`**. The structural delta is that v19 ships every
artifact required to regenerate the SFT shards from a clean checkout
in this directory and in three sibling `_v19_*_build/` trees, so the
pipeline result is reproducible end-to-end.

The chain produces four cumulative HF checkpoints (all four pushed;
`v19-recalibrate` is the headline):

| stage | checkpoint | answers the question |
|---|---|---|
| 1+2 | `asg-ai/athena-cti-sft-qwen25-14b-v19-core` | Does the v18.1 two-phase Core recipe (Phase A broad re-anchor + Phase B catalog drill) regenerate the v18-1-core baseline (CKT ≥ 70.0, RMS ≥ 55.0, ATE ≥ 60.0, VSP ≥ 80.0, RCM ≥ 67.5) when built from a clean v19 checkpoint with the v19-named shards? |
| 3 | `asg-ai/athena-cti-sft-qwen25-14b-v19-taa` | Does the v18.1 TAA Classic narrow drill recipe reproduce on top of `v19-core` (TAA Classic ≥ 40.0) without regressing the stage-1+2 axes by more than 2 pp? |
| 4 | `asg-ai/athena-cti-sft-qwen25-14b-v19-cse` | Does the v18.1 CSE letter-set drill recipe reproduce on top of `v19-taa` (CSE-TI ≥ 34.0, CSE-Malware ≥ 20.0) while keeping the v19-core gains within 2 pp on MCQ + TAA + RCM? Stage-3 RMS / ATE / VSP erosion is expected and is the target of Stage 5. |
| 5 | `asg-ai/athena-cti-sft-qwen25-14b-v19-recalibrate` | Does an equal-weight 3-shard low-LR replay (Phase A / Phase B / standalone TAA at 0.33 / 0.33 / 0.34, lr 1e-6, 2400 max-samples) recover RMS to ≥ 54.0 and lift MCQ above v18.2's 62.33 plateau without regressing the eight axes that v18.2 already passes? |

The vintage directory is self-contained per project convention. v19
is the **first vintage where every load-bearing artifact ships
in-repo** — the three stage manifests, the row-count gate, the master
plan, this README, and (under the sibling `_v19_*_build/` trees) the
build watchers, generators, gates, validators, and per-stage val/train
splitters.

```
05152026/
  README.md                        this document
  v19_plan.txt                     master plan (motivation in §1, deltas
                                   vs v18.2 in §2, row-count plan in §3,
                                   chained training recipe per stage in
                                   §4, falsification + sign-off in §5/§6)
  v19_row_count_gate.json          per-axis REJECT_IF_BELOW thresholds
                                   for the Core shard; carries v18p1
                                   floors verbatim (no axis change)
  Sophia-CTI-Templates-v19_core.txt  Core-shard manifest; body byte-
                                     identical to v18.1.txt (SHA256
                                     7e76b44f...)
  Sophia-CTI-Templates-v19_taa.txt   TAA-shard manifest; body byte-
                                     identical to v16.txt (SHA256
                                     843e26e4...)
  Sophia-CTI-Templates-v19_cse.txt   CSE-shard manifest; body byte-
                                     identical to v17.1.txt (SHA256
                                     9923f71f...)
```

Predecessor vintage directories that v19 carries content from
(verbatim, by SHA256 manifest):

| vintage | role in v19 |
|---|---|
| `05112026/` (v18.1) | Source of `Sophia-CTI-Templates-v18.1.txt` (carried as `v19_core.txt`) and `v18_1_row_count_gate.json` (carried as `v19_row_count_gate.json`). |
| `05092026/` (v16) | Source of `Sophia-CTI-Templates-v16.txt` (carried as `v19_taa.txt`). |
| `05102026/` (v17.1) | Source of `Sophia-CTI-Templates-v17.1.txt` (carried as `v19_cse.txt`). |
| `05132026/` (v18.2) | Source of the Stage 5 multi-shard recipe shape (cutoff 16384, packing off, lr 1e-6, eff_bs 4, --max-samples 2400, mix_strategy interleave_under). v19 changes ONLY the interleave_probs. |

Build infrastructure ships in three sibling build trees at the repo
root (forked from `_v18p1_build/`, `_v18_taa_build/`, `_v18_cse_build/`
with v19-named output paths and the v19 row-count gate path; the
Core fork pulls from `_v18p1_build/` rather than `_v18_build/` so the
v18.1 MCQ.EXT.* generator-merge skip is inherited):

```
_v19_core_build/                   Core (Stages 1+2) build artefacts
  _neo4j_check.py                  Phase 0 substrate validator
  watcher.sh                       post-build pipeline (substrate
                                   gate / seed-provenance / generator
                                   merges / TAA actor-balance / dedup
                                   / row-count gate / licence gate /
                                   stratified shuffle / val/train
                                   split / two-shard phase split)
  build_val_slice.py               per-axis val/train splitter
                                   (50 rows / shortname; seed=42)
  validate_corpus.py               cross-stage end-to-end validator

_v19_taa_build/                    Stage 3 (+TAA) build artefacts
  watcher.sh                       TAA Classic post-build pipeline
  build_val_slice.py               per-axis val/train splitter

_v19_cse_build/                    Stage 4 (+CSE) build artefacts
  watcher.sh                       CSE post-build pipeline
  letter_balance_gate.py           CSE letter-tuple distribution gate
  build_val_slice.py               per-axis val/train splitter
```

## 1. Why v19 exists — reproducibility-first rebuild

See `v19_plan.txt §1` for the full motivation. In short: v18.x built
with cross-vintage manifest dependencies (v18.1 referenced v16 and
v17.1 manifests by path; build watchers were gitignored "local
scratch"). Reproducing v18.x from a clean checkout therefore required
reconstructing both. v19 collapses both: every load-bearing artifact
is in-repo and self-contained under this vintage directory and the
three sibling `_v19_*_build/` trees.

The Stage 5 prob change (`0.25/0.40/0.35` → `0.33/0.33/0.34`) is the
sole training-recipe delta and is motivated by the v18.2 / v18.2.1 /
v18.2.2 prob-mix iteration history; see `v19_plan.txt §1.2`.

## 2. Run-book

### 2.1 Generation

Three independent shard builds, one per chain stage. The CTI-DB
substrate (`athena-cti-db` neo4j instance) must be reachable at
`tmpl_gen/data_generation/neo4j-local-config.json`.

```bash
cd /Users/pietro/code/Glaukopis    # or the cluster equivalent

# --- Stage 1+2: Core shard (v19 manifest) ---
python _v19_core_build/_neo4j_check.py
mkdir -p _v19_core_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05152026/Sophia-CTI-Templates-v19_core.txt \
     _v19_core_build/triples \
     SFT/data/ift_data_2026_05_15_v19_core.raw.json \
     2500 3500 > _v19_core_build/build.log 2>&1 &
echo "PID=$!" > _v19_core_build/build.pid
nohup bash _v19_core_build/watcher.sh > _v19_core_build/watcher.log 2>&1 &

# --- Stage 3: TAA Classic shard ---
mkdir -p _v19_taa_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05152026/Sophia-CTI-Templates-v19_taa.txt \
     _v19_taa_build/triples \
     SFT/data/ift_data_2026_05_15_v19_taa.raw.json \
     10 3500 > _v19_taa_build/build.log 2>&1 &
echo "PID=$!" > _v19_taa_build/build.pid
nohup bash _v19_taa_build/watcher.sh > _v19_taa_build/watcher.log 2>&1 &

# --- Stage 4: CSE shard ---
mkdir -p _v19_cse_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05152026/Sophia-CTI-Templates-v19_cse.txt \
     _v19_cse_build/triples \
     SFT/data/ift_data_2026_05_15_v19_cse.raw.json \
     10 3500 > _v19_cse_build/build.log 2>&1 &
echo "PID=$!" > _v19_cse_build/build.pid
nohup bash _v19_cse_build/watcher.sh > _v19_cse_build/watcher.log 2>&1 &
```

Final outputs (7 SFT JSON shards):

- `SFT/data/ift_data_2026_05_15_v19_core_a_kb_mcq_taa_soc_cm_ms_yn.json`
- `SFT/data/ift_data_2026_05_15_v19_core_b_rms_ate_vsp_rcm.json`
- `SFT/data/ift_data_2026_05_15_v19_core_val.json`
- `SFT/data/ift_data_2026_05_15_v19_taa.json`
- `SFT/data/ift_data_2026_05_15_v19_taa_val.json`
- `SFT/data/ift_data_2026_05_15_v19_cse.json`
- `SFT/data/ift_data_2026_05_15_v19_cse_val.json`

Validate end-to-end with `python _v19_core_build/validate_corpus.py`.

### 2.2 Training

Four launchers, run sequentially (each pulls the previous stage's
pushed HF checkpoint as its base):

```bash
bash SFT/autotrain/run_sft_qwen25_14b_v19_core.sh        # ~13 h on 8xH100 -> v19-core
bash SFT/autotrain/run_sft_qwen25_14b_v19_taa.sh         # ~6-8 h        -> v19-taa
bash SFT/autotrain/run_sft_qwen25_14b_v19_cse.sh         # ~4-6 h        -> v19-cse
bash SFT/autotrain/run_sft_qwen25_14b_v19_recalibrate.sh # ~95-115 min on 4xH100 -> v19-recalibrate
```

### 2.3 Bench

Standard 14B AthenaBench + CyberMetric + CyberSOCEval sweep against
each of the four pushed checkpoints. Sign-off gates per stage are in
`v19_plan.txt §5`. The `v19-recalibrate` checkpoint is graded against
the v18.2 §7.4 gate package (RMS ≥ 54.0, MCQ ≥ 62.0, TAA ≥ 40.0,
CSE-TI ≥ 34.0, CSE-Malware ≥ 20.0, ATE ≥ 62.0, RCM ≥ 67.5, VSP ≥
80.0, CyberMetric-2K ≥ 85.5, CyberMetric-10K ≥ 81.0).

## 3. Contamination posture

Inherited from v8 / v18.1 without modification. The v19 corpus is
built by `_v19_{core,taa,cse}_build/watcher.sh`, each of whose
Phase 5 runs `tmpl_gen/scripts/dedup_against_evals.py` against
`SFT/test/benchmark_data/` with `n=13` word-grams,
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
falsifiability list, scoped to a three-shard Core / TAA / CSE
pipeline of the same shape v19 uses: see
[`../05182026/README-21.md` §Contamination posture](../05182026/README-21.md#contamination-posture).

**v19-specific note.** v19 is a reproducibility-first rebuild of
v18.1 (see §1) and consumes the same template / row-count-gate
inputs through a freshly-forked watcher. The dedup contract,
threshold tunables, and eval-dir scope are unchanged from v18.1; the
per-shard `dedup_report.json` is regenerated against the
build-date-stamped corpus rather than reused from v18.1.

