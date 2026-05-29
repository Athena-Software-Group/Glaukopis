# Sophia CTI Templates — v17 (May 11, 2026 vintage)

v17 is the **first chained narrow-SFT vintage** in the project. It trains
on top of the v16 TAA specialist (`asg-ai/athena-cti-sft-qwen25-14b-v16`)
rather than off the frozen v12 baseline, and adds a single new
output-shape head — the CyberSOCEval `{"correct_answers": [...]}` JSON
letter-set — without re-introducing any of the TAA-attribution surface
v16 already learned. The branch produces one HF checkpoint:

| checkpoint | answers the question |
|---|---|
| `asg-ai/athena-cti-sft-qwen25-14b-v17` | Was v16's CyberSOCEval-TI 30.63%/CyberSOCEval-Malware 10.69% accuracy bound by *output shape* (avg_score 58.54%/45.15% suggested the right semantic content was already there), or by missing task knowledge? |

The vintage directory is self-contained per project convention:

```
05112026/
  Sophia-CTI-Templates-v17.txt   self-contained CSE-shape manifest (14 templates;
                                 zero AB.TAA / JS.TAA rows by design)
  v17_plan.txt                   master plan document (recipe, pipeline, sign-off,
                                 falsification criteria)
  v17_row_count_gate.json        per-axis REJECT_IF_BELOW thresholds (CSE-TI /
                                 CSE-MAL only)
  README.md                      this document (v16 post-mortem in §1; design
                                 rationale in §2; run-book in §3; chained-SFT
                                 risk model in §4)
```

## 1. Why v17 exists

### 1.1 v16 outcome

v16 (the v15 W1-rev TAA specialist) lifted CyberSOCEval-TI from 3.74%
(v12) to **30.63%** accuracy with `avg_score` 58.54%, and CyberSOCEval-
Malware from 7.06% to **10.69%** with `avg_score` 45.15%. TAA Classic
strict accuracy was 7% (essentially flat vs v12's 11%), but TAA Classic
**plausible accuracy was 88%** — the model is naming actors that the
`related_groups.csv` graph treats as related to ground truth (cluster
confusion: APT41↔APT38, APT37↔Lazarus, APT35↔APT33, APT28↔Turla, …).
See `tmpl_gen/templates/05102026/README.md` §1 for the v16-on-v15 W1
diagnosis.

### 1.2 The v16 → CSE accuracy gap is consistent with output-shape miss

The CyberSOCEval evaluator scores Jaccard similarity on a
`{"correct_answers": ["A","C"]}` JSON object wrapped in
`<json_object></json_object>` tags (TI shape) or returned bare (Malware
shape). v16 was trained on prose+letter `JS.TAA.*` templates whose
`Answer:` field is a single letter at the end of a chain-of-thought, **not
a JSON list of letters**. The 28pp gap between `accuracy` (30.63%) and
`avg_score` (58.54%) on TI is the signature of a model producing the
right semantic content but failing the formal extractor's shape check.

### 1.3 v17 hypothesis

If the v16 → CSE accuracy ceiling is bound by output shape rather than
task knowledge, then **chained narrow SFT on the missing CSE letter-set
shape only** — no new entity knowledge introduced — should lift CSE-Mal
and CSE-TI accuracy materially without regressing v16's TAA-attribution
head. Falsification criteria in `v17_plan.txt §4` (Outcomes A/B/C/D).

## 2. Design rationale (per-template Count: targets)

Full per-template Count: schedule in `Sophia-CTI-Templates-v17.txt` §"v17
declared-row budget summary". Highlights:

- **`JS.CSE.TI.GRP.{1,2,3}`** (~5,910 actuals, target 8,000): intrusion-
  set-anchored multi-select with 1/2/3 correct techniques; mirrors the
  CyberSOCEval-TI `threat_intel_reasoning` prompt scaffolding (synthetic
  intel-report prose with the named group's documented tradecraft inline)
  without using the actual CrowdStrike PDF corpus → **zero contamination
  by construction**.
- **`JS.CSE.TI.MAL.{1,2,3}`** (~1,687 actuals): malware-anchored multi-
  select; under-yields against the malware-anchor pool because the
  technique-distinct-from-malware-uses constraint stack is tight.
- **`JS.CSE.TI.CMP.1`** (Count bumped 1000 → **2500**, ~2,055 actuals):
  campaign-anchored multi-select; the small ~52-campaign anchor pool
  needed the Count: bump to lift the CSE-TI-other axis above its 1,400
  floor on the row-count gate.
- **`JS.CSE.TI.NEG.1`** (Count 1500, structural ~187 actuals): zero-
  correct training; collapses to one row per intrusion-set anchor under
  the 5-way `{force negap_i != negap_j}` constraint stack — same yield
  pattern documented for v14 `JS.TAA.NEG.1` in `05082026/v14`.
- **`JS.CSE.MAL.RPT.{1,2,3}`** (~5,039 actuals, target 4,500): malware-
  anchored detonation-report multi-select; bare `{"correct_answers":[...]}`
  output shape (no `<json_object>` wrapper) per the CyberSOCEval-Malware
  evaluator. Synthetic detonation report JSON (`vx_family`, `sha256:"hash"`,
  `mitre_attcks` list) seeded from the malware's documented attack-pattern
  set; mirrors PurpleLlama's `malware_analysis` scaffolding without using
  the formal Hybrid-Analysis sample set.
- **`JS.CSE.MAL.{TAC,TGT,NEG}.1`** (~3,753 actuals): tactic-axis,
  group-attribution-axis, and zero-correct variants for option-set
  diversity.

Per-actor cap stays at 60 for completeness but is **structurally non-
binding** for v17 because every JS.CSE.* shortname is "untouched" by
`taa_actor_balance.py` (the script only scopes against AB.TAA.{1-5} and
JS.TAA.{1-3}); the v17 watcher correspondingly sets `ACTOR_FLOOR=0`.

## 3. Run-book

### 3.1 Generation

```bash
cd /Users/pietro/code/Glaukopis     # or the cluster equivalent
# Pre-flight: confirm Neo4j athena-cti-db is reachable and populated
python _v17_build/_neo4j_check.py
# Expected: intrusion-sets >= 180; malware >= 600; attack-patterns >= 800

bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05112026/Sophia-CTI-Templates-v17.txt \
     _v17_build/triples \
     SFT/data/ift_data_2026_05_11_v17.raw.json \
     10 3500 \
     2>&1 | tee _v17_build/build.log &
echo "PID=$!" > _v17_build/build.pid

nohup bash _v17_build/watcher.sh > _v17_build/watcher.log 2>&1 &
```

The watcher polls the build PID, then runs Phases 4–8 (actor-balance
no-op, dedup, row-count gate against `v17_row_count_gate.json`, licence
gate, stratified shuffle, val/train split). Final outputs:

- `SFT/data/ift_data_2026_05_11_v17_cse.json`  (training shard, ~16.5K rows)
- `SFT/data/ift_data_2026_05_11_v17_val.json`  (held-out validation slice, ~400 rows)
- `_v17_build/watcher_status.json`             (per-phase row counts and reports)

### 3.2 Dataset registration

The launcher resolves dataset names through `SFT/data/dataset_info.json`.
The v17 entries (`ift_data_2026_05_11_v17_cse`, `ift_data_2026_05_11_v17_val`)
are registered alongside the v16 / v15 / v12 entries.

### 3.3 Training

```bash
bash SFT/autotrain/run_sft_qwen25_14b_v16_plus_v17_cse.sh
# defaults to ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v17
# base model: asg-ai/athena-cti-sft-qwen25-14b-v16   (CHAINED off v16, not v12)
# wall-time ~4-6 h on 8xH100
```

### 3.4 Bench

Standard 14B AthenaBench sweep against the v17 HF checkpoint; compare
against v16 (CSE-TI 30.63%, CSE-Malware 10.69%, TAA Classic 7%/88%) per
the decision matrix in `v17_plan.txt §4`. The full sweep covers TAA
Canonical / TAA Classic, athena-rms, cybermetric (80 + 2000 + 10000),
cybersoceval-malware, cybersoceval-ti.

## 4. Contamination posture

Inherited from v8 / v10 / v11 / v14 without modification. The v17
corpus is built by `_v17_build/watcher.sh`, whose Phase 5 runs
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
structural-overlap matrix (including the CSE family), adjacent
corpus-hygiene gates, and the falsifiability list: see
[`../05182026/README-21.md` §Contamination posture](../05182026/README-21.md#contamination-posture).

**v17-specific note.** v17 introduces the `JS.CSE.*` template family
that teaches the JSON-envelope output shape graded by CyberSOCEval
(`{"correct_answers": [...]}`, `{"behaviors": [...]}`). These
templates are generated **synthetically from athena-cti-db** (the
public MITRE / NIST / FIRST / CISA graph); the CrowdStrike PDF
corpus that backs CyberSOCEval's reports is **not** ingested into
the v17 build pipeline at any point. The CSE-shape surface therefore
has **zero verbatim contamination by construction**, independent of
the n=13 dedup pass. Structural overlap on the graph-derived facts
that both the SFT templates and the CyberSOCEval evaluation describe
is accepted by design, on the same terms as the rest of the
contamination posture.

