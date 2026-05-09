# Sophia CTI Templates — v16 (May 10, 2026 vintage)

v16 is the second specialist branch of the v15 parallel-branching
architecture (`tmpl_gen/templates/05082026/v15_plan.txt`). It is the
**v15 W1-rev**: same v12 base, same v9-narrow recipe, same single-
specialist topology — only the TAA shard is rebuilt from scratch with
Canonical alias-resolution data **purged** and the attribution and
JSON-shaped templates **bumped**. The branch produces one HF checkpoint:

| checkpoint | answers the question |
|---|---|
| `asg-ai/athena-cti-sft-qwen25-14b-v16` | Was v15 W1's failure to move TAA Classic caused by the v14 TAA shard being 68.5% Canonical alias data, or by a deeper template-bound TAA-accuracy ceiling? |

The vintage directory is self-contained per project convention:

```
05102026/
  Sophia-CTI-Templates-v16.txt   self-contained TAA-only manifest (CANON purged)
  v16_plan.txt                   master plan document (recipe, pipeline, sign-off)
  v16_row_count_gate.json        per-axis REJECT_IF_BELOW thresholds (TAA-attribution + TAA-IE-NEG)
  README.md                      this document (v15 W1 post-mortem in §1; design rationale in §2; run-book in §3)
```

## 1. Why v16 exists

### 1.1 v15 W1 outcome

v15 W1 (`asg-ai/athena-cti-sft-qwen25-14b-v12-plus-taa`) was the cheapest
test of the v15 parallel-branching hypothesis: train one narrow specialist
(TAA) off the frozen v12 baseline and bench. The result confirmed the
**topology**: every measured non-target axis stayed within bench noise
of v12, validating that narrow SFT applied independently to v12 does not
catastrophically forget adjacent capability. But the **TAA Classic axis
did not move** -- the bench result was statistically indistinguishable
from v12's 11.0 -- and TAA Canonical jumped +16.5pp.

### 1.2 Post-mortem: shard composition diagnosis

A `taa_actor_balance.py --dry-run` audit of the v14 TAA shard
`ift_data_2026_05_08_v14_taa` showed that the actor distribution itself
was healthy (top:bottom row ratio ~2x; no actor over 36 rows; bulk of the
98 actors in the 17-32 rows-per-actor bucket). The mode-collapse
hypothesis was falsified.

A shortname-prefix breakdown of the same shard then revealed the actual
composition:

| family prefix | rows | % of shard |
|---|---:|---:|
| `MISP.CANON.{1,2,3}` | 11,730 | 35.8% |
| `TAA.CANON.{1,2,3}` | 10,717 | 32.7% |
| `AB.TAA.{1,2,3,5}` (Classic, prose) | 3,910 | 11.9% |
| `AB.TAA.IE.{1,2}` + `AB.TAA.NEG.1` (refusal + paired) | 4,410 | 13.5% |
| `AB.TAA.4` (campaign-bound) | 112 | 0.3% |
| `JS.TAA.IE.1` + `JS.TAA.NEG.1` (JSON refusal + paired) | 1,476 | 4.5% |
| `JS.TAA.{1,2,3}` (JSON Classic attribution) | 428 | 1.3% |

**68.5% of v15 W1's training mass was Canonical alias-resolution data**
(`MISP.CANON` 35.8% + `TAA.CANON` 32.7%) -- a task NOT measured by the
formal TAA Classic benchmark. Only ~430 rows (1.3% of the shard) targeted
the JSON-shaped attribution surface that CyberSOCEval-TI exercises.

The v15 W1 plan (`v15_plan.txt §3`) described the shard as "CANON
excluded", but the v14 watcher was actually injecting Canonical data via
two separate generator phases (Phase 3 `taa_canon_generator.py` and
Phase 3e `misp_taa_generator.py`) that ran independently of the v14
manifest. The plan document under-described the watcher's behaviour and
the v15 W1 result is consistent with the corrected composition: the
+16.5pp on TAA Canonical reflects two-thirds of the shard being CANON
training data; the flat TAA Classic reflects the model receiving very
little Classic / JSON-attribution training relative to other vintages.

### 1.3 v16 hypothesis

If the v15 W1 TAA-Classic flatline was caused by **shard dilution**
(Canonical training crowding out the attribution surface), then a
shard-composition rebuild that drops Canonical entirely and bumps the
attribution + JSON families should move TAA Classic without any recipe
change. v16 is that experiment. The Canonical purge is total (zero
`TAA.CANON` and zero `MISP.CANON` rows in the manifest, and the v16
watcher drops the corresponding `taa_canon_generator.py` and
`misp_taa_generator.py` phases). The Count: bumps target ~22-26K actual
rows after balancing, of which ~12K are JSON-shaped attribution (vs ~430
in v15 W1) and ~14K are Classic / IE / NEG.

If v16 moves TAA Classic, dilution was the bottleneck and the v15
architecture is validated for production. If v16 still doesn't move TAA
Classic, the bottleneck is template-bound (5%-accurate vs 93%-plausible
pattern from v14.1 D-TAA) and v17 will need explicit hard-negative
attribution pairs and/or recipe changes (`v16_plan.txt §4` Outcome C).

## 2. Design rationale (per-template Count: targets)

Full delta table in `Sophia-CTI-Templates-v16.txt §"v15 W1 -> v16 deltas"`.
Highlights:

- **`AB.TAA.{1,2,3,5}` lifted to Count: 3000 each** (was default ~1500-
  2000). Per-actor cap lifted 40 -> 60 to give the bumped Counts more
  per-actor headroom; expected actuals ~3000/template given the 187
  intrusion-set anchor pool documented in the local Neo4j.
- **`AB.TAA.4` held at Count: 200** (campaign-bound; only 25 documented
  campaigns in `athena-cti-db`; raising Count: produces no new anchors).
- **`AB.TAA.IE.{1,2}` and `AB.TAA.NEG.1` lifted to Count: 2500** (was
  1500). IE rows draw from the unbound MITRE ATT&CK technique catalogue
  (835 attack-patterns, 91 tools); NEG rows use `Per_primary_grouping:
  false` so the (grp, rel) pair diversifies across rows.
- **`JS.TAA.{1,2,3}` lifted to Count: 2500 each** (was 400 -- the
  smoking gun). JSON-shaped attribution gets ~7,500 declared rows in v16
  vs ~430 actual in v15 W1.
- **`JS.TAA.IE.1` and `JS.TAA.NEG.1` lifted to Count: 2000** (was
  500-1000).

## 3. Run-book

### 3.1 Generation

```bash
cd /Users/pietro/code/Glaukopis     # or the cluster equivalent
# Pre-flight: confirm Neo4j athena-cti-db is reachable and populated
python _v16_build/_neo4j_check.py
# Expected: intrusion-sets >= 180; group-uses-attack-pattern >= 4000

bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05102026/Sophia-CTI-Templates-v16.txt \
     _v16_build/triples \
     SFT/data/ift_data_2026_05_10_v16.raw.json \
     10 3500 \
     2>&1 | tee _v16_build/build.log &
echo "PID=$!" > _v16_build/build.pid

nohup bash _v16_build/watcher.sh > _v16_build/watcher.log 2>&1 &
```

The watcher polls the build PID, then runs Phases 4-8 (actor-balance,
dedup, row-count gate, licence gate, stratified shuffle, val/train
split). Final outputs:

- `SFT/data/ift_data_2026_05_10_v16_taa.json`  (training shard)
- `SFT/data/ift_data_2026_05_10_v16_val.json`  (held-out validation slice)
- `_v16_build/watcher_status.json`             (per-phase row counts and reports)

### 3.2 Dataset registration

The launcher resolves dataset names through `SFT/data/dataset_info.json`.
The v16 entries (`ift_data_2026_05_10_v16_taa`, `ift_data_2026_05_10_v16_val`)
are registered in §"v16 (2026-05-10) -- v15 W1-rev TAA specialist".

### 3.3 Training

```bash
bash SFT/autotrain/run_sft_qwen25_14b_v12_plus_v16_taa.sh
# defaults to ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v16
# wall-time ~6-8 h on 8xH100
```

### 3.4 Bench

Standard 14B AthenaBench sweep against the v16 HF checkpoint; compare
against v12 (57.3 weighted total, 11.0 TAA Classic) and v15 W1 per the
decision matrix in `v16_plan.txt §4`.
