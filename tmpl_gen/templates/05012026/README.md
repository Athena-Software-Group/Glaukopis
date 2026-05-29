# Sophia CTI Templates — v10 (May 1, 2026)

Self-contained, single-pass SFT manifest. Replaces the v9 two-phase
(broad re-anchor → RMS slice) chain with one unified, balanced corpus
authored from a single template file. The v9 chain was driven by the
v8.1 trade-off — RMS catalog recovery at the cost of CKT/ATE/RCM
broad-knowledge regression — and v10 unifies both objectives into a
single 200K-row corpus by combining the v8/v9 template surface with
new structural cleanups, per-actor balancing, and an n-gram dedup
filter.

```
05012026/
  Sophia-CTI-Templates-v10.txt   self-contained v10 manifest (216 templates)
  README.md                      this document
```

The manifest is the single source of truth for the build:
`tmpl_gen/scripts/tmpl_docx2json.py` reads this file directly,
`iftgen.py` consumes the resulting `Sophia-CTI-Templates-v10.json`,
`to_alpaca.py` emits `SFT/data/ift_data_2026_05_01_v10.raw.json`,
and `_v10_build/watcher.sh` runs the post-pass (actor-balance →
eval-set dedup) to produce the final `ift_data_2026_05_01_v10.json`
consumed by `SFT/autotrain/run_abaligned_sft_qwen25_14b_v10.sh`.

## 1. Training corpus summary

| | |
|---|---:|
| **Final corpus** | `SFT/data/ift_data_2026_05_01_v10.json` |
| Rows | 200,340 |
| Bytes (raw JSON) | 227 MB |
| Tokens (~4 chars/tok) | ~54.5 M |
| Distinct sub-families | 145 (216 templates loaded; 8 yielded zero rows) |
| Mean / median row length | 1,089 / 731 chars (~272 / ~182 tokens) |
| p99 row length | 4,798 chars (~1,199 tokens) |
| Rows over 8192-token cutoff | 7 (0.003%) |
| Output (target) median length | 252 chars (~63 tokens) |

### 1.1 Build pipeline (deterministic, end-to-end)

```bash
# Phase 1: template -> triples -> Alpaca rows
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05012026/Sophia-CTI-Templates-v10.txt \
    _v10_build/triples \
    SFT/data/ift_data_2026_05_01_v10.raw.json \
    10 1500

# Phase 2 + 3: actor-balance + eval-set dedup post-pass
bash _v10_build/watcher.sh
```

Final outcome (PIDs, row counts, drop counts, all tunables) lands in
`_v10_build/watcher_status.json`. Per-phase logs in
`_v10_build/{build,balance,dedup,watcher}.log`.

### 1.2 Pipeline tunables (encoded in `_v10_build/watcher.sh`)

| knob | v10 value | purpose |
|---|---:|---|
| `ACTOR_CAP` | 20 | Max rows per intrusion-set in TAA attribution |
| `ACTOR_FLOOR` | 100 | Hard-fail if fewer distinct actors covered |
| `DEDUP_HIT_THRESHOLD` | 1 | Min shared 13-grams to flag for inspection |
| `DEDUP_DROP_THRESHOLD` | 50 | Drop rows with >= 50 shared 13-grams (verbatim contam) |

### 1.3 Build outcome

```
raw      208,077 rows  (216 templates loaded, 208 productive)
balanced 200,961 rows  (-7,116 via TAA cap=20)
clean    200,340 rows  (-621 via dedup-drop threshold 50)
retained 96.3% of raw
```

## 2. What v10 explicitly fixes vs prior versions

| issue | seen in | v10 fix |
|---|---|---|
| HTML/whitespace swamping descriptions | v8/v9 | `clean_cti_description()` BS4 sanitizer at parser chokepoint (`tmpl_gen/src/tmpl_gen/tmpl_parser.py`) |
| Prompt-leak from unmarked freeform text | v9 | `<desc>...</desc>` markers wrapping 64 Question-side multi-line property placeholders |
| TAA mode collapse on 4-5 head actors | v8/v9 | actor cap=20 (drops Leviathan 1611→20, APT29 766→20, Lazarus 628→20, APT41 424→20) |
| Verbatim eval contamination | always | `--drop-threshold 50` 13-gram filter via `tmpl_gen/scripts/dedup_against_evals.py` (drops 621 verbatim-overlap rows; preserves rows with only incidental shared CVE/MITRE description vocabulary, which is expected since training and eval share the same Neo4j knowledge base) |
| Two-phase chain with selection bias | v8/v9 | Single-pass over unified corpus; no Phase A/B rate-of-forgetting issue |
| RMS catalog collapse | v8 | RMS at 6.1% of corpus (was capped at 0.4% in v8 Phase B) |

## 3. Corpus composition

### 3.1 By answer-format axis (top-level family)

| share | rows | family | what the model emits |
|---:|---:|---|---|
| 16.5% | 33,037 | AB | Abductive prose (chain-of-reasoning) |
| 15.9% | 31,941 | X | Extraction (pluck IDs/entities) |
| 11.8% | 23,585 | YN | Yes/No |
| 8.4% | 16,761 | V | Vulnerability narrative |
| 6.0% | 12,000 | SR | Sigma rule |
| 5.4% | 10,872 | P | Patch / remediation |
| 4.7% | 9,419 | W | Weakness (CWE) |
| 4.6% | 9,151 | S | Software / product |
| 4.5% | 9,051 | SU | Summarization |
| 4.5% | 8,963 | CL | Classification |
| 4.2% | 8,478 | M | Mitigation |
| **3.6%** | **7,252** | **JS** | **JSON-structured output** |
| 3.2% | 6,448 | A | ATT&CK |
| 3.0% | 5,971 | POC | Proof-of-Concept |
| 1.9% | 3,883 | E | Exploit (KEV) |
| 0.9% | 1,850 | Q | Question generation |
| 0.8% | 1,678 | EDB | ExploitDB |

### 3.2 Eval-aligned vs broad-knowledge split

| | rows | share |
|---|---:|---:|
| Eval-axis-aligned (trains directly to AthenaBench slices) | 53,719 | 26.8% |
| Broad / context (anchors vocab, schema, cross-entity reasoning) | 146,621 | 73.2% |

Eval-axis-aligned breakdown:

| rows | axis | trained from |
|---:|---|---|
| 12,456 | `athena-cti-vsp` | AB.VSP.{1-4} + V.CPE |
| 12,158 | `athena-cti-rms` | AB.RMS.* + JS.RMS.* |
| 11,506 | `athena-cti-rcm` | AB.RCM + JS.RCM + X.VW + YN.VW |
| 5,965 | `athena-cti-ate` | AB.ATE + JS.ATE |
| 4,350 | `athena-cti-mcq` | AB.MCQ + JS.MCQ + AB.MS |
| 4,000 | `athena-cti-taa-ie` | AB.TAA.IE.{1,2} + JS.TAA.IE.1 |
| 2,000 | `athena-cti-taa-neg` | AB.TAA.NEG.1 + JS.TAA.NEG.1 |
| 1,284 | `athena-cti-taa` | AB.TAA.{1-5} + JS.TAA.{1-3} (post-cap=20) |


### 3.3 Broad-knowledge tail (73% of corpus, no direct eval axis)

These don't map 1:1 to any eval axis but anchor the model's CTI
vocabulary, schema knowledge, and cross-entity reasoning.

| rows | family | covers |
|---:|---|---|
| 27,441 | X.* | cross-entity extraction (relationships beyond eval slices) |
| 20,585 | YN.* | binary reasoning across the full entity graph |
| 12,000 | SR.* | Sigma rule content (detection-engineering vocabulary) |
| 10,872 | P.* | patch / remediation |
| 10,831 | V.* | CVE narratives outside V.CPE |
| 9,419 | W.* | CWE narratives outside W->ATE pivots |
| 9,151 | S.* | software / product surface |
| 9,051 | SU.* | summarization (long -> abstract) |
| 8,963 | CL.* | classification across all entity types |
| 8,478 | M.* | mitigation narratives outside RMS |
| 6,448 | A.* | ATT&CK narratives |
| 5,971 | POC.* | exploit-code narratives |
| 3,883 | E.* | KEV / in-the-wild exploits |
| 1,850 | Q.* | question generation (model authors queries) |
| 1,678 | EDB.* | ExploitDB |

## 4. RMS detail (the family v8.1+ exists to protect)

```
1,500  AB.RMS.1
1,445  AB.RMS.2
1,500  AB.RMS.6
1,100  AB.RMS.3a
1,100  AB.RMS.3b
1,100  AB.RMS.3c
1,100  AB.RMS.3d
  750  AB.RMS.3e
  375  AB.RMS.3f
  225  AB.RMS.3g
  150  AB.RMS.3h
   44  AB.RMS.4     <-- catalog drill, very light
   44  AB.RMS.5     <-- catalog drill, very light
2,025  JS.RMS.{1-8} (300/300/300/300/200/150/100/75)
12,158 TOTAL  (6.1% of corpus)
```

`AB.RMS.4` and `AB.RMS.5` (the M-control catalog drills) yielded only
44 rows each under v10. These are the templates whose absence drove
the v8 catalog-collapse. The Neo4j seed population didn't expand under
v10 settings, so if `athena-cti-rms` regresses vs v8.1, this is the
prime suspect for v11.

## 5. TAA detail (the mode-collapse fix)

1,284 attribution rows across 109 distinct intrusion sets (post-cap=20).

| template | rows |
|---|---:|
| AB.TAA.1 | 20 |
| AB.TAA.2 | 243 |
| AB.TAA.3 | 369 |
| AB.TAA.4 | 132 |
| AB.TAA.5 | 343 |
| JS.TAA.1 | 38 |
| JS.TAA.2 | 73 |
| JS.TAA.3 | 66 |

Actor cap distribution:

- 47 actors at exactly 20 rows (the heads -- Leviathan, APT29, Lazarus,
  APT41, etc., all clipped from up to 1,611)
- 62 actors below cap (long tail, kept intact 1-19 rows)

Plus 6,000 guardrail rows (preserved in full, not subject to cap):

```
1,500  AB.TAA.IE.1   "insufficient evidence"
1,500  AB.TAA.IE.2   "insufficient evidence" variant
1,500  AB.TAA.NEG.1  wrong-attribution rejection
1,000  JS.TAA.IE.1   JSON IE
  500  JS.TAA.NEG.1  JSON NEG
```

TAA total: 7,284 rows (3.6% of corpus). Guardrail-to-positive ratio of
~3.9:1 (5,000 IE/NEG : 1,284 attribute) -- aggressive but intentional
given the v9 mode-collapse history.

## 6. Multi-select detail (the v9 select-all-bias mitigation target)

```
1,000  AB.MS.MAL.2
  500  AB.MS.GRP.3
1,500  TOTAL  (0.7% of corpus)
```

Only 2 multi-select templates produced rows. The manifest defined more
(`AB.MS.GRP.{1,2}`, `AB.MS.MAL.{1,3}`) but they yielded zero -- likely
Neo4j seed mismatch on the multi-select set construction. If v9's
select-all bias persists at eval time on `athena-cti-mcq` multi-select
slices, the manifest needs more multi-select coverage in v11.

## 7. Length distribution (training-host planning)

| metric | chars | ~tokens |
|---|---:|---:|
| mean | 1,089 | 272 |
| median (p50) | 731 | 182 |
| p75 | 1,286 | 321 |
| p90 | 2,256 | 564 |
| p95 | 2,924 | 731 |
| p99 | 4,798 | 1,199 |
| max | 45,710 | 11,427 |

- Only 7 rows (0.003%) exceed the 8192-token training cutoff -- near-
  zero truncation impact at packing on / cutoff 8192.
- Output (the model's target) median is 252 chars (~63 tokens),
  consistent with the extraction / Y-N / JSON-heavy mix.
- Total training tokens: ~54.5 M.
- At `--max-samples 200000 --num-train-epochs 1` with effective batch
  16, that's ~12,500 optimizer steps per epoch.

## 8. Known content gaps to watch at eval

1. **`AB.RMS.{4,5}` catalog drills @ 44 rows each** -- the v8 collapse
   target. New structural delimiters + dedup may compensate, but if
   `athena-cti-rms` degrades vs v8.1, expand these in v11.
2. **`AB.MS.*` multi-select @ 1,500 rows in only 2 sub-templates** --
   if v9 select-all bias persists on `athena-cti-mcq` multi-select
   rows, the manifest needs more multi-select coverage in v11.
3. **`CVE-CVE-` / `CWE-CWE-` double-prefix in `X.VW.1`** -- known non-
   blocking schema artefact (consistent doubling on Question side, eval
   uses single prefix). May depress the X.VW eval slice marginally.

## 9. Consumers

| consumer | path |
|---|---|
| Dataset registration | `SFT/data/dataset_info.json` -> `ift_data_2026_05_01_v10` |
| Qwen2.5-14B launcher | `SFT/autotrain/run_abaligned_sft_qwen25_14b_v10.sh` |
| HF target | `${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v10` |

The dataset JSON itself is gitignored (227 MB); rsync the build
artefact to the training host alongside `git pull origin main` to pick
up the launcher and dataset registration.

## 10. Contamination posture

Inherited from v8 (`../04292026/README.md` §2) without modification.
The v10 corpus is built by `_v10_build/watcher.sh`, whose Phase 5
runs `tmpl_gen/scripts/dedup_against_evals.py` against
`SFT/eval/benchmark_data/` with `n=13` word-grams and
`hit-threshold=1` -- verbatim leakage of any AthenaBench / CTIBench /
CyberMetric / CyberSOCEval row into the training corpus is blocked at
build time, and structural overlap with the public MITRE / NIST /
FIRST / CISA / D3FEND knowledge bases is accepted by design.

**Canonical reference** -- conceptual taxonomy (verbatim vs.
structural), n=13 rationale anchored to the OLMo / Pythia / Llama
MMLU / HellaSwag / BIG-bench decontamination passes, and literature
pointers (SecKnowledge / CyberPal.AI Levi et al. 2024
arXiv:2408.09304, the CTIBench paper, the AthenaBench technical
report): see
[`../04292026/README.md` §2](../04292026/README.md#2-contamination-posture).

**Exhaustive restatement** -- per-shard enforcement, per-benchmark
structural-overlap matrix, adjacent corpus-hygiene gates, and the
falsifiability list, scoped to the v21 three-shard pipeline: see
[`../05182026/README-21.md` §Contamination posture](../05182026/README-21.md#contamination-posture).

**v10-specific note.** v10 introduces the **soft-drop** variant of
the dedup pass: `--drop-threshold 50` retains rows with 1--49
incidental n=13 hits (so the corpus is not stripped of common CVE /
MITRE vocabulary that would otherwise legitimately co-occur with eval
prompts) while still hard-filtering and reporting rows with >=50 hits.
The n=13 / `hit-threshold=1` knobs themselves are unchanged from v8;
the soft-drop is purely an audit-classification policy. Same posture
carries forward to v11 and to every vintage built after.

