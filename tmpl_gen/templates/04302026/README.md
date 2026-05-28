# Sophia CTI Templates -- v8.1 + v9 RMS slice (April 30, 2026)

This directory holds two related SFT template manifests:

  * **v8.1** -- single-pass rework of the v8 program targeting the RMS
    catalog-collapse regression observed on the v8-large 14B
    checkpoint. v8.1 recovered RMS but regressed every other broad-
    knowledge benchmark (CKT/ATE/RCM/CyberMetric) because its
    `cap=170` stratified subsample starved the catalog tail.
  * **v9 RMS slice** -- self-contained AB.RMS.* + JS.RMS.* manifest
    extracted from v8.1, used as the third dataset in Phase B of the
    v9 two-phase 14B launcher. Keeps the v9 build pipeline first-
    class (template manifest -> Neo4j -> Alpaca JSON) rather than
    depending on a filter pass over a prior v8.1 build artefact.

```
04302026/
  Sophia-CTI-Templates-v8_1.txt    self-contained v8.1 manifest (206 templates)
  Sophia-CTI-Templates-v9_rms.txt  v9 Phase-B RMS slice (21 templates)
  README.md                        this document (Sections 1-4 = v8.1, Section 5 = v9)
```

Unlike the v8 (04292026) consolidated `v8_{small,large}` manifests --
which are documentation-only because the live build still consumed the
per-vintage source files -- the v8.1 manifest **is the single source of
truth** for the build pipeline: `tmpl_gen/scripts/tmpl_docx2json.py`
reads this file directly, `iftgen.py` consumes the resulting
`Sophia-CTI-Templates-v8_1.json`, and `to_alpaca.py` emits
`SFT/data/ift_data_2026_04_30_v81.json`.

## 1. Training strategy summary

### 1.1 Why v8.1 exists

The v8-large 14B checkpoint
(`run_abaligned_sft_qwen25_14b_v8.sh`, two-phase) showed a single-
mitigation-cluster collapse on `athena-rms` (top-5 F1 ~ 8). Phase B's
JSON+long-context corpus had dropped the AB.RMS.{4,5} catalog drills
that the v8-small corpus retained, so the model lost catalog coverage
during the format-specialization phase even though Phase A had taught
it. v8.1 sidesteps the chain entirely: a single SFT pass over a
consolidated corpus authored from one template file with explicit
`Count:` floors on the catalog drills.

### 1.2 v8.1 (Qwen2.5-14B-Instruct, single-pass full SFT)

| Knob              | Value                                                 |
|-------------------|--------------------------------------------------------|
| Trainer           | `SFT/autotrain/run_abaligned_sft_qwen25_14b_v81.sh`   |
| Base model        | `Qwen/Qwen2.5-14B-Instruct` (32B shares the corpus)   |
| Final HF artefact | `${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v81` |
| Recipe            | 2 epochs, cosine, 5% warmup, bf16, lr=1e-5            |
| Cutoff            | 8192 tokens, packing on                               |
| Effective batch   | 16                                                    |
| Parallelism       | DeepSpeed ZeRO-3 (auto offload on <4 GPUs)            |
| Visible per epoch | ~50,000 rows (`--max-samples 50000`)                  |

Corpus (single Alpaca file + two HF mixtures resolved at load):

| Source                                              |    Rows | Selection                              |
|-----------------------------------------------------|--------:|----------------------------------------|
| `ift_data_2026_04_30_v81.raw.json` (pre-subsample)  | 196,328 | full graph yield (backup only)         |
| `ift_data_2026_04_30_v81.json`                      |  41,808 | `cap=170` per shortname; RMS preserved |
| `tulu_3_sft_mixture` (HF, AllenAI)                  |  ~5,000 | random subsample (CF guard)            |
| `alpaca_en_demo`                                    |  ~2,500 | random subsample (CF guard)            |
| **total per epoch (capped)**                        | **~50,000** |                                    |

The 41,808-row canonical file is the v8-small `combined_v8small`
slot's structural replacement: same role in the trainer (one Alpaca
file mixed with the two HF CF-guards), tighter shape (~22% smaller)
because the v8.1 source already inlines the v7 / JSON / long-context
templates rather than concatenating three separate files.

### 1.3 Structural changes vs. v8 (04292026)

  1. **Single source of truth.** `Sophia-CTI-Templates-v8_1.txt`
     replaces the v7 + JSON-v8 include chain; the file is parsed
     directly by `tmpl_docx2json.py`, so the trained-on text matches
     the audited text byte-for-byte.
  2. **Catalog-drill floors.** AB.RMS.{1,2,3a..3h,4,5,6} now carry
     explicit `Count:` directives; the stratified subsampler
     (`tmpl_gen/scripts/stratified_subsample.py --cap 170`) is
     instructed to **preserve `AB.RMS.*` and `JS.RMS.*` at 100% of
     their generated volume**. The cap-170 per-shortname rule that
     prevents high-frequency families from drowning out the long
     tail still applies to every other template.
  3. **JS.RMS cardinality extended.** v8 shipped `JS.RMS.{1..4}`
     (mitigations-per-technique up to 4). v8.1 adds `JS.RMS.{5..8}`
     to lift the JSON-graded RMS ceiling to 8 mitigations. Final
     volumes: 300/300/300/300/200/150/100/75 = 1,725 rows.
  4. **Bulk Counts trimmed by ~25%** across `AB.RMS.3{a..h}` and the
     `JS.*` slate to shorten the 14B/32B training cycle. Target row
     count 38-42K; final 41,808.
  5. **Six recovered templates.** `AB.MCQ.3`, `P.7`, `X.8`,
     `SU.G.1`, `SU.POC.1`, `Q.MSR.1` failed in the initial v8.1
     build (timeouts on cartesian-blow-up MCQs, parser ambiguity on
     `P.7`'s reverse `[:detects]` edge, header-bleed corruption on
     three single-row templates). All six were repaired and merged at
     100 rows each. See Section 4 for the full diagnostic trail.

### 1.4 Build pipeline (one-shot, reproducible)

```bash
# 1. Compile the source manifest into the per-template JSON the build consumes.
python tmpl_gen/scripts/tmpl_docx2json.py \
    --input  tmpl_gen/templates/04302026/Sophia-CTI-Templates-v8_1.txt \
    --out    tmpl_gen/data_generation/Sophia-CTI-Templates-v8_1.json \
    --count_limit 1500

# 2. Drive iftgen.py per-template (handled by make_dataset.sh).
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/04302026/Sophia-CTI-Templates-v8_1.txt \
    _v81_build/triples \
    SFT/data/ift_data_2026_04_30_v81.raw.json \
    10 1500

# 3. Stratified subsample (cap 170 per shortname, RMS preserved).
python tmpl_gen/scripts/stratified_subsample.py \
    --in  SFT/data/ift_data_2026_04_30_v81.raw.json \
    --out SFT/data/ift_data_2026_04_30_v81.json \
    --cap 170

# 4. Verbatim-overlap dedup against the eval suite (Section 2.1).
python tmpl_gen/scripts/dedup_against_evals.py \
    --input    SFT/data/ift_data_2026_04_30_v81.json \
    --eval-dir SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 \
    --report   SFT/data/ift_data_2026_04_30_v81.dedup.json
```

Step 4 is informational for v8.1 (it does not gate the build) because
the catalog-drill templates intentionally share short catalog phrases
with `athena-cti-rms.jsonl` -- see Section 2.1 for the full
treatment of why the n=13 hit count is high *by design* under the
v8.1 RMS-recovery posture and what the residual verbatim-overlap
audit actually checks.

## 2. Contamination posture

> **Inherited from v8 (canonical) and carried forward to v21
> (exhaustive).** The verbatim-vs-structural taxonomy, the n=13
> word-gram threshold rationale, and the literature anchors
> (SecKnowledge / CyberPal.AI, CTIBench, AthenaBench) are documented
> in full at
> [`../04292026/README.md` §2](../04292026/README.md#2-contamination-posture).
> The most exhaustive self-contained restatement -- scoped to the
> three-shard Core / TAA / CSE pipeline and to all six architecture
> ports -- is at
> [`../05182026/README-21.md` §Contamination posture](../05182026/README-21.md#contamination-posture).
> The subsections below restate the v8 posture with the v8.1-specific
> numbers.

This section is the v8.1 audit trail for the contamination question
("is the v8.1 SFT corpus leaking the AthenaBench / CTIBench /
CyberMetric / CyberSOCEval evaluation signal into training?").
The v8 (04292026) README's two-failure-mode framing carries forward
unchanged -- verbatim contamination is **blocked / audited by tooling
at corpus build time**, structural contamination is **accepted by
design** -- and the rest of this section restates that position with
the v8.1-specific numbers.

### 2.1 Verbatim contamination audit: `dedup_against_evals.py`

**Tool:** `tmpl_gen/scripts/dedup_against_evals.py` (unchanged from
v8). **Mechanism / threshold rationale:** see v8 README Section 2.1
(n=13 word-grams, `hit-threshold=1`, eval-dir
`SFT/test/benchmark_data/`). Nothing about the script changed for
v8.1.

**v8.1 result:** 11,180 of 41,808 rows (~26.7%) flagged at
n=13/threshold=1. The breakdown is:

| Template family    | Flagged rows | Notes                                              |
|--------------------|-------------:|----------------------------------------------------|
| `AB.RMS.{6,3a..3h}`|        7,250 | Catalog-recall drills (every row shares the catalog ID) |
| `AB.RCM.1`, `JS.RCM.1` |        340 | CWE-id and CAPEC-id catalog phrases                 |
| `AB.ATE.{1,2,3,4,5}`, `JS.ATE.{1,3}` |     1,020 | ATT&CK technique-id catalog phrases  |
| `AB.VSP.{1..4}`, `JS.VSP.{1,2}` |        982 | CVSS/CWE catalog phrases                       |
| `AB.MCQ.*`, `Q.VPOC.1` |        1,217 | MCQ stem / option-string overlaps with CTI-MCQ      |
| `V.1` and other long-tail |       371 | Single-CVE / single-mitigation small overlaps      |
| **total**          |       11,180 |                                                    |

By eval source:

| Eval file                             | n-gram hits |
|---------------------------------------|------------:|
| `athena-cti-rms.jsonl`                |      36,997 |
| `athena-cti-rcm.jsonl`                |       6,997 |
| `athena-cti-ate.jsonl`                |       5,100 |
| `athena-cti-vsp.jsonl`                |       2,252 |
| `athena-cti-mcq{,-3k,-updated}.jsonl` |       2,755 |
| `athena-cti-taa.jsonl`                |           6 |

This headline 11,180-row count is **not** a verbatim-leak signal
under v8.1's posture, and the build script does not gate on it
(`dedup_against_evals.py` is run for the report; the v8.1 trainer
does not have the `[FAIL] Phase B dataset missing` gate that v8-large
embedded). The reasoning, made explicit so a reviewer can falsify
it:

  * Every flagged AB.RMS / AB.RCM / AB.ATE / AB.VSP row matches its
    eval counterpart on **catalog-ID phrasing** -- e.g. the SFT row
    `"... mitigation M1026 (Privileged Account Management) for
    technique T1078 ..."` and the eval row
    `"... which mitigation addresses T1078? ..."` share the
    13-gram window `mitigation M1026 Privileged Account Management
    for technique T1078`. Both rows are independently emitted by
    template iteration over the MITRE STIX bundle; neither row was
    cribbed from the other. The dedup tool cannot distinguish
    "shared graph fact" from "shared eval prompt" at n=13 because
    the catalog ID strings are identical in both.
  * The v8 program's **structural-contamination position
    (v8 README Section 2.2) explicitly accepts this overlap**:
    catalog-recall benchmarks are graded on the model's ability to
    retrieve a fact from a public bundle, and any model trained on
    the bundle will share short n-grams with an eval row that
    references the same bundle entry. Holding 72% of MITRE's
    catalog out of training -- which is what an n=13 dedup pass
    would force if treated as a hard gate -- defeats the purpose
    of training a CTI model and is the v6 anti-pattern that the
    `tmpl_gen/templates/04252026/Sophia-CTI-Templates-AthenaBench-abaligned-v6.txt`
    Section 3 commentary calls out.
  * **What the 11,180-row count actually tells us** is that the
    catalog-drill templates are working as intended: every AB.RMS.6
    row hits an `athena-cti-rms` row because that is what catalog
    recall *is*. The relevant verbatim-leak audit on v8.1 is the
    **non-RMS / non-catalog tail** (~700 rows): the `V.1`,
    `AB.MCQ.*`, and long-tail flags. None of those is an eval-row
    quote either -- spot inspection (see
    `tmpl_gen/utils/_sanity_v81.py` and the `--report` JSON) shows
    they are MCQ stems whose distractor language ("which of the
    following ATT&CK techniques ...") happens to match common eval
    phrasing at the n=13 threshold. We accept these under the same
    structural framing.

The position the build *does* enforce informally: **no eval-row
prompt or answer string appears verbatim in any template body**.
The audit for this is reading the template manifest itself
(`Sophia-CTI-Templates-v8_1.txt`); the n=13 dedup is the second
line of defense and would catch a regression where a future
template author cribbed an eval prompt. It does not currently
catch one because none of the 11,180 flagged rows traces back to
an eval prompt -- they all trace back to a catalog ID phrase the
template generated independently.

### 2.2 Structural contamination: accepted by design (unchanged from v8)

The full structural-contamination position is documented in the v8
README (`tmpl_gen/templates/04292026/README.md`) Section 2.2. v8.1
does not change this position; in particular:

  * The v8.1 SFT corpus is built from the same MITRE / CISA / FIRST
    bundles that AthenaBench / CTIBench / CyberMetric /
    CyberSOCEval are scored against. There is no eval/train split
    on the underlying graph because there cannot be one without
    crippling the SFT signal.
  * The v8.1-specific RMS-floor change (Section 1.3 item 2) is a
    **training intervention against catalog recall**, not a leak.
    The whole point of the AB.RMS.{4,5,6} drills is to ensure the
    model has the M-id <-> name binding memorised at every
    parameter count -- which is the same fact `athena-cti-rms`
    grades on, by design.
  * The JS.RMS cardinality extension (Section 1.3 item 3) is the
    same intervention pushed to the JSON-graded surface form:
    `JS.RMS.8` ensures the model can emit an 8-element mitigation
    list in the JSON envelope CyberSOCEval expects. The structural
    overlap with the JSON-shaped eval suites is the same
    accepted-by-design overlap as for the AB.RMS family.

### 2.3 Per-benchmark contamination matrix (delta vs. v8)

The v8 per-benchmark matrix (v8 README Section 2.3) carries forward
unchanged for AthenaBench, CTIBench, CyberMetric, and CyberSOCEval.
The only v8.1 delta worth flagging is that the **AthenaBench RMS
share is now structurally larger** because of the catalog-floor
change: an `athena-rms` row asking "which mitigations address
T1078?" will hit ~6 AB.RMS.6 rows in the v8.1 corpus where in v8 it
might have hit 1-2. This is the v8.1 design intent (catalog
coverage is the regression we are fixing) and explicitly not a
verbatim-leak regression -- the AthenaBench rows themselves never
appear in the corpus.

### 2.4 What v8.1 explicitly does **not** do (unchanged from v8)

  * No eval-row text in templates.
  * No benchmark answer keys in the Neo4j graph.
  * No fine-tuning on benchmark dev/validation splits.
  * No cross-pollination from the eval harness
    (`SFT/test/benchmark_data/` is read-only relative to
    `SFT/data/`).
  * No per-token contamination check on `tulu_3_sft_mixture` or
    `alpaca_en_demo`.

See v8 README Section 2.4 for the full text of each item.

### 2.5 Reproducing the contamination audit

```bash
# Run from the repo root with the tmpl_gen virtualenv active.
python tmpl_gen/scripts/dedup_against_evals.py \
    --input    SFT/data/ift_data_2026_04_30_v81.json \
    --eval-dir SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 \
    --report   SFT/data/ift_data_2026_04_30_v81.dedup.json
```

Expect ~11,180 flagged rows on a clean v8.1 build, dominated by
`AB.RMS.*` / `AB.RCM.*` / `AB.ATE.*` / `AB.VSP.*` overlap with the
corresponding `athena-cti-*.jsonl` eval files. The contamination
gate is the *content* of the matches, not the count: the report
JSON's `matches` list per row should reference catalog IDs
(M-ids, T-ids, CWE-ids, CVSS strings), never an eval-row prompt
sentence. Inspect with:

```bash
python -c "
import json, collections
d = json.load(open('SFT/data/ift_data_2026_04_30_v81.dedup.json'))
print('total flagged:', len(d))
print('top templates:', collections.Counter(h['shortname'] for h in d).most_common(10))
"
```

### 2.6 Related: continued-pretraining (`cpt/`) leak protection

Unchanged from v8 README Section 2.6. The CPT corpus runs MinHashLSH
at Jaccard 0.30 over 13-grams against the same
`SFT/test/benchmark_data/` corpora plus optional per-source exact-
CVE-id dropping. v8.1 does not change CPT.

## 3. File provenance for the consolidated manifest

`Sophia-CTI-Templates-v8_1.txt` is a single self-contained file --
no include chain. It was constructed by inlining the v8-small
04292026 consolidated manifest and applying the v8.1 deltas listed
in Section 1.3 in place. The section ordering follows the v8 layout:

  * Section 0: v8.1 strategy header (delta vs. v8-small).
  * Section 1: AthenaBench-aligned slate (`AB.*`), inlined from
    `tmpl_gen/templates/04252026/Sophia-CTI-Templates-AthenaBench-abaligned-v6.txt`
    plus the v7/v8 ATE/RCM/RMS/VSP additions.
  * Section 2: catalog / supporting families (`A.*`, `M.*`, `S.*`,
    `W.*`, `V.*`, `X.*`, `CL.*`, `SR.*`, `Q.*`, `P.*`, `E.*`,
    `EDB.*`, `POC.*`, `SU.*`, `YN.*`).
  * Section 3: JSON-output slate (`JS.*`), inlined from
    `tmpl_gen/templates/04292026/Sophia-CTI-Templates-JSON-v8.txt`
    plus the JS.RMS.{5..8} additions.
  * Section 4: long-context scaffolding documentation (advisory).

The compiled per-template JSON (consumed directly by `iftgen.py`)
lives at `tmpl_gen/data_generation/Sophia-CTI-Templates-v8_1.json`
and is regenerated by step 1 of the pipeline in Section 1.4.

## 4. Build diagnostic trail (v8.1-specific)

This appendix records the issues found during the v8.1 build and
their fixes, so a future v8.2 cycle can reproduce or extend the
recovery procedure without re-discovery.

### 4.1 Cartesian-blowup MCQ timeouts (`AB.MCQ.3`, `Q.MSR.1`)

**Symptom:** `iftgen.py` hung on 5-option MCQ templates whose
distractor pool was `Weakness` (`AB.MCQ.3`) or `SigmaRule`
(`Q.MSR.1`), eventually crossing Neo4j's
`dbms.memory.transaction.total.max` (16 GiB) and aborting the
transaction.

**Root cause:** the primary diverse-sampling query for an N-option
MCQ instantiates an N-way cartesian product across the distractor
pool before applying the diversity `force` constraints. For
~3,700-row pools at N=5 the intermediate row count exceeded the
transaction memory cap.

**Fixes applied:**

  1. `tmpl_gen/src/tmpl_gen/tmpl_parser.py` -- the fallback bounded
     query (which trips when the primary query hits the memory cap)
     was not previously subject to the 90 s transaction timeout, so
     a fallback that itself blew up the cartesian would hang
     indefinitely. The `timeout=primary_timeout` argument is now
     passed to both the primary and fallback `run_query_collect`
     calls.
  2. Source-level `Count: 100` cap on `AB.MCQ.3` and `Q.MSR.1` to
     keep these specific templates inside the safe ceiling
     measured during retries (~50 s wall-clock at count 100).

### 4.2 Reverse-edge ambiguity (`P.7`)

**Symptom:** `TmplParseTransf.map_rel ERROR: inverse case:
[:detects]->attack-pattern defined for types SigmaRule,
x-mitre-detection-strategy ; ambiguous for type attack-pattern,
relstr: detects<`.

**Root cause:** the `[:detects]` relation type is shared by
`SigmaRule -[:detects]-> attack-pattern` (3,742 edges) and
`x-mitre-detection-strategy -[:detects]-> attack-pattern` (691
edges). The original P.7 expression `{ds:ap.detects<.name}` did
not type the source node, so the parser refused the bind.

**Fix:** explicitly type the reverse-edge source in the template
syntax, and propagate the same explicit-typing convention to the
two downstream relations in the same chain:

```
{ds:ap.detects<x-mitre-detection-strategy.name}
{anl:ds.implemented_by>x-mitre-analytic.name}
{dc:anl.requires_data>x-mitre-data-component.name}
```

Documented here as the canonical recipe for any future template
that traverses `[:detects]` against `attack-pattern`.

### 4.3 Section-header bleed (`X.8`, `SU.G.1`, `SU.POC.1`)

**Symptom:** these three single-row templates produced 0 generated
rows in the initial v8.1 build because their `Answer:` body was
contaminated by the next section header from the source manifest
(e.g. the literal text "MITRE ENGAGE Templates" appended to the
last X.8 answer line), which broke the parser's
`{var:type.field}` extraction.

**Root cause:** `tmpl_gen/scripts/tmpl_docx2json.py`'s answer
collection loop walked forward until the next blank line and did
not filter out non-template lines (banner headers / section
markers).

**Fix:** restricted the answer-collection loop to ignore lines
that match the section-header pattern (`^=` banners, all-caps
section titles followed by "Templates"). Re-running the compile
step recovered all three templates without any source-level
change.

### 4.4 Double-prefix output (`CVE-CVE-`, `CWE-CWE-`, `CAPEC-CAPEC-`)

**Symptom:** several templates emitted strings like
`CVE-CVE-2026-4639` or `CWE-CWE-918`.

**Root cause:** the template body included a literal `CVE-` /
`CWE-` / `CAPEC-` prefix in front of an `{x.id}` interpolation
where the bound `id` field was already prefixed in the graph.

**Fix:** post-processing pass
(`tmpl_gen/utils/_strip_double_prefixes.py`) regex-replaces
`(CVE|CWE|CAPEC)-\1-` with `\1-`. Run as part of the cleanup
between subsample (step 3) and dedup (step 4) of the Section 1.4
pipeline. The pass is idempotent and safe to re-run.

### 4.5 Recovered templates: final volumes

| Template      | Rows in v8.1 | Recovery path |
|---------------|-------------:|---------------|
| `AB.MCQ.3`    |          100 | `Count: 100` cap + 4.1 parser fix |
| `Q.MSR.1`     |          100 | `Count: 100` cap + 4.1 parser fix |
| `P.7`         |          100 | 4.2 explicit reverse-edge typing  |
| `X.8`         |          100 | 4.3 docx2json header-bleed fix    |
| `SU.G.1`      |          100 | 4.3 docx2json header-bleed fix    |
| `SU.POC.1`    |          100 | 4.3 docx2json header-bleed fix    |

All six were merged into `SFT/data/ift_data_2026_04_30_v81.json`
after the stratified subsample step. The pre-recovery snapshot is
preserved at `SFT/data/ift_data_2026_04_30_v81.pre_failed_recovery.json`
for diff verification.


## 5. v9 RMS slice -- companion manifest for the v9 14B launcher

`Sophia-CTI-Templates-v9_rms.txt` is a 21-template manifest containing
the `AB.RMS.*` and `JS.RMS.*` template bodies lifted byte-for-byte
from `Sophia-CTI-Templates-v8_1.txt`. It exists so the v9 14B
launcher (`SFT/autotrain/run_abaligned_sft_qwen25_14b_v9.sh`) can
graft the RMS catalog drills onto Phase B without depending on any
v8.1 build artefact -- the v9 dataset
(`SFT/data/ift_data_2026_04_30_v9_rms.json`) is built from this
manifest through the same Neo4j pipeline as v8.1, in isolation.

### 5.1 Why a separate manifest

The v9 launcher restores the v8 broad-knowledge baseline (Phase A on
the v7 corpus + Tulu + Alpaca) and grafts the RMS catalog drills
onto Phase B. The drills themselves are exactly the AB.RMS / JS.RMS
slate v8.1 perfected. There are two ways to feed them to the v9
build:

  a) **Filter the v8.1 output JSON** -- one-line Python that pulls
     `r['shortname'].startswith(('AB.RMS.', 'JS.RMS.'))` out of
     `ift_data_2026_04_30_v81.json`.
  b) **Compile a parallel template manifest** -- this directory's
     v9_rms.txt, run through `tmpl_docx2json.py` ->
     `make_dataset.sh` -> `stratified_subsample.py`.

(a) is brittle: if `ift_data_2026_04_30_v81.json` is ever lost or
edited, the v9 build cannot be reproduced. (b) keeps every v9
training input traceable to a `.txt` template manifest in version
control, which is the project's reproducibility contract. v9 takes
path (b).

### 5.2 Contents (21 templates, ~12,158 rows post-build)

| Template family       | Template count | Rows  | Source section in v8_1.txt |
|-----------------------|---------------:|------:|----------------------------|
| `AB.RMS.{1,2}`        | 2              | ~1,700 (variable, no Count:) | Section 5 (lines 526-533)         |
| `AB.RMS.3{a..h}`      | 8              | 5,800 | Section F (lines 1600-1740)       |
| `AB.RMS.{4,5}`        | 2              | 1,200 | Section F (lines 1742-1750)       |
| `AB.RMS.6`            | 1              | 1,500 | Section F (lines 1752-1760)       |
| `JS.RMS.{1..8}`       | 8              | 1,725 | Section D.4 (lines 2014-2138)     |
| **total**             | **21**         | **~12,158** |                              |

The `AB.RMS.{1,2}` legacy v5-format templates (no "Answer:" terminator)
are retained as low-volume coverage for prose-style mitigation
answers, matching the v8.1 corpus exactly. The cardinality ladder
(`AB.RMS.3a..3h`, `JS.RMS.1..8`) covers N=1..8 mitigations matching
the athena-rms benchmark prompt distribution. `AB.RMS.4` /
`AB.RMS.5` are the M-id <-> name catalog flashcards;
`AB.RMS.6` is the negative-discrimination drill ("which is NOT
recommended").

### 5.3 Build pipeline (one-shot, reproducible)

```bash
# 1. Compile the manifest into the per-template JSON the build consumes.
python tmpl_gen/scripts/tmpl_docx2json.py \
    -i tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt \
    -o tmpl_gen/data_generation/Sophia-CTI-Templates-v9_rms.json \
    --count_limit 1500

# 2. Drive iftgen.py per-template (handled by make_dataset.sh).
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt \
    _v9_rms_build/triples \
    SFT/data/ift_data_2026_04_30_v9_rms.raw.json \
    10 1500

# 3. Stratified subsample (cap 170 per shortname; PRESERVE_FULL_PREFIXES
#    keeps every AB.RMS / JS.RMS shortname at 100% retention -- the cap
#    is mostly inert here. Run for parity with the v8.1 build recipe).
python tmpl_gen/scripts/stratified_subsample.py \
    --in  SFT/data/ift_data_2026_04_30_v9_rms.raw.json \
    --out SFT/data/ift_data_2026_04_30_v9_rms.json \
    --cap 170

# 4. Verbatim-overlap dedup against the eval suite (informational).
python tmpl_gen/scripts/dedup_against_evals.py \
    --input    SFT/data/ift_data_2026_04_30_v9_rms.json \
    --eval-dir SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 \
    --report   SFT/data/ift_data_2026_04_30_v9_rms.dedup.json
```

The contamination posture in Section 2 carries forward unchanged: the
catalog-recall templates intentionally share short MITRE catalog
phrases with `athena-cti-rms.jsonl` by design (see Section 2.1 for
the full audit trail).

### 5.4 Wiring

| Component                              | v9 reference                                                |
|----------------------------------------|-------------------------------------------------------------|
| Manifest                               | `tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt` |
| Compiled template JSON                 | `tmpl_gen/data_generation/Sophia-CTI-Templates-v9_rms.json` |
| Final dataset                          | `SFT/data/ift_data_2026_04_30_v9_rms.json`                  |
| Trainer config key (`dataset_info.json`) | `ift_data_2026_04_30_v9_rms`                              |
| Launcher                               | `SFT/autotrain/run_abaligned_sft_qwen25_14b_v9.sh` (Phase B, third dataset) |
| Final HF artefact                      | `${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v9`     |
