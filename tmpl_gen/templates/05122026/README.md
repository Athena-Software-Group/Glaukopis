# Sophia CTI Templates — v17.1 (May 12, 2026 vintage)

v17.1 is the **data-fix recovery** of v17. It mirrors the v17 chained
narrow-SFT recipe byte-for-byte and changes only the corpus. The branch
produces one HF checkpoint:

| checkpoint | answers the question |
|---|---|
| `asg-ai/athena-cti-sft-qwen25-14b-v17-1` | When the v17 corpus defects are fixed, does the chained-SFT recipe lift CyberSOCEval-TI and CyberSOCEval-Malware accuracy without regressing the v16 TAA-attribution head — or was the v17 regression architectural (chained SFT competes with the v16 head) rather than data-driven? |

The vintage directory is self-contained per project convention:

```
05122026/
  Sophia-CTI-Templates-v17.1.txt   self-contained CSE-shape manifest;
                                   body byte-identical to v17 except every
                                   template now declares `Shuffle: mcq_multi`
  v17_1_plan.txt                   master plan document (RCA in §1.2/§1.3a;
                                   pipeline in §2; recipe in §3; falsification
                                   criteria in §4; sign-off in §5)
  v17_1_row_count_gate.json        per-axis REJECT_IF_BELOW thresholds
                                   (carried verbatim from v17)
  README.md                        this document (root-cause analysis in §1;
                                   the two engine fixes in §2; run-book in §3)
```

## 1. Why v17.1 exists — root-cause analysis of v17

The post-bench result on v17 was **Outcome D** per `v17_plan.txt §4`: the
chained SFT regressed every measured Athena axis simultaneously and did not
lift CSE-Malware or CSE-TI. Forensic analysis of the v17 training corpus
(`SFT/data/ift_data_2026_05_11_v17_cse.json`, 16,548 rows) revealed **two
distinct corpus-generation defects** — neither of which is an architectural
property of the chained-SFT recipe, both of which v17.1 isolates and fixes.

### 1.1 Bug A — engine: missing multi-select shuffler

The v17 manifest hard-codes the `correct_answers` letter list into the
Answer block of every JS.CSE.* template (e.g. `"correct_answers": ["A","B"]`)
and declares no `Shuffle:` directive. The engine had a `Shuffle: mcq` path
for single-letter MCQs but no equivalent for the multi-select letter-set
shape. **Fix:** added `_shuffle_mcq_options_multi` to
`tmpl_gen/src/tmpl_gen/tmpl_parser.py` and wired it to a new
`Shuffle: mcq_multi` directive that permutes the option block (A–H), parses
the JSON-letter-set answer marker (wrapped or bare), and rewrites the
`correct_answers` list to the shuffled positions.

### 1.2 Bug B — parser: dropped multi-paragraph question bodies (the actual root cause)

`tmpl_gen/scripts/tmpl_docx2json.py` truncated every JS.CSE.* template
body at the blank line that follows the `<desc>` wrapper. Its question-
collection loop appended only consecutive `[A-Z])` option lines after the
`Question:` header and broke out on any other line shape, so the rendered
v17 triples were **`Instruction:` + `Question:` body truncated at `</desc>`
+ `Answer:` `<hard-coded letter list>`**, with the "Based on the
intelligence above…" prose paragraph and ALL the `A) … E)` option lines
silently dropped. From the optimizer's perspective the input distribution
was essentially noise relative to the output: the model could satisfy the
training loss almost entirely by learning **"emit `A,B` regardless of
input"**. **Fix:** rewrote the question-collection loop to mirror the
answer-collection semantics already in the file — continue collecting
subsequent lines as part of the body until `Answer:`, a `{force …}`
constraint, a `Sample/Shuffle/Count/...` sentinel, or the next template
ID. v12 and v16 templates are unaffected; their `Question:` bodies are
single-line and pass through the new loop identically.

Bug B is the **single-line root cause** of the v17 regression; Bug A is
necessary-but-not-sufficient (a correctly-firing shuffler still has nothing
to shuffle if the option lines were never in the rendered prompt).

## 2. Build-time gate added in v17.1

A new Phase 6c gate (`_v17_1_build/letter_balance_gate.py`) catches the
mode-collapse pattern at the corpus level before training spend. It
rejects the build if any of A–E carries <8% or >32% of correct slots,
OR any single combo carries ≥15% of rows, OR <20 distinct combos appear.
The 20-combo floor reflects the manifest's combinatorial ceiling of 26
(= 1 zero-correct + C(5,1) + C(5,2) + C(5,3)). Run on the v17 corpus this
gate would have hard-failed at "letter E carries 0.0% of correct slots,
combo `[A,B]` carries 50.7% of rows, only 4 distinct combos".

## 3. Run-book

### 3.1 Generation

```bash
cd /Users/pietro/code/Glaukopis     # or the cluster equivalent
# Pre-flight: confirm Neo4j athena-cti-db is reachable and populated
python tmpl_gen/data_generation/neo4j_check.py
# Expected: intrusion-sets >= 180; malware >= 600; attack-patterns >= 800

mkdir -p _v17_1_build
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05122026/Sophia-CTI-Templates-v17.1.txt \
     _v17_1_build/triples \
     SFT/data/ift_data_2026_05_12_v17_1.raw.json \
     2500 3500 > _v17_1_build/build.log 2>&1 &
echo "PID=$!" > _v17_1_build/build.pid

nohup bash _v17_1_build/watcher.sh > _v17_1_build/watcher.log 2>&1 &
```

The watcher (in the local-only `_v17_1_build/`) polls the build PID, then
runs Phases 4–8 (actor-balance no-op, dedup, row-count gate against
`v17_1_row_count_gate.json`, licence gate, **letter-balance gate**,
stratified shuffle, val/train split). Final outputs:

- `SFT/data/ift_data_2026_05_12_v17_1_cse.json`  (training shard, ~18.8K rows)
- `SFT/data/ift_data_2026_05_12_v17_1_val.json`  (held-out validation slice, 350 rows = 50 × 7 axes)
- `_v17_1_build/watcher_status.json`             (per-phase row counts and reports)

### 3.2 Dataset registration

The launcher resolves dataset names through `SFT/data/dataset_info.json`.
The v17.1 entries (`ift_data_2026_05_12_v17_1_cse`,
`ift_data_2026_05_12_v17_1_val`) are registered alongside the v17 / v16 /
v15 / v12 entries.

### 3.3 Training

```bash
bash SFT/autotrain/run_sft_qwen25_14b_v16_plus_v17_1_cse.sh
# defaults to ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v17-1
# base model: asg-ai/athena-cti-sft-qwen25-14b-v16   (CHAINED off v16; same as v17)
# wall-time ~4-6 h on 8xH100
```

### 3.4 Bench

Standard 14B AthenaBench sweep against the v17.1 HF checkpoint; compare
against **both** v16 (the chaining baseline) and v17 (the broken-corpus
chained run). Decision matrix in `v17_1_plan.txt §4` (Outcomes A/B/C/D
overlay). If v17.1 lifts CSE-Malware and CSE-TI without TAA loss, the
v17 regression was the corpus and chained SFT is validated. If v17.1
reproduces the v17 regression, the problem is architectural and a v18
mergekit alpha sweep or parallel-branch architecture is indicated.
