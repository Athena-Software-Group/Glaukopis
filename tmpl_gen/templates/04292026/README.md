# Sophia CTI Templates -- v8 (April 29, 2026)

This directory consolidates the v8 SFT template manifests for both the
`v8-small` (Llama-3.1-8B-Instruct) and `v8-large` (Qwen2.5-32B-Instruct)
training runs, plus the JSON-output addendum that both variants share.

```
04292026/
  Sophia-CTI-Templates-v8_small.txt    self-contained v8-small manifest
  Sophia-CTI-Templates-v8_large.txt    self-contained v8-large manifest
  Sophia-CTI-Templates-JSON-v8.txt     v8 JSON-output template slate (inlined
                                       into the two consolidated files above
                                       and still referenced as a standalone
                                       file by the build pipeline)
  README.md                            this document
```

The two `v8_{small,large}` files are **documentation-only** as of the v8
release: the live build pipeline (`tmpl_gen/scripts/iftgen.py` driven by
`tmpl_gen/scripts/tmpl_docx2json.py`) still consumes the per-vintage
source files (`tmpl_gen/templates/04262026/Sophia-CTI-Templates-Combined-v7.txt`
and `tmpl_gen/templates/04292026/Sophia-CTI-Templates-JSON-v8.txt`).
A future v9 cycle that wires `--templates-file` to a single consolidated
input will switch the build to read from these consolidated files; until
then they exist to give reviewers a single self-contained artefact per
training variant.

## 1. Training strategy summary

### 1.1 v8-small (Llama-3.1-8B-Instruct, single-pass full SFT)

| Knob              | Value                                                 |
|-------------------|--------------------------------------------------------|
| Trainer           | `SFT/autotrain/run_abaligned_sft_llama31_8b_v8.sh`    |
| Base model        | `meta-llama/Llama-3.1-8B-Instruct`                    |
| Final HF artefact | `${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v8` |
| Recipe            | 2 epochs, cosine, 5% warmup, bf16, lr=1e-5            |
| Cutoff            | 8192 tokens, packing on                               |
| Effective batch   | 16                                                     |
| Parallelism       | DeepSpeed ZeRO-3 (no offload on >=2 x 80 GB GPUs)     |
| Visible per epoch | ~60,000 rows (`--max-samples 60000`)                  |

Corpus (combined Alpaca file + two HF mixtures resolved at load):

| Source                                              |    Rows | Selection                              |
|-----------------------------------------------------|--------:|----------------------------------------|
| `ift_data_2026_04_26_combined_v7`                   |  41,857 | `:250:strat` (cap per shortname)       |
| `ift_data_2026_04_29_json_v8`                       |   8,700 | full (no subsample)                    |
| `ift_data_2026_04_29_longctx_v8`                    |   2,000 | `:2000:random`                         |
| `ift_data_2026_04_29_combined_v8small.json`         |  52,557 | concatenation of the three above       |
| `tulu_3_sft_mixture` (HF, AllenAI)                  |  ~5,000 | random subsample (CF guard)            |
| `alpaca_en_demo`                                    |  ~2,500 | random subsample (CF guard)            |
| **total per epoch (capped)**                        | **~60,000** |                                    |

The Tulu and Alpaca additions are the catastrophic-forgetting guard:
prior CTI-only 8B runs collapsed general instruction-following enough
to noticeably regress CyberMetric (which is partly a general-knowledge
benchmark with a cyber framing).

### 1.2 v8-large (Qwen2.5-32B-Instruct, two-phase full SFT)

| Knob              | Value                                                 |
|-------------------|--------------------------------------------------------|
| Trainer           | `SFT/autotrain/run_abaligned_sft_qwen25_32b_v8.sh`    |
| Base model        | `Qwen/Qwen2.5-32B-Instruct`                           |
| Final HF artefact | `${HF_USERNAME}/athena-cti-sft-qwen25-32b-abaligned-v8` |
| Phase chaining    | Phase B's `--model` points at Phase A's output dir    |

**Phase A -- "broad knowledge re-anchor"**

| Datasets | `ift_data_2026_04_26_combined_v7`, `tulu_3_sft_mixture`, `alpaca_en_demo` |
|---|---|
| Cap | `--max-samples 250000` (1 epoch) |
| Cutoff | 4096, packing on, lr=1e-5, effective batch 16 |

**Phase B -- "format and long-context specialization"**

| Datasets | `ift_data_2026_04_29_json_v8`, `ift_data_2026_04_29_longctx_v8` |
|---|---|
| Cap | `--max-samples 50000` (1 epoch; corpus is ~12.7K rows so cap is a bound, not active) |
| Cutoff | 16384, **packing OFF**, lr=5e-6, effective batch 8 |

Packing is off in Phase B because long stitched reports must not be
packed across boundaries: cross-contamination between the end of one
report and the start of another would teach the model to ignore the
report-boundary scaffolding documented in
`Sophia-CTI-Templates-v8_*.txt` Section 3.

The two phases differ structurally because the v7 broad corpus is
~250K mostly-short Q-A pairs that derive no benefit from a 16K context
window or unpacked sequences; running them both at Phase B settings
would 4x-8x the total compute for negligible signal gain.

## 2. Contamination posture

This section is the audit trail for the contamination question
("are the v8 SFT corpora leaking the AthenaBench / CyberMetric /
CyberSOCEval / CTIBench evaluation signal into training?").
We separate the two failure modes the research community actually
distinguishes:

  * **Verbatim contamination** -- a literal eval prompt, eval answer
    string, or eval-row n-gram appearing in a training row. This is
    the failure mode that matters for benchmark validity, because the
    model can solve the eval row by memorisation.
  * **Structural contamination** -- the training data and the eval
    data sharing the same underlying knowledge base (the MITRE STIX
    bundles for ATT&CK / CAPEC / CWE, the NVD CVE feed, the CISA KEV
    catalog, the FIRST EPSS feed). The model trained on the graph
    can answer a benchmark item drawn from the same graph because
    the underlying fact is identical, not because the eval row was
    memorised.

The v8 program treats these two failure modes asymmetrically: verbatim
contamination is **blocked by tooling at corpus build time**;
structural contamination is **accepted by design**, with the rationale
documented per benchmark below. The remainder of this section makes
both positions explicit.

### 2.1 Verbatim contamination guard: `dedup_against_evals.py`

**Tool:** `tmpl_gen/scripts/dedup_against_evals.py`
**Reference:** `SFT/autotrain/run_abaligned_sft_qwen25_32b_v8.sh`
lines 85-95 require this script to have run cleanly against
`ift_data_2026_04_29_json_v8.json` and
`ift_data_2026_04_29_longctx_v8.json` before Phase B can be launched
(the launcher fails with `[FAIL] Phase B dataset missing` if the
file is absent, and the project convention is that the file is only
written to `SFT/data/` after passing dedup; see also the parallel
14B launcher `SFT/autotrain/run_abaligned_sft_qwen25_14b_v8.sh`
which embeds the same gate).

**Mechanism:**

  1. Tokenise every eval record's user-visible text (`question`,
     `prompt`, `report`, `context`, `input`, `instruction`, plus
     `answers`) into lowercase `[A-Za-z0-9]+` word tokens.
  2. Build the union set of distinct **n=13 word-grams** (default,
     overridable via `--n`) across every eval file under
     `--eval-dir SFT/test/benchmark_data/` -- which currently covers:
       * `athena_bench/` (`athena-cti-{ate,mcq,mcq-3k,mcq-updated,
         rcm,rms,vsp}.jsonl`, `athena_rms/`, `athena_taa/`,
         `mcq-patch.tsv`)
       * `cti_bench/` (`cti-{ate,mcq,rcm,rcm-2021,vsp}.tsv`,
         `cti_taa/`)
       * `cybermetricdataset/` (`CyberMetric-{80,500,2000,10000}-v1.json`)
       * `cve/`, `urlhaus/` (operational data sources)
  3. Index every n-gram to its source `eval_file:row_idx`.
  4. For every candidate Alpaca SFT row, tokenise the concatenation
     of its `instruction`, `input`, `output`; emit any n-gram that
     appears in the eval index; flag the row when its hit count
     against any **single** eval row reaches `--hit-threshold`
     (default 1).
  5. **Exit 1** when the flagged-row count exceeds `--max-fail`
     (default 0), which fails the build before training can start.

**Threshold rationale.** n=13 word tokens is the same window used by
the OLMo, Pythia, and Llama families' MMLU / HellaSwag / BIG-bench
decontamination passes: short enough to catch verbatim leakage while
long enough that incidental matches on common stock phrases (e.g.
"you are a cybersecurity expert that has been trained") do not
trigger. `hit-threshold=1` is the strict setting; one shared 13-gram
between an SFT row and an eval row is, in practice, never incidental
for technical CTI text and almost always indicates that the eval
row's question or answer string was inadvertently included in the
training corpus.

**What this catches:** any SFT row that quotes an eval row's
question, answer, multiple-choice option text, threat-report
excerpt, or explanatory paragraph at >=13 contiguous tokens of
overlap. This is the failure mode that motivated the script's
introduction in v8: when CyberSOCEval-shape JSON templates were
first authored it was easy to accidentally crib question phrasing
from an open-source example in the CyberSOCEval repo, and
`dedup_against_evals.py` would flag the row with the matching
`cybersoceval/*.jsonl:row_idx` so the template could be reworded
before the corpus shipped.

**What this does not catch:** semantically equivalent paraphrases
(e.g. an SFT row asking "Which CWE underlies CVE-2024-NNNN?" when
an eval row asks "What is the root-cause weakness for
CVE-2024-NNNN?"). Catching paraphrastic leakage requires
embedding-similarity dedup, which is not currently in the v8
pipeline; we address paraphrase risk structurally instead -- see
Section 2.3 -- and rely on the fact that the templates are built
from the underlying knowledge graph rather than from the eval text,
so paraphrastic overlap is only possible when both the SFT row and
the eval row independently describe the same graph fact (which is
the structural-contamination case addressed in Section 2.2).

### 2.2 Structural contamination: accepted by design

The MITRE ATT&CK STIX bundle, the CAPEC catalog, the CWE corpus,
the NVD CVE feed, the CISA KEV catalog, the FIRST EPSS feed, and
the MITRE ENGAGE matrix are all **public, canonical knowledge
bases**. The AthenaBench, CTIBench, CyberMetric, and CyberSOCEval
evaluation suites all draw their ground-truth answers from these
same bundles because they are the only authoritative sources of CTI
fact (there is no "held-out ATT&CK matrix"). Any model that has
seen these bundles in any form -- including every base model with a
2023+ pretrain cutoff -- has a structural overlap with these
benchmarks.

This is the established posture in the published CTI-LLM literature
(SecKnowledge / CyberPal.AI, Levi et al. 2024
arXiv:2408.09304; the CTIBench paper itself; the AthenaBench
technical report) and it is the position v8 takes explicitly:

  * The v8 SFT corpus is **built from the same MITRE / CISA / FIRST
    bundles** that the benchmarks are scored against. There is no
    eval/train split on the underlying graph because there cannot
    be one without crippling the SFT signal -- the v6 README
    excerpt in
    `tmpl_gen/templates/04252026/Sophia-CTI-Templates-AthenaBench-abaligned-v6.txt`
    (Section 3) makes this argument concretely for `athena-rms`:
    the benchmark samples 500 of ~691 techniques-with-mitigations
    from the same STIX bundle that `tmpl_gen` reads via Neo4j;
    holding out 72% of MITRE's catalog would defeat the purpose of
    training a CTI model.
  * We therefore frame these benchmarks as **catalog-recall and
    reasoning evaluations**, not as held-out generalisation
    evaluations: the question being measured is "given that the
    model has seen the underlying graph, can it retrieve the right
    fact and reason over it under the eval prompt's surface form",
    not "has the model generalised to a held-out subset of the
    knowledge base it has never seen".
  * The catalog-recall framing is what makes the v8-specific
    interventions sensible: AB.RMS.4/5 flashcard templates exist
    precisely because plain `(technique -> mitigation)` exposure in
    the catalog was insufficient repetition for the 8B parameter
    count to lock in the M-IDs (the v5 baseline hallucinated M1037
    descriptions; AB.RMS.4 drills the catalog ID->name binding to
    fix that). The fix is a training intervention against catalog
    recall, not a leak.

What is **not** acceptable under this framing -- and what the
verbatim contamination guard in Section 2.1 actively blocks -- is
the specific eval-row prompt or answer string from any of the four
eval suites appearing verbatim in any training row. The eval rows'
GPT-{4,5}-rewritten incident narratives, AthenaBench's specific
sampling of 500 of the 691 techniques-with-mitigations,
CyberMetric's specific MCQ phrasings, and CyberSOCEval's specific
JSON envelopes are all surface artefacts of those benchmarks and
have no place in the SFT corpus.

### 2.3 Per-benchmark contamination matrix

| Benchmark | Held-out from training? | What v8 SFT shares with it | Verbatim guard | Structural notes |
|---|---|---|---|---|
| **AthenaBench (`athena-rcm`, `athena-vsp`, `athena-ate`, `athena-taa`, `athena-rms`, `athena-mcq`)** | No -- structural overlap accepted (catalog-recall framing). | The underlying ATT&CK / CWE / CVE / KEV / EPSS graph. The benchmark prompts are GPT-{4,5}-rewritten incident narratives produced at benchmark generation time and are NOT in the SFT corpus. | `dedup_against_evals.py` indexes `athena_bench/*.jsonl` and the `athena_rms/`, `athena_taa/` subdirs at n=13. Any 13-gram overlap fails the build. | The catalog-recall framing is the published AthenaBench position. v8's RMS interventions (AB.RMS.{3a..3h,4,5,6}) target the catalog-coverage and cardinality gaps that the v5 baseline exposed; they do not encode any benchmark prompt. |
| **CTIBench (`cti-mcq`, `cti-rcm`, `cti-rcm-2021`, `cti-vsp`, `cti-ate`, `cti_taa`)** | No -- structural overlap accepted (same graph). | Same MITRE/NIST graph as the SFT corpus. CTIBench prompts are paraphrastic transformations of the underlying records. | Same n=13 fingerprint as above; `cti_bench/*.tsv` is in the index. | The CTIBench paper explicitly anticipates this overlap and grades models on the paraphrastic surface form. v8 templates do not import CTIBench-specific phrasings. |
| **CyberMetric (`CyberMetric-{80,500,2000,10000}-v1.json`)** | No -- structural overlap with general CTI knowledge accepted; the benchmark is partly a general-knowledge MCQ over public security material that overlaps the base model's pretrain. | Domain knowledge (concepts, definitions, CVE descriptions). The MCQ phrasings themselves are NOT in the SFT corpus. | All four CyberMetric files are in the n=13 index; verbatim leakage of any MCQ stem or option string fails the build. | The Tulu/Alpaca catastrophic-forgetting guards (Section 1) are part of the CyberMetric posture: they preserve general instruction-following so that CyberMetric scores reflect CTI-narrowing impact rather than instruction collapse, not so that the model can solve specific CyberMetric items. |
| **CyberSOCEval (Meta, malware-TI / threat-investigation JSON tasks)** | No -- structural overlap with the public CTI corpus accepted. | The JSON envelope shapes (e.g. `{"correct_answers": [...]}`, `{"behaviors": [...]}`) are matched in the `JS.*` template family. The specific eval prompts and ground-truth answers are NOT in the SFT corpus. | The CyberSOCEval source files (when available locally under `SFT/test/benchmark_data/`) are picked up by the same n=13 glob and any overlap fails the build. | The `JS.*` family is the v8 response to the v7 format-collapse failure mode (see `Sophia-CTI-Templates-JSON-v8.txt` Section A). It teaches the **shape** of a JSON-wrapped response over the SFT corpus's own graph-derived facts; CyberSOCEval grades the same shape over its held-out facts. |

### 2.4 What v8 explicitly does **not** do

To make the v8 contamination posture falsifiable, the following is the
list of practices we deliberately avoid and why:

  * **No eval-row text in templates.** No template's `Instruction:`,
    `Question:`, or `Answer:` body contains a verbatim phrase from any
    eval row. The n=13 guard fails the build if a future template ever
    introduces one.
  * **No benchmark answer keys in the graph.** The Neo4j build
    (`tmpl_gen/scripts/iftgen.py` -> `create_ATTACK_db`) ingests the
    upstream MITRE / NIST / FIRST / CISA bundles only. We do not load
    AthenaBench's GPT-rewritten narratives, CyberSOCEval's question
    JSON, CTIBench's paraphrases, or CyberMetric's MCQs into the
    graph. The graph contains only the canonical knowledge bases that
    the benchmarks themselves draw from.
  * **No fine-tuning on benchmark dev/validation splits.** AthenaBench
    and CTIBench publish small validation slices intended for prompt
    engineering. v8 does not consume any of these splits as training
    data; they are only used downstream by the eval harness in
    `SFT/test/`.
  * **No cross-pollination from the eval harness.** The eval harness
    in `SFT/test/pipelines/` reads from `SFT/test/benchmark_data/`
    only and never writes back into `SFT/data/`. This is enforced by
    the directory layout: training datasets live in `SFT/data/`,
    benchmarks live in `SFT/test/benchmark_data/`, and the build
    pipeline scripts in `tmpl_gen/scripts/` write only to the former.
  * **No per-token contamination check on `tulu_3_sft_mixture` or
    `alpaca_en_demo`.** These two HF mixtures are general
    instruction-following data and are unlikely to overlap CTI
    benchmarks at n=13; we do not run `dedup_against_evals.py` on
    them as a matter of routine. If a future eval suite is added that
    overlaps general-purpose instruction data, we would extend the
    guard; for the current v8 eval portfolio it is not a live risk.

### 2.5 Reproducing the contamination check

```bash
# Run from the repo root with the tmpl_gen virtualenv active.
python tmpl_gen/scripts/dedup_against_evals.py \
    --input    SFT/data/ift_data_2026_04_29_json_v8.json \
    --eval-dir SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 \
    --report   SFT/data/ift_data_2026_04_29_json_v8.dedup.json

python tmpl_gen/scripts/dedup_against_evals.py \
    --input    SFT/data/ift_data_2026_04_29_longctx_v8.json \
    --eval-dir SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 \
    --report   SFT/data/ift_data_2026_04_29_longctx_v8.dedup.json

# v8-small uses the same two source files (subsampled), so the
# combined v8-small file is dedup-clean by transitivity. To verify:
python tmpl_gen/scripts/dedup_against_evals.py \
    --input    SFT/data/ift_data_2026_04_29_combined_v8small.json \
    --eval-dir SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 \
    --report   SFT/data/ift_data_2026_04_29_combined_v8small.dedup.json
```

A clean run prints `scanned N corpus rows -> 0 flagged` and exits 0.
Any non-zero flag count writes the offending rows into the `--report`
JSON file and exits 1; the build is expected to halt and the
offending template(s) reworded before retrying.

### 2.6 Related: continued-pretraining (`cpt/`) leak protection

The continued-pretraining corpus under `cpt/` (a separate program
from the SFT corpus documented here) uses a stronger leak filter
(`cpt/process.py` `leak_filter`) because CPT documents are
free-form web text rather than graph-derived templates and so the
risk surface is larger. CPT runs MinHashLSH at Jaccard 0.30 over
13-grams against the same `SFT/test/benchmark_data/` corpora plus
optional per-source exact-CVE-id dropping (opt-in via
`drop_on_exact_id` in `cpt/sources.yaml`, default off for structural
taxonomies, on for NVD where the CVE record embeds the literal
CVSS-vector answer to CTIBench-VSP). The SFT pipeline does not
need MinHashLSH because templated rows are short and the n=13
exact-substring match at threshold 1 is already strict enough.

## 3. File provenance for the consolidated manifests

Each consolidated `Sophia-CTI-Templates-v8_{small,large}.txt` file
in this directory is the verbatim concatenation of:

  1. A variant-specific strategy header (Section 0 of the consolidated
     file).
  2. `tmpl_gen/templates/04262026/Sophia-CTI-Templates-Combined-v7.txt`
     (Section 1 of the consolidated file).
  3. `tmpl_gen/templates/04292026/Sophia-CTI-Templates-JSON-v8.txt`
     (Section 2 of the consolidated file).
  4. A long-context scaffolding appendix documenting
     `tmpl_gen/scripts/stitch_long_context.py` (Section 3 of the
     consolidated file).

The two consolidated files differ only in their Section 0 header
(single-pass vs two-phase strategy) and the per-section banners that
note Phase A/B sourcing for the v8-large variant. Sections 1, 2, and
3 are identical between the two files.
