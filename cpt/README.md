## CPT: Continued Pre-Training on a curated CTI corpus

Pivot away from template-driven SFT toward a pretraining-style update on
real threat-intelligence text. Targets the *base* Llama-3.1-8B
(`meta-llama/Llama-3.1-8B`), not the Instruct variant, on the hypothesis
that the SFT iterations v1-v4 were bottlenecked by the narrowness and
stylistic uniformity of template-generated training data rather than by
SFT itself.

### Layout

```
cpt/
  README.md           this file
  sources.yaml        declarative source registry (URLs, licenses, parsers)
  fetch.py            per-source fetchers -> cache/raw/<source>/
  parse.py            per-format parsers  -> cache/parsed/<source>.jsonl
  process.py          dedupe + benchmark-leak filter + quality gates + stats
  build_corpus.py     end-to-end driver: fetch -> parse -> process -> corpus/
  register_dataset.py append cti_corpus entry to SFT/data/dataset_info.json
  train_cpt.sh        LLaMA-Factory --stage pt launcher
  requirements.txt    extra deps (trafilatura, pymupdf, datasketch, feedparser)
  cache/              [gitignored] raw + parsed intermediates
  corpus/             [gitignored] final JSONL shards for training
```

### Usage (expected)

```bash
# 0. Install extra deps into the existing llm-sft env
conda activate llm-sft
pip install -r cpt/requirements.txt

# 1. Fetch + parse + process all sources enabled in sources.yaml
python cpt/build_corpus.py --out cpt/corpus --name cti_corpus_v1

# 2. Register the built corpus with LLaMA-Factory
python cpt/register_dataset.py --name cti_corpus_v1 \
    --file cpt/corpus/cti_corpus_v1.jsonl

# 3. Launch CPT. Default: base Llama-3.1-8B, LoRA r=32, 1 epoch, 1 H100.
bash cpt/train_cpt.sh --dataset cti_corpus_v1 \
    --repo-id asg-ai/athena-cti-cpt-llama31-8b-v1
```

### Do we need SFT after CPT?

Yes, in a limited form, if we want the AthenaBench/CTIBench suites in
`SFT/test/` to work unchanged. The existing evaluators issue instruction-
style prompts and parse structured output (`Answer: X`, `{...}` dicts,
etc.). A pure CPT on the *base* model will produce fluent CTI prose but
not reliably honor those output formats.

Three endpoints from this pipeline:

| Path                                     | Follow-on                                | Bench compatibility                 |
| ---------------------------------------- | ---------------------------------------- | ----------------------------------- |
| CPT base, no SFT                         | none                                     | Needs ICL prompting; current eval breaks |
| CPT base + tiny chat SFT (~1-2k rows)    | `SFT/autotrain/run_abaligned_sft_v5.sh`  | Works as-is; preferred endpoint     |
| CPT Instruct directly                    | none                                     | Works; risk of eroding Instruct alignment |

The corpus built here is reusable across all three. `train_cpt.sh` flips
between base and Instruct via `--model`.

### Sources (see `sources.yaml` for canonical list)

| Source                          | License     | Parser   | Est. raw text |
| ------------------------------- | ----------- | -------- | ------------: |
| MITRE ATT&CK (enterprise/mobile/ics STIX) | CC-BY       | stix     |       ~3-5 MB |
| MITRE CAPEC                     | CC-BY       | stix/xml |         ~1 MB |
| MITRE CWE                       | public      | xml      |         ~2 MB |
| NVD CVE feeds (2020-2025)       | public      | json     |      ~60-80 MB |
| CISA KEV catalog                | public      | json     |         ~1 MB |
| CISA advisories (AA-/ICSA-)     | public      | html     |      ~20-30 MB |
| The DFIR Report                 | CC-BY-NC-SA | html     |       ~6-8 MB |
| Vendor threat reports (PDFs)    | varies      | pdf      |      ~30-50 MB |
| Vendor blogs (Mandiant/Talos/Unit42/Microsoft TI/Sekoia/Volexity) | varies, scrape-with-attribution | rss+html | ~15-25 MB |
| Sigma rule corpus (SigmaHQ)     | MIT         | yaml     |         ~3 MB |
| MISP OSINT feeds / abuse.ch     | varies      | json/csv |      ~10-20 MB |

**Corpus total target: 150-250 MB raw text, ~35-55M Llama tokens after dedupe.**

### Leak protection

`process.py` runs two filters against the AthenaBench / CTIBench test
splits to prevent contamination:
1. **Exact-id filter**: drop any doc whose CVE/technique/actor id appears
   verbatim as a test-set answer key.
2. **13-gram overlap filter**: minhash near-dup between each doc and the
   test TSVs at a Jaccard threshold of 0.3 (tunable in `sources.yaml`).

Both filters log dropped-doc counts to `cache/leak_report.json` for audit.

### Compute estimates (1x H100 80GB, bf16, flash-attn 2, packed seq_len=4096)

Throughput calibrations are conservative ballparks; tune after the first
real run with `SFT/utils/registry.py` tracking token throughput in the
train log.

| Regime                       | Tok/s    | 50M tokens x 1 epoch | 100M tok x 1 ep |
| ---------------------------- | -------: | -------------------: | --------------: |
| Llama-3.1-8B LoRA r=32, 1xH100  |  ~30,000 |              ~28 min |        ~56 min |
| Llama-3.1-8B LoRA r=32, 2xH100  |  ~55,000 |              ~15 min |        ~30 min |
| Llama-3.1-8B full-param, 2xH100 (ZeRO-2) |  ~22,000 |              ~38 min |        ~76 min |
| Llama-3.1-8B full-param, 4xH100 (ZeRO-2) |  ~42,000 |              ~20 min |        ~40 min |

At ~$2-3/GPU-hour (RunPod/Lambda H100 on-demand, 2026 rates):

| Target regime                     | Est. cost / 50M tok run | Cost for 10-run sweep |
| --------------------------------- | ----------------------: | --------------------: |
| LoRA r=32, 1xH100                 |                 $1-1.50 |                $10-15 |
| LoRA r=32, 2xH100                 |                 $1-1.50 |                $10-15 |
| Full-param, 2xH100 ZeRO-2         |                 $2.50-4 |                $25-40 |
| Full-param, 4xH100 ZeRO-2         |                 $2.50-4 |                $25-40 |

**Dominant cost is curation engineering, not GPU.** The fetch+parse+dedupe
pipeline is the part that takes real human time; a single CPT iteration
is well under $5.

### Hyperparameter starting point (documented in `train_cpt.sh` defaults)

- `--stage pt` (LLaMA-Factory pretraining stage; no chat template applied)
- `--cutoff 4096` with packing on (dense token coverage; CPT wants this)
- `--lr 1e-4` LoRA / `--lr 2e-5` full-param, cosine, warmup 3%
- `--epochs 1` (CPT overfits fast; single pass is the right default)
- LoRA r=32, alpha=64, dropout=0.05, target=all
- `--bf16`, flash-attn 2, `--packing True`

The anti-collapse lever from the v1-v4 post-mortem: **one epoch, not
three**. Fine-tune hyperparameters once the first CPT run lands and we
have a reference throughput + eval number.
