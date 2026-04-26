---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- cybersecurity
- cyber-threat-intelligence
- cti
- mitre-attack
- llama-3.1
- sft
- athenabench
---

# athena-cti-sft-llama31-8b-abaligned-v7

Full-parameter SFT of `meta-llama/Llama-3.1-8B-Instruct` aligned to the
[AthenaBench](https://github.com/Athena-Software-Group/Glaukopis) Cyber
Threat Intelligence (CTI) benchmark suite. v7 is the first checkpoint
in the `athena-cti-sft-llama31-8b-abaligned-*` family that lands above
the v0 base-model baseline on **every** athena task and recovers the
Risk Mitigation Strategy (RMS) task from the v6 regression.

## Intended use

Knowledge-graph-grounded CTI question answering across MITRE ATT&CK,
CAPEC, CWE, CVE, CISA KEV, FIRST EPSS, and MITRE ENGAGE. The training
mix is task-aligned to the six AthenaBench subtasks:

| Task | What it asks | Output format |
|---|---|---|
| `athena-mcq` | 5-way multiple choice over CTI facts | single letter `A..E` |
| `athena-rcm` | binary relevance / classification | `Yes` / `No` |
| `athena-vsp` | CVSS v3 base-score prediction | numeric severity |
| `athena-ate` | ATT&CK technique extraction from prose | T-IDs |
| `athena-taa` | threat-actor attribution | actor name + plausibility |
| `athena-rms` | Risk Mitigation Strategy: pick *N* MITRE Mitigation IDs that best mitigate a given attack pattern | `Answer: M####, M####, ...` |

Out-of-scope: open-ended generation, code synthesis, non-CTI domains.

## Validated AthenaBench results (suite=athena, version=1)

Single H100 via vLLM, `--tp 1 --max-len 4096 --batch 128`, end-to-end
1m51s for the full 8,100-row sweep.

| Task | Rows | Metric | v7 | v0 baseline | v6 |
|---|---:|---|---:|---:|---:|
| `athena-mcq` | 3000 | accuracy | **57.60 %** | ~50 % | ~50 % |
| `athena-rcm` | 2000 | accuracy | **65.80 %** | ~55 % | ~60 % |
| `athena-vsp` | 2000 | accuracy (MAD 1.92) | **75.02 %** | ~70 % | ~70 % |
| `athena-ate` | 500 | accuracy | **50.00 %** | ~45 % | ~45 % |
| `athena-taa` | 100 | combined acc (strict 17.0 % / plausible 82.0 %) | **49.50 %** | low double-digits strict | low double-digits strict |
| `athena-rms` | 500 | strict F1 (plausible 64.32 %) | **62.64 %** | **5.88 %** | **0.00 %** |

The RMS recovery (`+56.76 pp` strict F1 over the v0 baseline,
`+62.64 pp` over the v6 regression) is the headline result.

## Training recipe

| Setting | Value |
|---|---|
| Method | full-parameter SFT (DeepSpeed ZeRO-3, no LoRA) |
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Dataset | `ift_data_2026_04_26_combined_v7` (~181k rows: v5 broad CTI coverage + v7 RMS addendum) + `alpaca_en_demo` mix-in |
| Epochs | 3 |
| Learning rate | 1e-5 cosine, 5 % warmup |
| Precision | bf16 |
| `cutoff_len` | 4096 |
| Effective batch size | 16 |
| Packing | on |
| Chat template | `llama3` |
| Optimizer | AdamW (LlamaFactory defaults) |
| Trainer | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) + DeepSpeed ZeRO-3 |
| Launcher | [`SFT/autotrain/run_abaligned_sft_v7.sh`](https://github.com/Athena-Software-Group/Glaukopis/blob/main/SFT/autotrain/run_abaligned_sft_v7.sh) |

## What changed vs v6

v6 collapsed `athena-rms` to 0.00 % strict F1. The v7 template slate
and launcher fix three structural bugs identified in the v6 post-mortem:

1. **Output truncation** — v6 inlined the full mitigation description
   per item and clipped at `cutoff_len=2048`. ~80 % of RMS rows were
   right-truncated mid-explanation, so the trained model never learned
   the terminal `Answer:` line. v7 doubles the cutoff to 4096 and
   shortens per-mitigation clauses to `{mitre_id} ({name})`.
2. **Missing `Answer:` terminator** — the AthenaBench RMS post-processor
   expects a final line `Answer: M####, M####, ...`. v6 ended with
   "Therefore, the recommended ..." and emitted zero compliant
   responses. Every variable-N v7 template ends with the literal
   directive.
3. **Cardinality coverage gap** — v6 trained on N=3..5 only; the
   benchmark distribution peaks at N=1 (39 %) and N=2 (24 %). v7 adds
   `RMS.3a..3h` covering N=1..8 to match the benchmark mass.

## Reproducing

```bash
# Inference (vLLM, single H100, ~16 GB weights + ~60 GB KV cache):
vllm serve asg-ai/athena-cti-sft-llama31-8b-abaligned-v7 \
    --tensor-parallel-size 1 --max-model-len 4096 --port 8000

# AthenaBench sweep (from a Glaukopis checkout):
cd SFT/test
BENCH_CONDA_ENV=ctibench bash utils/serve_and_bench.sh \
    athena-cti-sft-llama31-8b-abaligned-v7-vllm \
    --tp 1 --max-len 4096 --port 8000 \
    -- --suite athena --version 1 --batch 128 --overwrite --yes
```

Training reproduction: see
[`SFT/autotrain/README.md` § *v7 recipe and results*](https://github.com/Athena-Software-Group/Glaukopis/blob/main/SFT/autotrain/README.md#v7-recipe-and-results).

## Limitations

- **CTI scope only.** Performance outside MITRE ATT&CK / CAPEC / CWE /
  CVE / KEV / EPSS / ENGAGE territory is not measured and likely
  degraded vs the base Llama-3.1-8B-Instruct.
- **Hallucinated identifiers.** The model may emit syntactically
  plausible but non-existent MITRE IDs (T####, M####, CWE-####). Always
  validate downstream.
- **Single-language.** English only.
- **License inheritance.** Subject to the [Llama 3.1 Community License
  Agreement](https://www.llama.com/llama3_1/license/) of the base
  model.

## Citation

If you use this model, please cite the Glaukopis project:

```bibtex
@software{glaukopis_athena_cti_sft_v7_2026,
  title  = {athena-cti-sft-llama31-8b-abaligned-v7},
  author = {Athena Software Group},
  year   = {2026},
  url    = {https://huggingface.co/asg-ai/athena-cti-sft-llama31-8b-abaligned-v7}
}
```
