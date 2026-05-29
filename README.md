# Glaukopis

**Knowledge-Grounded Supervised Fine-Tuning for Cyber Threat Intelligence.**

Glaukopis is the **training pillar** of the Athena Labs CTI stack — a reproducible, instrumented SFT pipeline that turns a structured CTI knowledge graph into a domain-aligned Small Language Model (SLM) deployable behind a customer's perimeter. Within the larger Athena Security Group toolchain:

```
Ariadne   →   Sophia       →   Glaukopis   →   Minerva   →   Pallas / Promachos
(graph)       (templates)      (SFT)           (RLVR)        (production)
```

Glaukopis sits inside a three-pillar architecture engineered to deliver accurate, structured, verifiable CTI outputs into production SOCs:

| Pillar | Component | Function |
|---|---|---|
| **Measure** | **AthenaBench** | Six-task dynamic CTI benchmark (CKT/MCQ, RCM, ATE, TAA, RMS, VSP) with live API connectors to MITRE ATT&CK / NVD / CWE / CAPEC / EPSS. Ground truth stays *current*, not stale. |
| **Train** | **Glaukopis (this repo)** | Knowledge-graph-driven Instruction Fine-Tuning. Converts the Ariadne CTI graph + curated OSINT corpora into structured supervised examples via the Sophia template engine, then runs LlamaFactory-based SFT. |
| **Verify** | **Minerva** | Reinforcement Learning with Verifiable Rewards. Programmatic verifiers (T-code exists? JSON valid? CWE↔CVE mapping holds?) replace opaque preference models. |

This repository covers the **Measure** and **Train** loop end-to-end:

1. Build the CTI knowledge graph from public threat-intelligence sources (`athena_cti_db`).
2. Generate IFT data from the graph via the Sophia template engine (`tmpl_gen`).
3. Supervised fine-tune base LLMs (Llama-3.1-8B, Qwen2.5-14B / 32B) on the IFT data via LlamaFactory (`SFT`).
4. Benchmark the resulting models on AthenaBench + general-reasoning suites (`SFT/eval`).

---

## Why a domain-trained CTI SLM?

The frontier-API tax is structurally incompatible with production SOC economics, and frontier models — trained as generalists — guess at MITRE T-codes, hallucinate CVE → CWE mappings, and emit free-text where STIX / JSON is required.

### Cost (per single AthenaBench sweep, ~6.6K rows × ~3K avg output tokens ≈ ~20M output tokens)

| Model | Input $/1M | Output $/1M | Est. cost / sweep | Vs. self-hosted |
|---|---:|---:|---:|---:|
| GPT-5.5-Pro | 30.00 | 180.00 | **~$3,900** | **~1,950×** |
| GPT-5.5 | 5.00 | 30.00 | ~$650 | ~325× |
| Gemini-3.1-Pro | 2.00 | 12.00 | ~$260 | ~130× |
| GPT-5.2 (high reasoning) | 1.75 | 14.00 | ~$300 | ~150× |
| DeepSeek-V4-Pro (HF) | 1.74 | 3.48 | ~$87 | ~43× |
| Gemini-3-Flash | 0.50 | 3.00 | ~$65 | ~32× |
| DeepSeek-V3.2-Exp (HF) | 0.27 | 0.40 | ~$11 | ~5.5× |
| Qwen2.5-14B (HF Router) | 0.20 | 0.20 | ~$6 | ~3× |
| **Self-hosted Glaukopis-Qwen2.5-14B** | — | — | **~$2** (amortized GPU-hour, on-prem) | **1× (baseline)** |

Per-token rate cards live in [`SFT/eval/pipelines/api_usage.py`](SFT/eval/pipelines/api_usage.py); the per-sweep aggregator is [`SFT/eval/utils/build_cost_summary.py`](SFT/eval/utils/build_cost_summary.py), with tracked output at `SFT/eval/responses/cost_summary.{csv,tsv}`. Multiply by every analyst query in production and the frontier curve becomes prohibitive.

### Sovereignty

| Axis | Hosted frontier API | Self-hosted Glaukopis SLM (8B / 14B / 32B) |
|---|---|---|
| Data sovereignty | All prompts and completions traverse third-party infrastructure | Zero egress; threat data, IR notes, host artifacts never leave the customer perimeter |
| Air-gap compatible | No | **Yes** — runs in fully air-gapped enclaves with Ollama / vLLM |
| Latency floor | Network RTT + provider queue | Local PCIe — millisecond-class first-token latency |
| Cost model | Per-token, asymptotically unbounded | Fixed CapEx; marginal cost ≈ electricity |
| Domain alignment | Generalist; CTI is one of thousands of long-tail domains | Trained on CTI standards; emits canonical IDs and valid schemas by construction |
| Auditability | Black-box | Open weights, open templates, open verifiers; outputs traceable to a graph walk |

---

## Current state (Q2 2026)

- **Production target:** **Glaukopis-Qwen2.5-32B-Instruct (v21-recal-32b)** — 4-stage chain (Core → TAA → CSE → Recalibrate) on the v21 byte-clone of the v18.1 templates + gates; 32B-tuned Stage-4 recipe (LR 3e-6, Phase-B-heavy 0.15 / 0.60 / 0.25 mix, max-samples 3600). Full SFT, ZeRO-3 + 8-bit AdamW, ~6.5h on 8× H100. **Total 65.0 / Weighted 62.9** under the AthenaBench 50/50 TAA blend — tops all 14B / 8B / MoE peers on absolute leaderboard. HF: [`asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b`](https://huggingface.co/asg-ai).
- **14B matched-baseline:** Glaukopis-Qwen2.5-14B (v21-recalibrate) — Total 61.0 / Weighted 59.6; retained as the fast-iteration anchor on the same template set.
- **MoE port:** Glaukopis-Qwen3-30B-A3B-Thinking-2507 (v21-cse) — Total 63.4 / Weighted 60.9 at the CSE stage; Stage-4 sweeps perturb expert routing, so the chain is closed at Stage 3 for this architecture.
- **Historical anchor:** Glaukopis-Llama-3.1-8B v7 — RMS 62.6 strict-F1, MCQ 57.6, VSP 75.0. The RMS.3a..3h cardinality-stratified template family was the original unlock.
- **Best published RLVR checkpoint:** Minerva-Llama-8B (RLVR layered on v7-class SFT) places **7th overall** on AthenaBench while outperforming GPT-4 and Gemini-2.5-Flash on the two highest-stakes operational axes (RMS, VSP).
- **Open lineage:** every checkpoint is on Hugging Face under [`asg-ai/athena-cti-sft-*`](https://huggingface.co/asg-ai); every recipe is in-repo.

### AthenaBench snapshot (Q1–Q2 2026, 50/50 TAA-Classic + TAA-Canonical blend)

| Rank | Model | Class | Total | RMS | VSP |
|---:|---|---|---:|---:|---:|
| 1 | Gemini-3-Pro | Frontier | **69.7** | 43.1 | 90.7 |
| 2 | GPT-5.2 (high reasoning) | Frontier | 67.1 | 35.6 | 86.1 |
| 3 | **Glaukopis-Qwen2.5-32B v21-recal-32b** | **SLM (this repo)** | **65.0** | 50.1 | 85.0 |
| 4 | Glaukopis-Qwen3-30B-A3B v21-cse (MoE) | SLM (this repo) | 63.4 | 50.1 | 85.0 |
| 5 | Glaukopis-Qwen2.5-14B v21-recalibrate | SLM (this repo) | 61.0 | — | — |
| 6 | GPT-4o | Frontier | 58.0 | 20.2 | 84.7 |
| 7 | Minerva-Llama-8B (Athena Labs, RLVR on v7) | SLM + RLVR | 56.3 | 41.2 | 87.6 |
| 8 | Gemini-2.5-Flash | Frontier | 54.0 | 13.4 | 78.5 |
| 10 | GPT-4 | Frontier | 51.4 | 15.1 | 84.7 |

Full per-axis breakdown across all v21 ports lives in [`SFT/eval/Glaukopis Results.xlsx`](SFT/eval/Glaukopis%20Results.xlsx) (canonical, tracked) and `tmpl_gen/templates/05182026/README-21.md` (per-stage commentary).

### v21 acceptance floors (carried from v18.1, applied across all 5 ports)

| Axis | Floor | Anchor |
|---|---:|---|
| MCQ | ≥ 75.0 | recover 8B-era 77.6 peak on larger bases |
| RMS (strict F1) | ≥ 64.0 | recover Llama v7's 65.8 |
| VSP | ≥ 84.0 | recover historical 86.7 |
| ATE / RCM / SOC / CM | ≥ v18-core − 2pp | no-regression guards |

For the live in-flight context — current sweep state, frontier comparator status, cost-tracking instrumentation, and the full v0..v21 lineage table — see [`PROJECT_STATE.md`](PROJECT_STATE.md).

---

## Repository Layout

| Directory | Purpose | Detailed README |
|-----------|---------|-----------------|
| [`athena_cti_db/`](athena_cti_db/) | Populates a Neo4j graph database with CTI data from MITRE ATT&CK, CAPEC, CWE, CVE, CISA KEV, FIRST EPSS, and MITRE ENGAGE. Produces the graph that feeds `tmpl_gen`. | [`athena_cti_db/README.md`](athena_cti_db/README.md) |
| [`tmpl_gen/`](tmpl_gen/) | Generates structured text (IFT Alpaca-format triples) from graph-based Sophia CTI templates over the Neo4j CTI DB. Produces training data for `SFT`. | [`tmpl_gen/README.md`](tmpl_gen/README.md) |
| [`SFT/`](SFT/) | LlamaFactory-based SFT / LoRA training pipeline for Qwen2.5-32B / 14B-Instruct, Qwen3-30B-A3B-Thinking-2507 (MoE), Foundation-Sec-8B-Instruct, and Llama-3.1-8B-Instruct on the IFT dataset, plus the `SFT/eval/` benchmarking suite. | [`SFT/README.md`](SFT/README.md) |
| [`SFT/eval/`](SFT/eval/) | Benchmarking framework for evaluating LLMs on CTI tasks (ATHENA-RCM/VSP/ATE/TAA/RMS/MCQ) and general NLP tasks (GLUE, SuperGLUE, MMLU, MMLU-Pro, CyberMetric, URLhaus, NVD CVE). | [`SFT/eval/README.md`](SFT/eval/README.md) |

---

## Typical End-to-End Workflow

Each stage has its own environment and dependencies; see the per-submodule README for details.

### 1. Build the CTI graph database (`athena_cti_db`)

Populates a local Neo4j instance with the full CTI graph (MITRE ATT&CK, CAPEC, CWE, CVE, KEV, EPSS, ENGAGE).

```bash
cd athena_cti_db/
./install.sh
export NEO4J_URL="neo4j://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export NEO4J_DB="neo4j"
./populate.sh
```

### 2. Generate IFT training data (`tmpl_gen`)

Runs the template → triples → Alpaca pipeline against the Neo4j CTI DB.

```bash
cd tmpl_gen/
./install.sh -e
cd data_generation/
./make_dataset.sh ../templates/Sophia-CTI-Templates.docx results_dir alpaca.json
```

Configure Neo4j connection parameters in `data_generation/neo4j-local-config.json` before running.

### 3. Fine-tune the base model (`SFT`)

Place the Alpaca dataset produced above in `SFT/data/` and register it in `data/dataset_info.json`, then run LoRA training via LlamaFactory.

```bash
cd SFT/
conda create -n llm-sft python=3.11 -y && conda activate llm-sft
pip install -e .
pip install -r requirements/metrics.txt -r requirements/deepspeed.txt
bash ift_training_qwen_2.5_14b.sh      # or ift_training_llama3_8b.sh
```

After training, merge the LoRA adapters into a standalone model with `llamafactory-cli export`.

### 4. Benchmark the fine-tuned model (`SFT/eval`)

```bash
cd SFT/eval/
conda create -n ctibench python=3.11 -y && conda activate ctibench
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
git lfs pull

python inference.py athena-mcq <model_name> --batch 5 --version 1 \
    --data_path benchmark_data/athena_bench/athena-mcq.tsv
```

Results and metrics are written to `SFT/eval/responses/<model_name>/<task>/`.

---

## Requirements Summary

| Stage | Key requirements |
|-------|-----------------|
| `athena_cti_db` | Python 3.8+, Neo4j 5.x Desktop with APOC, ~20 GB disk |
| `tmpl_gen` | Python (editable install of `tmpl_gen`), running Neo4j CTI DB |
| `SFT` | Python 3.11+, CUDA 12.4-compatible NVIDIA GPU (A100 80 GB recommended for 14B), WandB + Hugging Face tokens |
| `SFT/eval` | Python 3.11+, Git LFS, GPU for local HF models, API keys for hosted models (OpenAI / Google / etc.) |

---

## Status

Active development. Individual submodule APIs, template syntax, schema, and benchmark tasks may evolve.
