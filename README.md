# Glaukopis

End-to-end Cyber Threat Intelligence (CTI) Supervised Fine-Tuning (SFT) training and evaluation pipeline.

Glaukopis is one stage of a larger CTI LLM toolchain:

```
Ariadne  →  Sophia  →  Glaukopis  →  Alkidemos
```

Within this repository, the pipeline covers:

1. Building a CTI knowledge graph from public threat-intelligence sources (`athena_cti_db`).
2. Generating Instruction Fine-Tuning (IFT) data from the graph using text templates (`tmpl_gen`).
3. Supervised fine-tuning of base LLMs on the IFT data with LoRA (`SFT`).
4. Benchmarking the resulting models on CTI and general-reasoning tasks (`athena_bench`).

---

## Repository Layout

| Directory | Purpose | Detailed README |
|-----------|---------|-----------------|
| [`athena_cti_db/`](athena_cti_db/) | Populates a Neo4j graph database with CTI data from MITRE ATT&CK, CAPEC, CWE, CVE, CISA KEV, FIRST EPSS, and MITRE ENGAGE. Produces the graph that feeds `tmpl_gen`. | [`athena_cti_db/README.md`](athena_cti_db/README.md) |
| [`tmpl_gen/`](tmpl_gen/) | Generates structured text (IFT Alpaca-format triples) from graph-based Sophia CTI templates over the Neo4j CTI DB. Produces training data for `SFT`. | [`tmpl_gen/README.md`](tmpl_gen/README.md) |
| [`SFT/`](SFT/) | LlamaFactory-based SFT / LoRA training pipeline for Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct on the IFT dataset. Produces fine-tuned models for `athena_bench`. | [`SFT/README.md`](SFT/README.md) |
| [`athena_bench/`](athena_bench/) | Benchmarking framework for evaluating LLMs on CTI tasks (ATHENA-RCM/VSP/ATE/TAA/RMS/MCQ) and general NLP tasks (GLUE, SuperGLUE, MMLU, CyberMetric, URLhaus, NVD CVE). | [`athena_bench/README.md`](athena_bench/README.md) |

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

### 4. Benchmark the fine-tuned model (`athena_bench`)

```bash
cd athena_bench/
conda create -n ctibench python=3.11 -y && conda activate ctibench
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
git lfs pull

python inference.py athena-mcq <model_name> --batch 5 --version 1 \
    --data_path benchmark_data/athena_bench/athena-mcq.tsv
```

Results and metrics are written to `athena_bench/responses/<model_name>/<task>/`.

---

## Requirements Summary

| Stage | Key requirements |
|-------|-----------------|
| `athena_cti_db` | Python 3.8+, Neo4j 5.x Desktop with APOC, ~20 GB disk |
| `tmpl_gen` | Python (editable install of `tmpl_gen`), running Neo4j CTI DB |
| `SFT` | Python 3.11+, CUDA 12.4-compatible NVIDIA GPU (A100 80 GB recommended for 14B), WandB + Hugging Face tokens |
| `athena_bench` | Python 3.11+, Git LFS, GPU for local HF models, API keys for hosted models (OpenAI / Google / etc.) |

---

## Status

Active development. Individual submodule APIs, template syntax, schema, and benchmark tasks may evolve.
