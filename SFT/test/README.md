# Athena CTI Bench

A benchmarking framework for evaluating Large Language Models (LLMs) on Cyber Threat Intelligence (CTI) tasks.

## Overview

Athena CTI Bench provides a comprehensive evaluation suite for assessing Large Language Model (LLM) performance across fifteen key Cyber Threat Intelligence (CTI) and general reasoning tasks.

#### Core CTI Benchmarks:

- **MCQ**: Multiple Choice Questions on CTI knowledge
- **RCM**: Relationship Classification for Malware entities
- **VSP**: Vulnerability Severity Prediction
- **ATE**: Automated Threat Entity extraction
- **TAA**: Threat Actor Attribution

#### Athena Advanced CTI Benchmarks:

- **ATHENA-RCM (Root Cause Mapping):** Maps each CVE narrative to a structured root cause category, enabling models to connect qualitative descriptions to remediation insights.
- **ATHENA-VSP (Vulnerability Severity Prediction):** Predicts the CVSS vector and overall severity score from the vulnerability record.
- **ATHENA-ATE (Attack Technique Enumeration):** Identifies relevant MITRE ATT&CK techniques from adversary scenarios.
- **ATHENA-TAA (Threat Actor Attribution):** Attributes threat reports to the correct actor or collective based on extracted evidence.
- **ATHENA-RMS (Risk Mitigation Strategy):** Generates prioritized defensive actions that address an identified attack path or technique set.
- **ATHENA-MCQ (Multiple Choice Questions):** Builds question banks from authoritative CTI sources such as MITRE ATT&CK, CAPEC, CWE, and CISA advisories using the `athena_scrape` pipeline.

#### Additional Benchmarks:

- **URLHAUS (IOC BENCHMARKING):** Evaluates model performance on identifying and classifying malicious vs. benign URLs using recent IOC data.
- **CVE (Vulnerability Benchmarking):** Tests understanding of vulnerabilities from the latest 90 days of NVD data, including descriptions and CVSS metrics.
- **CYBERMETRIC (CTI MCQ Evaluation):** Assesses CTI knowledge through multiple-choice questions based on cybersecurity metrics and terminology.
- **GLUE (General Language Understanding Evaluation):** Evaluates language understanding tasks like sentence similarity, entailment, and sentiment.
- **SUPERGLUE (Advanced NLP Evaluation):** A more challenging version of GLUE focusing on reasoning and comprehension.
- **MMLU (Massive Multitask Language Understanding):** Tests broad knowledge across academic and professional subjects.
- **MMLU-Pro (Robust MMLU Successor):** Harder, reasoning-focused successor to MMLU from TIGER-AI-Lab; ~12K questions across 14 academic domains, up to 10 answer options per question, evaluated zero-shot CoT against the upstream "The answer is (X)" extraction contract.

## Setup

### Automated Setup (Linux + CUDA)

The recommended path on a Linux CUDA host is the scripted installer under
[`utils/`](utils/):

```bash
cd SFT/utils
./setup.sh                          # defaults: CUDA 12.4, python=3.11, envs=llm-sft + ctibench
./setup.sh --mode test              # benchmarking stack only -> 'ctibench' env
./setup.sh --cuda cu121             # target a different CUDA toolkit
./setup.sh --env-name ctibench-dev  # collapse both stacks into one custom env (with --mode all)
./setup.sh --no-flash-attn          # skip flash-attn (e.g. unsupported GPU)
./setup.sh --lfs-pull               # opt in to 'git lfs pull' for data/ (see note below)
./setup.sh --no-conda-init          # skip modifying your shell rc
./setup.sh --cuda cpu               # CPU-only install (also skips flash-attn)
./setup.sh --mode vllm              # isolated 'vllm' env (see Local vLLM section)
./setup.sh --help
```

The script is idempotent and handles:
1. Bootstrapping Miniconda to `$HOME/miniconda3` if `conda` is not on `PATH`.
2. Creating/reusing the conda envs (default: `llm-sft` for training +
   `ctibench` for benchmarking; `--mode test` alone creates only `ctibench`).
3. Installing the CUDA-matched PyTorch wheels (`cu124` by default).
4. Installing `requirements.txt` and (optionally) `flash-attn`.
   - `flash-attn` is **optional and non-fatal**: the runtime uses PyTorch SDPA
     by default (see *Attention backend* below), so a failed flash-attn
     install no longer aborts setup.
   - When requested, the script installs the matching prebuilt wheel from
     GitHub releases directly (avoids the known EXDEV / cross-device-link
     build bug).
5. Installing Git LFS (but **not** running `git lfs pull` by default — see below).
6. Printing a PyTorch/CUDA verification summary.
7. Running `conda init` for your shell (unless `--no-conda-init` is given) so
   that `conda activate` works in any new terminal.

After it finishes, start a new shell (or `exec bash`) to pick up the conda
shell hook, then activate the env and run the smoke test:

```bash
exec bash                 # or open a new terminal
conda activate ctibench
./utils/smoke_test.sh
```

### Attention backend

HuggingFace models are loaded with PyTorch's SDPA attention by default. SDPA
dispatches to flash / memory-efficient CUDA kernels under the hood and avoids
the `transformers` × `flash-attn` version-mismatch failures that surface on
Qwen2-based models (DeepHat, Qwen2.5-*, etc.).

Override via env var if you have a working flash-attn build and want to use it
explicitly:

```bash
ATHENA_ATTN_IMPL=flash_attention_2 python inference.py ...
ATHENA_ATTN_IMPL=sdpa              python inference.py ...   # default
ATHENA_ATTN_IMPL=eager             python inference.py ...   # last-resort fallback
```

If the requested implementation fails to load, the loader falls back through
`sdpa` and then `eager` automatically.

### Git LFS and the `data/` directory

The scripted installer no longer runs `git lfs pull` by default because:

- The benchmark files under `benchmark_data/` and `benchmark_data_mini/` used by
  `inference.py` are tracked as **regular git files** — they do not need LFS.
- The tree under `data/` (MCQ scrape output, NVD daily snapshots, processed
  MITRE ATT&CK bundles, TAA working files, etc.) consists entirely of LFS
  pointer files whose backing objects are largely **missing from the LFS
  server**. Running `git lfs pull` against this repo currently fails with
  `Object does not exist on the server: [404]`.
- `data/` is only needed when you are **regenerating** benchmark datasets
  (see the *Dataset creation* section below). It can be reproduced from
  scratch via the `athena_scrape` pipelines.

If you do need `data/` populated, either:

```bash
./setup.sh --lfs-pull              # best-effort fetch; non-fatal if objects are missing
# or, after setup:
git lfs pull                       # best-effort; expect 404s for missing objects
```

…and then regenerate whatever is still missing using the commands under
*Dataset creation*.

### Manual Setup

If you prefer to install by hand:

```bash
conda create -n ctibench python=3.11 -y
conda activate ctibench
pip install -r requirements.txt
# (Important) Install Flash Attention for Hugging Face model performance
pip install flash-attn --no-build-isolation

conda install -c conda-forge git-lfs -y
git lfs install
# 'git lfs pull' is optional; see the 'Git LFS and the data/ directory' note above
```

Git LFS is required to fetch the large files stored in `data/`.

## Usage

### Running Inference

The framework supports multiple LLM providers including OpenAI GPT models, Meta Llama models, and others.

Basic command structure:

```bash
python inference.py task subtask model_name --athena-cti-lnd --batch N --rows N --version N --data_path PATH --cleanup
```

Parameters:
- `task`: One of `mcq`, `rcm`, `vsp`, `ate`, `taa`, `cve`, `urlhaus`,`cybermetric`, `athena-ate`, `athena-taa`, `athena-rcm`, `athena-rms`, `athena-vsp`,
`athena-mcq`, `glue`, `superglue`, `mmlu`, `mmlu-pro`
- `subtask`: (Optional) Subtask name for GLUE or SUPERGLUE (e.g., cola, sst2).
- `model_name`: Model identifier (e.g., `gpt4`, `gemini-2.5-flash`, `llama-3-70b`)
- `--athena-cti-lnd`: (Optional) Enables the web search preview tool, used only for CVE tasks with GPT-5 or Gemini models.
- `--batch`: (Optional) Number of concurrent workers (useful for API-based models like GPT or Gemini).
- `--rows`: Optional number of data rows to process (default: all)
- `--version`: (Optional) Run version number (default: 1). Use higher numbers for new or resumed runs.
- `--data_path`: Optional custom data path (default: `benchmark_data/cti_bench/cti-<task>.tsv`)
- `--cleanup`: (Optional parameter) Force cleanup of model from memory after each inference only when using non-api models on runpod for free the gpu memory space in order to run the current task in the gpu memory.

### Example

Run inference on complete MCQ questions dataset using GPT-4 Turbo:

```bash
python inference.py mcq gpt-4-turbo --data_path benchmark_data\cti_bench\cti-mcq.tsv
```

Note: --data_path is optional 
Run inference on 5 rows MCQ questions using GPT-4 Turbo:

```bash
python inference.py mcq gpt-4-turbo --rows 5 --data_path benchmark_data\cti_bench\cti-mcq.tsv
```

Run inference for the ATHENA-MCQ task using GPT-5:
```bash
python inference.py athena-mcq gpt5 --batch 5 --version 2 --data_path benchmark_data\athena_bench\athena-mcq.tsv --cleanup
```

### HuggingFace Inference Providers (hosted API)

Any model key ending in `-hf` is routed through the HuggingFace Inference
Providers router (`https://router.huggingface.co/v1`) rather than loaded onto
a local GPU. The router auto-selects a backing provider (Together, Fireworks,
Sambanova, Cerebras, Novita, etc.) based on what has the model hosted and the
account's provider preference order.

This is the fastest path for large or reasoning models (e.g.
`deepseek-r1-14b`) where per-row inference locally would take 10s+.

**One-time account setup** (no model deployment required — the models are
already hosted by the providers):

1. Log in at https://huggingface.co and open **Settings → Access Tokens**.
   Create a token with scope **"Make calls to Inference Providers"**
   (read-only is sufficient). Copy the token.
2. Open **Settings → Billing** and either:
   - subscribe to **HF Pro** (includes monthly inference credits), or
   - enable **pay-as-you-go** by attaching a payment method. Usage is then
     billed per token at the backing provider's rate.
3. Optionally open **Settings → Inference Providers** to reorder which
   provider gets tried first for each model.

**Repo setup**:

Put the token in the usual `.env` file the rest of the framework already
reads (or export it in the shell):

```bash
# SFT/test/.env
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`huggingface_hub` is already pinned in `requirements.txt`, so no new install
step is needed.

**Available `-hf` model keys**:

| Key | Backing model |
|---|---|
| `deepseek-r1-14b-hf` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| `deepseek-r1-70b-hf` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` |
| `qwen3-14b-hf`       | `Qwen/Qwen3-14B` |
| `qwen3-32b-hf`       | `Qwen/Qwen3-32B` |
| `qwen2.5-14b-hf`     | `Qwen/Qwen2.5-14B-Instruct` |
| `llama-3-70b-hf`     | `meta-llama/Meta-Llama-3-70B-Instruct` |
| `llama3.3-70b-hf`    | `meta-llama/Llama-3.3-70B-Instruct` |
| `deepseek-v3.2-exp-hf` | `deepseek-ai/DeepSeek-V3.2-Exp` |

Additional keys can be registered by adding an entry to `model_mapping` in
`pipelines/models.py` with the `-hf` suffix.

**Example — single task with concurrency**:

```bash
python inference.py athena-mcq deepseek-r1-14b-hf --batch 32 --version 1
```

`--batch N` fires N concurrent HTTP requests (via `ThreadPoolExecutor`); the
provider's continuous batching handles the rest. Typical throughput for a
14B reasoning model on Together: 30-60× vs local single-GPU HF.

**Example — full sweep**:

```bash
./utils/run_benchmark.sh deepseek-r1-14b-hf --batch 32 --overwrite --yes
```

Note: do **not** use `run_benchmark_parallel.sh` for `-hf` models. That script
shards across GPUs, which only makes sense for local inference. The sequential
`run_benchmark.sh` + `--batch N` is the right tool for hosted inference, since
concurrency is already handled at the HTTP layer.

**Cost estimation** (very rough, depends on provider's per-token rate):

| Model | Typical rate | Full 6-task sweep |
|---|---|---|
| `deepseek-r1-14b-hf` | ~$0.30 / 1M tokens combined | ~$10-15 |
| `llama3.3-70b-hf`    | ~$0.80 / 1M tokens combined | ~$20-30 |

Check `https://huggingface.co/<model-id>?inference_provider=...` for the
current per-provider pricing.

### Local vLLM server (`-vllm` suffix)

Any model key ending in `-vllm` is routed through a local
[vLLM](https://docs.vllm.ai/) OpenAI-compatible HTTP server rather than the
default transformers/`device_map="auto"` load path. vLLM loads the model once
and handles concurrent requests via continuous batching, so `--batch N` in
`run_benchmark.sh` maps to N in-flight `/v1/chat/completions` requests — on a
single H100 this is typically 10–20× the throughput of the transformers path
for an 8B model at the same accuracy.

This is the intended inference path for **private fine-tuned models** (CPT,
SFT) that the HF Inference Providers router cannot serve. The HF-hosted
`-hf` path remains the right choice for large public models
(`deepseek-r1-70b-hf` etc.); `-vllm` is the right choice for local custom
models.

**One-time env setup** (separate conda env to keep vllm's torch pin
isolated from the training / llamafactory env):

```bash
bash SFT/utils/setup.sh --mode vllm    # creates 'vllm' conda env
```

**Two-terminal workflow**:

```bash
# Terminal 1 — start the server. Foreground; Ctrl-C tears it down.
conda activate vllm
bash SFT/test/utils/serve_vllm.sh \
    --model asg-ai/athena-cti-cpt-llama31-8b-v1 \
    --tp 2                              # tensor-parallel-size (2xH100)

# Terminal 2 — point the benchmark harness at localhost:8000.
conda activate llm-sft                  # or ctibench if --split-envs
cd SFT/test/utils
./run_benchmark.sh athena-cti-cpt-llama31-8b-v1-vllm \
    --suite athena --batch 64 --version 2
```

**Chat-template handling**: `serve_vllm.sh` probes the model repo for a
`chat_template` on startup. Fine-tuned models (CPT/SFT) ship one and need
nothing extra. Base models (e.g. `meta-llama/Llama-3.1-8B`) do not, which
makes `/v1/chat/completions` return 400. The script detects this, matches
the repo name against a family table, and auto-applies a bundled jinja
from `utils/chat_templates/`:

| Repo pattern     | Bundled template |
|---|---|
| `llama-3` / `llama3` | `utils/chat_templates/llama3.jinja` |

Override with `--chat-template <path>` or disable with
`--no-auto-template`. Add additional families by dropping a new jinja
into `utils/chat_templates/` and extending the pattern match in
`serve_vllm.sh`.

The same auto-apply runs on the **local transformers path**
(`HuggingFaceModel.load_model` in `pipelines/models.py`). When a base
model's tokenizer has no `chat_template` and the repo id matches a
known family, the bundled jinja is assigned to
`tokenizer.chat_template` at load time so that `generate()` takes the
chat-formatted branch instead of feeding a raw continuation prompt.
Without this, base Llama-3.1-8B emits `<|end_of_text|>` as token 1 on
open-ended CTI prompts (ATE, RMS, TAA, RCM) and returns empty strings,
producing misleading 0% baselines.

**Available `-vllm` model keys**:

| Key | Backing model |
|---|---|
| `llama-3-8b-base-vllm`                    | `meta-llama/Llama-3.1-8B`                         |
| `llama-3-8b-vllm`                         | `meta-llama/Meta-Llama-3.1-8B-Instruct`           |
| `qwen3-32b-vllm`                          | `Qwen/Qwen3-32B`                                  |
| `athena-cti-cpt-llama31-8b-v1-vllm`       | `asg-ai/athena-cti-cpt-llama31-8b-v1`             |
| `athena-cti-sft-llama31-8b-abaligned-v4-vllm` | `asg-ai/athena-cti-sft-llama31-8b-abaligned-v4` |

Additional keys can be registered by adding an entry to `model_mapping` in
`pipelines/models.py` with the `-vllm` suffix.

**Config via env** (consumed by `VLLMModel` on the client side):

```bash
VLLM_BASE_URL=http://localhost:8000/v1   # default
VLLM_API_KEY=EMPTY                       # vllm ignores; SDK needs non-empty
```

Override when running the benchmark against a non-default port or a
remote vllm host.

**Limitations**:

- The served model and the benchmark alias must point at the same HF repo
  id (see `model_mapping` — the `-vllm` alias value matches the non-suffix
  alias). vLLM checks `model` against what it loaded and 404s on mismatch.
- `--cleanup` on `inference.py` is a no-op for `-vllm` models: there are
  no local weights in the benchmark process to evict.
- Do **not** install `vllm` into the `llm-sft` / `ctibench` envs. vllm
  pins torch precisely; mixing causes one or the other to break at import.
  Always use the isolated `vllm` env from `setup.sh --mode vllm`.

## Evaluation

Results are automatically evaluated after inference and saved in the responses/<model_name>/<task_name>/ directory. The framework calculates task-specific metrics:

- ATHENA-MCQ, MCQ, ATHENA-RCM, RCM, CYBERMETRIC, URLHAUS, MMLU: Accuracy
- ATHENA-VSP, VSP: Mean Absolute Deviation (MAD)
- ATHENA-ATE, ATE: F1 Score
- ATHENA-TAA, TAA: Correct and Plausible Accuracy
- GLUE: Accuracy,F1-score,Pearson,Spearmanr
- SUPERGLUE: Accuracy, F1-score, Exact Match
- MMLU-Pro: Overall accuracy, per-category accuracy, parse-error rate (returns a nested dict so the per-task pretty-printer renders the 14-domain breakdown beneath the headline number)

Evaluate predictions separately (optional):
The task_evaluation.py script can be used to manually evaluate model outputs after inference.
It loads a saved response file, runs the appropriate evaluation logic for each task, and prints the final metrics to the console.

```bash
# Example: Evaluate ATHENA-MCQ task results from GPT-5
python task_evaluation.py athena-mcq gpt5 responses/gpt5/athena-mcq/athena-mcq_all_v1_gpt5_response.jsonl
```

## Dataset creation

Dataset generation utilities live in the `athena_data` package and are driven by
`athena_data/config.yaml`.

### ATHENA-RCM and ATHENA-VSP

RCM (Root Cause Mapping) and VSP (CVSS Vector Severity Prediction) use NVD CVE records.

1. Download NVD records:

   ```bash
   python -m athena_data.common.download_nvd_records --config athena_data/config.yaml
   ```

2. Build benchmark files:

   ```bash
   python -m athena_data.cve.create_cve_data --config athena_data/config.yaml
   ```

   This writes `benchmark_data/athena_bench/athena-cti-rcm.jsonl` and `benchmark_data/athena_bench/athena-cti-vsp.jsonl`.

### ATHENA-TAA

TAA (Threat Actor Attribution) relies on URLs to threat reports listed in
`data/taa/threat_actor_url.csv`.

1. Extract and anonymize reports:

   ```bash
   python -m athena_data.taa.create_taa_data --config athena_data/config.yaml
   ```

2. Create the benchmark file:

   ```bash
   python -m athena_data.taa.make_benchmark --config athena_data/config.yaml
   ```

   The final dataset is written to `benchmark_data/athena_bench/athena-cti-taa.jsonl`.


### ATHENA-MCQ

Use the `scrape/athena_scrape` CLI to pull fresh URLs and generate the corpus in one step.

1. Collect the latest URL catalog.

   ```bash
   python -m athena_scrape collect --sources mitre_attack cwe capec cisa_ics cisa_csa --out data/raw/urls/all_urls.csv
   ```

2. Fetch each URL and write cleaned text plus markdown summaries.

   ```bash
   python -m athena_scrape build-corpus \
       --url-csv data/raw/urls/all_urls.csv \
       --raw-dir data/raw/mcq_data \
       --processed-dir data/processed/mcq
   ```

3. Draft the MCQ question plan (sampling + question counts).

   ```bash
   python -m athena_scrape plan-mcq \
       --url-csv data/raw/urls/all_urls.csv \
       --processed-dir data/processed/mcq \
       --raw-dir data/raw/mcq_data \
       --out data/processed/mcq/mcq_plan.tsv
   ```

4. Generate MCQ questions with OpenAI (JSONL output).

   ```bash
   python -m athena_data.mcq.create_mcq_data --config athena_data/config.yaml
   ```

   This reads `data/processed/mcq/mcq_plan.tsv`, shows a progress bar over all rows,
   calls the MCQ prompt with reasoning_effort=medium, parses the returned TSV rows into
   structured records, and writes JSONL to `benchmark_data/athena_bench/athena-cti-mcq.jsonl` including
   source metadata from the plan (e.g., `url_id`, `url`, `source_type`, paths, counts).

   CLI behavior and options:

   - If `benchmark_data/athena_bench/athena-cti-mcq.jsonl` already exists and you do not pass `--rebuild`, the script does not call the API. Instead, it inserts a formatted evaluation prompt into any rows missing a `prompt` field and exits.
   - Force prompt hydration only (no API calls):

     ```bash
     python -m athena_data.mcq.create_mcq_data --config athena_data/config.yaml --hydrate-only
     ```

   - Force full regeneration via API (ignore existing JSONL):

     ```bash
     python -m athena_data.mcq.create_mcq_data --config athena_data/config.yaml --rebuild
     ```
#### ATHENA-MCQ Manual Validation and Patch Application

After generating `benchmark_data/athena_bench/athena-cti-mcq.jsonl`, we support manual correction of problematic questions via a patch sheet `benchmark_data/athena_bench/mcq-patch.tsv`.

- The patch file must contain at least the columns: `id` (0-based row index in the JSONL), `question` (exact text), and `answer` (one of `A`-`E` or `X`).
- During application, the script asserts that the `question` text for each patched `id` exactly matches the corresponding JSONL row to ensure alignment.
- The script reports counts for:
  - How many patched rows have `answer` = `X`.
  - Among remaining, how many had `correct_answer` not in `[A, B, C, D, E]` originally but have a valid letter in the patch.
  - Among remaining, how many had a valid original letter but are replaced by a different letter in the patch.
  - Among remaining, how many have the same letter between original and patch.
- It then writes a new file `benchmark_data/athena_bench/athena-cti-mcq-updated.jsonl` that adds an `updated_answer` field: for patched rows this is the patched `answer`; for all others this is copied from the original `correct_answer`. The original JSONL is left unchanged.

Apply the patch and produce the updated file:

```bash
python -m athena_data.mcq.apply_mcq_patch \
  --input benchmark_data/athena_bench/athena-cti-mcq.jsonl \
  --patch benchmark_data/athena_bench/mcq-patch.tsv \
  --output benchmark_data/athena_bench/athena-cti-mcq-updated.jsonl
```

#### ATHENA-MCQ 3k Subset Curation

To create a 3,000-question subset for evaluation while excluding rows marked as invalid (`updated_answer = X`), run:

```bash
python -m athena_data.mcq.make_mcq_subset \
  --input benchmark_data/athena_bench/athena-cti-mcq-updated.jsonl \
  --output benchmark_data/athena_bench/athena-cti-mcq-3k.jsonl \
  --size 3000 \
  --seed 42
```

Notes:
- The script filters out any rows with `updated_answer` equal to `X`.
- It samples the requested number of questions uniformly at random with the provided seed.
- In the subset file, the `answer` field is set to match `updated_answer` for consistency.

### ATHENA-RMS and ATHENA-ATE

RMS (Risk Mitigation Strategy) and ATE (Attack Technique Enumeration) are built
from MITRE ATT&CK technique data.

1. Generate scenarios and build both benchmarks:

   ```bash
   python -m athena_data.mitre_attck.create_mitre_attck_data --config athena_data/config.yaml
   ```

   This downloads the ATT&CK bundle, creates GPT-5 scenarios, and writes
   processed records to `data/processed/mitre_attck/`. The script also produces
   the benchmark files `benchmark_data/athena_bench/athena-cti-rms.jsonl` and
   `benchmark_data/athena_bench/athena-cti-ate.jsonl`.

### MINI ATHENA BENCHMARKS
Create smaller subsets for quick iteration and CI checks. The script adds a `prompt_hash` to all original benchmark files (computed as SHA-256 of the `prompt`) and then writes mini subsets to `benchmark_data_mini/`.

Rules:
- MCQ, MCQ3k, ATE, RCM, RMS, VSP: sample 10% (ceil) uniformly at random.
- TAA: sample a fixed 25 records.

Run:

```bash
python -m athena_data.create_mini_benchmarks \
  --config athena_eval/config.yaml \
  --out-dir benchmark_data_mini \
  --update-originals \
  --seed 42
```

Notes:
- `--update-originals` writes `prompt_hash` back into the original benchmark JSONL files so you can reuse existing runs by matching on `prompt_hash` if needed.
- Mini files retain `prompt_hash` for easy mapping between full and mini datasets.
- The script enforces that `prompt_hash` values are unique within each task file (derived from `prompt`). If duplicate prompts exist in a task (which would cause identical hashes), the script raises an error so you can deduplicate or adjust the data before proceeding.

### CVE (NVD) — Recent 90 Days Data
CVE benchmark data is created by fetching recent NVD vulnerability records.
This script collects all CVEs modified in the last 90 days and saves them in JSONL format in `benchmark_data/cve`.
```bash
   python -m scrape.nvd_cve.nvd_data
   ```

### URLHAUS (IOC Benchmark) — Malicious IOC Collection
1. Fetch Malicious IOCs
    Retrieve the latest malicious URLs from the URLHaus API for benchmarking and it is saved in `benchmark_data/urlhaus`.
    ```bash
    python -m scrape.urlhaus.fetch_urlhaus
    ```
2. Generate Benign Lookalike IOCs (using GPT-5)
   Randomly select 100 malicious IOCs from the URLHaus dataset and generate 100 benign lookalike URLs using GPT-5. The ouput file is saved in `benchmark_data/urlhaus`.
   ```bash
   python -m scrape.urlhaus.generate_benign_iocs --input_csv --output_csv model --num_iocs N --include_all
    ```
   CLI Arguments:
   - `input_csv`: Path to the input CSV containing malicious IOCs. Default: benchmark_data/urlhaus/urlhaus_full_ioc.csv
   - `output_csv`: Path to the output CSV file. Default: benchmark_data/urlhaus/urls_benchmark_<YYYYMMDD>.csv
   - `model`: Model name to use (e.g., gpt5). Required
   - `num_iocs`: Number of malicious IOCs to process. Default: 100
   - `include_all`: Include all malicious IOCs in output before generating benigns. Default: False
   
   Notes:
   - Output CSV automatically includes a date tag (YYYYMMDD) in the filename.
   - If `--include_all` is used, all malicious URLs from the input file are written first, followed by generated benign URLs.
   - If `--include_all` is not used, the final output will contain only num_iocs malicious URLs and their corresponding generated benign URLs (resulting in a total of num_iocs * 2 rows).

## Project Structure

- `benchmarks/`: Task-specific benchmark implementations
- `pipelines/`: Core components for data loading, model inference, and evaluation
- `data/`: Intermediate data files generated from scraping Athena CTI tasks
- `benchmark_data/`: Benchmark datasets for Athena-CTI, CTI, CVE, and IOC evaluation tasks.
- `benchmark_data_mini/` : Mini Benchmark datasets for Athena-CTI.
- `athena_data/`: Scripts to generate ready-to-benchmark JSONL files for Athena CTI tasks
- `scrape/`: Scripts for scraping data from various CTI sources
- `responses/`: Generated model responses and evaluation results