# Fine-Tuning LLMs with LlamaFactory (SFT / CPT)

This document describes the end-to-end workflow for fine-tuning a large
language model for Cyber Threat Intelligence (CTI) using
[LlamaFactory](https://github.com/hiyouga/LlamaFactory). The canonical
pipeline is **templates → train → test**: IFT datasets are generated
from Sophia CTI templates via [`tmpl_gen`](../tmpl_gen/README.md),
continued pre-training (CPT) and supervised fine-tuning (SFT) runs are
launched from [`cpt/`](../cpt/README.md) and
[`autotrain/`](autotrain/README.md), and evaluation is driven from
[`test/`](test/README.md) against the AthenaBench, CyberMetric, and
CyberSOCEval suites. The current canonical SFT target is
**Qwen-2.5-14B-Instruct** under the **v20** chain (four sequential
launchers producing the cumulative `v20-core → v20-taa → v20-cse →
v20-recalibrate` checkpoints; `v20-recalibrate` is the headline
release). The earlier **Llama-3.1-8B-Instruct + v7** recipe remains a
fully documented 8B baseline (62.64 % strict F1 on `athena-rms`). CPT
continues to target **Llama-3.1-8B** (base) for corpus-level adaptation.

For the full v20 chain — template manifests, dataset shards, launchers,
recipes, and sign-off gates — see [`SFT_FLOW.md`](SFT_FLOW.md).

---

## Table of Contents

1. [Quick primer: AthenaBench workflow](#quick-primer-athenabench-workflow)
2. [Environment Setup](#environment-setup)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [CUDA and PyTorch Verification](#cuda-and-pytorch-verification)
6. [Pipeline reference: templates → train → test](#pipeline-reference-templates--train--test)
7. [Notes](#notes)

---

## Quick primer: AthenaBench workflow

The `SFT/` tree plus the top-level `cpt/` tree together cover the full
AthenaBench fine-tuning loop: host setup, continued pre-training (CPT),
supervised fine-tuning (SFT), and benchmarking via either a local vLLM
server or HuggingFace's Inference Providers API. The four subsections
below are the minimum set of commands a new contributor needs; each
points at the authoritative launcher and its own `--help` / README for
full flag coverage.

### a) Set up a fresh Linux + CUDA host

[`SFT/utils/setup.sh`](utils/setup.sh) is idempotent: it installs
Miniconda (if missing), creates the `llm-sft` (training) and `ctibench`
(benchmarking) conda envs, installs CUDA-matched PyTorch + LlamaFactory
(editable) into the former and the `SFT/test/` benchmark stack into the
latter, bootstraps `SFT/.env` from `.env.example`, and runs `conda init`
for your shell. Pass `--env-name FOO` together with `--mode all` to
collapse both stacks into a single named env instead.

```bash
cd ~/Glaukopis/SFT
./utils/setup.sh                        # defaults: CUDA 12.4, py=3.11, envs=llm-sft + ctibench
$EDITOR .env                            # fill in HF_TOKEN (write scope) + HF_USERNAME
exec bash                               # pick up the conda shell hook
conda activate llm-sft                  # training; use ctibench for benchmarks
```

Separate vLLM env (kept isolated so vLLM's torch pin does not clobber the
training env):

```bash
./utils/setup.sh --mode vllm            # creates the 'vllm' conda env
```

Full flag reference: `./utils/setup.sh --help`.

### b) Train an SFT model

The canonical pipeline is the **v20 chain** — full-parameter SFT of
`Qwen-2.5-14B-Instruct` across four sequential launchers, each
consuming the prior stage's pushed HF checkpoint as its base. The
template manifests, row-count gates, build watchers, and master plan
all live under [`tmpl_gen/templates/05162026/`](../tmpl_gen/templates/05162026/);
see [`SFT_FLOW.md`](SFT_FLOW.md) for the full description.

```bash
# Confirm the seven v20 JSON shards are on-host (gitignored; rsync from workstation,
# or build via tmpl_gen/data_generation/make_dataset.sh -- see SFT_FLOW.md §2).
ls -lh SFT/data/ift_data_2026_05_16_v20_{core_a_kb_mcq_taa_soc_cm_ms_yn,core_b_rms_ate_vsp_rcm,core_val,taa,taa_val,cse,cse_val}.json

conda activate llm-sft
cd SFT/autotrain

# Stage 1+2: broad re-anchor (Phase A) + AthenaBench catalog drill (Phase B).
./run_sft_qwen25_14b_v20_core.sh         # ~13 h on 8xH100, ~26 h on 4xH100 -> v20-core

# Stage 3: TAA Classic narrow drill on top of v20-core.
./run_sft_qwen25_14b_v20_taa.sh          # ~6-8 h on 8xH100              -> v20-taa

# Stage 4: CyberSOCEval letter-set narrow drill on top of v20-taa.
./run_sft_qwen25_14b_v20_cse.sh          # ~4-6 h on 8xH100              -> v20-cse

# Stage 5: 3-shard interleaved low-LR recalibration on top of v20-cse.
./run_sft_qwen25_14b_v20_recalibrate.sh  # ~95-115 min on 4xH100         -> v20-recalibrate
```

Each launcher pushes its merged full-weight checkpoint to
`hf://${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-{core,taa,cse,recalibrate}`
on exit 0. ZeRO-3 CPU offload is auto-enabled at < 4 GPUs (Core / TAA /
CSE) and < 8 GPUs (Recalibrate); override with `--offload` /
`--no-offload`. All four accept `--dry-run` to print the
`llamafactory-cli` invocation without executing.

**Legacy 8B baseline** — full-parameter SFT of `Llama-3.1-8B-Instruct`
on the consolidated v7 dataset (~181 k rows). First Llama-3.1-8B run
to land above the v0 baseline on every athena task; in particular
`athena-rms` recovered from 5.88 % (v0) / 0.00 % (v6) to
**62.64 % strict F1**. Full recipe details, hyperparameter rationale,
validated benchmark scores, and troubleshooting are in
[`autotrain/README.md`](autotrain/README.md) (*v7 recipe and results*
section).

```bash
./run_abaligned_sft_v7.sh                # Llama-3.1-8B, v7, 3 epochs, lr=1e-5, cutoff 4096
```

Older Llama-3.1-8B launchers (`run_abaligned_sft.sh`,
`run_abaligned_sft_v4.sh`, etc.) are retained for provenance; see the
table in [`autotrain/README.md`](autotrain/README.md).

### c) Train a CPT (continued pre-training) model

CPT lives at the repo root under [`cpt/`](../cpt/README.md) because the
corpus build pipeline (fetch + parse + dedupe + benchmark-leak filter)
is a separate concern from instruction tuning. The launcher drives
LlamaFactory with `--stage pt` (no chat template, packed raw text, 1
epoch by default).

```bash
conda activate llm-sft
pip install -r cpt/requirements.txt

# 1. Build the corpus (fetches sources listed in cpt/sources.yaml).
python cpt/build_corpus.py --out cpt/corpus --name cti_corpus_v1

# 2. Register it with LlamaFactory (appends to SFT/data/dataset_info.json).
python cpt/register_dataset.py --name cti_corpus_v1 \
    --file cpt/corpus/cti_corpus_v1.jsonl

# 3. Launch CPT. Default: base Llama-3.1-8B, LoRA r=32, 1 epoch, 1 H100.
bash cpt/train_cpt.sh --dataset cti_corpus_v1 \
    --repo-id asg-ai/athena-cti-cpt-llama31-8b-v1
```

Source catalog, hyperparameter starting points, and leak-protection
rules live in [`cpt/README.md`](../cpt/README.md).

### d) Benchmark on AthenaBench (vLLM and HF Inference Providers)

Two transports are supported, selected by the suffix on the model alias
registered in [`test/pipelines/models.py`](test/pipelines/models.py):
`-vllm` for a local vLLM server (right choice for private CPT/SFT
models), `-hf` for HuggingFace Inference Providers (right choice for
large public models where hosted tok/s beats local compute), and no
suffix for the default transformers / `device_map="auto"` path.

**Local vLLM server** — two-terminal workflow:

```bash
# Terminal 1 — serve the model (Ctrl-C to tear down).
conda activate vllm
bash SFT/test/utils/serve_vllm.sh \
    --model asg-ai/athena-cti-sft-llama31-8b-abaligned-v3 --tp 2

# Terminal 2 — run the sweep against http://localhost:8000.
conda activate llm-sft
cd SFT/test/utils
./run_benchmark.sh athena-cti-sft-llama31-8b-abaligned-v3-vllm \
    --suite athena --batch 64 --version 1
```

`serve_vllm.sh` auto-applies a bundled chat template for base models
that do not ship one (e.g. `meta-llama/Llama-3.1-8B`); CPT/SFT repos
that carry their own template are used as-is.

**HuggingFace Inference Providers** (hosted API; no local GPU):

```bash
# One-time: put an 'Inference Providers'-scoped token in SFT/test/.env
#   HUGGINGFACE_TOKEN=hf_xxx
conda activate llm-sft
cd SFT/test
./utils/run_benchmark.sh deepseek-r1-14b-hf --batch 32 --overwrite --yes
```

Any alias ending in `-hf` routes through `https://router.huggingface.co/v1`;
`--batch N` fires N concurrent HTTPS requests.

**Local transformers / HF path** (sequential, no server) — default when
the alias has neither `-vllm` nor `-hf` suffix. Useful for transport
parity checks against a vLLM run of the same model:

```bash
conda activate llm-sft
cd SFT/test/utils
./run_benchmark.sh athena-cti-sft-llama31-8b-abaligned-v3 \
    --suite athena --version 1
```

Alias tables, cost estimates, and per-transport limitations are in
[`test/README.md`](test/README.md) (*Local vLLM server* and *HuggingFace
Inference Providers* sections).

---

## Environment Setup

A Linux environment is required. Use Anaconda or Miniconda to manage a dedicated Python environment with Python 3.11 or higher.

### Automated Setup (Linux + CUDA)

The recommended path is the scripted installer under [`utils/`](utils/):

```bash
cd SFT/utils
./setup.sh                                 # defaults: CUDA 12.4, env=llm-sft, python=3.11
./setup.sh --cuda cu121                    # target a different CUDA toolkit
./setup.sh --env-name llm-sft-dev          # custom env name
./setup.sh --extras "metrics deepspeed vllm"  # install additional requirement groups
./setup.sh --no-flash-attn                 # skip flash-attn (e.g. unsupported GPU)
./setup.sh --no-conda-init                 # skip modifying your shell rc
./setup.sh --cuda cpu                      # CPU-only install (also skips flash-attn)
./setup.sh --help
```

The script is idempotent and handles:
1. Bootstrapping Miniconda to `$HOME/miniconda3` if `conda` is not on `PATH`.
2. Creating/reusing the conda env (default `llm-sft`, Python 3.11).
3. Installing the CUDA-matched PyTorch wheels (`cu124` by default).
4. Installing LlamaFactory in editable mode (`pip install -e .`).
5. Installing optional requirement groups from `requirements/` (default: `metrics` + `deepspeed`).
6. Installing `wandb` and `huggingface_hub`.
7. (Optionally) installing `flash-attn` — installs the matching prebuilt
   wheel from GitHub releases directly (avoids the known EXDEV /
   cross-device-link build bug).
8. Printing a PyTorch/CUDA/LlamaFactory verification summary.
9. Running `conda init` for your shell (unless `--no-conda-init` is given) so
   that `conda activate` works in any new terminal.

After it finishes, start a new shell (or `exec bash`) to pick up the conda
shell hook, then activate the env and log in to the experiment/model services:

```bash
exec bash                 # or open a new terminal
conda activate llm-sft
wandb login
huggingface-cli login
```

### Manual Setup

```bash
conda create -n llm-sft python=3.11 -y
conda activate llm-sft

which python
python --version
```

---

## Prerequisites

- An NVIDIA GPU with sufficient VRAM (A100 80 GB recommended for 14B-parameter models); RunPod A100 SXM is used for training and benchmarking both LLMs
- CUDA toolkit compatible with PyTorch (CUDA 12.4 is tested)
- A [Weights & Biases](https://wandb.ai/) API key for experiment tracking
- A [Hugging Face](https://huggingface.co/) token with write access (required for model upload)

---

## Installation

If you are not using [`utils/setup.sh`](utils/setup.sh), install LlamaFactory in editable mode along with the optional dependency groups by hand:

```bash
pip install -e .
pip install -r requirements/metrics.txt -r requirements/deepspeed.txt
pip install wandb huggingface_hub
```

Dependencies are defined in `pyproject.toml`. The `requirements/` directory contains optional dependency groups (`metrics.txt`, `deepspeed.txt`, `vllm.txt`, etc.).

---

## CUDA and PyTorch Verification

Before training, confirm that PyTorch detects your GPU and that CUDA versions are compatible:

```bash
python - << 'EOF'
import torch
import subprocess

print("=== TORCH INFO ===")
print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    print("device capability:", torch.cuda.get_device_capability(0))

print("\n=== NVIDIA-SMI ===")
subprocess.run(["nvidia-smi"])
EOF
```
Recommended version of CUDA and PyTorch:

CUDA stack
Torch CUDA: 12.4
Driver CUDA: 12.7
Driver version: 565.57.01

PyTorch
Version: 2.6.0+cu124

If there is a version mismatch between PyTorch and the installed CUDA driver, reinstall PyTorch targeting the correct CUDA version:

```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Pipeline reference: templates → train → test

The AthenaBench workflow is split across three modules, each with its
own README. This section is a cross-reference map; use the per-module
docs for anything beyond the orientation below.

### Templates — generating the IFT dataset

Sophia CTI templates drive IFT triple generation from a Neo4j CTI
graph (MITRE ATT&CK, CAPEC, CWE, CVE, CISA KEV, FIRST EPSS, MITRE
ENGAGE). The pipeline lives at [`tmpl_gen/`](../tmpl_gen/README.md);
the end-to-end entry point is `tmpl_gen/data_generation/make_dataset.sh`,
which wraps:

1. `docx2json.sh` — extract templates from a `.docx` to JSON
2. `tmpl2triples.sh` — expand templates against the CTI DB
3. `triples2alpaca.sh` — merge triples into an Alpaca-format dataset

The canonical AthenaBench-aligned templates are under
`tmpl_gen/templates/<date>/Sophia-CTI-Templates-AthenaBench-aligned*.txt`.
Output datasets (e.g. `ift_data_2026_04_23_trimmed_v3.json`) are placed
in [`SFT/data/`](data/) and registered in
[`SFT/data/dataset_info.json`](data/dataset_info.json) with the
Alpaca-column mapping LlamaFactory expects
(`instruction` → `system`, `input` → `prompt`, `output` → `response`).

Dataset JSON files are gitignored; rsync them onto the training host
from your workstation or re-generate in place. Full template syntax,
Neo4j connection parameters, and schema-validation tooling are
documented in [`tmpl_gen/README.md`](../tmpl_gen/README.md).

### Train — SFT and CPT launchers

The canonical SFT pipeline is the four-stage v20 chain on
Qwen-2.5-14B-Instruct. Each stage consumes the prior stage's pushed
HF checkpoint as its base model; only the final stage's checkpoint is
the headline release. CPT lives at the repo root and targets the
Llama-3.1-8B base. Older Llama-3.1-8B and Qwen-2.5-14B SFT launchers
(v3 / v4 / v7 / v8.x / v9 / v10–v19) are retained for provenance.

| Stage | Launcher | Base model | Push target |
|---|---|---|---|
| **v20 / 1+2 Core** | [`autotrain/run_sft_qwen25_14b_v20_core.sh`](autotrain/run_sft_qwen25_14b_v20_core.sh) | `Qwen/Qwen2.5-14B-Instruct` | `…/athena-cti-sft-qwen25-14b-v20-core` |
| **v20 / 3 TAA** | [`autotrain/run_sft_qwen25_14b_v20_taa.sh`](autotrain/run_sft_qwen25_14b_v20_taa.sh) | `asg-ai/athena-cti-sft-qwen25-14b-v20-core` | `…/athena-cti-sft-qwen25-14b-v20-taa` |
| **v20 / 4 CSE** | [`autotrain/run_sft_qwen25_14b_v20_cse.sh`](autotrain/run_sft_qwen25_14b_v20_cse.sh) | `asg-ai/athena-cti-sft-qwen25-14b-v20-taa` | `…/athena-cti-sft-qwen25-14b-v20-cse` |
| **v20 / 5 Recalibrate** | [`autotrain/run_sft_qwen25_14b_v20_recalibrate.sh`](autotrain/run_sft_qwen25_14b_v20_recalibrate.sh) | `asg-ai/athena-cti-sft-qwen25-14b-v20-cse` | `…/athena-cti-sft-qwen25-14b-v20-recalibrate` |
| **8B baseline (v7)** | [`autotrain/run_abaligned_sft_v7.sh`](autotrain/run_abaligned_sft_v7.sh) | `meta-llama/Llama-3.1-8B-Instruct` | `…/athena-cti-sft-llama31-8b-abaligned-v7` |
| **CPT** | [`cpt/train_cpt.sh`](../cpt/train_cpt.sh) | `meta-llama/Llama-3.1-8B` (base) | `…/athena-cti-cpt-llama31-8b-<run-id>` |

Every launcher accepts `--dry-run` (print the `llamafactory-cli`
command and exit) and auto-configures DeepSpeed offload based on
visible GPU count. On exit 0 the merged full-weight checkpoint is
pushed to `hf://${HF_USERNAME}/<repo-id>`. Per-stage recipe details
(cutoff, packing, eff_bs, LR schedule, sign-off gates) are in
[`SFT_FLOW.md`](SFT_FLOW.md); flag-reference, checkpoint layout, and
historical-launcher index are in
[`autotrain/README.md`](autotrain/README.md) and
[`cpt/README.md`](../cpt/README.md).

### Test — AthenaBench evaluation

Benchmark sweeps are launched from
[`test/utils/run_benchmark.sh`](test/utils/run_benchmark.sh). The
transport is selected by the suffix on the model alias registered in
[`test/pipelines/models.py`](test/pipelines/models.py):

| Alias suffix | Transport | Use when |
|---|---|---|
| `-vllm` | Local vLLM OpenAI-compatible server (`test/utils/serve_vllm.sh`) | Benchmarking private CPT/SFT checkpoints; high-throughput |
| `-hf` | HuggingFace Inference Providers (hosted API) | Large public models where hosted tok/s beats local compute |
| *(none)* | Local transformers, `device_map="auto"` | Transport-parity checks; no batching |

All three transports share the same prompt templates, scoring code,
and response-cache directory layout under `test/responses/<model>/`.
The transformers path is sequential (no `--batch`); the `-vllm` and
`-hf` paths accept `--batch N` for N concurrent requests. Alias
tables, cost estimates, per-task row counts, and response-diff
tooling (`test/utils/diff_hf_vllm_responses.py`) are documented in
[`test/README.md`](test/README.md).

---

## Notes

- **GPU requirements**: The v20 Qwen-2.5-14B chain is sized for
  8×H100 80 GB (Phase A / Phase B / TAA / CSE) and 4×H100 80 GB
  (Recalibrate); Phase B (cutoff 16384, packing off) is the
  memory-tight stage. The v20 launchers auto-flip ZeRO-3 CPU offload
  on at < 4 GPUs (Core / TAA / CSE) and < 8 GPUs (Recalibrate);
  override with `--offload` / `--no-offload`. The legacy 8B v7
  full-parameter recipe runs on a single A100 80 GB with ZeRO-3 + CPU
  offload, or on ≥ 2× 80 GB without offload. See
  [`autotrain/README.md`](autotrain/README.md) and
  [`SFT_FLOW.md`](SFT_FLOW.md) for per-stage memory budgets.
- **CUDA compatibility**: Verify that your PyTorch CUDA version
  matches your driver's supported CUDA version before starting
  training. Mismatches cause silent failures or crashes.
- **Checkpoint paths**: Training runs write to `saves/<model>/<ts>/`.
  The launchers merge and upload on exit 0; intermediate checkpoints
  are not committed to the HF repo.
- **Secrets**: Put `HF_TOKEN` and `HF_USERNAME` in
  [`SFT/.env`](.env.example); never commit real tokens. The setup
  script bootstraps `.env` from `.env.example` if missing.
- **Dataset files**: `SFT/data/*.json` training sets are gitignored
  due to size (tens of MB). Rsync them onto the training host from a
  workstation or regenerate from `tmpl_gen` before launching a run.
