# Fine-Tuning LLMs with LlamaFactory (SFT / LoRA)

This document describes the end-to-end workflow for supervised fine-tuning (SFT) of a large language model using [LlamaFactory](https://github.com/hiyouga/LlamaFactory), with LoRA adapters. The current configuration targets **Qwen2.5-14B-Instruct** on a custom instruction-following dataset (`ift_data`).

---

## Table of Contents

1. [Quick primer: AthenaBench workflow](#quick-primer-athenabench-workflow)
2. [Environment Setup](#environment-setup)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [CUDA and PyTorch Verification](#cuda-and-pytorch-verification)
6. [Dataset Preparation](#dataset-preparation)
7. [Training](#training)
8. [Merging LoRA Adapters](#merging-lora-adapters)
9. [Local Inference](#local-inference)
10. [Uploading to Hugging Face](#uploading-to-hugging-face)
11. [Notes](#notes)

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
Miniconda (if missing), creates the `llm-sft` conda env, installs
CUDA-matched PyTorch + LlamaFactory (editable) + the `SFT/test/`
benchmark stack, bootstraps `SFT/.env` from `.env.example`, and runs
`conda init` for your shell.

```bash
cd ~/Glaukopis/SFT
./utils/setup.sh                        # defaults: CUDA 12.4, env=llm-sft, py=3.11
$EDITOR .env                            # fill in HF_TOKEN (write scope) + HF_USERNAME
exec bash                               # pick up the conda shell hook
conda activate llm-sft
```

Separate vLLM env (kept isolated so vLLM's torch pin does not clobber the
training env):

```bash
./utils/setup.sh --mode vllm            # creates the 'vllm' conda env
```

Full flag reference: `./utils/setup.sh --help`.

### b) Train an SFT model

Full-parameter SFT of `Llama-3.1-8B-Instruct` on the AthenaBench-aligned
v3 dataset (`ift_data_2026_04_23_trimmed_v3`) via LlamaFactory +
DeepSpeed ZeRO-3, launched by
[`autotrain/run_abaligned_sft.sh`](autotrain/run_abaligned_sft.sh):

```bash
# Confirm the 37 MB training file is on-host (gitignored; rsync from workstation).
ls -lh SFT/data/ift_data_2026_04_23_trimmed_v3.json

conda activate llm-sft
cd SFT/autotrain
./run_abaligned_sft.sh --dry-run        # inspect the llamafactory-cli command first
./run_abaligned_sft.sh                  # 3 epochs, lr=1e-5, bf16, ZeRO-3
```

On exit 0 the merged full-weight model is pushed to
`hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v3`. ZeRO-3 CPU
offload is auto-enabled on single-GPU hosts; override with `--offload` /
`--no-offload`.

LoRA variant on the v4 dataset (MCQ + TAA dropped; the adapter is merged
at upload time):

```bash
./run_abaligned_sft_v4.sh               # LoRA r=16, 1 epoch
```

Full recipe details, hyperparameter rationale, and troubleshooting are
in [`autotrain/README.md`](autotrain/README.md).

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

## Dataset Preparation

Place your training data inside the `data/` directory. The dataset must be a JSON file in Alpaca format, where each entry contains three fields:

| Field         | Description                                      |
|---------------|--------------------------------------------------|
| `instruction` | System-level instruction or context              |
| `input`       | The user prompt or question                      |
| `output`      | The expected model response                      |

Example entry from `data/ift_data.json`:

```json
{
    "instruction": "You are a cybersecurity expert ...",
    "input": "Assess remediation urgency for KEV ...",
    "output": "KEV Microsoft Windows Common Log File System ..."
}
```

Register the dataset in `data/dataset_info.json` by adding an entry that maps your file's column names to LlamaFactory's expected schema:

```json
"ift_data": {
    "file_name": "ift_data.json",
    "columns": {
        "system": "instruction",
        "prompt": "input",
        "response": "output"
    }
}
```

---

## Training

Training scripts are provided for different model configurations:

| Script | Model | Description |
|--------|-------|-------------|
| `ift_training.sh` | Qwen2.5-14B-Instruct | General training script for Qwen 2.5-14B |
| `ift_training_qwen_2.5_14b.sh` | Qwen2.5-14B-Instruct | Optimized configuration for Qwen 2.5-14B Instruct |
| `ift_training_llama3_8b.sh` | Llama-3.1-8B-Instruct | Training configuration for Llama 3.1-8B Instruct |

Run any script directly:

```bash
bash ift_training.sh
```

This launches SFT with LoRA on Qwen2.5-14B-Instruct using the `ift_data` dataset with a 5% validation split. The full command inside the script is:

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --do_eval True \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ift_data \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 150000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking False \
    --report_to wandb \
    --output_dir saves/Qwen2.5-14B-Instruct/lora/train_${TIMESTAMP} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 18000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all \
    --val_size 0.2 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 4
```

### LoRA Hyperparameters

LoRA (Low-Rank Adaptation) inserts small trainable matrices into the model's existing layers while keeping the base weights frozen. This drastically reduces the number of trainable parameters and GPU memory usage compared to full fine-tuning.

| Parameter      | Value  | Description                                                                                                  |
|----------------|--------|--------------------------------------------------------------------------------------------------------------|
| `lora_rank`    | 64      | Rank of the low-rank decomposition matrices. Lower rank = fewer parameters and less capacity. Common values: 4, 8, 16, 32. Higher rank captures more complex adaptations but increases memory and risks overfitting on small datasets. |
| `lora_alpha`   | 128     | Scaling factor applied to the LoRA output. The effective learning rate for LoRA layers is scaled by `alpha / rank` (here 16 / 8 = 2.0). A ratio of 2:1 (alpha:rank) is a standard starting point. |
| `lora_dropout` | 0.05   | Dropout probability applied to LoRA layers during training. Provides light regularization to reduce overfitting. |
| `lora_target`  | `all`  | Applies LoRA adapters to all linear layers in the model (attention projections, MLP layers, etc.) rather than a subset. This gives the adapter maximum expressiveness. |

### Training Hyperparameters

| Parameter                     | Value          | Description                                                                                      |
|-------------------------------|----------------|--------------------------------------------------------------------------------------------------|
| `learning_rate`               | 5e-05          | Peak learning rate. Decayed via cosine schedule over the training run.                           |
| `lr_scheduler_type`           | `cosine`       | Cosine annealing schedule. Gradually reduces the learning rate to near zero by the end of training. |
| `num_train_epochs`            | 1.0            | Single pass over the dataset. Sufficient for large datasets to avoid overfitting.                |
| `max_samples`                 | 150000         | Upper limit on training samples. If the dataset has more, only the first 150k are used.          |
| `per_device_train_batch_size` | 4              | Samples per GPU per forward pass. Maximum tested on A100 80 GB for this model.                   |
| `gradient_accumulation_steps` | 8              | Accumulates gradients over 8 steps before updating weights. Effective batch size = 4 x 8 = 32.  |
| `warmup_steps`                | 0              | No learning rate warmup. Training begins at the full learning rate immediately.                  |
| `max_grad_norm`               | 1.0            | Gradient clipping threshold to prevent training instability.                                     |
| `optim`                       | `adamw_torch`  | AdamW optimizer (PyTorch implementation) with decoupled weight decay.                            |
| `bf16`                        | True           | Uses bfloat16 mixed precision to reduce memory usage and improve throughput on supported GPUs.   |
| `cutoff_len`                  | 2048           | Maximum token length per training sample. Sequences longer than this are truncated.              |
| `val_size`                    | 0.2            | 20% of data reserved for evaluation.                                                            |
| `eval_strategy` / `eval_steps`| `steps` / 100  | Runs evaluation every 100 training steps.                                                        |
| `save_steps`                  | 100            | Saves a checkpoint every 100 training steps.                                                     |
| `report_to`                   | `wandb`        | Logs all metrics to Weights & Biases for experiment tracking.                                    |

Checkpoints and training artifacts (including loss plots) are saved to:

```
saves/Qwen2.5-14B-Instruct/lora/train/
```

---

## Merging LoRA Adapters

After training, merge the LoRA adapters back into the base model to produce a standalone model. A merge configuration is provided at `examples/merge_lora/qwen2.5_lora_sft.yaml`:

```yaml
### model 
model_name_or_path: Qwen/Qwen2.5-14B-Instruct # Hugging Face path of model
adapter_name_or_path: saves/Qwen2.5-14B-Instruct/lora/train_2026-04-04-21-52-50
template: qwen
finetuning_type: lora

### export
export_dir: models/qwen2.5_14b_sft_lora
export_device: cpu
export_legacy_format: false
```

Update `adapter_name_or_path` to point to the checkpoint directory from your training run, then execute:

```bash
llamafactory-cli export examples/merge_lora/qwen2.5_lora_sft.yaml
```

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

The merged model (including tokenizer, config, and safetensors shards) is written to `models/qwen2.5_14b_sft_lora/`. Do not use a quantized base model or `quantization_bit` when merging.

---

## Local Inference

Run inference on the merged model using `inference_local.py`:

```bash
python inference_local.py
```

The script loads the model from `models/qwen2.5_14b_sft_lora` and generates a response using the chat template. Edit the `messages` list in the script to change the system prompt or user query. Key settings:

- `device_map="auto"` distributes the model across available GPUs.
- `torch_dtype="auto"` uses the model's native precision (bfloat16).
- Generation uses greedy decoding (`do_sample=False`) with a 1024-token limit.

---

## Uploading to Hugging Face

The `upload_to_hf.py` script uploads the merged model to a private Hugging Face repository. Before running, replace the placeholder values:

1. Set your Hugging Face write token in the `login()` call.
2. Set your target `repo_id` (e.g., `your-username/qwen2.5-14b-sft-lora`).

```bash
python upload_to_hf.py
```

The model is uploaded as a **private** repository. Anyone using it for inference will need access to the repository and a valid Hugging Face token.

---

## Notes

- **GPU requirements**: Training Qwen2.5-14B-Instruct with LoRA at batch size 4 requires approximately 80 GB of GPU VRAM (tested on NVIDIA A100 80 GB).
- **CUDA compatibility**: Verify that your PyTorch CUDA version matches your driver's supported CUDA version before starting training. Mismatches cause silent failures or crashes.
- **Checkpoint paths**: The training script appends a timestamp to the output directory (e.g., `train_2026-04-04-21-52-50`). Update the merge YAML accordingly after each training run.
- **Secrets**: Do not commit `upload_to_hf.py` with a real Hugging Face token. Use environment variables or a `.env` file instead.
