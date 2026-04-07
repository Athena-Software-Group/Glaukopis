# Fine-Tuning LLMs with LlamaFactory (SFT / LoRA)

This document describes the end-to-end workflow for supervised fine-tuning (SFT) of a large language model using [LlamaFactory](https://github.com/hiyouga/LlamaFactory), with LoRA adapters. The current configuration targets **Qwen2.5-14B-Instruct** on a custom instruction-following dataset (`ift_data`).

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [CUDA and PyTorch Verification](#cuda-and-pytorch-verification)
5. [Dataset Preparation](#dataset-preparation)
6. [Training](#training)
7. [Merging LoRA Adapters](#merging-lora-adapters)
8. [Local Inference](#local-inference)
9. [Uploading to Hugging Face](#uploading-to-hugging-face)
10. [Notes](#notes)

---

## Environment Setup

A Linux environment is required. Use Anaconda or Miniconda to manage a dedicated Python environment with Python 3.11 or higher.

```bash
conda create -n llm-sft python=3.11 -y
conda activate llm-sft

which python
python --version
```

---

## Prerequisites

- An NVIDIA GPU with sufficient VRAM (A100 80 GB recommended for 14B-parameter models)
- CUDA toolkit compatible with PyTorch (CUDA 12.4 is tested)
- A [Weights & Biases](https://wandb.ai/) API key for experiment tracking
- A [Hugging Face](https://huggingface.co/) token with write access (required for model upload)

---

## Installation

Clone the repository and install LlamaFactory in editable mode along with optional dependencies:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory

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

The training command is provided as a standalone script at `ift_training.sh`. Run it directly:

```bash
bash ift_training.sh
```

This launches SFT with LoRA on Qwen2.5-14B-Instruct using the `ift_data` dataset with a 20% validation split. The full command inside the script is:

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
    --ddp_timeout 180000000 \
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
| `lora_rank`    | 8      | Rank of the low-rank decomposition matrices. Lower rank = fewer parameters and less capacity. Common values: 4, 8, 16, 32. Higher rank captures more complex adaptations but increases memory and risks overfitting on small datasets. |
| `lora_alpha`   | 16     | Scaling factor applied to the LoRA output. The effective learning rate for LoRA layers is scaled by `alpha / rank` (here 16 / 8 = 2.0). A ratio of 2:1 (alpha:rank) is a standard starting point. |
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
