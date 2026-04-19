# SFT via HuggingFace AutoTrain

End-to-end automation for fine-tuning a base LLM (default
`meta-llama/Llama-3.1-8B-Instruct`) on the internal instruction-following
dataset (`SFT/data/ift_data.json`) via
[HuggingFace AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced),
pushing the resulting full-weight model to the HF Hub, and benchmarking it
against [`athena_bench`](../../athena_bench).

This pipeline is intentionally kept separate from the LLaMA-Factory flow
under [`SFT/utils/`](../utils/): use AutoTrain when you want a hosted-friendly,
low-config full-parameter SFT; use LLaMA-Factory when you need fine-grained
control (LoRA, DPO, GPTQ merge, etc.).

## Layout

| File | Purpose |
|---|---|
| `setup.sh` | Create an isolated `autotrain` conda env and install `autotrain-advanced`. |
| `prepare_dataset.sh` | Convert `ift_data.json` into a chat-templated JSONL and upload it as an HF dataset repo. |
| `autotrain_llama3_8b_sft.yml` | AutoTrain config — full-parameter SFT of Llama-3.1-8B-Instruct (bf16, cosine, 3 epochs). |
| `train.sh` | Launch `autotrain --config <yaml>` with logging, optional `--nohup` detach, and optional `--cuda-devices` pinning. |
| `run_athenabench.sh` | Register the trained model in `athena_bench/pipelines/models.py` (idempotent), run a smoke test, then the full sweep. |

## Prerequisites

- A Linux box with CUDA and **≥ 80 GB effective VRAM** for the default
  full-SFT config (1× A100-80G, 1× H100-80G, or 2× A100-40G sharded).
  Drop to `peft: true` + `quantization: int4` in the YAML if you only have
  a 24 GB consumer GPU.
- Conda (`setup.sh` assumes `conda` is on `PATH`; run
  [`SFT/utils/setup.sh`](../utils/setup.sh) or install Miniconda first).
- HF credentials in `SFT/autotrain/.env` (a gitignored file auto-created
  from `.env.example` on first `setup.sh` run). You need a write-scope
  token for `HF_TOKEN` and your namespace for `HF_USERNAME`. All three
  runtime scripts (`prepare_dataset.sh`, `train.sh`, `run_athenabench.sh`)
  auto-source this file — no manual `export` required.
- License acceptance for the base model
  (`meta-llama/Llama-3.1-8B-Instruct` is gated — visit the model page once
  with the same account whose token you're using).

## Quick start

```bash
# 1. Create the autotrain conda env (isolated from llm-sft).
#    Also copies .env.example -> .env on first run so step 2 has a file to edit.
./setup.sh
conda activate autotrain

# 2. Fill in HF credentials ONCE in .env (gitignored).
#    Replace HF_TOKEN=hf_xxx_replace_me and HF_USERNAME=your-hf-username:
$EDITOR SFT/autotrain/.env
chmod 600 SFT/autotrain/.env        # recommended
# Every script below automatically sources this file; no 'export' needed.

# 3. Convert + upload the training dataset -> hf://datasets/${HF_USERNAME}/athena-ift
./prepare_dataset.sh

# 4. Train (foreground); on success AutoTrain pushes the model to
#    hf://${HF_USERNAME}/llama3.1-8b-athena-ift
./train.sh
# or, to detach:
./train.sh --nohup

# 5. Register the model in athena_bench and benchmark it
./run_athenabench.sh
```

## Script reference

### `setup.sh`

Creates `conda env autotrain` (python 3.11) and installs
`autotrain-advanced==0.8.36` (the latest stable on PyPI). AutoTrain pins
its entire dependency tree exactly (`transformers==4.48.0`,
`huggingface-hub==0.27.0`, `accelerate==1.2.1`, …); the script installs
it in a single pass and does not upgrade anything afterwards. Safe to
rerun; pass `--recreate` to nuke and rebuild the env from scratch.

```bash
./setup.sh [--env-name NAME] [--python VERSION]
           [--autotrain-version SPEC]
           [--recreate] [--no-conda-init]
```

### `prepare_dataset.sh`

Applies the base model's chat template to each row of `ift_data.json`
(`instruction` → system, `input` → user, `output` → assistant) and writes
the result as a single-column JSONL with a `text` field — the shape
AutoTrain expects when `chat_template: null`. Then it creates (or reuses)
an HF dataset repo and uploads `<split>.jsonl`.

```bash
./prepare_dataset.sh [--src PATH] [--base-model HF_ID]
                     [--dataset-repo USER/NAME] [--split-name train]
                     [--private] [--overwrite]
```

### `autotrain_llama3_8b_sft.yml`

Default training config. Key settings:

- `peft: false`, `quantization: null` — full-parameter SFT, all weights trainable.
- `block_size: 2048`, `model_max_length: 8192`, `epochs: 3`.
- `batch_size: 1`, `gradient_accumulation: 8` (effective batch 8).
- `lr: 1.0e-5`, cosine scheduler, 5 % warmup.
- `mixed_precision: bf16`, `gradient_checkpointing: true`.
- `hub.push_to_hub: true` — pushes to `${HF_USERNAME}/llama3.1-8b-athena-ift`.

To run a different base model or dataset, copy the file and point `train.sh`
at it via `--config`.

### `train.sh`

Thin wrapper around `autotrain --config <yaml>`. Logs to
`<project_name>_<UTC-timestamp>.log` next to the script.

```bash
./train.sh [--config PATH] [--cuda-devices LIST] [--nohup]
```

### `run_athenabench.sh`

1. Verifies the pushed HF model repo is readable.
2. Patches `athena_bench/pipelines/models.py` with the new alias
   (idempotent: exits 0 if the alias already maps to the same repo, fails
   loudly if it maps to a different one).
3. Activates the `ctibench` conda env.
4. Runs a 2-row smoke test on `athena-mcq` (version 99, disposable).
5. If the smoke test passes, runs the full 6-task benchmark sweep via
   [`athena_bench/utils/run_benchmark.sh`](../../athena_bench/utils/run_benchmark.sh).

```bash
./run_athenabench.sh [--repo-id USER/NAME] [--alias NAME]
                     [--env-name NAME] [--smoke-only]
                     [--rows N] [--batch N]
                     [--tasks "athena-mcq athena-rcm ..."]
```

## Troubleshooting

- **`autotrain: command not found`** — You forgot `conda activate autotrain`
  after `setup.sh`.
- **401 on tokenizer download in `prepare_dataset.sh`** — The base model is
  gated; accept its license on huggingface.co using the same account whose
  token you're using, then retry.
- **OOM at step 0** — You're likely on < 80 GB VRAM. Either shard across
  GPUs (set `--cuda-devices "0,1"` and let Accelerate split) or switch the
  YAML to LoRA + int4 (`peft: true`, `quantization: int4`,
  `target_modules: all-linear`).
- **Run finishes but no repo on the Hub** — `HF_TOKEN` is read-only or the
  `hub:` block in the YAML lost its env substitution; re-export `HF_TOKEN`
  with write scope and rerun `train.sh`.
- **Alias conflict in `run_athenabench.sh`** — The registry already has a
  different repo under that alias; pass `--alias <unique-name>` or edit
  `athena_bench/pipelines/models.py` manually.
