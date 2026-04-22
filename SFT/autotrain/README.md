# AthenaBench-aligned SFT

Single-command launcher for the AthenaBench-aligned full-parameter SFT of
`meta-llama/Llama-3.1-8B-Instruct` on `ift_data_2026_04_22` (138,343 rows
from the `04222026/` template family: native MCQ shuffling, Description→ID
ATE direction, full TAA/VSP/RCM/RMS coverage).

> **Historical note.** This directory previously wrapped
> [HuggingFace AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced).
> AutoTrain is unmaintained (last release 0.8.36, 2025-01) and pins
> `transformers==4.48.0`, which conflicts with LLaMA-Factory's `>=4.55.0`.
> The pipeline has been migrated to LLaMA-Factory + DeepSpeed ZeRO-3
> inside the unified `llm-sft` conda env. The directory name is kept for
> continuity with existing logs and model aliases.

## Layout

| File | Purpose |
|---|---|
| `run_abaligned_sft.sh` | Launch full-parameter SFT on `ift_data_2026_04_22` via `../utils/run_train.sh` with DeepSpeed ZeRO-3. Pushes the merged model to `${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned` on success. |
| `run_athenabench.sh` | Register the trained+pushed model in `athena_bench/pipelines/models.py` (idempotent), run a smoke test, then the full 6-task sweep. |

## Prerequisites

- Linux box with CUDA and ≥ 2× 80 GB GPUs (H100/H200/A100-80G). Full SFT of
  an 8B model in bf16 needs DeepSpeed ZeRO-3 sharding across ≥ 2 GPUs.
- `llm-sft` conda env created by [`../utils/setup.sh`](../utils/setup.sh).
  Single setup script, single env — no separate `autotrain` env.
- HF credentials in `SFT/.env` (auto-created from `SFT/.env.example` on
  first `setup.sh` run; also honours `SFT/.env.local` and the legacy
  `SFT/autotrain/.env`). Required keys: `HF_TOKEN` (write scope),
  `HF_USERNAME`.
- License acceptance for `meta-llama/Llama-3.1-8B-Instruct` on huggingface.co
  using the same account whose token you're using.
- The training file `SFT/data/ift_data_2026_04_22.json` (144 MB, gitignored)
  must be present on the training host. Transfer it out-of-band from the
  workstation where it was generated:
  ```bash
  rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_04_22.json \
        ~/Glaukopis/SFT/data/
  ```
  LLaMA-Factory reads it directly via `SFT/data/dataset_info.json` — there
  is no HF-dataset-repo round-trip.

## Quick start

```bash
# 1. One-time: env + credentials (on the training host)
cd ~/Glaukopis/SFT
./utils/setup.sh              # creates llm-sft env, bootstraps SFT/.env
$EDITOR .env                  # fill in HF_TOKEN, HF_USERNAME
conda activate llm-sft

# 2. Ensure the training data is present (144 MB, not in git)
ls -lh data/ift_data_2026_04_22.json || \
    rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_04_22.json data/

# 3. Launch full-parameter SFT (writes to SFT/saves/..., pushes to HF on exit 0)
cd autotrain
./run_abaligned_sft.sh

# 4. After training pushes the model, benchmark it
./run_athenabench.sh --alias athena-cti-sft-llama31-8b-abaligned
```

## `run_abaligned_sft.sh`

Thin wrapper around `../utils/run_train.sh` with the ab-aligned defaults
baked in:

- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Dataset: `ift_data_2026_04_22,alpaca_en_demo` (the alpaca mix-in is
  anti-forgetting regularization; see `alpaca_en_demo` in
  `../data/dataset_info.json`)
- `--finetuning full` (full-parameter SFT, all weights trainable)
- 3 epochs, lr=1e-5 cosine, 5 % warmup, bf16
- `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`
  → effective batch 16 on a 2-GPU node
- `cutoff_len=2048`, `save_steps=500`, `save_total_limit=3`
- `--deepspeed examples/deepspeed/ds_z3_config.json` (ZeRO-3 sharding,
  required to fit the 8B model in bf16 on 2× 80 GB)
- `--report-to wandb` (override with `--report-to none`)
- Post-training HF push to `${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned`

```bash
./run_abaligned_sft.sh [--repo-id USER/NAME] [--output-dir DIR]
                       [--report-to wandb|none] [--dry-run]
                       [--extra "--additional --llamafactory --flags"]
```

`--dry-run` prints the `llamafactory-cli train` invocation and the HF push
command without executing anything.

The underlying launcher (`../utils/run_train.sh`) handles timestamped
output dirs, git-sha snapshotting into `train_config.json`, tee'd logs at
`train.log`, and the merge-free upload (full SFT saves a merged model
directly, so `upload_to_hf.py --merged-dir` is used instead of the LoRA
`--adapter-dir` path).

## `run_athenabench.sh`

1. Verifies the pushed HF model repo is readable.
2. Patches `athena_bench/pipelines/models.py` with the new alias
   (idempotent: exits 0 if the alias already maps to the same repo, fails
   loudly if it maps to a different one).
3. Activates the `ctibench` conda env.
4. Runs a 2-row smoke test on `athena-mcq` (version 99, disposable).
5. If the smoke test passes, runs the full 6-task benchmark sweep via
   [`../../athena_bench/utils/run_benchmark.sh`](../../athena_bench/utils/run_benchmark.sh).

```bash
./run_athenabench.sh [--repo-id USER/NAME] [--alias NAME]
                     [--env-name NAME] [--smoke-only]
                     [--rows N] [--batch N]
                     [--tasks "athena-mcq athena-rcm ..."]
```

## Troubleshooting

- **`llamafactory-cli: command not found`** — activate the env first:
  `conda activate llm-sft`.
- **`training dataset not found: .../ift_data_2026_04_22.json`** — the 144
  MB dataset is gitignored; transfer it via rsync (see Prerequisites).
- **401 on base-model download** — Llama-3.1-8B-Instruct is gated; accept
  the license on huggingface.co using the same account whose token you're
  using, then retry.
- **OOM at step 0** — you're on < 2× 80 GB. Options: reduce `cutoff_len`
  (e.g. `--extra "--cutoff_len 1536"`), lower `per_device_train_batch_size`
  (`--extra "--per_device_train_batch_size 1 --gradient_accumulation_steps 8"`),
  or fall back to LoRA via `../utils/run_train.sh` directly
  (`--finetuning lora`, which is the default).
- **Run finishes but no repo on the Hub** — `HF_TOKEN` is read-only or
  missing; fix it in `SFT/.env` and rerun `upload_to_hf.py --merged-dir <output_dir>`
  manually (training output is preserved under `SFT/saves/`).
- **Alias conflict in `run_athenabench.sh`** — the registry already has a
  different repo under that alias; pass `--alias <unique-name>` or edit
  `athena_bench/pipelines/models.py` manually.
