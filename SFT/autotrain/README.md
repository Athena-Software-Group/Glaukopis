# AthenaBench-aligned SFT

Single-command launcher for the AthenaBench-aligned full-parameter SFT of
`meta-llama/Llama-3.1-8B-Instruct` on `ift_data_2026_04_23_trimmed_v3`
(15,625 rows from the `04222026/` trimmed template family: 12 templates
across MCQ/RCM/VSP/ATE/RMS, TAA training coverage deferred pending
re-score of the prior checkpoint with the fixed actor-resolution logic
— see `tmpl_gen/templates/04222026/Sophia-CTI-Templates-AthenaBench-aligned-trimmed.txt`
for the full rationale).

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
| `run_abaligned_sft.sh` | Launch full-parameter SFT on `ift_data_2026_04_23_trimmed_v3` via `../utils/run_train.sh` with DeepSpeed ZeRO-3. Pushes the merged model to `${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v3` on success. |
| `run_abaligned_sft_v4.sh` | LoRA r=16, 1 epoch on `ift_data_2026_04_23_trimmed_v4` (MCQ + TAA dropped). Pushes to `…-abaligned-v4`. |
| `run_abaligned_sft_v5.sh` | Full-parameter SFT on `ift_data_2026_04_24_abaligned_v5` (broad CTI coverage, all six athena tasks). Pushes to `…-abaligned-v5`. |
| `run_abaligned_sft_v5_lora.sh` | LoRA variant of the v5 recipe. Pushes to `…-abaligned-v5-lora`. |
| `run_abaligned_sft_v6.sh` | Full-parameter SFT on the v5 dataset + the v6 RMS-only addendum. **Regressed** athena-rms from 5.88% → 0.00% F1 (truncation + missing terminator + N=3..5-only coverage). Kept for provenance; do not use. |
| `run_abaligned_sft_v7.sh` | Llama-3.1-8B v7 baseline. Full-parameter SFT on `ift_data_2026_04_26_combined_v7` (~181k rows). Pushes to `…-abaligned-v7`. See [v7 recipe and results](#v7-recipe-and-results) below. |
| `run_abaligned_sft_qwen25_14b_v7.sh` | Qwen2.5-14B v7 capacity test. Same dataset and recipe as the 8B v7 launcher, base swapped to `Qwen/Qwen2.5-14B-Instruct`. |
| `run_abaligned_sft_qwen25_14b_v8.sh` | Two-phase Qwen2.5-14B v8. Phase A re-anchors on the v7 corpus; Phase B specialises on `ift_data_2026_04_29_json_v8` + `ift_data_2026_04_29_longctx_v8` at cutoff 16384. Pushes to `…-abaligned-v8`. **Superseded** at 14B by v9 because Phase B dropped the AB.RMS catalog drills (RMS catalog-collapse regression). |
| `run_abaligned_sft_qwen25_14b_v81.sh` | Single-pass Qwen2.5-14B v8.1. Trains on `ift_data_2026_04_30_v81` (~42k rows, cap=170 stratified subsample with AB.RMS / JS.RMS preserved at 100%). Recovered RMS but regressed CKT/ATE/RCM/CyberMetric because the cap=170 rule starved every other catalog family (V/W/X/S/P/M cut to 0.09–0.16× of v7). **Superseded** by v9. |
| `run_abaligned_sft_qwen25_14b_v9.sh` | **Current canonical 14B recipe.** Two-phase: Phase A identical to v8 (broad-knowledge re-anchor), Phase B grafts the v8.1 `AB.RMS.*` + `JS.RMS.*` slice (~12.2k rows) onto v8's JSON + long-context mix. Pushes to `…-abaligned-v9`. See [v9 recipe](#v9-recipe) below. |
| `run_athenabench.sh` | Register the trained+pushed model in `SFT/test/pipelines/models.py` (idempotent), run a smoke test, then the full 6-task sweep. |

## Prerequisites

- Linux box with CUDA. Recommended: ≥ 2× 80 GB GPUs (H100/H200/A100-80G) for
  pure ZeRO-3 sharding. Single-GPU hosts are supported via automatic ZeRO-3
  CPU offload (needs ~100 GB spare CPU RAM; costs ~30–50% throughput).
- `llm-sft` conda env created by [`../utils/setup.sh`](../utils/setup.sh).
  Single setup script, single env — no separate `autotrain` env.
- HF credentials in `SFT/.env` (auto-created from `SFT/.env.example` on
  first `setup.sh` run; also honours `SFT/.env.local` and the legacy
  `SFT/autotrain/.env`). Required keys: `HF_TOKEN` (write scope),
  `HF_USERNAME`.
- License acceptance for `meta-llama/Llama-3.1-8B-Instruct` on huggingface.co
  using the same account whose token you're using.
- The training file `SFT/data/ift_data_2026_04_23_trimmed_v3.json` (37 MB,
  gitignored) must be present on the training host. Transfer it out-of-band
  from the workstation where it was generated:
  ```bash
  rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_04_23_trimmed_v3.json \
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

# 2. Ensure the training data is present (37 MB, not in git)
ls -lh data/ift_data_2026_04_23_trimmed_v3.json || \
    rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_04_23_trimmed_v3.json data/

# 3. Launch full-parameter SFT (writes to SFT/saves/..., pushes to HF on exit 0)
cd autotrain
./run_abaligned_sft.sh

# 4. After training pushes the model, benchmark it
./run_athenabench.sh --alias athena-cti-sft-llama31-8b-abaligned-v3
```

## `run_abaligned_sft.sh`

Thin wrapper around `../utils/run_train.sh` with the ab-aligned defaults
baked in:

- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Dataset: `ift_data_2026_04_23_trimmed_v3,alpaca_en_demo` (the alpaca
  mix-in is anti-forgetting regularization; see `alpaca_en_demo` in
  `../data/dataset_info.json`)
- `--finetuning full` (full-parameter SFT, all weights trainable)
- 3 epochs, lr=1e-5 cosine, 5 % warmup, bf16
- `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`
  → effective batch 16 on a 2-GPU node
- `cutoff_len=2048`, `save_steps=500`, `save_total_limit=3`
- DeepSpeed ZeRO-3 sharding. Config auto-selected by GPU count:
  - ≥ 2 GPUs: `examples/deepspeed/ds_z3_config.json` (GPU-only sharding)
  - 1 GPU:   `examples/deepspeed/ds_z3_offload_config.json` (optimizer + params offloaded to CPU)
  Override with `--offload` (force CPU offload) or `--no-offload` (force
  GPU-only; will OOM on < 2× 80 GB for 8B full SFT).
- `--report-to wandb` (override with `--report-to none`)
- Post-training HF push to `${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v3`

```bash
./run_abaligned_sft.sh [--repo-id USER/NAME] [--output-dir DIR]
                       [--report-to wandb|none]
                       [--offload | --no-offload]
                       [--dry-run]
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
2. Patches `SFT/test/pipelines/models.py` with the new alias
   (idempotent: exits 0 if the alias already maps to the same repo, fails
   loudly if it maps to a different one).
3. Activates the `ctibench` conda env.
4. Runs a 2-row smoke test on `athena-mcq` (version 99, disposable).
5. If the smoke test passes, runs the full 6-task benchmark sweep via
   [`../test/utils/run_benchmark.sh`](../test/utils/run_benchmark.sh).

```bash
./run_athenabench.sh [--repo-id USER/NAME] [--alias NAME]
                     [--env-name NAME] [--smoke-only]
                     [--rows N] [--batch N]
                     [--tasks "athena-mcq athena-rcm ..."]
```

## v7 recipe and results

`run_abaligned_sft_v7.sh` is the current canonical full-parameter SFT
recipe. It supersedes v6, which regressed `athena-rms` from the v0
baseline of 5.88% strict F1 to 0.00% due to three structural bugs in
the RMS-only addendum templates and launcher (output truncation at
`cutoff_len=2048`, missing `Answer:` terminator, and N=3..5-only
cardinality coverage that mismatched the benchmark's N=1..8
distribution). v7 fixes all three.

### Training configuration

| Setting | Value | Notes vs v6 |
|---|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` | unchanged |
| Method | full-parameter SFT (DeepSpeed ZeRO-3) | unchanged |
| Dataset | `ift_data_2026_04_26_combined_v7` (~181k rows: v5 broad coverage + v7 RMS addendum) + `alpaca_en_demo` mix-in | merged into a single file (was v5 + v6-addendum split) |
| Epochs | 3 | unchanged |
| Learning rate | 1e-5 cosine, 5 % warmup | unchanged |
| Precision | bf16 | unchanged |
| `cutoff_len` | **4096** | doubled from 2048 — v6 truncated ~80 % of RMS rows mid-explanation |
| Effective batch | 16 | unchanged |
| Per-device batch / grad-accum | 1 / 8 (≤ 3 GPUs) or 2 / 2 (≥ 4 GPUs) | halved per-device + doubled grad-accum to absorb the 2× cutoff growth in activation memory |
| Packing | on | unchanged |
| `save_steps` / `eval_steps` | 200 | halved (packed-sequence count roughly halves at 4096) |
| Pushed repo | `${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v7` | new repo |

Template-side changes (in
[`tmpl_gen/templates/04262026/Sophia-CTI-Templates-AthenaBench-abaligned-v7.txt`](../../tmpl_gen/templates/04262026/Sophia-CTI-Templates-AthenaBench-abaligned-v7.txt)):

- **Variable-N** RMS templates `RMS.3a..3h` covering N=1..8 (matches
  the benchmark mass distribution; v6 collapsed to N=1 in 98.4 % of
  responses because it only saw N=3..5 in training).
- **Per-mitigation clauses reduced** to `{coa.mitre_id} ({coa.name})`
  (no inline `{coa.description}`); estimated output stays under
  ~600 chars at N=8.
- **Mandatory `Answer:` terminator** — every variable-N template (and
  RMS.6) ends with a literal `Answer: M####, M####, ...` final line,
  matching the AthenaBench RMS post-processor's extraction regex.
- Instruction text aligned verbatim with the benchmark prompt
  ("Return exactly N mitigation IDs ...").

### Validated AthenaBench results (suite=athena, version=1)

Run on a single H100 via vLLM (`utils/serve_and_bench.sh
athena-cti-sft-llama31-8b-abaligned-v7-vllm --tp 1 --max-len 4096
--port 8000 -- --suite athena --version 1 --batch 128 --overwrite
--yes`); end-to-end wall clock 1m51s.

| Task | Rows | Metric | v7 | v0 baseline | v6 |
|---|---:|---|---:|---:|---:|
| `athena-mcq` | 3000 | accuracy | **57.60 %** | ~50 % | ~50 % |
| `athena-rcm` | 2000 | accuracy | **65.80 %** | ~55 % | ~60 % |
| `athena-vsp` | 2000 | accuracy (MAD 1.92) | **75.02 %** | ~70 % | ~70 % |
| `athena-ate` | 500 | accuracy | **50.00 %** | ~45 % | ~45 % |
| `athena-taa` | 100 | combined accuracy (strict 17.0 % / plausible 82.0 %) | **49.50 %** | low double-digits strict | low double-digits strict |
| `athena-rms` | 500 | strict F1 (plausible 64.32 %) | **62.64 %** | **5.88 %** | **0.00 %** |

The RMS recovery (`+56.76 pp` strict F1 over the v0 baseline,
`+62.64 pp` over the v6 regression) is the headline result and
confirms all three template/launcher fixes were necessary.

### Reproducing v7

```bash
# On the training host (≥ 2× 80 GB GPUs recommended).
ls -lh SFT/data/ift_data_2026_04_26_combined_v7.json   # ~193 MB, gitignored

conda activate llm-sft
cd SFT/autotrain
./run_abaligned_sft_v7.sh --dry-run    # inspect the llamafactory-cli command first
./run_abaligned_sft_v7.sh              # 3 epochs, ZeRO-3, pushes to HF on exit 0
```

On exit 0 the merged full-weight checkpoint is at
`hf://${HF_USERNAME}/athena-cti-sft-llama31-8b-abaligned-v7`. The
benchmark sweep above can then be run from any host with vLLM and a
single H100.

## v9 recipe

`run_abaligned_sft_qwen25_14b_v9.sh` is the current canonical 14B
launcher. It supersedes both `run_abaligned_sft_qwen25_14b_v8.sh`
(RMS catalog-collapse in Phase B) and
`run_abaligned_sft_qwen25_14b_v81.sh` (broad regression on
CKT/ATE/RCM/CyberMetric driven by the cap=170 stratified subsample).

### Why v9 exists

The v8.1 14B sweep recovered athena-rms (+8.9 pp F1 over v8) but
regressed the broad knowledge tasks vs the v8 14B baseline:

| Task | v8 14B | v8.1 14B | Δ |
|---|---:|---:|---:|
| CKT | 77.6 | 63.2 | **−14.4** |
| ATE | 47.6 | 30.2 | **−17.4** |
| RCM | 64.5 | 54.0 | **−10.5** |
| CyberMetric (avg 2k/10k) | 88.2 | 83.1 | **−5.1** |
| RMS (F1) | 36.0 | 44.9 | +8.9 |

Root cause was traced to `tmpl_gen/scripts/stratified_subsample.py
--cap 170` with `AB.RMS.*` / `JS.RMS.*` hard-coded as 100%-retained.
Every other catalog family was capped at 170 rows/shortname, cutting
V/W/X/S/P/M from 9–35k rows down to ~1–4k each (0.09–0.16× of v7).
Combined with Tulu/Alpaca dilution, v8.1 saw ~85k CTI example-passes
vs v7's ~540k and v8's ~262k — a 3–6× compute deficit on exactly the
knowledge surface that drives CKT/ATE/RCM/CyberMetric.

### Phase shape (v8 broad-knowledge baseline + v8.1 RMS slice)

| Phase | Datasets | Recipe |
|---|---|---|
| **A** | `ift_data_2026_04_26_combined_v7,tulu_3_sft_mixture,alpaca_en_demo` | 1 epoch, lr 1e-5, cutoff 4096, packing on, eff. batch 16 (identical to v8 Phase A) |
| **B** | `ift_data_2026_04_29_json_v8,ift_data_2026_04_29_longctx_v8,ift_data_2026_04_30_v81_rms` | 1 epoch, lr 5e-6, cutoff 16384, packing off, eff. batch 8, `group_by_length` on |

The `ift_data_2026_04_30_v81_rms` dataset is the
`AB.RMS.*` + `JS.RMS.*` slice of the v8.1 corpus (12,158 rows: 10,433
catalog drills + 1,725 JSON-shaped variants). It is built as a
one-shot from the v8.1 file:

```bash
python -c "
import json
d = json.load(open('SFT/data/ift_data_2026_04_30_v81.json'))
keep = [r for r in d if (r.get('shortname') or '').startswith(('AB.RMS.', 'JS.RMS.'))]
json.dump(keep, open('SFT/data/ift_data_2026_04_30_v81_rms.json', 'w'), ensure_ascii=False)
"
```

Registered in `SFT/data/dataset_info.json` under the
`ift_data_2026_04_30_v81_rms` key.

### Reproducing v9

```bash
# On the training host (>=2x 80 GB GPUs recommended; 4x for no offload).
ls -lh SFT/data/ift_data_2026_04_26_combined_v7.json \
       SFT/data/ift_data_2026_04_29_json_v8.json \
       SFT/data/ift_data_2026_04_29_longctx_v8.json \
       SFT/data/ift_data_2026_04_30_v81_rms.json

conda activate llm-sft
cd SFT/autotrain
./run_abaligned_sft_qwen25_14b_v9.sh --dry-run    # inspect both phases
./run_abaligned_sft_qwen25_14b_v9.sh              # both phases, push to HF on exit 0
```

On exit 0 the merged Phase B checkpoint is at
`hf://${HF_USERNAME}/athena-cti-sft-qwen25-14b-abaligned-v9`. Phase A
is intermediate; only Phase B is pushed.

## Troubleshooting

- **`llamafactory-cli: command not found`** — activate the env first:
  `conda activate llm-sft`.
- **`training dataset not found: .../ift_data_2026_04_23_trimmed_v3.json`**
  — the 37 MB dataset is gitignored; transfer it via rsync (see Prerequisites).
- **401 on base-model download** — Llama-3.1-8B-Instruct is gated; accept
  the license on huggingface.co using the same account whose token you're
  using, then retry.
- **OOM at step 0 on a single GPU** — full SFT of 8B with AdamW (fp32 m+v)
  needs ~96 GB of GPU RAM, which exceeds 1× 80 GB. The launcher auto-enables
  CPU offload on single-GPU hosts; if you overrode with `--no-offload`,
  drop that flag. If OOM persists even with offload, lower
  `per_device_train_batch_size` (`--extra "--per_device_train_batch_size 1 --gradient_accumulation_steps 8"`)
  or reduce `cutoff_len` (`--extra "--cutoff_len 1536"`).
- **OOM at step 0 on multi-GPU** — reduce batch size per the previous bullet,
  or fall back to LoRA via `../utils/run_train.sh` directly
  (`--finetuning lora`, which is the default).
- **Run finishes but no repo on the Hub** — `HF_TOKEN` is read-only or
  missing; fix it in `SFT/.env` and rerun `upload_to_hf.py --merged-dir <output_dir>`
  manually (training output is preserved under `SFT/saves/`).
- **Alias conflict in `run_athenabench.sh`** — the registry already has a
  different repo under that alias; pass `--alias <unique-name>` or edit
  `SFT/test/pipelines/models.py` manually.
