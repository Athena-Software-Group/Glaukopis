# AthenaBench-aligned SFT

Full-parameter SFT launchers driven by LLaMA-Factory + DeepSpeed ZeRO-3
inside the unified `llm-sft` conda env. The canonical SFT recipe is the
four-stage **v20 chain** on `Qwen/Qwen2.5-14B-Instruct`; the chain is
documented end-to-end in [`../SFT_FLOW.md`](../SFT_FLOW.md). Earlier
8B (Llama-3.1) and 14B (Qwen2.5 v7..v19) launchers are retained for
provenance and regression comparison.

**Planned extensions (post-v20-recalibrate sign-off):** the v20 chain
is recipe-portable and will be ported to two additional base models
under the same five-stage topology:

| Planned launcher (v20 family) | Base model | Notes |
|---|---|---|
| `run_sft_llama31_8b_v20_*.sh` | `meta-llama/Llama-3.1-8B-Instruct` | Replaces the v7 8B baseline; reuses v20 data shards verbatim. Smaller effective batch / shorter wall-time. |
| `run_sft_gemma4_31b_v20_*.sh` | `google/gemma-4-31B-it` | 31B SFT, requires tp=2 + ZeRO-3 (no offload on 8×H100 80 GB). Re-uses the `gemma-4-31b-it-vllm` alias for eval. |

Both extensions inherit the same `_v20_*_build/` datasets and sign-off
gate matrix (see `SFT_FLOW.md §4`); only the base-model pointer and
the per-stage `eff_bs` / `cutoff` budgets change. Recipe deltas and
launcher names will land in this directory and `SFT_FLOW.md §3` once
the Qwen-2.5-14B reference chain ships.

> **Historical note.** This directory previously wrapped
> [HuggingFace AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced).
> AutoTrain is unmaintained (last release 0.8.36, 2025-01) and pins
> `transformers==4.48.0`, which conflicts with LLaMA-Factory's `>=4.55.0`.
> The pipeline has been migrated to LLaMA-Factory + DeepSpeed ZeRO-3
> inside the unified `llm-sft` conda env. The directory name is kept
> for continuity with existing logs and model aliases.

## Layout

### v20 chain (canonical, Qwen-2.5-14B-Instruct)

| File | Stage | Base → push target |
|---|---|---|
| `run_sft_qwen25_14b_v20_core.sh` | 1+2 (Phase A re-anchor + Phase B catalog drill) | `Qwen/Qwen2.5-14B-Instruct` → `…/athena-cti-sft-qwen25-14b-v20-core` |
| `run_sft_qwen25_14b_v20_taa.sh` | 3 (TAA Classic refresher) | `…/v20-core` → `…/athena-cti-sft-qwen25-14b-v20-taa` |
| `run_sft_qwen25_14b_v20_cse.sh` | 4 (CSE letter-set drill) | `…/v20-taa` → `…/athena-cti-sft-qwen25-14b-v20-cse` |
| `run_sft_qwen25_14b_v20_recalibrate.sh` | 5 (3-shard interleaved replay) | `…/v20-cse` → `…/athena-cti-sft-qwen25-14b-v20-recalibrate` (**headline**) |
| `run_sft_qwen25_14b_v20_chain.sh` | **3 → 4 → 5 wrapper** (sequential TAA → CSE → Recalibrate; `--include-core` to also run 1+2 first) | gates each stage on the prior stage's HF push being readable |

Per-stage recipes (cutoff, packing, LR, eff_bs, eval/save), wall-time
budgets, and sign-off gates are in [`../SFT_FLOW.md`](../SFT_FLOW.md).

### v21 chain (v18.1 byte-identical replay, Qwen-2.5-14B-Instruct)

v21 is a clean re-derivation of the v18.1 chain on freshly built datasets
(date stamp `2026_05_18`) using the SAME templates, row-count gates,
shuffles, and per-axis Counts as v18.1, and the SAME training
hyperparameters (`lr`, `cutoff`, `eff_bs`, `packing`, `max-samples`,
`save/eval-steps`). Only the dataset filenames and HF push targets
change. Purpose: recover the v18.1 Core optimum and isolate whether the
v19/v20 regression is data-build variance vs recipe drift. See
[`../../tmpl_gen/templates/05182026/v21_plan.txt`](../../tmpl_gen/templates/05182026/v21_plan.txt)
for the replication recipe.

| File | Stage | Base → push target |
|---|---|---|
| `run_sft_qwen25_14b_v21_core.sh` | 1+2 (Phase A re-anchor + Phase B catalog drill) | `Qwen/Qwen2.5-14B-Instruct` → `…/athena-cti-sft-qwen25-14b-v21-core` |
| `run_sft_qwen25_14b_v21_plus_taa.sh` | 3 (TAA Classic refresher) | `…/v21-core` → `…/athena-cti-sft-qwen25-14b-v21-taa` |
| `run_sft_qwen25_14b_v21_final.sh` | 4 (CSE letter-set drill, **headline**) | `…/v21-taa` → `…/athena-cti-sft-qwen25-14b-v21-cse` |

### Legacy 14B launchers (Qwen-2.5-14B, retained for regression)

| File | Purpose |
|---|---|
| `run_sft_qwen25_14b_v19_{core,taa,cse,recalibrate}.sh` | v19 5-stage chain (v20 predecessor; superseded by v20 axis-density rebalance). |
| `run_sft_qwen25_14b_v18*.sh`, `…_v18p1_*.sh`, `…_v18p2*.sh` | v18 / v18.1 / v18.2.x chains (v18.2 was the prior production ship candidate). |
| `run_sft_qwen25_14b_v{11,12,13,14,14_1,16,17}*.sh` | Pre-v18 single-shard and parallel-branch experiments. |
| `run_abaligned_sft_qwen25_14b_v{7,8,8small,81,9,10}.sh` | Pre-v11 experiments (v9 was the canonical 14B before v11; see [v9 recipe](#v9-recipe)). |

### Legacy 8B launchers (Llama-3.1-8B-Instruct, retained for baseline)

| File | Purpose |
|---|---|
| `run_abaligned_sft_v7.sh` | **Llama-3.1-8B v7 baseline.** Full-parameter SFT on `ift_data_2026_04_26_combined_v7` (~181k rows). Pushes to `…-abaligned-v7`. 62.64 % strict F1 on `athena-rms`. See [v7 recipe and results](#v7-recipe-and-results) below. |
| `run_abaligned_sft_v{3,4,5,5_lora,6}.sh` | Pre-v7 8B experiments (v3 trimmed / v4 LoRA / v5 broad / v5-lora / v6 RMS-addendum regression). |

### Post-train helper

| File | Purpose |
|---|---|
| `run_athenabench.sh` | Register the trained+pushed model in `SFT/test/pipelines/models.py` (idempotent), run a smoke test, then the full benchmark sweep. |

## Prerequisites

- Linux box with CUDA. The v20 Qwen-2.5-14B chain is sized for
  8× H100 80 GB (Core / TAA / CSE) and 4× H100 80 GB (Recalibrate);
  the launchers auto-flip ZeRO-3 CPU offload on at < 4 GPUs (Core /
  TAA / CSE) and < 8 GPUs (Recalibrate). The legacy 8B v7 recipe
  needs ≥ 2× 80 GB GPUs for pure ZeRO-3 sharding, or a single 80 GB
  GPU with auto-enabled CPU offload (~100 GB spare CPU RAM; ~30-50%
  throughput cost).
- `llm-sft` conda env created by [`../utils/setup.sh`](../utils/setup.sh).
  Single setup script, single env — no separate `autotrain` env.
- HF credentials in `SFT/.env` (auto-created from `SFT/.env.example`
  on first `setup.sh` run; also honours `SFT/.env.local` and the
  legacy `SFT/autotrain/.env`). Required keys: `HF_TOKEN` (write
  scope), `HF_USERNAME`. Optional: `GITHUB_TOKEN` (private-repo
  pulls), `WANDB_API_KEY` (set `--report-to none` to skip).
- License acceptance on huggingface.co (same account as `HF_TOKEN`)
  for whichever base model the chosen launcher targets:
  `Qwen/Qwen2.5-14B-Instruct` (v20 chain), `meta-llama/Llama-3.1-8B-Instruct`
  (v7 baseline, planned v20 8B extension), or `google/gemma-4-31B-it`
  (planned v20 31B extension).
- The v20 training shards live under `SFT/data/ift_data_2026_05_16_v20_*.json`
  (gitignored). Either rsync them from the build host or regenerate
  in place via the `_v20_*_build/` watchers (see `SFT_FLOW.md §2`):
  ```bash
  rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_05_16_v20_*.json \
        ~/Glaukopis/SFT/data/
  ```
  LLaMA-Factory reads them directly via `SFT/data/dataset_info.json` —
  there is no HF-dataset-repo round-trip.

## Quick start (v20 chain)

```bash
# 1. One-time: env + credentials (on the training host)
cd ~/Glaukopis/SFT
./utils/setup.sh              # creates llm-sft env, bootstraps SFT/.env
$EDITOR .env                  # fill in HF_TOKEN, HF_USERNAME, GITHUB_TOKEN
conda activate llm-sft

# 2. Ensure the v20 training shards are present (gitignored)
ls -lh data/ift_data_2026_05_16_v20_*.json || \
    rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_05_16_v20_*.json data/

# 3a. Stage-by-stage (each launcher writes to SFT/saves/..., pushes
#     the merged checkpoint to HF on exit 0, and becomes the base
#     model for the next stage):
cd autotrain
./run_sft_qwen25_14b_v20_core.sh          # Stage 1+2, ~13h on 8xH100
./run_sft_qwen25_14b_v20_taa.sh           # Stage 3,   ~6-8h
./run_sft_qwen25_14b_v20_cse.sh           # Stage 4,   ~4-6h
./run_sft_qwen25_14b_v20_recalibrate.sh   # Stage 5,   ~95-115min on 4xH100

# 3b. Or run TAA -> CSE -> Recalibrate as a single chained job. Each
#     stage is gated on the prior stage's HF push being readable; on
#     any non-zero exit the chain halts and leaves the partial state
#     intact for restart via --start-stage.
./run_sft_qwen25_14b_v20_chain.sh                                # TAA -> CSE -> Recalibrate
./run_sft_qwen25_14b_v20_chain.sh --include-core                 # full 5-stage chain
./run_sft_qwen25_14b_v20_chain.sh --start-stage cse              # resume from Stage 4
./run_sft_qwen25_14b_v20_chain.sh --start-stage recalibrate      # only Stage 5

# 4. After Stage 5 pushes, benchmark the headline checkpoint
./run_athenabench.sh --alias athena-cti-sft-qwen25-14b-v20-recalibrate
```

Each v20 launcher accepts `--dry-run`, `--repo-id`, `--report-to`,
`--offload`/`--no-offload`, and `--extra "..."`. The chain wrapper
also accepts `--probs`, `--max-samples`, `--lr` (forwarded to
Recalibrate only) and `--skip-readiness-check` (skip the pre-stage
HF probe). See `SFT_FLOW.md §3` for the per-stage recipe table and
`§4` for the sign-off gate matrix.

## Quick start (legacy 8B v7 baseline)

```bash
ls -lh data/ift_data_2026_04_26_combined_v7.json || \
    rsync -avP workstation:Glaukopis/SFT/data/ift_data_2026_04_26_combined_v7.json data/

cd autotrain
./run_abaligned_sft_v7.sh                 # 3 epochs, ZeRO-3, pushes on exit 0
./run_athenabench.sh --alias athena-cti-sft-llama31-8b-abaligned-v7
```

## `run_sft_qwen25_14b_v20_*.sh` (canonical v20 chain)

Four sequential launchers, each a thin wrapper around
`../utils/run_train.sh` with the v20-stage defaults baked in. Common
flags across all four:

- `--finetuning full` (full-parameter SFT, all weights trainable)
- bf16, DeepSpeed ZeRO-3 (GPU-only sharding on ≥ 4 GPUs for
  Core/TAA/CSE and ≥ 8 GPUs for Recalibrate; CPU offload auto-flips
  on below those thresholds). Override with `--offload` /
  `--no-offload`.
- Post-training merge + HF push via
  [`../scripts/upload_to_hf.py`](../scripts/upload_to_hf.py).
- `--dry-run`, `--repo-id USER/NAME`, `--output-dir DIR`,
  `--report-to wandb|none`, and `--extra "--additional --llamafactory --flags"`
  flags are accepted by every launcher.

Per-stage knobs that differ (full table in
[`../SFT_FLOW.md §3`](../SFT_FLOW.md)):

| Stage | Datasets (LF names)                                                                                          | Cutoff | Pack | LR    | eff_bs | Push target                                       |
|-------|--------------------------------------------------------------------------------------------------------------|-------:|------|------:|-------:|---------------------------------------------------|
| Core Phase A | `…_v20_core_a_kb_mcq_taa_soc_cm_ms_yn` + `tulu_3_sft_mixture` + `alpaca_en_demo`                      |   8192 | on   | 1e-5  |     16 | *(intermediate; not pushed)*                      |
| Core Phase B | `…_v20_core_b_rms_ate_vsp_rcm`                                                                        |  16384 | off  | 5e-6  |      8 | `…/athena-cti-sft-qwen25-14b-v20-core`            |
| TAA          | `…_v20_taa`                                                                                           |   4096 | on   | 5e-6  |     16 | `…/athena-cti-sft-qwen25-14b-v20-taa`             |
| CSE          | `…_v20_cse`                                                                                           |   4096 | on   | 5e-6  |     16 | `…/athena-cti-sft-qwen25-14b-v20-cse`             |
| Recalibrate  | `…_v20_core_a` (0.25) + `…_v20_core_b` (0.40) + `…_v20_taa` (0.35), `mix_strategy=interleave_under` |  16384 | off  | 1e-6  |      4 | `…/athena-cti-sft-qwen25-14b-v20-recalibrate`     |

Stage-specific notes:

- **Core (Stage 1+2).** A single launcher runs both phases
  back-to-back; only Phase B's merged checkpoint is pushed. The
  launcher carries a `--skip-eval` knob for ≤ 4-GPU configurations
  (Phase B eval logits `[1, 16384, 152064]` fp32 ≈ 10 GiB per rank
  exceed no-offload ZeRO-3 headroom and OOM at the first
  `eval_steps=400` boundary). Eval is monitoring-only — disabling
  changes zero training-state weights.
- **TAA (Stage 3).** Byte-identical to v18.1+TAA. v14.1 hot-fix
  (`--disable_gradient_checkpointing True`) auto-applies on 8×H100
  only; re-enabled on < 8 GPUs to stay within per-rank budget.
- **CSE (Stage 4).** CyberSOCEval letter-set drill. Carries 5
  `Count: 1500 → 2500` bumps on TI-actor / TI-other templates
  (manifest detail in `tmpl_gen/templates/05162026/v20_plan.txt §3.3`).
- **Recalibrate (Stage 5).** Reverts v19's experimental equal-weight
  probs `0.33/0.33/0.34` to v18.2's asymmetric `0.25/0.40/0.35`
  (Phase A / Phase B / TAA); the asymmetric mix weights Phase B
  highest, which is the recalibrate emphasis v20 needs given v19's
  Phase B-axis regression. Intra-training eval is **disabled**
  (LlamaFactory requires `len(eval_dataset) == len(interleave_probs)`
  and the three natural eval shards would dedupe to two; sign-off is
  via the post-train bench sweep).

```bash
./run_sft_qwen25_14b_v20_core.sh         [--skip-eval] [--repo-id ...] [--report-to wandb|none] [--offload|--no-offload] [--dry-run] [--extra "..."]
./run_sft_qwen25_14b_v20_taa.sh          [--repo-id ...] [--report-to wandb|none] [--offload|--no-offload] [--dry-run] [--extra "..."]
./run_sft_qwen25_14b_v20_cse.sh          [--repo-id ...] [--report-to wandb|none] [--offload|--no-offload] [--dry-run] [--extra "..."]
./run_sft_qwen25_14b_v20_recalibrate.sh  [--repo-id ...] [--report-to wandb|none] [--offload|--no-offload] [--dry-run] [--extra "..."]
```

`--dry-run` prints the `llamafactory-cli train` invocation and the
HF push command without executing anything. `../utils/run_train.sh`
handles timestamped output dirs, git-sha snapshotting into
`train_config.json`, tee'd logs at `train.log`, and the merge +
upload step.

### `run_sft_qwen25_14b_v20_chain.sh` (sequential wrapper)

Runs Stages 3 → 4 → 5 (and optionally Stage 1+2 first) as a single
unattended job. Each stage launches its own per-stage script under
the hood, so per-stage logs, output dirs, and HF pushes are unchanged
— the wrapper only adds chaining and a pre-stage HF-readability
probe so a silent push failure cannot waste the next stage's
compute. Aggregate progress is teed to
`SFT/saves/v20_chain_<ts>/chain.log` and each stage's stdout to
`SFT/saves/v20_chain_<ts>/<stage>.log`.

```bash
./run_sft_qwen25_14b_v20_chain.sh [--start-stage taa|cse|recalibrate]
                                  [--include-core]
                                  [--report-to wandb|none]
                                  [--offload | --no-offload]
                                  [--probs P_A,P_B,P_TAA]     # recalibrate only
                                  [--max-samples N]           # recalibrate only
                                  [--lr LR]                   # recalibrate only
                                  [--skip-readiness-check]
                                  [--dry-run]
```

Behaviour:

- `--start-stage taa` (default) runs TAA → CSE → Recalibrate; assumes
  `…/v20-core` is already on HF (probed first).
- `--start-stage cse` resumes from Stage 4; assumes `…/v20-taa` is on HF.
- `--start-stage recalibrate` resumes from Stage 5; assumes `…/v20-cse` is on HF.
- `--include-core` runs Stage 1+2 first (rare; Core is normally run
  standalone because its wall-time is comparable to the rest of the
  chain combined and you usually want a sign-off review on `v20-core`
  before committing to TAA / CSE / Recalibrate).
- `--skip-readiness-check` disables the per-stage HF probe (useful
  when running fully offline with `--base-model <local-dir>` overrides
  threaded through the per-stage launchers; pass via `--extra` on
  each stage if needed).
- `--dry-run` propagates to every stage; the chain prints each stage's
  `llamafactory-cli` invocation and the post-train HF push command
  without executing anything.

### Porting v20 to Llama-3.1-8B-Instruct and gemma-4-31B-it

The v20 launchers are recipe-portable. The follow-on ports
(`run_sft_llama31_8b_v20_*.sh` and `run_sft_gemma4_31b_v20_*.sh`)
keep the dataset shards, stage topology, LR schedule, packing, and
sign-off gates byte-identical; only the base-model pointer and the
per-stage memory budgets change:

| Knob                           | Qwen-2.5-14B (reference)            | Llama-3.1-8B (planned)              | Gemma-4-31B (planned)               |
|--------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|
| Base model                     | `Qwen/Qwen2.5-14B-Instruct`         | `meta-llama/Llama-3.1-8B-Instruct`  | `google/gemma-4-31B-it`             |
| Phase B `per_device_bs`        | 1                                   | 2                                   | 1                                   |
| Phase B `grad_accum_steps`     | 8 (eff_bs 8 on 8 GPUs)              | 4 (eff_bs 8 on 8 GPUs)              | 16 (eff_bs 8 on 8 GPUs, ZeRO-3)     |
| Recalibrate sizing             | 4× H100 80 GB                       | 2× H100 80 GB                       | 4× H100 80 GB                       |
| Eval transport (alias suffix)  | `…-vllm` (`tp=4` for cse-ti @ 65K)  | `…-vllm` (`tp=1` for cse-ti @ 65K)  | `…-vllm` (`tp=2` for cse-ti @ 49K)  |

Sign-off gates (`SFT_FLOW.md §4`) are **carried verbatim** across all
three bases — gates are recipe-level, not capacity-level. If the
8B port misses a gate by > 5 pp, the residual gap is attributable to
parameter-count headroom rather than the recipe (v7 8B vs v9 14B
baseline gap was 8-12 pp on catalog axes); Gemma-4-31B is expected to
match or exceed Qwen-2.5-14B at every axis since the parameter budget
is larger.

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
| **B** | `ift_data_2026_04_29_json_v8,ift_data_2026_04_29_longctx_v8,ift_data_2026_04_30_v9_rms` | 1 epoch, lr 5e-6, cutoff 16384, packing off, eff. batch 8, `group_by_length` on |

The `ift_data_2026_04_30_v9_rms` dataset is the AB.RMS.* + JS.RMS.*
catalog-drill corpus (~12,158 rows: 10,433 catalog drills + 1,725
JSON-shaped variants). It is built first-class from its own template
manifest -- not filtered out of any prior build artefact -- so the v9
pipeline is fully reproducible from the source manifest alone:

```bash
# 1. Compile the manifest into the per-template JSON the build consumes.
python tmpl_gen/scripts/tmpl_docx2json.py \
    -i tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt \
    -o tmpl_gen/data_generation/Sophia-CTI-Templates-v9_rms.json \
    --count_limit 1500

# 2. Drive iftgen.py per-template (handled by make_dataset.sh).
bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/04302026/Sophia-CTI-Templates-v9_rms.txt \
    _v9_rms_build/triples \
    SFT/data/ift_data_2026_04_30_v9_rms.raw.json \
    10 1500

# 3. Stratified subsample (cap 170 per shortname; AB.RMS / JS.RMS
#    preserved at 100% by PRESERVE_FULL_PREFIXES, so this step is
#    mostly inert here -- run for parity with v8.1 / v8 builds).
python tmpl_gen/scripts/stratified_subsample.py \
    --in  SFT/data/ift_data_2026_04_30_v9_rms.raw.json \
    --out SFT/data/ift_data_2026_04_30_v9_rms.json \
    --cap 170
```

The 21-template manifest is documented in
`tmpl_gen/templates/04302026/README.md` Section 5 and registered in
`SFT/data/dataset_info.json` under the `ift_data_2026_04_30_v9_rms`
key.

### Reproducing v9

```bash
# On the training host (>=2x 80 GB GPUs recommended; 4x for no offload).
ls -lh SFT/data/ift_data_2026_04_26_combined_v7.json \
       SFT/data/ift_data_2026_04_29_json_v8.json \
       SFT/data/ift_data_2026_04_29_longctx_v8.json \
       SFT/data/ift_data_2026_04_30_v9_rms.json

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
