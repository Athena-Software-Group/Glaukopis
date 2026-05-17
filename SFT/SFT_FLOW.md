# SFT End-to-End Flow (v20 final)

This document is the single-source description of the Glaukopis SFT
pipeline as of the **v20** vintage (May 16, 2026), which is the final
SFT variant. It complements [`README.md`](README.md) (quick-start +
environment) and [`autotrain/README.md`](autotrain/README.md)
(per-launcher flag reference) by describing how the three modules
fit together — **templates → train → test** — and what each v20
artifact does.

```
tmpl_gen/                                 SFT/                                    SFT/test/
+--------------------------+               +-----------------------------+         +---------------------+
|  Sophia CTI templates    | --(IFT JSON)-> | LlamaFactory full-param SFT |--push--> | AthenaBench / CM /  |
|  + Neo4j CTI graph       |               |  Qwen-2.5-14B-Instruct      |  to HF  | CyberSOCEval sweeps |
|  (05162026/ vintage)     |               |  4-stage v20 chain          |         | (vLLM / HF Router)  |
+--------------------------+               +-----------------------------+         +---------------------+
```

The chain produces four cumulative HF checkpoints
(`v20-core → v20-taa → v20-cse → v20-recalibrate`); the last is the
headline release.

---

## 1. Templates — `tmpl_gen/` and the v20 vintage

`tmpl_gen` (see [`../tmpl_gen/README.md`](../tmpl_gen/README.md)) is
the IFT triple generator. Templates are plain-text manifests of CTI
prompts whose placeholders (`{var:NodeType.property}`,
`{var1.rel>TargetType.property}`, `{force …}`, `<* … *>`) are
expanded against an `athena-cti-db` Neo4j instance carrying MITRE
ATT&CK, CAPEC, CWE, CVE, CISA KEV, FIRST EPSS, and MITRE ENGAGE
nodes. The end-to-end build is
[`tmpl_gen/data_generation/make_dataset.sh`](../tmpl_gen/data_generation/make_dataset.sh),
which wraps three steps:

1. `docx2json.sh` — extract templates from `.docx` to JSON (or take
   a manifest `.txt`/`.json` directly).
2. `tmpl2triples.sh` — drive `iftgen.py` per template against the
   CTI graph.
3. `triples2alpaca.sh` — merge per-template triple JSON files into
   one Alpaca-format dataset
   (`instruction` → `system`, `input` → `prompt`, `output` → `response`).

### v20 template vintage — `tmpl_gen/templates/05162026/`

The v20 vintage directory is self-contained — every load-bearing
artifact ships in-repo.

| File | Role |
|---|---|
| `README.md` | Vintage overview (read first for orientation) |
| `v20_plan.txt` | Master plan: §1 motivation, §2 deltas vs v19, §3 row-count plan, §4 training recipe, §5 sign-off gates, §6 falsification |
| `Sophia-CTI-Templates-v20_core.txt` | Core manifest (Stages 1+2 of the chain). Body byte-identical to v19_core; v20 raises the `count_max` ceiling 3500 → 5500 at `make_dataset.sh` invocation time to lift ATE / RCM density |
| `Sophia-CTI-Templates-v20_taa.txt` | TAA Classic manifest (Stage 3). Carried verbatim from v19_taa |
| `Sophia-CTI-Templates-v20_cse.txt` | CyberSOCEval letter-set manifest (Stage 4). 5 `Count: 1500` directives bumped to `Count: 2500` on `JS.CSE.TI.{GRP.2, GRP.3, MAL.2, MAL.3, NEG.1}` to lift the TI-actor / TI-other axes |
| `v20_row_count_gate.json` | Per-axis `REJECT_IF_BELOW` thresholds for the Core shard (ATE floor 12 500 → 18 500; RCM floor 9 000 → 40 000) |
| `v20_taa_row_count_gate.json` | TAA per-axis floors (carried from v19_taa) |
| `v20_cse_row_count_gate.json` | CSE per-axis floors (TI-actor 5 000 → 8 500; TI-other 1 400 → 2 400) |

### Sibling build trees (repo root)

Three `_v20_*_build/` directories drive the post-`make_dataset.sh`
pipeline. Each contains:

- `watcher.sh` — runs substrate gate → seed-provenance → generator
  merges → dedup → row-count gate → licence gate → stratified shuffle
  → val/train split → (Core only) two-shard phase split.
- `build_val_slice.py` — per-axis val/train splitter (50 rows /
  shortname, seed=42).
- `_neo4j_check.py` — Phase 0 substrate validator (Core only).
- `letter_balance_gate.py` — CSE letter-tuple distribution gate
  (CSE only).
- `validate_corpus.py` — cross-stage end-to-end validator (Core
  only; spot-checks all three shards).

### v20 dataset shards (final outputs)

All seven shards land in [`SFT/data/`](data/) and are pre-registered
in [`SFT/data/dataset_info.json`](data/dataset_info.json) with the
Alpaca-column mapping LlamaFactory expects.

| Shard | Used by stage | Rough size |
|---|---|---:|
| `ift_data_2026_05_16_v20_core_a_kb_mcq_taa_soc_cm_ms_yn.json` | Stage 1 (Phase A), Stage 5 | broad re-anchor mix |
| `ift_data_2026_05_16_v20_core_b_rms_ate_vsp_rcm.json` | Stage 2 (Phase B), Stage 5 | AthenaBench catalog drill |
| `ift_data_2026_05_16_v20_core_val.json` | Stage 1/2 eval (val_size 0) | per-axis 50 rows / shortname |
| `ift_data_2026_05_16_v20_taa.json` | Stage 3, Stage 5 | TAA Classic narrow drill |
| `ift_data_2026_05_16_v20_taa_val.json` | Stage 3 eval | |
| `ift_data_2026_05_16_v20_cse.json` | Stage 4 | CyberSOCEval letter-set drill |
| `ift_data_2026_05_16_v20_cse_val.json` | Stage 4 eval | |

All `*.json` training files are gitignored; rsync from the build
workstation or regenerate in place (see §2 below).

---

## 2. Building the v20 datasets (from scratch)

Run the three shard builds concurrently. Each lives under its own
`_v20_*_build/` tree at the repo root; the `watcher.sh` enforces the
gates and produces the shards listed above.

```bash
cd /Users/pietro/code/Glaukopis      # or the cluster equivalent

# Stage 1+2: Core (count_max 5500 lifts ATE / RCM density)
python _v20_core_build/_neo4j_check.py
mkdir -p _v20_core_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05162026/Sophia-CTI-Templates-v20_core.txt \
     _v20_core_build/triples \
     SFT/data/ift_data_2026_05_16_v20_core.raw.json \
     2500 5500 > _v20_core_build/build.log 2>&1 &
echo "PID=$!" > _v20_core_build/build.pid
nohup bash _v20_core_build/watcher.sh > _v20_core_build/watcher.log 2>&1 &

# Stage 3: TAA Classic (manifest verbatim from v19)
mkdir -p _v20_taa_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05162026/Sophia-CTI-Templates-v20_taa.txt \
     _v20_taa_build/triples \
     SFT/data/ift_data_2026_05_16_v20_taa.raw.json \
     10 3500 > _v20_taa_build/build.log 2>&1 &
echo "PID=$!" > _v20_taa_build/build.pid
nohup bash _v20_taa_build/watcher.sh > _v20_taa_build/watcher.log 2>&1 &

# Stage 4: CSE (5 TI Count: bumps applied in v20_cse.txt)
mkdir -p _v20_cse_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05162026/Sophia-CTI-Templates-v20_cse.txt \
     _v20_cse_build/triples \
     SFT/data/ift_data_2026_05_16_v20_cse.raw.json \
     10 3500 > _v20_cse_build/build.log 2>&1 &
echo "PID=$!" > _v20_cse_build/build.pid
nohup bash _v20_cse_build/watcher.sh > _v20_cse_build/watcher.log 2>&1 &
```

Validate end-to-end with `python _v20_core_build/validate_corpus.py`.

---

## 3. Train — the v20 chain (Qwen-2.5-14B-Instruct)

Four sequential launchers under [`autotrain/`](autotrain/). Each
consumes the prior stage's pushed HF checkpoint as its base; only
each stage's final merged checkpoint is pushed (no intermediate
shards uploaded). All four wrap
[`utils/run_train.sh`](utils/run_train.sh) → `llamafactory-cli train`
with DeepSpeed ZeRO-3 and bf16, source `SFT/.env` for
`HF_TOKEN`/`HF_USERNAME`, and accept `--dry-run`, `--repo-id`,
`--report-to wandb|none`, and `--offload`/`--no-offload`.

| # | Launcher | Datasets (LF names) | Cutoff | Pack | LR | eff_bs | Eval/save | Base → push |
|---:|---|---|---:|---|---:|---:|---|---|
| 1 | `run_sft_qwen25_14b_v20_core.sh` **Phase A** | `ift_data_2026_05_16_v20_core_a_kb_mcq_taa_soc_cm_ms_yn` + `tulu_3_sft_mixture` + `alpaca_en_demo` | 8192 | on | 1e-5 | 16 | 500 | `Qwen/Qwen2.5-14B-Instruct` → (intermediate) |
| 2 | `run_sft_qwen25_14b_v20_core.sh` **Phase B** | `ift_data_2026_05_16_v20_core_b_rms_ate_vsp_rcm` | 16384 | off | 5e-6 | 8 | 400 | Phase A dir → `…/v20-core` |
| 3 | `run_sft_qwen25_14b_v20_taa.sh` | `ift_data_2026_05_16_v20_taa` | 4096 | on | 5e-6 | 16 | 100 | `…/v20-core` → `…/v20-taa` |
| 4 | `run_sft_qwen25_14b_v20_cse.sh` | `ift_data_2026_05_16_v20_cse` | 4096 | on | 5e-6 | 16 | 100 | `…/v20-taa` → `…/v20-cse` |
| 5 | `run_sft_qwen25_14b_v20_recalibrate.sh` | `core_a` (0.25) + `core_b` (0.40) + `taa` (0.35), `interleave_under` | 16384 | off | 1e-6 | 4 | 200 | `…/v20-cse` → `…/v20-recalibrate` |

### Per-stage notes (rationale and gotchas)

- **Stage 1+2 (Core)** runs both phases back-to-back inside the single
  `run_sft_qwen25_14b_v20_core.sh` launcher. Phase A is the broad
  re-anchor (packed, cutoff 8192); Phase B is the AthenaBench catalog
  drill (unpacked, cutoff 16384). Only Phase B's merged checkpoint is
  pushed. `--skip-eval` is the targeted Phase B OOM mitigation on
  ≤ 4-GPU configurations — the eval logits `[1, 16384, 152064]` fp32
  at ~10 GiB exceed per-rank headroom under no-offload ZeRO-3 and
  crash GPU 1 at the first `eval_steps=400` boundary. Eval is
  monitoring-only (no `load_best_model_at_end`), so disabling it
  changes zero training-state weights.

- **Stage 3 (TAA)** is byte-identical to the v18.1+TAA recipe. v14.1
  hot-fix (`--disable_gradient_checkpointing True`) is auto-applied
  on 8×H100 only; re-enabled on < 8 GPUs to keep activation
  footprint within per-rank budget.

- **Stage 4 (CSE)** is the CyberSOCEval letter-set drill. The
  manifest carries five `Count: 1500 → 2500` bumps on TI-actor /
  TI-other templates to lift the CSE-TI axis back toward v18.2.

- **Stage 5 (Recalibrate)** is a 3-shard low-LR interleaved
  touch-up. `interleave_probs = 0.25 / 0.40 / 0.35` is the v18.2
  production mix that v20 reverts to from v19's experimental
  `0.33 / 0.33 / 0.34` (the equal-weight mix starved Phase B of
  recalibrate exposure, see `v20_plan.txt §1.2`). At
  `--max-samples 2400` with `max(P) = 0.40` the interleaved
  dataset yields `2400 / 0.40 = 6000` rows ≈ 1500 optimizer
  steps ≈ 80-100 min on 4×H100. Intra-training eval is **disabled**
  (LlamaFactory requires `len(eval_dataset) == len(interleave_probs)`,
  and the three natural eval shards would dedupe to two; sign-off is
  via the post-train bench sweep).

### Wall-time budget

| Stage | 8×H100 80 GB | 4×H100 80 GB | Notes |
|---|---:|---:|---|
| Core (Phase A + Phase B) | ~13 h | ~26 h | Phase B is the bottleneck (cutoff 16384, packing off, ~0.9 k tok/s) |
| TAA | ~6-8 h | ~12-14 h | |
| CSE | ~4-6 h | ~8-12 h | |
| Recalibrate | n/a | ~95-115 min | Sized for 4 GPUs |
| **Total** | **~24 h** + 100 min | **~48 h** + 100 min | sequential |

### Where outputs land

- Training writes to `SFT/saves/Qwen_Qwen2.5-14B-Instruct/full/v20_<stage>_<ts>/`
  with intermediate checkpoints every `save_steps`,
  `save_total_limit 2`, `save_only_model True`.
- On exit 0 the launcher merges and uploads via
  [`scripts/upload_to_hf.py`](scripts/upload_to_hf.py) to
  `${HF_USERNAME}/athena-cti-sft-qwen25-14b-v20-{core,taa,cse,recalibrate}`.
- `train_config.json` (effective flags + git sha) and `train.log`
  (tee'd stdout/stderr) are written to the same output dir.

---

## 4. Test — benchmark sweeps

The pushed checkpoints are evaluated via the
[`test/`](test/README.md) pipeline against three suites: AthenaBench
(MCQ, RCM, VSP, ATE, TAA, RMS), CyberMetric (2K / 10K), and
CyberSOCEval (letter-set tasks). Two transports:

- `-vllm` alias suffix — local vLLM OpenAI-compatible server
  ([`test/utils/serve_vllm.sh`](test/utils/serve_vllm.sh)); the
  right choice for the four private v20 checkpoints.
- `-hf` alias suffix — HuggingFace Inference Providers (hosted API);
  the right choice for large public-model baselines (DeepSeek,
  Kimi, Gemma, Qwen 3.x).

```bash
# Two-terminal local vLLM sweep against v20-recalibrate.
conda activate vllm
bash SFT/test/utils/serve_vllm.sh \
    --model asg-ai/athena-cti-sft-qwen25-14b-v20-recalibrate --tp 4

conda activate llm-sft
cd SFT/test/utils
./run_benchmark.sh athena-cti-sft-qwen25-14b-v20-recalibrate-vllm \
    --suite athena --batch 64 --version 1
./run_benchmark.sh athena-cti-sft-qwen25-14b-v20-recalibrate-vllm \
    --suite cybersoceval --batch 64
```

### v20-recalibrate sign-off gates (`v20_plan.txt §5`, `tmpl_gen/templates/05162026/README.md §2.3`)

| axis | gate | rationale |
|---|---:|---|
| `athena-rms` (strict F1) | ≥ 54.0 | v18.2 §7.4 |
| `athena-mcq` | ≥ 62.0 | v18.2 §7.4 |
| `athena-taa` (Classic) | ≥ 40.0 | v18.2 §7.4 |
| `cse-ti` | ≥ 34.0 | v18.2 plateau target |
| `cse-malware` | ≥ 20.0 | v18.2 floor |
| `athena-ate` | ≥ 62.0 | v20 axis-density target (vs v19 regression) |
| `athena-rcm` | ≥ 67.5 | v20 axis-density target (vs v19 regression) |
| `athena-vsp` | ≥ 80.0 | v18.2 floor |
| `cybermetric-2k` | ≥ 85.5 | v18.2 floor |
| `cybermetric-10k` | ≥ 81.0 | v18.2 floor |

### Falsification

If `v20-recalibrate` misses ATE ≥ 62.0 or RCM ≥ 67.5 despite the
density bumps, the residual gap is attributable to the basin-shift
diagnosed in `v20_plan.txt §1.1` (8×H100 → 4×H100 build); a follow-on
v21 would need to either revert the build hardware or restructure the
Stage 5 mix further toward Phase B.

---

## 5. Lineage — why v20 is the final variant

v20 is the **targeted axis-density rebalance** on top of v19's
reproducibility-first rebuild of the v18.x architecture. Two
behavioural deltas vs v19:

1. **Core `count_max` 3500 → 5500** (Stage 1+2 build only). Lifts the
   realised per-axis yield for the regressed catalog axes:
   ATE 16 967 → 19 843 (+17 %), RCM 28 772 → 44 651 (+55 %). RMS /
   VSP / MCQ / MS / TAA / SOC / CM templates carry explicit `Count:`
   ceilings, so the bump is a no-op for those families (± 1 %).
2. **CSE TI `Count: 1500 → 2500`** on five JS.CSE.TI templates
   (GRP.2, GRP.3, MAL.2, MAL.3, NEG.1). Lifts CSE-TI-actor declared
   total 11 000 → 13 000 (+18 %) and CSE-TI-other 4 000 → 5 000
   (+25 %).
3. **Stage 5 `interleave_probs` revert** to v18.2's asymmetric
   `0.25 / 0.40 / 0.35` (Phase A / Phase B / TAA), from v19's
   experimental equal-weight `0.33 / 0.33 / 0.34`. The asymmetric
   mix weights Phase B (catalog drill) highest, which is the
   recalibrate emphasis the v20 chain needs given v19's Phase B-axis
   regression.

Predecessor lineage carried forward: chain topology and per-stage
recipe shapes are v18.1 → v18.2 → v19 → v20 byte-identical (only
dataset names, HF push targets, and the three deltas above change).
The v7 Llama-3.1-8B baseline (62.64 % strict F1 on `athena-rms`)
remains the documented 8B reference; the v20 chain is the documented
14B production recipe.

---

## 6. File map

```
SFT/
  README.md                       quick-start + environment
  SFT_FLOW.md                     this document
  .env.example                    HF_TOKEN / HF_USERNAME / WANDB / GITHUB_TOKEN
  utils/
    setup.sh                      idempotent installer (conda + envs + git auth)
    run_train.sh                  thin llamafactory-cli wrapper used by every launcher
    cleanup_disk.sh               disk reclaim helper (called by setup.sh)
  autotrain/
    run_sft_qwen25_14b_v20_core.sh         Stage 1+2 (Phase A + Phase B)
    run_sft_qwen25_14b_v20_taa.sh          Stage 3
    run_sft_qwen25_14b_v20_cse.sh          Stage 4
    run_sft_qwen25_14b_v20_recalibrate.sh  Stage 5 (final)
    run_abaligned_sft_v7.sh                Llama-3.1-8B v7 baseline (legacy)
    run_abaligned_sft_qwen25_14b_v{7,8,8small,81,9,10}.sh   pre-v11 lineage
    run_sft_qwen25_14b_v{11,12,13,14,14_1,16,17,18,18p1,18p2,19}*.sh  v11..v19 lineage
    README.md                              per-launcher flag reference
    model_cards/                           HF model-card seeds
  data/
    ift_data_2026_05_16_v20_*.json         seven v20 shards (gitignored)
    dataset_info.json                      LlamaFactory dataset registry
  scripts/
    upload_to_hf.py                        merge + push helper (called on exit 0)
    vllm_infer.py                          local vLLM batch inference
  test/
    README.md                              eval + transport reference
    utils/
      run_benchmark.sh                     sweep launcher
      serve_vllm.sh                        local vLLM server
    pipelines/
      models.py                            alias → repo registry (`-vllm`, `-hf` suffixes)
      api_usage.py                         HF Router cost / usage tracking

tmpl_gen/
  README.md                                template syntax + iftgen.py
  data_generation/
    make_dataset.sh                        docx2json → tmpl2triples → triples2alpaca
  templates/
    README.md                              category / source / design strategy
    05162026/                              ** v20 vintage (this is the final one) **
      README.md
      v20_plan.txt
      Sophia-CTI-Templates-v20_{core,taa,cse}.txt
      v20{,_taa,_cse}_row_count_gate.json

_v20_core_build/                           Stage 1+2 build pipeline (root)
_v20_taa_build/                            Stage 3 build pipeline (root)
_v20_cse_build/                            Stage 4 build pipeline (root)
```

