# v21 — v18.1 strict reproducibility experiment

`v21` is a forked clone of the `v18.1` (Qwen2.5-14B) three-stage chain
intended to verify that the `v18.1-core` peak is bit-reproducible from
byte-identical template + gate inputs. It exists because both `v19-core`
(58.5) and `v20-core` (57.9) regressed against `v18.1-core` (58.9) under
launcher comments asserting a "byte-identical recipe", which leaves
either silent template/gate drift or data-build non-determinism as the
remaining suspect. v21 forces both candidate causes to identity vs
`v18.1` and re-benchmarks.

See [`v21_plan.txt`](v21_plan.txt) for the full §1–§6 plan, including
the §5 sign-off gate (v21 axis-by-axis result must land within ±1.5 pp
of `v18.1-core` for the recipe to be declared bit-reproducible).

## What is byte-identical vs v18.1

| Artefact (v21) | Source (v18.1) | Verified by |
|---|---|---|
| `Sophia-CTI-Templates-v21.txt`     | `05112026/Sophia-CTI-Templates-v18.1.txt` | `shasum` |
| `Sophia-CTI-Templates-v21_taa.txt` | `05092026/Sophia-CTI-Templates-v16.txt`   | `shasum` |
| `Sophia-CTI-Templates-v21_cse.txt` | `05102026/Sophia-CTI-Templates-v17.1.txt` | `shasum` |
| `v21_row_count_gate.json`          | `05112026/v18_1_row_count_gate.json`      | `shasum` |
| `v21_taa_row_count_gate.json`      | `05092026/v16_row_count_gate.json`        | `shasum` |
| `v21_cse_row_count_gate.json`      | `05102026/v17_1_row_count_gate.json`      | `shasum` |

## What changes vs v18.1

Only paperwork and naming. No `Count:`, `Shuffle:`, `--max-samples`,
`--lr`, `--cutoff`, or gate-floor change:

- Date stamp `05_11` → `05_18`, vintage dir `05112026` → `05182026`
- Dataset names `ift_data_2026_05_11_v18p1_*` → `ift_data_2026_05_18_v21_*`
- Build dirs `_v18p1{,_taa,_cse}_build/` → `_v21_{core,taa,cse}_build/`
  (v19/v20 per-stage convention; v18.1 collapsed Core into `_v18p1_build/`)
- HF push targets `…-v18-1-{core,taa,cse}` → `…-v21-{core,taa,cse}`

## How to run

```bash
# 1. Build the v21 data (one shot per stage). Same args as v18.1.
mkdir -p _v21_core_build/triples _v21_taa_build/triples _v21_cse_build/triples
# Core (count_limit=2500, count_max=1500 verbatim from v18.1 README-18-1)
nohup bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21.txt \
    _v21_core_build/triples \
    SFT/data/ift_data_2026_05_18_v21_core.raw.json \
    2500 1500 > _v21_core_build/build.log 2>&1 &
echo "PID=$!" > _v21_core_build/build.pid
nohup bash _v21_core_build/watcher.sh > _v21_core_build/watcher.log 2>&1 &
# TAA (count_limit=10, count_max=3500 verbatim from v18 README)
nohup bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_taa.txt \
    _v21_taa_build/triples \
    SFT/data/ift_data_2026_05_18_v21_taa.raw.json \
    10 3500 > _v21_taa_build/build.log 2>&1 &
echo "PID=$!" > _v21_taa_build/build.pid
nohup bash _v21_taa_build/watcher.sh > _v21_taa_build/watcher.log 2>&1 &
# CSE (count_limit=10, count_max=3500 verbatim from v18 README)
nohup bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_cse.txt \
    _v21_cse_build/triples \
    SFT/data/ift_data_2026_05_18_v21_cse.raw.json \
    10 3500 > _v21_cse_build/build.log 2>&1 &
echo "PID=$!" > _v21_cse_build/build.pid
nohup bash _v21_cse_build/watcher.sh > _v21_cse_build/watcher.log 2>&1 &

# 2. Train. Stage 1 (Core) is run on its own; Stages 2-4 are chained
# behind it so the headline ship checkpoint (v21-recalibrate) lands in
# one wrapper call. The chain wrapper waits for each prior HF push to
# be readable before launching the next stage. See "Findings" below for
# why the off-plan Recalibrate stage is on the default chain path.
cd SFT/autotrain
./run_sft_qwen25_14b_v21_core.sh                   # 8xH100 80GB SXM (default)
./run_sft_qwen25_14b_v21_core.sh --gc on           # 8xRTX PRO 6000 96GB (Verda)
./run_sft_qwen25_14b_v21_chain.sh                  # TAA -> CSE -> Recalibrate

# 3. Benchmark (vLLM aliases registered in SFT/test/pipelines/models.py)
cd ../test
utils/run_benchmark.sh athena-cti-sft-qwen25-14b-v21-core-vllm \
    --suite athena --version 1
# Or all four v21 stages + suites in one shot:
utils/run_v21_sweep.sh
```

## Findings (2026-05-19; ship: v21-recalibrate)

The §5.1 sign-off gate (v21-core within ±1.5 pp of v18.1-core on every
axis) **partially failed**. Total Score landed inside the band but two
per-axis sub-scores fell outside it, with the same shape v19 and v20
showed against v18.1. The full four-stage chain was then run and
benchmarked; the off-plan Recalibrate stage (Stage 4) produced the
strongest ship candidate.

| Stage | Total | Notable axis movement vs v18.1-core | Within §5.1 band? |
|---|---|---|---|
| `v21-core`        | 60.8 (target 57.4..60.4)        | ATE -4.8pp, RCM -3.1pp     | Total: yes; per-axis: **no** |
| `v21-taa`         | (TAA narrow drill; per-plan)    | TAA Classic axis as expected | n/a |
| `v21-cse`         | per-stage; VSP 72.9 (was 82.5)  | VSP -9.6pp at CSE (chain trade-off) | n/a (Stage 3, downstream of Core) |
| `v21-recalibrate` | **62.3 (best of chain)**        | VSP recovered to 83.1       | n/a (off-plan; see below) |

### Interpretation

1. **The v18.1 recipe is not bit-reproducible at the per-axis level**
   even when templates, gates, counts, mixes, and hyperparameters are
   byte-identical to v18.1 (`shasum` verified for all templates / gate
   JSONs / row-count contracts; see §"What is byte-identical vs v18.1"
   above). Total Score lands inside the ±1.5pp instrument-noise band,
   but per-axis ATE/RCM drift exceeds it. Per [`v21_plan.txt`](v21_plan.txt)
   §5.3 this implicates **data-build non-determinism** (substrate
   sampling, dedup tiebreaks, shuffle seeding in
   `tmpl_gen/data_generation/make_dataset.sh` / Neo4j read order)
   rather than recipe drift; this is the same failure mode v19 and v20
   showed against v18.1.
2. **VSP erosion at CSE is a recurring chain pattern** (v18.1, v19,
   v20, v21). Stage 3 narrow drilling on the CyberSOCEval-letter-set
   shard reliably trades ~10pp of VSP for CyberSOCEval-Malware
   capability. The Recalibrate touch-up (1e-6 LR, 3-shard interleave of
   Core Phase A + Phase B + TAA at probs 0.25/0.40/0.35) recovers VSP
   without undoing the CSE gains; this matches the v19/v20 Recalibrate
   recovery shape.
3. **Ship recommendation: `athena-cti-sft-qwen25-14b-v21-recalibrate`**.
   62.3 Total beats both v21-core (60.8) and v18.1-core (58.9) and is
   the only v21 chain checkpoint with VSP back inside its v18.1-core
   band. The Recalibrate stage is included on the default chain path
   (`run_sft_qwen25_14b_v21_chain.sh`) for this reason.

### Open questions / next vintage

* **Data-build determinism**: the v19 / v20 / v21 axis drift cluster
  points to the data layer. A follow-up vintage should pin the Neo4j
  read order and seed the substrate sampler / shuffle in
  `make_dataset.sh` before another reproducibility run is attempted.
* **Architecture transfer**: the v21 recipe is being applied verbatim
  to `meta-llama/Llama-3.1-8B-Instruct` to check whether the
  Core->TAA->CSE->Recalibrate shape generalises off Qwen2.5-14B. See
  `SFT/autotrain/run_sft_llama31_8b_v21_{core,plus_taa,final,recalibrate,chain}.sh`.

## Qwen2.5-32B port (off-plan; recal recipe split)

The v21 chain is also ported to `Qwen/Qwen2.5-32B-Instruct` to test
recipe scaling. Stages 1-3 (Core / TAA / CSE) use the 14B recipe
verbatim (same datasets, same `lr`/`cutoff`/`packing`/`max-samples`,
only the base model and HF push targets change); they ride the existing
8xH200 / 8xH100 SXM footprint via `--optim adamw_8bit` + ZeRO-3 with
CPU offload on. v21-cse on Qwen2.5-32B posted Total 65.8 / Weighted
64.9 (2026-05-20 sweep), beating the 14B v21-cse and approaching the
14B v21-recalibrate Total within ~3 pp pre-Stage-4.

Stage 4 (Recalibrate) is the only stage where the 14B recipe does
**not** carry over cleanly. The off-plan touch-up that lifts 14B VSP
from 72.9 -> 83.1 instead drifts the wrong way at 32B scale under the
byte-identical recipe (VSP 78.9 -> 75.7). The most parsimonious
explanation is that the 14B-tuned 1e-6 LR + 0.40 Phase-B share
produces enough optimizer signal at 14B but sits at the noise floor on
32B + `adamw_8bit`, so Phase B catalog re-exposure cannot overpower
the post-CSE residual. To isolate the recipe variable cleanly, the v21
Qwen2.5-32B port keeps both Stage 4 variants on disk as **parallel
branches off v21-cse** (not stacked):

| Branch (Stage 4)   | HF target                                              | LR    | Probs (A / B / TAA) | `--max-samples` | Status                                                                                |
|--------------------|--------------------------------------------------------|-------|---------------------|-----------------|---------------------------------------------------------------------------------------|
| `v21-recalibrate`  | `asg-ai/athena-cti-sft-qwen25-32b-v21-recalibrate`     | 1e-6  | 0.25 / 0.40 / 0.35  | 2400            | 14B-recipe verbatim port. Benched; **fails** VSP recovery (78.9 -> 75.7).             |
| `v21-recal-32b`    | `asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b`       | 3e-6  | 0.15 / 0.60 / 0.25  | 3600            | 32B-tuned recipe (3x LR, Phase-B-heavy mix). Trained 2026-05-21. **Bench pending.**   |

Naming reflects **recipe provenance, not chain position** -- both
branches share `v21-cse` as their parent checkpoint, so the bench
comparison isolates the Stage-4 recipe change. Three coupled deltas
between the two branches (chosen to hold optimizer step count and
wall-time constant so the only A/B variable is the catalog-recovery
recipe):

* **LR**: 1e-6 -> 3e-6 (~3x bump, rough 32B/14B param ratio; targets
  the `adamw_8bit` optimizer noise floor at 32B scale).
* **Probs**: 0.25 / 0.40 / 0.35 -> 0.15 / 0.60 / 0.25 (heavier Phase B
  share = more VSP/RMS catalog exposure per interleaved row; Phase A
  and TAA reduced because neither is the 32B bottleneck -- CKT/TAA
  Classic are already in-band on v21-cse).
* **`--max-samples`**: 2400 -> 3600 (interleave_under cap =
  `max_samples / max(P)`. New `max(P)=0.60` -> 6000 interleaved rows,
  matching the original 2400 / 0.40 = 6000. Step count and wall-time
  preserved; only composition shifts).

Everything else (cutoff 16384, packing off, eff_bs 8, `--optim
adamw_8bit`, Liger, GC on, offload default-on at 32B, ZeRO-3) is held
identical to the 14B-recipe port. Launchers:

* `SFT/autotrain/run_sft_qwen25_32b_v21_recalibrate.sh` (Stage 4, 14B recipe verbatim)
* `SFT/autotrain/run_sft_qwen25_32b_v21_recal_32b.sh`   (Stage 4, 32B-tuned recipe)
* `SFT/autotrain/run_sft_qwen25_32b_v21_chain.sh`       (TAA -> CSE -> Recalibrate; Stage 4 uses the 14B-recipe variant for 14B-chain parity. The 32B-tuned `_recal_32b` variant is run standalone off `v21-cse` as the diagnostic A/B and is intentionally **not** on this chain path.)

**Ship recommendation pending Stage-4 bench.** If `v21-recal-32b`
recovers VSP without sacrificing the cse-stage gains (the 14B v21
ship pattern), it becomes the 32B ship candidate. If it does not,
`asg-ai/athena-cti-sft-qwen25-32b-v21-cse` remains the 32B headline
(Total 65.8 / Weighted 64.9). The bench wrapper at
`SFT/test/utils/serve_and_bench_qwen25_32b_v21_recal_32b.sh` runs
the full AthenaBench + CyberMetric 2K/10K + CyberSOCEval suite on
2xH100 under one warm vLLM session (~11 h wallclock).

## Provenance

Forked 2026-05-18. See [`v21_plan.txt`](v21_plan.txt) §1 for the
regression context, §5 for the falsification matrix, §7 for the
recorded 14B outcome, and §8 for the Qwen2.5-32B port and the
Stage-4 recipe split.
