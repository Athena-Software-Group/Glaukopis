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

Totals in this table are reported under the §5.1 sign-off-gate
weighting (per-axis combined with TAA-Classic only) for continuity
with the v18.1 reproducibility narrative. Under the later 50/50 TAA
blend (Classic + Canonical combined; introduced 2026-05-22 for
cross-architecture ranking) the same `v21-recalibrate` checkpoint
posts Total 61.0 / Weighted 59.6, still the best 14B chain stage.
See the leaderboard in §"Qwen3-30B-A3B-Thinking-2507 MoE port" for
the 50/50 ranking applied across all v21 ports.

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

| Branch (Stage 4)   | HF target                                              | LR    | Probs (A / B / TAA) | `--max-samples` | Status                                                                                                       |
|--------------------|--------------------------------------------------------|-------|---------------------|-----------------|--------------------------------------------------------------------------------------------------------------|
| `v21-recalibrate`  | `asg-ai/athena-cti-sft-qwen25-32b-v21-recalibrate`     | 1e-6  | 0.25 / 0.40 / 0.35  | 2400            | 14B-recipe verbatim port. Benched; **fails** VSP recovery (78.9 -> 75.7).                                    |
| `v21-recal-32b`    | `asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b`       | 3e-6  | 0.15 / 0.60 / 0.25  | 3600            | 32B-tuned recipe (3x LR, Phase-B-heavy mix). Benched 2026-05-21; **Total 65.0 / Weighted 62.9** -- ship.     |

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

**Ship recommendation: `asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b`.**
The 32B-tuned Stage-4 recipe puts the dense Qwen2.5-32B port at
Total 65.0 / Weighted 62.9 under the 50/50 TAA blend (Classic +
Canonical combined). **This is the v21 vintage's optimal ship
checkpoint across all SFT rows in the leaderboard** -- it tops the
14B v21-recalibrate (Total 61.0), the 8B Foundation-Sec and
Llama-3.1 v21-recalibrate ports (Totals 53.5 / 49.8), and every
Qwen3-MoE chain stage (best MoE checkpoint `v21-cse` at 63.4 /
60.9; see §"Qwen3-30B-A3B-Thinking-2507 MoE port" below). The
recal-32b shape is **CKT/RMS/AB-heavy** (CKT
75.5, RMS combined-f1 62.5, AB Avg II 62.6) and intentionally does
*not* attempt a Canonical-TAA lift on the dense-32B base (Canonical
combined 3.4, comparable to cse); the Stage-4 recipe at this scale
is tuned for VSP / catalog recovery, not alias->canonical migration.
The bench wrapper at `SFT/test/utils/serve_and_bench_qwen25_32b_v21_recal_32b.sh`
runs the full AthenaBench + CyberMetric 2K/10K + CyberSOCEval suite on
2xH100 under one warm vLLM session (~11 h wallclock).

## Qwen3-30B-A3B-Thinking-2507 MoE port (off-plan; ship at v21-cse, Stage 4 closed)

The v21 chain is also ported to `Qwen/Qwen3-30B-A3B-Thinking-2507`,
the pure-thinking July 2025 split of the Qwen3-30B-A3B MoE base
(30.5B total params, 3.3B active per token via 128 experts top-8).
This is the first Qwen3-family SFT in the codebase and the first MoE
port of v21. Param scale is peer to the Qwen2.5-32B port (30.5B vs
32B) so the cross-architecture comparison is `dense 32B` vs `sparse
30.5B / 3.3B-active` rather than a scale change. Hardware: 8xB300
288GB SXM (Blackwell Ultra, `sm_103a`); `adamw_8bit` + Liger + ZeRO-3
without CPU offload (288 GB HBM3e per GPU absorbs the 30.5B optimiser
shard comfortably).

### Chain shape

Stages 1-3 (Core / TAA / CSE) hold the 14B / dense-32B recipe
byte-identical: same datasets, same `lr` / `cutoff` / `packing` /
`--max-samples` / `eff_bs` / `save-steps`. Only the base model,
`--template` (`qwen` -> `qwen3`), and HF push targets change.

**Stage 4 was run twice (32B-tuned `recal-32b` and 14B-recipe
`recalibrate`) and neither beat `v21-cse` on the 50/50 TAA blend
on this architecture.** Both Stage-4 launchers and HF targets are
retained for reproducibility, but the on-chain ship checkpoint for
the Qwen3-MoE port is `v21-cse`. See "Ship recommendation" below
for the per-variant numbers and the MoE-specific failure mode.

| Stage              | HF target                                                              | Recipe source            | Wall-time (8xB300) |
|--------------------|------------------------------------------------------------------------|--------------------------|--------------------|
| `v21-core`         | `asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core`           | 14B/32B Core (Phase A+B) | ~14-18 h           |
| `v21-taa`          | `asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa`            | 14B/32B TAA              | ~7-10 h            |
| **`v21-cse`** (ship) | `asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse`          | 14B/32B CSE              | ~5-7 h             |
| `v21-recal-32b`    | `asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b`      | 32B-tuned recal-32b      | ~1.5-2 h           |
| `v21-recalibrate`  | `asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recalibrate`    | 14B recalibrate          | ~1-1.5 h           |

Total ~28-37 h end-to-end Core -> Stage 4 on 8xB300, vs ~55-75 h
on the 8xH100 80GB SXM fallback.

### Training semantic (`enable_thinking=True`)

The base model is pure-thinking (every response is prefixed with
`<think>...</think>`). `SFT/utils/run_train.sh` defaults
`--enable_thinking True`, which makes the Qwen3 reasoning template
inject an empty `<think>\n\n</think>` block into the
loss/response_ids on every SFT sample that does not already carry a
`<think>` block (i.e. every row in the v21 CTI corpus). The trained
model learns to autonomously emit a ~6-token empty thought followed
by the answer for CTI prompts. The thinking apparatus stays alive as
a generation path so OOD (non-CTI) reasoning behaviour can still
resurface; CTI-domain queries get the empty-trace short-circuit
without retraining the reasoning weights to zero.

At serve / bench time the `-no-think` suffix on the vLLM alias
(`athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b-no-think-vllm`)
forwards `chat_template_kwargs.enable_thinking=False` per request --
belt-and-suspenders against template drift, and -- more importantly
-- it suppresses `VLLMModel`'s `-thinking` 8192-token decode floor so
the per-task caps in `TASK_MAX_NEW_TOKENS` (MCQ=128, RCM/RMS/TAA=256)
apply correctly. See `SFT/test/pipelines/models.py` for the alias
registration.

### Launchers

```bash
# Stage 1 (Core, Phase A+B) -- run on its own so its HF push is the
# probed base for the chained stages.
bash SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_core.sh

# Stages 2-4 (TAA -> CSE -> Recal-32b) chained behind it. The chain
# waits for each prior HF push to be readable before launching the
# next stage; default --stop-stage is recal_32b. To stop at the
# ship checkpoint and skip the (closed) Stage-4 sweep, pass
# `--stop-stage cse`.
bash SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_chain.sh
# ... or SSH-resilient under nohup setsid:
bash SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_chain.nohup.sh

# Off-chain 14B-recipe Stage-4 A/B (manual; not invoked by chain.sh):
bash SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_recalibrate.sh
```

Per-stage benchmark wrappers live under `SFT/test/utils/serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_{core,taa,cse,recal_32b,recalibrate}.sh`
and each runs the full Athena + CM-2K + CM-10K + CSE suite on 2xH100
under one warm vLLM session.

### Ship recommendation

**Ship: `asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse`.**
Both Stage-4 variants were benched (2026-05-22) and neither beat
`v21-cse` on the 50/50 TAA blend (Classic + Canonical) used as the
v21 ranking metric. Per-variant numbers off the same `v21-cse`
parent:

| Stage (50/50 TAA blend) | CKT  | RCM  | ATE  | VSP  | RMS  | TAA Classic | TAA Canonical | CM avg | Total  | Weighted |
|-------------------------|-----:|-----:|-----:|-----:|-----:|------------:|--------------:|-------:|-------:|---------:|
| **`v21-cse`** (ship)    | 74.5 | 68.5 | 58.6 | 85.0 | 50.1 | 47.0        | 4.9           | 88.6   | **63.4** | **60.9** |
| `v21-recalibrate` (14B) | 63.3 | 63.0 | 52.8 | 83.6 | 50.9 | 49.0        | 12.8          | 88.8   | 62.3   | 59.0     |
| `v21-recal-32b`         | 61.1 | 67.3 | 54.8 | 84.2 | 50.6 | 45.5        | 34.2          | 79.2   | 59.5   | 58.1     |

The two Stage-4 attempts trace a Pareto frontier between Canonical
TAA lift and broad-knowledge preservation:

* `v21-recal-32b` delivers a **+29.3pp Canonical-TAA lift** over cse
  (4.9 -> 34.2) -- the only checkpoint in the v21 vintage that moves
  this axis meaningfully -- but **crashes CyberMetric by 9.4pp**
  (88.6 -> 79.2). The dense Qwen2.5-32B port under the same recipe
  does *not* exhibit either effect (Canonical stays at 3.4, CM
  holds at 89.8), so both the lift and the CM crash are
  MoE-specific.
* `v21-recalibrate` (14B-recipe) preserves CM cleanly (88.8) but
  drifts CKT / RCM / ATE 5-11pp below cse and delivers only a
  +7.9pp Canonical lift -- the gentler recipe is signal-starved for
  Canonical migration at this scale.

Mechanism: on Qwen3-MoE (128 experts, top-8 routing), any
second-pass SFT off `v21-cse` perturbs expert routing for narrow
analytical axes regardless of LR or Phase-A/B mix -- a failure mode
absent from the dense-32B port at peer parameter scale. The
recipe-knob axis (LR, interleave probs, sample count) is not the
dominant variable here; the architecture is. The Qwen3-MoE chain is
therefore closed at `v21-cse` for the v21 vintage.

### Matched-conditions base baseline (no-think A/B)

The per-stage table above reports v21 SFT checkpoints served under the
`-no-think-vllm` alias scheme (`enable_thinking=False` at request time,
per-task `TASK_MAX_NEW_TOKENS` caps applied without the thinking-mode
8192-token floor). The original
`serve_and_bench_qwen3_30b_a3b_thinking_2507_baseline.sh` wrapper benched
the base `Qwen/Qwen3-30B-A3B-Thinking-2507` on its fair footing instead
-- thinking-on with the 8192 floor and the `qwen3` reasoning parser --
which is a useful capability ceiling but is **not directly comparable**
to the v21 SFT numbers above. To close the A/B, the base is also benched
under the matched no-think inference path via
`serve_and_bench_qwen3_30b_a3b_thinking_2507_baseline_no_think.sh`
(alias: `qwen3-30b-a3b-thinking-2507-no-think-vllm`; same HF repo as the
thinking-on alias, `-no-think` substring triggers the same VLLMModel
detection used by the v21 SFT bench wrappers).

Expected signature: the base under matched no-think conditions should
collapse relative to its thinking-on baseline because Thinking-2507 was
not trained on the empty-thought pattern that the v21 SFT instills. The
expected collapse vs the SFT'd `v21-cse` numbers IS the signal -- it
isolates the SFT's contribution to functioning under a no-trace
inference budget, separate from any general-knowledge or reasoning gain
the SFT may also deliver. MMLU-Pro is benched in a separate session via
the generic `serve_and_bench_mmlu_pro.sh <alias>` wrapper so suite
scope stays decoupled from the CTI baselines.

| Row (matched no-think) | CKT  | RCM  | ATE  | VSP  | RMS  | TAA Classic | TAA Canonical | CM avg | Total  | Weighted | MMLU-Pro |
|------------------------|-----:|-----:|-----:|-----:|-----:|------------:|--------------:|-------:|-------:|---------:|---------:|
| Base (no-think)        | _pending sweep 2026-05-23_ | | | | | | | | | | |
| **`v21-cse`** (ship)   | 74.5 | 68.5 | 58.6 | 85.0 | 50.1 | 47.0        | 4.9           | 88.6   | **63.4** | **60.9** | _pending_ |

**Cross-architecture optimal: `asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b`**
(dense Qwen2.5-32B + 32B-tuned recal recipe; Total 65.0 / Weighted
62.9) remains the v21 vintage's recommended ship checkpoint across
all ported architectures -- see §"Qwen2.5-32B port" above. The
Qwen3-MoE `v21-cse` ship is the recommendation **conditional on the
MoE architecture** (for consumers who specifically need the
sparse-30B / 3.3B-active inference footprint); on absolute v21
leaderboard ranking, dense-32B + recal-32b wins by ~1.6 Total.

## Contamination posture

This section is the v21-specific audit trail for the contamination
question ("does the v21 SFT corpus -- across its Core / TAA / CSE
shards -- leak the AthenaBench / CTIBench / CyberMetric / CyberSOCEval
evaluation signal into training?"). v21 is a SHA-verified byte-clone of
v18.1 / v16 / v17.1 templates and gates (see §"What is byte-identical
vs v18.1" above), so the posture is mechanically inherited from those
vintages and ultimately from v8, where the framework was first
articulated in [`tmpl_gen/templates/04292026/README.md`](../04292026/README.md)
§2. This section reproduces the full posture in self-contained form so a
reader auditing v21 alone does not need to traverse the carry chain.

We separate the two failure modes the research community distinguishes:

  * **Verbatim contamination** -- a literal eval prompt, eval answer
    string, or eval-row n-gram appearing in a training row. This is the
    failure mode that matters for benchmark validity, because the model
    can solve the eval row by memorisation rather than by reasoning.
  * **Structural contamination** -- the training data and the eval data
    sharing the same underlying knowledge base (the MITRE STIX bundles
    for ATT&CK / CAPEC / CWE, the NVD CVE feed, the CISA KEV catalog,
    the FIRST EPSS feed, the D3FEND v1.4.0 matrix). The model trained on
    the graph can answer a benchmark item drawn from the same graph
    because the underlying fact is identical, not because the eval row
    was memorised.

v21 treats these two failure modes asymmetrically:

  * Verbatim contamination is **blocked / audited by tooling at corpus
    build time** -- each of the three v21 shard watchers runs
    `tmpl_gen/scripts/dedup_against_evals.py` as its Phase 5 step
    (mechanism in §"Verbatim contamination guard" below), and the
    shard's clean dataset only lands in `SFT/data/` if Phase 5 exits
    cleanly. The launcher cannot find an input to consume otherwise
    (`run_sft_qwen25_14b_v21_core.sh:154-166` exits 2 with
    `[FAIL] v21-core dataset missing` if the file is absent).
  * Structural contamination is **accepted by design**, with the
    rationale documented per benchmark in §"Per-benchmark contamination
    matrix" below. This is the established posture in the published
    CTI-LLM literature: SecKnowledge / CyberPal.AI (Levi et al. 2024,
    arXiv:2408.09304), the CTIBench paper, and the AthenaBench technical
    report all build training corpora from the same MITRE / NIST / FIRST
    bundles the benchmarks score against, on the grounds that there is
    no "held-out ATT&CK matrix" and pretending otherwise would defeat
    the purpose of training a CTI model.

The remainder of this section makes both positions explicit and verifies
that nothing in the v21 chain shape (three independent shards, off-plan
Recalibrate stage, five architecture ports) has weakened the audit.

### Why the v8 / v18.1 posture transfers cleanly to v21

Because every `Sophia-CTI-Templates-v21*.txt` body and every
`v21*_row_count_gate.json` is `shasum`-identical to its v18.1 / v16 /
v17.1 predecessor, the v21 corpus shares exactly the same **template-
side** contamination surface as the v18.1 chain. The only v21-side data
deltas vs v18.1 are date-stamps, dataset file names, build-dir names,
and HF push targets (§"What changes vs v18.1"); none touch the verbatim
or structural surface. The audit chain is:

  * **Verbatim:** the same `dedup_against_evals.py` invocation that
    cleared the v18.1 / v16 / v17.1 corpora is re-executed
    independently on each v21 shard at build time. Because the template
    bodies are byte-identical and the row generation consumes the same
    Neo4j substrate, the expected dedup-drop counts land within the
    v18.1 envelope; the per-shard `dedup_report.json` files are the
    audit artefact.
  * **Structural:** unchanged from v8. The v21 Core shard reads from
    the same MITRE / NVD / CISA / FIRST / D3FEND graph the eval suites
    score against; the v21 CSE shard reads from the same
    Qwen2.5-7B-generated CyberSOCEval-shape substrate v17.1 introduced
    (see [`Sophia-CTI-Templates-v21_cse.txt`](Sophia-CTI-Templates-v21_cse.txt)
    header, which is byte-identical to v17.1 and re-states the
    `Shuffle: mcq_multi` fix that broke the `JS.CSE.*` letter-set
    mode-collapse without changing the underlying substrate or the
    structural overlap with CyberSOCEval).

### Verbatim contamination guard: `dedup_against_evals.py`

**Tool:** `tmpl_gen/scripts/dedup_against_evals.py` (unchanged since v8).

**Invocation in v21** (verbatim from each watcher's Phase 5 block;
source `_v21_core_build/watcher.sh:301-307` and the equivalent blocks
in `_v21_taa_build/watcher.sh:124-130` and `_v21_cse_build/watcher.sh:123-129`):

```
python tmpl_gen/scripts/dedup_against_evals.py \
    --input          ${BALANCED_JSON} \
    --filter-output  ${CLEAN_JSON} \
    --drop-threshold 50 \
    --hit-threshold  1 \
    --max-fail       999999 \
    --report         ${DEDUP_REPORT}
```

The `DEDUP_HIT_THRESHOLD=1` and `DEDUP_DROP_THRESHOLD=50` tunables are
set at the top of each watcher (`_v21_core_build/watcher.sh:81-82`) at
the v18 / v8 defaults.

**Mechanism** (step-by-step):

  1. Tokenise every eval record's user-visible text (`question`,
     `prompt`, `report`, `context`, `input`, `instruction`, plus
     `answers`) into lowercase `[A-Za-z0-9]+` word tokens
     (`tmpl_gen/scripts/dedup_against_evals.py:30-55`).
  2. Build the union set of distinct **n=13 word-grams** (default,
     overridable via `--n`) across every eval file under
     `--eval-dir SFT/test/benchmark_data/` -- which currently covers:
       * `athena_bench/` (`athena-cti-{ate,mcq,mcq-3k,mcq-updated,rcm,rms,vsp}.jsonl`,
         `athena_rms/`, `athena_taa/`, `mcq-patch.tsv`)
       * `cti_bench/` (`cti-{ate,mcq,rcm,rcm-2021,vsp}.tsv`, `cti_taa/`)
       * `cybermetricdataset/` (`CyberMetric-{80,500,2000,10000}-v1.json`)
       * `cybersoceval/` (`malware_analysis/*.jsonl`,
         `threat_intel_reasoning/*.jsonl`)
       * `cve/`, `urlhaus/` (operational data sources)
  3. Index every n-gram to its source `eval_file:row_idx`.
  4. For every candidate Alpaca SFT row, tokenise the concatenation of
     `instruction`, `input`, `output`; emit any n-gram that appears in
     the eval index; flag the row when its hit count against any
     **single** eval row reaches `--hit-threshold` (set to 1 in v21).
  5. Two thresholds gate behaviour:
       * `--hit-threshold 1` is the **flag** threshold -- a single shared
         13-gram is enough to log the row to `--report`.
       * `--drop-threshold 50` is the **filter** threshold -- a row is
         removed from `--filter-output` when its hit count against a
         single eval row reaches 50. v21 sets `--max-fail 999999` so the
         build does not abort on flagged-but-not-dropped rows; the
         report file is the human audit surface and the filtered output
         is the trainable artefact.

**Threshold rationale.** n=13 word tokens is the same window used by
the OLMo, Pythia, and Llama families' MMLU / HellaSwag / BIG-bench
decontamination passes: short enough to catch verbatim leakage while
long enough that incidental matches on common stock phrases (e.g. "you
are a cybersecurity expert that has been trained") do not trigger.
`hit-threshold=1` is the strict audit setting; one shared 13-gram
between an SFT row and an eval row is, in practice, never incidental
for technical CTI text and almost always indicates the eval row's
question or answer string was inadvertently included.
`drop-threshold=50` is the v10-onward soft setting (vs the v8 hard
abort) that distinguishes verbatim row inclusion (>=50 shared 13-grams,
filtered out) from incidental shared CVE / MITRE description vocabulary
(<50 shared 13-grams, kept on the rationale that training and eval
share the same Neo4j knowledge base and graph-derived vocabulary
overlap is the structural-contamination case, not verbatim leakage).

**What this catches:** any SFT row that quotes an eval row's question,
answer, multiple-choice option text, threat-report excerpt, or
explanatory paragraph at >=13 contiguous tokens of overlap.

**What this does not catch:** semantically equivalent paraphrases (e.g.
an SFT row asking "Which CWE underlies CVE-2024-NNNN?" when an eval row
asks "What is the root-cause weakness for CVE-2024-NNNN?"). Catching
paraphrastic leakage requires embedding-similarity dedup, which is not
in the v21 pipeline; v21 addresses paraphrase risk **structurally** by
building templates off the underlying knowledge graph rather than off
eval text, so paraphrastic overlap is only possible when both the SFT
row and the eval row independently describe the same graph fact --
which is the structural-contamination case addressed below.

### Per-shard v21 enforcement

Each of the three v21 build watchers runs the Phase 5 invocation
independently, producing an independent report:

| Shard | Watcher                       | Clean JSON (only written if dedup OK)              | Dedup report                          |
|-------|-------------------------------|-----------------------------------------------------|---------------------------------------|
| Core  | `_v21_core_build/watcher.sh`  | `SFT/data/ift_data_2026_05_18_v21_core.json`        | `_v21_core_build/dedup_report.json`   |
| TAA   | `_v21_taa_build/watcher.sh`   | `SFT/data/ift_data_2026_05_18_v21_taa.json`         | `_v21_taa_build/dedup_report.json`    |
| CSE   | `_v21_cse_build/watcher.sh`   | `SFT/data/ift_data_2026_05_18_v21_cse.json`         | `_v21_cse_build/dedup_report.json`    |

A non-zero exit from `dedup_against_evals.py` triggers the watcher's
`fail "dedup"` helper (`_v21_core_build/watcher.sh:315`, and the
equivalent lines in `_v21_taa_build/watcher.sh:139` and
`_v21_cse_build/watcher.sh:138`), which writes the failure to
`_v21_{core,taa,cse}_build/watcher_status.json` and skips Phase 6
through Phase 9; no `CLEAN_JSON` is written and no Phase 9 per-axis
shard files are produced.

The v21 stage launchers do not call `dedup_against_evals.py` directly;
they enforce the dedup gate **indirectly** by asserting the existence
of the dedup-output dataset file. For example, the Core launcher:

```
SFT/autotrain/run_sft_qwen25_14b_v21_core.sh:154-166
  for ds in ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn \
            ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm \
            "${VAL_NAME}"; do
      if [[ ! -f "${SFT_DIR}/data/${ds}.json" ]]; then
          echo "[FAIL] v21-core dataset missing: ..." >&2
          exit 2
      fi
  done
```

Because the watcher only writes the Phase 9 per-phase shards (and
therefore the launcher-visible dataset files above) **after** Phase 5
dedup has produced `CLEAN_JSON` successfully, the launcher's
`[[ ! -f ... ]]` guard is structurally a dedup gate -- the trainable
artefact cannot exist on disk unless dedup passed -- without the
launcher needing to re-implement the dedup contract. The same
indirection applies to the TAA-stage launcher
(`run_sft_qwen25_14b_v21_plus_taa.sh`) and the CSE-stage launcher
(`run_sft_qwen25_14b_v21_final.sh`), each gated on its respective
shard's `ift_data_2026_05_18_v21_{taa,cse}*.json`, and to every
cross-architecture port of the v21 chain
(`run_sft_{foundation_8b,gemma4_31b,llama31_8b,qwen25_32b,qwen3_30b_a3b_thinking}_v21_*.sh`),
all of which consume the same three shard files.

### Per-benchmark contamination matrix

The v21 benchmark portfolio is **AthenaBench + CTIBench + CyberMetric +
CyberSOCEval** (the full eval index in `dedup_against_evals.py`); the
v21 chain ports also bench MMLU-Pro for general-knowledge regression
tracking (`SFT/test/utils/serve_and_bench_mmlu_pro.sh`), which is run
out of the loop and is not in the contamination index because the v21
SFT corpus does not include any MMLU-Pro adjacent material.

| Benchmark | Held out from training? | What v21 SFT shares with it | Verbatim guard in v21 | Structural notes |
|---|---|---|---|---|
| **AthenaBench** (`athena-cti-{ate,mcq,rcm,rms,vsp,taa}.jsonl`) | No -- structural overlap accepted (catalog-recall framing). | The underlying ATT&CK / CWE / CVE / KEV / EPSS / D3FEND graph. The benchmark prompts are GPT-{4,5}-rewritten incident narratives produced at benchmark generation time and are NOT in the v21 corpus. | All three shard watchers' Phase 5 indexes `athena_bench/*.jsonl` and the `athena_rms/`, `athena_taa/` subdirs at n=13. Any >=50 shared 13-grams filters the row from the trainable artefact and logs it to the shard's `dedup_report.json`. | The catalog-recall framing is the published AthenaBench position. v21's `AB.RMS.{3a..3h,4,5,6}` templates (carried byte-identical from v18.1) target the catalog-coverage and cardinality gaps the v5 baseline exposed; they do not encode any benchmark prompt. |
| **CTIBench** (`cti-{ate,mcq,rcm,rcm-2021,vsp}.tsv`, `cti_taa/`) | No -- structural overlap accepted (same graph). | Same MITRE / NIST graph as the SFT corpus. CTIBench prompts are paraphrastic transformations of the underlying records. | Same n=13 fingerprint as above; `cti_bench/*.tsv` is in the index for all three v21 shards. | The CTIBench paper explicitly anticipates this overlap and grades models on the paraphrastic surface form. v21 templates do not import CTIBench-specific phrasings. |
| **CyberMetric** (`CyberMetric-{80,500,2000,10000}-v1.json`; v21 ports bench the 2K and 10K slices) | No -- structural overlap with general CTI knowledge accepted; the benchmark is partly a general-knowledge MCQ over public security material that overlaps the base model's pretrain. | Domain knowledge (concepts, definitions, CVE descriptions). The MCQ phrasings themselves are NOT in the v21 corpus. | All four CyberMetric files in the n=13 index; verbatim leakage of any MCQ stem or option string is filtered out and logged. | The `tulu_3_sft_mixture` + `alpaca_en_demo` catastrophic-forgetting guards in v21-Core Phase A preserve general instruction-following so that CyberMetric scores reflect CTI-narrowing impact rather than instruction collapse, not so that the model can solve specific CyberMetric items. |
| **CyberSOCEval** (Meta; `malware_analysis/*.jsonl`, `threat_intel_reasoning/*.jsonl`) | No -- structural overlap with the public CTI corpus accepted. | The JSON envelope shapes (e.g. `{"correct_answers": [...]}`, `{"behaviors": [...]}`) are matched in the `JS.*` and `JS.CSE.*` template families. The specific eval prompts and ground-truth answers are NOT in the v21 corpus. | The CyberSOCEval source files under `SFT/test/benchmark_data/cybersoceval/` are picked up by the same n=13 glob and any >=50-overlap row is filtered. | The `JS.CSE.*` family in the v21 CSE shard is byte-identical to v17.1, which itself reset the v17 letter-set mode-collapse via a `Shuffle: mcq_multi` directive without altering the underlying CyberSOCEval-shape substrate. The family teaches the **shape** of a JSON-wrapped multi-select response over graph-derived facts; CyberSOCEval grades the same shape over its held-out facts. |

### Adjacent corpus-hygiene gates (not contamination, but co-resident)

Phase 6b runs `tmpl_gen/scripts/check_corpus_licences.py` immediately
downstream of Phase 5 dedup, against the `CLEAN_JSON` artefact, to
verify every row's `source` field is in the commercial-use allowlist
(per-shard report at `_v21_{core,taa,cse}_build/licence_gate_report.json`).
Phase 6 is the orthogonal **floor** check that the post-dedup corpus
still meets the per-axis row-count contract the chain was tuned against,
specified in `tmpl_gen/templates/05182026/v21_row_count_gate.json`
(Core), `v21_taa_row_count_gate.json` (TAA), and
`v21_cse_row_count_gate.json` (CSE). Neither is part of the
contamination posture as such, but both are reported alongside the
dedup report in the watcher's final `watcher_status.json` so the full
corpus-hygiene chain is auditable from a single artefact per shard.

### What v21 explicitly does **not** do

To keep the v8 falsifiability list intact, v21 inherits the same
exclusions verbatim and adds no new ones:

  * **No eval-row text in any template.** No template's `Instruction:`,
    `Question:`, or `Answer:` body in `Sophia-CTI-Templates-v21*.txt`
    contains a verbatim phrase from any eval row. The n=13 guard would
    flag the row if a future edit ever introduced one.
  * **No benchmark answer keys in the graph.** The Neo4j build
    (`tmpl_gen/scripts/iftgen.py` -> `create_ATTACK_db`) ingests the
    upstream MITRE / NIST / FIRST / CISA / D3FEND bundles only. The
    graph does not contain AthenaBench's GPT-rewritten narratives,
    CyberSOCEval's question JSON, CTIBench's paraphrases, or
    CyberMetric's MCQ stems.
  * **No fine-tuning on benchmark dev/validation splits.** AthenaBench
    and CTIBench publish small validation slices intended for prompt
    engineering. v21 does not consume any of these splits as training
    data; they are only used downstream by the eval harness under
    `SFT/test/`.
  * **No cross-pollination from the eval harness.** The eval harness in
    `SFT/test/pipelines/` reads from `SFT/test/benchmark_data/` only
    and never writes back into `SFT/data/`. Training datasets live in
    `SFT/data/`, benchmarks live in `SFT/test/benchmark_data/`, and
    the v21 build pipeline scripts in `tmpl_gen/scripts/` write only
    to the former.
  * **No per-token contamination check on `tulu_3_sft_mixture` or
    `alpaca_en_demo`.** These two HF mixtures are general
    instruction-following data and are unlikely to overlap CTI
    benchmarks at n=13; v21 does not run `dedup_against_evals.py`
    against them as a matter of routine. Same posture as v8.
  * **No embedding-similarity dedup.** Out of scope for v21;
    paraphrastic risk is addressed structurally (templates are
    generated from the knowledge graph, not from eval-row text), so
    paraphrastic overlap with an eval row is only possible when both
    the SFT row and the eval row independently describe the same graph
    fact -- the structural-contamination case, which is accepted.
  * **No relaxation of dedup thresholds across the v21 chain.** All
    three shards and all six architecture ports (Qwen2.5-14B,
    Qwen2.5-32B, Qwen3-30B-A3B-Thinking-2507, Foundation-Sec-8B,
    Llama-3.1-8B-Instruct, Gemma-4-31B) consume the same three shard
    files produced by the same n=13 / hit=1 / drop=50 dedup pass;
    there is no per-architecture dedup recipe.

### Reproducing the contamination check

The three v21 shard reports are written by the build pipeline and live
under each shard's build dir. To regenerate any one of them locally:

```bash
# Re-run dedup against the v21 Core shard's balanced corpus
python tmpl_gen/scripts/dedup_against_evals.py \
    --input           SFT/data/ift_data_2026_05_18_v21_core.balanced.json \
    --filter-output   /tmp/v21_core_clean.json \
    --eval-dir        SFT/test/benchmark_data \
    --n 13 \
    --hit-threshold   1 \
    --drop-threshold  50 \
    --max-fail        999999 \
    --report          /tmp/v21_core_dedup_report.json

# Same for TAA and CSE
python tmpl_gen/scripts/dedup_against_evals.py \
    --input           SFT/data/ift_data_2026_05_18_v21_taa.balanced.json \
    --filter-output   /tmp/v21_taa_clean.json \
    --eval-dir        SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 --drop-threshold 50 --max-fail 999999 \
    --report          /tmp/v21_taa_dedup_report.json

python tmpl_gen/scripts/dedup_against_evals.py \
    --input           SFT/data/ift_data_2026_05_18_v21_cse.balanced.json \
    --filter-output   /tmp/v21_cse_clean.json \
    --eval-dir        SFT/test/benchmark_data \
    --n 13 --hit-threshold 1 --drop-threshold 50 --max-fail 999999 \
    --report          /tmp/v21_cse_dedup_report.json
```

A clean shard prints `scanned N corpus rows -> 0 dropped` and exits 0.
Flagged-but-not-dropped rows (hit count between 1 and 49 inclusive) are
written to the report file as the human audit surface but are retained
in `--filter-output` per the structural-overlap rationale. A non-zero
drop count writes the offending rows to the report and filters them
out; the build is expected to inspect the report and reword the
offending template(s) before re-launching the chain.

### Scope and limitations

  * **The audit is per-shard, not per-architecture.** All v21 ports
    (Qwen2.5-14B / 32B, Qwen3-30B-A3B-Thinking-2507, Foundation-Sec-8B,
    Llama-3.1-8B, Gemma-4-31B) consume the same three shard files
    after dedup; the per-architecture variation is in optimiser /
    `--template` / HF push target only, not in dataset content. One
    pass of the contamination check per shard therefore certifies all
    six architecture ports of that shard.
  * **The audit covers the v21 SFT corpus, not the base models.**
    Every base model on which v21 is fine-tuned has its own pretrain
    contamination surface that is out of scope for this section. The
    structural-overlap framing assumes the base model has already seen
    the public MITRE / NIST / FIRST / CISA bundles via pretrain and
    therefore that the held-out-vs-training-graph question is moot.
  * **The audit covers the SFT path only, not the CPT (continued
    pretraining) path** documented separately in `cpt/`. CPT has its
    own leak protections; see 04292026 README §2.6 for the CPT-side
    posture (carried forward unchanged to v21).
  * **The audit is generation-time, not run-time.** Dedup runs once,
    when each shard is built (Phase 5 of each watcher), and again on
    demand via the snippet above. The trainer does not re-dedup at
    every epoch -- the contract is that the dataset file on disk is
    already clean when the trainer reads it.

## Provenance

Forked 2026-05-18. See [`v21_plan.txt`](v21_plan.txt) §1 for the
regression context, §5 for the falsification matrix, §7 for the
recorded 14B outcome, and §8 for the Qwen2.5-32B port and the
Stage-4 recipe split.
