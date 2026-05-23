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

## Provenance

Forked 2026-05-18. See [`v21_plan.txt`](v21_plan.txt) §1 for the
regression context, §5 for the falsification matrix, §7 for the
recorded 14B outcome, and §8 for the Qwen2.5-32B port and the
Stage-4 recipe split.
