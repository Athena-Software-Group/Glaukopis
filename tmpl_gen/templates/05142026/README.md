# Sophia CTI Templates — v18.2.2 (May 14, 2026 vintage)

v18.2.2 is the **third iteration** of the Stage 4 multi-shard replay
touch-up that sits on top of the v18.1 chain. It is a **recipe-only**
change — no manifest delta, no new shards, no new generators — and
exists to recover the §7.4 gate package that v18.2 missed by a hair
(MCQ −7.67, RMS −0.28) and that v18.2.1 then strictly regressed
against (4 gates failing instead of 2). The published v18.1 chain
(`-v18-1-core` → `-v18-1-taa` → `-v18-1-cse`) is **not re-trained**;
v18.2.2 chains a single low-LR, three-shard interleaved replay onto
the existing `asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse` checkpoint
and pushes a new HF repo for regression comparison.

The branch produces one new HF checkpoint:

| checkpoint | answers the question |
|---|---|
| `asg-ai/athena-cti-sft-qwen25-14b-v18-2-2` | When the v18.2.1 prob mix is reverted to v18.2's (Phase A 0.25 / Phase B 0.40 / TAA 0.35) **and** `--max-samples` is cut from 3000 to 1500 per dataset (−50% steps vs v18.2.1, −38% vs v18.2), does the resulting "smaller, not different" Stage 4 preserve the RMS gain (≥55.0 combined_f1) while reducing the MCQ damage (≥70.0 accuracy) and protect ATE / RCM from sliding below their floors — without regressing the eight axes that v18.2 already passes? |

The vintage directory is self-contained per project convention. Because
v18.2.2 is recipe-only, **no manifest, no row-count gate, and no build
artefacts** ship from this directory; the only documents are the plan,
the launch checklist, and this README. The training shards are reused
verbatim from `05112026/` (the v18.1 build).

```
05142026/
  README.md     this document
  READY         pre-launch status marker; go/no-go checklist; JSON shard
                inventory (all reused from v18.1; no scp required); run-book
                pointers
  plan.txt      master plan (motivation in §1, recipe deltas in §2, recipe
                in §3, falsification + sign-off gates in §4, escalation
                paths in §5, sign-off in §6); cross-references
                tmpl_gen/templates/05132026/v18_2_plan.txt §8 for the
                long-form v18.2.1 post-mortem and trade-ratio derivation
                rather than re-deriving them here
```

Predecessor vintage directories that v18.2.2 depends on:

| vintage | role |
|---|---|
| `05112026/` (v18.1) | Source of the three training shards (Phase A / Phase B / standalone TAA), already on the 4xH100 host. v18.1's `-v18-1-cse` checkpoint is the base model for v18.2.2's Stage 4. |
| `05132026/` (v18.2 / v18.2.1) | Source of the multi-shard recipe shape, the `interleave_under` rationale, the `--do_eval False --eval_strategy no` rationale, and §8 of `v18_2_plan.txt` (the v18.2.1 post-mortem and trade-ratio derivation that motivates the v18.2.2 recipe deltas). |

## 1. Why v18.2.2 exists — the v18.2 / v18.2.1 gate-package sub-floor

The v18.1 chain delivered the designed CSE uplift (CSE-TI +16 pp,
CSE-Malware +16 pp at Stage 3) but eroded RMS combined_f1 by 11.6 pp
at the CSE stage (v18.1-core 57.69 → v18.1-cse 46.34). v18.2 was a
Stage 4 touch-up over `-v18-1-cse` using a 3-shard interleave at
probs 0.25/0.40/0.35 (Phase A / Phase B / TAA) with `--max-samples
2400` per dataset (~1500 optimizer steps, ~80–100 min on 4xH100).
v18.2 recovered RMS to 54.72 (within 0.28 pp of the 55.0 gate) and
held 8 of 10 axes; MCQ stuck at 62.33 (gate 70.0).

v18.2.1 was a prob-and-step rebalance designed to recover MCQ
(Phase A 0.25 → 0.35) and close the RMS hairline (Phase B 0.40 →
0.45) while dropping TAA share (0.35 → 0.20) on the hypothesis that
the standalone TAA shard's short-form structured generation competed
with MCQ's letter-decoder pattern. Step count was bumped to
`--max-samples 3000` per dataset (~1667 steps, ~95–115 min). The
2026-05-14 bench made the result clear:

| axis            | v18.1-cse | v18.2 | v18.2.1 | gate   | v18.2.1 verdict |
|-----------------|----------:|------:|--------:|-------:|-----------------|
| MCQ             |     72.03 | 62.33 |   63.17 |  ≥70.0 | MISS (−6.83)    |
| RMS combined_f1 |     46.34 | 54.72 |   50.37 |  ≥55.0 | MISS (−4.63)    |
| ATE             |     58.60 | 63.20 |   62.40 |  ≥63.0 | MISS (−0.60)    |
| RCM             |     69.60 | 72.55 |   66.80 |  ≥67.5 | MISS (−0.70)    |
| VSP             |     76.29 | 83.87 |   82.65 |  ≥80.0 | PASS            |
| TAA Classic     |        —  | 47.50 |   47.00 |  ≥40.0 | PASS            |
| CSE-TI          |     36.07 | 41.25 |   41.79 |  ≥34.0 | PASS            |
| CSE-Malware     |     22.37 | 24.14 |   23.48 |  ≥20.0 | PASS            |
| CyberMetric-2K  |     87.70 | 88.95 |   89.35 |  ≥85.5 | PASS            |
| CyberMetric-10K |     83.40 | 83.94 |   84.17 |  ≥81.0 | PASS            |

v18.2.1 fails 4 gates vs v18.2's 2 — a strict regression of the gate
package. The trade-ratio diagnosis (full derivation in
`tmpl_gen/templates/05132026/v18_2_plan.txt §8.2.3`) compares the
MCQ-for-RMS exchange rate of each Stage 4 variant:

| variant         |    dMCQ |   dRMS | trade ratio (\|dRMS/dMCQ\|) |
|-----------------|--------:|-------:|----------------------------:|
| cse → v18.2     | −9.70   | +8.38  | 0.86                         |
| cse → v18.2.1   | −8.86   | +4.03  | 0.45                         |

v18.2.1's trade is **half as efficient** as v18.2's: Stage 4
multi-shard at higher prob/step counts loses RMS install efficiency
without buying back MCQ. The mechanism for the Phase B prob-bump
backfire is that Phase B is a 4-task shard (rms+ate+vsp+rcm), so
raising its interleave probability raises **all four** signals in
lockstep — the RMS-specific ratio inside the mix did not improve, and
the +22% step bump then over-trained the dilution. ATE −0.80 and
RCM −5.75 both regressing under the larger Phase B share confirm
this; both axes carry their own §7.4 floors and Phase B is their only
install vehicle in the Stage 4 mix.

The first-order implication: **the v18.2 prob mix was the right one**,
v18.2 was on the better side of the diminishing-returns curve, and
v18.2.1 pushed past the optimum on **both** probs and step count.

## 2. v18.2 → v18.2.2 deltas (recipe only; no corpus delta)

v18.2.2 is therefore RECIPE-DRIVEN, not corpus-driven. No template
or manifest changes. The Stage 1 / 2 / 3 corpora **and** the v18.2
dataset shards are untouched. v18.2.2 adds exactly one new training
launcher and one new bench wrapper; the only knob differences vs
v18.2 are `--max-samples` and a new push target:

| knob                  | v18.2          | v18.2.1        | **v18.2.2**       |
|-----------------------|----------------|----------------|-------------------|
| Phase A interleave prob | 0.25         | 0.35           | **0.25** (revert) |
| Phase B interleave prob | 0.40         | 0.45           | **0.40** (revert) |
| TAA interleave prob     | 0.35         | 0.20           | **0.35** (revert) |
| `--max-samples` (per ds) | 2400        | 3000           | **1500** (NEW)    |
| est. optimizer steps    | ~1500        | ~1667          | **~937**          |
| est. wallclock 4xH100   | ~80–100 min  | ~95–115 min    | **~50–65 min**    |
| HF push target          | `…-v18-2`    | `…-v18-2-1`    | **`…-v18-2-2`**   |

Unchanged across all three iterations: lr 1e-6, cutoff 16384, packing
off, effective batch 4, `mix_strategy=interleave_under`, gradient
checkpointing on, DeepSpeed offload on (4xH100 + cutoff 16384 +
packing off forces offload to avoid OOM), base model
`asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse`, datasets
`ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn` (Phase A),
`ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm` (Phase B), and
`ift_data_2026_05_11_v18p1_taa` (standalone TAA). v18.2.2 is a fresh
Stage 4 over v18.1-cse (NOT a continuation of v18.2 or v18.2.1); it
is directly comparable to both predecessors on every axis.

The hypothesis (full version in `05132026/v18_2_plan.txt §8.3`):
the regression is **over-exposure damage** from a too-long Stage 4
eroding the Phase A and CSE drill circuits. Reducing the step count
without changing probs should preserve the RMS gain while reducing
MCQ damage and protect ATE / RCM from sliding below their floors.
The `--max-samples` math (LlamaFactory applies the cap **per dataset
before** interleaving): 1500 / max(probs) = 1500 / 0.40 = 3750 final
training samples, ~937 optimizer steps at effective batch 4.

## 3. JSON SFT data artefacts (all reused; no scp required)

Because v18.2.2 is recipe-only, **no new shards** are produced and no
new shards need to be transferred to the 4xH100 training host. All
three datasets are reused **verbatim** from the v18.1 build
(`05112026/`) and were already shipped to the host during the v18.2 /
v18.2.1 runs. They are also already registered in
`SFT/data/dataset_info.json`.

| shard | path | size | interleave prob |
|---|---|---:|:---:|
| Phase A (MCQ-bearing; protects MCQ axis) | `SFT/data/ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn.json` | 258 MB (~246 MiB) | 0.25 |
| Phase B (catalog drill; primary RMS install) | `SFT/data/ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm.json` | 145 MB (~139 MiB) | 0.40 |
| TAA Classic (protects TAA Classic axis) | `SFT/data/ift_data_2026_05_11_v18p1_taa.json` | 24 MB (~23 MiB) | 0.35 |

Pre-flight verification on the training host (the launcher itself
aborts with `[FAIL] v18.2.2 multi-replay dataset missing` if any of
the three files is absent):

```bash
test -f ~/Glaukopis/SFT/data/ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn.json && \
test -f ~/Glaukopis/SFT/data/ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm.json && \
test -f ~/Glaukopis/SFT/data/ift_data_2026_05_11_v18p1_taa.json && \
echo "v18p2p2 shards present"
```

## 4. Run-book

### 4.1 Generation

**Not applicable.** v18.2.2 is recipe-only; the three Phase A / Phase B /
TAA shards are reused verbatim from the v18.1 build. See
`tmpl_gen/templates/05112026/README-18-1.md §3` for the original build
record (counts, gate margins, fingerprint cross-validation).

### 4.2 Dataset registration

The launcher resolves dataset names through `SFT/data/dataset_info.json`.
The v18p1 entries used by v18.2.2 are already registered:

- `ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn`
- `ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm`
- `ift_data_2026_05_11_v18p1_taa`

No edits to `dataset_info.json` are required.

### 4.3 Training (Stage 4; multi-shard replay)

Single launcher; auto-detects GPU count and forces DeepSpeed CPU
offload on for any non-8x configuration to avoid OOM at cutoff 16384
with packing off:

```bash
# ~50-65 min on 4xH100 80GB
bash SFT/autotrain/run_sft_qwen25_14b_v18p2p2_multi_replay.sh
# defaults to ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-2-2
# base of stage 4 : asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse
```

Stage 4 runs the three-shard interleave (cutoff 16384, packing off,
lr 1e-6, effective batch 4, `--max-samples 1500` per dataset, 1 epoch,
save every 200 steps, intra-training eval DISABLED via
`--do_eval False --eval_strategy no`). The eval-disable rationale is
the same as v18.2 / v18.2.1 (LlamaFactory's loader keys datasets by
name, so the 3:3 shard:prob alignment requirement cannot be satisfied
with the available val shards without a fresh data build for a
touch-up where sign-off is via the bench suites). Train loss is
logged at `logging_steps=5` throughout. Only the final merged model
is pushed to HF.

### 4.4 Bench

Standard 14B AthenaBench + CyberMetric + CyberSOCEval sweep against
`asg-ai/athena-cti-sft-qwen25-14b-v18-2-2` via the new wrapper
(2xH100, ~8 min full sweep):

```bash
BENCH_CONDA_ENV=ctibench bash SFT/test/utils/serve_and_bench_v18p2p2_multi_replay.sh
```

Sign-off criteria (full table in `plan.txt §4`; expected ranges in
parentheses are **predictions**, not gates):

- RMS combined_f1 ≥ 55.0  (v18.2: 54.72; expected 53–56)
- MCQ accuracy    ≥ 70.0  (v18.2: 62.33; expected 65–70)
- ATE             ≥ 63.0  (v18.2: 63.20; expected stable)
- VSP             ≥ 80.0  (v18.2: 83.87; expected stable)
- RCM             ≥ 67.5  (v18.2: 72.55; expected 70–73)
- TAA Classic combined ≥ 40.0  (v18.2: 47.50; expected 45–50)
- CSE-TI          ≥ 34.0  (v18.2: 41.25; expected stable or up)
- CSE-Malware     ≥ 20.0  (v18.2: 24.14; expected stable or up)
- CyberMetric-2K  ≥ 85.5  (v18.2: 88.95; expected stable)
- CyberMetric-10K ≥ 81.0  (v18.2: 83.94; expected stable)

If §4 passes, `asg-ai/athena-cti-sft-qwen25-14b-v18-2-2` is promoted
to the headline v18.2 ship by alias swap in
`SFT/test/pipelines/models.py` — **not** by overwrite of v18.1-cse,
v18-2, or v18-2-1 (those repos are retained for regression
comparison).

If MCQ recovers but RMS misses by 0.5–2 pp, the next iteration is a
short Phase-A-only anchor pass on top of v18.2.2 (escalation path (a)
in `plan.txt §5`). If both MCQ and RMS still miss and the trade
ratio is **worse** than v18.2's 0.86, Stage 4 multi-shard has hit its
ceiling and the next iteration is v18.3 — a Core-stage Phase B
re-tune at the source (escalation path (d); see
`05132026/v18_2_plan.txt §5.5(c)` for the Core-stage mechanics).
