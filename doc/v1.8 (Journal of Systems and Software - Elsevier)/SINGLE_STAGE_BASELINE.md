# Single-stage baseline: is Reviewer 2's point closed?

*Prepared 2026-10-25 for the v1.8 (Journal of Systems and Software) submission. Every number below is reproduced by `analysis/v18_additions.py` (output in `data/v18_additions.json`) from the results workbook and the training shards, or is quoted from the v1.7 manuscript.*

## Decision (2026-10-25, after this assessment)

The authors chose to treat the **first stage of the build, the Core checkpoint, as the study's single-stage baseline**. It is a model fine-tuned once on the core shards, and it is retained and evaluated with the same harness and per-item records as every later checkpoint. v1.8 now:

- defines this baseline in §6.3;
- reports the full chain against it with paired item-level intervals in §8.3;
- answers RQ4 from it.

The later stages add **+3.2** points to the three-benchmark mean (paired interval +2.2 to +4.2) and change the following, all with intervals that exclude zero:

| Measure | Change |
|---|---|
| CyberSOCEval malware, strict | +18.6 |
| CyberSOCEval threat intelligence, strict | +7.9 |
| CKT | +2.7 |
| Balanced AthenaBench score | −2.3 |
| Canonical TAA | −15.0 |
| VSP | −5.3 |
| RCM | −2.7 |
| MMLU-Pro | −1.8 |

The earlier single-pass releases appear only as one paragraph of design history. The matched-control launcher stays in the repository (`931a087`) but is not used by the paper.

**What this framing does and does not answer.** It answers "what do the later stages add over a single-stage model?" with paired statistics. It does not answer Reviewer 2's literal request, a single-stage control "trained on an equivalent total token budget or update count", because the Core checkpoint never saw the TAA and CSE shards. v1.8 says this in one sentence, in the internal-validity threat (§11), and lists a single pass over all four shards as a future control (§12). If a reviewer or the committee presses the point, the launcher below runs that control.

The rest of this document is the assessment as written before the decision.

## Short answer

**Not yet.** v1.7, and v1.8 as drafted, answer the point by no longer claiming that staging helps and by reporting the single-stage runs the project already had. That removes the over-claim but leaves the line of attack open:

- The paper still proposes and describes a four-stage pipeline.
- None of the existing single-stage runs was trained on the same data as the chain.
- The only 32B single-stage run saw about twice the chain's CTI exposure.

Dr. Cardei's condition ("provided you can also claim you used comparable hyperparameters") is met for the optimizer settings but not for the data or the token budget. Reviewer 2 asked for exactly those: "no single-stage (monolithic) multi-task control model was trained on an equivalent total token budget or update count."

Closing the point takes one training run plus one benchmark sweep. The data already exist and the launcher is ready (see "The matched control" below).

## What the reviewer asked

From review COSE-D-26-05507, Reviewer 2, point 1:

> The central empirical claim surrounds a 4-stage SFT schedule. However, no single-stage (monolithic) multi-task control model was trained on an equivalent total token budget or update count. […] Without a matched single-stage control, it is impossible to verify whether the 4-stage schedule provides a genuine learning advantage or merely introduces unnecessary complexity.

Point 2 adds that hyperparameters change at every stage boundary. For a software-engineering journal, the phrase "merely introduces unnecessary complexity" is the sharper form of the objection.

## What the record contains: the early single-stage runs

### 32B: one run (release v7, 26 April 2026)

| Setting | Single-stage v7 | Staged chain v21 | Comparable? |
|---|---|---|---|
| Base, framework, optimizer | Qwen2.5-32B-Instruct, LlamaFactory full SFT, ZeRO-3, 8-bit AdamW, Liger, bf16, cosine, 5% warm-up | same | **yes** |
| Learning rate | 1e-5 | 1e-5 (Stage 1A), 5e-6 (1B, 2, 3), 3e-6 (4) | partly: first stage only |
| Effective batch | 16 | 16 / 8 | partly |
| Sequence limit, packing | 4,096 packed | 8,192 packed / 16,384 unpacked / 4,096 packed | partly |
| CTI corpus | v7, 180,533 rows (no CSE letter-set or glossary-MCQ families) | v21, 256,119 rows | **no** |
| Epochs | 3 | 1 per stage | **no** |
| CTI row presentations | 541,599 | 262,119 | **no** (2.07×) |
| CTI text presented | 539.0 M chars | 305.9 M chars (Stages 1–3) | **no** (1.76×) |
| General instruction data | 999 Alpaca rows | 240,000 Tülu-3 + 999 Alpaca | **no** |

Results on the measures both runs were evaluated on, with Classic attribution only:

| Model | CKT | ATE | RCM | RMS | VSP | TAA Cl. | A_Cl | CSE C | CM Y | B_Cl |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 52.9 | 18.2 | 59.5 | 7.6 | 78.8 | 48.0 | 44.2 | 42.6 | 90.2 | 59.0 |
| Single-stage v7 (3 epochs) | 71.0 | 57.4 | 62.1 | 68.5 | 83.4 | 52.5 | **65.8** | 39.6 | 88.6 | **64.7** |
| Staged chain, Recal-32B | 75.5 | 65.8 | 67.1 | 62.5 | 77.8 | 50.5 | **66.5** | 42.6 | 89.8 | **66.3** |

The workbook holds two evaluations of the base model. They differ by up to 2.0 points on a task and 0.7 on A_Cl, so differences of that size between runs evaluated at different times are noise.

### 14B: three single-pass runs and three phased ones (development base)

| Release | Shape | A_Cl (aligned) | B_Cl (3-benchmark) |
|---|---|---|---|
| Base | — | 40.0 | 56.6 |
| v7 | single pass, 3 epochs | 60.0 | 59.4 |
| v10 | single pass, 1 epoch | 58.4 | 60.6 |
| **v11** | **single pass, 1 epoch, same hyperparameters as the chain's Stage 1A** | **56.1** | **60.8** |
| v9 | two phases | 60.6 | 60.4 |
| **v12** | **three phases** | **63.2** | **60.2** |
| v21 | four stages | 64.9 | 62.3 |

**v11 → v12 is the closest pair.** v11 used the chain's Stage-1A hyperparameters (lr, sequence limit, packing, batch, general mixture and cap). But v12 also rebuilt four template families that had reached only 22–48% of their planned row counts in v11. The project's own plan recorded the confound at the time: "Single-pass viability -- conflated with corpus underbuild" (`tmpl_gen/templates/05052026/v12_plan.txt`).

### What the existing evidence suggests

- At 32B the single-stage run is within 0.7 points of the chain on the aligned benchmark.
- At 14B the phased releases lead on the aligned benchmark but not on the three-benchmark mean, because they score lower on CyberSOCEval.
- A realistic expectation is that the matched control lands near the chain.

Either outcome closes the attack, because the comparison will exist:

- **Chain ahead outside the bootstrap interval:** the schedule is justified on this data.
- **Tie, or control ahead:** the paper says so. Staging's value is then diagnostic and operational (attributable regressions, re-runnable stages), not accuracy. JSS's aims statement explicitly welcomes negative results.

The control also tests the interference mechanism. If it avoids the canonical-attribution and VSP collapses, that supports the paper's account of Stage 3.

## The matched control

**Data.** One epoch over exactly the rows the chain's Stages 1–3 consumed (497,118 rows):

- Core-A: 168,859 rows
- Tülu-3 SFT mixture, capped at 240,000 as in Stage 1A
- Alpaca demo: 999 rows
- Core-B: 49,581 rows
- TAA: 20,359 rows
- CSE: 17,320 rows

The ~6,000-row Stage-4 replay is omitted because its rows are already in the union. The shard files are on disk (`SFT/data/ift_data_2026_05_18_v21_*.json`), so no data rebuild is needed.

**Settings.** The Stage-1A configuration, which already governs 82% of those rows:

- learning rate 1e-5, one epoch, cosine schedule, 5% warm-up
- 8,192-token packed sequences, effective batch 16
- 8-bit AdamW, Liger, ZeRO-3, gradient checkpointing

By a character-length estimate, only about 8 Core-B rows exceed 8,192 tokens.

**Sensitivity run.** A second run at lr 5e-6, the rate of Stages 1B–3, brackets the one setting the narrow shards see differently.

**Selection rule (fix before looking at benchmarks).** Report both runs. The primary control is the one with the lower mean eval loss on `core_val`, `taa_val` and `cse_val`.

**Differences that remain (state them in the paper):**

- order of presentation
- the learning rate seen by the narrow shards
- fewer optimizer steps on Core-B rows, because they are packed here and were unpacked at batch 8 in the chain

**Evaluation.** The full suite, including Canonical TAA and MMLU-Pro, on the same 2×H100 vLLM settings. The run keeps per-item files, which allows paired bootstrap differences against Recal-32B (`doc/v1.5 (Computers & Security)/analysis/` bootstrap script).

**Cost (planning estimate, not metered).**

| Item | Estimate |
|---|---|
| Training, one run, 8×H100 | ~18–30 h (the Stage-1A figure of 16–18 h plus the packed Core-B/TAA/CSE rows) |
| Upper bound per run | the chain's own Stages 1–3 budget, 48–60 h |
| Training cost at $2.50 per GPU-hour | ~$360–600 per run; ≤ $1,200 at the bound |
| Evaluation | the chain's checkpoints took ~15–17 min each on 2×H100 per their run manifests, plus server start-up |

**Commands.** The dry run was checked locally on 2026-10-25.

```bash
# training host, 8xH100, conda env llm-sft
cd SFT/autotrain
./run_sft_qwen25_32b_v21_single_stage.sh                # primary, lr 1e-5 -> asg-ai/...-v21-single-stage
./run_sft_qwen25_32b_v21_single_stage.sh --lr 5e-06     # sensitivity      -> asg-ai/...-v21-single-stage-lr5e-06

# eval host, 2xH100, conda env vllm
BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/serve_and_bench_qwen25_32b_v21_single_stage.sh
MODEL_ALIAS=athena-cti-sft-qwen25-32b-v21-single-stage-lr5e-06-vllm \
  BENCH_CONDA_ENV=ctibench bash SFT/eval/utils/serve_and_bench_qwen25_32b_v21_single_stage.sh
```

New files:

- `SFT/autotrain/run_sft_qwen25_32b_v21_single_stage.sh`
- `SFT/eval/utils/serve_and_bench_qwen25_32b_v21_single_stage.sh`
- two aliases in `SFT/eval/pipelines/models.py`

None of these is committed yet.

## Corrections already made in v1.8 to how v1.7 described the 32B run

- v1.7's table caption called the run "one pass"; it was **one stage of three epochs**.
- v1.7 said the run's "update budget … is comparable to the staged chain's". That holds only for total rows (544,596 vs ≈503,000). It saw about twice the CTI rows and almost no general data. v1.8 drops the "comparable budget" claim; §6.3 describes the run as one stage of three epochs on an earlier corpus, as design history only.

## Recommendation (before the decision above)

Run the primary control (lr 1e-5) before submitting to JSS. Add the sensitivity run if the budget allows.

Submitting v1.8 as it stands is honest, but it invites the same objection from the first reviewer who reads Section 6.3, and Dr. Cardei expects committee members to raise it too.

Lower-priority controls, in order:

1. Interleaved Stage 2+3, which tests the remedy for the attribution collapse.
2. Three seeded Stage-4 runs.
