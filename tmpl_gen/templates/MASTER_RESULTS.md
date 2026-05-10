# Master Benchmark Results -- Pinned Notes

Persistent scratch pad for the Glaukopis SFT vintages so a context-restored
agent can recover the most recent benchmark numbers without asking the user
to repaste them. Keep this file alongside the dated lineage subdirs --
agents reading `tmpl_gen/templates/README.md` should pick this up next.

> **Status:** the source paste from the user (2026-05-10) was truncated by
> conversation summarization. Only the tail of the table survived intact.
> The header row below shows just the column-group banner that survived;
> per-column subheaders (per-axis Strict-F1 / F1 / Acc breakdowns) were
> dropped and need to be reconstructed or repasted before any new row is
> added with confidence.

## Suite layout (column groups, partial)

```
[ AthenaBench ........................ ] [ CyberSOCEval .... ] [ CyberMetr... ]
```

AthenaBench axes in use across the v7..v17.1 sweep (per the dated plan
files): RMS, ATE, VSP, RCM, CKT (MCQ), SOC, CM, TAA, CSE.  CyberSOCEval
covers Malware-Analysis and Threat-Intel-Reasoning sub-suites.

## Surviving rows (verbatim, tab-separated)

```
asg-ai/athena-cti-sft-qwen25-14b-v14p1                    SFT      55.3  51.8  66.0  44.5  48.0  46.6  82.8   5.0  93.0  49.0  15.8  82.7  49.3  50.9  58.6   4.8  38.1  15.3  48.5  26.7  88.2  83.8  86.0  57.1  54.2
Qwen/Qwen2.5-14B-Instruct                                 INSTRUCT 36.8  16.0  57.1   6.2   6.2   6.2  71.8  16.0  88.0  52.0   .     .     .    34.0  40.0  23.3  53.0  47.2  68.5  41.2  91.0  85.9  88.5  56.5  48.7
asg-ai_athena-cti-sft-llama31-8b-abaligned-v8 (small)     SFT      62.5  46.4  63.4  39.8  42.9  41.6  83.7   8.0  91.0  49.5   .     .     .    50.6  57.9   7.0  42.4  16.2  46.8  28.1  53.8  52.2  53.0  46.3  47.5
```

Trailing fragment from a truncated higher-up row (vintage unknown):
`... 83.1  56.6  54.1`

## Cross-referenced anchors (from v18_plan.txt §1, authoritative)

| axis           | v12  | v16  | v17.1 | athena-v8 (8B) | v18 target |
|----------------|------|------|-------|----------------|------------|
| CKT (MCQ)      | 70.4 | 72.1 | 70.0  | 77.6           | 77.6       |
| ATE            | 55.1 | 56.9 | 56.6  | 50.6           | 61.0       |

These are the only per-axis numbers explicitly preserved in the lineage
docs.  The full per-axis table for v7, v9..v11, v13, v14, v15, v17 is in
the user's Excel master sheet and has *not* been mirrored into the repo.

## v18 architecture (pinned)

* Base: `Qwen/Qwen2.5-14B-Instruct`, full-parameter SFT (no LoRA).
* Pattern: v17.1-style **chained three-stage** SFT.
  1. **Stage 1 (Core)**  -- v12-shape recipe (Phase A broad + Phase B axis;
     no TAA in this stage).  Launcher:
     `SFT/autotrain/run_sft_qwen25_14b_v18_core.sh`.  Manifest:
     `tmpl_gen/templates/05132026/Sophia-CTI-Templates-v18.txt`.
  2. **Stage 2 (+TAA Classic)** -- chained off Stage 1 checkpoint, reuses
     the v16 TAA Classic manifest at
     `tmpl_gen/templates/05102026/Sophia-CTI-Templates-v16.txt`.
     Launcher: `SFT/autotrain/run_sft_qwen25_14b_v18_plus_taa.sh`.
  3. **Stage 3 (+CSE, final)** -- chained off Stage 2 checkpoint, reuses
     the v17.1 CSE manifest at
     `tmpl_gen/templates/05122026/Sophia-CTI-Templates-v17.1.txt`.
     Launcher: `SFT/autotrain/run_sft_qwen25_14b_v18_final.sh` (publishes
     the final `asg-ai/athena-cti-sft-qwen25-14b-v18-core-plus-taa-cse` checkpoint).
* Why this shape: the v12..v17.1 chain successfully recovered CSE-TI /
  CSE-Malware without forgetting AthenaBench heads, but CKT (MCQ) and ATE
  refused to lift.  v18 lifts the MCQ floor by ~50% and adds glossary-
  sourced MCQ + new ATE narratives in the v18 Core manifest to break the
  ceiling on those two axes specifically.
* Archived predecessor: an earlier monolithic v18 draft (single 3-phase
  recipe with TAA.CANON in Phase C) was retired after the v15 W1 audit
  fingered TAA.CANON as the wrong TAA flavour for AthenaBench TAA Classic.
  The dead launcher lives at
  `SFT/autotrain/run_sft_qwen25_14b_v18.sh.monolithic.bak`.

## Pinned project decisions (as of 2026-05-10, user-confirmed)

* **Active baseline model is `Qwen/Qwen2.5-14B-Instruct`.**  Llama-3.1-8B
  is no longer the comparison anchor for new sweeps; treat any open task
  that still names it as the baseline as superseded.  The v0..v8 Llama
  rows in the master results table stay as historical reference but new
  benchmark passes target the Qwen base + Qwen-derived SFT vintages
  (v9..v18).
* **DeepSeek frontier sweep is deferred**, not cancelled.  The
  `DeepSeek-V4-Pro-hf` partial-sweep numbers (~88.8% MCQ early result)
  are valid but no further DeepSeek work is queued; revisit only when
  the user re-opens the topic.  Same posture for the other `-hf`
  frontier aliases (`gpt5.5-pro`, `gemini-3.1-pro`, etc.) -- rate cards
  remain in `api_usage.PRICING_PER_1K` so they're ready when reactivated.
* **Cost-tracking patch is COMPLETE on disk** across the four files
  (`SFT/test/pipelines/api_usage.py`, `SFT/test/pipelines/models.py`,
  `SFT/test/inference.py`, `SFT/test/utils/_print_sweep_summary.py`).
  End-to-end smoke-test on 2026-05-10 confirmed `_render_cost_block`
  emits a markdown table and JSON block from a synthetic checkpoint,
  silently omitting tasks with no recorded usage (correct behavior for
  local vLLM / HF tasks).  The patch is committed and on `main`; an
  earlier session note that called it "uncommitted and paused" was
  stale.

## Open follow-ups (next time memory is restored)

1. Ask the user to repaste the full master results table (all rows + the
   real per-column header) and overwrite the verbatim block above.
2. Once v18 Stage 1 finishes, append a `v18-core` row here from the
   AthenaBench sweep before Stage 2 even kicks off, so the chain's
   intermediate numbers are captured (v17.1 missed this step).
3. **(Already done -- do not re-run.)** Baseline for the new active
   model `Qwen/Qwen2.5-14B-Instruct` is captured in the master results
   table above (the `INSTRUCT` row).  Both AthenaBench and the Qwen-side
   of CyberSOCEval / CyberMetric have numbers in that row; if a future
   memory-restored agent thinks it needs to "run the Qwen baseline,"
   it does not -- those values are the baseline.  The legacy plan to
   baseline `Llama-3.1-8B-Instruct` on CyberSOCEval is dropped.

## v18 HF naming convention (locked 2026-05-10)

The three v18 chained-stage HF repos use additive suffixes so each
checkpoint's lineage is self-evident from its name and so any of the
three can be benchmarked independently:

| stage | launcher | HF repo |
|---|---|---|
| 1 | `SFT/autotrain/run_sft_qwen25_14b_v18_core.sh` | `asg-ai/athena-cti-sft-qwen25-14b-v18-core` |
| 2 | `SFT/autotrain/run_sft_qwen25_14b_v18_plus_taa.sh` | `asg-ai/athena-cti-sft-qwen25-14b-v18-core-plus-taa` |
| 3 | `SFT/autotrain/run_sft_qwen25_14b_v18_final.sh` | `asg-ai/athena-cti-sft-qwen25-14b-v18-core-plus-taa-cse` |

History (do NOT re-run / do NOT re-rename):
* Stage 2 was originally pushed as `asg-ai/athena-cti-sft-qwen25-14b-v18-plus-taa`
  on 2026-05-10 and renamed in-place via the HF `repos/move` API to
  `…-v18-core-plus-taa` the same day.  The old name returns HTTP 307
  -> the new canonical name, so any pinned reference still resolves.
* Stage 3 launcher was originally `run_sft_qwen25_14b_v18.sh` (and
  briefly `run_sft_qwen25_14b_v18_cse.sh`) before settling on
  `run_sft_qwen25_14b_v18_final.sh` to surface that it is the v18
  publish step.  The pre-pivot monolithic v18 script lives at
  `SFT/autotrain/run_sft_qwen25_14b_v18.sh.monolithic.bak` and is
  unrelated to the chained launchers above.
