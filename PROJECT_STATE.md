# Glaukopis Project State

Workspace-rooted continuity file. **Read this first** at session start
(or after VS Code restart) to recover SFT lineage, current benchmarking
front, and in-flight work without rebuilding context from chat history.

Authoritative sources always win over this file. Whenever a section
points at a per-vintage README or manifest, treat that file as truth
and treat this one as an index. Update protocol is in §6.

## 1. Project at a glance

- **Goal**: SFT a CTI-specialised assistant; evaluate on AthenaBench
  (6 tasks: RMS / ATE / VSP / RCM / MCQ / SOC) plus CKT, CyberMetric
  and (new) CyberSOCEval.
- **Base model lineage**: Llama-3.1-8B-Instruct (v0–v8), then Qwen2.5-14B
  (v8small/v8/v8.1/v9/v10/v11/v12/v13) with 32B serial after 14B passes
  the per-vintage §6 / §8 gates.
- **Latest validated checkpoint**: **v12** (`asg-ai/athena-cti-sft-qwen25-14b-v12`)
  — 14B sweep weighted total **57.3** (passed v12 plan §8 with caveats:
  RMS regression vs v9, SOC regression vs v11, Phase C net-negative on TAA).
- **Latest in-flight checkpoint**: **v13** — first best-of-vintage
  composition; targets weighted total ≥ 58.5 with §6 floors per axis.
  See `tmpl_gen/templates/05072026/README.md` §6 for pass criteria.

## 2. SFT lineage (authoritative pointers)

Every vintage directory holds the manifest, plan, row-count gate, and
(from v8 onwards) a per-vintage README that is the source of truth for
that vintage. The README in `tmpl_gen/templates/05072026/` carries the
canonical `v0 → v13` table in §7 — defer to it on any conflict.

| ver | date       | dir                              | base    | corpus       | headline result / role |
|-----|------------|----------------------------------|---------|--------------|------------------------|
| v0  | 2026-03-22 | `tmpl_gen/templates/04022026/`*  | L31-8B  | hand-crafted | M/A/W/V/S/P/E/X core (still in every later manifest as Section C) |
| v3  | 2026-04-23 | `tmpl_gen/templates/04232026/`   | L31-8B  | abaligned v4 | first AthenaBench-aligned slate |
| v4–v6 | 2026-04-24..25 | `04242026/`, `04252026/`   | L31-8B  | abaligned    | distractor pattern + AB.* family established |
| v7  | 2026-04-26 | `tmpl_gen/templates/04262026/`   | L31-8B  | combined     | **L31-8B RMS=62.64 strict F1** (the 8B baseline) |
| v8  | 2026-04-29 | `tmpl_gen/templates/04292026/`   | Q25-14B | small+large  | JS.* (JSON), long-context scaffolding; **MCQ=77.6 14B peak** |
| v8.1| 2026-04-30 | `tmpl_gen/templates/04302026/`   | Q25-14B | single-pass  | RMS catalog-collapse fix; AB.RMS.{4,5} `Count:` floors |
| v9  | 2026-04-30 | `tmpl_gen/templates/04302026/` (`*v9_rms.txt`) | Q25-14B | two-phase | **RMS=65.8** (v13 baseline target); recipe: cutoff 8192, packing ON, batch 16 |
| v10 | 2026-05-01 | `tmpl_gen/templates/05012026/`   | Q25-14B | 200,340      | single-pass; HTML sanitiser; **VSP=86.7 14B peak**; total=54.1 |
| v11 | 2026-05-03 | `tmpl_gen/templates/05032026/`   | Q25-14B | 198,994      | SOC.* + TAA.CANON.* + D3FEND v1.4.0; **SOC=44.7 14B peak**; total **54.3 (failed §8)** |
| v12 | 2026-05-05 | `tmpl_gen/templates/05052026/`   | Q25-14B | 260,589      | 4 generators (TAA.CANON/CM/MCQ.EXT/SOC.GEN); 3-phase recipe; total **57.3** (RMS=63.3, VSP=84.6, SOC=39.3 regressions) |
| **v13** | **2026-05-07** | `tmpl_gen/templates/05072026/` | Q25-14B | **~275,000 (target)** | **In flight.** Best-of-vintage (RMS←v9, MCQ←v8, VSP←v10, SOC←v11); MISP CC-0 TAA expansion (~12K rows); licence-allowlist gate; reverts to 2-phase v9-shape recipe |

*v0 substrate per-family design strategy is documented in
`tmpl_gen/templates/README.md`.

Model registrations live in `SFT/test/pipelines/models.py` under
`model_mapping` (search `athena-cti-sft-`); aliases follow
`...-{ver}-vllm` for local vLLM serving and `...-{ver}-hf` for the HF
Inference Providers route.

## 3. Build pipeline & training recipes

For each vintage, the per-version README §1 (build pipeline) and §4
(training recipe) are authoritative. Cross-cutting references:

- **Manifest syntax / template DSL**: `tmpl_gen/templates/README.md`
- **End-to-end SFT pipeline doc**: `SFT/README.md`
- **Inference / Eval pipeline doc**: `SFT/test/README.md`
- **CyberSOCEval scoring (Jaccard)**: `SFT/test/pipelines/evaluation/cybersoceval_eval.py`
- **AthenaBench task definitions**: `SFT/test/pipelines/evaluation/`
- **Phase split logic (v12+)**: `tmpl_gen/scripts/split_corpus_for_phases.py`
- **Licence-allowlist gate (v13+)**: `tmpl_gen/scripts/check_corpus_licences.py`
- **MISP TAA generator (v13+)**: `tmpl_gen/scripts/misp_taa_generator.py`

## 4. Frontier / baseline benchmarking front

Parallel track to the SFT lineage: comparing v-checkpoints against
hosted frontier models on AthenaBench via the HF Inference Providers
route (`router.huggingface.co/v1`), driven by `SFT/test/inference.py`
with `-hf` model aliases.

- **Active comparator**: `deepseek-v4-pro-hf` — superior stability
  vs Together/Fireworks routes; 100% accuracy in initial MCQ probes;
  AthenaBench sweep in flight.
- **Abandoned comparators** (do **not** retry without protocol change):
  - `gemma-4-*-hf` and `kimi-k2.6-*-hf`: high 502/504 rate on HF Router
    under parallel load.
  - `qwen3.5-plus-hf`: empty-content responses from thinking-mode
    channel split; needs `extra_body={"enable_thinking": False}` or
    `reasoning_content` fallback before re-attempt.
- **Token-cap fix**: `HFInferenceModel.generate` now floors
  `max_tokens` at 2048 (was 128/256, sized for terse local models;
  starved hosted frontier models). See `SFT/test/pipelines/models.py`.
- **Cost tracking**: rate cards in `SFT/test/pipelines/api_usage.py`
  (`PRICING_PER_1K`); `add_tokens` warns rather than crashes on
  unknown models. DeepSeek-V4-Pro $1.74/$3.48 per 1K in/out;
  DeepSeek-V3.2-Exp $0.27/$0.40.

## 5. In-flight work threads

State as of 2026-05-07. Update §6 protocol when these change.

1. **v13 14B SFT sweep** — manifest assembled (3,337 lines, 235
   shortnames, 249 templates emitted); raw corpus built
   (`SFT/data/ift_data_2026_05_07_v13.raw.json`, 204,695 rows,
   249 MB). **Watcher phases 2–9 not yet authored**; dataset
   registration + two-phase launcher are committed (commit `423cfe2`).
   Blocker for production sweep: `_v13_build/watcher.sh` needs to be
   written before phases 3–3e can run.
2. **DeepSeek-V4-Pro AthenaBench sweep** — in flight on the HF
   Inference Providers route. Monitor for routing stability and
   completion before scoring against v12/v13.
3. **CyberSOCEval baseline (Llama-3.1-8B-Instruct)** — data confirmed
   present on remote host (609 malware + 588 threat-intel questions
   under `SFT/test/benchmark_data/cybersoceval/`). Pending: `--rows 2`
   probe against `cybersoceval-malware` and `cybersoceval-ti` to
   verify routing, prompt assembly for large reports, and the
   json-block answer extraction end-to-end.
4. **Cost-tracking instrumentation** — `api_usage.py`, `models.py`,
   `inference.py` edits made in-session. Working tree is clean as of
   2026-05-07; verify with `git log -p api_usage.py | head -80` that
   the DeepSeek rate cards and graceful `add_tokens` are committed.
   `_print_sweep_summary.py` markdown cost block is the next
   deliverable on this thread.

## 6. Update protocol

This file must be touched whenever any of the following happens:
- A new SFT vintage starts (add row to §2, point at the new dir).
- A vintage's status flips (in-flight → completed/failed; update §1
  "latest validated" and the §2 row).
- A frontier comparator is added or abandoned (§4).
- An in-flight thread closes or a new one opens (§5).

Per-vintage detail belongs in the vintage's own README, **not here**.
This file is an index; keep it under ~150 lines so it loads in one
`view` call. When in conflict with a per-vintage README or with git
history, defer to those — and update this file to match.

Quick verification commands at session start:
```bash
git log --oneline -15                      # what's been committed since
git status --short                         # any uncommitted edits
ls tmpl_gen/templates/ | tail -5           # most recent vintages on disk
grep -n "athena-cti-sft-qwen25" SFT/test/pipelines/models.py | tail -10
```
