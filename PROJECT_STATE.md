# Glaukopis Project State

Workspace-rooted continuity file. **Read this first** at session start
(or after VS Code restart) to recover SFT lineage, current benchmarking
front, and in-flight work without rebuilding context from chat history.

Authoritative sources always win over this file. Whenever a section
points at a per-vintage README or manifest, treat that file as truth
and treat this one as an index. Update protocol is in §7.

## 1. Project at a glance

- **Goal**: SFT a CTI-specialised assistant; evaluate on AthenaBench
  (CKT/MCQ, RCM, ATE, VSP, RMS, TAA Classic + Canonical) plus
  CyberMetric 2K/10K, CyberSOCEval (Malware + Threat-Intel), and
  MMLU-Pro as the cross-architecture general-reasoning anchor.
- **Production base model**: **`Qwen/Qwen2.5-32B-Instruct`** (dense
  32B). The 14B base (`Qwen2.5-14B-Instruct`) is retained as the
  matched-baseline + faster-iteration anchor; Llama-3.1-8B-Instruct,
  Foundation-Sec-8B-Instruct, and Qwen3-30B-A3B-Thinking-2507 (MoE,
  30.5B total / 3.3B active) are ported architecture comparators.
- **Latest shipped vintage**: **v21** — 5-architecture chain
  (Core → TAA → CSE → Recalibrate/Recal-32b), built as a SHA-verified
  byte-clone of v18.1 templates + gates. Cross-architecture ship
  checkpoint: **`asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b`**
  (Total **65.0** / Weighted **62.9** under the 50/50 TAA blend, tops
  all 14B / 8B / MoE peers on absolute leaderboard).
- **Per-architecture ship checkpoints (v21):**
  - **dense 32B (ship)**: `qwen25-32b-v21-recal-32b` — 32B-tuned recipe (LR 3e-6, Phase-B-heavy mix)
  - dense 14B: `qwen25-14b-v21-recalibrate` — Total 61.0 / Weighted 59.6
  - MoE 30B / 3.3B-active: `qwen3-30b-a3b-thinking-2507-v21-cse` — Total 63.4 / Weighted 60.9 (Stage 4 closed; MoE expert-routing failure mode)
  - 8B Foundation-Sec: `foundation-8b-v21-recalibrate` — Total 53.5
  - 8B Llama-3.1: `llama31-8b-v21-recalibrate` — Total 49.8
- **Results store**: **`SFT/eval/Glaukopis Results.xlsx`** is the
  canonical live spreadsheet covering the full CTI benchmark suite
  (AthenaBench + CyberMetric + CyberSOCEval) plus MMLU-Pro plus
  cost numbers per row. Dated snapshots (e.g. `Glaukopis Results -
  05272026.xlsx`) are frozen for posterity. Companion tracked CSVs:
  `SFT/eval/responses/cost_summary.{csv,tsv}` and
  `SFT/eval/responses/mmlu_pro_summary.{csv,tsv}`.
- **Paper sources**: `doc/` holds the LaTeX manuscripts, figures,
  bibliographies, and rendered PDFs for the Glaukopis paper across
  three publisher formats: `doc/ACM/`, `doc/IEEE/`, `doc/Oxford/`
  (each with its own `figures/` tree). LaTeX build caches
  (`.texpadtmp/`) and packaged drafts (`*.zip`) are gitignored.
  Active ICTAI/IEEE draft is `doc/IEEE/glaukopis_ieee_abridged-v1.3.tex`.
  Material is governed by the top-level `LICENSE.txt` (PolyForm
  Noncommercial 1.0.0).

## 2. SFT lineage (authoritative pointers)

Every vintage directory under `tmpl_gen/templates/<MMDDYYYY>/` holds
the manifest, plan, row-count gate, and (from v8 onwards) a per-
vintage README that is the source of truth for that vintage. Per-
vintage detail belongs there; the table below is the index — collapsed
to milestones for brevity, see git log under `tmpl_gen/templates/`
for the full v0..v21 chain.

| ver | date       | dir                            | base               | role / headline result |
|-----|------------|--------------------------------|--------------------|-----------------------|
| v0  | 2026-03-22 | `04022026/`*                   | L31-8B             | M/A/W/V/S/P/E/X core (still in every later manifest as Section C) |
| v7  | 2026-04-26 | `04262026/`                    | L31-8B             | **L31-8B RMS=62.64 strict F1** (the historical 8B anchor) |
| v8  | 2026-04-29 | `04292026/`                    | Q25-14B            | JSON / long-context scaffolding; **MCQ=77.6 14B peak** |
| v9  | 2026-04-30 | `04302026/` (`*v9_rms.txt`)    | Q25-14B            | **RMS=65.8** two-phase peak; recipe substrate carried forward |
| v12 | 2026-05-05 | `05052026/`                    | Q25-14B            | 4-generator pass; total **57.3** |
| v17.1 | 2026-05-10 | `05102026/` (README-17-1.md) | Q25-14B            | CSE letter-set `Shuffle: mcq_multi` fix |
| v18.1 | 2026-05-11 | `05112026/` (README-18-1.md) | Q25-14B            | 3-stage chain (Core+TAA+CSE); **58.9 14B baseline** |
| v19 / v20 | 2026-05-15..17 | `05152026/`, `05162026/` | Q25-14B          | v18.1-replay attempts (58.5 / 57.9); flagged data-build non-determinism as the unresolved variable |
| **v21** | **2026-05-18** | **`05182026/`** (README-21.md) | **Q25-32B (ship)** + 14B + MoE + 8B (×2) | **Byte-clone of v18.1; 5-architecture chain; Stage-4 recipe split (`recal-32b` is the 32B-tuned variant and the cross-arch ship)** |

*v0 per-family design strategy is documented in `tmpl_gen/templates/README.md`.

Model registrations live in `SFT/eval/pipelines/models.py` under
`model_mapping` (search `athena-cti-sft-`); aliases follow
`...-{ver}-vllm` for local vLLM serving and `...-{ver}-hf` for the HF
Inference Providers route. Qwen3 MoE rows carry both a thinking-on
`-vllm` alias and a matched-conditions `-no-think-vllm` alias against
the same HF repo for the A/B isolation pattern (see README-21.md
§"Matched-conditions base baseline").

## 3. v21 architecture ports

v21 is the first vintage ported across multiple base architectures
using the **same three shard files** (Core / TAA / CSE) after dedup.
Per-architecture variation is in optimiser / `--template` / HF push
target only, not dataset content — so one contamination check per
shard certifies all five ports of that shard.

| port              | base                                      | best stage              | Total | Weighted | notes |
|-------------------|-------------------------------------------|-------------------------|------:|---------:|-------|
| dense 32B         | `Qwen/Qwen2.5-32B-Instruct`               | **`recal-32b`** (ship)  | **65.0** | **62.9** | 32B-tuned Stage 4 (LR 3e-6, Phase-B-heavy, max-samples 3600). |
| dense 14B         | `Qwen/Qwen2.5-14B-Instruct`               | `recalibrate`           | 61.0 | 59.6     | 14B-recipe Stage 4 (LR 1e-6, balanced 0.25/0.40/0.35 mix). |
| MoE 30B           | `Qwen/Qwen3-30B-A3B-Thinking-2507`        | `cse` (Stage 4 closed)  | 63.4 | 60.9     | Stage-4 sweeps perturb expert routing; chain closed at CSE for this arch. |
| 8B (Foundation-Sec) | `fdtn-ai/Foundation-Sec-8B-Instruct`    | `recalibrate`           | 53.5 | —        | Domain-prepared 8B base. |
| 8B (Llama-3.1)    | `meta-llama/Llama-3.1-8B-Instruct`        | `recalibrate`           | 49.8 | —        | Architecture-transfer reference. |

Stage 4 recipe split (dense-32B only) — two parallel branches off `v21-cse`, both retained on disk for the recipe A/B:
- `recalibrate` — 14B-recipe verbatim port; **fails** VSP recovery at 32B (78.9→75.7)
- **`recal-32b`** — 32B-tuned recipe (3× LR, 0.15/0.60/0.25 probs, max-samples 3600) — **ship**

Launchers: `SFT/autotrain/run_sft_{qwen25_14b,qwen25_32b,qwen3_30b_a3b_thinking,foundation_8b,llama31_8b}_v21_{core,taa,cse,recalibrate,recal_32b,chain}.sh`.
Per-stage bench wrappers: `SFT/eval/utils/serve_and_bench_<arch>_v21_<stage>.sh`.
Sweep aggregator: `SFT/eval/utils/run_v21_sweep.sh`.

## 4. Build pipeline & training recipes

For each vintage, the per-version README §1 (build pipeline) and §4
(training recipe) are authoritative. Cross-cutting references:

- **Manifest syntax / template DSL**: `tmpl_gen/templates/README.md`
- **End-to-end SFT pipeline doc**: `SFT/README.md`
- **Inference / Eval pipeline doc**: `SFT/eval/README.md`
- **Cost aggregator**: `SFT/eval/utils/build_cost_summary.py` → `SFT/eval/responses/cost_summary.{csv,tsv}` (tracked)
- **AthenaBench task definitions**: `SFT/eval/pipelines/evaluation/`
- **CyberSOCEval scoring (Jaccard)**: `SFT/eval/pipelines/evaluation/cybersoceval_eval.py`
- **Phase split logic**: `tmpl_gen/scripts/split_corpus_for_phases.py`
- **Licence-allowlist gate**: `tmpl_gen/scripts/check_corpus_licences.py`
- **MISP TAA generator**: `tmpl_gen/scripts/misp_taa_generator.py`
- **Verbatim dedup against eval set** (n=13, hit=1, drop=50): `tmpl_gen/scripts/dedup_against_evals.py` — invoked as Phase 5 of every v21 shard watcher (`_v21_{core,taa,cse}_build/watcher.sh`)
- **Contamination posture**: README-21.md §"Contamination posture" is authoritative for v21 (and back-ported to v8.1..v20 per commit `2e46df8`); see §"Per-benchmark contamination matrix" for the verbatim-vs-structural framing per eval source

## 5. Frontier / baseline benchmarking

Parallel track to the SFT lineage: comparing v21 checkpoints against
hosted frontier models on AthenaBench / CSE / CM / MMLU-Pro via the
HF Inference Providers route (`router.huggingface.co/v1`), driven by
`SFT/eval/inference.py` with `-hf` aliases.

- **Per-token cost tracking**: complete and committed on `main` across
  `SFT/eval/pipelines/{api_usage.py,models.py}` +
  `SFT/eval/inference.py` + `SFT/eval/utils/_print_sweep_summary.py`.
  Per-row spend rolls into `cost_summary.{csv,tsv}`; cost-revalidation
  chain wrapper at `SFT/eval/utils/run_cost_revalidation_chain.sh`.
- **Rate cards**: `SFT/eval/pipelines/api_usage.py:PRICING_PER_1K`.
  Confirmed entries include `gpt-5.5` ($5/$30), `gpt-5.2` ($1.75/$14),
  `gemini-3-flash` ($0.50/$3), `gemini-2.5-flash`, `deepseek-v4-pro-hf`
  ($1.74/$3.48), `deepseek-v3.2-exp-hf` ($0.27/$0.40),
  `deepseek-v3.1-terminus-hf`. Self-hosted vLLM rows use the
  2×H100 @ $2.50/hr GPU-hour basis.
- **Latest cost picture** (per full sweep = AthenaBench + CSE + CM +
  MMLU-Pro, from `cost_summary.csv`):
  `gemini-3-flash` ~$1,154 (verbose output blows the budget);
  `gpt-5.5` ~$508; `gpt-5.2` ~$210;
  self-hosted `qwen3-30b-MoE` ~$192; `deepseek-v4-pro-hf` ~$126;
  `deepseek-v3.2-exp-hf` ~$11; every self-hosted Q25-32B v21 SFT
  checkpoint lands at **~$2–3** end-to-end.
- **Abandoned comparators** (do not retry without protocol change):
  `gemma-4-*-hf`, `kimi-k2.6-*-hf` (502/504 storm), `qwen3.5-plus-hf`
  (empty-content from thinking-mode channel split).

## 6. In-flight work threads

State as of 2026-06-24. Update §7 protocol when these change.

1. **IEEE ICTAI 2026 paper revision** — active source is
   `doc/IEEE/glaukopis_ieee_abridged-v1.3.tex`. The current draft
   softens novelty claims, removes the internal `cardei2025ift`
   reference, expands limitations, adds real qualitative examples from
   v21 response artifacts, compresses Ariadne operational details, adds
   result interpretation for ATE/TAA/VSP/recalibration, and includes a
   CyberPal/SecKnowledge positioning table.
2. **Response artifacts** — real v21 stage response JSONLs used for the
   qualitative table live locally under `SFT/test/responses/` and should
   remain untracked unless explicitly requested. Do not silently migrate
   these artifacts to `SFT/eval/responses/`; this bundle came from the
   remote test path.
3. **Open evidence gaps for the paper** — still missing matched
   single-stage-vs-chain training, repeated-seed/significance analysis,
   and a formal metric-level error taxonomy for VSP/TAA/ATE. These are
   acknowledged as limitations rather than implied claims.

## 7. Update protocol

This file must be touched whenever any of the following happens:
- A new SFT vintage starts (add row to §2; create new vintage dir).
- A vintage's status flips (in-flight → completed/failed; update §1
  "latest shipped" + the §2 row + §3 port table).
- A new architecture port lands (extend §3).
- A frontier comparator is added or abandoned (§5).
- An in-flight thread closes or a new one opens (§6).
- The cost-rate cards drift (§5 third bullet).

Per-vintage detail belongs in the vintage's own README, **not here**.
This file is an index; keep it under ~200 lines so it loads in one
`view` call. When in conflict with a per-vintage README or with git
history, defer to those — and update this file to match.

Quick verification commands at session start:
```bash
git log --oneline -15                      # what's been committed since
git status --short                         # any uncommitted edits
ls tmpl_gen/templates/ | tail -5           # most recent vintages on disk
grep -nE "v21" SFT/eval/pipelines/models.py | tail -15
head -25 SFT/eval/responses/cost_summary.csv
```
