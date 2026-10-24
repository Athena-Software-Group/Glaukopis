# Glaukopis v1.5 — verification memorandum and status of the open items

**Prepared:** 2026-09-06
**Target venue:** Elsevier *Computers & Security* (single-anonymized review; author-year references; elsarticle class)
**Source:** `doc/v1.4 (abridged)/glaukopis_abridged_v1.4.tex` (ChatGPT author-review draft of 2026-09-05), rewritten as `glaukopis_cs_v1.5.tex` after re-deriving every number from the primary artifacts in this repository.
**What was not done:** no model training and no model inference. All new numbers are recomputed from artifacts that already existed (workbook, run manifests, cost ledger, build reports, per-item prediction files).

## 1. Verification of the v1.4 numerical claims

| v1.4 claim (from the 2026-09-05 memo) | Verdict | Evidence |
|---|---|---|
| The AthenaBench composite ("AB Avg II") of the SFT rows was computed with Classic TAA only, while the caption claimed a 50/50 blend | **Confirmed, and the mechanism is now known.** In `SFT/eval/Glaukopis Results.xlsx`, sheet `Final wMMLUPro-v21`, column V is a formula over T (the 50/50 blend) for the hosted rows, a hard-coded Classic-only value for the five 32B SFT rows (e.g. V12 = 66.533 = mean of the five other tasks and Classic TAA), and a P-based (Classic) formula for the base row. Column E ("Avg CTI Benchmark") inherits the mismatch. The sheet `Final  (v21) 14B 32B` uses the blend for every row: ship 65.0028, base 57.694. | `analysis/make_tables.py` → `data/aggregates_uniform_convention.csv` (columns `sheet_AB_AvgII_asis`, `B_bal_C3`) |
| Balanced three-benchmark mean 65.0 (ship) vs 57.7 (base) | **Confirmed** (65.003 / 57.694). Gain 7.31. | same |
| "Avg Total (w/MMLU)" does not follow the stated (10×CTI-test + MMLU)/11 rule | **Confirmed; the true rule is now identified.** It is a 9-component mean of MMLU-Pro, CKT, ATE, RCM, RMS combined-F1, VSP, TAA blend, CSE composite and CM mean (base 49.31, ship 62.83). The paper no longer uses it. "Avg CTI Test" is a consistent 10-test mean (strict RMS F1, strict CSE accuracies, TAA blend) and is reported as the "ten-test mean". | same |
| Frontier rows need reconciliation before comparison | **Resolved.** Hosted rows were always computed with the blend, so the corrected local rows are directly comparable: GPT-5.5 72.9, DeepSeek-V4-Pro 70.4, Gemini-3-Flash 69.8, GPT-5.2 67.5, DeepSeek-V3.2-Exp 66.0, ship 65.0, Gemini-2.5-Flash 63.4, base 57.7. | same |
| Cost totals may not be single-pass costs | **Confirmed and quantified.** `build_cost_summary.py` globs `summary_*.json`, which matches the model-level `summary_model.json` in addition to the per-suite files, so every local row is exactly 2.000× the run-record elapsed time (verified for all five 32B checkpoints). Ship single pass: 1,020 s on 2×H100 = $1.42 (CTI-only $1.02) vs ledger $2.83. API rows are built from a cumulative checkpoint and may include reruns; the remote run records were not available. | `data/local_cost_run_records.csv` |
| Contamination filter drops only rows with ≥50 shared 13-grams | **Confirmed and quantified.** Core: 219,675 scanned, 34,897 flagged (15.9%), 684 removed; TAA: 0 flagged; CSE: 41 flagged, 0 removed. | `data/build_reports_v21.csv`, `_v21_*_build/dedup_report.json` |
| Filter coverage of TSV/CyberSOCEval unclear | **Resolved: not covered.** The index held 18 JSON/JSONL files (AthenaBench task files, CyberMetric, CVE and URLhaus feeds). CyberSOCEval is fetched at run time into a directory that did not exist at build time, and the CTIBench TSVs are outside the glob. The dedup reports contain no CyberSOCEval or CTIBench hits. `README-21.md` overstates coverage. | `dedup.log` ("indexing 18 eval files"), `SFT/eval/utils/fetch_cybersoceval_data.py` |
| Evaluation snapshot date (May 24 vs June 24) | **Resolved from run manifests.** CTI suites for the five 32B checkpoints ran 2026-05-20T23:29Z to 2026-05-21T19:26Z; MMLU-Pro on 2026-05-23 (ship) and 2026-05-26 (others). The workbook file is dated 2026-06-24. | `SFT/test/responses/*/summary_*_all_v1.json` |
| Training timings | **Partially resolved.** No trainer logs are in the repository. `SFT/SFT_FLOW.md` documents a planning budget: Core 26–30 h, +TAA 13–17 h, CSE 9–13 h, Recal-32b 3–4 h; total 51–64 h on 8×H100 (408–512 GPU-h; $1,020–$1,280 at $2.50/GPU-h). Reported as a planning figure, not a metered bill. | `SFT/SFT_FLOW.md`, `SFT/README.md` |
| Hyperparameter table | **Verified against launchers.** 1A: lr 1e-5, cutoff 8192, packing on, max-samples 240,000 (per dataset), eff. batch 16; 1B: 5e-6, 16384, off, 70,000, 8; TAA/CSE: 5e-6, 4096, on, 33,000, 16; Recal-32b: 3e-6, 16384, off, 3,600 per shard, probs 0.15/0.60/0.25; Recalibrate (14B recipe): 1e-6, 2,400, 0.25/0.40/0.35. The v1.3 figure's annotations were wrong and the figure was replaced. | `SFT/autotrain/run_sft_qwen25_32b_v21_*.sh` |
| Bibliography | **Updated and re-verified entry by entry on 2026-09-06 (DBLP, ACL Anthology, arXiv, proceedings.com, MITRE PDF).** Corrections made in that pass: LlamaFactory author list reduced to the five authors of the ACL 2024 demo record; CTIBench pages 50805–50825 and DOI 10.52202/079017-1607 added; MMLU-Pro DOI 10.52202/079017-3018 added; MITRE report noted as MP180360 (July 2018, revised March 2020); the OLMo citation re-attributed to paragraph-level leakage removal because that paper does not use 13-gram screening (GPT-3 does). **Earlier updates:** AthenaBench → ACSAC Workshops 2025, pp. 443–450, DOI 10.1109/ACSACW69556.2025.00072 (DBLP). CyberMetric → IEEE CSR 2024, pp. 296–302, DOI 10.1109/CSR61664.2024.10679494. CyberSOCEval → arXiv:2509.20166 with the full author list. Minerva, Levi et al. 2025, Qwen2.5, Tülu 3 verified on arXiv. Added GPT-3/OLMo (13-gram precedent), Gururangan 2020, LlamaFactory, ZeRO, Efron & Tibshirani, CVSS v3.1. | `glaukopis_cs_v1.5.bib` |

New finding not in v1.4: the CyberSOCEval composite used by every workbook row is `AVERAGE(W:Y)` = (malware strict accuracy + malware Jaccard + threat-intel strict accuracy)/3, which omits the threat-intel Jaccard score. The paper states the convention explicitly; a symmetric four-metric mean changes B_bal by at most 2.7 points, changes the base-to-final gain from 7.31 to 7.16, and does not change the ordering of any reported row.

## 2. Status of the Section-5 items from the 2026-09-05 memo

| Item | Status in v1.5 | What remains |
|---|---|---|
| Reconcile existing artifacts (workbook formulas, task manifests, prediction exports, evaluation dates, API-counter boundaries, local runtime summaries, training timings, decontamination reports) | **Done** for everything that exists locally (see Section 1). | API run records for the hosted models and trainer logs live on the remote machine, if at all; the base model's per-item files are not local. |
| Independent evaluation on later or independently constructed incidents | **Not possible without new inference.** Replaced by the strongest analysis available offline: entity and gold-fact overlap between the training shards and each AthenaBench task, and the final checkpoint's score on overlapping vs non-overlapping items (Table 6 of the paper). | Build an entity-disjoint / temporally held-out incident set, freeze the graph, evaluate once. |
| Core→CSE and matched single-stage controls | **Not possible without GPU training.** The paper claims only observable transitions and states the missing controls. | Train the two controls (≈ 25–45 h each on 8×H100 by the documented budgets). |
| Quantify uncertainty | **Item-level part done:** 2,000-replicate paired bootstrap over the five checkpoints for every task and aggregate (Table 7, Table 8, Figure 2). **Seed part not possible.** | ≥3 independently seeded Stage-4 runs (≈ 3–4 h each) and, ideally, seeded replication of the whole chain. |
| Canonical-TAA error classification | **Done** (Table 9): the collapse is semantic — at the CSE checkpoint 124/133 responses name a wrong but real group with well-formed output; at +TAA 64/133 name a non-canonical alias of the correct group. | — |
| VSP/ATE error taxonomy | **Done** for VSP (Table 10: exploitability metrics AV/PR/S drive the regression) and ATE (accuracy tracks the training frequency of the gold technique; the scorer only checks the parent technique). | — |
| Parse-failure accounting for CyberSOCEval | **Done:** malware parse failures 0.7% (Core) → 17.1% (CSE) → 5.9% (ship); penalized accuracies reported. | — |
| Qualitative examples | **Verified** against the artifacts (ATE id 38, RCM id 10, CKT id 15) and reported with item ids. | — |
| Journal template | **Done:** elsarticle, author-year, highlights, CRediT, competing interests, funding, data availability, Elsevier generative-AI declaration, line numbers. | Author confirmations listed in `SUBMISSION_CHECKLIST.md`. |

## 3. Reviewer concerns (ICTAI submission 236) and how v1.5 answers them

- **R1/R2/R4 — staging vs. more task-aligned data (no matched control).** Claim narrowed to observable transitions with paired intervals; controls listed as required. Unchanged from v1.4 in substance, now with intervals.
- **R1/R2/R4 — benchmark alignment / no untouched benchmark.** New quantified evidence: n-gram screen counts and coverage limits; entity and gold-fact overlap per task; overlapping-vs-non-overlapping item scores showing no inflation on RCM/RMS/VSP and exposure effects on ATE and attribution.
- **R2/R4 — no confidence intervals or repeated runs.** Paired item bootstrap added; seed replication explicitly still missing.
- **R1/R2/R4 — cost accounting.** Ledger double count found and corrected; API accumulation risk stated; training budget and break-even given as planning figures with the formulas as the claim.
- **R1 — Classic/Canonical divergence explanation.** Resolved by the response taxonomy (semantic, not format).
- **R1 — Foundation-Sec negative finding.** Now a negative result on the consistent aggregate (B_bal −2.9) rather than a +0.4 footnote.
- **R3 — problem formulation, references.** Formulation retained from v1.4; bibliography rebuilt with verified published records.
- **R4 — anonymity.** Moot for a single-anonymized journal; disclosures are explicit.

## 4. Repository observations for the authors (not changed by this revision)

1. `SFT/eval/utils/build_cost_summary.py`: exclude `summary_model.json` from the `summary_*.json` glob (or glob `summary_*_all_v*.json`) and regenerate `cost_summary.{csv,tsv}`; every local row will halve.
2. `SFT/eval/Glaukopis Results.xlsx`, sheet `Final wMMLUPro-v21`: replace the hard-coded V12–V16 and the P-based V17 with `=AVERAGE(T,M,L,I,H,G)`; consider `AVERAGE(W:Z)` or `AVERAGE(W,Y)` for the CSE composite and re-derive every dependent column.
3. `tmpl_gen/templates/05182026/README-21.md`: the contamination section claims CyberSOCEval and CTIBench coverage that the executed build did not have; run `fetch_cybersoceval_data.py` before the dedup step and extend the glob to TSV, then re-run the screen and re-publish the counts.
4. The "~6.5 h on 8×H100" phrase in the top-level README conflicts with the 51–64 h budget in `SFT_FLOW.md`; reconcile.
5. The RCM/VSP AthenaBench items overlap the Core shard on 660 and 446 CVE identifiers; a future release should hold out those CVEs from the corpus or report scores on the disjoint subset by default.
