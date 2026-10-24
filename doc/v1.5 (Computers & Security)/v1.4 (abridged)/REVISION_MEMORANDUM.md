# Glaukopis v1.4 — revision memorandum and reviewer-response map

**Prepared:** September 5, 2026  
**Requested output:** `glaukopis_abridged_v1.4.tex`  
**Input version actually supplied:** `glaukopis_ieee_abridged-v1.3(1).tex`, with its v1.3 PDF; long-form v1.1 TEX/PDF; ICTAI submission 236 reviews.  
**Status:** Substantive author-review journal revision. Not an assertion that missing controls, repeated seeds, independent holdouts, or matched cost measurements have been completed.

## 1. Editorial and scientific decision

The revised title is **Glaukopis: Auditable Graph-Grounded Instruction Tuning for Cyber Threat Intelligence**. The central contribution is the Ariadne/Sophia query-to-target provenance mechanism. The multi-stage schedule is retained, but described as an instrumented configuration that exposes task interference, not as a causally validated improvement over single-stage SFT.

The revision remains nine pages in a journal-neutral IEEE layout. It is not yet converted to the official IEEE Access class. The missing external figure dependency has been removed; the pipeline boundaries and training configuration are given in native LaTeX tables. This also avoids retaining the original schematic's hyperparameter/packing inconsistencies. All four original authors are preserved, with their supplied affiliations rather than the unused RIT affiliation in the longer manuscript.

## 2. Evidence actually inspected

The two supplied manuscripts and all four reviews were read. The public repository was inspected through the connected GitHub service, read-only, at commit:

`c4d3a2f45e0a08672c87c2b46c3c4be109440c38`

Relevant files included `PROJECT_STATE.md`, `README.md`, the IEEE bibliography, `SFT/eval/responses/cost_summary.csv`, `SFT/eval/utils/_print_sweep_summary.py`, `SFT/eval/utils/build_cost_summary.py`, and `tmpl_gen/scripts/dedup_against_evals.py`.

The Mac path `/Users/pietro/code/Glaukopis` is not mounted in this session. No files on that machine or in GitHub were modified. Raw prediction artifacts described by the repository as local/untracked were not available for a new item-level analysis. The repository's results workbook was identified, but its binary contents could not be materialized here. Its exact formulas and unrounded values have therefore not been claimed as audited.

No training, hosted-model inference, new experimental evaluation, bootstrap confidence interval, or significance test was run. The included Python script only recomputes arithmetic from existing reported scores and the tracked cost ledger.

## 3. Reviewer-by-reviewer response map

The responses below describe changes actually made. They are not a claim of acceptance, a representation that the ICTAI reviewers have seen the revision, or a substitute for the receiving journal's requested cover letter.

### Reviewer 1 — weak accept

| Concern | Change in v1.4 | Status and residual evidence |
|---|---|---|
| Four-stage framing suggests additive gains although individual tasks regress. | Title, abstract, Sections I–II, V, VII and X now make data provenance primary. Stage transitions are cumulative observations, not matched ablations. The VSP and Canonical-TAA failures are stated directly. | Addressed in framing; causal superiority is deliberately not claimed. |
| Core-plus-CSE control would test whether Stage 2 is necessary. | Sections V and IX identify Core→CSE and Core→CSE→Recal as the appropriate skip-stage controls. | New training control still required; no result invented. |
| Benchmark alignment needs main-text treatment and an untouched benchmark. | Ownership overlap is disclosed in the introduction. Section VI maps each training shard to its targeted evaluation family. Section VII quantifies that all positive cross-benchmark gain comes from AthenaBench. | Addressed in disclosure and arithmetic; no genuinely independent CTI holdout has been demonstrated. |
| Add training cost and amortized break-even; report CTI-only cost. | Section VIII and Table VI reaggregate the actual repository ledger into CTI-only and MMLU-Pro expenditure. Training-cost, lifecycle-cost, and break-even equations are included with their conditions. | CTI arithmetic completed. A verified full-chain training ledger and matched single-pass inference accounting remain unavailable. |
| Stage-4 recipe selection rests on a small single-run difference. | Section VII treats “ship” as a project release label, not a statistically validated optimum. Repeated training seeds and paired item uncertainty are distinguished. | Editorial claim corrected. Repeated Stage-4 runs still needed. |
| State which operational workloads the model should not handle. | Sections II and IX explicitly exclude sole reliance for autonomous attribution, high-consequence containment, unrestricted tool use, legal interpretation, and unvalidated out-of-domain reasoning. | Addressed as operational boundaries, not as a new safety evaluation. |
| Disclose AthenaBench authorship. | Peter Worth and Salman Ahmad are named as overlapping authors in Section I; developer/evaluator relationships are disclosed again at the end. | Addressed. |
| Explain Classic/Canonical TAA divergence. | Section VII describes competing hypotheses and a concrete response-level classification audit. | Hypothesis distinguished from observation. Prediction-level error counts are not available. |
| Promote Foundation-Sec negative finding. | Section VII-C/D gives the already-specialized base a dedicated paragraph and retains its −17.6 MMLU-Pro result. The reported +0.4 legacy CTI statistic is explicitly qualified because its aggregate formula is unreconciled. | Addressed without relabeling an unknown weighted metric. |

**Additional correction affecting this review:** the review accurately reads the original table, but that table's AthenaBench column is numerically consistent with Classic TAA alone, not its claimed 50/50 blend. Under the explicit balanced definition in v1.4, AthenaBench peaks at +TAA rather than Core. The underlying task scores have not changed.

### Reviewer 2 — weak reject

| Concern | Change in v1.4 | Status and residual evidence |
|---|---|---|
| Data, ordering and hyperparameters co-vary; no matched single-stage comparison. | Section II formally separates the data-construction mechanism from the standard SFT objective. Sections V and IX explicitly limit causal inference. | Claim narrowed; matched single-stage experiment remains needed for a scheduling-superiority claim. |
| Contamination/alignment and catastrophic interference. | Section VI puts the exact n-gram thresholds, schema/fact overlap, auxiliary-mixture exclusions, and file-coverage issues in the main text. Section VII reports the TAA collapse and the non-improving external benchmark summaries. | Reporting improved; no assertion of contamination-free evaluation. |
| No confidence intervals or repeated runs. | Single-run status is explicit. No significance is assigned to the Stage-4 margin. | New seed runs and item-level analysis remain required. |
| Cost mixes local GPU accounting and API price, omitting training. | Section VIII labels both accounting regimes, identifies cumulative ledger semantics, removes headline 45–408× savings, and gives conditional amortization equations. | Misleading interpretation removed; empirical TCO/break-even not fabricated. |

### Reviewer 3 — reject

| Concern | Change in v1.4 | Status and residual evidence |
|---|---|---|
| No clear problem formulation or technical benefit. | New Section II defines graph snapshot, template query, binding, rendered training target, provenance, and standard training objective. Section I explains the specific traceability and diagnosis problems. | Addressed with a technical formulation rather than a generic “cybersecurity LLM” pitch. |
| Reference [11] appears to describe duplicated concurrent work. | Software is explicitly cited as source-available implementation, not a second peer-reviewed paper. End matter distinguishes the long and abridged manuscript versions from the separate AthenaBench benchmark. | The attached identity-bearing v1.3 has a software reference at [11]. The exact anonymized submitted bibliography was not provided, so the reviewer's specific reference cannot be conclusively reconstructed. Authors must confirm the publication/submission status of the longer manuscript. |
| Most references are arXiv and writing is not academic enough. | Bibliography rebuilt around verified published versions for FLAN, InstructGPT, FLAN-T5, CyberPal.AI, CTIBench, MMLU-Pro, vLLM, Open-CyKG, and the CyberBench workshop. Incorrect identifiers and unverified KG citations removed. Technical claims are narrower and terminology normalized. | Verified preprints remain labeled as preprints; software and standards are not counted as refereed papers. See bibliography audit below. |

### Reviewer 4 — reject

| Concern | Change in v1.4 | Status and residual evidence |
|---|---|---|
| Distinctive project and public repository defeat conference double-blind review. | This is an author-identified journal working draft. Authorship, funding, benchmark overlap, and software provenance are explicit. | Appropriate for an author-identified route such as IEEE Access. A journal that requires double-anonymous review still needs its own anonymization/artifact procedure. Journal submission does not automatically remove every anonymity obligation. |
| No matched single-stage baseline and no repeated-seed evidence. | Superiority claims removed; controls and uncertainty requirements stated. | Not experimentally resolved. |
| Shared facts/schemas and no independent incidents. | Alignment moved forward and quantified; controls described precisely. | Independent holdout still needed. |
| Cost excludes graph construction, fine-tuning and maintenance. | Conditional lifecycle equations and cost-boundary discussion added; unvalidated full-suite savings no longer headline the paper. | Complete measured lifecycle cost not available. |

## 4. Numerical audit: what changed and why

### 4.1 AthenaBench aggregation is mislabeled in the supplied tables

Using the supplied task scores:

- Base, **Classic-only**: `(52.9 + 18.2 + 59.5 + 7.6 + 78.8 + 48.0) / 6 = 44.1667`, which rounds to the original 44.2.
- Final, **Classic-only**: `(75.5 + 65.8 + 67.1 + 62.5 + 77.8 + 50.5) / 6 = 66.5333`, which rounds to the original 66.5.

Those values do not use the stated 50/50 attribution blend. Applying that blend instead:

| Checkpoint | Recomputed balanced AthenaBench | Recomputed three-benchmark mean |
|---|---:|---:|
| Base | 40.3333 | 57.7111 |
| Core | 64.8750 | 61.8583 |
| +TAA | 66.0167 | 61.7389 |
| CSE | 62.4417 | 64.4139 |
| Recal-32B | 62.6083 | 65.0028 |

The final-minus-base gain remains about 7.3 points. This is **post-hoc arithmetic on existing rounded summary scores**, not a new benchmark run. The repository README's “Total 65.0” is consistent with this result, but the results workbook was not examined and this observation is not proof of all its formulas.

The final production tables should be generated from unrounded task/item results and a versioned scoring definition. Do not compare the new balanced local mean to a hosted model's differently defined legacy aggregate.

### 4.2 Legacy secondary averages remain inconsistent

The original statement that Avg Total extends the ten-test Avg CTI Test with MMLU-Pro is contradicted by the printed values:

- Base: `(10 × 49.6 + 69.0) / 11 = 51.3636`, not 49.3.
- Final: `(10 × 62.9 + 57.4) / 11 = 62.4`, not 62.8.

The v1.4 main conclusions no longer use these columns. Their original values are preserved in `data/legacy_headline_table_UNRECONCILED.csv`; they have not been silently overwritten with guesses about the workbook's weighting.

### 4.3 Generalization claim is narrower than the headline mean

Final versus base:

- AthenaBench balanced: +22.2750, contributing +7.4250 to the three-benchmark mean.
- CyberSOCEval: 42.6 → 42.6, contributing zero.
- CyberMetric: 90.2 → 89.8, contributing −0.1333.

Every positive contribution to the mean therefore comes from AthenaBench. This directly addresses the reviewers' alignment concern without inventing an untouched-benchmark experiment.

### 4.4 Other corrected interpretations

The MoE port's MMLU-Pro score of 53.0 is below the dense 32B score of 57.4. Stage 4 does not restore VSP above the parent or base. A hypotheses-only explanation such as “optimizer noise floor” or “MoE routing caused the loss” is not stated as demonstrated causation. All-source nightly refresh and all-training-rows graph materialization were narrowed to match the longer implementation description and the auxiliary general-data mixture.

## 5. Cost-accounting audit

Pinned source files:

- [Recorded cost CSV](https://github.com/Athena-Software-Group/Glaukopis/blob/c4d3a2f45e0a08672c87c2b46c3c4be109440c38/SFT/eval/responses/cost_summary.csv)
- [Per-sweep summary code](https://github.com/Athena-Software-Group/Glaukopis/blob/c4d3a2f45e0a08672c87c2b46c3c4be109440c38/SFT/eval/utils/_print_sweep_summary.py)
- [Cost aggregation code](https://github.com/Athena-Software-Group/Glaukopis/blob/c4d3a2f45e0a08672c87c2b46c3c4be109440c38/SFT/eval/utils/build_cost_summary.py)

The API summary's own documentation says counters can be restored and accumulated across reruns, including overwrite runs. The aggregate utility sums API checkpoint entries by model/family and local elapsed time over available summary files. These semantics do not independently certify one matched sweep per model. Whether a given total is inflated, complete, or correctly scoped requires its actual run records; the code audit alone does not establish the number of repeats.

The revised cost table therefore reports ledger amounts, including a CTI-only subtotal. It does not call them verified per-inference costs. The supporting CSV includes arithmetic ratios but labels them **ledger ratios**. It also preserves the supplied local accounting assumption of two H100s at $2.50/GPU-hour; that is not a current cloud-price recommendation.

The README's approximate 6.5-hour / eight-H100 description lacks a stage-complete training ledger and cannot safely be multiplied into a measured full-chain bill. Even a correct successful-run bill would need separate treatment of graph/data construction, unsuccessful research runs, repeat deployment, and ongoing operations.

## 6. Contamination implementation audit

[Inspected filter](https://github.com/Athena-Software-Group/Glaukopis/blob/c4d3a2f45e0a08672c87c2b46c3c4be109440c38/tmpl_gen/scripts/dedup_against_evals.py)

The distinction between `hit-threshold=1` and `drop-threshold=50` is central. One shared 13-gram is logged, not necessarily removed. A record shorter than 62 words cannot contain 50 distinct 13-grams, so complete short duplicates can evade the removal threshold under unique record identifiers. This is a property of the method, not an assertion that a specific training row is contaminated.

Additional items to resolve before using “decontaminated” as an unqualified guarantee:

- Default file globs cover JSON/JSONL, not the long manuscript's claimed TSV inputs. Demonstrate any conversion path and exact files indexed.
- The extractor handles named fields and `answers`; confirm benchmark-specific gold labels, options, nested structures, and auxiliary corpora are covered where required.
- Evaluation keys use file basename and row index. Replace them with unique full relative paths or immutable IDs and test collision handling.
- General Tülu/Alpaca mixtures were described as outside the routine scan. Audit them separately.
- Text overlap does not establish fact-, entity-, incident-, or time-disjointness. Measure those dimensions and document permitted catalogue overlap.

The revision does not claim to have rerun this scan or to have measured newly removed rows.

## 7. Bibliography audit

| Legacy entry/problem | Action in the revised bibliography |
|---|---|
| CyberPal.AI cited as a 2024 preprint. | Replaced with AAAI 2025, 39(23), 24402–24412, DOI 10.1609/aaai.v39i23.34618. |
| Scaling Instruction-Finetuned Language Models cited only as arXiv. | Replaced with JMLR 25(70), 1–53, 2024; author metadata corrected. |
| CTIBench linked to arXiv:2406.07508. | That identifier resolves to an unrelated particle-physics paper. Replaced with the NeurIPS 2024 publisher record; authors corrected to Alam, Bhusal, **Le Nguyen**, and Rastogi. |
| CyberBench linked to arXiv:2407.08665. | That identifier resolves to an unrelated tunnel-junction/randomness paper. Replaced with the authors' AICS/AAAI workshop record. |
| Open-CyKG attributed to Wu/Wang, 2022. | Replaced with **Injy Sarhan and Marco Spruit**, Knowledge-Based Systems 233, 107524, 2021, DOI 10.1016/j.knosys.2021.107524. Its graph-construction contribution is not misrepresented as a graph-IFT experiment. |
| InstructGPT, FLAN, MMLU-Pro and vLLM bibliographic status. | Used the published conference records rather than presenting all as unpublished arXiv works. |
| Several generic KG entries with incomplete/unverified identifiers (Fieblinger/Tjoa “Actionable…”, CTIKG, K-CTIAA, CyberKG) and the incomplete SecEval reference. | Removed from the revised cited subset instead of inventing publication details. Reintroduce only with verified primary records and supported claims. |
| AthenaBench. | The public arXiv record was verified and retained. The previously mentioned IEEE workshop DOI could not be confirmed from an accessible primary publication record during this audit; provide the publisher citation export before changing the bibliography to that version. This is not a conclusion that no proceedings version exists. |
| SEvenLLM, CyberMetric, CyberPal follow-on and Minerva. | Retained as explicitly labeled preprints where a later peer-reviewed version was not verified in this session. Minerva title and authors corrected. |
| Project reference described as an “open-source implementation.” | Reclassified as a source-available software artifact with pinned commit and the repository's noncommercial license, not a peer-reviewed paper. |

Verified primary records used include [AAAI CyberPal](https://ojs.aaai.org/index.php/AAAI/article/view/34618), [JMLR FLAN-T5](https://jmlr.org/papers/v25/23-0870.html), [NeurIPS CTIBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5acd3c628aa1819fbf07c39ef73e7285-Abstract-Datasets_and_Benchmarks_Track.html), [CyberBench author record](https://zefang-liu.github.io/publication/liu2024cyberbench), [NeurIPS MMLU-Pro](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ad236edc564f3e3156e1b2feafb99a24-Abstract.html), and the individual preprint records linked in the `.bib`.

## 8. Pre-submission evidence priorities

### Gate A — resolve from existing local artifacts before spending GPU time

Recover the results workbook with formulas and unrounded task outputs; adopt one explicit attribution/benchmark aggregation rule; reconcile all hosted reference rows under that same rule. Confirm the real evaluation timestamps and immutable task manifests. Separate matched per-run API deltas from cumulative counters and duplicate local summaries. Recover per-pass training runtimes and exact completed-token/update counts. Re-audit decontamination coverage and publish the actual post-filter counts and manifests. Confirm related-manuscript publication/submission status and the final reference list.

### Gate B — inexpensive analyses if prediction logs are present

Classify Canonical-TAA errors from the paired +TAA/CSE/final responses. Report parser failures separately from semantic misclassification. Bootstrap paired item scores using the appropriate independent unit; this estimates evaluation-item uncertainty only. Check duplicates and source-entity overlaps using fixed, documented thresholds. These analyses can improve evidence without retraining, but none has been represented as completed in v1.4.

### Gate C — minimum targeted new experiments for a stronger journal submission

1. A Core→CSE control, preferably with the same final recalibration fork, to test whether TAA belongs in the retained chain.
2. A matched one-stage mixed-data control with the same actual examples/tokens, optimizer update budget, and base model; distinguish equal-token from equal-wall-clock comparisons where both cannot be achieved simultaneously.
3. Several independent Stage-4 training seeds (the reviewer proposes three) evaluated under a frozen selection rule. Report all runs rather than the best seed.
4. An external incident or temporally held-out CTI set with no shard designed against it, accompanied by entity/fact/source overlap reporting. A new benchmark name alone is not evidence of independence.

A two-factor data-source × schedule study would be stronger still, but is a larger scope than the minimal journal repair. Do not add such results to the paper until actually run.

## 9. Journal submission mechanics

The document is not presently a submission-ready IEEE Access template. IEEE Access requires its own template, author biographies for all authors, source/PDF agreement, an ORCID-linked submitting account, and AI-text disclosure. The included disclosure describes assistance honestly without claiming that the authors have already validated every statement. Final author review and any journal-specific disclosure adjustments are required.

The new bibliography references real published versions where verified, but a journal cannot be made suitable merely by increasing the proportion of refereed citations. Its contribution, evidence, and reproducibility still matter. The most urgent issues are formula/run reconciliation and an appropriately bounded empirical claim, not decorative citation replacement.
