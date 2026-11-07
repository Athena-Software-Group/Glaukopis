# AAIML 2027, paper L3544 — response to the review

Decision: weak accept (score 1). Ratings: Originality and Relevance *good*; Technical Soundness, Clarity, and Significance *fair*.
Camera-ready source: `aaiml2027_camera_ready.tex` / `.bib` (10 pages, IEEEtran V1.8b conference class, byte-identical to the organizers' `review/conference-latex-template.zip`).
Reviewed version: `aaiml2027_blind_submission (11).tex` (unchanged).

| # | Reviewer request | Status | Where |
|---|---|---|---|
| 1 | Single-stage mixture baseline with matched data and tokens; control that omits the attribution layer | **Partly**: Core used as the single-stage baseline with paired intervals; matched mixture and skip-attribution runs not yet done | §VII-C "Layered versus Single-Stage Training"; §VIII-B "Causal attribution" |
| 2 | Narrow the leakage claim; CyberSOCEval and general data were never screened | **Done**: post-hoc screen run, wording narrowed everywhere | §VI; contributions; §VIII-B; §IX |
| 3 | Core-A/B arithmetic; "240K each" vs "240K total"; actual examples and tokens per stage | **Done** | §III-D; Fig. 2; Table III (`tab:train`) |
| 4 | VSP "fact present" (445) vs 7 complete-answer items; explain "fact absent"; separate entity from answer exposure; account for exclusions | **Done** | Table VI (`tab:overlap`) and the paragraph after it |
| 5a | Qwen3 base scored with thinking on, fine-tuned checkpoint with thinking off; evaluate under matched serving | **Partly**: MMLU-Pro matched result from existing runs; CTI suite not yet re-run | §VII-F "Transfer to Other Base Models" |
| 5b | Repeat key training runs with multiple seeds | **Not done**: stated as a limitation | §VIII-B "Uncertainty and reproducibility" |
| 6 | Larger Fig. 2 and table text, earlier abbreviations, metric types, explicit SLM definition | **Done** | Abstract; §I; Figs. 1–2; Tables; Table IV (`tab:eval`) |

The open items (1, 5a, 5b) need GPU runs; see `RUN_PLAN_JARVIS.md`. `%% RUN-SLOT` comments in the .tex mark the sentences to replace when results arrive.

## 1. Layered versus single-stage training

**Changed.** A new subsection, §VII-C *Layered versus Single-Stage Training*, treats the Core checkpoint as the single-stage baseline. Core is one fine-tuning pass of the base model on the core shards, with no attribution, SOC-format or recalibration layer, scored with the same harness and per-item records as every later checkpoint. Paired differences, final (Recal-32B) minus Core, with item-bootstrap 95% intervals:

- $B_{bal}$ +3.2 [2.2, 4.2] and $A_{bal}$ −2.3 [−3.4, −1.1].
- Gains: CyberSOCEval malware +18.6, threat intelligence +7.9, CKT +2.7.
- Losses: Canonical TAA −15.0, VSP −5.3, RCM −2.7, MMLU-Pro −1.8.

The text calls layering "a trade rather than a uniform improvement". It also states that this comparison does not separate the value of the added attribution and SOC data from the value of training on them in separate layers. Finally, it states that a matched single pass over all four shards, and a chain that omits the attribution layer, "were not run for this paper". The *Causal attribution* limitation says the same.

**Still open.** Two runs would close this point:
- **G1:** the matched single-pass mixture. Its launcher is ready, and its token budget is verified: it trains 349.4M tokens, 99.4% of the chain's 351.6M. The 2.2M difference is the Layer-4 replay, whose rows are already in the union.
- **G2:** the Core → CSE (→ Recal) chain without the TAA layer.

## 2. Leakage claims

**Changed.** The build-time gaps were closed after the fact, using the build-time rule (13-gram windows, flagged at ≥1 shared, removal threshold ≥50); script `analysis/posthoc_screen.py`, output `data/posthoc_screen.json`. Results, stated in §VI:

- **Against CyberSOCEval** (1,197 questions plus their answer options), no Core or TAA row shares any 13-gram. 86 of the 17,320 CSE rows do, none sharing more than 4.
- **Against AthenaBench, CyberMetric and CyberSOCEval together**, 22 of the 240,000 Tülu-3 rows used (none more than 12) and 1 of the 999 Alpaca rows share a 13-gram.
- **No row reaches the removal threshold.**

The screen is described as verbatim-only: it cannot detect paraphrase.

Every "not explained by leakage" claim has been replaced with the reviewer's narrower wording. The paper now says the audit "does not establish leakage as the explanation for the gains, but it cannot exclude it". It also notes that paraphrased and repeated public facts remain uncontrolled. This wording is used in the overlap section, the *Benchmark alignment* limitation, the contributions list and the conclusion. None of the old wording remains.

## 3. Counts and tokens

**Changed.**
- **The 218,440 vs. 218,441 discrepancy.** After the screen removed 684 rows and 550 were held out for validation, one training row was found identical to a validation row and dropped. 219,675 − 684 − 550 − 1 = 218,440 = 168,859 (Core-A) + 49,581 (Core-B). §III-D now says so.
- **"Up to 240K each" vs. "240K total".** LlamaFactory's `max_samples` cap applies to each dataset separately (`select(range(N))`). It truncates Tülu-3 to its first 240,000 rows, 239,985 of which are usable conversations, and leaves Alpaca at its full 999 rows. Fig. 2 and the text now give these actual numbers.
- **Actual examples and tokens per stage.** Table III now has Examples, Tokens (M) and Response tokens (M) columns, plus a Stage-1A row that splits Core-A, Tülu-3 and Alpaca. Tokens were counted by replaying LlamaFactory's encoding of every row with the Qwen2.5 tokenizer, including its per-stage truncation rule (`analysis/stage_tokens.py` → `data/stage_tokens.json`). Layer-4 figures are expectations, because that stage samples rows at random. The text adds that general data supply 87% of Stage 1A's 314.9M tokens.

| Pass | Examples | Tokens (M) | Response tokens (M) |
|---|---:|---:|---:|
| 1A Core-A + Tülu-3 + Alpaca | 409,843 | 314.9 | 175.7 |
| 1B Core-B | 49,581 | 22.2 | 7.4 |
| 2 TAA | 20,359 | 5.0 | 1.3 |
| 3 CSE | 17,320 | 7.3 | 0.3 |
| 4 Recal (expected) | ≈6,000 | ≈2.2 | ≈0.7 |

## 4. Entity versus complete-answer exposure

**Changed.** Table VI was rebuilt with three exclusive groups (script `analysis/overlap_three_way.py`, output `data/overlap_three_way.json`), and its caption accounts for every item:
- **No exposure:** the item's entity never appears in training.
- **Entity only:** the entity appears, but never in the same row as the complete gold answer.
- **Complete answer:** the entity and the complete gold answer appear in one training row.

For VSP the groups hold 1,554, 439 and 7 items (total 2,000). The two items whose final-checkpoint vectors cannot be parsed are excluded from scoring, one in each of the first two groups.

This explains the reviewer's numbers. The old "fact present" group (445 scored items) was entity exposure: 439 + 7 = 446 items, minus the one excluded. Only 7 of those items had the complete answer in training. The old "fact absent" group was the 1,554 unexposed items, 1,553 of them scored.

The text now discusses each task separately:
- **VSP.** The 7 complete-answer items score 89.2 against 77.3 and 77.9 for the other groups. The paper notes that seven items cannot support a conclusion either way.
- **RCM.** Complete-answer exposure does not raise the score: 66.4, versus 64.3 unexposed and 74.2 entity only.
- **RMS.** Exposed sets score lower, because they are larger and set F1 is harder on them.
- **ATE.** Accuracy rises with training frequency across tertiles (62.3 → 64.7 → 70.5).
- **Canonical TAA.** At the +TAA checkpoint, items whose alias mapping appears in training score 52.6, versus 30.6 for the rest.

## 5. Qwen3 serving and seeds

**5a. Changed, partly.** While checking this point I confirmed how each Qwen3 number was served:
- **Paper's original statement:** accurate, as far as the code history shows. When the CSE checkpoint was scored, its wrapper used the `-no-think` alias. Its cache directory carries the HF repo name because the run predates the 23 May 2026 switch to alias-keyed caches. This is inferred from git history; the per-item response files, which are not on this machine, would confirm it.
- **The base:** scored with thinking on.
- **Existing matched run:** the same CSE checkpoint was also scored on MMLU-Pro with thinking on, through the alias built for this A/B test (`athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm`). It scored 53.00, against 52.98 without thinking (`SFT/eval/responses/mmlu_pro_summary.csv`). The base scores 77.58 under the same serving.

The paper now says: "Its base was scored with thinking mode and its fine-tuned checkpoint without; served with thinking, as its base was, the checkpoint still scores 53.0 on MMLU-Pro, so that loss is not a serving artifact, but the CTI comparison was not repeated under matched serving."

**Still open: G3.** Score the CSE checkpoint on the CTI suite under thinking-on serving, using the existing wrapper. This is about 1 h on 2×H100.

Scoring the base with thinking off is the weaker option, for two reasons:
- **The base is thinking-only.** A comment in `models.py` notes that with thinking switched off at request time it still tends to emit a trace and truncate. Its scores would therefore understate the base.
- **The fine-tuned model was trained for this.** It learned to emit an empty trace, so thinking-on serving costs it nothing.

**5b. Not done.** The *Uncertainty and reproducibility* limitation states that the intervals cover item sampling, not training-seed variance, and that every checkpoint is one training run. The Stage-4 launcher now takes `--seed N` (G4). The released run used LlamaFactory's default seed, 42.

## 6. Presentation and terminology

**Changed.**
- **Figures.** Fig. 2 text enlarged (`\footnotesize`/`\scriptsize`) with resized boxes; Fig. 1 redrawn as a compact one-row pipeline.
- **Tables.** Dense numeric tables raised to `\footnotesize`.
- **Abbreviations.** CVE, SOC, CTI, NVD, EPSS, CWE, CAPEC and LLM are spelled out at first use. The six task abbreviations are defined in §I, and TAA and CSE in the Fig. 1 caption. "Benchmark A" is now AthenaBench throughout.
- **Metric types.** Table IV has a "Score (type)" column (accuracy, F1, or partial credit). The text states that $T_{bal}$, $A_{bal}$ and $B_{bal}$ are averages of task scores, not accuracies.
- **SLM definition.** Stated in the abstract and §I: open-weight models of 8B to 32B parameters that a two-GPU host can fine-tune and serve.

## Other camera-ready changes

- **De-anonymized:**
  - author block with FAU and Athena affiliations, plus Cardei's ORCID;
  - AthenaBench citation (ACSAC Workshops 2025);
  - artifact repository URL;
  - Levi et al. updated to ICML 2026;
  - Dong et al. (ACL 2024) added.
- **Acknowledgment:** funding (Athena Software Group Inc.), competing interests (Worth's shares; Cardei's advisory role and incentive options; Worth's AthenaBench co-authorship), and AI-assistance disclosure.
- **Ariadne graph counts** from the v21 release: 214,651 nodes with 31 labels; 330,935 relationships of 41 types.
- **Cut for space:** the Qualitative Examples subsection, the roadmap paragraph, and most of the future work (condensed to one paragraph).

## Camera-ready checklist (due 30 Nov 2026)

- [ ] Decide which runs from `RUN_PLAN_JARVIS.md` to fund. Fill the `%% RUN-SLOT` sentences, making equal cuts, because page 10 is full.
- [ ] Final read by both authors. Cardei's e-mail address is not in the author block; add it if he wants one.
- [ ] IEEE copyright line: insert the exact string the organizers supply, if any. A commented placeholder with instructions is in the .tex header.
- [ ] Page charge: 10 pages = 6 included + 4 extra at $70 each ($280), assuming the extra-page fee is unchanged on the registration form.
- [ ] Complete `review/registration_form_full paper.docx`.
- [ ] E-mail the final PDF, the LaTeX source, and the registration form to aaiml_conf@163.com.
- [ ] In that e-mail, ask the organizers to update the title. The acceptance letter shows an older one ("Layered Instruction Fine Tuning Cyber Threat Intelligence for Small Language Models: A Functional Approach"); the paper and review form use "Graph-Grounded Layered Instruction Fine-Tuning of Small Language Models for Cyber Threat Intelligence".
- [ ] Confirm which author will present in person in Tokyo (29–31 March 2027). Attendance is mandatory for publication.
