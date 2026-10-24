# Glaukopis v1.6 — response to the Computers & Security reviews and change record

**Prepared:** 2026-10-24
**Source reviewed:** `doc/v1.5 (Computers & Security)/glaukopis_cs_v1.5.tex` (submission COSE-D-26-05507, rejected; Reviewer 1 dated 17 Oct 2026, Reviewer 2 dated 22 Oct 2026)
**Result:** `doc/v1.6/glaukopis_v1.6.tex` + `glaukopis_v1.6.bib`, built for resubmission to a different journal (see §4).
**What was not done:** no model training and no model inference. Every new number is computed from artifacts that already existed in the repository (training shards, per-item VSP prediction files, the results workbook, the cost ledger) by `analysis/v16_additions.py`, which writes `data/v16_additions.json`. Nothing in the paper depends on a number that script cannot reproduce.

The section numbers the reviewers cite are those of v1.5 (§4.1 Ariadne, §7.2 transitions, §7.5 examples, §8 cost). v1.6 inserts a new §2 (overview) and several subsections, so the numbering has shifted; the map in §3 below gives both.

---

## 1. Reviewer 1 — exposition, background, baselines, interpretation

| # | Comment (paraphrased) | What v1.6 does | Where |
|---|---|---|---|
| 1.1 | Abstract is too technical; should start from the field and context, then the approach, then what is measured and how. | Abstract rewritten in that order: what CTI work is → why a locally served specialized model is needed → what Glaukopis does (graph-traced data, retained checkpoints) → what is measured and against what → results → the limit of the claim. 245 words. | Abstract |
| 1.2 | Paper assumes the reader has read a previous paper; explanations anchored in components not yet introduced; "written as a to-do list"; choices not justified. | (a) Introduction rewritten to define CTI, the catalogs, and the problem before any system name appears; Ariadne, Sophia and Glaukopis are each defined in one sentence at first use, in the same paragraph. (b) New §2 "System overview and terminology" with a glossary table (Table 2) defining every recurring name, shard, stage, benchmark and task acronym. (c) The paragraph describing how this version differs from the conference draft, and the workbook-audit narrative in the aggregation section, are removed from the body; the one fact a reader needs (why 65.0/57.7 rather than 66.3/59.0) is kept as a two-sentence note. (d) Design-choice rationale added: why a knowledge graph and why Neo4j (§5.1), why these twelve sources and the 2024+ cutoff (§5.1), why Qwen2.5-32B (§6.1), why four stages in this order and what each stage is for (§6.1), why the 13-gram screen (§7.4, unchanged), and what each baseline controls for (§7.2). | §1, §2, §5.1, §6.1, §7.2 |
| 1.3 | "Sophia instead obtains the CTI target from an executed query over the Ariadne knowledge graph" appears before either is defined. | The sentence is gone; both are defined before they are used (§1 ¶3), and §2 repeats the definitions in a table. | §1, Table 2 |
| 1.4 | The sentence about Worth and Ahmad co-authoring AthenaBench needs logical steps between authorship and evaluation. | Rewritten as a full paragraph (§1 ¶4) and a dedicated subsection "Evaluation independence" (§7.3): the benchmark authors also designed the training templates, so the two share knowledge of the task schemas even though they were built by different pipelines from different inputs; the overlap is quantified (§7.5) but the shared design cannot be removed by measurement; therefore AthenaBench is read as the aligned benchmark and CyberSOCEval/CyberMetric/MMLU-Pro as the independent evidence, and the results are presented in those three tiers. | §1, §7.3 |
| 1.5 | Need a set of baseline methods, preferably the best in the field, compared experimentally; Table 1 is only theoretical. | New §7.2 "Baselines and reference models" states the five experimental comparisons already in the paper and what each controls: the zero-shot base; Foundation-Sec-8B-Instruct (security-specialized open model); six hosted frontier models under the same harness (GPT-5.5, GPT-5.2, DeepSeek-V4-Pro, DeepSeek-V3.2-Exp, Gemini-3-Flash, Gemini-2.5-Flash); a single-stage fine-tuning run on the same base (new, §6.3); four architecture ports. Foundation-Sec is now discussed explicitly as a baseline in §8.7 (56.4 vs 65.0 on B_bal; strong on knowledge, weak on mapping). Why the systems in Table 1 are not compared by score is stated: none releases weights evaluated on this suite, so a score comparison would require retraining them, which was outside budget. | §7.2, §6.3, §8.7 |
| 1.6 | §4.1: be precise about what each of the 12 sources is (e.g. Sigma is a rule format, not a dataset). | New Table 3 lists each source, exactly what is loaded (e.g. Sigma: the SigmaHQ rule repository, each rule with its ATT&CK tags; ExploitDB: the exploit index rows referencing a loaded CVE; NVD: CVSS vectors and CPE folded onto CVE nodes), the retention rule (full vs 2024+), and the node labels it creates. Read from `athena_cti_db/FUNCTIONAL_SCOPE.md` and the populator's source definitions. | Table 3 |
| 1.7 | §4.1: report exact graph sizes, not "approximately". | The schema is now reported exactly (26 node labels, 37 relationship types, from `tmpl_gen/_db_schema.json`). The node/edge counts remain approximate in the text because no build log with exact counts exists in the repository and the graph is rebuilt per snapshot; the text explains that the count is a property of a build and names the query that produces it. **Author action:** if the v21 Neo4j instance is still up, run the two Cypher count queries in the TODO comment at §5.1 and replace the sentence with a per-label table. | §5.1 |
| 1.8 | §7.2: results are presented but not interpreted. | Interpretation added after every result: why CyberSOCEval is flat (format lost at Stage 1, regained at Stage 3 — the components move in opposite directions) and why CyberMetric does not move (§8.1); what the Core transition actually teaches (format + catalog recall) and how the attribution stage works (the taxonomy shows wrong-group → alias → exact) (§8.2); why Stage 4 recovers broad but not narrow capabilities (≈900 TAA rows and 3,600 Core-B rows at 3e-6 against 17,320 rows at 5e-6) (§8.2); a mechanistic account of the attribution collapse from the CSE shard's content (new Table 12, §8.3); the direction of the VSP errors (§8.4); which MMLU-Pro subjects are lost and why (new §8.6); what the hosted comparison means (§8.8). | §8 throughout |
| 1.9 | §7.5: examples are listed, not explained. | Each of the three examples is now explained as an instance of a distinct error type (granularity; sibling confusion in the CWE hierarchy; capability vs consequence in ATT&CK) with what the discrimination requires and which training family drills it; the selection method is stated (chosen as clear instances, not sampled). | §8.5 |
| 1.10 | §8 (costs): dollar figures go stale; questionable in a scientific paper. | Section renamed "Inference efficiency" (§9). Primary measures are now physical and durable: wall-clock seconds on fixed hardware for local models, input/output tokens for hosted models (new columns in Table 17, from the ledger). Dollar conversions appear once, dated (May 2026), so the reader can substitute current rates. The break-even derivation is reduced to one paragraph with the formula as the claim; the section is about half its former length. | §9 |

## 2. Reviewer 2 — methodology

| # | Comment (paraphrased) | What v1.6 does | Where |
|---|---|---|---|
| 2.1 | No matched single-stage control; cannot tell whether 4 stages help. | Two changes. (a) The paper no longer presents the 4-stage schedule as its central empirical claim: the abstract, introduction, contributions and conclusion now say explicitly that no advantage is claimed for the schedule, and that the contribution is the traceable data method plus the diagnostic protocol. (b) New §6.3 reports the closest single-stage run in the project record: Qwen2.5-32B-Instruct trained in one stage on the earlier v7 Sophia corpus (180,533 rows, 3 epochs, lr 1e-5) — a comparable update budget — evaluated with the same harness (Table 7). It reaches A_Cl = 65.8 vs 66.5 for the staged chain and B_Cl = 64.7 vs 66.3; the chain is ahead on CKT/ATE/RCM, the single pass on RMS/VSP/Classic-TAA. The text states precisely why it is not a matched control (different corpus vintage) and specifies the matched control that would settle the question (one pass over the union of the v21 shards, same realized tokens and updates; 25–45 h on 8×H100). **This is the one item that can only be fully closed by training; see §5 below.** | §1, §6.2–6.3, Table 7, §10, §11 |
| 2.2 | Hyperparameters change concurrently at stage boundaries, so transitions cannot isolate causal effects. | Agreed and now stated as a named subsection (§6.2 "What changes between stages") listing the four co-varying quantities and the reason they co-vary; every "stage effect" sentence in §8 is phrased as the effect of the whole configuration change. Table 5 unchanged. | §6.2 |
| 2.3 | Canonical TAA collapses 36.8 → 3.4; 124/133 wrong-but-valid group names; G-codes cease. | This was reported in v1.5 as a negative finding; v1.6 explains it. New Table 12 profiles the two narrow shards over every row: the 20,359 TAA rows end in free-text attribution (66% with a G-code, 52% naming a benchmark group); the 17,320 CSE rows have a median output of 3 words and contain **zero** G-codes, group names or T-codes — every output is a JSON letter set — while 51% of CSE *inputs* name a benchmark group. Stage 3 therefore presents group names 8,801 times as context and never as a target, immediately after Stage 2 trained the opposite role, which is exactly the "wrong but well-formed group, no G-code" signature. The remedy (interleave the format drill with the attribution shard) is stated as a hypothesis for a controlled run. Operational consequence stated in §10: do not use the checkpoint for attribution. | §8.3, Table 12, §10 |
| 2.4 | VSP exact-vector match 25.6% → 13.6%; PR/AV/S mismatches explode; distance scoring masks it. | Table 14 gains a *metric-level accuracy* column (share of the 8×2,000 metric values correct: 83.1% → 76.6% → 74.8%), which does not share the distance score's tolerance. New direction analysis from the per-item files: of 583 AV errors at the final checkpoint, 552 are gold Local → predicted Network; of 791 PR errors, 657 are gold Low → None; at +TAA the same confusions were small and balanced. 84% of the 658 locally exploitable test CVEs are misjudged as remote. Interpreted as collapse toward the modal exploitability profile after an epoch of three-word targets (the CSE shard has almost no CVSS content, so it is not conflicting supervision). Operational consequence stated (§10): vectors biased toward over-estimated exposure; recommend reporting metric-level accuracy and error direction with any CVSS distance score. | §8.4, Table 14, §10 |
| 2.5 | MMLU-Pro −11.6 (32B), −24.6 (MoE); gains only on AthenaBench; CyberSOCEval zero net gain. | Stated as the headline limit in the abstract and §1 rather than discovered in the results. New §8.6 locates the MMLU-Pro loss by subject (physics −23.2, chemistry −22.7, business −18.7, math −18.6 vs health −4.0, history −3.2, philosophy −0.4) and attributes it to shortened reasoning (the base's MMLU-Pro run takes 9–18× longer because it reasons at length; the general mixture is present only in Stage 1A; Stage 3 trains 3-word targets), with the untested remedies named. §8.1 explains why CyberSOCEval's composite is flat (components move in opposite directions; format lost then regained) and says plainly that no transfer of CTI reasoning to a benchmark built by others has been shown. | Abstract, §1, §8.1, §8.6, §10 |
| 2.6 | Developer–evaluator overlap (AthenaBench co-authors; Athena funding); no entity-disjoint / out-of-domain / blind testing. | Disclosed in §1 ¶4 and in the new §7.3, with the three-tier reading of the benchmarks. The overlap audit (§7.5, Table 10: entity overlap, gold-fact overlap, and scores on overlapping vs non-overlapping items) is retained and now framed as the partial answer it is; §10 states the decisive test (frozen graph, development-set selection, later incident set built without access to the templates) and that it has not been run. The honest negative results on the independent benchmarks and on the two 8B ports are pointed to as the pattern an inflated evaluation would not produce. Competing-interest statement unchanged. | §1, §7.3, §7.5, §10 |
| 2.7 | No citations for GPT-5.5, GPT-5.2, DeepSeek-V4-Pro, DeepSeek-V3.2-Exp, Gemini-3-Flash-Preview, Gemini-2.5-Flash. | Added and cited at first use (§7.2): OpenAI "Introducing GPT-5.5" (23 Apr 2026) with system card; "Update to GPT-5 System Card: GPT-5.2" (11 Dec 2025); DeepSeek-V4 technical report (Apr 2026, HF release); DeepSeek-V3.2-Exp technical report (Sep 2025, GitHub); Gemini 3 Flash model card (17 Dec 2025); Gemini 2.5 technical report (Comanici et al., arXiv:2507.06261) with the Flash model card. Also added for the ported bases: Foundation-Sec-8B technical report (Kassianik et al., arXiv:2504.21039), Llama 3 herd (arXiv:2407.21783), Qwen3 technical report (arXiv:2505.09388). Each hosted entry records the alias and month under which the model was evaluated. | §7.2, `.bib` |

## 3. Section map, v1.5 → v1.6

| v1.5 | v1.6 | Change |
|---|---|---|
| Abstract | Abstract | rewritten (context first; 245 words) |
| 1 Introduction | 1 Introduction | rewritten; roadmap paragraph added |
| — | **2 System overview and terminology** (Table 2 glossary) | new |
| 2 Problem formulation | 3 Problem formulation | unchanged |
| 3 Background and related work | 4 Background and related work | unchanged |
| 4.1 Ariadne's source and refresh boundaries | 5.1 Why a knowledge graph, and why these sources (Table 3) | rewritten; source table added; schema size exact |
| 4.2–4.3 | 5.2–5.3 | unchanged |
| 5 Multi-stage training | 6.1 Base model and rationale for the stages; 6.2 What changes between stages; **6.3 A single-stage reference run (Table 7)** | rationale added; co-variation named; new reference run |
| 6.1 Task portfolio | 7.1 | audit-narrative sentences removed |
| — | **7.2 Baselines and reference models** | new |
| 6.2 Alignment | 7.3 Evaluation independence | rewritten |
| 6.3–6.6 | 7.4–7.7 | 7.6 (aggregation) shortened: workbook-audit narrative replaced by a two-sentence note |
| 7.1 | 8.1 | interpretation of CyberSOCEval / CyberMetric / per-task pattern added |
| 7.2 | 8.2 | interpretation of each transition and of Stage 4's partial recovery added |
| 7.3 | 8.3 (Table 12 new) | shard-content mechanism added |
| 7.4 | 8.4 (Table 14 extended) | metric-level accuracy and error direction added |
| 7.5 | 8.5 | examples interpreted |
| — | **8.6 Where general reasoning is lost** | new |
| 7.6 | 8.7 | Foundation-Sec discussed as baseline |
| 7.7 | 8.8 | interpretation sentence added |
| 8 Evaluation expenditure and lifecycle economics | 9 Inference efficiency (Table 17 reworked) | reframed on wall-clock and tokens; dollars dated; break-even cut to one paragraph |
| 9 Threats to validity | 10 Threats to validity | reorganized into six named paragraphs matching the reviewers' concerns; controls specified concretely |
| 10 Conclusion | 11 Conclusion | rewritten |

(Table/figure numbers above are those produced by the generic-class preview build; the Elsevier/Springer class may number floats differently but the order is the same.)

## 4. Numbers that are new in v1.6 and where they come from

| Number | Source artifact | Script |
|---|---|---|
| TAA shard: 20,359 rows; 13,454 (66.1%) outputs with G-code; 10,664 (52.4%) naming a benchmark group; 6,905 (33.9%) with T-code; median 29 words | `SFT/data/ift_data_2026_05_18_v21_taa.json` | `v16_additions.py` §1 |
| CSE shard: 17,320 rows; 0 G-codes / group names / T-codes in outputs; 8,801 (50.8%) inputs naming a benchmark group; median 3 words | `SFT/data/ift_data_2026_05_18_v21_cse.json` | same |
| VSP metric-level accuracy 82.6 / 83.1 / 76.6 / 77.0 / 74.8 | `SFT/test/responses/…-v21-*-vllm/athena-vsp/*_scored.jsonl` | §2 |
| VSP error direction (AV L→N 552 of 583; PR L→N 657 of 791 at Recal-32B; 83/33 and 166/98 at +TAA); gold distribution 65% Network, 36% PR:None; 658 Local items | same | §2 |
| MMLU-Pro per-subject deltas | workbook sheet `MMLU Pro` | §3 |
| Single-stage v7 32B row: 71.0 / 57.4 / 62.1 / 68.5 / 83.4 / 52.5; C 39.6; Y 88.6; A_Cl 65.8; B_Cl 64.7 | workbook sheet `Glaukopis (TEST)`; recipe from `SFT/autotrain/run_abaligned_sft_qwen25_32b_v7.sh`; corpus from `tmpl_gen/templates/04262026/` | §4 |
| Hosted input/output tokens (Table 17) | workbook sheet `Cost` / `SFT/eval/responses/cost_summary.csv` | §5 |
| Local wall-clock seconds (Table 17) | derived from the v1.5 run-record costs at $2.50/GPU-h × 2 GPUs (1,020 s for Recal-32B verified in v1.5 against the run manifests) | arithmetic |
| 26 node labels, 37 relationship types | `tmpl_gen/_db_schema.json` | counted |

Minor note: the metric-level accuracy is computed with the script's own vector parser; its exact-match counts differ from Table 14's existing counts by up to 9 items (e.g. Core 492 vs 483) because the v1.5 script used a stricter final-line rule. Table 14 keeps the v1.5 counts for the existing columns; the metric-level column agrees to 0.1 point under either parser.

## 5. What still cannot be answered without compute, in priority order

Both venues rejected on the same first point. If any training budget is available before resubmission, this is the order in which to spend it; each item slots into an existing table without restructuring the paper.

1. **Matched single-stage control (closes R2.1, strengthens R1.5).** One pass over the union of the four v21 shards (`core_a` + general mixture, `core_b`, `taa`, `cse`), same base, same effective batch, learning rate 5e-6, cutoff 8,192 packed, `max_samples` set so that realized tokens ≈ the chain's. Planning budget 25–45 h on 8×H100. Evaluate with `SFT/eval/utils/serve_and_bench_qwen25_32b_v21_<stage>.sh` on the full suite. Insert as a row in Table 7 and Table 8; rewrite the last paragraph of §6.3.
2. **Interleaved Stage 2+3 run (tests the mechanism in §8.3 and §8.4).** From the Core checkpoint, one pass over `taa` and `cse` interleaved 50/50 at 5e-6, cutoff 4,096. If Canonical TAA stays near 36.8 and VSP metric-level accuracy near 83, the interference account is confirmed and the remedy is demonstrated. 13–17 h.
3. **Three seeded Stage-4 runs (closes the "single run" caveat in §7.7).** 3–4 h each.
4. **Exact graph counts (closes R1.7 completely).** Five minutes if the Neo4j instance exists.
5. **Entity-disjoint held-out incident set (R2.6).** Requires building new items; the largest effort and the one the committee is least likely to require for this paper.

## 6. Author actions before upload (carried over from v1.5 plus new)

1. Set `\journal{}` and, if the venue is Springer (IJIS, Cybersecurity), port to `sn-jnl.cls`: the body is class-agnostic; only the front matter (`\author`/`\affiliation`/`\ead`/`\cortext`, `keyword`, `highlights`) and the bibliography style change. For IEEE Access, port to `ieeeaccess.cls`.
2. Confirm the legal entity name (TODO at the affiliation line): the affiliation, funding, competing-interest statement and `glaukopis2026software` bib entry say "Athena Security Group, LLC"; use the correct form consistently.
3. Confirm RevolveAI city/country, every ORCID, and the CRediT roles for S. Ahmad and M. M. Khan.
4. Deposit the supplementary package (per-item artifacts, build reports, `analysis/` scripts including `v16_additions.py`, `data/` CSV/JSON) and insert the DOI in Data availability.
5. Re-check the AI-use declaration wording against the target journal's policy.
6. Cover letter: state that an abridged version was rejected by ICTAI 2026 and the full version by Computers & Security, that both sets of reviews were addressed (attach this document's §1–2 as the response), and that the manuscript is not under consideration elsewhere.
7. The `%% TODO` at §5.1 (graph counts) — run the queries or delete the comment.
