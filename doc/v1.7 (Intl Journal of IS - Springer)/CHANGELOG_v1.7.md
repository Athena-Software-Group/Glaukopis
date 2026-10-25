# v1.7.2 (2026-10-25): content backported from v1.8 (JSS)

The manuscript was rebuilt from `doc/v1.8 (Journal of Systems and Software - Elsevier)/glaukopis_jss.tex`. It keeps Springer's svjour3 two-column format, numbered references (`spmpsci`), front matter, and "Statements and Declarations". The submitted v1.7.1 source is preserved in `previous/glaukopis_ijis_v1.7.1_submitted.tex` / `.bib`. The PDF is now 34 pages.

- **Title.** "Glaukopis: Layered Instruction Fine-Tuning for Cyber Threat Intelligence LLM Training". The stages are described as layered, each on the checkpoint of the one before.
- **Single-stage baseline.**
  - The Stage-1 (Core) checkpoint is the study's single-stage baseline (§6.3).
  - New §8.3 reports the full chain against it with paired intervals: +3.2 on the three-benchmark mean (+2.2 to +4.2); −2.3 on the balanced AthenaBench score; −15.0 on Canonical TAA; −5.3 on VSP.
  - The unmatched v7 comparison table and its "comparable update budget" claim are replaced by one paragraph of design history.
- **Exact graph statistics.**
  - 214,651 nodes, 330,935 relationships, 31 labels and 41 relationship types, in §5.1, in Figure 1 (Fig1.pdf/.eps regenerated), and in the new Appendix D table.
  - These replace "~280K nodes, ~480K edges" and "26 labels, 37 types".
- **Data pipeline.** A quality-checks table records what each check detects and misses, including the two silent failures (v11 under-build, v17 parser defect) that led to new gates. The row-count gate is described as per task axis.
- **Framing carried over from v1.8.**
  - Research questions RQ1–RQ5.
  - Background on engineering ML systems and their data.
  - The forgetting literature, and Dong et al. (2024) on mixed versus sequential SFT.
  - Answers to the RQs and seven lessons learned.
  - Threats to validity by construct, internal, external and conclusion validity, and reliability.
- **Author facts.**
  - ORCIDs for both authors.
  - Athena Software Group Inc. (294 Queens Ct, West Palm Beach, FL 33401) and FAU's postal address.
  - Competing interests: Worth is founder, president, CEO and board chairman and holds shares; Cardei advises Athena as Chief Scientist and holds incentive options.
  - Funding statement with the sponsor's role.
  - Data availability: code in the public repository; shards, per-item results and build reports on request. "Online Resource 1" is no longer referenced.
- **Bibliography.** Same 60 records as v1.8.

Not changed: the cover letters (`COVER_LETTER_IJIS.*`) remain the record of the original submission.

# v1.7 (2026-10-24) — expansion of the IJIS manuscript with the Oxford-draft material

Base: v1.6 content in svjour3 format (22 pages). Result: 29 pages. No numbers changed; every added
statement was checked against the repository artifacts, and the Oxford claims that conflict with them
were left out (see README).

## Added

- **§4 Background and related work**, rewritten and expanded from two paragraphs to five subsections:
  CTI and its public catalogs (what ATT&CK, CVE/NVD, CWE, CAPEC, CVSS, D3FEND, ENGAGE and the
  exploitation/detection sources are, and why they form a graph); instruction fine-tuning and staged
  training (FLAN, InstructGPT, FLAN-T5, domain-adaptive training — with the staging claim kept to what
  the article can support); graph-grounded instruction data (three research lines, with verified CTI
  knowledge-graph citations); security-specific instruction data and models (CyberPal/SecKnowledge detail);
  cyber-security benchmarks (why domain-specific, the four families, and why a portfolio).
  Addresses Reviewer 1's "lacks the necessary background" and "choices not justified".
- **§5.1 source selection**: the nineteen candidate sources surveyed in August 2025 (from
  `training/Cyber Threat Intel Datasets & Sources 08122025.xlsx`) and the three criteria that separate
  the twelve selected ones (stable machine-readable identifiers; cross-references to other catalogs;
  checkable redistribution licence). Addresses "justify the choices".
- **§5.2 Sophia** expanded from one subsection to five parts: template language (with the M.1 single-node
  example and the CVE/CWE/CAPEC multi-hop example), record format and family identifiers (so that the
  `AB.*`, `JS.CSE.*`, `YN.*` names used in the analyses are explained), the three-step generation
  pipeline with the row-count gate, balancing/screening/hold-out, and schema validation with its limits.
- **§5.3** provenance paragraph expanded: the release is a hash-verified byte-identical replay of the
  preceding release with a ±1.5-point sign-off gate that two task scores failed; non-determinism in the
  data build is the stated explanation.
- **§11 Research directions** (new): controls first; interference-aware scheduling; verifiable-reward
  training on graph-derived answers; held-out incident evaluation; paraphrastic leakage; surface diversity
  and rationales; larger bases and operational dimensions.
- **Appendix A** The evaluation suites: AthenaBench, CyberSOCEval, CyberMetric described from their
  papers and from the harness as executed.
- **Appendix B** The contamination screen: mechanism, enforcement, per-benchmark posture table
  (Table 18) — corrected from the Oxford version: CyberSOCEval and CTIBench were *not* in the release
  index.
- **Appendix C** Example training records: one verbatim row from each major family (AB.MCQ, X.MT,
  YN.MN, CL.V, AB.RCM, AB.RMS, AB.ATE, AB.VSP, AB.TAA, JS.CSE.TI.GRP), including two rendering defects
  found in the released shards (`CWE-CWE-319`; an empty detection-guidance clause) and one source
  artifact (ATT&CK legacy mitigation IDs). These are reported because they show what
  "graph-consistent but not necessarily correct" means and because a reviewer who samples the data
  will find them.
- Seven verified references (Fieblinger et al. 2024; CTINexus; K-CTIAA; SecEval; CyberSecEval 1 and 2;
  WMDP). The Oxford entries for Open-CyKG (wrong authors), CTIKG, CyberKG and the internal
  `cardei2025ift` report were not used.

## Cross-checks performed with the new `results/` and `training/` folders

- `results/Glaukopis Results - 05272026.xlsx` is numerically identical to `SFT/eval/Glaukopis Results.xlsx`
  on every row of the headline sheet; `results/Athena-CTI-LLM SFT Benchmarking 05242026.xlsx` differs only
  in a model-name typo (`ftdn-ai`). The paper's numbers are therefore stable across the May snapshots.
- `training/athena_cti_instruction_format_requirements.txt` and the IFT guides confirm the Alpaca
  three-field format and the task families described in §5.2; no numbers in the paper depend on them.
- The `training/samples/IFT_*.jsonl` files are early (2025) per-source samples that predate Sophia's
  template language; the appendix uses rows from the actual v21 shards instead.

## Author actions added by this version

- Appendix C exposes two template defects. Decide whether to (a) keep them documented as is, or (b) fix
  the templates (`CL.V.1`, `AB.ATE.1`), regenerate, and note the fix. (a) is honest and costs nothing;
  (b) requires regenerating the Core shard and would invalidate the reported results unless retrained.
