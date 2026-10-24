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
