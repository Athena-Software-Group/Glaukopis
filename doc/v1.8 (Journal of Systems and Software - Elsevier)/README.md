# Journal of Systems and Software (Elsevier): submission package, v1.8

Created 2026-10-25 from `doc/v1.7 (Intl Journal of IS - Springer)/glaukopis_ijis.tex`. The v1.7 content was re-framed for a software-engineering readership and ported to Elsevier's two-column CAS class (`cas-dc`).

- `SUBMISSION_CHECKLIST_JSS.md`: requirement-by-requirement check against the journal's Guide for Authors, plus the remaining author actions.
- `SINGLE_STAGE_BASELINE.md`: how the paper answers the single-stage-baseline point raised by the earlier reviewers and Dr. Cardei.

## Files

| File | Purpose |
|---|---|
| `glaukopis_jss.tex`, `glaukopis_jss.bib`, `glaukopis_jss.bbl` | Article source and bibliography (author–year, `cas-model2-names.bst`). |
| `glaukopis_jss.pdf` | Compiled article (27 pages, two-column). |
| `glaukopis_jss_supplementary.tex` / `.pdf` | Supplementary material, formerly Appendices A–C plus graph statistics: S1 evaluation suites; S2 contamination screen with Table S1; S3 verbatim training records; S4 graph statistics with Table S2. Its references to article sections and tables resolve through `xr-hyper` from `glaukopis_jss.aux`. |
| `build.sh` | Builds the article, then the supplement (`pdflatex → bibtex → pdflatex ×2` each). |
| `highlights.txt` | Five highlights (≤ 85 characters), for separate upload; not typeset in the PDF. |
| `VITAE.md` → `vitae.docx` | Author biographies (≤ 100 words each), as the guide requires. Sources: the Athena leadership page and Cardei's FAU pages. |
| `DECLARATION_OF_INTEREST.md` → `Declaration_of_Interest_Statement.docx` | Separate Declaration of Interest Statement in Elsevier's declarations-tool format, to upload as "Declaration of Interest Statement". Complete: Worth (employment, board, shares); Cardei (advisory, incentive options). |
| `COVER_LETTER_JSS.md` | Cover letter. The "Prior history" paragraph is marked for you to confirm. |
| `SUBMISSION_CHECKLIST_JSS.md` | Guide-by-guide check, files to upload, author actions. |
| `SINGLE_STAGE_BASELINE.md` | The single-stage-baseline framing, with the record behind it. |
| `analysis/v18_additions.py` → `data/v18_additions.json` | Reproduces the earlier-single-pass numbers quoted in §6.3 and the related budget figures from the workbook and shards. Run: `GLAUKOPIS_ROOT=/path/to/Glaukopis python3 analysis/v18_additions.py` (needs `openpyxl`). |
| `analysis/graph_counts.cypher` → `data/graph_counts_v21.json` | The Cypher queries and output behind the graph statistics in §5.1 and Table S2. Run 2026-10-25 against the release's Neo4j instance (`athena-cti-db`, last written 2026-05-04); its entity counts match the v21 build's substrate report. |
| `figures/` | `Figure_1`/`Figure_2` as vector PDF/EPS plus their TikZ sources. The article compiles the same figures inline. |
| `cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Elsevier CAS files the build needs. |
| `Guide for authors - Journal of Systems and Software ….pdf` | Elsevier's guide, downloaded 2026-10-25. Untracked: it is Elsevier's page and the repository is public. |
| Other template files | Elsevier's elsarticle/CAS download (templates, samples, `doc/`, `figs/`, `thumbnails/`, other `.bst`/`.cls`). Not part of the submission. |

## What changed from v1.7

- **Venue and format.**
  - Elsevier CAS two-column class with author–year references.
  - Elsevier declarations: CRediT via `\printcredits`, competing interest, funding with the sponsor's role, ethics, data availability, and the generative-AI declaration with the guide's title.
  - Highlights and vitae as separate files; appendices moved to the supplement.
- **Software-engineering framing.**
  - New title and abstract (248 words).
  - Five research questions (RQ1–RQ5).
  - §4.1 on engineering ML systems and their data: SE4AI, data validation, datasheets and model cards, leakage, LLM threats in SE research.
  - Table 5, the data-quality checks: what each detects, what it misses, and two silent failures that led to new gates.
  - §10: answers to the RQs and seven lessons learned.
  - §11: threats to validity by construct, internal, external, conclusion, reliability (Wohlin et al. 2012).
- **Staging.**
  - The Stage-1 (Core) checkpoint is the study's single-stage baseline; §6.3 defines it.
  - New §8.3 reports the full chain against it with paired item-level intervals. The later stages add 3.2 points to the three-benchmark mean (interval +2.2 to +4.2) and cost 2.3 on the balanced AthenaBench score, 15.0 on Canonical attribution, and 5.3 on severity prediction.
  - RQ4, lesson L7, the internal-validity threat and the conclusion follow this framing.
  - The earlier single-pass releases (32B v7; 14B v7/v10/v11) are summarized in one paragraph as design history.
  - §4.3 cites the forgetting literature and Dong et al. (2024).
- **Author facts.**
  - The competing-interest statement names the authors' Athena roles: founder and CEO (Worth), Chief Scientist (Cardei).
  - Data availability: code and templates are in the public repository; shards, per-item files and build reports are available on request.
  - ORCIDs for both authors.
  - Athena Software Group Inc., 294 Queens Ct, West Palm Beach, FL 33401, and FAU's postal address.
  - The software reference cites the public repository with an access date.

No result number from v1.7 changed. The new numbers are the paired differences already reported in the paired-differences table (Table 11), and the earlier-single-pass figures reproduced by `analysis/v18_additions.py`.

## Remaining markers

None. The Neo4j graph counts are filled in: 214,651 nodes, 330,935 relationships, 31 labels, 41 types. They replace v1.7's approximate "~280K nodes, ~480K edges" and the "26 labels, 37 types" taken from an April schema snapshot, both of which were wrong.

## Related repository files

The matched single-stage-control launcher (`SFT/autotrain/run_sft_qwen25_32b_v21_single_stage.sh`), its bench wrapper, and two model aliases were committed on 2026-10-25 (`931a087`). With the baseline framing above, the paper does not depend on them. They remain available if a reviewer asks for a single pass over all four shards.

## Build

`bash build.sh` with TeX Live 2024. For Texifier, see the note at the end of `SUBMISSION_CHECKLIST_JSS.md`.
