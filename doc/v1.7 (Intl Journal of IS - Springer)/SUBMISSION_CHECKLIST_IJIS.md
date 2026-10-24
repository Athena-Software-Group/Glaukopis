# International Journal of Information Security (Springer) — submission checklist for Glaukopis

Checked against https://link.springer.com/journal/10207/submission-guidelines on 2026-10-24.
Manuscript: `glaukopis_ijis.tex` (v1.7: the v1.6 content ported to svjour3, plus the expanded background, Sophia, research-directions and appendix material from the Oxford drafts; see CHANGELOG_v1.7.md).

## Requirements and how the manuscript meets them

| Guideline | Requirement | Status |
|---|---|---|
| Peer review | **Single-blind.** The page gives no anonymization instructions. | Author-identified manuscript is correct. Names, affiliations, email, ORCID, funding and competing interests stay in the PDF. **Do not anonymize.** |
| Template | Springer LaTeX macro package (svjour3) with the **"twocolumn"** option. | `\documentclass[twocolumn]{svjour3}`; `svjour3.cls` + `svglov3.clo` in this folder (your download is the EPJC variant of the same package; the class is the same). |
| Source files | Complete editable source incl. style files and figures, plus a compiled PDF. "Failing to submit a complete set of editable source files will result in your article not being considered for review." | Upload: `glaukopis_ijis.tex`, `glaukopis_ijis.bib`, `glaukopis_ijis.bbl`, `svjour3.cls`, `svglov3.clo`, `spmpsci.bst`, `Fig1.eps`/`Fig1.pdf`, `Fig2.eps`/`Fig2.pdf`, `glaukopis_ijis.pdf`. (The figures are also embedded as TikZ source in the .tex, so the PDF builds without the figure files.) |
| Title page | Concise title; all author names; affiliations with institution, city, state, country; corresponding author with active email; 16-digit ORCID for each author if available. | `\title`, `\author`, `\institute`, `\email`. **TODO:** ORCID for Cardei (only Worth's is present). |
| Abstract | 150–250 words; no undefined abbreviations or unspecified references. | 245 words; CTI defined at first use; MITRE ATT&CK and MMLU-Pro are proper names; no citations. |
| Keywords | 4–6. | 6 (`\keywords`). |
| Headings | Decimal numbering, no more than three levels. | Two numbered levels + run-in `\paragraph` headings. |
| Abbreviations | Define at first mention. | All defined at first use; glossary Table 1 (Section 2) repeats them. |
| Footnotes | Numbered consecutively; table footnotes as lowercase superscript letters. | No text footnotes; table notes use ^a, ^b, ^c. |
| References | **Numbered, in square brackets**; list numbered consecutively; only cited works; DOIs where available; journal names abbreviated. | `spmpsci.bst` (Springer's numbered style; alphabetical order, `[n]` citations, prints DOI/URL). Every entry is cited. |
| Tables | Arabic numerals, cited consecutively; caption above; footnotes below. | svjour3 places captions above; all tables are `table*` (two-column span) and cited in order. |
| Figures | Electronic; EPS preferred for vector; fonts embedded; file names `Fig1`, `Fig2`; captions in the text starting "Fig. n"; readable in grayscale; ≤174 mm wide. | `Fig1.eps`/`.pdf` (177 × 128 mm) and `Fig2.eps`/`.pdf` (168 × 109 mm) exported from the TikZ/pgfplots source (graphics program: TikZ/PGF 3.1 via pdfLaTeX). Both are vector with embedded fonts; Fig. 1 uses pale fills plus text labels, Fig. 2 one series with error bars, so both read in grayscale. Captions are produced by the class ("Fig. 1 …"). |
| Statements and Declarations | Before the reference list: Competing interests (mandatory), Funding (full funder names), Data availability (mandatory for original research), Author contributions (recommended; CRediT acceptable), ethics statements where relevant. | Section "Statements and Declarations": Funding; Competing interests; Ethical approval and dual use; Data availability; Author contributions (CRediT); Use of large language models. |
| AI use | LLMs cannot be authors; LLM use must be documented in Methods or equivalent; copy-editing need not be declared. | Documented twice: one sentence in Section 7.1 (evaluation protocol = methods) and the "Use of large language models" declaration. |
| Dual-use research | Must be identified in the manuscript. | "Ethical approval and dual use" paragraph. |
| Research data | Data Availability Statement explaining access; deposit in a repository; cite datasets with a persistent identifier. | Statement present; **TODO:** deposit Online Resource 1 (per-item artifacts, build reports, `analysis/` scripts, `data/`) in Zenodo and insert the DOI. |
| Supplementary information | Referred to as "Online Resource n"; each file carries article title, journal, authors, corresponding author's affiliation and email; captions required; text as PDF, spreadsheets as csv/xlsx. | Manuscript refers to "Online Resource 1". **TODO:** assemble the package (see below) with a one-page PDF cover listing the metadata. |
| Prior submission / conference extension | Original, not published, not under review elsewhere; conference papers must be disclosed with ≥40% new material. | Neither prior submission (ICTAI 2026 abridged; Computers & Security full) was published, so this is not a conference extension. Disclose both in the cover letter (template in `COVER_LETTER_IJIS.md`), not in the manuscript. |
| Plagiarism screening | Automatic. | The ICTAI abridged draft and C&S draft share text with this manuscript; neither is published, so no overlap with published work. arXiv preprint: none posted. |
| Self-citation | Excessive self-citation discouraged. | 4 self-citations of 46 (AthenaBench, Minerva, the software repository, and Fieblinger et al., which shares a co-author with AthenaBench) (AthenaBench, Minerva, the software repository), all necessary. |
| Open access | Open Choice hybrid; APC on the "How to publish with us" page. | The journal page (2026-10-24) headlines a move to fully open access; **check the APC (listed as US$3,690) and FAU's Springer Nature agreement before submitting; a discretionary waiver can only be requested at submission.** |
| Reviewer suggestions | Optional; independent, institutional emails. | Prepare 3–4 names if desired (no AthenaBench co-authors, no Athena, RevolveAI or FAU EECS affiliates). |

## Online Resource 1 — what to deposit

1. `SFT/test/responses/athena-cti-sft-qwen25-32b-v21-{core,taa,cse,recalibrate,recal-32b}-vllm/` (per-item `_scored.jsonl`, CSVs, `summary_*` manifests)
2. `_v21_{core,taa,cse}_build/{dedup_report,watcher_status,row_count_gate_report}.json`
3. `doc/v1.5 (Computers & Security)/analysis/` scripts and `data/` CSVs (bootstrap, entity overlap, overlap subsets, make_tables)
4. `doc/v1.6/analysis/v16_additions.py` and `doc/v1.6/data/v16_additions.json`
5. A one-page `README.pdf` stating: article title, journal name, author names, corresponding author's affiliation and email, and a caption per file (Springer requires this metadata in each supplementary file).

## Author actions before upload (in order)

1. Fill the `%% TODO (authors)` items in the .tex: Cardei's ORCID; legal entity name (trade name vs Athena Software Group, Inc.) used consistently in the institute line, Funding, Competing interests and the `glaukopis2026software` bib entry.
2. Decide whether to run the matched single-stage control first (`doc/v1.6/RESPONSE_TO_REVIEWERS_v1.6.md` §5). If run, it is one row in Table 7 and one row in Table 8 plus a rewrite of the last paragraph of Section 6.3.
3. Deposit Online Resource 1; insert the DOI in the Data availability paragraph; remove the TODO comment.
4. Run `bash build.sh`; check that the log reports no undefined citations or references and that the page count is ~29.
5. Confirm APC coverage or request the waiver at submission.
6. Submit via the journal's "Submit manuscript" link with the cover letter.
