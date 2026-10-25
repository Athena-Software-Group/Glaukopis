# Journal of Systems and Software (Elsevier): submission checklist for Glaukopis v1.8

**Checked:** 2026-10-25, against the Guide for Authors PDF in this folder (`Guide for authors - Journal of Systems and Software - ISSN 0164-1212 _ ScienceDirect.com by Elsevier.pdf`).

**Files checked:**

- `glaukopis_jss.tex`: Elsevier CAS double-column, 27 pages.
- `glaukopis_jss_supplementary.tex`: 7 pages.

## Requirements and status

| Guide section | Requirement (quoted or paraphrased) | Status |
|---|---|---|
| Aims and scope | All of software engineering, including "Software Engineering for AI systems"; welcomes "reports of practical experience … replication studies and studies with negative results." | Framed as a software-engineering-for-AI study: RQ1–RQ5, data-quality checks (Table 5), lessons learned (§10.2), threats to validity (§11). |
| Article type | Regular papers; the *In Practice* track (Applied Research Reports; Practitioner Insights ≤ 4 pages, industry first author); *New Ideas and Trends Papers*. | Submit as a regular paper. An Applied Research Report is the fallback. |
| **Length** | "It is encouraged that authors submit full-length papers of less than … 18 pages double-column. If your manuscript is longer, please include an explanation in your submission as to why the length is justified." | **27 pages** (main text to p. 24; then declarations and references). Appendices already moved to the supplement. The justification is in the cover letter's "Length" paragraph. |
| Peer review | Single anonymized. | Author-identified; do not anonymize. |
| File format / LaTeX | `.tex` source; double-column allowed only for LaTeX. | Done. |
| Title page | Authors; affiliations with full postal address including country; corresponding author's e-mail. | Done: both ORCIDs; FAU, 777 Glades Road, Boca Raton, FL 33431; Athena Software Group Inc., 294 Queens Ct, West Palm Beach, FL 33401. |
| Abstract | ≤ 250 words, no references. | 248 words. |
| Keywords | 1–7; avoid multi-word keywords joined by "and" or "of". | 7. |
| Highlights | Required: separate editable file with "highlights" in its name, 3–5 bullets, ≤ 85 characters each. | `highlights.txt`: 5 bullets, 80–85 characters. |
| Graphical abstract | Encouraged. | None (optional). |
| Tables | Editable; cited in order; captions; notes below; no vertical rules. | Compliant. |
| Figures | Separate files, logical names, vector EPS/PDF; LaTeX may embed. | Embedded TikZ, plus `figures/Figure_1.pdf` and `figures/Figure_2.pdf`. |
| Supplementary material | Cite in the text; concise caption per file; submit with the manuscript. | Done. The article cites Sections S1–S3 and Table S1, and has a "Supplementary material" section captioning the file. The supplement carries the title, authors, affiliations and contact. |
| Appendices | Lettered, with tables numbered "Table A.1". | None remain in the article. The supplement numbers its table S1. |
| CRediT | Required, using the CRediT role names. | `\credit{}` and `\printcredits`. |
| Competing interests | In the manuscript **and** in the Elsevier declarations tool (upload its Word file). | Section present. **At submission:** complete the declarations tool. |
| Funding | "Funding: This work was supported by …"; the sponsor's role, or a statement that it had none. | Done (Athena Software Group Inc.; its role was exercised through the authors). |
| Generative AI | Section before the references, with the guide's title and wording. | Done. |
| Research data | **Option C**: deposit the data in a repository and cite and link it, or state why it cannot be shared; a data statement is required at submission. | Done. Code, templates, harness, workbook and analysis scripts are in the public GitHub repository, and the benchmarks are cited individually. The training shards (≈ 340 MB), per-item result files (≈ 290 MB) and build-check reports are "too large to host in the repository and are available from the corresponding author on reasonable request". That is the reason Option C requires. Use the same wording in Editorial Manager's data statement. |
| Software references | Creator, title, repository, date or version, `[software]`. | `glaukopis2026software`: GitHub, `[software]`, accessed 25 October 2026 (the guide accepts an access date when no version is pinned). |
| References | Any consistent style; author–year in the text; real sources; preprints marked. | Done. |
| Vitae | ≤ 100 words per author, editable. | `vitae.docx` / `VITAE.md` (97 and 96 words), from the Athena leadership page and Cardei's FAU pages. |
| Language | American or British English, not both. | American English. British spellings remain only in verbatim training records (S3). |
| Open Science | Optional opt-in, checked after acceptance. | Decide at submission. The repository is public and source-available (PolyForm Noncommercial). |
| Preprints | Optional SSRN posting at submission. | Your choice. |

## Files to upload

- `glaukopis_jss.tex`, `glaukopis_jss.bib`, `glaukopis_jss.bbl`
- `cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst`
- `glaukopis_jss.pdf`
- `figures/Figure_1.pdf`, `figures/Figure_2.pdf`
- `glaukopis_jss_supplementary.pdf`, as Supplementary material
- `highlights.txt`
- `vitae.docx`
- the declarations-tool Word file
- the cover letter

## Two Editorial Manager steps explained

**Article type.** This is not about layout. Editorial Manager's first screen asks which category of contribution you are submitting, from a drop-down of the journal's types: research papers, the *In Practice* track, *New Ideas and Trends Papers*, and special-issue ("VSI: …") entries. Choose the regular research-paper entry; its exact label is whatever the drop-down shows. The two-column layout is a separate matter: the PDF is already two-column, and the guide permits that for LaTeX submissions.

**Declarations tool.** Elsevier asks every author group to fill in a short web form, the *Declaration of Interest tool*. Editorial Manager links it during submission, and it is also reachable from Elsevier's "declaration of competing interests" author page. You tick the categories that apply and type a short statement; the tool then produces a Word file, which you upload at the "attach files" step as the item type *Declaration of Interest Statement*. The content must match the manuscript's "Declaration of competing interest" section:

- **Funding:** Athena Software Group Inc. funded the work and develops Glaukopis.
- **Employment / leadership:** P. J. Worth Jr. is founder, president and CEO of Athena Software Group Inc.; I. Cardei is its Chief Scientist.
- **Stock ownership:** tick it if either author holds equity in Athena. The manuscript does not currently say so. If you tick it, add "and holds equity in" to the manuscript's declaration so the two match.
- **Other relationships:** P. J. Worth Jr. is a co-author of the AthenaBench benchmark used in the evaluation.

## Author actions before upload

1. Run `bash build.sh`. Check that neither log reports undefined citations or references.
2. In Editorial Manager:
   - choose the research-paper article type;
   - complete the declarations tool and upload its Word file (see above);
   - fill in the data statement;
   - decide on SSRN and the Open Science opt-in.

## Texifier note

`cas-dc.cls` and `cas-common.sty` load from this folder but need `expl3`/`xparse` and the `stix`/`inconsolata` fonts, which Texifier's 2017-era bundle may lack. The supplement also needs `xr-hyper`. If Texifier fails, switch it to the system TeX Live 2024: `bash build.sh` builds both files cleanly there.
