# Journal of Systems and Software (Elsevier): submission checklist for Glaukopis v1.8

**Checked:** 2026-10-25, against the Guide for Authors PDF in this folder (`Guide for authors - Journal of Systems and Software - ISSN 0164-1212 _ ScienceDirect.com by Elsevier.pdf`).

**Manuscript:** `glaukopis_jss.tex` (Elsevier CAS double-column class, 31 pages).

## Requirements and status

| Guide section | Requirement (quoted or paraphrased from the guide) | Status |
|---|---|---|
| Aims and scope | All of software engineering, including "Software Engineering for AI systems" and "Artificial Intelligence, data analytics and big data applied in software engineering"; "welcomes reports of practical experience …, as well as replication studies and studies with negative results." | Framed as a software-engineering-for-AI study: research questions RQ1–RQ5, data-quality checks (Table 5), lessons learned (§10.2), Wohlin-style threats to validity (§11). |
| Article type | Regular papers, the *In Practice* track (Applied Research Reports; Practitioner Insights ≤ 4 pages, industry first author), and *New Ideas and Trends Papers*. | Submit as a regular paper. An Applied Research Report is a fallback; it asks for the industrial setting and lessons learned, which the paper already has. |
| **Length** | "It is encouraged that authors submit full-length papers of less than 36 pages single-column or 18 pages double-column. If your manuscript is longer, please include an explanation in your submission as to why the length is justified." | **Over: 31 pages double-column** (main text to p. 25, appendices to p. 28, then declarations and references). The cover letter has a justification paragraph. **Decision:** keep it, or condense first (see "Length options" below). |
| Peer review | Single anonymized; at least two reviewers. | Author-identified manuscript. Do not anonymize. |
| File format / LaTeX | Submit `.tex` source; "Double-column formatting is only permitted for LaTeX submissions." The Elsevier template is encouraged. | Two-column CAS (`cas-dc`), LaTeX source. |
| Title page | Title; authors; affiliations with "the full name and postal address of each affiliation, including the country name"; corresponding author's e-mail. | Both ORCIDs present (Cardei 0009-0000-1050-768X). FAU address complete. **TODO:** postal address for Athena Software Group Inc. (`%% TODO` at the affiliation). |
| Abstract | ≤ 250 words, stands alone, no references, abbreviations defined. | 246 words; CTI defined; no citations. |
| Keywords | 1–7, in English; avoid multi-word keywords joined by "and" or "of". | 7, none joined by "and" or "of". |
| Highlights | **Required**: a separate editable file with "highlights" in the file name, 3–5 bullets of ≤ 85 characters including spaces. | `highlights.txt`: 5 bullets, 82–85 characters. Not typeset in the PDF. |
| Graphical abstract | Encouraged, not required: ≥ 1328 px wide, readable at 13 cm. | None. Optional: a cropped Figure 1 would serve. |
| Tables | Editable; cited; numbered in order; captions; notes below; no vertical rules or shading. | Compliant (booktabs; notes below the tables). |
| Figures | Separate files with logical names (e.g. `Figure_1`); vector as EPS or PDF; text graphics may be embedded in LaTeX. | Embedded TikZ, plus `figures/Figure_1.pdf`/`.eps` and `figures/Figure_2.pdf`/`.eps` for upload. |
| Appendices | Lettered A, B, …; tables numbered "Table A.1", equations "(A.1)". | Done (Table B.1). |
| CRediT | Required, using the CRediT role names. | `\credit{}` per author with the guide's role names, printed by `\printcredits`. |
| Competing interests | Declare in the manuscript **and** complete the Elsevier declarations tool; upload its Word file at "attach/upload files". | Section present (Athena Software Group Inc. funding and affiliation; Worth's AthenaBench co-authorship). **TODO:** complete the declarations tool at submission. |
| Funding | Standard form "Funding: This work was supported by …"; declare the sponsor's role in design, data, analysis, writing, and the decision to submit, or state that it had none. | Rewritten in that form with a sponsor-role sentence. **TODO:** confirm the sentence (`%% TODO`). |
| Generative AI | A section before the references titled "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", in the guide's template wording. | Done, with the guide's title and wording; methods use is also stated in §7.1. |
| Research data | **Option C**: deposit the data in a repository **and cite and link the dataset in the article** (reference marked `[dataset]`), or explain why it cannot be shared. A data statement is required at submission. | Data availability section present. **TODO:** deposit on Zenodo or Mendeley Data, add the `[dataset]` bib entry (template in the `%% TODO` under Data availability) and cite it. Fill in the data statement in Editorial Manager. |
| Software references | Creator, title, repository with a persistent identifier (DOI preferred), date, version, `[software]`. | `glaukopis2026software` cites GitHub at a commit. **Recommended:** archive the release on Zenodo for a DOI. **TODO:** update the commit hash after this commit. |
| References | Any consistent style at submission; author–year citations in the text; DOIs where available; every reference must be real; mark preprints. | Author–year (`natbib` + `cas-model2-names`); 60 entries, all cited, preprints marked "arXiv preprint". |
| Article structure | Numbered sections; numbered cross-references. Acknowledgements, if any, go in a separate section directly before the references. | Compliant. No acknowledgements section (add one if anyone should be thanked). |
| Vitae | "Please submit a short (maximum 100 words) biography of each author", in an editable format. | **TODO:** fill in `VITAE.md` and upload it as .docx. |
| Language | American or British English, not both. | American English. British spellings remain only inside verbatim training records (Appendix C). |
| Open Science | Optional JSS Open Science Initiative: checked after acceptance and does not affect the decision; "If material cannot be shared (e.g., due to NDAs), authors should state this explicitly." | **Decide** whether to opt in. The repository is public and source-available (PolyForm Noncommercial), not open source; say so. Confirm the `asg-ai` Hugging Face checkpoints are public. |
| Preprints | Allowed; an optional SSRN posting is offered during submission. | Your choice. Posting would also give a citable DOI for the dissertation. |
| Submission declaration | Not published, not under consideration elsewhere, approved by all authors. | True. Prior rejections are disclosed in the cover letter. |

## Length options (decide before upload)

1. **Submit at 31 pages** with the justification paragraph in `COVER_LETTER_JSS.md`. The guide permits this.
2. **Move Appendices A–C to supplementary material**, which saves about three pages. Appendix C, with its verbatim records, makes the best standalone supplement.
3. **Condense to about 20–22 pages:**
   - shorten §4 background, especially §4.2 (the CTI catalogs list) and §4.6 (benchmarks);
   - merge Tables 2 and 4 into the text;
   - trim §8.5 (qualitative examples) and §12 (research directions, which overlaps §11);
   - shorten §9 (inference efficiency) to one paragraph and one table.

   This is a substantive edit; do it after the matched control, because that run also changes §6.3.

## Files to upload

- `glaukopis_jss.tex`, `glaukopis_jss.bib`, `glaukopis_jss.bbl`
- `cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst`
- `figures/Figure_1.pdf`, `figures/Figure_2.pdf`
- `glaukopis_jss.pdf`
- `highlights.txt` (keep "highlights" in the file name)
- vitae, as .docx
- the declarations-tool Word file
- the cover letter

## Replication package (deposit in Zenodo or Mendeley Data, then cite it as `[dataset]`)

1. `SFT/test/responses/athena-cti-sft-qwen25-32b-v21-{core,taa,cse,recalibrate,recal-32b}-vllm/`: per-item `_scored.jsonl` files and `summary_*` manifests. Add the matched control's directory once it exists.
2. `_v21_{core,taa,cse}_build/*_report.json`: the row-count, licence, letter-balance, substrate, seed-provenance and dedup reports behind Table 5.
3. Analysis scripts and their outputs:
   - `doc/v1.5 (Computers & Security)/analysis/` and `data/`
   - `doc/v1.6/analysis/v16_additions.py` and its `data/v16_additions.json`
   - `doc/v1.8 …/analysis/v18_additions.py` and its `data/v18_additions.json`
4. A README listing the article title, journal, authors, the corresponding author's contact, and one caption per file.

## Author actions before upload (in order)

1. **Decide on the matched single-stage control** (`SINGLE_STAGE_BASELINE.md`). Recommended: run it first, then update every `MATCHED-CONTROL` / `STAGING-CLAIM` marker, the last highlight, and the cover letter.
2. **Decide on length** (options above).
3. Fill the remaining `%% TODO (authors)` items:
   - Athena Software Group Inc. postal address;
   - the funding sponsor-role sentence;
   - the replication-package DOI and `[dataset]` reference;
   - the software commit or DOI;
   - the Neo4j graph counts (optional; carried over from v1.7).
4. Fill in `VITAE.md`.
5. Edit the cover letter's "Prior history" paragraph to match what happened at IJIS.
6. Run `bash build.sh`. Check that no citation or reference is undefined. One overfull box at `\maketitle` and an italic-monospace font substitution are harmless.
7. In Editorial Manager:
   - choose the research-paper type;
   - complete the declarations tool;
   - fill in the data statement;
   - decide on the SSRN preprint and the Open Science opt-in.

## Texifier note

`cas-dc.cls` and `cas-common.sty` are loaded from this folder, but they need `expl3`/`xparse` features and the `stix`/`inconsolata` fonts that may be missing from Texifier's bundled 2017-era package set. If the Texifier build fails or falls back to Computer Modern, switch Texifier to the system TeX Live 2024: `bash build.sh` builds cleanly with it. pgfplots stays at `compat=1.15` for Texifier compatibility.
