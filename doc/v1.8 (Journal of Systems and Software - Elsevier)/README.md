# Journal of Systems and Software (Elsevier): submission package, v1.8

Created 2026-10-25 from `doc/v1.7 (Intl Journal of IS - Springer)/glaukopis_ijis.tex`. The v1.7 content was re-framed for a software-engineering readership and ported to Elsevier's two-column CAS class (`cas-dc`).

- `SUBMISSION_CHECKLIST_JSS.md`: requirements, files to upload, and remaining author actions.
- `SINGLE_STAGE_BASELINE.md`: the assessment of Reviewer 2's single-stage-control point, raised again by Dr. Cardei, and the protocol for closing it.

## Files

| File | Purpose |
|---|---|
| `glaukopis_jss.tex`, `glaukopis_jss.bib`, `glaukopis_jss.bbl` | Manuscript source and bibliography (author–year, `cas-model2-names.bst`). |
| `glaukopis_jss.pdf` | Compiled manuscript (31 pages, two-column). |
| `build.sh` | `pdflatex → bibtex → pdflatex ×2`. |
| `highlights.txt` | Five highlights (≤ 85 characters), for separate upload; not typeset in the PDF. |
| `COVER_LETTER_JSS.md` | Cover letter, with bracketed items to confirm. |
| `VITAE.md` | Template for the author biographies the guide requires (at most 100 words each). |
| `SUBMISSION_CHECKLIST_JSS.md` | Requirement-by-requirement check, replication-package contents, and author actions. |
| `SINGLE_STAGE_BASELINE.md` | Whether the single-stage point is closed (it is not yet), what the record contains, and the matched-control protocol and commands. |
| `analysis/v18_additions.py` → `data/v18_additions.json` | Reproduces every number new in v1.8 (Tables 7–9; row and character budgets; base re-evaluation drift) from the workbook and shards. Run: `GLAUKOPIS_ROOT=/path/to/Glaukopis python3 analysis/v18_additions.py` (needs `openpyxl`). |
| `figures/` | `Figure_1`/`Figure_2` as vector PDF/EPS plus their TikZ sources, copied unchanged from v1.7 (the v1.7 `Fig1`/`Fig2` files, renamed to the guide's naming convention). The manuscript compiles the same figures inline. |
| `cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Elsevier CAS bundle files the build needs. |
| `Guide for authors - Journal of Systems and Software ….pdf` | Elsevier's guide, downloaded 2026-10-25; left uncommitted because it is Elsevier's page and the repository is public. |
| Other template files | Elsevier's elsarticle/CAS download, as you placed it: templates, samples, `doc/`, `figs/`, `thumbnails/`, the other `.bst` and `.cls` files. Not part of the submission. |

## What changed from v1.7

**Venue and format.**

- Elsevier CAS two-column class, author–year references.
- Elsevier declarations: CRediT via `\printcredits`, competing interest, funding, ethics, data availability, and the generative-AI declaration in Elsevier's wording.
- Highlights moved to a separate file.

**Software-engineering framing.**

- New title.
- New abstract (246 words).
- Introduction with five research questions (RQ1–RQ5).
- New §4.1 on engineering ML systems and their data:
  - SE4AI: Amershi et al. 2019, Sculley et al. 2015, Martínez-Fernández et al. 2022, Paleyes et al. 2022
  - data validation and documentation: Breck et al. 2019, datasheets, model cards
  - leakage: Kapoor & Narayanan 2023
  - LLM threats in SE research: Sallou et al. 2024
- New Table 5, the data-quality checks: what each detects, what it misses, and two historical silent failures that led to gates.
- New §10 with answers to the RQs and seven lessons learned.
- Threats to validity reorganized into construct, internal, external, conclusion and reliability (Wohlin et al. 2012).

**Staging question.**

- §4.3 now cites the forgetting literature (McCloskey & Cohen 1989; Kirkpatrick et al. 2017; Luo et al. 2023) and the controlled mixed-vs-sequential SFT study of Dong et al. 2024.
- §6.3 is rewritten:
  - Table 7 compares the configuration of the 32B single-stage run, the chain, and the matched control.
  - Table 8 is the v7 result table with its caption corrected: one stage of three epochs, not "one pass".
  - Table 9 is new: the 14B single-pass and phased releases.
  - A paragraph defines the matched control.
- v1.7's claim that the single-stage run had a "comparable update budget" is replaced by the exact exposure figures (2.07× CTI rows, 1.76× CTI text, almost no general data).

**Small corrections.**

- The row-count gate is per task axis, not per template.
- The conference-draft note in §7.6 is reworded for readers who never saw the draft.
- The baseline list says "single-stage runs" (plural) and that none holds the data fixed.

**Changes from the JSS Guide for Authors (read 2026-10-25).**

- Cardei's ORCID added.
- Legal entity is Athena Software Group Inc. throughout: affiliation, declarations, software reference.
- FAU postal address added.
- Generative-AI declaration retitled to the guide's wording.
- Funding statement now uses the guide's form with a sponsor-role sentence.
- Appendix tables numbered per appendix (Table B.1).
- American spelling outside verbatim records.
- CRediT role names match the guide.
- Figure files renamed `Figure_N`.
- Added a vitae template and a length-justification paragraph in the cover letter (the paper is 31 two-column pages against the encouraged 18).

No result number from v1.7 changed. The new numbers are those produced by `analysis/v18_additions.py`.

## Markers to resolve before submitting

`grep -n 'TODO (authors)\|MATCHED-CONTROL\|STAGING-CLAIM' glaukopis_jss.tex`

## Related repository changes (uncommitted)

- `SFT/autotrain/run_sft_qwen25_32b_v21_single_stage.sh`: matched-control launcher. Dry run checked.
- `SFT/eval/utils/serve_and_bench_qwen25_32b_v21_single_stage.sh`: full-suite bench wrapper, including MMLU-Pro.
- `SFT/eval/pipelines/models.py`: two vLLM aliases for the control and its learning-rate sensitivity run.

## Build

`bash build.sh` with TeX Live 2024. For Texifier, see the note at the end of `SUBMISSION_CHECKLIST_JSS.md`.
