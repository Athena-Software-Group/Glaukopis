# International Journal of Information Security (Springer) — submission package

Created 2026-10-24 by porting `doc/v1.6/glaukopis_v1.6.tex` (the post–Computers & Security revision)
to Springer's svjour3 class in the journal's required two-column format with numbered references,
then expanded (v1.7) with the background, Sophia, research-directions and appendix material from the
longer Oxford-format drafts, corrected against the audited facts. `CHANGELOG_v1.7.md` lists the additions.
See `SUBMISSION_CHECKLIST_IJIS.md` for the requirement-by-requirement check and the remaining author actions,
and `COVER_LETTER_IJIS.md` for the cover letter.

## Files to upload

- `glaukopis_ijis.tex`, `glaukopis_ijis.bib`, `glaukopis_ijis.bbl` — manuscript source and bibliography
- `svjour3.cls`, `svglov3.clo`, `spmpsci.bst` — class, class options and Springer's numbered bibliography style
- `Fig1.eps` / `Fig1.pdf`, `Fig2.eps` / `Fig2.pdf` — the two figures as separate vector files (also embedded as TikZ in the .tex)
- `glaukopis_ijis.pdf` — compiled manuscript (29 pages, two-column)

## Build

`bash build.sh` (pdflatex → bibtex → pdflatex ×2). The three style files above must stay next to the .tex.
`Fig1_src.tex` / `Fig2_src.tex` regenerate the standalone figure files (`pdflatex FigN_src.tex`, then `pdftops -eps`).

## Folder housekeeping

`template/` holds the files that came with the downloaded svjour3/EPJC package (templates, user guide,
`spphys.bst`, `svepjc3.clo`) and is not part of the submission. `figures/` holds the project's historical
figure assets and is not referenced by this manuscript.

## Relation to the Oxford drafts

The Oxford-format drafts (`doc/Oxford/glaukopis_oup-v1.10.tex` and earlier) were the source of the v1.7
additions. Not everything in them was carried over: their statements that the contamination screen
covered CyberSOCEval and CTIBench, that Stage 4 recovers VSP, and that the four-stage chain is "the
load-bearing element of the recipe" are contradicted by the audited artifacts and were not reproduced;
their bibliography entries for several knowledge-graph and benchmark papers had placeholder authors and
were replaced with verified records.
