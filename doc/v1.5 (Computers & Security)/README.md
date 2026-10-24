# Glaukopis v1.5 — Computers & Security submission package

Created 2026-09-06 from `doc/v1.4 (abridged)/` after re-verifying every number against the repository's primary artifacts. Supersedes v1.4 for journal submission.

## Contents

- `glaukopis_cs_v1.5.tex` / `.pdf` — manuscript in `elsarticle` (preprint, author-year), 31 pages with line numbers.
- `highlights.txt` — the five Highlights, uploaded to Elsevier as a separate file. The `highlights` environment in the .tex is commented out on purpose: enabling it makes elsarticle 3.x prepend an extra unnumbered Highlights page to the manuscript PDF.
- `glaukopis_cs_v1.5.bib` — bibliography with verified publication records.
- `build.sh` — `pdflatex` → `bibtex` → `pdflatex` ×2.
- Typesetting note: the preamble carries a guarded compatibility block for `elsarticle` < 3.0, and pins `pgfplots` to `compat=1.15`. TexpadTeX/Texifier bundles elsarticle 1.2.0 (2009) and pgfplots 1.15 (2017): the former lacks `\affiliation` and the `highlights` environment, so without the block their arguments were typeset as raw text on a spurious first page and the affiliation labels did not resolve; the latter rejects `compat=1.18`. TeX Live 2024 (elsarticle 3.3, pgfplots 1.18.1) accepts both settings, and Figure 2 renders pixel-identically under `compat=1.15` and `compat=1.18`. Both toolchains now build 31 pages with the title page first.
- `REVISION_MEMORANDUM_v1.5.md` — what was verified, what changed, status of every open item from the 2026-09-05 memo, reviewer-concern map, repository fixes recommended to the authors.
- `SUBMISSION_CHECKLIST.md` — journal requirements and author actions before upload.
- `analysis/` — scripts that produce every new number in the paper (no inference):
  - `bootstrap_v21_32b.py` — paired item-level bootstrap over the five local 32B checkpoints (`bootstrap_results.json`).
  - `entity_overlap.py` — entity and gold-fact overlap between training shards and AthenaBench items (`entity_overlap.json`).
  - `overlap_subsets.py` — checkpoint scores on overlapping vs non-overlapping items (`overlap_subsets.json`).
  - `make_tables.py` — regenerates the aggregate, cost and build-report tables from the workbook, ledger, run manifests and build reports into `data/`.
- `data/` — CSV outputs of `make_tables.py`.

## Inputs the scripts read (read-only)

- `SFT/eval/Glaukopis Results.xlsx` (cached values; formulas audited separately)
- `SFT/eval/responses/cost_summary.csv`
- `SFT/test/responses/athena-cti-sft-qwen25-32b-v21-*-vllm/` (local, untracked per-item artifacts and run manifests)
- `SFT/eval/benchmark_data/athena_bench/` and `SFT/data/ift_data_2026_05_18_v21_{core,taa,cse}.json`
- `_v21_{core,taa,cse}_build/` build reports

Run with the Anaconda base interpreter (`/opt/anaconda3/bin/python`), which has `openpyxl`.
