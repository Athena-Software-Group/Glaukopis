# Glaukopis v1.4 author-review package

This package was created from the supplied abridged v1.3 and long-form v1.1 papers, ICTAI reviews, and read-only public-repository inspection. It does **not** alter `/Users/pietro/code/Glaukopis` or the GitHub repository.

## Contents

- `glaukopis_abridged_v1.4.tex`: revised nine-page journal-neutral manuscript.
- `glaukopis_abridged_v1.4.pdf`: compiled, visually inspected PDF.
- `glaukopis_v1.4_verified.bib`: corrected bibliographic subset with verified publication versions where available.
- `glaukopis_abridged_v1.4.bbl`: generated bibliography for reproducible compilation.
- `REVISION_MEMORANDUM.md`: detailed responses to all reviewers, new numerical/code-audit findings, unresolved evidence gates.
- `JOURNAL_SHORTLIST.md`: journal ranking, metrics, verified review/fee data, FAU requirement and approval wording.
- `SOURCE_MANIFEST.json`: input hashes and repository paths/version used for this revision.
- `data/`: transcribed source numbers, derived balanced averages and CTI-only ledger arithmetic, plus unreconciled legacy columns.
- `tools/recompute_revision_tables.py`: dependency-free arithmetic reproduction; no training or inference.
- `build.sh`: local LaTeX build script.

## Status before submission

The manuscript is deliberately marked **author-review draft**. The new 65.0 vs 57.7 balanced mean is a recomputation from existing rounded task scores, not a new evaluation. Individual experimental scores were preserved. The legacy secondary averages and hosted-reference attribution conventions need exact workbook/item-level reconciliation. Cost records are presented as expenditure ledgers because cumulative counters and multi-summary aggregation do not prove one matched pass per model. Missing controls, repeated seeds, and independent held-out incidents remain missing; they have not been replaced with fabricated results.

The manuscript uses standard `IEEEtran` in journal mode and requires no external figures. It is not the official IEEE Access class and lacks the required biographies. Convert to the selected journal's template only after choosing the venue and closing the scientific audit gates. Keep appropriate AI-assistance disclosure and confirm all author/submission declarations.

## Build

A TeX installation with IEEEtran, xurl, microtype, listings, and standard AMS packages is required.

```bash
cd /path/to/Glaukopis_v1_4
bash build.sh
```

The script uses `bibtex` when present, otherwise `bibtex8`, and does not require internet access. Alternatively, the included `.bbl` permits PDFLaTeX compilation without regenerating the bibliography.

Recompute the arithmetic separately:

```bash
python3 tools/recompute_revision_tables.py
```

That command rewrites only this package's `data/` files. It does not inspect or modify the original project.

## Minimal local evidence to close the audit

Use a read-only export of the exact results workbook with formulas, the v21 task-score exports, relevant per-run summaries and prediction files, API counter snapshots/run manifests, per-pass training timing/configuration logs, and the actual decontamination reports. Do not include credentials, API keys, `.env` files, unrelated customer telemetry, private database credentials, model weights, or the entire development directory merely for convenience. Existing artifacts may resolve much of the reporting audit before any new GPU runs are needed.
