# Glaukopis v1.6 — revision after the Computers & Security rejection (2026-10-24)

Created from `doc/v1.5 (Computers & Security)/` in response to reviews COSE-D-26-05507.
Supersedes v1.5 for journal submission. Target venue to be set in `\journal{}` (see
`JOURNAL_RECOMMENDATION.md`).

## Contents

- `glaukopis_v1.6.tex` / `glaukopis_v1.6.bib` — manuscript (elsarticle, author-year). Same
  preamble and compatibility shim as v1.5. Five `%% TODO (authors)` comments mark items only
  the authors can settle (entity name, ORCIDs, CRediT roles, graph counts, data DOI).
- `glaukopis_v1.6_PREVIEW_generic_class.pdf` — content preview built with a generic article
  class because the build environment lacked `elsarticle.cls`. Layout, float numbering and
  bibliography style differ from the real class; the text, tables and figures are final.
  Rebuild with `bash build.sh` on a machine with elsarticle (Texifier / TeX Live) for the
  submission PDF.
- `RESPONSE_TO_REVIEWERS_v1.6.md` — every review comment mapped to the change made, the
  v1.5→v1.6 section map, provenance of every new number, the experiments that still need
  compute (in priority order), and the pre-upload author checklist.
- `JOURNAL_RECOMMENDATION.md` — venue analysis and the top-three recommendation.
- `highlights.txt` — five highlights (≤85 characters each), for venues that use them.
- `analysis/v16_additions.py` — reproduces every number new in v1.6 from existing artifacts
  (no training or inference). Run with `GLAUKOPIS_ROOT=/path/to/Glaukopis python3 analysis/v16_additions.py`;
  writes `data/v16_additions.json`. Needs `openpyxl` for the workbook-derived tables.
- `data/v16_additions.json` — its output.
- The v1.5 `analysis/` scripts (bootstrap, entity overlap, overlap subsets, make_tables) are
  unchanged and still produce the tables carried over from v1.5; copy them alongside when
  depositing the supplementary package.

## Inputs the new script reads (read-only)

- `SFT/data/ift_data_2026_05_18_v21_{taa,cse}.json`
- `SFT/eval/benchmark_data/athena_bench/athena_taa_canonical/athena-cti-taa-canonical.jsonl`
- `SFT/test/responses/athena-cti-sft-qwen25-32b-v21-*-vllm/athena-vsp/*_scored.jsonl` (local, untracked)
- `SFT/eval/Glaukopis Results.xlsx` (sheets `MMLU Pro`, `Glaukopis (TEST)`, `Cost`)
