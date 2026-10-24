# Computers & Security submission checklist (v1.5)

Journal facts checked 2026-09-06 (Elsevier / ScienceDirect author pages and policy pages; the ScienceDirect guide-for-authors page itself blocked automated fetching, so re-read it once in a browser before uploading):

- Peer review: single-anonymized; at least two reviewers. Author-identified manuscript is correct; no anonymization needed.
- Template: `elsarticle` (preprint, 12pt, `authoryear`). Reference style: name-and-year (`elsarticle-harv`). Line numbers are on (`lineno`) for reviewer convenience.
- Highlights: 3–5 bullets, max 85 characters each (five bullets in `highlights.txt`, longest 79 characters). Uploaded as a separate file, not typeset into the manuscript PDF.
- Abstract: kept under 250 words. Keywords: 6 (maximum 6).
- Required end sections present: CRediT authorship statement, Declaration of competing interest, Funding, Data availability, Declaration of generative AI and AI-assisted technologies in the writing process (Elsevier wording, placed before the references).
- Graphical abstract: optional; not provided.

## Author actions before upload

1. Confirm the affiliation details for RevolveAI (city, country) and every author's ORCID (`\affiliation[rai]` and `\cortext` in the .tex).
2. Confirm the CRediT roles written for S. Ahmad and M. M. Khan (marked `TODO` in the .tex).
3. Confirm the competing-interest text (AthenaBench co-authorship; Athena Security Group funding and affiliations).
4. Deposit the supplementary package in a persistent archive (Zenodo or similar) and insert the DOI in the Data availability section:
   - `SFT/test/responses/athena-cti-sft-qwen25-32b-v21-{core,taa,cse,recalibrate,recal-32b}-vllm/` (per-item `_scored.jsonl`, CSVs, and `summary_*` manifests) — currently untracked and local only;
   - `_v21_{core,taa,cse}_build/{dedup_report,watcher_status,row_count_gate_report}.json`;
   - this folder's `analysis/` scripts and `data/` CSVs.
5. Decide whether to keep the base-model cost row (ledger only) or drop it; the text already flags it.
6. Check the Elsevier AI-declaration wording against the current policy page at upload time (wording can change).
7. Cover letter: state that an abridged version was submitted to and rejected by ICTAI 2026, that the ICTAI reviews motivated the added audits, and that the manuscript is not under consideration elsewhere. Disclose that a longer internal manuscript version exists and has not been published.
8. Consider inviting the committee chair to confirm that a *Computers & Security* acceptance satisfies the FAU CS PhD publication expectation before submission (the 2026-09-05 memo has the wording).

## Optional strengthening if time permits before submission (each requires compute)

- Three seeded Stage-4 runs (3–4 h each on 8×H100) evaluated under a frozen rule — replaces the "single run" caveat for the recipe-selection claim.
- Core→CSE control (skip Stage 2) — tests whether the attribution stage is needed.
- Re-run the 13-gram screen with CyberSOCEval fetched and TSV files included — replaces the coverage caveat.
- Evaluate the base model with per-item retention so that base-to-final paired intervals can be reported.
