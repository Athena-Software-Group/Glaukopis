AAIML 2027 DOUBLE-BLIND SUBMISSION PACKAGE
===========================================

Main file:
  aaiml2027_blind_submission.tex

Included dependencies/artifacts:
  aaiml2027_blind_submission.bbl  - generated IEEE bibliography; included so BibTeX is NOT required for a normal compile
  aaiml2027_blind_submission.bib  - cited-only bibliography source, included if you want to regenerate the BBL
  IEEEtran.cls                    - conference class supplied with the venue/template
  IEEEtran.bst                    - bibliography style used to generate the BBL
  figures/pipeline_greek_blind.pdf - blind pipeline figure created for this conference version
  aaiml2027_blind_submission.pdf  - verified compiled output

Recommended local compile (uses the included .bbl):
  pdflatex aaiml2027_blind_submission.tex
  pdflatex aaiml2027_blind_submission.tex

If you edit bibliography entries and need to regenerate the .bbl:
  pdflatex aaiml2027_blind_submission.tex
  bibtex aaiml2027_blind_submission
  pdflatex aaiml2027_blind_submission.tex
  pdflatex aaiml2027_blind_submission.tex

The manuscript intentionally references:
  figures/pipeline_greek_blind.pdf
That figure IS part of this package and must stay under the figures/ subdirectory.

Double-blind notes:
- Authors and affiliations are anonymous.
- Benchmark A and the source repository are intentionally anonymized in the submitted bibliography.
- Ariadne and Sophia component names are intentionally retained.
- Do not substitute the long-form journal BBL into this review package because it reveals author/project identities.

Verified output:
- 10 pages total, including references.
- Page numbers enabled.
- PDF metadata Author = Anonymous.
