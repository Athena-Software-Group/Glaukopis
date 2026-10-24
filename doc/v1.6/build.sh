#!/usr/bin/env bash
# Build the v1.6 manuscript (pdflatex + bibtex). Requires the elsarticle class
# (TeX Live: texlive-publishers) or, for a Springer/IEEE venue, the ported front matter.
set -euo pipefail
cd -- "$(dirname -- "$0")"
main=glaukopis_v1.6
command -v pdflatex >/dev/null || { echo 'pdflatex is required.' >&2; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
bibtex "$main" >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$main.tex" | tail -n 3
grep -E "Warning: (Citation|Reference)" "$main.log" || true
echo "Built: $PWD/$main.pdf"
