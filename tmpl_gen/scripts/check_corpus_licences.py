#!/usr/bin/env python3
"""Build-time licence-allowlist gate for the v13 training corpus
(tmpl_gen/templates/05072026/v13_plan.txt §4.5 and §10).

Reads an Alpaca-format SFT JSON corpus and asserts that every row's
`source` tag is in the §10.1 commercial-use allowlist. Halts the build
non-zero if any row carries a denylisted or unknown tag, so a non-
permissive source cannot silently land in a published v13 checkpoint.

Why this exists:
  v12 had no row-level licence audit. v13 is the first vintage authored
  under an explicit commercial-use guardrail (HF redistribution + down-
  stream packaging without inheriting NC/SA clauses). The Source: parser
  directive (v13_plan.txt §5.5) populates the row-level tag; this script
  is the build-time enforcement.

Usage:
  python tmpl_gen/scripts/check_corpus_licences.py \\
      --input  SFT/data/ift_data_2026_05_07_v13.json \\
      --report _v13_build/licence_gate_report.json

  # Dev builds with Alpaca CC-BY-NC mixed in:
  python tmpl_gen/scripts/check_corpus_licences.py \\
      --input  ... --report ... --allow-nc
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# v13_plan.txt §10.1: permissive sources approved for commercial v13 builds.
ALLOWED_SOURCES: set[str] = {
    "mitre-attack-custom",      # MITRE ATT&CK Terms of Use
    "mitre-d3fend-custom",      # MITRE D3FEND Terms of Use
    "misp-galaxy-cc0",          # CC-0 1.0 / public domain
    "athena-cti-db-internal",   # this repo's curated knowledge tables
    "tulu-3-odc-by",            # AllenAI Tulu-3, ODC-BY 1.0 (handled at trainer mix)
    "nist-iso-concept",         # NIST CSF / ISO 27001 concept paraphrases (CM.*)
}

# v13_plan.txt §10.1: only allowed in dev builds via --allow-nc.
NC_SOURCES: set[str] = {
    "alpaca-cc-by-nc-4",
}

# v13_plan.txt §10.2: explicit denylist; surfaced separately for clarity.
DENIED_SOURCES: set[str] = {
    "crowdstrike-proprietary",
    "mandiant-proprietary",
    "thaicert-cc-by-nc-sa",
    "eternal-liberty-cc-by-sa",
    "recorded-future-proprietary",
    "intel471-proprietary",
}


def classify(tag: str, allow_nc: bool) -> str:
    if tag in ALLOWED_SOURCES:
        return "allowed"
    if tag in NC_SOURCES:
        return "allowed-nc" if allow_nc else "denied-nc"
    if tag in DENIED_SOURCES:
        return "denied"
    return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--report", type=Path,
                   help="Optional JSON path for the per-tag report.")
    p.add_argument("--allow-nc", action="store_true",
                   help="Treat alpaca-cc-by-nc-4 as allowed (dev builds only; "
                        "v13 production builds MUST omit this flag).")
    p.add_argument("--key-field", default="source")
    p.add_argument("--max-examples", type=int, default=5,
                   help="Number of denied/unknown row examples to surface "
                        "in the report per offending tag (default: 5).")
    args = p.parse_args()

    rows = json.loads(args.input.read_text())
    print(f"loaded {len(rows):,} rows from {args.input}", file=sys.stderr)

    tag_counts: Counter[str] = Counter()
    missing = 0
    for r in rows:
        tag = r.get(args.key_field)
        if not tag:
            missing += 1
            tag_counts["<MISSING>"] += 1
        else:
            tag_counts[tag] += 1

    print(f"distinct source tags: {len(tag_counts):,}\n", file=sys.stderr)

    by_status: dict[str, list[str]] = {
        "allowed": [], "allowed-nc": [], "denied-nc": [],
        "denied": [], "unknown": [],
    }
    for tag in tag_counts:
        status = "unknown" if tag == "<MISSING>" else classify(tag, args.allow_nc)
        by_status[status].append(tag)

    examples: dict[str, list[dict]] = {}
    if args.max_examples > 0:
        for tag in by_status["denied"] + by_status["denied-nc"] + by_status["unknown"]:
            ex: list[dict] = []
            for r in rows:
                rt = r.get(args.key_field) or "<MISSING>"
                if rt == tag:
                    ex.append({k: r.get(k, "") for k in
                               ("shortname", "instruction", "input", "output")})
                if len(ex) >= args.max_examples:
                    break
            examples[tag] = ex

    fail_tags = by_status["denied"] + by_status["denied-nc"] + by_status["unknown"]
    fail_rows = sum(tag_counts[t] for t in fail_tags)

    for tag, n in tag_counts.most_common():
        status = "unknown" if tag == "<MISSING>" else classify(tag, args.allow_nc)
        marker = "  " if status in ("allowed", "allowed-nc") else "!!"
        print(f"{marker} {tag:<32s} {n:>8,d}  [{status}]", file=sys.stderr)

    report = {
        "input": str(args.input),
        "total_rows": len(rows),
        "missing_source_field": missing,
        "allow_nc": args.allow_nc,
        "tag_counts": dict(tag_counts),
        "by_status": by_status,
        "examples": examples,
        "fail_tag_count": len(fail_tags),
        "fail_row_count": fail_rows,
        "outcome": "fail" if fail_tags else "ok",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.report}", file=sys.stderr)

    if fail_tags:
        print(f"\nFAIL: {len(fail_tags)} non-allowlist source tag(s) "
              f"covering {fail_rows:,} rows:", file=sys.stderr)
        for t in fail_tags:
            status = "unknown" if t == "<MISSING>" else classify(t, args.allow_nc)
            print(f"  - {t}: {tag_counts[t]:,} rows ({status})", file=sys.stderr)
        return 1
    print("\nlicence gate PASSED.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
