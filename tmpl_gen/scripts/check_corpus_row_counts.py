#!/usr/bin/env python3
"""Build-time row-count gate for the v12+ training corpus
(tmpl_gen/templates/05052026/v12_plan.txt §4.1).

Reads an Alpaca-format SFT JSON corpus and a per-axis plan JSON, counts
rows in each axis's shortname patterns, and exits non-zero if any axis
is below its `reject_if_below` value (or, optionally, above
`max_allowed`).

Why this exists:
  v11's build silently undershot ATE (48% of plan), VSP (41%), RCM
  (24%), MCQ (22%), and shipped 0 rows for the new CM.* family while
  AB.TAA.* overshot 245%. None of these were detectable until the v11
  bench sweep returned regressed numbers. This script makes the same
  failure loud at build time.

Usage:
  python tmpl_gen/scripts/check_corpus_row_counts.py \\
      --input  SFT/data/ift_data_2026_05_05_v12.json \\
      --plan   tmpl_gen/templates/05052026/v12_row_count_gate.json \\
      --report _v12_build/row_count_gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def matches(shortname: str, patterns: list[str]) -> bool:
    for p in patterns:
        if p.endswith("*"):
            if shortname.startswith(p[:-1]):
                return True
        elif shortname == p:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--report", type=Path,
                   help="Optional JSON path for the per-axis report.")
    p.add_argument("--key-field", default="shortname")
    args = p.parse_args()

    rows = json.loads(args.input.read_text())
    plan = json.loads(args.plan.read_text())

    sn_counts = Counter(r.get(args.key_field, "") for r in rows)
    print(f"loaded {len(rows):,} rows from {args.input}", file=sys.stderr)
    print(f"distinct {args.key_field}s: {len(sn_counts):,}\n", file=sys.stderr)

    failures: list[dict] = []
    report_axes: list[dict] = []
    for axis in plan["axes"]:
        name = axis["axis"]
        patterns = axis["shortname_patterns"]
        target = axis.get("target", 0)
        floor = axis.get("reject_if_below", 0)
        ceil = axis.get("max_allowed")
        actual = sum(n for sn, n in sn_counts.items() if matches(sn, patterns))
        matched_sns = sorted(sn for sn in sn_counts if matches(sn, patterns))
        status = "OK"
        if actual < floor:
            status = "FAIL_LOW"
        elif ceil is not None and actual > ceil:
            status = "FAIL_HIGH"
        entry = {
            "axis": name,
            "target": target,
            "actual": actual,
            "reject_if_below": floor,
            "max_allowed": ceil,
            "status": status,
            "matched_shortnames": matched_sns,
            "patterns": patterns,
        }
        report_axes.append(entry)
        marker = "  " if status == "OK" else "!!"
        print(f"{marker} {name:<18s} actual={actual:>6,d}  "
              f"target={target:>6,d}  floor={floor:>6,d}"
              + (f"  ceil={ceil:>6,d}" if ceil is not None else "")
              + f"  [{status}]",
              file=sys.stderr)
        if status != "OK":
            failures.append(entry)

    report = {
        "input": str(args.input),
        "plan": str(args.plan),
        "total_rows": len(rows),
        "axes": report_axes,
        "failures": failures,
        "outcome": "fail" if failures else "ok",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.report}", file=sys.stderr)

    if failures:
        print(f"\nFAIL: {len(failures)} axis/axes did not pass the gate:",
              file=sys.stderr)
        for f in failures:
            print(f"  - {f['axis']}: {f['actual']:,} ({f['status']})",
                  file=sys.stderr)
        return 1
    print("\ngate PASSED.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
