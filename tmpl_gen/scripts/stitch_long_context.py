#!/usr/bin/env python3
"""Stitch existing Alpaca rows into long-context pseudo threat reports.

Builds the v8 long-context corpus by concatenating N (5-15) existing rows'
``output`` fields into a single multi-section "report body", then appending
one of the source rows' ``input`` as the question to be answered. The result
is an Alpaca row whose ``input`` is in the 8K-16K-token range that the
Phase B v8 training pass operates over.

Selection heuristic: rows are sampled until the running character total is
within --target-chars-min/max; the question target is the row whose output
contributes the largest single block (so the answer remains anchored to a
named span of the report body).

Usage:
  python stitch_long_context.py \\
      --input  SFT/data/ift_data_2026_04_24_v5.json \\
      --output SFT/data/ift_data_2026_04_29_longctx_v8.json \\
      --count 4000 --target-chars-min 24000 --target-chars-max 56000

Run with --self-test to execute inline unit tests instead.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

REPORT_INSTRUCTION = (
    "You are a senior CTI analyst at an enterprise Security Operations "
    "Center reading a long threat intelligence report. The report is "
    "composed of multiple sections describing related MITRE ATT&CK "
    "techniques, mitigations, weaknesses, vulnerabilities, threat groups, "
    "and exploit context. Read the entire report carefully, then answer "
    "the single question at the end of the report."
)
REPORT_HEADER = "THREAT INTELLIGENCE REPORT\n" + ("=" * 60)
QUESTION_HEADER = "\n\nQUESTION\n" + ("-" * 60) + "\n"


def stitch(rows: list[dict], n_min: int, n_max: int,
           cmin: int, cmax: int, rng: random.Random,
           max_attempts: int = 24) -> dict | None:
    """Sample 5-15 rows whose total output chars is in [cmin, cmax]."""
    for _ in range(max_attempts):
        n = rng.randint(n_min, n_max)
        sample = rng.sample(rows, k=min(n, len(rows)))
        total = sum(len(r.get("output", "")) for r in sample)
        if cmin <= total <= cmax:
            return _assemble(sample, rng)
    sample = rng.sample(rows, k=min(n_max, len(rows)))
    return _assemble(sample, rng)


def _assemble(sample: list[dict], rng: random.Random) -> dict:
    target_idx = max(range(len(sample)),
                     key=lambda i: len(sample[i].get("output", "")))
    sections = []
    for i, r in enumerate(sample, 1):
        sn = r.get("shortname", "section")
        body = (r.get("output") or "").strip()
        sections.append(f"\nSECTION {i} ({sn})\n{'-' * 60}\n{body}")
    target = sample[target_idx]
    body = REPORT_HEADER + "".join(sections) + QUESTION_HEADER + (target.get("input") or "").strip()
    return {
        "instruction": REPORT_INSTRUCTION + " " + (target.get("instruction") or "").strip(),
        "input": body,
        "output": (target.get("output") or "").strip(),
        "shortname": "LC." + (target.get("shortname") or "stitched"),
    }


def filter_eligible(rows: Iterable[dict], min_out_chars: int = 200) -> list[dict]:
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not r.get("input") or not r.get("output"):
            continue
        if len(r["output"]) < min_out_chars:
            continue
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=False, help="Source Alpaca JSON.")
    p.add_argument("--output", "-o", required=False, help="Destination Alpaca JSON.")
    p.add_argument("--count", "-c", type=int, default=4000,
                   help="Number of stitched rows to emit.")
    p.add_argument("--n-stitch-min", type=int, default=5)
    p.add_argument("--n-stitch-max", type=int, default=15)
    p.add_argument("--target-chars-min", type=int, default=24000,
                   help="Lower bound on stitched input character length (~6K tokens).")
    p.add_argument("--target-chars-max", type=int, default=56000,
                   help="Upper bound on stitched input character length (~14K tokens).")
    p.add_argument("--min-source-chars", type=int, default=200,
                   help="Drop source rows shorter than this from the pool.")
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return _self_test()

    if not args.input or not args.output:
        p.error("--input and --output are required (unless --self-test)")

    src = json.loads(Path(args.input).read_text())
    pool = filter_eligible(src, min_out_chars=args.min_source_chars)
    if len(pool) < args.n_stitch_max:
        sys.exit(f"Not enough eligible rows: {len(pool)} < {args.n_stitch_max}")

    rng = random.Random(args.seed)
    out = []
    for _ in range(args.count):
        row = stitch(pool, args.n_stitch_min, args.n_stitch_max,
                     args.target_chars_min, args.target_chars_max, rng)
        if row is not None:
            out.append(row)

    Path(args.output).write_text(json.dumps(out, indent=2))
    chars = [len(r["input"]) for r in out]
    print(f"wrote {len(out)} rows to {args.output} "
          f"input chars: min={min(chars)} mean={sum(chars)//len(chars)} max={max(chars)}")


def _self_test():
    rng = random.Random(0)
    pool = [{"instruction": "Q", "input": f"q{i}", "output": "x" * 5000,
             "shortname": f"T.{i}"} for i in range(20)]
    row = stitch(pool, 5, 8, 20000, 40000, rng)
    assert row is not None and "SECTION 1" in row["input"]
    assert 20000 <= len(row["input"]) <= 60000, len(row["input"])
    assert row["shortname"].startswith("LC."), row["shortname"]
    print("self-test ok")


if __name__ == "__main__":
    main()
