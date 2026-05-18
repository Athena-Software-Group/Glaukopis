#!/usr/bin/env python3
"""Build the v18p1-core held-out validation slice (v18_1_plan.txt §6;
ported verbatim from _v18_build/build_val_slice.py with v18p1 paths).

Reads the clean v18p1-core corpus
(SFT/data/ift_data_2026_05_18_v21_core.json), samples N rows per
AthenaBench axis, and writes:
  - SFT/data/ift_data_2026_05_18_v21_core_val.json   (~50 rows per axis)
  - SFT/data/ift_data_2026_05_18_v21_core_train.json (corpus minus val)

Axis derivation from the row-level `shortname` field:
  AB.RMS.* / JS.RMS.*               -> RMS
  AB.MCQ.* / JS.MCQ.*               -> MCQ
  AB.ATE.{1..8} / JS.ATE.{1..3}     -> ATE
  AB.RCM.{1..4} / JS.RCM.{1,2}      -> RCM
  AB.VSP.* / V.CPE                  -> VSP
  AB.TAA.{1..5} / JS.TAA.{1..3}     -> TAA
  AB.TAA.IE.* / JS.TAA.IE.*         -> TAA-IE
  AB.TAA.NEG.* / JS.TAA.NEG.*       -> TAA-NEG
  AB.MS.* / JS.MS.*                 -> MS
  SOC.*                             -> SOC
  CM.*                              -> CM
  (TAA.CANON.* dropped per the v18 chain pivot; carried into v18p1)

Other shortnames (broad-knowledge X.*, YN.*, M.*, A.*, etc.) are not
sampled -- they are not directly axis-aligned to AthenaBench eval slices.

Sampling is deterministic given --seed. Output JSON files use indent=2
to match the v11/v12/v13/v18 build pipeline convention. Reads from the
row-count-gated CLEAN corpus BEFORE the stratified shuffle so the
per-axis sampling is deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


AXIS_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^SOC\."),                              "SOC"),
    (re.compile(r"^CM\."),                               "CM"),
    (re.compile(r"^(?:AB|JS)\.TAA\.IE\."),               "TAA-IE"),
    (re.compile(r"^(?:AB|JS)\.TAA\.NEG\."),              "TAA-NEG"),
    (re.compile(r"^(?:AB|JS)\.TAA(?:\.[1-9])?$"),        "TAA"),
    (re.compile(r"^(?:AB|JS)\.MS\."),                    "MS"),
    (re.compile(r"^(?:AB|JS)\.RMS\."),                   "RMS"),
    (re.compile(r"^(?:AB|JS)\.MCQ\."),                   "MCQ"),
    (re.compile(r"^(?:AB|JS)\.ATE(?:\.|$)"),             "ATE"),
    (re.compile(r"^(?:AB|JS)\.RCM(?:\.|$)"),             "RCM"),
    (re.compile(r"^(?:AB|JS)\.VSP\."),                   "VSP"),
    (re.compile(r"^V\.CPE(?:\.|$)"),                     "VSP"),
]


def axis_for(shortname: str) -> str | None:
    for pat, axis in AXIS_RULES:
        if pat.match(shortname):
            return axis
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("SFT/data/ift_data_2026_05_18_v21_core.json"))
    p.add_argument("--val-out", type=Path,
                   default=Path("SFT/data/ift_data_2026_05_18_v21_core_val.json"))
    p.add_argument("--train-out", type=Path,
                   default=Path("SFT/data/ift_data_2026_05_18_v21_core_train.json"))
    p.add_argument("--per-axis", type=int, default=50,
                   help="rows to sample per AthenaBench axis (default: 50)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = json.loads(args.input.read_text())
    print(f"[load] {len(rows):,} rows from {args.input}", file=sys.stderr)

    by_axis: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        ax = axis_for(r.get("shortname", ""))
        if ax is not None:
            by_axis[ax].append(i)

    print(f"[axes] {len(by_axis)} AthenaBench-aligned axes detected:",
          file=sys.stderr)
    for ax in sorted(by_axis):
        print(f"  {ax:<10s}  {len(by_axis[ax]):>6,d} candidate rows",
              file=sys.stderr)

    rng = random.Random(args.seed)
    val_idx: set[int] = set()
    for ax in sorted(by_axis):
        cands = by_axis[ax]
        n = min(args.per_axis, len(cands))
        picks = rng.sample(cands, n)
        val_idx.update(picks)
        if n < args.per_axis:
            print(f"[warn] axis {ax}: only {n} rows available "
                  f"(< requested {args.per_axis})", file=sys.stderr)

    val_rows = [rows[i] for i in sorted(val_idx)]
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]

    assert len(val_rows) + len(train_rows) == len(rows), \
        f"row-count mismatch: val={len(val_rows)} train={len(train_rows)} total={len(rows)}"

    args.val_out.write_text(json.dumps(val_rows, indent=2))
    args.train_out.write_text(json.dumps(train_rows, indent=2))
    print(f"[write] val   {len(val_rows):>6,d} -> {args.val_out}",
          file=sys.stderr)
    print(f"[write] train {len(train_rows):>6,d} -> {args.train_out}",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
