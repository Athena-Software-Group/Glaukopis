#!/usr/bin/env python3
"""Build the v19-TAA held-out validation slice (chained TAA-Classic refresher
shard for v19; ported from _v18_taa_build/build_val_slice.py with paths
re-pointed at the v19 TAA shard).

Reads the clean v19-TAA corpus (SFT/data/ift_data_2026_05_15_v19_taa.json),
samples N rows per AthenaBench axis, and writes:
  - SFT/data/ift_data_2026_05_15_v19_taa_val.json   (~50 rows per axis)
  - SFT/data/ift_data_2026_05_15_v19_taa.json       (corpus minus val)

The v19-TAA shard is generated from tmpl_gen/templates/05152026/Sophia-CTI-
Templates-v19_taa.txt (byte-identical to v16.txt; TAA Classic only, CANON
purged) and chained on top of the v19-core base in
run_sft_qwen25_14b_v19_plus_taa.sh.

Sampling is deterministic given --seed.
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
    (re.compile(r"^(?:AB|JS)\.TAA\.IE\."),               "TAA-IE"),
    (re.compile(r"^(?:AB|JS)\.TAA\.NEG\."),              "TAA-NEG"),
    (re.compile(r"^(?:AB|JS)\.TAA(?:\.[1-9])?$"),        "TAA"),
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
                   default=Path("SFT/data/ift_data_2026_05_15_v19_taa.shuffled.json"))
    p.add_argument("--val-out", type=Path,
                   default=Path("SFT/data/ift_data_2026_05_15_v19_taa_val.json"))
    p.add_argument("--train-out", type=Path,
                   default=Path("SFT/data/ift_data_2026_05_15_v19_taa.json"))
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
