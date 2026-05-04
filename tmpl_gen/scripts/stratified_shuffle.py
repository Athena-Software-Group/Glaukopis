#!/usr/bin/env python3
"""Stratified pre-shuffle for v11+ training corpora (Option A from
tmpl_gen/templates/05032026/v11_plan.txt §6.3).

Reads an Alpaca-format SFT JSON corpus, groups rows by their `shortname`
field, shuffles within each shortname (seeded), then emits the rows in a
deterministic order computed by stride-based interleaving so that every
shortname is distributed uniformly across the whole corpus.

Why this exists:
  v10's RMS regression is partly attributable to LLaMA-Factory's default
  uniform-random shuffle: with ~12K RMS rows in a 200K-row corpus, large
  contiguous windows could contain very few RMS examples, starving the
  M-control reasoning consolidation. Stride interleaving guarantees that
  in any window of ~total_rows/family_size consecutive positions, each
  shortname is represented once.

Algorithm (Bresenham-style stride scheduling):
  - For each shortname f with k_f rows in a corpus of N total rows,
    define stride_f = N / k_f.
  - Maintain a min-heap keyed on `next_emit_pos` per shortname (initial
    next_emit_pos = stride_f / 2 so first emits are spread across the
    start, not all bunched at position 0).
  - Pop the lowest next_emit_pos, append the next row from that shortname,
    push back with next_emit_pos += stride_f. Tie-break by stable counter.

The output preserves all input rows (no drops, no duplicates) and is
deterministic given --seed. Out is a flat JSON consumed by the standard
training launcher; no LLaMA-Factory changes required.

Usage:
  python tmpl_gen/scripts/stratified_shuffle.py \\
      --input  SFT/data/ift_data_2026_05_03_v11.json \\
      --output SFT/data/ift_data_2026_05_03_v11.shuffled.json \\
      --seed 42 --validate
"""

import argparse
import heapq
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def family_key(row: dict, key_field: str) -> str:
    val = row.get(key_field)
    if isinstance(val, str) and val:
        return val
    return "_unknown_"


def stride_interleave(family_rows: dict[str, list[dict]]) -> list[dict]:
    total = sum(len(v) for v in family_rows.values())
    counter = itertools.count()
    heap: list[tuple[float, int, str]] = []
    cursors: dict[str, int] = {}
    strides: dict[str, float] = {}

    for fam, rows in family_rows.items():
        if not rows:
            continue
        strides[fam] = total / len(rows)
        cursors[fam] = 0
        heapq.heappush(heap, (strides[fam] / 2.0, next(counter), fam))

    out: list[dict] = []
    while heap:
        pos, _, fam = heapq.heappop(heap)
        out.append(family_rows[fam][cursors[fam]])
        cursors[fam] += 1
        if cursors[fam] < len(family_rows[fam]):
            heapq.heappush(heap, (pos + strides[fam], next(counter), fam))
    return out


def validate_distribution(rows: list[dict], key_field: str,
                          window: int) -> None:
    fam_total = Counter(family_key(r, key_field) for r in rows)
    n = len(rows)
    print(f"\nvalidation (window={window}):", file=sys.stderr)
    print(f"  total rows: {n:,}  distinct {key_field}: {len(fam_total):,}",
          file=sys.stderr)
    print(f"\ntop-15 {key_field} by row count (target per-window count):",
          file=sys.stderr)
    for fam, k in fam_total.most_common(15):
        target = (k * window) / n
        print(f"  {k:6d}  target/window={target:6.2f}  {fam}",
              file=sys.stderr)

    if n < window * 2:
        return

    # Spot-check three windows: start, middle, end.
    for label, start in [("start", 0),
                         ("mid", (n // 2) - (window // 2)),
                         ("end", n - window)]:
        win = rows[start:start + window]
        seen = Counter(family_key(r, key_field) for r in win)
        # Report the top-3 most over-represented in this window.
        deviations = []
        for fam, k in fam_total.items():
            target = (k * window) / n
            actual = seen.get(fam, 0)
            if target >= 0.5:  # only flag families with non-trivial expected count
                deviations.append((abs(actual - target) / max(target, 1.0), fam, actual, target))
        deviations.sort(reverse=True)
        worst = deviations[:3]
        print(f"\n  {label} window [{start:,}:{start + window:,}] — "
              f"distinct families={len(seen)}; worst-3 dev:", file=sys.stderr)
        for ratio, fam, a, t in worst:
            print(f"    {a:4d} (target {t:5.1f})  rel_err={ratio:.2f}  {fam}",
                  file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--key-field", default="shortname",
                   help="Row field used as stratification key (default: shortname).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for within-family shuffle (default: 42).")
    p.add_argument("--validate", action="store_true",
                   help="Print per-window family distribution sanity check.")
    p.add_argument("--validate-window", type=int, default=512,
                   help="Window size for --validate (default: 512).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = json.loads(args.input.read_text())
    print(f"loaded {len(rows):,} rows from {args.input}", file=sys.stderr)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[family_key(r, args.key_field)].append(r)

    print(f"grouped into {len(grouped):,} distinct {args.key_field} values",
          file=sys.stderr)
    if "_unknown_" in grouped:
        print(f"WARN: {len(grouped['_unknown_']):,} rows had no `{args.key_field}` field",
              file=sys.stderr)

    for fam in grouped:
        rng.shuffle(grouped[fam])

    out_rows = stride_interleave(grouped)
    assert len(out_rows) == len(rows), \
        f"row-count mismatch: in={len(rows)} out={len(out_rows)}"

    if args.validate:
        validate_distribution(out_rows, args.key_field, args.validate_window)

    args.output.write_text(json.dumps(out_rows, indent=2))
    print(f"\nwrote {len(out_rows):,} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
