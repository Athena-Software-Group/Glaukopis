#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified subsampler for Alpaca-format SFT corpora.

Builds a compact training file by drawing a per-shortname cap from each
input. Used to derive the v8-small Llama-3.1-8B corpus from
ift_data_2026_04_26_combined_v7.json without saturating the 8B
parameter count and inducing catastrophic forgetting.

Algorithm:
  - Group rows by `shortname` (rows missing the field are bucketed under "_").
  - Per source, draw min(per_shortname_cap, len(bucket)) rows uniformly at
    random with a fixed seed (deterministic across runs).
  - Concatenate the per-source outputs in the order they were supplied.

Per-source caps default to None (= keep all rows) so callers can mix
"draw N per shortname from corpus A" with "include all of corpus B".
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load_alpaca(path: Path) -> list[dict]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected list of objects, got {type(data).__name__}")
    return data


def stratified_sample(rows: list[dict], cap: int, seed: int, label: str) -> list[dict]:
    """Per-shortname cap. cap<=0 means no subsampling (return rows as-is)."""
    if cap is None or cap <= 0:
        return list(rows)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r.get("shortname") or "_"].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    kept_per_bucket: dict[str, int] = {}
    for sn, bucket in buckets.items():
        if len(bucket) <= cap:
            sampled = bucket
        else:
            sampled = rng.sample(bucket, cap)
        out.extend(sampled)
        kept_per_bucket[sn] = len(sampled)
    print(f"  [{label}] {len(rows):,} rows / {len(buckets):,} shortnames -> "
          f"{len(out):,} rows after cap={cap}")
    return out


def random_sample(rows: list[dict], n: int, seed: int, label: str) -> list[dict]:
    """Uniform random subsample without per-shortname stratification."""
    if n is None or n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    out = rng.sample(rows, n)
    print(f"  [{label}] {len(rows):,} rows -> {len(out):,} rows after random sample")
    return out


def parse_source(spec: str) -> tuple[Path, int, str]:
    """Parse `path[:cap[:mode]]` where mode in {strat, random}. Default cap=0 (all)."""
    parts = spec.split(":")
    path = Path(parts[0])
    cap = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    mode = parts[2] if len(parts) > 2 else "strat"
    if mode not in ("strat", "random"):
        raise ValueError(f"unknown mode {mode!r} in {spec!r}; expected strat|random")
    return path, cap, mode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", "-s", action="append", required=True,
                    metavar="PATH[:CAP[:MODE]]",
                    help="Repeatable. CAP=0 keeps all rows; MODE=strat (per-shortname cap) "
                         "or random (uniform). Examples: "
                         "SFT/data/x.json:300:strat, SFT/data/y.json:2000:random, "
                         "SFT/data/z.json (keep all).")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output Alpaca JSON file (list of {instruction,input,output,shortname?}).")
    ap.add_argument("--seed", type=int, default=20260429,
                    help="RNG seed for reproducibility (default: 20260429).")
    ap.add_argument("--shuffle-output", action="store_true",
                    help="Shuffle the concatenated output once before writing. Off by "
                         "default to keep source ordering visible in diffs.")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for spec in args.source:
        path, cap, mode = parse_source(spec)
        if not path.is_file():
            print(f"[FAIL] source not found: {path}", file=sys.stderr)
            sys.exit(2)
        label = path.stem
        rows = load_alpaca(path)
        if mode == "strat":
            sampled = stratified_sample(rows, cap, args.seed, label)
        else:
            sampled = random_sample(rows, cap, args.seed, label)
        all_rows.extend(sampled)

    if args.shuffle_output:
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(all_rows, f, indent=4)
    print(f"\nWrote {len(all_rows):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
