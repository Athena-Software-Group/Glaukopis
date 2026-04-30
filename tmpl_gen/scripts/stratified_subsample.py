#!/usr/bin/env python
"""Stratified per-template subsample of an Alpaca SFT JSON.

Preserves the small AB.RMS.* / JS.RMS.* catalog drills in full (they are scarce
and high-value for catalog recall).  All other templates are capped at --cap
rows.  Sampling is deterministic given --seed.
"""
import argparse, json, random
from collections import defaultdict

# Templates we never subsample (full retention to protect catalog recall).
PRESERVE_FULL_PREFIXES = (
    "AB.RMS.",   # ATT&CK / RMS catalog drills
    "JS.RMS.",
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--cap", type=int, default=200,
                    help="max rows per template (templates with fewer rows kept in full)")
    ap.add_argument("--seed", type=int, default=20260430)
    args = ap.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)
    print(f"in:  {args.in_path}: {len(data)} rows")

    by_t = defaultdict(list)
    for r in data:
        sn = r.get("shortname") or r.get("template") or r.get("source") or "?"
        by_t[sn].append(r)

    rng = random.Random(args.seed)
    out = []
    kept_per_t = {}
    for sn, rows in by_t.items():
        preserve = any(sn.startswith(p) for p in PRESERVE_FULL_PREFIXES)
        n = len(rows)
        if preserve or n <= args.cap:
            keep = rows
        else:
            keep = rng.sample(rows, args.cap)
        kept_per_t[sn] = (n, len(keep), preserve)
        out.extend(keep)

    rng.shuffle(out)
    with open(args.out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"out: {args.out_path}: {len(out)} rows  (cap={args.cap})")

    # Top-line family stats
    fam = defaultdict(int)
    for sn, (n_in, n_out, _) in kept_per_t.items():
        fam[sn.split('.')[0]] += n_out
    print("\nrows by family prefix (after subsample):")
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"  {k:6s} {v:6d}")

    # Templates fully preserved
    preserved = [(sn, n_in) for sn, (n_in, n_out, p) in kept_per_t.items() if p]
    print(f"\nfully preserved templates (n={len(preserved)}):")
    for sn, n_in in sorted(preserved):
        print(f"  {sn:24s} {n_in:5d}")

if __name__ == "__main__":
    main()
