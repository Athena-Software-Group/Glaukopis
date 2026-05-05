#!/usr/bin/env python3
"""TAA actor-balance post-pass for the v10+ training corpus.

Audits actor distribution across the actor-attribution TAA shortnames
(AB.TAA.{1-5} and JS.TAA.{1-3}) and:
  - drops rows over the per-actor cap (mode-collapse guard),
  - rejects the build if fewer than --min-actors distinct intrusion sets
    are represented (under-coverage guard).

Out of scope (left untouched): AB.TAA.IE.*, JS.TAA.IE.*, AB.TAA.NEG.*,
JS.TAA.NEG.* — those intentionally rely on synthetic / paired actors and
should not be capped or counted toward the floor.

Usage:
  python tmpl_gen/scripts/taa_actor_balance.py \\
      --input  SFT/data/ift_data_2026_05_01_v10.raw.json \\
      --output SFT/data/ift_data_2026_05_01_v10.json \\
      --max-per-actor 8 --min-actors 150
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Shortnames whose output line names a single attributed actor; cap + floor
# checks apply to these.
ATTRIB_AB = {"AB.TAA.1", "AB.TAA.2", "AB.TAA.3", "AB.TAA.4", "AB.TAA.5"}
ATTRIB_JS = {"JS.TAA.1", "JS.TAA.2", "JS.TAA.3"}
ATTRIB_ALL = ATTRIB_AB | ATTRIB_JS

# Out-of-scope: synthetic / paired actor families. Do not touch.
OUT_OF_SCOPE = re.compile(r"^(AB|JS)\.TAA\.(IE|NEG)\.\d+$")

# AB extractor: "Therefore, the adversary is <NAME>." (final sentence)
RE_AB_ACTOR = re.compile(r"Therefore,\s+the\s+adversary\s+is\s+(.+?)\.?\s*$",
                         re.MULTILINE)

# JS extractor: "actor": "<NAME>"  (first occurrence in the JSON output)
RE_JS_ACTOR = re.compile(r'"actor"\s*:\s*"([^"]+)"')


def extract_actor(row: dict) -> str | None:
    sn = row.get("shortname", "")
    out = row.get("output", "") or ""
    if sn in ATTRIB_AB:
        m = RE_AB_ACTOR.search(out)
        return m.group(1).strip() if m else None
    if sn in ATTRIB_JS:
        m = RE_JS_ACTOR.search(out)
        if not m:
            return None
        actor = m.group(1).strip()
        return None if actor == "INSUFFICIENT_EVIDENCE" else actor
    return None


def audit(rows: list[dict]) -> tuple[dict[str, list[int]], list[int], list[int]]:
    """Return (actor -> [row indices], in_scope_indices, untouched_indices)."""
    actor_to_idx: dict[str, list[int]] = defaultdict(list)
    in_scope, untouched = [], []
    for i, r in enumerate(rows):
        sn = r.get("shortname", "")
        if sn in ATTRIB_ALL:
            actor = extract_actor(r)
            if actor is None:
                # In-scope shortname but couldn't parse actor — keep it,
                # surface as a warning so the manifest author can investigate.
                untouched.append(i)
                continue
            actor_to_idx[actor].append(i)
            in_scope.append(i)
        else:
            untouched.append(i)
    return actor_to_idx, in_scope, untouched


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-per-actor", type=int, default=8)
    p.add_argument("--min-actors", type=int, default=150)
    p.add_argument("--max-rows-per-family-total", type=int, default=0,
                   help="Hard cap on total in-scope attribution rows AFTER the "
                        "per-actor cap. 0 disables (default). v12: set 3500 to "
                        "match plan §3.1 budget and prevent the v11 ACTOR_CAP=40 "
                        "overshoot (8,560 actual vs 3,500 planned).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Audit + histogram only; do not write output.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = json.loads(args.input.read_text())
    print(f"loaded {len(rows):,} rows from {args.input}", file=sys.stderr)

    # Quick sanity: count out-of-scope TAA so the operator can see they are
    # honoured separately.
    oos_count = sum(1 for r in rows if OUT_OF_SCOPE.match(r.get("shortname", "")))
    print(f"out-of-scope TAA rows (IE/NEG, untouched): {oos_count:,}",
          file=sys.stderr)

    actor_to_idx, in_scope, untouched = audit(rows)
    distinct = len(actor_to_idx)
    in_scope_n = sum(len(v) for v in actor_to_idx.values())
    print(f"in-scope actor-attribution rows: {in_scope_n:,} across "
          f"{distinct:,} distinct actors", file=sys.stderr)

    # Histogram (top 20 + tail summary)
    hist = Counter({a: len(v) for a, v in actor_to_idx.items()})
    print("\ntop-20 actors by row count:", file=sys.stderr)
    for actor, n in hist.most_common(20):
        print(f"  {n:5d}  {actor}", file=sys.stderr)
    long_tail = sum(1 for n in hist.values() if n <= args.max_per_actor)
    over_cap = sum(1 for n in hist.values() if n > args.max_per_actor)
    print(f"\nactors at-or-below cap ({args.max_per_actor}): {long_tail:,}",
          file=sys.stderr)
    print(f"actors over cap: {over_cap:,}", file=sys.stderr)

    # Floor check
    if distinct < args.min_actors:
        print(f"\nFAIL: {distinct} distinct actors < min_actors={args.min_actors}. "
              f"Aborting; rebuild the corpus with more TAA Count: or new templates.",
              file=sys.stderr)
        sys.exit(2)

    # Cap: randomly sample max-per-actor rows per actor, drop the rest.
    keep_idx: set[int] = set(untouched)
    dropped = 0
    for actor, idxs in actor_to_idx.items():
        if len(idxs) <= args.max_per_actor:
            keep_idx.update(idxs)
        else:
            sampled = rng.sample(idxs, args.max_per_actor)
            keep_idx.update(sampled)
            dropped += len(idxs) - args.max_per_actor

    print(f"\ndropped {dropped:,} over-cap rows; "
          f"keeping {len(keep_idx):,} of {len(rows):,} total",
          file=sys.stderr)

    # Second-pass family-total cap (v12 §4.3). Applies only to the in-scope
    # attribution rows; untouched rows (broad knowledge, IE/NEG, etc.) are
    # not counted against the cap. Sampling is uniform across actors after
    # the per-actor cap, so the cap doesn't bias toward any single actor.
    if args.max_rows_per_family_total > 0:
        in_scope_keep = sorted(i for i in keep_idx if i in set(in_scope))
        if len(in_scope_keep) > args.max_rows_per_family_total:
            sampled = set(rng.sample(in_scope_keep,
                                     args.max_rows_per_family_total))
            removed = len(in_scope_keep) - args.max_rows_per_family_total
            keep_idx = (set(keep_idx) - set(in_scope_keep)) | sampled
            print(f"family-total cap: {len(in_scope_keep):,} > "
                  f"{args.max_rows_per_family_total:,}; "
                  f"sampled down to {args.max_rows_per_family_total:,} "
                  f"(removed {removed:,})", file=sys.stderr)
        else:
            print(f"family-total cap: {len(in_scope_keep):,} <= "
                  f"{args.max_rows_per_family_total:,}; no further sampling",
                  file=sys.stderr)

    if args.dry_run:
        print("dry-run: not writing output.", file=sys.stderr)
        return

    out_rows = [rows[i] for i in sorted(keep_idx)]
    args.output.write_text(json.dumps(out_rows, indent=2))
    print(f"wrote {len(out_rows):,} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
