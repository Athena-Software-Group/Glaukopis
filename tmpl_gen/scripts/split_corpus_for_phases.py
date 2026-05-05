#!/usr/bin/env python3
"""Split the v12 stratified-shuffled corpus into per-phase training shards
(tmpl_gen/templates/05052026/v12_plan.txt §6.3).

Reads the post-shuffle clean corpus and the held-out val slice, then
writes three disjoint Alpaca-format JSON files:

  --out-broad             everything NOT in Phase B or Phase C
                          (broad knowledge + SOC + CM + AthenaBench
                          families not consumed by B/C)
  --out-rms-ate-vsp-rcm   AB.RMS.* + JS.RMS.* + AB.ATE + JS.ATE +
                          AB.VSP.* + V.CPE + AB.RCM + JS.RCM +
                          X.VW.* + YN.VW.*
  --out-taa-canon         TAA.CANON.*

Val rows are excluded from all three shards (identified by exact
instruction+input+output triple match against the val slice).

Splits preserve the input row order (stratified-shuffle order is
inherited per-shard so LLaMA-Factory's per-epoch shuffle operates on
already-stratified shards).

Usage:
  python tmpl_gen/scripts/split_corpus_for_phases.py \\
      --input  SFT/data/ift_data_2026_05_05_v12.shuffled.json \\
      --val    SFT/data/ift_data_2026_05_05_v12_val.json \\
      --out-broad           SFT/data/ift_data_2026_05_05_v12_broad.json \\
      --out-rms-ate-vsp-rcm SFT/data/ift_data_2026_05_05_v12_rms_ate_vsp_rcm.json \\
      --out-taa-canon       SFT/data/ift_data_2026_05_05_v12_taa_canon.json \\
      --report _v12_build/phase_split_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PHASE_B_PATTERNS: list[str] = [
    "AB.RMS.*", "JS.RMS.*",
    "AB.ATE", "AB.ATE.*", "JS.ATE", "JS.ATE.*",
    "AB.VSP.*", "V.CPE", "V.CPE.*",
    "AB.RCM", "AB.RCM.*", "JS.RCM", "JS.RCM.*",
    "X.VW.*", "YN.VW.*",
]
PHASE_C_PATTERNS: list[str] = ["TAA.CANON.*"]


def matches(shortname: str, patterns: list[str]) -> bool:
    for p in patterns:
        if p.endswith("*"):
            if shortname.startswith(p[:-1]):
                return True
        elif shortname == p:
            return True
    return False


def row_key(r: dict) -> tuple[str, str, str]:
    return (r.get("instruction", "") or "",
            r.get("input", "") or "",
            r.get("output", "") or "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--val", required=True, type=Path)
    p.add_argument("--out-broad", required=True, type=Path)
    p.add_argument("--out-rms-ate-vsp-rcm", required=True, type=Path)
    p.add_argument("--out-taa-canon", required=True, type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--key-field", default="shortname")
    args = p.parse_args()

    rows = json.loads(args.input.read_text())
    val_rows = json.loads(args.val.read_text())
    val_keys = {row_key(r) for r in val_rows}
    print(f"loaded {len(rows):,} rows from {args.input}", file=sys.stderr)
    print(f"loaded {len(val_rows):,} val rows from {args.val}", file=sys.stderr)

    broad: list[dict] = []
    phase_b: list[dict] = []
    phase_c: list[dict] = []
    val_overlap = 0
    for r in rows:
        if row_key(r) in val_keys:
            val_overlap += 1
            continue
        sn = r.get(args.key_field, "")
        if matches(sn, PHASE_C_PATTERNS):
            phase_c.append(r)
        elif matches(sn, PHASE_B_PATTERNS):
            phase_b.append(r)
        else:
            broad.append(r)

    total_out = len(broad) + len(phase_b) + len(phase_c)
    assert total_out + val_overlap == len(rows), \
        f"row accounting mismatch: out={total_out} val_skip={val_overlap} in={len(rows)}"

    print(f"\nval rows excluded (exact i/i/o match): {val_overlap:,}",
          file=sys.stderr)
    print(f"phase A (broad): {len(broad):,}", file=sys.stderr)
    print(f"phase B (rms+ate+vsp+rcm): {len(phase_b):,}", file=sys.stderr)
    print(f"phase C (taa_canon): {len(phase_c):,}", file=sys.stderr)

    for path, shard in [(args.out_broad, broad),
                        (args.out_rms_ate_vsp_rcm, phase_b),
                        (args.out_taa_canon, phase_c)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(shard, indent=2))
        print(f"wrote {len(shard):,} rows to {path}", file=sys.stderr)

    if args.report:
        def hist(rows: list[dict]) -> dict:
            return dict(Counter(r.get(args.key_field, "") for r in rows)
                        .most_common(50))
        report = {
            "input": str(args.input),
            "val": str(args.val),
            "input_rows": len(rows),
            "val_overlap_excluded": val_overlap,
            "phase_a_broad_rows": len(broad),
            "phase_b_rms_ate_vsp_rcm_rows": len(phase_b),
            "phase_c_taa_canon_rows": len(phase_c),
            "phase_a_top50_shortnames": hist(broad),
            "phase_b_top50_shortnames": hist(phase_b),
            "phase_c_top50_shortnames": hist(phase_c),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.report}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
