#!/usr/bin/env python3
"""Split the v12+ stratified-shuffled corpus into per-phase training shards.

v12 mode (default; v12_plan.txt §6.3): three disjoint shards.

  --out-broad             everything NOT in Phase B or Phase C
                          (broad knowledge + SOC + CM + AthenaBench
                          families not consumed by B/C)
  --out-rms-ate-vsp-rcm   AB.RMS.* + JS.RMS.* + AB.ATE + JS.ATE +
                          AB.VSP.* + V.CPE + AB.RCM + JS.RCM +
                          X.VW.* + YN.VW.*
  --out-taa-canon         TAA.CANON.*

v13 mode (--two-phase; v13_plan.txt §6.3): two shards. v13 drops v12's
Phase C; TAA.CANON folds into Phase A (broad_plus_canon), and SOC.*
appears in BOTH shards (intersection: SOC sees two epochs of supervision,
which v9's SOC retention shape proved is not over-training and which
v12's SOC regression to 39.3 proved was needed).

  --out-broad-plus-canon  everything except the axis shard, PLUS SOC.*
                          (broad + TAA.* + TAA.CANON.* + MISP.CANON.* +
                          CM.* + AB.MCQ.* + AB.MCQ.EXT.* + AB.MS.* + SOC.*)
  --out-axis              AB.RMS.* + JS.RMS.* + AB.ATE + JS.ATE +
                          AB.VSP.* + V.CPE + AB.RCM + JS.RCM + X.VW + YN.VW
                          + SOC.* (intersected from broad_plus_canon)

Val rows are excluded from all output shards (identified by exact
instruction+input+output triple match against the val slice).

Splits preserve input row order (stratified-shuffle order inherited
per-shard so LLaMA-Factory's per-epoch shuffle operates on already-
stratified shards).

Usage (v12 three-phase):
  python tmpl_gen/scripts/split_corpus_for_phases.py \\
      --input  SFT/data/ift_data_2026_05_05_v12.shuffled.json \\
      --val    SFT/data/ift_data_2026_05_05_v12_val.json \\
      --out-broad           SFT/data/ift_data_2026_05_05_v12_broad.json \\
      --out-rms-ate-vsp-rcm SFT/data/ift_data_2026_05_05_v12_rms_ate_vsp_rcm.json \\
      --out-taa-canon       SFT/data/ift_data_2026_05_05_v12_taa_canon.json \\
      --report _v12_build/phase_split_report.json

Usage (v13 two-phase):
  python tmpl_gen/scripts/split_corpus_for_phases.py \\
      --input  SFT/data/ift_data_2026_05_07_v13.shuffled.json \\
      --val    SFT/data/ift_data_2026_05_07_v13_val.json \\
      --two-phase \\
      --out-broad-plus-canon SFT/data/ift_data_2026_05_07_v13_broad_plus_canon.json \\
      --out-axis             SFT/data/ift_data_2026_05_07_v13_axis.json \\
      --report _v13_build/phase_split_report.json
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

# v13 §6.3: AXIS shard = v12 Phase B + SOC.* (SOC sees two epochs).
V13_AXIS_PATTERNS: list[str] = PHASE_B_PATTERNS + ["SOC.*"]
# v13 §6.3: SOC.* lives in BOTH shards. Used to also include SOC.*
# rows in the broad_plus_canon shard even though they match the axis
# pattern (intentional duplication; not a routing error).
V13_DUAL_SHARD_PATTERNS: list[str] = ["SOC.*"]


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
    p.add_argument("--two-phase", action="store_true",
                   help="v13 mode: emit two shards (--out-broad-plus-canon "
                        "+ --out-axis) instead of v12's three. Drops Phase C "
                        "and folds TAA.CANON.* / MISP.CANON.* into the "
                        "broad shard; SOC.* lives in BOTH shards.")
    # v12 three-shard outputs (default mode).
    p.add_argument("--out-broad", type=Path)
    p.add_argument("--out-rms-ate-vsp-rcm", type=Path)
    p.add_argument("--out-taa-canon", type=Path)
    # v13 two-shard outputs (--two-phase mode).
    p.add_argument("--out-broad-plus-canon", type=Path)
    p.add_argument("--out-axis", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--key-field", default="shortname")
    args = p.parse_args()

    if args.two_phase:
        if not (args.out_broad_plus_canon and args.out_axis):
            p.error("--two-phase requires --out-broad-plus-canon and --out-axis")
    else:
        missing = [n for n, v in [("--out-broad", args.out_broad),
                                  ("--out-rms-ate-vsp-rcm", args.out_rms_ate_vsp_rcm),
                                  ("--out-taa-canon", args.out_taa_canon)] if not v]
        if missing:
            p.error("default (three-shard) mode requires " + ", ".join(missing))

    rows = json.loads(args.input.read_text())
    val_rows = json.loads(args.val.read_text())
    val_keys = {row_key(r) for r in val_rows}
    print(f"loaded {len(rows):,} rows from {args.input}", file=sys.stderr)
    print(f"loaded {len(val_rows):,} val rows from {args.val}", file=sys.stderr)

    if args.two_phase:
        return _emit_two_phase(args, rows, val_keys)
    return _emit_three_phase(args, rows, val_keys)


def _hist(rows: list[dict], key_field: str) -> dict:
    return dict(Counter(r.get(key_field, "") for r in rows).most_common(50))


def _emit_three_phase(args, rows: list[dict], val_keys: set) -> int:
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
        report = {
            "mode": "three-phase",
            "input": str(args.input),
            "val": str(args.val),
            "input_rows": len(rows),
            "val_overlap_excluded": val_overlap,
            "phase_a_broad_rows": len(broad),
            "phase_b_rms_ate_vsp_rcm_rows": len(phase_b),
            "phase_c_taa_canon_rows": len(phase_c),
            "phase_a_top50_shortnames": _hist(broad, args.key_field),
            "phase_b_top50_shortnames": _hist(phase_b, args.key_field),
            "phase_c_top50_shortnames": _hist(phase_c, args.key_field),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.report}", file=sys.stderr)

    return 0


def _emit_two_phase(args, rows: list[dict], val_keys: set) -> int:
    broad_plus_canon: list[dict] = []
    axis: list[dict] = []
    val_overlap = 0
    soc_dual = 0
    for r in rows:
        if row_key(r) in val_keys:
            val_overlap += 1
            continue
        sn = r.get(args.key_field, "")
        in_axis = matches(sn, V13_AXIS_PATTERNS)
        in_dual = matches(sn, V13_DUAL_SHARD_PATTERNS)
        if in_axis:
            axis.append(r)
        if (not in_axis) or in_dual:
            broad_plus_canon.append(r)
        if in_dual:
            soc_dual += 1

    print(f"\nval rows excluded (exact i/i/o match): {val_overlap:,}",
          file=sys.stderr)
    print(f"broad_plus_canon (Phase A): {len(broad_plus_canon):,}", file=sys.stderr)
    print(f"axis (Phase B):             {len(axis):,}", file=sys.stderr)
    print(f"  SOC.* dual-shard rows:    {soc_dual:,} (in BOTH shards)",
          file=sys.stderr)

    for path, shard in [(args.out_broad_plus_canon, broad_plus_canon),
                        (args.out_axis, axis)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(shard, indent=2))
        print(f"wrote {len(shard):,} rows to {path}", file=sys.stderr)

    if args.report:
        report = {
            "mode": "two-phase",
            "input": str(args.input),
            "val": str(args.val),
            "input_rows": len(rows),
            "val_overlap_excluded": val_overlap,
            "broad_plus_canon_rows": len(broad_plus_canon),
            "axis_rows": len(axis),
            "soc_dual_shard_rows": soc_dual,
            "broad_plus_canon_top50_shortnames": _hist(broad_plus_canon, args.key_field),
            "axis_top50_shortnames": _hist(axis, args.key_field),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.report}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
