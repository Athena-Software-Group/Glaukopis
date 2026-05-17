#!/usr/bin/env python3
"""End-to-end validation of the v19 SFT shards (v17.1-pattern chain).

Checks the seven shards LlamaFactory will actually load across the three
chained stages -- v20-core (core_a_kb_mcq_taa_soc_cm_ms_yn /
core_b_rms_ate_vsp_rcm / core_val), v19-plus-taa (taa / taa_val),
v19 (cse / cse_val) -- for:
JSON well-formedness, required-field presence, schema consistency,
empty-string sanity, per-stage train/val disjointness, MCQ letter
balance, ATE family lift vs v12, source-tag allowlist, and duplicate-row
counts. Emits a single PASS/FAIL summary.
"""

from __future__ import annotations
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

DATA = Path("SFT/data")
SHARDS = {
    "core_a_kb_mcq_taa_soc_cm_ms_yn": DATA / "ift_data_2026_05_16_v20_core_a_kb_mcq_taa_soc_cm_ms_yn.json",
    "core_b_rms_ate_vsp_rcm":   DATA / "ift_data_2026_05_16_v20_core_b_rms_ate_vsp_rcm.json",
    "core_val":                 DATA / "ift_data_2026_05_16_v20_core_val.json",
    "taa":                      DATA / "ift_data_2026_05_16_v20_taa.json",
    "taa_val":                  DATA / "ift_data_2026_05_16_v20_taa_val.json",
    "cse":                      DATA / "ift_data_2026_05_16_v20_cse.json",
    "cse_val":                  DATA / "ift_data_2026_05_16_v20_cse_val.json",
}
DISJOINT_PAIRS = (
    ("core_a_kb_mcq_taa_soc_cm_ms_yn", "core_val"),
    ("core_b_rms_ate_vsp_rcm",         "core_val"),
    ("taa",                    "taa_val"),
    ("cse",                    "cse_val"),
)
REQUIRED_FIELDS = ("instruction", "input", "output", "shortname", "source")
ALLOWED_SOURCES = {"athena-cti-db-internal"}

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL: {msg}")


def row_hash(r: dict) -> str:
    return hashlib.sha256(
        f"{r.get('instruction','')}||{r.get('input','')}||{r.get('output','')}".encode()
    ).hexdigest()


def main() -> int:
    print("=== v19 corpus validation ===\n")
    loaded: dict[str, list[dict]] = {}

    # 1. JSON well-formedness + required fields
    for name, p in SHARDS.items():
        print(f"[{name}] {p}")
        if not p.exists():
            fail(f"{name}: shard missing")
            continue
        try:
            rows = json.loads(p.read_text())
        except Exception as e:
            fail(f"{name}: JSON parse error: {e}")
            continue
        if not isinstance(rows, list):
            fail(f"{name}: top-level is not a list")
            continue
        loaded[name] = rows
        print(f"  rows: {len(rows):,}  size: {p.stat().st_size/1e6:.1f} MB")

        missing = collections.Counter()
        empty_in = empty_out = 0
        for r in rows:
            for f in REQUIRED_FIELDS:
                if f not in r or r.get(f) is None:
                    missing[f] += 1
            if not str(r.get("input", "")).strip():
                empty_in += 1
            if not str(r.get("output", "")).strip():
                empty_out += 1
        for f, n in missing.items():
            fail(f"{name}: {n:,} rows missing field '{f}'")
        if empty_out:
            fail(f"{name}: {empty_out:,} rows have empty output")
        print(f"  empty input: {empty_in:,}  empty output: {empty_out:,}")

        # source allowlist
        srcs = collections.Counter(r.get("source") for r in rows)
        bad = {s: n for s, n in srcs.items() if s not in ALLOWED_SOURCES}
        if bad:
            fail(f"{name}: non-allowlisted source tags: {bad}")
        print(f"  distinct source tags: {len(srcs)}  -> {dict(srcs)}")

        # internal duplicates
        hashes = collections.Counter(row_hash(r) for r in rows)
        dups = sum(1 for h, c in hashes.items() if c > 1)
        if dups:
            print(f"  WARN: {dups:,} duplicate row groups inside {name}")

    # 2. each train shard disjoint from its paired val (exact-match);
    #    pairs are per-stage in the v17.1 chained pattern.
    for train_name, val_name in DISJOINT_PAIRS:
        if train_name not in loaded or val_name not in loaded:
            continue
        val_hashes = {row_hash(r) for r in loaded[val_name]}
        leaks = sum(1 for r in loaded[train_name] if row_hash(r) in val_hashes)
        if leaks:
            fail(f"{train_name}: {leaks:,} {val_name} rows leaked into training shard")
        else:
            print(f"  [disjoint] {train_name} ∩ {val_name} = 0 rows  OK")

    # 3. MCQ letter balance (core Phase A KB+MCQ+TAA+SOC+CM+MS+YN shard)
    pa_key = "core_a_kb_mcq_taa_soc_cm_ms_yn"
    if pa_key in loaded:
        rx = re.compile(r"\bTherefore,\s*([A-H])\.")
        letters = collections.Counter()
        mcq = [r for r in loaded[pa_key]
               if r["shortname"].startswith(("AB.MCQ", "JS.MCQ"))]
        for r in mcq:
            m = rx.search(r.get("output", ""))
            if m:
                letters[m.group(1)] += 1
        total = sum(letters.values()) or 1
        print(f"\n[MCQ] {len(mcq):,} rows in {pa_key}; {total:,} parsed Answer letters")
        for lt in sorted(letters):
            pct = 100 * letters[lt] / total
            mark = "  " if 17.0 <= pct <= 23.0 else "!!"
            if not (17.0 <= pct <= 23.0):
                fail(f"MCQ letter {lt} balance out-of-band: {pct:.1f}%")
            print(f"  {mark}{lt}: {letters[lt]:>5,d} ({pct:.1f}%)")

    # 4. ATE family count in core Phase B axis shard (lift vs v12 ~10,500)
    if "core_b_rms_ate_vsp_rcm" in loaded:
        ate = [r for r in loaded["core_b_rms_ate_vsp_rcm"]
               if r["shortname"].startswith(("AB.ATE", "JS.ATE"))]
        print(f"\n[ATE] {len(ate):,} rows in core_b_rms_ate_vsp_rcm (v12 baseline ~10,500)")
        if len(ate) < 12000:
            fail(f"ATE total {len(ate):,} below v19 floor 12,000")

    # 5. summary
    print("\n=== summary ===")
    if failures:
        print(f"FAIL: {len(failures)} validation issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: all v19 shards validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
