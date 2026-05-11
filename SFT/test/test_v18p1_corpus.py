#!/usr/bin/env python3
# v18.1 Core SFT corpus regression check. Run from repo root:
#     python SFT/test/test_v18p1_corpus.py
# Exits 0 on success, 1 on any failure. No pytest dependency.

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "SFT" / "data"
GATE = REPO / "_v18p1_build" / "row_count_gate_report.json"

SHARDS = {
    "phase_a": DATA / "ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn.json",
    "phase_b": DATA / "ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm.json",
    "val":     DATA / "ift_data_2026_05_11_v18p1_core_val.json",
    "train":   DATA / "ift_data_2026_05_11_v18p1_core_train.json",
}

EXPECTED_ROWS = {"phase_a": 258_403, "phase_b": 71_447, "val": 550, "train": 329_850}
EXPECTED_AB_MCQ_789 = {"AB.MCQ.7": 797, "AB.MCQ.8": 397, "AB.MCQ.9": 445}
LETTER_BALANCE_MAX = 0.30   # any single A-E letter <= 30% per AB.MCQ.{1..9}
LETTER_RE = re.compile(r"\b([A-E])\b")
PASSED, FAILED = [], []

def check(label, ok, detail=""):
    (PASSED if ok else FAILED).append(label)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))

def load(path):
    with open(path) as fh:
        return json.load(fh)

# 1. Shards present + row counts
print("\n1. shard presence + row counts")
shards_data = {}
for name, path in SHARDS.items():
    if not path.exists():
        check(f"{name} exists", False, str(path))
        continue
    rows = load(path)
    shards_data[name] = rows
    check(f"{name} rows == {EXPECTED_ROWS[name]:,}", len(rows) == EXPECTED_ROWS[name], f"got {len(rows):,}")

# 2. Alpaca schema (instruction / input / output) + shortname tag
print("\n2. schema (alpaca + shortname)")
required = {"instruction", "input", "output", "shortname"}
for name, rows in shards_data.items():
    if not rows:
        continue
    missing = [i for i, r in enumerate(rows[:1000]) if not required.issubset(r.keys())]
    check(f"{name} schema OK on first 1000 rows", not missing, f"missing keys at idx {missing[:3]}")

# 3. MCQ letter balance for AB.MCQ.{1..9}
print("\n3. AB.MCQ.{1..9} correct-answer letter balance")
def mcq_letter(row):
    out = (row.get("output") or "").strip()
    if "Therefore," in out:
        tail = out.rsplit("Therefore,", 1)[1]
        m = LETTER_RE.search(tail)
        if m:
            return m.group(1)
    m = re.search(r"\b([A-E])\b\.?\s*$", out)
    return m.group(1) if m else None

phase_a = shards_data.get("phase_a", [])
mcq_rows = [r for r in phase_a if str(r.get("shortname", "")).startswith("AB.MCQ.")
            and str(r.get("shortname", "")).split(".")[-1] in {"1","2","3","4","5","6","7","8","9"}]
per_template = {}
for r in mcq_rows:
    sn = r["shortname"]
    letter = mcq_letter(r)
    if letter:
        per_template.setdefault(sn, Counter())[letter] += 1
for sn in sorted(per_template):
    total = sum(per_template[sn].values())
    if total < 50:
        continue
    pct = {l: per_template[sn][l] / total for l in "ABCDE"}
    worst_letter, worst_pct = max(pct.items(), key=lambda kv: kv[1])
    detail = f"n={total}  worst={worst_letter}={worst_pct:.1%}  dist=" + " ".join(f"{l}={pct[l]:.0%}" for l in "ABCDE")
    check(f"{sn} max letter share <= {LETTER_BALANCE_MAX:.0%}", worst_pct <= LETTER_BALANCE_MAX, detail)

# 4. AB.MCQ.{7,8,9} substrate-bound row counts (counted across train+val,
#    i.e. the full clean corpus before the val slice was siphoned off)
print("\n4. AB.MCQ.{7,8,9} substrate-bound row counts (train+val)")
all_rows = shards_data.get("train", []) + shards_data.get("val", [])
sn_counts = Counter(r.get("shortname") for r in all_rows)
for sn, expected in EXPECTED_AB_MCQ_789.items():
    actual = sn_counts.get(sn, 0)
    check(f"{sn} == {expected} rows", actual == expected, f"got {actual}")

# 5. Train/val 0-overlap (fingerprint over instruction|input|output)
print("\n5. train/val disjointness (fingerprint hash)")
def fp(r):
    blob = f"{r.get('instruction','')}\x1f{r.get('input','')}\x1f{r.get('output','')}".encode()
    return hashlib.sha1(blob).hexdigest()

train_fps = {fp(r) for r in shards_data.get("train", [])}
val_fps = {fp(r) for r in shards_data.get("val", [])}
overlap = train_fps & val_fps
check("train/val fingerprint overlap == 0", not overlap, f"{len(overlap)} overlapping rows")
check("|train| unique fingerprints > 0.99 * row_count",
      len(train_fps) > 0.99 * len(shards_data.get("train", [])),
      f"|fps|={len(train_fps):,} of {len(shards_data.get('train', [])):,}")

# 6. Row-count gate: every axis status == OK
print("\n6. row-count gate (build report)")
if GATE.exists():
    rep = load(GATE)
    for axis in rep.get("axes", []):
        check(f"axis {axis['axis']:18s} status OK", axis.get("status") == "OK",
              f"actual={axis.get('actual')} floor={axis.get('reject_if_below')}")
else:
    check("row_count_gate_report.json present", False, str(GATE))

print(f"\n=== summary: {len(PASSED)} passed, {len(FAILED)} failed ===")
if FAILED:
    print("FAILED:")
    for f in FAILED:
        print(f"  - {f}")
sys.exit(0 if not FAILED else 1)
