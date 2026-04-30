#!/usr/bin/env python
"""Convert each of the 5 recovered failed-template triples to Alpaca,
apply the same post-processing (CVE-CVE-/CWE-CWE- prefix collapse), and
merge into the canonical v8.1 dataset."""
import json, os, re, shutil, subprocess

DATA   = "SFT/data/ift_data_2026_04_30_v81.json"
BACKUP = "SFT/data/ift_data_2026_04_30_v81.pre_failed_recovery.json"
SHORTNAMES = ("AB.MCQ.3", "P.7", "X.8", "SU.G.1", "SU.POC.1")

PATTERNS = [
    (re.compile(r"\bCVE-CVE-"),     "CVE-"),
    (re.compile(r"\bCWE-CWE-"),     "CWE-"),
    (re.compile(r"\bCAPEC-CAPEC-"), "CAPEC-"),
]
def fix(text):
    if not text: return text
    for pat, repl in PATTERNS:
        text = pat.sub(repl, text)
    return text

# 1) one-shot back-up
if not os.path.exists(BACKUP):
    shutil.copy(DATA, BACKUP)
    print(f"backup -> {BACKUP}")

# 2) convert each triples_<sn>/  ->  alpaca via to_alpaca.py
all_extra = []
for sn in SHORTNAMES:
    triples_dir = f"_failed_test/triples_{sn}"
    out         = f"_failed_test/{sn}_alpaca.json"
    cmd = [
        "tmpl_gen/venv/bin/python", "tmpl_gen/scripts/to_alpaca.py",
        "--results_dir", triples_dir,
        "--output",      out,
        "--count_max",   "100",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    last_line = (proc.stdout + proc.stderr).strip().splitlines()[-1]
    rows = json.load(open(out))
    n_fixed = 0
    for r in rows:
        for f in ("instruction","input","output"):
            old = r.get(f, "") or ""
            new = fix(old)
            if new != old:
                r[f] = new
                n_fixed += 1
    print(f"  {sn:10s} converted={len(rows):3d}  prefix_fixes={n_fixed:3d}  ({last_line})")
    all_extra.extend(rows)

# 3) merge into the dataset
base = json.load(open(DATA))
merged = base + all_extra
print(f"\nbase rows  : {len(base)}")
print(f"adding     : {len(all_extra)}")
print(f"merged total: {len(merged)}")
json.dump(merged, open(DATA, "w"), ensure_ascii=False)
print(f"wrote: {DATA}")
