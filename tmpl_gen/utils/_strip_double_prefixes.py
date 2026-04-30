#!/usr/bin/env python
"""Collapse double-prefix tokens emitted by templates that put a literal
"CVE-" / "CWE-" / "CAPEC-" / "T" / "M" before a {var} whose .id already
carries the prefix.

Concretely:
    "CVE-CVE-2024-1141" -> "CVE-2024-1141"
    "CWE-CWE-444"       -> "CWE-444"
    "CAPEC-CAPEC-66"    -> "CAPEC-66"

Operates in place on SFT/data/ift_data_2026_04_30_v81.json.  Use --dry-run
to preview counts.  Backup is written once.
"""
import json, re, shutil, sys
from pathlib import Path

DATA_PATH   = Path("SFT/data/ift_data_2026_04_30_v81.json")
BACKUP_PATH = Path("SFT/data/ift_data_2026_04_30_v81.pre_dedup_prefix.json")

PATTERNS = [
    (re.compile(r"\bCVE-CVE-"),       "CVE-"),
    (re.compile(r"\bCWE-CWE-"),       "CWE-"),
    (re.compile(r"\bCAPEC-CAPEC-"),   "CAPEC-"),
]

def fix(text: str) -> tuple[str, int]:
    if not text:
        return text, 0
    n = 0
    for pat, repl in PATTERNS:
        new, k = pat.subn(repl, text)
        n += k
        text = new
    return text, n

def main():
    apply = "--apply" in sys.argv
    data = json.load(open(DATA_PATH))
    n_rows = len(data)

    edits_per_field = {"instruction": 0, "input": 0, "output": 0}
    rows_touched = 0
    sample_diffs = []
    for r in data:
        touched = False
        for f in edits_per_field:
            old = r.get(f, "")
            new, k = fix(old)
            if k > 0:
                edits_per_field[f] += k
                if apply:
                    r[f] = new
                if not touched and len(sample_diffs) < 4:
                    sample_diffs.append((r.get("shortname"), f, old[:200], new[:200]))
                touched = True
        if touched:
            rows_touched += 1

    print(f"rows touched: {rows_touched} / {n_rows}")
    print(f"replacements per field: {edits_per_field}")
    print()
    for sn, f, old, new in sample_diffs:
        print(f"--- {sn} {f} ---")
        print(f"  old: {old}")
        print(f"  new: {new}")
        print()

    if not apply:
        print("(dry run; pass --apply to write changes)")
        return

    if not BACKUP_PATH.exists():
        shutil.copy(DATA_PATH, BACKUP_PATH)
        print(f"backup written: {BACKUP_PATH}")

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"wrote: {DATA_PATH}")

if __name__ == "__main__":
    main()
