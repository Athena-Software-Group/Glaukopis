#!/usr/bin/env python
"""Sanity check the v8.1 SFT dataset.

Checks:
  1. Schema:    every row has instruction/input/output (str, non-empty for output).
  2. Shape:     no NULL or absurdly small/large rows; print length histogram.
  3. Coverage:  one sample per family + per AB.RMS / JS.RMS template (catalog drills).
  4. RMS:       for each preserved RMS template, confirm output ID format
                (`M\\d{4}` / `T\\d{4}(\\.\\d{3})?` etc.).
  5. JSON shape: for JS.* templates, the output should be valid JSON.
"""
import json, re, random
from collections import defaultdict, Counter

PATH = "SFT/data/ift_data_2026_04_30_v81.json"
data = json.load(open(PATH))
print(f"loaded {len(data)} rows from {PATH}\n")

# 1. Schema
missing = Counter()
empty = Counter()
for r in data:
    for k in ("instruction", "input", "output"):
        if k not in r:
            missing[k] += 1
        elif not isinstance(r.get(k), str):
            missing[f"{k}:nonstr"] += 1
        elif k == "output" and not r[k].strip():
            empty[k] += 1
print("schema check:")
print(f"  missing fields    : {dict(missing)}")
print(f"  empty output rows : {dict(empty)}")
print()

# 2. Length histogram (chars)
def lh(field):
    lens = [len(r.get(field, "") or "") for r in data]
    lens.sort()
    n = len(lens)
    return {
        "min": lens[0], "p10": lens[n//10], "p50": lens[n//2],
        "p90": lens[(9*n)//10], "p99": lens[(99*n)//100], "max": lens[-1],
        "mean": sum(lens)//n,
    }
for f in ("instruction", "input", "output"):
    print(f"length({f:11s}): {lh(f)}")
print()

# 3. Per-template counts
by_t = defaultdict(list)
for r in data:
    sn = r.get("shortname") or "?"
    by_t[sn].append(r)
print(f"unique templates: {len(by_t)}")
families = defaultdict(int)
for sn, rows in by_t.items():
    families[sn.split('.')[0]] += len(rows)
print("rows by family:", dict(families))
print()

# 4. RMS catalog drills
print("=== RMS catalog drill check ===")
M_RE = re.compile(r"\bM\d{4}\b")
T_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
for prefix in ("AB.RMS", "JS.RMS"):
    drills = sorted(sn for sn in by_t if sn.startswith(prefix))
    for sn in drills:
        rows = by_t[sn]
        out_sample = rows[0].get("output", "")
        m_ids = M_RE.findall(out_sample)
        t_ids = T_RE.findall(out_sample)
        flag = " " if (m_ids or t_ids) else "!"
        print(f"  {flag} {sn:14s} n={len(rows):4d}  m_ids={m_ids[:3]}  t_ids={t_ids[:3]}")
print()

# 5. JS.* outputs are valid JSON
print("=== JS.* JSON-shape check ===")
js_total = 0; js_bad = 0
bad_examples = []
for sn, rows in by_t.items():
    if not sn.startswith("JS."):
        continue
    for r in rows:
        js_total += 1
        out = r.get("output", "").strip()
        # outputs may be a JSON object embedded in larger text; require it to start with { or [
        try:
            json.loads(out)
        except Exception as e:
            js_bad += 1
            if len(bad_examples) < 3:
                bad_examples.append((sn, str(e)[:80], out[:200]))
print(f"  JS rows: {js_total}; non-JSON output: {js_bad}")
for sn, err, snippet in bad_examples:
    print(f"    [{sn}] err={err}")
    print(f"      {snippet}")
print()

# 6. Random visual samples
print("=== random visual samples (1 per template family) ===")
rnd = random.Random(20260430)
seen = set()
samples = []
order = list(by_t.items())
rnd.shuffle(order)
for sn, rows in order:
    fam = sn.split('.')[0]
    if fam in seen: continue
    seen.add(fam)
    samples.append((sn, rnd.choice(rows)))
for sn, r in samples:
    print(f"\n--- {sn} ---")
    print(f"  INSTR : {(r.get('instruction','') or '')[:200]}")
    print(f"  INPUT : {(r.get('input','') or '')[:300]}")
    print(f"  OUTPUT: {(r.get('output','') or '')[:300]}")

# 7. CL.* (cloze) and SU.* (summary) sanity
print("\n=== CL.* / SU.* output shape ===")
for prefix in ("CL.", "SU."):
    rows = [r for sn, rs in by_t.items() if sn.startswith(prefix) for r in rs]
    if not rows: continue
    print(f"  {prefix} n={len(rows)} sample output={rows[0].get('output','')[:200]}")
