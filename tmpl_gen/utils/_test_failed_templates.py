#!/usr/bin/env python
"""Run each of the 5 v8.1 build-failure templates in isolation through
iftgen to determine which still fail with the recompiled (header-bleed
fixed) JSON, and how each one fails."""
import json, os, shutil, subprocess, sys, time

RECOMPILED = "_qmsr1_test/Sophia-CTI-Templates-v8_1.recompiled.json"
TMPL_OUT   = "_failed_test"
SHORTNAMES = ("AB.MCQ.3", "P.7", "X.8", "SU.G.1", "SU.POC.1")
COUNT_MAX  = 100   # default; AB.MCQ.3 cartesian likely needs lower

os.makedirs(TMPL_OUT, exist_ok=True)
all_t = json.load(open(RECOMPILED))
by_sn = {t["shortname"]: t for t in all_t}

results = {}
for sn in SHORTNAMES:
    t = by_sn.get(sn)
    if not t:
        results[sn] = ("MISSING_FROM_RECOMPILED", 0, 0.0)
        continue
    sub_path = f"{TMPL_OUT}/{sn}.json"
    res_dir  = f"{TMPL_OUT}/triples_{sn}"
    json.dump([t], open(sub_path, "w"), indent=2)
    if os.path.isdir(res_dir): shutil.rmtree(res_dir)
    cmd = [
        "tmpl_gen/venv/bin/python", "tmpl_gen/scripts/iftgen.py",
        "--cmd", "generate",
        "--genconf", "tmpl_gen/data_generation/gencfg_default_neo4j.json",
        "--dbconf",  "tmpl_gen/data_generation/neo4j-local-config.json",
        "--tmpl",    sub_path,
        "--results_dir", res_dir,
        "--count_max",   str(COUNT_MAX),
    ]
    print(f"\n========== {sn} (count_max={COUNT_MAX}) ==========")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        out  = (proc.stdout + proc.stderr).strip().splitlines()
        for ln in out[-6:]:
            print(f"  > {ln}")
    except subprocess.TimeoutExpired:
        print("  > SUBPROCESS TIMED OUT (>240s)")
    elapsed = time.time() - t0
    # Read the per-template result file if it exists
    rep_path = f"{res_dir}/_results-report.json"
    n_rows = 0; exc = ""
    if os.path.isfile(rep_path):
        rep = json.load(open(rep_path))
        if rep.get("results"):
            r0 = rep["results"][0]
            n_rows = r0.get("generated_count", 0) or 0
            exc_raw = (r0.get("exception", "") or "")
            exc = (exc_raw.splitlines()[0][:140] if exc_raw else "")
    print(f"  rows={n_rows}  elapsed={elapsed:.1f}s  exc={exc}")
    results[sn] = ("OK" if n_rows > 0 else "FAIL", n_rows, elapsed)

print("\n========== SUMMARY ==========")
for sn, (st, n, e) in results.items():
    print(f"  {sn:10s} {st:6s} rows={n:4d}  {e:.1f}s")
