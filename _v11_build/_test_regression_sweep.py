#!/usr/bin/env python3
"""F3 emitter regression sweep against live Neo4j.

For each (shortname, gencfg, expected_min_rows) tuple:
- Isolate the template from Sophia-CTI-Templates-v10.txt
- Run iftgen with the given gencfg
- Read generated_count from the result JSON
- Check elapsed time and row count

Templates exercised:
- AB.MS.GRP.1   per_primary  -- canonical F3 case
- AB.MS.MAL.1   per_primary  -- F3 with malware as dependent type
- AB.TAA.1      per_primary  -- threat actor attribution (different shape)
- AB.MCQ.1      legacy       -- single-answer MCQ via legacy emitter
"""
from __future__ import annotations
import json, subprocess, sys, time, pathlib

ROOT    = pathlib.Path(__file__).resolve().parent.parent
TG      = ROOT / "tmpl_gen"
PY      = str(TG / "venv" / "bin" / "python")
DBCONF  = TG / "data_generation" / "neo4j-local-config.json"
GC_DEF  = TG / "data_generation" / "gencfg_default_neo4j.json"
GC_PP   = TG / "data_generation" / "gencfg_per_primary_neo4j.json"
SRC     = TG / "templates" / "05012026" / "Sophia-CTI-Templates-v10.txt"
DOCX2J  = TG / "scripts" / "tmpl_docx2json.py"
IFTGEN  = TG / "scripts" / "iftgen.py"
WORK    = ROOT / "_v11_build" / "_regression_sweep"
WORK.mkdir(parents=True, exist_ok=True)


def isolate(shortname: str, dst: pathlib.Path) -> None:
    text = SRC.read_text().splitlines(keepends=True)
    out, capture = [], False
    for line in text:
        if line.startswith(f"{shortname} Instruction:"):
            capture = True
        elif capture and line and not line[0].isspace() and not line.startswith(
                ("{", "Sample:", "Shuffle:", "Count:", "A)", "B)", "C)", "D)",
                 "E)", "Question:", "Answer:")):
            if "Instruction:" in line:
                capture = False
        if capture:
            out.append(line)
    dst.write_text("".join(out))


def run_one(shortname: str, gencfg: pathlib.Path, count_max: int,
            timeout_s: int = 180) -> dict:
    iso_txt  = WORK / f"{shortname}.txt"
    iso_json = WORK / f"{shortname}.json"
    res_dir  = WORK / f"results_{shortname}"
    isolate(shortname, iso_txt)
    if iso_txt.stat().st_size == 0:
        return {"shortname": shortname, "rows": 0, "elapsed": 0.0,
                "exit": -1, "note": "isolation produced empty file"}
    subprocess.check_call([PY, str(DOCX2J), "-i", str(iso_txt),
                           "-o", str(iso_json), "--count_limit", str(count_max)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res_dir.exists():
        for f in res_dir.iterdir():
            f.unlink()
        res_dir.rmdir()
    res_dir.mkdir(parents=True)
    t0 = time.time()
    p = subprocess.Popen(
        [PY, str(IFTGEN), "--cmd", "generate", "--genconf", str(gencfg),
         "--dbconf", str(DBCONF), "--tmpl", str(iso_json),
         "--results_dir", str(res_dir), "--count_max", str(count_max)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        return {"shortname": shortname, "rows": 0, "elapsed": time.time()-t0,
                "exit": -1, "note": f"TIMEOUT @ {timeout_s}s"}
    elapsed = time.time() - t0
    rows = 0
    note = ""
    for f in sorted(res_dir.iterdir()):
        if f.name == "_results-report.json":
            continue
        d = json.loads(f.read_text())
        rows += int(d.get("generated_count") or 0)
    if "WARN: primary query hit" in out:
        note = "primary->fallback"
    if "Generated: 0" in out:
        note = (note + "; gen=0").strip("; ")
    return {"shortname": shortname, "rows": rows, "elapsed": round(elapsed, 2),
            "exit": p.returncode, "note": note, "gencfg": gencfg.name}


def main():
    # Per-template min row expectations are calibrated against the actual
    # graph cardinality, not the requested Count: (DISTINCT dedup on
    # with-replacement sampling + bounded combo space caps yield well below
    # the requested row count for templates whose joined product is small).
    cases = [
        ("AB.MS.GRP.1", GC_PP,  1500, 1000),  # F3 canonical case
        ("AB.MS.MAL.1", GC_PP,  1500,  500),  # F3, smaller mal-using anchor pool
        ("AB.TAA.1",    GC_PP,   500,  100),  # F3, threat-actor attribution
        ("AB.MCQ.1",    GC_DEF,  500,  100),  # legacy emitter, unchanged path
    ]
    print(f"{'shortname':<14} {'gencfg':<32} {'rows':>6} {'t(s)':>6}  note")
    print("-" * 78)
    results = []
    for shortname, gc, cmax, _min in cases:
        r = run_one(shortname, gc, cmax, timeout_s=180)
        results.append(r)
        print(f"{shortname:<14} {gc.name:<32} {r['rows']:>6} {r['elapsed']:>6}  {r.get('note','')}")
    print("-" * 78)
    fails = []
    for r, (sn, _, _, mn) in zip(results, cases):
        if r["rows"] < mn:
            fails.append(f"{sn}: rows={r['rows']} < expected_min={mn}")
    if fails:
        print("\nFAIL:\n  " + "\n  ".join(fails))
        sys.exit(1)
    print("\nALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
