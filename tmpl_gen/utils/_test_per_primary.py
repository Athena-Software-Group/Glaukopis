#!/usr/bin/env python3
"""Quick test of AB.MCQ.3 with and without per_primary_grouping enabled.

Runs iftgen on a single isolated AB.MCQ.3 template at Count: 50 to compare
generation time / success between the default (huge cartesian materialised)
and per_primary_grouping=True (CALL subquery with LIMIT 1 per primary).
"""
from __future__ import annotations
import json, sys, time, subprocess, pathlib, tempfile, os

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_TMPL = ROOT / "templates" / "04302026" / "Sophia-CTI-Templates-v8_1.txt"
DOCX2JSON = ROOT / "scripts" / "tmpl_docx2json.py"
IFTGEN = ROOT / "scripts" / "iftgen.py"
DBCONF = ROOT / "data_generation" / "neo4j-local-config.json"
GENCFG_BASE = ROOT / "data_generation" / "gencfg_default_neo4j.json"
PY = str(ROOT / "venv" / "bin" / "python")


def isolate(shortname: str, txt_in: pathlib.Path, txt_out: pathlib.Path) -> None:
    text = txt_in.read_text().splitlines(keepends=True)
    out, capture = [], False
    for line in text:
        if line.startswith(f"{shortname} Instruction:"):
            capture = True
        elif capture and line and not line[0].isspace() and not line.startswith(("{", "Sample:", "Shuffle:", "Count:", "A)", "B)", "C)", "D)", "E)", "Question:", "Answer:")):
            if "Instruction:" in line:
                capture = False
        if capture:
            out.append(line)
    txt_out.write_text("".join(out))


def run_one(label: str, gencfg: pathlib.Path, tmpl_json: pathlib.Path, results_dir: pathlib.Path) -> None:
    if results_dir.exists():
        for f in results_dir.iterdir():
            f.unlink()
        results_dir.rmdir()
    results_dir.mkdir(parents=True)
    t0 = time.time()
    print(f"\n=== {label} ===")
    p = subprocess.Popen(
        [PY, str(IFTGEN), "--cmd", "generate",
         "--genconf", str(gencfg),
         "--dbconf", str(DBCONF),
         "--tmpl", str(tmpl_json),
         "--results_dir", str(results_dir),
         "--count_max", "50"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        out, _ = p.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        print(f"  [TIMEOUT after {time.time()-t0:.1f}s]")
        return
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  exit:    {p.returncode}")
    print("  --- output (last 20 lines) ---")
    for ln in out.splitlines()[-20:]:
        print(f"  {ln}")
    for f in sorted(results_dir.iterdir()):
        d = json.loads(f.read_text())
        print(f"  result {f.name}: generated={d.get('generated_count')}")


def main() -> None:
    workdir = pathlib.Path("_per_primary_test")
    workdir.mkdir(exist_ok=True)
    txt_iso = workdir / "ab_mcq3_iso.txt"
    isolate("AB.MCQ.3", SRC_TMPL, txt_iso)
    print(f"isolated template ({txt_iso.stat().st_size} bytes):")
    print(txt_iso.read_text())

    json_iso = workdir / "ab_mcq3_iso.json"
    subprocess.check_call([PY, str(DOCX2JSON), "--input", str(txt_iso), "--out", str(json_iso)])

    gencfg_orig = workdir / "gencfg_default.json"
    gencfg_orig.write_text(GENCFG_BASE.read_text())

    gencfg_pp = workdir / "gencfg_perprimary.json"
    cfg = json.loads(GENCFG_BASE.read_text())
    cfg["per_primary_grouping"] = True
    gencfg_pp.write_text(json.dumps(cfg, indent=2))

    run_one("default (no per_primary_grouping)", gencfg_orig, json_iso, workdir / "results_default")
    run_one("per_primary_grouping=True", gencfg_pp, json_iso, workdir / "results_perprimary")


if __name__ == "__main__":
    main()
