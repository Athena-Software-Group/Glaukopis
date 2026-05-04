#!/usr/bin/env python3
"""v11 anchor-fixation fix validation against live Neo4j.

Runs AB.MS.GRP.1 (the canonical v10 anchor-fixation failure case) in
isolation under two gencfg variants and compares row counts + anchor
diversity:

    A. gencfg_default_neo4j.json  (per_primary_grouping unset)
       -> exercises the legacy DISTINCT-prefix Cypher path
    B. gencfg_per_primary_neo4j.json  (per_primary_grouping=true,
       allow_nullprops=true)
       -> exercises the new UNWIND-with-replacement prefix +
          ORDER BY rand() inner CALL subquery

Pass criteria for B (the v11 production path):
- generated rows >= 1000 (target was 1500; AB.MS.GRP.1 v10 yielded 0)
- distinct grp anchors >= 50 (i.e. not collapsed to a single anchor)

No commits, no side effects outside _v11_build/_anchor_fix_test/.
"""
from __future__ import annotations
import json, subprocess, sys, time, pathlib, collections

ROOT    = pathlib.Path(__file__).resolve().parent.parent
TMPL_GEN= ROOT / "tmpl_gen"
PY      = str(TMPL_GEN / "venv" / "bin" / "python")
DBCONF  = TMPL_GEN / "data_generation" / "neo4j-local-config.json"
GENCFG_A= TMPL_GEN / "data_generation" / "gencfg_default_neo4j.json"
GENCFG_B= TMPL_GEN / "data_generation" / "gencfg_per_primary_neo4j.json"
SRC_TMPL= TMPL_GEN / "templates" / "05012026" / "Sophia-CTI-Templates-v10.txt"
DOCX2J  = TMPL_GEN / "scripts" / "tmpl_docx2json.py"
IFTGEN  = TMPL_GEN / "scripts" / "iftgen.py"

WORKDIR = ROOT / "_v11_build" / "_anchor_fix_test"
WORKDIR.mkdir(parents=True, exist_ok=True)


def isolate_template(shortname: str, src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy the template block named `shortname` (e.g. AB.MS.GRP.1) into dst.
    Uses the same line-capture logic as tmpl_gen/utils/_test_per_primary.py."""
    text = src.read_text().splitlines(keepends=True)
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


def run_iftgen(label: str, gencfg: pathlib.Path, tmpl_json: pathlib.Path,
               results_dir: pathlib.Path, count_max: int) -> dict:
    if results_dir.exists():
        for f in results_dir.iterdir():
            f.unlink()
        results_dir.rmdir()
    results_dir.mkdir(parents=True)
    print(f"\n=== {label} ===")
    print(f"  gencfg: {gencfg.name}")
    print(f"  count_max: {count_max}")
    t0 = time.time()
    p = subprocess.Popen(
        [PY, str(IFTGEN), "--cmd", "generate",
         "--genconf", str(gencfg),
         "--dbconf", str(DBCONF),
         "--tmpl", str(tmpl_json),
         "--results_dir", str(results_dir),
         "--count_max", str(count_max)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        out, _ = p.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        print(f"  [TIMEOUT after {time.time()-t0:.1f}s]")
        return {"label": label, "rows": 0, "anchors": 0, "timeout": True}
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.2f}s   exit: {p.returncode}")
    print("  --- iftgen tail (last 12 lines) ---")
    for ln in out.splitlines()[-12:]:
        print(f"  {ln}")
    # iftgen result files: {generated_count, generated_strings:[...], query, ...}
    # Anchor extraction: AB.MS.GRP.1 question text contains "threat group <NAME> (G####)".
    import re as _re
    anchor_rx = _re.compile(r"threat group ([^(]+?) \(G\d+\)")
    rows, grp_counts = 0, collections.Counter()
    for f in sorted(results_dir.iterdir()):
        if f.name == "_results-report.json":
            continue
        d = json.loads(f.read_text())
        gs = d.get("generated_strings", []) or []
        rows += len(gs)
        for s in gs:
            m = anchor_rx.search(s)
            if m:
                grp_counts[m.group(1).strip()] += 1
        print(f"  result {f.name}: rows={len(gs)} generated_count={d.get('generated_count')}")
    print(f"  TOTAL rows: {rows}    distinct grp anchors: {len(grp_counts)}")
    if grp_counts:
        top5 = grp_counts.most_common(5)
        print(f"  top-5 anchors: {top5}")
    return {"label": label, "rows": rows, "anchors": len(grp_counts),
            "elapsed_s": round(elapsed, 2), "exit": p.returncode,
            "top5": grp_counts.most_common(5)}


def main() -> None:
    iso_txt = WORKDIR / "ab_ms_grp1.txt"
    iso_json = WORKDIR / "ab_ms_grp1.json"
    isolate_template("AB.MS.GRP.1", SRC_TMPL, iso_txt)
    print(f"isolated template ({iso_txt.stat().st_size} bytes):\n---")
    print(iso_txt.read_text())
    print("---")
    subprocess.check_call([PY, str(DOCX2J), "-i", str(iso_txt), "-o", str(iso_json),
                           "--count_limit", "1500"])

    results = []
    results.append(run_iftgen("A. legacy default gencfg (no per_primary)",
                              GENCFG_A, iso_json, WORKDIR / "results_A", 1500))
    results.append(run_iftgen("B. v11 per-primary gencfg (UNWIND fix)",
                              GENCFG_B, iso_json, WORKDIR / "results_B", 1500))

    print("\n===================== SUMMARY =====================")
    for r in results:
        print(f"  {r['label']:<48}  rows={r['rows']:<6} anchors={r['anchors']:<4} t={r.get('elapsed_s', '?')}s")
    b = results[1]
    pass_rows    = b["rows"] >= 1000
    pass_anchors = b["anchors"] >= 50
    print(f"\n  Pass (rows >= 1000):    {'YES' if pass_rows else 'NO'}  ({b['rows']})")
    print(f"  Pass (anchors >= 50):  {'YES' if pass_anchors else 'NO'}  ({b['anchors']})")
    sys.exit(0 if (pass_rows and pass_anchors) else 1)


if __name__ == "__main__":
    main()
