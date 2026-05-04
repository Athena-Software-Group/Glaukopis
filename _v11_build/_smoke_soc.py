"""Smoke-test each SOC.* template against the live athena-cti-db Neo4j
DB by extracting the JSON for each shortname and running it through
TmplGenNeo4j with count_max=3 (just enough to confirm Cypher binds).
Reports rows generated per template; non-zero == OK."""
import json, sys, pathlib, tempfile, os
sys.path.insert(0, "tmpl_gen/src")

V11_JSON = pathlib.Path("/tmp/v11_check.json")
DBCFG = pathlib.Path("tmpl_gen/data_generation/neo4j-local-config.json")
GENCFG = pathlib.Path("tmpl_gen/data_generation/gencfg_per_primary_neo4j.json")

assert V11_JSON.exists() and DBCFG.exists() and GENCFG.exists(), "missing inputs"

all_tmpls = json.loads(V11_JSON.read_text())
soc = [t for t in all_tmpls if t.get("shortname", "").startswith("SOC.")]
target = sys.argv[1] if len(sys.argv) > 1 else None
if target:
    soc = [t for t in soc if t["shortname"] == target]
print(f"Found {len(soc)} SOC templates")

from tmpl_gen.tmpl_parser import TmplGenNeo4j  # type: ignore

results = {}
for t in soc:
    sn = t["shortname"]
    print(f"\n=== {sn} (Count: {t.get('count_limit', '-')}) ===")
    # cap at 3 to keep fast
    t_capped = dict(t); t_capped["count_limit"] = 3
    one_file = pathlib.Path(tempfile.mkstemp(suffix=".json")[1])
    one_file.write_text(json.dumps([t_capped]))
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="soc_smoke_"))
    options = {
        "gen_conf_file": str(GENCFG),
        "templates_file": str(one_file),
        "neo4j_conf_file": str(DBCFG),
        "results_dir": str(out_dir),
        "count_max": 3,
        "verbose": 1 if target else 0,
        "allow_nullprops": True,
    }
    try:
        tg = TmplGenNeo4j(options)
        lst = tg.load_templates(options["templates_file"])
        (gen, fail) = tg.generate(lst, do_print=False)
        rows = 0
        sample = None
        for p in out_dir.glob("*.json"):
            if p.name.startswith("_"):
                continue
            try:
                d = json.loads(p.read_text())
                if isinstance(d, dict) and "generated_strings" in d:
                    rows += len(d["generated_strings"])
                    if sample is None and d["generated_strings"]:
                        sample = d["generated_strings"][0]
            except Exception:
                pass
        results[sn] = rows
        print(f"  -> {rows} rows  (gen={gen} fail={fail})")
        if sample:
            print(f"  sample: {sample[:280]}")
    except Exception as e:
        results[sn] = f"ERR: {e}"
        print(f"  ERR: {e}")

print("\n========== SUMMARY ==========")
for sn, r in results.items():
    print(f"  {sn:30s} {r}")
