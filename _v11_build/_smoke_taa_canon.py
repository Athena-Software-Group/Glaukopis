"""Smoke-test TAA.CANON.1/2 binding against athena-cti-db."""
import json, sys, pathlib, tempfile
sys.path.insert(0, "tmpl_gen/src")
from tmpl_gen.tmpl_parser import TmplGenNeo4j  # type: ignore

V11_JSON = pathlib.Path("/tmp/v11_check.json")
DBCFG = pathlib.Path("tmpl_gen/data_generation/neo4j-local-config.json")
GENCFG = pathlib.Path("tmpl_gen/data_generation/gencfg_per_primary_neo4j.json")

all_tmpls = json.loads(V11_JSON.read_text())
targets = [t for t in all_tmpls if t.get("shortname", "").startswith("TAA.CANON.")]
print(f"Found {len(targets)} TAA.CANON templates")

for t in targets:
    sn = t["shortname"]
    print(f"\n=== {sn} (Count: {t.get('count_limit', '-')}) ===")
    t_capped = dict(t); t_capped["count_limit"] = 3
    one_file = pathlib.Path(tempfile.mkstemp(suffix=".json")[1])
    one_file.write_text(json.dumps([t_capped]))
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="taa_canon_smoke_"))
    options = {
        "gen_conf_file": str(GENCFG),
        "templates_file": str(one_file),
        "neo4j_conf_file": str(DBCFG),
        "results_dir": str(out_dir),
        "count_max": 3,
        "verbose": 0,
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
        print(f"  -> {rows} rows  (gen={gen} fail={fail})")
        if sample:
            print(f"  sample: {sample[:300]}")
    except Exception as e:
        print(f"  ERR: {e}")
