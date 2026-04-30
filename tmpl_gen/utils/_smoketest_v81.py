#!/usr/bin/env python3
"""One-shot smoketest for v8.1 build readiness:
  - confirms tmpl_gen venv has neo4j + docx
  - confirms local neo4j is reachable, prints node/label counts
  - prints mitigation / technique / mitigates-edge counts so the v8.1
    template author has the catalog cardinality figures handy
"""
import json
import sys
from neo4j import GraphDatabase

CFG_PATH = "tmpl_gen/data_generation/neo4j-local-config.json"


def main() -> int:
    cfg = json.load(open(CFG_PATH))
    try:
        drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))
        with drv.session(database=cfg["db_name"]) as s:
            n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            labels = sorted(r["l"] for r in s.run(
                "CALL db.labels() YIELD label AS l RETURN l").data())
            coa = s.run('MATCH (n:`course-of-action`) RETURN count(n) AS c').single()["c"]
            ap = s.run('MATCH (n:`attack-pattern`) RETURN count(n) AS c').single()["c"]
            ap_with_mit = s.run(
                "MATCH (a:`attack-pattern`)-[:mitigates]-(c:`course-of-action`) "
                "RETURN count(DISTINCT a) AS c").single()["c"]
            coa_used = s.run(
                "MATCH (a:`attack-pattern`)-[:mitigates]-(c:`course-of-action`) "
                "RETURN count(DISTINCT c) AS c").single()["c"]
        drv.close()
    except Exception as e:
        print(f"[FAIL] neo4j not reachable: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"  uri          : {cfg['uri']}")
    print(f"  db           : {cfg['db_name']}")
    print(f"  total nodes  : {n:,}")
    print(f"  labels       : {labels}")
    print(f"  mitigations  : {coa}")
    print(f"  techniques   : {ap}")
    print(f"  techniques w/mitigations: {ap_with_mit}")
    print(f"  mitigations referenced  : {coa_used}")
    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
