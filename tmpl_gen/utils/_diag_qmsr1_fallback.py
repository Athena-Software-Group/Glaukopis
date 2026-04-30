#!/usr/bin/env python
"""Run Q.MSR.1's fallback query directly (with a hard wall-clock timeout)
to determine what actually fails: syntax error, cartesian explosion, or
something else."""
import json, time, signal
from neo4j import GraphDatabase

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))
DB = cfg["db_name"]

# Reconstructed exactly like tmpl_parser.py would emit (str_with == "").
PRIMARY = (
    "MATCH (sr:SigmaRule) WITH DISTINCT sr ORDER BY rand() LIMIT 50 "
    "MATCH (sr:SigmaRule), (sr:SigmaRule)-[:detects]->(ap:`attack-pattern`), "
    "(negap1:`attack-pattern`), (negap2:`attack-pattern`), "
    "(negap3:`attack-pattern`), (negap4:`attack-pattern`)     "
    "WHERE elementId(negap1) <> elementId(ap) AND "
    "elementId(negap2) <> elementId(ap) AND "
    "elementId(negap3) <> elementId(ap) AND "
    "elementId(negap4) <> elementId(ap) AND "
    "elementId(negap1) <> elementId(negap2) AND "
    "elementId(negap1) <> elementId(negap3) AND "
    "elementId(negap1) <> elementId(negap4) AND "
    "elementId(negap2) <> elementId(negap3) AND "
    "elementId(negap2) <> elementId(negap4) AND "
    "elementId(negap3) <> elementId(negap4)     "
    "RETURN DISTINCT sr.id, sr.title, sr.description, ap.mitre_id, ap.name, "
    "negap1.mitre_id, negap1.name, negap2.mitre_id, negap2.name, "
    "negap3.mitre_id, negap3.name, negap4.mitre_id, negap4.name     "
    "ORDER BY rand()     LIMIT 50"
)

FALLBACK = (
    "MATCH (sr:SigmaRule) WITH DISTINCT sr ORDER BY rand() LIMIT 50 "
    "MATCH (sr:SigmaRule), (sr:SigmaRule)-[:detects]->(ap:`attack-pattern`), "
    "(negap1:`attack-pattern`), (negap2:`attack-pattern`), "
    "(negap3:`attack-pattern`), (negap4:`attack-pattern`)     "
    "WHERE elementId(negap1) <> elementId(ap) AND "
    "elementId(negap2) <> elementId(ap) AND "
    "elementId(negap3) <> elementId(ap) AND "
    "elementId(negap4) <> elementId(ap) AND "
    "elementId(negap1) <> elementId(negap2) AND "
    "elementId(negap1) <> elementId(negap3) AND "
    "elementId(negap1) <> elementId(negap4) AND "
    "elementId(negap2) <> elementId(negap3) AND "
    "elementId(negap2) <> elementId(negap4) AND "
    "elementId(negap3) <> elementId(negap4)     "
    "LIMIT 50     RETURN DISTINCT sr.id, sr.title, sr.description, "
    "ap.mitre_id, ap.name, negap1.mitre_id, negap1.name, negap2.mitre_id, "
    "negap2.name, negap3.mitre_id, negap3.name, negap4.mitre_id, negap4.name     "
    "ORDER BY rand()"
)

def try_query(label, q, timeout=20.0):
    print(f"\n========== {label} (timeout {timeout}s) ==========")
    print(f"query (first 200): {q[:200]}")
    t0 = time.time()
    try:
        with drv.session(database=DB) as s:
            tx = s.begin_transaction(timeout=timeout)
            try:
                rs = list(tx.run(q))
                tx.commit()
                print(f"  OK   rows={len(rs)}  elapsed={time.time()-t0:.2f}s")
            except Exception as e:
                try: tx.close()
                except Exception: pass
                raise
    except Exception as e:
        kind = type(e).__name__
        msg = str(e).splitlines()[0][:200]
        print(f"  FAIL {kind}: {msg}  elapsed={time.time()-t0:.2f}s")

# Production count_limit is 250; rerun both at 250 to reproduce the build.
PRIMARY_250  = PRIMARY.replace("LIMIT 50", "LIMIT 250")
FALLBACK_250 = FALLBACK.replace("LIMIT 50", "LIMIT 250")
try_query("FALLBACK @50",  FALLBACK,     timeout=30.0)
try_query("FALLBACK @250", FALLBACK_250, timeout=120.0)
drv.close()
