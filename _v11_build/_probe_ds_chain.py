#!/usr/bin/env python3
"""Verify x-mitre-data-source connectivity + detection-strategy chain
so SOC.TRIAGE.DS.1 can bind against a real path."""
import json
from neo4j import GraphDatabase
cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

queries = [
    ("all rels touching x-mitre-data-source",
     "MATCH (a:`x-mitre-data-source`)-[r]-(b) "
     "RETURN type(r) AS t, labels(b) AS bl, count(*) AS c "
     "ORDER BY c DESC LIMIT 10"),
    ("dc node sample with all rels",
     "MATCH (dc:`x-mitre-data-component`)-[r]-(b) "
     "RETURN type(r) AS t, labels(b) AS bl, count(*) AS c "
     "ORDER BY c DESC LIMIT 10"),
    ("dc -> ds rel sample",
     "MATCH (dc:`x-mitre-data-component`)-[r]->(ds:`x-mitre-data-source`) "
     "RETURN type(r) AS t, count(*) AS c LIMIT 5"),
    ("dc <- ds rel sample",
     "MATCH (dc:`x-mitre-data-component`)<-[r]-(ds:`x-mitre-data-source`) "
     "RETURN type(r) AS t, count(*) AS c LIMIT 5"),
    ("detection-strategy keys",
     "MATCH (n:`x-mitre-detection-strategy`) RETURN keys(n) AS k LIMIT 1"),
    ("detection-strategy count",
     "MATCH (n:`x-mitre-detection-strategy`) RETURN count(n) AS c"),
    ("detection-strategy rels",
     "MATCH (n:`x-mitre-detection-strategy`)-[r]-(b) "
     "RETURN type(r) AS t, labels(b) AS bl, count(*) AS c "
     "ORDER BY c DESC LIMIT 10"),
    ("end-to-end ds chain",
     "MATCH (ds:`x-mitre-data-source`)<-[r1]-(dc:`x-mitre-data-component`)"
     "<-[:requires_data]-(an:`x-mitre-analytic`)"
     "<-[:implemented_by]-(det:`x-mitre-detection-strategy`)"
     "-[:detects]->(ap:`attack-pattern`) "
     "RETURN type(r1) AS r1, ds.name AS ds, dc.name AS dc, "
     "an.name AS an, det.name AS det, ap.mitre_id AS ap_mid, ap.name AS ap_name "
     "LIMIT 2"),
    ("end-to-end ds chain reverse rel direction",
     "MATCH (ds:`x-mitre-data-source`)-[r1]->(dc:`x-mitre-data-component`)"
     "<-[:requires_data]-(an:`x-mitre-analytic`)"
     "<-[:implemented_by]-(det:`x-mitre-detection-strategy`)"
     "-[:detects]->(ap:`attack-pattern`) "
     "RETURN type(r1) AS r1, ds.name AS ds, dc.name AS dc, "
     "an.name AS an, det.name AS det, ap.mitre_id AS ap_mid, ap.name AS ap_name "
     "LIMIT 2"),
]

with d.session(database=cfg["db_name"]) as s:
    for label, q in queries:
        print(f"=== {label} ===")
        try:
            for row in s.run(q):
                for k, v in dict(row).items():
                    sv = str(v)
                    print(f"  {k}: {sv[:240]}")
        except Exception as e:
            print(f"  ERR: {e}")
d.close()
