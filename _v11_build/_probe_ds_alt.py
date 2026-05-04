#!/usr/bin/env python3
"""Check whether attack-pattern carries x_mitre_data_sources text and
how many techniques cite each data source name, so SOC.TRIAGE.DS.1
can join via property text instead of a missing edge."""
import json
from neo4j import GraphDatabase
cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

queries = [
    ("attack-pattern keys (filter to data-source-ish)",
     "MATCH (n:`attack-pattern`) RETURN [k IN keys(n) WHERE toLower(k) CONTAINS 'data'] AS k LIMIT 1"),
    ("attack-pattern w/ x_mitre_data_sources count",
     "MATCH (n:`attack-pattern`) WHERE n.x_mitre_data_sources IS NOT NULL RETURN count(n) AS c"),
    ("sample x_mitre_data_sources values",
     "MATCH (n:`attack-pattern`) WHERE n.x_mitre_data_sources IS NOT NULL "
     "RETURN n.mitre_id AS aid, n.name AS aname, n.x_mitre_data_sources AS ds LIMIT 3"),
    ("data-source name vs x_mitre_data_sources cross-ref",
     "MATCH (ds:`x-mitre-data-source`) "
     "OPTIONAL MATCH (ap:`attack-pattern`) WHERE ANY(s IN ap.x_mitre_data_sources WHERE s STARTS WITH ds.name) "
     "RETURN ds.name AS ds, ds.mitre_id AS dsmid, count(DISTINCT ap) AS ap_count "
     "ORDER BY ap_count DESC LIMIT 10"),
    ("dc keys + sample (does dc.name match ds.name?)",
     "MATCH (dc:`x-mitre-data-component`) RETURN dc.name AS dcname, dc.mitre_id AS dcmid, substring(dc.description,0,120) AS d LIMIT 5"),
    ("ds.name list",
     "MATCH (n:`x-mitre-data-source`) RETURN n.name AS name, n.mitre_id AS mid ORDER BY n.name LIMIT 50"),
    ("sigma analytic-strategy combined coverage",
     "MATCH (det:`x-mitre-detection-strategy`)-[:detects]->(ap:`attack-pattern`) "
     "RETURN count(DISTINCT det) AS dets, count(DISTINCT ap) AS aps"),
    ("d3fend counters distinct attack-patterns + tactic distrib",
     "MATCH (d:D3FENDTechnique)-[r:counters]->(ap:`attack-pattern`) "
     "RETURN count(DISTINCT d) AS d3, count(DISTINCT ap) AS aps, count(*) AS edges"),
    ("d3fend tactic counts on counters edges",
     "MATCH (d:D3FENDTechnique)-[r:counters]->(:`attack-pattern`) "
     "RETURN r.def_tactic AS tac, count(*) AS c ORDER BY c DESC LIMIT 10"),
    ("malware count w/ x_mitre_aliases",
     "MATCH (n:malware) WHERE n.x_mitre_aliases IS NOT NULL AND size(n.x_mitre_aliases) > 0 RETURN count(n) AS c"),
    ("malware sample aliases",
     "MATCH (n:malware) WHERE n.x_mitre_aliases IS NOT NULL AND size(n.x_mitre_aliases) > 1 "
     "RETURN n.mitre_id AS mid, n.name AS name, n.x_mitre_aliases AS aliases LIMIT 3"),
    ("intrusion-set keys for aliases",
     "MATCH (n:`intrusion-set`) RETURN [k IN keys(n) WHERE toLower(k) CONTAINS 'alias'] AS k LIMIT 1"),
    ("intrusion-set w/ aliases count",
     "MATCH (n:`intrusion-set`) WHERE n.x_mitre_aliases IS NOT NULL AND size(n.x_mitre_aliases) > 1 RETURN count(n) AS c"),
    ("intrusion-set sample aliases",
     "MATCH (n:`intrusion-set`) WHERE n.x_mitre_aliases IS NOT NULL AND size(n.x_mitre_aliases) > 1 "
     "RETURN n.mitre_id AS mid, n.name AS name, n.x_mitre_aliases AS aliases LIMIT 3"),
]

with d.session(database=cfg["db_name"]) as s:
    for label, q in queries:
        print(f"=== {label} ===")
        try:
            for row in s.run(q):
                for k, v in dict(row).items():
                    sv = str(v)
                    print(f"  {k}: {sv[:300]}")
        except Exception as e:
            print(f"  ERR: {e}")
d.close()
