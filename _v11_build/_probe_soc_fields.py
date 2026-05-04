#!/usr/bin/env python3
"""Probe Neo4j to confirm node properties + relationships used by the
v11 SOC.* templates so the manifest binds against real fields."""
import json
from neo4j import GraphDatabase
cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

queries = [
    ("malware sample keys", "MATCH (n:malware) WHERE n.description IS NOT NULL RETURN keys(n) AS k LIMIT 1"),
    ("malware count w/ desc", "MATCH (n:malware) WHERE n.description IS NOT NULL AND n.description <> '' RETURN count(n) AS c"),
    ("xm-analytic keys", "MATCH (n:`x-mitre-analytic`) RETURN keys(n) AS k LIMIT 1"),
    ("xm-analytic count", "MATCH (n:`x-mitre-analytic`) RETURN count(n) AS c"),
    ("xm-data-source keys", "MATCH (n:`x-mitre-data-source`) RETURN keys(n) AS k LIMIT 1"),
    ("xm-data-source count", "MATCH (n:`x-mitre-data-source`) RETURN count(n) AS c"),
    ("xm-data-comp keys", "MATCH (n:`x-mitre-data-component`) RETURN keys(n) AS k LIMIT 1"),
    ("D3FENDTechnique keys", "MATCH (n:D3FENDTechnique) WHERE (n)-[:counters]->() RETURN keys(n) AS k LIMIT 1"),
    ("d3 counters rel keys", "MATCH (d:D3FENDTechnique)-[r:counters]->(:`attack-pattern`) RETURN keys(r) AS k LIMIT 1"),
    ("an->dc rel direction", "MATCH (a:`x-mitre-analytic`)-[r]->(b) RETURN type(r) AS t, labels(b) AS bl, count(*) AS c LIMIT 5"),
    ("an<-? rel direction", "MATCH (a:`x-mitre-analytic`)<-[r]-(b) RETURN type(r) AS t, labels(b) AS bl, count(*) AS c LIMIT 5"),
    ("ds<-dc rel", "MATCH (ds:`x-mitre-data-source`)<-[r]-(b) RETURN type(r) AS t, labels(b) AS bl, count(*) AS c LIMIT 5"),
    ("ds->? rel", "MATCH (ds:`x-mitre-data-source`)-[r]->(b) RETURN type(r) AS t, labels(b) AS bl, count(*) AS c LIMIT 5"),
    ("dc->ap rel", "MATCH (dc:`x-mitre-data-component`)-[r]->(b) RETURN type(r) AS t, labels(b) AS bl, count(*) AS c LIMIT 5"),
    ("dc<-ap rel", "MATCH (dc:`x-mitre-data-component`)<-[r]-(b) RETURN type(r) AS t, labels(b) AS bl, count(*) AS c LIMIT 5"),
    ("malware sample row", "MATCH (n:malware) WHERE n.description IS NOT NULL AND n.description <> '' RETURN n.id AS id, n.mitre_id AS mid, n.name AS name, substring(n.description, 0, 200) AS desc, n.aliases AS aliases LIMIT 1"),
    ("xma sample row", "MATCH (n:`x-mitre-analytic`) RETURN n.id AS id, n.name AS name, substring(coalesce(n.description, ''), 0, 200) AS desc LIMIT 2"),
    ("xmds sample row", "MATCH (n:`x-mitre-data-source`) RETURN n.id AS id, n.name AS name, n.mitre_id AS mid, substring(coalesce(n.description, ''), 0, 200) AS desc LIMIT 2"),
    ("d3 sample counter", "MATCH (d:D3FENDTechnique)-[r:counters]->(a:`attack-pattern`) RETURN d.d3fend_id AS did, d.name AS dname, substring(d.definition, 0, 150) AS ddef, a.mitre_id AS aid, a.name AS aname, r.def_artifact AS da, r.def_artifact_rel AS dar, r.off_artifact AS oa, r.off_artifact_rel AS oar LIMIT 2"),
    ("sigma fp sample", "MATCH (n:SigmaRule) WHERE n.falsepositives IS NOT NULL AND n.falsepositives <> '' RETURN n.id AS id, n.title AS title, n.falsepositives AS fp LIMIT 1"),
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
