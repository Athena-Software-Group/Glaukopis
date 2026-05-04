"""Quick Neo4j probe to understand AB.MS.GRP.1 expected yield."""
import json, sys
from neo4j import GraphDatabase
cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))
with drv.session(database=cfg["db_name"]) as s:
    queries = [
        ("MATCH (g:`intrusion-set`) RETURN count(g) AS n", "intrusion-set count"),
        ("MATCH (g:`intrusion-set`)-[:uses]->(a:`attack-pattern`) RETURN count(DISTINCT g) AS gn, count(*) AS ap_total", "grp w/ ap edges"),
        ("MATCH (g:`intrusion-set`)-[:uses]->(a:`attack-pattern`) WITH g, count(a) AS k WHERE k >= 2 RETURN count(g) AS n, min(k) AS min_k, max(k) AS max_k, avg(k) AS avg_k", "grp eligible for AB.MS.GRP.1"),
        ("MATCH (a:`attack-pattern`) RETURN count(a) AS n_ap", "attack-pattern total"),
        ("MATCH (m:`course-of-action`) WHERE m.mitre_id STARTS WITH 'M1' RETURN count(m) AS n", "M-controls"),
    ]
    for q, label in queries:
        rec = s.run(q).single()
        print(f"{label:<32}  {dict(rec)}")
drv.close()
