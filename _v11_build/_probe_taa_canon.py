"""Probe intrusion-set alias inventory to design TAA.CANON.1/2 templates.
Counts groups with aliases, expansion factor, and shows sample shape."""
import json, sys
sys.path.insert(0, "tmpl_gen/src")
from neo4j import GraphDatabase

cfg = json.loads(open("tmpl_gen/data_generation/neo4j-local-config.json").read())
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

def q(cy):
    with d.session(database=cfg.get("db_name", "neo4j")) as s:
        return [dict(r) for r in s.run(cy)]

print("=== intrusion-set node count + alias coverage ===")
for r in q("MATCH (g:`intrusion-set`) RETURN count(g) AS total"):
    print(" total intrusion-set:", r)
for r in q("MATCH (g:`intrusion-set`) WHERE g.aliases IS NOT NULL AND size(g.aliases) > 1 RETURN count(g) AS with_aliases"):
    print(" with >1 alias:", r)
for r in q("MATCH (g:`intrusion-set`) WHERE g.aliases IS NOT NULL RETURN avg(size(g.aliases)) AS avg_aliases, max(size(g.aliases)) AS max_aliases"):
    print(" alias size stats:", r)

print("\n=== sample aliases ===")
for r in q("MATCH (g:`intrusion-set`) WHERE size(g.aliases) >= 3 RETURN g.name AS name, g.aliases AS aliases LIMIT 5"):
    print(" ", r)

print("\n=== uses-attack-pattern coverage (for TAA.CANON.2) ===")
for r in q("MATCH (g:`intrusion-set`)-[:uses]->(ap:`attack-pattern`) RETURN count(DISTINCT g) AS groups, count(*) AS edges"):
    print(" ", r)

d.close()
