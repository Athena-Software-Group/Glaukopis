"""Probe Neo4j substrate cardinality for candidate scenario-MCQ angles
(v18p1 MCQ shortfall recovery; one-off, not part of the watcher).
"""
import json
import time
from pathlib import Path
from neo4j import GraphDatabase

CFG = json.loads(Path("tmpl_gen/data_generation/neo4j-local-config.json").read_text())
time.sleep(60)  # clear AuthenticationRateLimit window
drv = GraphDatabase.driver(CFG["uri"], auth=tuple(CFG["auth"]))
db = CFG["db_name"]


def q(c, label):
    with drv.session(database=db) as s:
        v = s.run(c).single()[0]
        print(f"  {label:60s} {v}")


print("=== node-label cardinality (scenario-MCQ candidates) ===")
for c, lbl in [
    ("MATCH (n:malware) RETURN count(n)", "malware nodes"),
    ("MATCH (n:tool) RETURN count(n)", "tool nodes"),
    ("MATCH (n) WHERE 'x-mitre-data-source' IN labels(n) RETURN count(n)", "data-source nodes"),
    ("MATCH (n) WHERE 'x-mitre-data-component' IN labels(n) RETURN count(n)", "data-component nodes"),
    ("MATCH (n:`intrusion-set`) RETURN count(n)", "intrusion-set (group) nodes"),
    ("MATCH (n:`attack-pattern`) RETURN count(n)", "attack-pattern (technique) nodes"),
    ("MATCH (n:`attack-pattern`) WHERE n.x_mitre_is_subtechnique=true RETURN count(n)", "sub-technique nodes"),
    ("MATCH (n:`course-of-action`) RETURN count(n)", "course-of-action (mitigation) nodes"),
    ("MATCH (n) WHERE 'x-mitre-tactic' IN labels(n) RETURN count(n)", "tactic nodes"),
    ("MATCH (n:campaign) RETURN count(n)", "campaign nodes"),
    ("MATCH (n:Weakness) RETURN count(n)", "Weakness (CWE) nodes"),
    ("MATCH (n:CAPEC) RETURN count(n)", "CAPEC nodes"),
    ("MATCH (n:CVE) RETURN count(n)", "CVE nodes"),
]:
    try:
        q(c, lbl)
    except Exception as e:
        print(f"  {lbl:60s} ERR: {type(e).__name__}: {str(e)[:80]}")

print()
print("=== edge cardinality (scenario-MCQ anchor candidates) ===")
for c, lbl in [
    ("MATCH (m:malware)-[:uses]->(t:`attack-pattern`) RETURN count(DISTINCT m)", "DISTINCT malware that USES technique"),
    ("MATCH (m:malware)-[:uses]->(t:`attack-pattern`) RETURN count(*)", "(malware,technique) USES edges"),
    ("MATCH (g:`intrusion-set`)-[:uses]->(m:malware) RETURN count(DISTINCT m)", "DISTINCT malware USED-BY group"),
    ("MATCH (g:`intrusion-set`)-[:uses]->(m:malware) RETURN count(*)", "(group,malware) USES edges"),
    ("MATCH (t:tool)-[:uses]->(ap:`attack-pattern`) RETURN count(DISTINCT t)", "DISTINCT tool that USES technique"),
    ("MATCH (g:`intrusion-set`)-[:uses]->(t:tool) RETURN count(DISTINCT t)", "DISTINCT tool USED-BY group"),
    ("MATCH (ds)-[:detects]->(ap:`attack-pattern`) WHERE 'x-mitre-data-source' IN labels(ds) RETURN count(DISTINCT ds)", "DISTINCT data-source that DETECTS technique"),
    ("MATCH (ap:`attack-pattern`)-[:`subtechnique-of`]->(par:`attack-pattern`) RETURN count(*)", "sub-technique -> parent edges"),
    ("MATCH (c:campaign)-[:`attributed-to`]->(g:`intrusion-set`) RETURN count(DISTINCT c)", "DISTINCT campaign attributed-to group"),
    ("MATCH (c:campaign)-[:uses]->(t:`attack-pattern`) RETURN count(DISTINCT c)", "DISTINCT campaign USES technique"),
    ("MATCH (ap:`attack-pattern`)-[:`mitigated-by`]->(coa:`course-of-action`) RETURN count(*)", "(technique,mitigation) edges"),
]:
    try:
        q(c, lbl)
    except Exception as e:
        print(f"  {lbl:60s} ERR: {type(e).__name__}: {str(e)[:80]}")

print()
print("=== sample malware/tool/group description sizes (memory check) ===")
with drv.session(database=db) as s:
    for r in s.run("MATCH (n:malware) WHERE n.description IS NOT NULL RETURN max(size(n.description)) AS mx, avg(size(n.description)) AS av, count(*) AS n"):
        print(f"  malware  desc max={r['mx']:>6}  avg={int(r['av'] or 0):>5}  n={r['n']}")
    for r in s.run("MATCH (n:tool) WHERE n.description IS NOT NULL RETURN max(size(n.description)) AS mx, avg(size(n.description)) AS av, count(*) AS n"):
        print(f"  tool     desc max={r['mx']:>6}  avg={int(r['av'] or 0):>5}  n={r['n']}")
    for r in s.run("MATCH (n:`intrusion-set`) WHERE n.description IS NOT NULL RETURN max(size(n.description)) AS mx, avg(size(n.description)) AS av, count(*) AS n"):
        print(f"  group    desc max={r['mx']:>6}  avg={int(r['av'] or 0):>5}  n={r['n']}")

drv.close()
