"""Probe substrate yield for AB.TAA.NEG.1 / JS.TAA.NEG.1 under per-primary
grouping vs legacy form. Establishes whether the v10>v11 raw-row drop
(1500->1106 and 500->378) is due to per_primary_grouping=true or a hard
substrate ceiling we can't recover from."""
import json
from neo4j import GraphDatabase

cfg = json.load(open('tmpl_gen/data_generation/neo4j-local-config.json'))
d = GraphDatabase.driver(cfg['uri'], auth=tuple(cfg['auth']), database=cfg.get('db_name'))

LIMIT_AB = 1500
LIMIT_JS = 500

q_per_primary = (
    "MATCH (grp:`intrusion-set`) "
    "WITH collect(DISTINCT grp) AS _allprim, count(DISTINCT grp) AS _nprim "
    "UNWIND range(1, $limit) AS _dup_i "
    "WITH _allprim[toInteger(rand() * _nprim)] AS grp "
    "CALL (grp) { "
    "  MATCH (grp)-[:uses]->(ap1:`attack-pattern`) "
    "  WITH grp, ap1 ORDER BY rand() LIMIT 1 "
    "  MATCH (grp)-[:uses]->(ap2:`attack-pattern`) "
    "  WHERE elementId(ap2) <> elementId(ap1) "
    "  WITH grp, ap1, ap2 ORDER BY rand() LIMIT 1 "
    "  MATCH (grp)-[:uses]->(mw:malware) "
    "  WITH grp, ap1, ap2, mw ORDER BY rand() LIMIT 1 "
    "  MATCH (rel:`intrusion-set`)-[:uses]->(ap1) "
    "  WHERE rel.mitre_id <> grp.mitre_id "
    "  MATCH (rel)-[:uses]->(ap2) "
    "  MATCH (rel)-[:uses]->(mw) "
    "  WITH grp, ap1, ap2, mw, rel ORDER BY rand() LIMIT 1 "
    "  RETURN ap1, ap2, mw, rel "
    "} "
    "RETURN count(*) AS n"
)

q_legacy = (
    "MATCH (grp:`intrusion-set`)-[:uses]->(ap1:`attack-pattern`), "
    "      (grp)-[:uses]->(ap2:`attack-pattern`), "
    "      (grp)-[:uses]->(mw:malware), "
    "      (rel:`intrusion-set`)-[:uses]->(ap1), "
    "      (rel)-[:uses]->(ap2), "
    "      (rel)-[:uses]->(mw) "
    "WHERE elementId(ap1) <> elementId(ap2) AND rel.mitre_id <> grp.mitre_id "
    "RETURN DISTINCT grp.mitre_id, ap1.mitre_id, ap2.mitre_id, mw.mitre_id, rel.mitre_id "
    "ORDER BY rand() LIMIT $limit"
)

print("=== AB.TAA.NEG.1 substrate probe ===")
print(f"target LIMIT: {LIMIT_AB}")

print("\n[a] per-primary grouping (v11 default):")
with d.session() as s:
    for row in s.run(q_per_primary, limit=LIMIT_AB, timeout=180):
        print(f"  yield: {dict(row)}")

print("\n[b] legacy form (v10 default):")
with d.session() as s:
    n = 0
    for row in s.run(q_legacy, limit=LIMIT_AB, timeout=180):
        n += 1
    print(f"  yield: {n} (target: {LIMIT_AB})")

print(f"\n=== JS.TAA.NEG.1 substrate probe ===")
print(f"target LIMIT: {LIMIT_JS}")

print("\n[c] per-primary grouping (v11 default):")
with d.session() as s:
    for row in s.run(q_per_primary, limit=LIMIT_JS, timeout=180):
        print(f"  yield: {dict(row)}")

print("\n[d] legacy form (v10 default):")
with d.session() as s:
    n = 0
    for row in s.run(q_legacy, limit=LIMIT_JS, timeout=180):
        n += 1
    print(f"  yield: {n} (target: {LIMIT_JS})")

print("\n=== Summary ===")
print("Substrate ceiling probes complete.")
print("If [b] >= LIMIT_AB and [d] >= LIMIT_JS, switching these two")
print("templates to per_primary_grouping=false fully recovers raw yield.")

d.close()
