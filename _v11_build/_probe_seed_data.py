#!/usr/bin/env python3
"""One-shot probe: what alias / sigma / soc / cm seed data is available."""
import json
import os
from neo4j import GraphDatabase

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

with d.session(database=cfg["db_name"]) as s:
    print("=== intrusion-set property keys ===")
    r = s.run("MATCH (g:`intrusion-set`) WITH keys(g) AS k UNWIND k AS p RETURN DISTINCT p ORDER BY p")
    for row in r:
        print(" ", row["p"])

    print("\n=== sample intrusion-set rows w/ alias-shaped props ===")
    r = s.run(
        "MATCH (g:`intrusion-set`) "
        "WHERE g.aliases IS NOT NULL OR g.x_mitre_aliases IS NOT NULL OR g.synonyms IS NOT NULL "
        "RETURN g.name AS name, g.aliases AS a, g.x_mitre_aliases AS xa, g.synonyms AS syn LIMIT 5"
    )
    for row in r:
        print(" ", dict(row))

    print("\n=== node label inventory ===")
    r = s.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
    for row in r:
        print(" ", row["label"])

d.close()

print("\n=== athena_cti_db/utils/threat_data/sigma contents ===")
sd = "athena_cti_db/utils/threat_data/sigma"
if os.path.isdir(sd):
    for root, dirs, files in os.walk(sd):
        depth = root[len(sd):].count(os.sep)
        if depth > 2:
            continue
        print(f"  {root}/  ({len(files)} files)")
        if depth == 2 and files:
            print(f"    e.g. {files[:3]}")

print("\n=== fetch_cybersoceval_data.py purpose ===")
with open("SFT/test/utils/fetch_cybersoceval_data.py") as f:
    head = "".join(f.readlines()[:40])
print(head)
