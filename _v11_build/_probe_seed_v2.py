#!/usr/bin/env python3
"""Probe TAA.CANON / SOC.SIGMA / SOC.MAL feasibility from current Neo4j state."""
import json
from neo4j import GraphDatabase

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

with d.session(database=cfg["db_name"]) as s:
    print("--- intrusion-set alias counts ---")
    r = s.run(
        "MATCH (g:`intrusion-set`) WHERE g.aliases IS NOT NULL AND size(g.aliases) > 1 "
        "RETURN count(g) AS groups_w_multi_alias, sum(size(g.aliases) - 1) AS total_noncanon_aliases"
    )
    for row in r:
        print(" ", dict(row))

    print("--- SigmaRule count + key inventory ---")
    r = s.run("MATCH (s:SigmaRule) RETURN count(s) AS n")
    for row in r:
        print("  total:", row["n"])
    r = s.run("MATCH (s:SigmaRule) WITH keys(s) AS k UNWIND k AS p RETURN DISTINCT p ORDER BY p")
    print("  keys:", [row["p"] for row in r])

    print("--- SigmaRule sample row ---")
    r = s.run("MATCH (s:SigmaRule) RETURN s LIMIT 1")
    for row in r:
        sample = dict(row["s"])
        for k, v in sample.items():
            sv = str(v)
            print(f"   {k}: {sv[:140]}{'...' if len(sv) > 140 else ''}")

    print("--- SigmaRule -> attack-pattern edges (for SOC.SIGMA reasoning) ---")
    r = s.run("MATCH (s:SigmaRule)-[r]->(t) RETURN type(r) AS rel, labels(t) AS to_label, count(*) AS n ORDER BY n DESC LIMIT 10")
    for row in r:
        print(" ", dict(row))

    print("--- malware: count + with description ---")
    r = s.run(
        "MATCH (m:malware) "
        "RETURN count(m) AS total, "
        "sum(CASE WHEN m.description IS NOT NULL AND m.description <> '' THEN 1 ELSE 0 END) AS with_desc"
    )
    for row in r:
        print(" ", dict(row))

d.close()
