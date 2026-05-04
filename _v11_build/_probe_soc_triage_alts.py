#!/usr/bin/env python3
"""Probe non-eval-contaminated seed sources for SOC.TRIAGE-style templates."""
import json
from neo4j import GraphDatabase

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

with d.session(database=cfg["db_name"]) as s:
    print("--- (1) Sigma falsepositives field availability ---")
    r = s.run(
        "MATCH (s:SigmaRule) "
        "RETURN count(s) AS total, "
        "sum(CASE WHEN s.falsepositives IS NOT NULL AND size(s.falsepositives) > 0 THEN 1 ELSE 0 END) AS with_fp"
    )
    for row in r:
        print(" ", dict(row))

    print("--- (2) attack-pattern x_mitre_detection field (per-technique analyst hints) ---")
    r = s.run("MATCH (a:`attack-pattern`) WITH keys(a) AS k UNWIND k AS p RETURN DISTINCT p ORDER BY p")
    keys = [row["p"] for row in r]
    print("  keys:", keys)
    if "x_mitre_detection" in keys:
        r = s.run(
            "MATCH (a:`attack-pattern`) "
            "RETURN count(a) AS total, "
            "sum(CASE WHEN a.x_mitre_detection IS NOT NULL AND a.x_mitre_detection <> '' THEN 1 ELSE 0 END) AS with_det"
        )
        for row in r:
            print(" ", dict(row))

    print("--- (3) x-mitre-data-source / data-component nodes (alert-source reasoning) ---")
    for label in ("x-mitre-data-source", "x-mitre-data-component", "x-mitre-detection-strategy", "x-mitre-analytic"):
        r = s.run(f"MATCH (n:`{label}`) RETURN count(n) AS n")
        for row in r:
            print(f"  {label}: {row['n']}")

    print("--- (4) Detection_Method label (early sigma->ATT&CK style)? ---")
    r = s.run("MATCH (n:Detection_Method) RETURN count(n) AS n")
    for row in r:
        print(" ", dict(row))
    r = s.run("MATCH (n:Detection_Method) RETURN n LIMIT 1")
    for row in r:
        sample = dict(row["n"])
        for k, v in sample.items():
            sv = str(v)
            print(f"   {k}: {sv[:100]}{'...' if len(sv) > 100 else ''}")

d.close()
