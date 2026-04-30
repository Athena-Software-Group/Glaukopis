#!/usr/bin/env python
"""Inspect schema around detection-strategy / analytic / data-component
to determine the right tmpl syntax for P.7's `{ds:ap.detects<...}` rule."""
import json
from neo4j import GraphDatabase
cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

with drv.session(database=cfg["db_name"]) as s:
    print("--- node labels containing 'detect' or 'analytic' or 'data' ---")
    for r in s.run("CALL db.labels() YIELD label RETURN label ORDER BY label"):
        l = r["label"]
        if any(x in l.lower() for x in ("detect", "analytic", "data")):
            print(f"  {l}")
    print()
    print("--- detection-strategy outgoing edges ---")
    for r in s.run(
        "MATCH (n:`x-mitre-detection-strategy`)-[r]->(t) "
        "RETURN type(r) AS rt, labels(t)[0] AS dst, count(*) AS c "
        "ORDER BY c DESC LIMIT 15"):
        print(f"  (x-mitre-detection-strategy)-[:{r['rt']}]->(:{r['dst']}) x{r['c']}")
    print()
    print("--- detection-strategy incoming edges ---")
    for r in s.run(
        "MATCH (n:`x-mitre-detection-strategy`)<-[r]-(t) "
        "RETURN type(r) AS rt, labels(t)[0] AS src, count(*) AS c "
        "ORDER BY c DESC LIMIT 15"):
        print(f"  (x-mitre-detection-strategy)<-[:{r['rt']}]-(:{r['src']}) x{r['c']}")
    print()
    print("--- attack-pattern -[detects]- edges  ---")
    for r in s.run(
        "MATCH (n:`attack-pattern`)-[r:detects]-(t) "
        "RETURN type(r) AS rt, labels(t)[0] AS lbl, count(*) AS c"):
        print(f"  attack-pattern-detects: {r}")
    for r in s.run(
        "MATCH (n:`attack-pattern`)<-[r:detects]-(t) "
        "RETURN type(r) AS rt, labels(t)[0] AS lbl, count(*) AS c"):
        print(f"  ap<-[detects]-: {r}")
    print()
    print("--- analytic outgoing edges ---")
    for r in s.run(
        "MATCH (n:`x-mitre-analytic`)-[r]->(t) "
        "RETURN type(r) AS rt, labels(t)[0] AS dst, count(*) AS c "
        "ORDER BY c DESC LIMIT 15"):
        print(f"  (x-mitre-analytic)-[:{r['rt']}]->(:{r['dst']}) x{r['c']}")

drv.close()
