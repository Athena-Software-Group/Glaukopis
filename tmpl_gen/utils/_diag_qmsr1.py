#!/usr/bin/env python
"""Diagnose Q.MSR.1 cartesian explosion: count node populations and try
running the actual generated query with a hard timeout to confirm
whether neo4j enforces it."""
import json, time
from neo4j import GraphDatabase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tmpl_gen"))
from neo4j_env import neo4j_auth  # credentials from NEO4J_USER/NEO4J_PASSWORD or .env

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=neo4j_auth(cfg))
DB_NAME = cfg["db_name"]

QUERIES = [
    ("SigmaRule count",
     "MATCH (n:SigmaRule) RETURN count(n) AS c"),
    ("attack-pattern count",
     "MATCH (n:`attack-pattern`) RETURN count(n) AS c"),
    ("attack-pattern with mitre_id",
     "MATCH (n:`attack-pattern`) WHERE n.mitre_id IS NOT NULL RETURN count(n) AS c"),
    ("SigmaRule -[:detects]-> attack-pattern edges",
     "MATCH (sr:SigmaRule)-[:detects]->(ap:`attack-pattern`) RETURN count(*) AS c"),
]

with drv.session(database=DB_NAME) as s:
    for label, q in QUERIES:
        print(f"--- {label} ---\n  {q}")
        print(f"  -> {s.run(q).single()['c']}")
    print()

drv.close()
