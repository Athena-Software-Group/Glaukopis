#!/usr/bin/env python3
"""End-to-end test for the new D3FEND ingest path in populate_neo4j_complete.py.

Reuses the already-downloaded d3fend.json + d3fend-full-mappings.json from
/tmp/d3fend_probe/ to skip the network step. Runs process_d3fend_data() and
then execute_queries() against the live athena-cti-db to verify the new
D3FENDTactic + D3FENDTechnique nodes and counters/parent edges land cleanly.
"""
import json
import os
import sys
import shutil
from pathlib import Path

# Point to the live local Neo4j before importing the populate module
cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
host_port = cfg["uri"].replace("bolt://", "")
os.environ["NEO4J_URL"] = f"bolt://{host_port}"
os.environ["NEO4J_USER"] = cfg["auth"][0]
os.environ["NEO4J_PASSWORD"] = cfg["auth"][1]
os.environ["NEO4J_DB"] = cfg["db_name"]

sys.path.insert(0, "athena_cti_db/threat_framework")
import populate_neo4j_complete as pn

# Stage cached files into a fresh dir so process_d3fend_data sees them
test_dir = Path("/tmp/d3fend_test_ingest")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)
for f in ("d3fend.json", "d3fend-full-mappings.json"):
    src = Path("/tmp/d3fend_probe") / f
    if not src.exists():
        print(f"ERROR: cached file missing {src}; run probe first")
        sys.exit(1)
    shutil.copy(src, test_dir / f)
print(f"--- staged d3fend cache to {test_dir} ---")

# Generate queries
print("--- running process_d3fend_data ---")
queries = pn.process_d3fend_data(test_dir)
print(f"  total queries generated: {len(queries):,}")

# Sample
from collections import Counter
kinds = Counter()
for q in queries:
    s = q["statement"]
    if "create_node_query" in s or "MERGE (n:" in s and "stix_id" in s:
        kinds["node_create"] += 1
    elif "MERGE (c)-[:parent]->" in s:
        kinds["parent_edge"] += 1
    elif "MERGE (d)-[r:counters]->" in s:
        kinds["counters_edge"] += 1
    else:
        kinds["other"] += 1
print(f"  by kind: {dict(kinds)}")

# Add d3fend constraints first (idempotent)
print("--- ensuring D3FEND constraints exist ---")
constraints = [
    "CREATE CONSTRAINT stix_id_d3fend_technique IF NOT EXISTS FOR (n:D3FENDTechnique) REQUIRE n.stix_id IS UNIQUE",
    "CREATE CONSTRAINT d3fend_technique_id IF NOT EXISTS FOR (n:D3FENDTechnique) REQUIRE n.d3fend_id IS UNIQUE",
    "CREATE CONSTRAINT stix_id_d3fend_tactic IF NOT EXISTS FOR (n:D3FENDTactic) REQUIRE n.stix_id IS UNIQUE",
    "CREATE CONSTRAINT d3fend_tactic_id IF NOT EXISTS FOR (n:D3FENDTactic) REQUIRE n.d3fend_id IS UNIQUE",
]
for c in constraints:
    try:
        pn.execute_queries([{"statement": c}])
    except Exception as e:
        print(f"  constraint warn: {e}")

# Execute the ingest queries
print("--- executing ingest against athena-cti-db ---")
pn.execute_queries(queries)

# Verify
print("--- verifying ---")
from neo4j import GraphDatabase
d = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))
with d.session(database=cfg["db_name"]) as s:
    for q, label in [
        ("MATCH (t:D3FENDTactic) RETURN count(t) AS n", "D3FENDTactic"),
        ("MATCH (t:D3FENDTechnique) RETURN count(t) AS n", "D3FENDTechnique"),
        ("MATCH (:D3FENDTechnique)-[:parent]->() RETURN count(*) AS n", "parent edges"),
        ("MATCH (:D3FENDTechnique)-[:counters]->(:`attack-pattern`) RETURN count(*) AS n", "counters edges"),
    ]:
        for row in s.run(q):
            print(f"  {label}: {row['n']}")
    print("--- sample D3FENDTechnique ---")
    for row in s.run("MATCH (t:D3FENDTechnique) RETURN t LIMIT 1"):
        for k, v in dict(row["t"]).items():
            sv = str(v)
            print(f"   {k}: {sv[:120]}{'...' if len(sv) > 120 else ''}")
    print("--- sample counters edge w/ context ---")
    for row in s.run(
        "MATCH (d:D3FENDTechnique)-[r:counters]->(a:`attack-pattern`) "
        "RETURN d.d3fend_id AS d3id, d.name AS d3name, "
        "a.mitre_id AS mid, a.name AS aname, "
        "r.def_artifact AS def_art, r.def_artifact_rel AS def_rel, "
        "r.off_artifact_rel AS off_rel, r.off_artifact AS off_art "
        "LIMIT 3"
    ):
        print("  ", dict(row))
d.close()
