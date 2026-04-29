import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ["NEO4J_URL"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
with d.session(database=os.environ["NEO4J_DB"]) as s:
    print("=== Weakness -> CAPEC ===")
    for r in s.run("MATCH (a:Weakness)-[r]->(b:CAPEC) RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"):
        print(" ", r["t"], r["c"])
    print("=== CAPEC -> Weakness (reverse) ===")
    for r in s.run("MATCH (a:CAPEC)-[r]->(b:Weakness) RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"):
        print(" ", r["t"], r["c"])
    print("=== CAPEC -> attack-pattern ===")
    for r in s.run("MATCH (a:CAPEC)-[r]->(b:`attack-pattern`) RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"):
        print(" ", r["t"], r["c"])
    print("=== attack-pattern -> CAPEC (reverse) ===")
    for r in s.run("MATCH (a:`attack-pattern`)-[r]->(b:CAPEC) RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"):
        print(" ", r["t"], r["c"])
    print("=== sample CAPEC keys ===")
    for r in s.run("MATCH (c:CAPEC) RETURN keys(c) AS k LIMIT 1"):
        print(" ", r["k"])
    print("=== sample Weakness keys ===")
    for r in s.run("MATCH (w:Weakness) RETURN keys(w) AS k LIMIT 1"):
        print(" ", r["k"])
    print("=== Weakness any outgoing rels ===")
    for r in s.run("MATCH (w:Weakness)-[r]->(x) RETURN type(r) AS t, labels(x) AS lbl, count(*) AS c ORDER BY c DESC LIMIT 20"):
        print(" ", r["t"], r["lbl"], r["c"])
    print("=== CAPEC any outgoing rels ===")
    for r in s.run("MATCH (c:CAPEC)-[r]->(x) RETURN type(r) AS t, labels(x) AS lbl, count(*) AS c ORDER BY c DESC LIMIT 20"):
        print(" ", r["t"], r["lbl"], r["c"])
d.close()
