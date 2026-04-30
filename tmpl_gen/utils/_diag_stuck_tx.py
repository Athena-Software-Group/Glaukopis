#!/usr/bin/env python
"""Diagnose / kill the currently-running stuck Neo4j transaction(s)."""
from neo4j import GraphDatabase
import json, time, sys

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))
with drv.session(database=cfg["db_name"]) as s:
    txs = list(s.run("SHOW TRANSACTIONS").data())
    print("=== active txs ===")
    for t in txs:
        cq = str(t.get('currentQuery', ''))
        if 'SHOW TRANSACTIONS' in cq:
            continue
        print(f"  - {t.get('transactionId')} elapsed={t.get('elapsedTime')}")
        print(f"    q: {cq[:300]}")

    if "--kill" in sys.argv:
        for t in txs:
            cq = str(t.get('currentQuery', ''))
            if 'SHOW TRANSACTIONS' in cq:
                continue
            tid = t.get('transactionId')
            print(f"  killing {tid}")
            s.run(f"TERMINATE TRANSACTION '{tid}'")
        time.sleep(2)
        txs = list(s.run("SHOW TRANSACTIONS").data())
        print("=== after kill ===")
        for t in txs:
            cq = str(t.get('currentQuery', ''))[:120]
            print(f"  - {t.get('transactionId')} elapsed={t.get('elapsedTime')} q={cq}")

    print()
    print("=== node counts ===")
    sigma_q = "MATCH (sr:SigmaRule) RETURN count(sr) AS c"
    ap_q = "MATCH (ap:`attack-pattern`) RETURN count(ap) AS c"
    det_q = "MATCH (sr:SigmaRule)-[:detects]->(ap:`attack-pattern`) RETURN count(*) AS c"
    for q in [sigma_q, ap_q, det_q]:
        print(q, '->', list(s.run(q).data()))
drv.close()
