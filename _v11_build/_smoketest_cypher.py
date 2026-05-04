"""Direct Cypher smoketest of three candidate AB.MS.GRP.1 forms.
Bypasses iftgen / parser entirely so we can iterate on Cypher shape
without re-running the full pipeline."""
import json, time, collections, datetime
from neo4j import GraphDatabase
cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

LIMIT = 1500
TIMEOUT_S = 30  # client-side python driver wait
TX_TIMEOUT = 30.0  # server-side tx timeout (seconds)

CANDIDATES = {
    "F1_unwind_orderby_rand_call": (
        f"MATCH (grp:`intrusion-set`) "
        f"WITH collect(DISTINCT grp) AS _allprim, count(DISTINCT grp) AS _nprim "
        f"UNWIND range(1, {LIMIT}) AS _dup_i "
        f"WITH _allprim[toInteger(rand() * _nprim)] AS grp "
        f"CALL (grp) {{ "
        f"  MATCH (grp:`intrusion-set`)-[:uses]->(ap1:`attack-pattern`), "
        f"        (grp:`intrusion-set`)-[:uses]->(ap2:`attack-pattern`), "
        f"        (negap1:`attack-pattern`), (negap2:`attack-pattern`), (negap3:`attack-pattern`) "
        f"  WHERE elementId(ap1) <> elementId(ap2) "
        f"    AND NOT elementId(negap1) IN [elementId(ap1), elementId(ap2)] "
        f"    AND NOT elementId(negap2) IN [elementId(ap1), elementId(ap2), elementId(negap1)] "
        f"    AND NOT elementId(negap3) IN [elementId(ap1), elementId(ap2), elementId(negap1), elementId(negap2)] "
        f"  RETURN ap1, ap2, negap1, negap2, negap3 ORDER BY rand() LIMIT 1 }} "
        f"RETURN DISTINCT grp.name AS gn, ap1.mitre_id AS a1, ap2.mitre_id AS a2, "
        f"  negap1.mitre_id AS n1, negap2.mitre_id AS n2, negap3.mitre_id AS n3 "
        f"LIMIT {LIMIT}"
    ),
    "F2_unwind_deterministic_call": (
        f"MATCH (grp:`intrusion-set`) "
        f"WITH collect(DISTINCT grp) AS _allprim, count(DISTINCT grp) AS _nprim "
        f"UNWIND range(1, {LIMIT}) AS _dup_i "
        f"WITH _allprim[toInteger(rand() * _nprim)] AS grp "
        f"CALL (grp) {{ "
        f"  MATCH (grp:`intrusion-set`)-[:uses]->(ap1:`attack-pattern`), "
        f"        (grp:`intrusion-set`)-[:uses]->(ap2:`attack-pattern`), "
        f"        (negap1:`attack-pattern`), (negap2:`attack-pattern`), (negap3:`attack-pattern`) "
        f"  WHERE elementId(ap1) <> elementId(ap2) "
        f"    AND NOT elementId(negap1) IN [elementId(ap1), elementId(ap2)] "
        f"    AND NOT elementId(negap2) IN [elementId(ap1), elementId(ap2), elementId(negap1)] "
        f"    AND NOT elementId(negap3) IN [elementId(ap1), elementId(ap2), elementId(negap1), elementId(negap2)] "
        f"  RETURN ap1, ap2, negap1, negap2, negap3 LIMIT 1 }} "
        f"RETURN DISTINCT grp.name AS gn, ap1.mitre_id AS a1, ap2.mitre_id AS a2, "
        f"  negap1.mitre_id AS n1, negap2.mitre_id AS n2, negap3.mitre_id AS n3 "
        f"LIMIT {LIMIT}"
    ),
    "F3_step_by_step_per_anchor": (
        f"MATCH (grp:`intrusion-set`) "
        f"WITH collect(DISTINCT grp) AS _allprim, count(DISTINCT grp) AS _nprim "
        f"UNWIND range(1, {LIMIT}) AS _dup_i "
        f"WITH _allprim[toInteger(rand() * _nprim)] AS grp "
        f"CALL (grp) {{ "
        f"  MATCH (grp)-[:uses]->(ap1:`attack-pattern`) "
        f"    WITH ap1 ORDER BY rand() LIMIT 1 "
        f"  MATCH (grp)-[:uses]->(ap2:`attack-pattern`) "
        f"    WHERE ap2 <> ap1 "
        f"    WITH ap1, ap2 ORDER BY rand() LIMIT 1 "
        f"  MATCH (negap1:`attack-pattern`) "
        f"    WHERE NOT negap1 IN [ap1, ap2] "
        f"    WITH ap1, ap2, negap1 ORDER BY rand() LIMIT 1 "
        f"  MATCH (negap2:`attack-pattern`) "
        f"    WHERE NOT negap2 IN [ap1, ap2, negap1] "
        f"    WITH ap1, ap2, negap1, negap2 ORDER BY rand() LIMIT 1 "
        f"  MATCH (negap3:`attack-pattern`) "
        f"    WHERE NOT negap3 IN [ap1, ap2, negap1, negap2] "
        f"    WITH ap1, ap2, negap1, negap2, negap3 ORDER BY rand() LIMIT 1 "
        f"  RETURN ap1, ap2, negap1, negap2, negap3 }} "
        f"RETURN DISTINCT grp.name AS gn, ap1.mitre_id AS a1, ap2.mitre_id AS a2, "
        f"  negap1.mitre_id AS n1, negap2.mitre_id AS n2, negap3.mitre_id AS n3 "
        f"LIMIT {LIMIT}"
    ),
}

def run_with_tx_timeout(q, label):
    print(f"\n========== {label} ==========", flush=True)
    t0 = time.time()
    try:
        with drv.session(database=cfg["db_name"]) as s:
            tx = s.begin_transaction(timeout=TX_TIMEOUT)
            try:
                rows = list(tx.run(q))
                tx.commit()
            except Exception:
                try: tx.rollback()
                except Exception: pass
                raise
        elapsed = time.time() - t0
        anchors = collections.Counter(r["gn"] for r in rows)
        print(f"  rows={len(rows):<6} distinct_grp={len(anchors):<4} elapsed={elapsed:.1f}s")
        top = anchors.most_common(5)
        print(f"  top-5 anchors: {top}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:200]}")

for name, q in CANDIDATES.items():
    run_with_tx_timeout(q, name)
drv.close()
