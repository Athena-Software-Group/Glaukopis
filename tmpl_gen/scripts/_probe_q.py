import os, time
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ["NEO4J_URL"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

# Per-primary grouping form: pre-sample primaries, then CALL-subquery one row per primary
q_pp = """MATCH (cwe:Weakness) WITH DISTINCT cwe ORDER BY rand() LIMIT 3
CALL (cwe) {
  MATCH (cwe)-[:related_attack_pattern]->(cap:CAPEC),
        (negcap1:CAPEC), (negcap2:CAPEC), (negcap3:CAPEC), (negcap4:CAPEC)
  WHERE elementId(negcap1) <> elementId(negcap3)
    AND elementId(negcap1) <> elementId(negcap2)
    AND elementId(negcap1) <> elementId(negcap4)
    AND elementId(negcap3) <> elementId(negcap4)
    AND elementId(negcap2) <> elementId(negcap3)
    AND elementId(negcap2) <> elementId(negcap4)
    AND elementId(cap)     <> elementId(negcap1)
    AND elementId(cap)     <> elementId(negcap3)
    AND elementId(cap)     <> elementId(negcap2)
    AND elementId(cap)     <> elementId(negcap4)
  RETURN cap, negcap1, negcap2, negcap3, negcap4 LIMIT 1
}
RETURN DISTINCT cwe.id, cap.id, negcap1.id, negcap2.id, negcap3.id, negcap4.id
LIMIT 3"""

# Bounded fallback that current code uses (LIMIT after MATCH, before RETURN)
q_bf = """MATCH (cwe:Weakness) WITH DISTINCT cwe ORDER BY rand() LIMIT 3
MATCH (cwe)-[:related_attack_pattern]->(cap:CAPEC),
      (negcap1:CAPEC), (negcap2:CAPEC), (negcap3:CAPEC), (negcap4:CAPEC)
WHERE elementId(negcap1) <> elementId(negcap3)
  AND elementId(negcap1) <> elementId(negcap2)
  AND elementId(negcap1) <> elementId(negcap4)
  AND elementId(negcap3) <> elementId(negcap4)
  AND elementId(negcap2) <> elementId(negcap3)
  AND elementId(negcap2) <> elementId(negcap4)
  AND elementId(cap)     <> elementId(negcap1)
  AND elementId(cap)     <> elementId(negcap3)
  AND elementId(cap)     <> elementId(negcap2)
  AND elementId(cap)     <> elementId(negcap4)
LIMIT 3
RETURN DISTINCT cwe.id, cap.id, negcap1.id, negcap2.id, negcap3.id, negcap4.id"""

for name, q in [("per-primary CALL", q_pp), ("bounded fallback", q_bf)]:
    print(f"\n=== {name} ===")
    t0 = time.time()
    try:
        with d.session(database=os.environ["NEO4J_DB"]) as s:
            rows = list(s.run(q))
            print(f"  rows: {len(rows)}  elapsed: {time.time()-t0:.2f}s")
            for r in rows[:3]: print(" ", r)
    except Exception as e:
        print(f"  FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {str(e)[:200]}")
d.close()
