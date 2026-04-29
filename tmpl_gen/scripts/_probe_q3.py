import os, time
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ["NEO4J_URL"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

# JS.MCQ.4 primary query, but with cwe.description STRIPPED from RETURN
q4_no_desc = """MATCH (cwe:Weakness) WITH DISTINCT cwe ORDER BY rand() LIMIT 3
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
RETURN DISTINCT cwe.id, cwe.name, cap.id, cap.name,
       negcap1.id, negcap1.name, negcap2.id, negcap2.name,
       negcap3.id, negcap3.name, negcap4.id, negcap4.name
ORDER BY rand() LIMIT 3"""

# Same but using CALL subquery per-primary form
q4_per_primary = """MATCH (cwe:Weakness) WITH DISTINCT cwe ORDER BY rand() LIMIT 3
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
RETURN DISTINCT cwe.id, cwe.name, cwe.description,
       cap.id, cap.name, negcap1.id, negcap1.name,
       negcap2.id, negcap2.name, negcap3.id, negcap3.name,
       negcap4.id, negcap4.name
LIMIT 3"""

for name, q in [("JS.MCQ.4 no-description", q4_no_desc), ("JS.MCQ.4 per-primary CALL with desc", q4_per_primary)]:
    print(f"=== {name} ===")
    t0 = time.time()
    try:
        with d.session(database=os.environ["NEO4J_DB"]) as s:
            rows = list(s.run(q))
            print(f"  rows: {len(rows)}  elapsed: {time.time()-t0:.2f}s")
            for r in rows[:1]:
                print("  cwe:", r["cwe.id"], "cap:", r["cap.id"], "negs:", [r[f"negcap{i}.id"] for i in (1,2,3,4)])
    except Exception as e:
        print(f"  FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {str(e)[:200]}")
    print()
d.close()
