import json, os, time
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ["NEO4J_URL"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

for f in ["/tmp/v8_smoke_results/t_00003_JS.MCQ.4.json", "/tmp/v8_smoke_results/t_00005_JS.MCQ.6.json"]:
    print("===", f, "===")
    obj = json.load(open(f))
    q = obj["query"]
    print("QUERY (full):\n", q, "\n")
    t0 = time.time()
    try:
        with d.session(database=os.environ["NEO4J_DB"]) as s:
            rows = list(s.run(q))
            print(f"  rows: {len(rows)}  elapsed: {time.time()-t0:.2f}s")
            for r in rows[:2]: print(" ", r)
    except Exception as e:
        print(f"  FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {str(e)[:300]}")
    print()
d.close()
