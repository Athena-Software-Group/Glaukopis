import os
from neo4j import GraphDatabase

url = os.environ['NEO4J_URL']
drv = GraphDatabase.driver(url, auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
with drv.session(database=os.environ['NEO4J_DB']) as s:
    print("=== node label counts (target labels) ===")
    for lbl in ["CVE", "Weakness", "CAPEC", "attack-pattern", "x-mitre-tactic",
                "course-of-action", "intrusion-set", "EPSS", "KEV",
                "SigmaRule", "ExploitDBEntry", "GithubPoC"]:
        try:
            q = "MATCH (n:`" + lbl + "`) RETURN count(n) AS c"
            r = s.run(q).single()
            print(f"  {lbl:20s} {r['c']:>10,}")
        except Exception as e:
            print(f"  {lbl:20s} ERROR {e}")
    print()
    print("=== relationship counts (target rels) ===")
    for rel in ["has_weaponized_exploit", "has_poc", "detects",
                "problemType", "impacts", "map_ap", "achieves", "scores", "known_exploit"]:
        q = "MATCH ()-[r:`" + rel + "`]->() RETURN count(r) AS c"
        r = s.run(q).single()
        print(f"  {rel:25s} {r['c']:>10,}")
    print()
    print("=== CVE.cpe_matches presence ===")
    r = s.run("MATCH (c:CVE) WHERE c.cpe_matches IS NOT NULL RETURN count(c) AS c").single()
    print(f"  CVEs with cpe_matches: {r['c']:,}")
    r = s.run("MATCH (c:CVE) WHERE c.cpe_matches IS NOT NULL RETURN c.id AS id, substring(c.cpe_matches,0,180) AS cpe LIMIT 3").data()
    for row in r:
        print(f"  {row['id']}: {row['cpe']}")
    print()
    for lbl in ["SigmaRule", "ExploitDBEntry", "GithubPoC"]:
        q = "MATCH (n:`" + lbl + "`) RETURN keys(n) AS k LIMIT 1"
        r = s.run(q).single()
        print(f"=== {lbl} keys: {r['k'] if r else 'no nodes'}")
drv.close()
