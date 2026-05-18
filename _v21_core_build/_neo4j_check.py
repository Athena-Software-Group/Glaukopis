#!/usr/bin/env python3
"""Substrate validation for v18: confirms local Neo4j athena-cti-db is
populated with the entities the v18 manifest will traverse.

Probes the v18-critical paths:
  AB.ATE.{9}     SigmaRule.title + .description -> sr.detects>attack-pattern
  AB.ATE.{10}    malware.description -> mw.uses>attack-pattern
  AB.ATE.{11}    intrusion-set.description -> grp.uses>attack-pattern
  AB.MCQ.{1,2,5} attack-pattern + course-of-action + intrusion-set hard-neg
  TAA.CANON.*    intrusion-set.aliases populated

Run as Phase 0 of _v18_build/watcher.sh; exits 1 if any entity floor or
traversal floor is unmet so the build halts before make_dataset starts."""

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

HERE = Path(__file__).resolve().parent
CFG_PATH = HERE.parent / "tmpl_gen" / "data_generation" / "neo4j-local-config.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON path for entity/traversal results.")
    args = p.parse_args()

    cfg = json.loads(CFG_PATH.read_text())
    drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))
    failed = []
    entity_results: list[dict] = []
    traversal_results: list[dict] = []
    with drv.session(database=cfg["db_name"]) as s:
        print("--- node label counts ---")
        labels = [r["label"] for r in s.run(
            "CALL db.labels() YIELD label RETURN label ORDER BY label")]
        for lbl in labels:
            n = s.run(
                "MATCH (n) WHERE labels(n)[0]=$lbl RETURN count(n) AS c",
                lbl=lbl,
            ).single()["c"]
            print(f"  {lbl:40s} {n:>8d}")

        print("--- relationship type counts ---")
        rtypes = [r["t"] for r in s.run(
            "CALL db.relationshipTypes() YIELD relationshipType AS t "
            "RETURN t ORDER BY t")]
        for rt in rtypes:
            n = s.run(
                "MATCH ()-[r]->() WHERE type(r)=$rt RETURN count(r) AS c",
                rt=rt,
            ).single()["c"]
            print(f"  {rt:40s} {n:>8d}")

        print("--- v18-critical entity floors ---")
        floors = [
            ("intrusion-set",  "MATCH (n:`intrusion-set`) RETURN count(n) AS c",  180),
            ("malware",        "MATCH (n:malware) RETURN count(n) AS c",          600),
            ("attack-pattern", "MATCH (n:`attack-pattern`) RETURN count(n) AS c", 800),
            ("course-of-action", "MATCH (n:`course-of-action`) RETURN count(n) AS c", 200),
            ("x-mitre-tactic",   "MATCH (n:`x-mitre-tactic`) RETURN count(n) AS c", 14),
            ("x-mitre-data-source", "MATCH (n:`x-mitre-data-source`) RETURN count(n) AS c", 30),
            ("x-mitre-data-component", "MATCH (n:`x-mitre-data-component`) RETURN count(n) AS c", 100),
        ]
        for name, q, floor in floors:
            try:
                got = s.run(q).single()["c"]
            except Exception as e:
                got = -1
                print(f"  {name:30s} ERR ({e})")
                failed.append(name)
                entity_results.append({"name": name, "got": got, "floor": floor, "ok": False, "error": str(e)})
                continue
            mark = "OK" if got >= floor else "FAIL"
            print(f"  {name:30s} got={got:>6d}  floor={floor:>5d}  {mark}")
            entity_results.append({"name": name, "got": got, "floor": floor, "ok": got >= floor})
            if got < floor:
                failed.append(name)

        print("--- v18-critical traversal probes ---")
        probes = [
            (
                "intrusion-set->uses->attack-pattern (AB.ATE.11, AB.MCQ.5)",
                "MATCH (g:`intrusion-set`)-[:uses]->(t:`attack-pattern`) RETURN count(*) AS c",
                4000,
            ),
            (
                "intrusion-set with description >100 chars (AB.ATE.11 anchor)",
                "MATCH (g:`intrusion-set`) WHERE g.description IS NOT NULL AND size(g.description) > 100 RETURN count(g) AS c",
                150,
            ),
            (
                "malware->uses->attack-pattern (AB.ATE.{8,10})",
                "MATCH (m:malware)-[:uses]->(a:`attack-pattern`) RETURN count(*) AS c",
                8000,
            ),
            (
                "malware with description >80 chars AND uses>=1 (AB.ATE.10 anchor)",
                "MATCH (m:malware)-[:uses]->(a:`attack-pattern`) WHERE m.description IS NOT NULL AND size(m.description) > 80 RETURN count(DISTINCT m) AS c",
                500,
            ),
            (
                "SigmaRule->detects->attack-pattern (AB.ATE.9)",
                "MATCH (s:SigmaRule)-[:detects]->(a:`attack-pattern`) RETURN count(*) AS c",
                3000,
            ),
            (
                "SigmaRule with description >30 chars (AB.ATE.9 anchor)",
                "MATCH (s:SigmaRule) WHERE s.description IS NOT NULL AND size(s.description) > 30 RETURN count(s) AS c",
                2500,
            ),
            (
                "intrusion-set with aliases populated (TAA.CANON.*)",
                "MATCH (g:`intrusion-set`) WHERE g.aliases IS NOT NULL RETURN count(g) AS c",
                150,
            ),
            (
                "course-of-action->mitigates->attack-pattern (AB.MCQ.2)",
                "MATCH (c:`course-of-action`)-[:mitigates]->(a:`attack-pattern`) RETURN count(*) AS c",
                1000,
            ),
            (
                "technique->subtechnique-of->parent (AB.ATE.7)",
                "MATCH (s:`attack-pattern`)-[:`subtechnique-of`]->(p:`attack-pattern`) RETURN count(*) AS c",
                400,
            ),
            (
                "technique->achieves->tactic (AB.MCQ.1, JS.MCQ.1)",
                "MATCH (a:`attack-pattern`)-[:achieves]->(t:`x-mitre-tactic`) RETURN count(*) AS c",
                800,
            ),
        ]
        for name, q, floor in probes:
            try:
                got = s.run(q).single()["c"]
            except Exception as e:
                print(f"  {name:65s} ERR ({e})")
                failed.append(name)
                traversal_results.append({"name": name, "got": -1, "floor": floor, "ok": False, "error": str(e)})
                continue
            mark = "OK" if got >= floor else "FAIL"
            print(f"  {name:65s} got={got:>6d}  floor={floor:>5d}  {mark}")
            traversal_results.append({"name": name, "got": got, "floor": floor, "ok": got >= floor})
            if got < floor:
                failed.append(name)
    drv.close()

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "ok": not failed,
            "failed": failed,
            "entities": entity_results,
            "traversals": traversal_results,
        }, indent=2))

    if failed:
        print(f"\n[neo4j_check] FAILED checks: {failed}")
        return 1
    print("\n[neo4j_check] all v18-critical floors met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
