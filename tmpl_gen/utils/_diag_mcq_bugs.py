#!/usr/bin/env python3
"""Diagnostic for the two zero-row templates in the v8.1 build:
   - AB.MCQ.4: {cap:cwe.related_attack_pattern>CAPEC.id}
   - JS.MCQ.2: {coa:ap.mitigates<course-of-action.mitre_id} + neg coa selection

Confirms whether the relationship name / direction / property pattern that
the template uses actually yields rows in the local athena-cti-db.
"""
import json
from neo4j import GraphDatabase

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

with drv.session(database=cfg["db_name"]) as s:
    print("=== AB.MCQ.4 -- Weakness <-> CAPEC ===")
    rows = s.run(
        "MATCH (w:Weakness)-[r]->(c:CAPEC) "
        "RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC LIMIT 10"
    ).data()
    print(f"  outgoing W->CAPEC : {rows}")
    rows = s.run(
        "MATCH (w:Weakness)<-[r]-(c:CAPEC) "
        "RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC LIMIT 10"
    ).data()
    print(f"  incoming W<-CAPEC : {rows}")
    rows = s.run("MATCH (c:CAPEC) RETURN keys(c)[..20] AS k LIMIT 1").data()
    print(f"  CAPEC keys        : {rows}")
    rows = s.run("MATCH (w:Weakness) RETURN keys(w)[..20] AS k LIMIT 1").data()
    print(f"  Weakness keys     : {rows}")
    rows = s.run(
        "MATCH (c:CAPEC) WHERE c.related_weaknesses IS NOT NULL "
        "RETURN c.id AS id, c.related_weaknesses[..3] AS rw LIMIT 5"
    ).data()
    print(f"  CAPEC.related_weaknesses sample: {rows}")
    rows = s.run(
        "MATCH (w:Weakness) WHERE w.related_attack_patterns IS NOT NULL "
        "RETURN w.id AS id, w.related_attack_patterns[..3] AS rap LIMIT 5"
    ).data()
    print(f"  Weakness.related_attack_patterns sample: {rows}")

    print()
    print("=== JS.MCQ.2 -- ap <-mitigates- course-of-action ===")
    rows = s.run(
        "MATCH (ap:`attack-pattern`)<-[r:mitigates]-(c:`course-of-action`) "
        "RETURN count(*) AS n"
    ).data()
    print(f"  ap<-mitigates-coa edges: {rows}")
    rows = s.run(
        "MATCH (c:`course-of-action`) WHERE c.mitre_id STARTS WITH 'M1' "
        "RETURN count(*) AS n"
    ).data()
    print(f"  coa with mitre_id starting M1: {rows}")
    rows = s.run(
        "MATCH (c:`course-of-action`) WHERE c.mitre_id IS NOT NULL "
        "RETURN c.mitre_id AS mid LIMIT 8"
    ).data()
    print(f"  sample mitre_ids: {rows}")
    rows = s.run(
        "MATCH (ap:`attack-pattern`)<-[:mitigates]-(c:`course-of-action`) "
        "WITH ap, count(c) AS n RETURN n, count(*) AS aps "
        "ORDER BY n DESC LIMIT 12"
    ).data()
    print(f"  histogram of mitigations per ap: {rows}")

    # JS.MCQ.2 specifically asks: pick ap, then find ONE coa1 that mitigates it,
    # AND four neg coa that don't mitigate it. With Sample: ap and force constraints
    # being WHERE neq, the question is whether the joined query is missing.
    # Mimic the parser's likely query shape:
    print()
    print("=== mimic JS.MCQ.2 join shape (5-coa MCQ) ===")
    q = (
        "MATCH (ap:`attack-pattern`)<-[:mitigates]-(coa:`course-of-action`), "
        "      (negcoa1:`course-of-action`), (negcoa2:`course-of-action`), "
        "      (negcoa3:`course-of-action`), (negcoa4:`course-of-action`) "
        "WHERE coa.mitre_id IS NOT NULL "
        "  AND negcoa1.mitre_id IS NOT NULL AND negcoa2.mitre_id IS NOT NULL "
        "  AND negcoa3.mitre_id IS NOT NULL AND negcoa4.mitre_id IS NOT NULL "
        "  AND negcoa1.mitre_id <> coa.mitre_id "
        "  AND negcoa2.mitre_id <> coa.mitre_id "
        "  AND negcoa3.mitre_id <> coa.mitre_id "
        "  AND negcoa4.mitre_id <> coa.mitre_id "
        "  AND negcoa1.mitre_id <> negcoa2.mitre_id "
        "  AND negcoa1.mitre_id <> negcoa3.mitre_id "
        "  AND negcoa1.mitre_id <> negcoa4.mitre_id "
        "  AND negcoa2.mitre_id <> negcoa3.mitre_id "
        "  AND negcoa2.mitre_id <> negcoa4.mitre_id "
        "  AND negcoa3.mitre_id <> negcoa4.mitre_id "
        "RETURN ap.mitre_id AS ap, coa.mitre_id AS coa, "
        "  negcoa1.mitre_id AS n1, negcoa2.mitre_id AS n2, "
        "  negcoa3.mitre_id AS n3, negcoa4.mitre_id AS n4 LIMIT 3"
    )
    rows = s.run(q).data()
    print(f"  join rows: {len(rows)}; first: {rows[:1]}")

drv.close()
