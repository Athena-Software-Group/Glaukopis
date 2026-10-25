// Node and relationship counts of the release graph (Table S2 of the supplementary material).
// Run against the athena-cti-db database of the Neo4j instance used for the v21 build, e.g.
//   cypher-shell -a bolt://localhost:7687 -d athena-cti-db --format plain < analysis/graph_counts.cypher
// Output of the 2026-10-25 run (Neo4j 2026.03.1; store last written 2026-05-04): data/graph_counts_v21.json
MATCH (n) RETURN count(n) AS nodes;
MATCH ()-[r]->() RETURN count(r) AS relationships;
CALL db.labels() YIELD label RETURN count(label) AS labels;
CALL db.relationshipTypes() YIELD relationshipType RETURN count(relationshipType) AS reltypes;
MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY n DESC;
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC;
MATCH (n) RETURN size(labels(n)) AS nlabels, count(*) AS n ORDER BY nlabels;
