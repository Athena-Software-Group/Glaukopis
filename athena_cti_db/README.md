# Athena CTI DB

`athena_cti_db` builds and populates the **Athena Threat Intelligence graph database** on a local Neo4j instance. It ingests 12 public CTI sources — MITRE ATT&CK, CAPEC, CWE, ENGAGE, D3FEND, CVE Project, NVD, CISA KEV, FIRST EPSS, Sigma, ExploitDB, and PoC-in-GitHub — modelling all entities and cross-framework relationships in Neo4j.

The populated graph is the upstream data source for [`tmpl_gen`](../tmpl_gen/), which traverses it to generate Instruction Fine-Tuning data.

The primary entry point is [`threat_framework/populate_neo4j_complete.py`](threat_framework/populate_neo4j_complete.py), which downloads, parses, and loads all CTI data into Neo4j.

> - Full setup, Neo4j configuration, environment variables, and troubleshooting → **[`README_LOCAL_SETUP.md`](README_LOCAL_SETUP.md)**
> - Functional footprint — every data source, node label, and graph relationship → **[`FUNCTIONAL_SCOPE.md`](FUNCTIONAL_SCOPE.md)**

---

## Directory Layout

```
athena_cti_db/
├── install.sh                      # installs Python dependencies
├── populate.sh                     # runs the full population pipeline
├── requirements.txt
├── README.md                       # this file
├── README_LOCAL_SETUP.md           # detailed setup guide
└── threat_framework/
    └── populate_neo4j_complete.py  # downloads and loads all CTI sources
```

After the population script runs, downloaded data is cached under `threat_data/` (MITRE ATT&CK, ENGAGE, CVE, CWE, CAPEC, CISA KEV, EPSS).

---

## Prerequisites

- Python 3.8 or higher
- Neo4j Desktop 5.x with the **APOC** plugin
- Git (required for cloning the MITRE CTI and CVE repositories)
- 8 GB RAM minimum (16 GB recommended), 20 GB free disk space

Neo4j database settings required in the Neo4j Desktop configuration:

```properties
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*
server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g
```

---

## Quick Start

```bash
cd athena_cti_db/

# 1. Python environment
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
./install.sh

# 2. Neo4j connection parameters
export NEO4J_URL="neo4j://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export NEO4J_DB="neo4j"

# 3. Populate the graph
./populate.sh
```

Or, equivalently:

```bash
cd threat_framework/
python populate_neo4j_complete.py
```

CVE ingestion (200K+ entries) is the longest step — expect 30–90 minutes total depending on system and network speed.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URL` | `neo4j://127.0.0.1:7687` | Bolt connection URL |
| `NEO4J_USER` | `neo4j` | Database username |
| `NEO4J_PASSWORD` | *(required)* | Database password |
| `NEO4J_DB` | `neo4j` | Target database name |

---

## Data Sources

| Source | Format | Scope |
|--------|--------|-------|
| MITRE ATT&CK (`mitre/cti`) | Git / STIX 2.1 JSON | full (Enterprise) |
| MITRE ENGAGE (`mitre/engage`) | Git / JSON | full |
| MITRE CAPEC | XML | full |
| MITRE CWE | ZIP / XML | full |
| MITRE D3FEND | JSON-LD + SPARQL JSON | pinned v1.4.0 |
| CVE Project (`CVEProject/cvelistV5`) | Git sparse / JSON | 2024+ |
| NVD CPE/CVSS bulk feeds | NDJSON | 2024+ |
| CISA KEV | JSON feed | full |
| FIRST EPSS | Gzipped CSV (yesterday) | 2024+ |
| Sigma rules (`SigmaHQ/sigma`) | Git sparse / YAML | full `rules/` |
| ExploitDB | Git sparse / CSV | 2024+ CVE |
| PoC-in-GitHub (`nomi-sec/PoC-in-GitHub`) | Git sparse / JSON | 2024+ |

URLs and acquisition details: [`README_LOCAL_SETUP.md`](README_LOCAL_SETUP.md). Per-source content, node labels, and graph relationships: [`FUNCTIONAL_SCOPE.md`](FUNCTIONAL_SCOPE.md).

---

## What the Script Does

1. **Downloads CTI data** (skipped if already cached in `threat_data/`).
2. **Parses and transforms** each source into nodes and relationships.
3. **Populates Neo4j** — creates constraints, inserts nodes (Tactics, Techniques, CAPEC, CWE, CVE, KEV, Engage, EPSS), and builds all intra- and cross-framework relationships.

---

## Troubleshooting

Common issues and fixes (connection refused, APOC errors, CVE download failures, heap memory, etc.) are documented in [`README_LOCAL_SETUP.md`](README_LOCAL_SETUP.md#troubleshooting).

---

## Status

Active development. Data sources, schema, and relationships may evolve as the CTI graph is extended.
