# Athena CTI Database: Local Setup

`athena_cti_db` provides a population script for building the Athena Threat Intelligence graph database on a local Neo4j instance. It ingests data from MITRE ATT&CK, CAPEC, CWE, CVE, CISA KEV, FIRST EPSS, and MITRE ENGAGE, and models all entities and cross-framework relationships in Neo4j.

The primary entry point is [`threat_framework/populate_neo4j_complete.py`](threat_framework/populate_neo4j_complete.py), which downloads, parses, and loads all CTI data into Neo4j.

---

## Directories

- **[`threat_framework/`](threat_framework/)** — Population script and data pipeline.
  - `populate_neo4j_complete.py` — downloads all CTI sources and populates the Neo4j graph

### Scripts (all under `utils/`)

- **[`utils/setup.sh`](utils/setup.sh)** — end-to-end orchestrator: TCP-preflights Neo4j, creates a venv, installs dependencies, verifies auth + target DB is online, then runs `populate.sh`. Honours `utils/.env` for `NEO4J_*` settings.
- **[`utils/install.sh`](utils/install.sh)** — installs Python dependencies from `requirements.txt` into the active environment.
- **[`utils/populate.sh`](utils/populate.sh)** — populate-only wrapper. Safe to run from cron / launchd to refresh the always-refresh sources (EPSS + NVD current year).

---

## Directory Structure

```
athena_cti_db/
├── README.md
├── README_LOCAL_SETUP.md
├── FUNCTIONAL_SCOPE.md
├── requirements.txt
├── utils/
│   ├── setup.sh
│   ├── install.sh
│   ├── populate.sh
│   └── threat_data/                      # cached source data (created on first run)
└── threat_framework/
    └── populate_neo4j_complete.py
```

After running the population script, downloaded data is cached under `utils/threat_data/`:

```
utils/threat_data/
├── cti/                                  # MITRE ATT&CK STIX JSON files (git clone)
├── engage/                               # MITRE ENGAGE JSON files (git clone)
├── cve/cves/                             # CVE 5.0 JSON files, by year (sparse clone, 2024+)
├── nvd/                                  # NVD CVE 2.0 bulk feeds (per-year ndjson batches; current year auto-refreshed)
├── cwe/                                  # CWE XML data
├── capec_latest.xml                      # CAPEC XML
├── known_exploited_vulnerabilities.json  # CISA KEV
├── epss_scores_data.csv                  # EPSS scores (yesterday's snapshot; auto-refreshed every run)
├── d3fend/                               # MITRE D3FEND ontology + ATT&CK mappings (pinned v1.4.0)
├── sigma/                                # SigmaHQ detection rules (sparse clone of rules/)
├── exploitdb/                            # ExploitDB files_exploits.csv (sparse clone)
└── poc_github/                           # nomi-sec/PoC-in-GitHub year folders (sparse clone, 2024+)
```

> The full functional inventory (sources, schema, axes) lives in [`FUNCTIONAL_SCOPE.md`](./FUNCTIONAL_SCOPE.md). Source of truth for source URLs is `populate_neo4j_complete.py::DATA_SOURCES`.

---

## Installation

### Prerequisites

- **Python** 3.8 or higher
- **Neo4j Desktop** 5.x — [download here](https://neo4j.com/download/)
- **Git** — required for cloning MITRE CTI and CVE repositories
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk Space**: 20 GB free minimum

### 1. Set Up Neo4j

1. Install and launch Neo4j Desktop
2. Create a new project, then add a **Local DBMS** — choose a name and a strong password, select Neo4j 5.x
3. Open the **Plugins** tab and install **APOC**
4. Open **Settings** for the database and add:

```properties
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g
```

5. Start the database — wait for the status to show **Active**

Connection defaults once running:
- **Bolt URL**: `neo4j://localhost:7687`
- **HTTP URL**: `http://localhost:7474`
- **Username**: `neo4j`

### 2. Set Up Python Environment

```bash
cd athena_cti_db/

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

./utils/install.sh
```

Or install directly with pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Credentials

Set these environment variables before running the script:

**macOS/Linux**:
```bash
export NEO4J_URL="neo4j://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export NEO4J_DB="neo4j"
```

**Windows (PowerShell)**:
```powershell
$env:NEO4J_URL="neo4j://127.0.0.1:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="your-password"
$env:NEO4J_DB="neo4j"
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URL` | `neo4j://127.0.0.1:7687` | Bolt connection URL |
| `NEO4J_USER` | `neo4j` | Database username |
| `NEO4J_PASSWORD` | *(required)* | Database password |
| `NEO4J_DB` | `neo4j` | Target database name |

---

## How to Populate the Database

### Quick Start

End-to-end (preflight + install + populate):

```bash
cd athena_cti_db/
./utils/setup.sh
```

Populate only (assumes deps and Neo4j are ready):

```bash
./utils/populate.sh
```

Or run the populator script directly:

```bash
python threat_framework/populate_neo4j_complete.py
```

### What the Script Does

1. **Downloads CTI data** into `utils/threat_data/`. Refresh policy varies by source (see [`README.md`](README.md#auto-sync--refresh-semantics) §Auto-Sync for the full table):
   - **Always-refresh**: FIRST EPSS (yesterday's snapshot, overwritten), NVD current year (gzip feed, re-downloaded)
   - **Skip-if-cached (HTTPS)**: CAPEC, CWE, CISA KEV, MITRE D3FEND (pinned v1.4.0)
   - **Skip-if-cached (Git)**: MITRE ATT&CK (git clone, STIX2 JSON), MITRE ENGAGE, CVE Project (sparse, 2024+), Sigma (sparse `rules/`), ExploitDB (sparse `files_exploits.csv`), PoC-in-GitHub (sparse, 2024+ year folders)

2. **Parses and transforms** each source into Cypher MERGE statements

3. **Populates Neo4j**:
   - Creates uniqueness constraints (`stix_id` and natural-key `id`/`d3fend_id` per label)
   - Inserts nodes: Tactics, Techniques, CAPEC, CWE, CVE, KEV, Engage, EPSS,
     D3FENDTactic, D3FENDTechnique, SigmaRule, ExploitDBEntry, GithubPoC
   - Creates all intra- and cross-framework relationships (see [`FUNCTIONAL_SCOPE.md`](FUNCTIONAL_SCOPE.md) §5)

The populator is idempotent — every write is a `MERGE` against the per-label uniqueness constraints, so re-runs are safe and only the changed sources actually affect the graph.

### Expected Output

```
INFO - Starting complete threat intelligence data import...
INFO - Cloning MITRE CTI repository...
INFO - Processing ATT&CK data...
INFO - Processing CAPEC data...
INFO - Processing CWE data...
INFO - Processing CVE data...
INFO - Progress: Processed 10,000 / XXXXX CVE files
INFO - Processing KEV data...
INFO - Processing Engage data...
INFO - Downloading and updating EPSS data...
INFO - ============================================================
INFO - Import completed successfully!
INFO - ============================================================
```

> CVE ingestion (200K+ entries) is the longest step — expect 30–90 minutes total depending on system and network speed. Do not interrupt the process.

---

## Data Sources

All twelve sources are defined in `populate_neo4j_complete.py::DATA_SOURCES`. See [`FUNCTIONAL_SCOPE.md`](./FUNCTIONAL_SCOPE.md) for licence posture, retention, and node-label mapping per source.

| Source | URL | Format |
|--------|-----|--------|
| **MITRE ATT&CK** | https://github.com/mitre/cti.git | Git / STIX2 JSON |
| **MITRE ENGAGE** | https://github.com/mitre/engage.git | Git / JSON |
| **CAPEC** | http://capec.mitre.org/data/xml/capec_latest.xml | XML |
| **CWE** | http://cwe.mitre.org/data/xml/cwec_latest.xml.zip | ZIP / XML |
| **MITRE D3FEND** | https://d3fend.mitre.org/ontologies/d3fend/{version}/d3fend.json (+ `…/d3fend-full-mappings.json`) | JSON-LD / SPARQL-JSON (pinned v1.4.0) |
| **CVE** | https://github.com/CVEProject/cvelistV5.git | Git sparse-checkout / JSON (2024+) |
| **NVD** | https://nvd.nist.gov/feeds/json/cve/2.0 | HTTPS / per-year gzip (2024+) |
| **CISA KEV** | https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json | JSON Feed |
| **EPSS** | https://epss.cyentia.com/epss_scores-{date}.csv.gz | Gzipped CSV |
| **Sigma** | https://github.com/SigmaHQ/sigma.git | Git clone / YAML rules |
| **ExploitDB** | https://gitlab.com/exploit-database/exploitdb.git | Git sparse-clone / CSV |
| **PoC-in-GitHub** | https://github.com/nomi-sec/PoC-in-GitHub.git | Git sparse-clone / per-CVE JSON (2024+) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Connection refused** | Verify database is Active (green) in Neo4j Desktop; check port 7687 is not blocked |
| **Authentication failed** | Confirm `NEO4J_PASSWORD` matches the database password; reset in Neo4j Desktop if needed |
| **Memory error / Java heap space** | Increase heap in database settings (Step 1.4); close other apps and restart Neo4j |
| **APOC procedures not found** | Reinstall APOC plugin; ensure `dbms.security.procedures.unrestricted=apoc.*` is set; restart DB |
| **CVE download fails** | Check internet connection; CVE repo is 500 MB+; place manually in `threat_data/cve/` if needed |
| **No data in Neo4j after run** | Check logs for errors; verify `threat_data/` was populated; ensure sufficient disk space |
| **Python module not found** | Activate the virtual environment; run `pip install --upgrade -r requirements.txt` |
| **Script is very slow** | Expected for large datasets; use SSD storage; allocate more heap memory in settings |

---

## Status

This project is in active development. Data sources, schema, and relationships may evolve as the CTI graph is extended.
