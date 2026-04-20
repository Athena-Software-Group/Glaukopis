#!/bin/bash

# End-to-end setup for the Athena CTI Neo4j database:
#   1. Validates Neo4j connection parameters (host/port reachable, auth valid,
#      target database exists and is online)
#   2. Creates/activates a Python virtual environment
#   3. Installs dependencies (delegates to install.sh)
#   4. Populates the database (delegates to populate.sh)
#
# Usage:
#   ./setup.sh [-d DB_NAME] [-H HOST] [-P PORT] [-u USER] [--no-venv] [--skip-populate]
#
# Configuration precedence (highest wins):
#   1. CLI flags
#   2. Shell environment variables already exported before invocation
#   3. athena_cti_db/utils/.env (gitignored; holds NEO4J_URL/USER/PASSWORD/DB)
#   4. Built-in defaults
#
# Recognised variables:
#   NEO4J_URL       default: neo4j://127.0.0.1:7687
#   NEO4J_USER      default: neo4j
#   NEO4J_PASSWORD  (required, no default)
#   NEO4J_DB        default: athena-cti-db

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env if present. Existing shell env takes precedence over file entries.
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    echo "Loading config from ${ENV_FILE}"
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        [[ -z "${_line}" || "${_line}" =~ ^[[:space:]]*# ]] && continue
        _key="${_line%%=*}"
        _val="${_line#*=}"
        _key="${_key// /}"
        [[ -z "${_key}" ]] && continue
        # strip surrounding single or double quotes from the value
        if [[ "${_val}" =~ ^\"(.*)\"$ ]] || [[ "${_val}" =~ ^\'(.*)\'$ ]]; then
            _val="${BASH_REMATCH[1]}"
        fi
        if [[ -z "${!_key:-}" ]]; then
            export "${_key}=${_val}"
        fi
    done < "${ENV_FILE}"
    unset _line _key _val
fi

DB_NAME="${NEO4J_DB:-athena-cti-db}"
NEO4J_HOST="127.0.0.1"
NEO4J_PORT="7687"
NEO4J_USER="${NEO4J_USER:-neo4j}"
USE_VENV=1
RUN_POPULATE=1

if [[ -n "${NEO4J_URL:-}" ]]; then
    _url_stripped="${NEO4J_URL#*://}"
    NEO4J_HOST="${_url_stripped%%:*}"
    NEO4J_PORT="${_url_stripped##*:}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--db)            DB_NAME="$2"; shift 2 ;;
        -H|--host)          NEO4J_HOST="$2"; shift 2 ;;
        -P|--port)          NEO4J_PORT="$2"; shift 2 ;;
        -u|--user)          NEO4J_USER="$2"; shift 2 ;;
        --no-venv)          USE_VENV=0; shift ;;
        --skip-populate)    RUN_POPULATE=0; shift ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; started=1; next} started{exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

NEO4J_URL="neo4j://${NEO4J_HOST}:${NEO4J_PORT}"

if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
    echo "ERROR: NEO4J_PASSWORD is not set. Export it and re-run." >&2
    exit 1
fi

export NEO4J_URL NEO4J_USER NEO4J_PASSWORD NEO4J_DB="${DB_NAME}"

echo "=== Athena CTI DB Setup ==="
echo "  Repo dir     : ${REPO_DIR}"
echo "  Neo4j URL    : ${NEO4J_URL}"
echo "  Neo4j user   : ${NEO4J_USER}"
echo "  Database     : ${DB_NAME}"
echo "  Venv         : $([ ${USE_VENV} -eq 1 ] && echo yes || echo no)"
echo "  Populate     : $([ ${RUN_POPULATE} -eq 1 ] && echo yes || echo skipped)"
echo

# ── Preflight 1: TCP reachability (no Python deps needed) ─────────────────────
echo "[1/5] Checking TCP connectivity to ${NEO4J_HOST}:${NEO4J_PORT}..."
if ! python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(3)
try:
    s.connect(('${NEO4J_HOST}', int('${NEO4J_PORT}')))
    sys.exit(0)
except Exception as e:
    print(f'  connection failed: {e}', file=sys.stderr); sys.exit(1)
finally:
    s.close()
"; then
    echo "ERROR: Neo4j not reachable on ${NEO4J_HOST}:${NEO4J_PORT}." >&2
    echo "  Start the DBMS in Neo4j Desktop (or equivalent) and retry." >&2
    exit 1
fi
echo "  OK"

# ── Venv + dependencies ───────────────────────────────────────────────────────
if [[ ${USE_VENV} -eq 1 ]]; then
    VENV_DIR="${REPO_DIR}/venv"
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "[2/5] Creating virtualenv at ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    else
        echo "[2/5] Reusing virtualenv at ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
else
    echo "[2/5] Skipping venv (--no-venv); using active Python: $(command -v python)"
fi

echo "[3/5] Installing Python dependencies..."
bash "${SCRIPT_DIR}/install.sh"

# ── Preflight 2: auth + target database exists and online ─────────────────────
echo "[4/5] Verifying Neo4j auth and database '${DB_NAME}' is online..."
python - <<'PYEOF'
import os, sys
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

url, user, pw, db = (os.environ['NEO4J_URL'], os.environ['NEO4J_USER'],
                     os.environ['NEO4J_PASSWORD'], os.environ['NEO4J_DB'])
try:
    drv = GraphDatabase.driver(url, auth=(user, pw))
    drv.verify_connectivity()
except AuthError as e:
    print(f"ERROR: Neo4j auth failed for user '{user}': {e}", file=sys.stderr); sys.exit(1)
except ServiceUnavailable as e:
    print(f"ERROR: Neo4j unreachable at {url}: {e}", file=sys.stderr); sys.exit(1)

with drv.session(database='system') as s:
    rows = {r['name']: r for r in s.run('SHOW DATABASES').data()}

if db not in rows:
    print(f"ERROR: Database '{db}' does not exist on this DBMS.", file=sys.stderr)
    print(f"  Existing: {sorted(rows.keys())}", file=sys.stderr)
    print(f"  Create it in Neo4j Browser:  CREATE DATABASE `{db}`;", file=sys.stderr)
    drv.close(); sys.exit(1)

status = rows[db].get('currentStatus') or rows[db].get('requestedStatus')
if status != 'online':
    print(f"ERROR: Database '{db}' exists but status='{status}', expected 'online'.", file=sys.stderr)
    print(f"  In Neo4j Browser:  START DATABASE `{db}`;", file=sys.stderr)
    drv.close(); sys.exit(1)

with drv.session(database=db) as s:
    s.run('RETURN 1').single()

drv.close()
print(f"  OK  auth=ok  db='{db}'  status=online  writable=yes")
PYEOF

# ── Populate ──────────────────────────────────────────────────────────────────
if [[ ${RUN_POPULATE} -eq 1 ]]; then
    echo "[5/5] Populating database (this can take 45–75 minutes)..."
    bash "${SCRIPT_DIR}/populate.sh"
else
    echo "[5/5] Populate skipped (--skip-populate)."
fi

echo
echo "=== Setup complete ==="
