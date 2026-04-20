#!/bin/bash

# End-to-end setup for tmpl_gen on this workstation:
#   1. Loads config from tmpl_gen/utils/.env
#   2. Creates a dedicated conda env (Python >= 3.10) for tmpl_gen, or falls
#      back to a plain virtualenv at tmpl_gen/venv if conda is unavailable
#      or --no-conda is passed
#   3. Installs tmpl_gen and its dependencies (editable install)
#   4. Preflights Neo4j: TCP reachability, auth, target DB online, has nodes
#
# Usage:
#   ./setup.sh [--conda-env NAME] [--python-version X.Y] [--no-conda]
#              [--python PATH] [--recreate] [--skip-install] [--skip-preflight] [-h]
#
# Notes:
#   tmpl_gen's tmpl_parser.py uses datetime.UTC (Python 3.11+), so the default
#   conda Python is 3.12. Older 3.10 envs will fail at runtime.
#
# Configuration precedence (highest wins):
#   1. CLI flags
#   2. Shell env vars already exported
#   3. tmpl_gen/utils/.env (gitignored)
#   4. Built-in defaults
#
# Recognised variables:
#   NEO4J_URL              default: neo4j://127.0.0.1:7687
#   NEO4J_USER             default: neo4j
#   NEO4J_PASSWORD         (required, no default)
#   NEO4J_DB               default: athena-cti-db
#   TMPL_GEN_CONDA_ENV     default: tmpl_gen    (conda env name)
#   TMPL_GEN_PY_VERSION    default: 3.12        (python version for new conda env)
#   TMPL_GEN_PYTHON        default: unset       (override interpreter for venv fallback)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Load .env (existing shell env wins) ───────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    echo "Loading config from ${ENV_FILE}"
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        [[ -z "${_line}" || "${_line}" =~ ^[[:space:]]*# ]] && continue
        _key="${_line%%=*}"; _val="${_line#*=}"; _key="${_key// /}"
        [[ -z "${_key}" ]] && continue
        if [[ "${_val}" =~ ^\"(.*)\"$ ]] || [[ "${_val}" =~ ^\'(.*)\'$ ]]; then
            _val="${BASH_REMATCH[1]}"
        fi
        [[ -z "${!_key:-}" ]] && export "${_key}=${_val}"
    done < "${ENV_FILE}"
    unset _line _key _val
fi

NEO4J_URL="${NEO4J_URL:-neo4j://127.0.0.1:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_DB="${NEO4J_DB:-athena-cti-db}"

USE_CONDA=1
RUN_INSTALL=1
RUN_PREFLIGHT=1
RECREATE_ENV=0
CONDA_ENV_NAME="${TMPL_GEN_CONDA_ENV:-tmpl_gen}"
PY_VERSION_TARGET="${TMPL_GEN_PY_VERSION:-3.12}"
PYTHON_BIN="${TMPL_GEN_PYTHON:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)        CONDA_ENV_NAME="$2"; shift 2 ;;
        --python-version)   PY_VERSION_TARGET="$2"; shift 2 ;;
        --no-conda)         USE_CONDA=0; shift ;;
        --python)           PYTHON_BIN="$2"; USE_CONDA=0; shift 2 ;;
        --recreate)         RECREATE_ENV=1; shift ;;
        --skip-install)     RUN_INSTALL=0; shift ;;
        --skip-preflight)   RUN_PREFLIGHT=0; shift ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; started=1; next} started{exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ── Locate conda (prefer it; fall back to venv) ───────────────────────────────
CONDA_BIN=""
if [[ ${USE_CONDA} -eq 1 ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
    elif [[ -x /opt/anaconda3/bin/conda ]]; then
        CONDA_BIN="/opt/anaconda3/bin/conda"
    elif [[ -x /opt/miniforge3/bin/conda ]]; then
        CONDA_BIN="/opt/miniforge3/bin/conda"
    elif [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
        CONDA_BIN="${HOME}/miniforge3/bin/conda"
    else
        echo "WARN: conda not found; falling back to venv. Install Miniforge for arm64 macOS:" >&2
        echo "      https://conda-forge.org/download/  (or brew install --cask miniforge)" >&2
        USE_CONDA=0
    fi
fi

# ── Parse host/port from NEO4J_URL for the TCP check ──────────────────────────
_url_stripped="${NEO4J_URL#*://}"
NEO4J_HOST="${_url_stripped%%:*}"
NEO4J_PORT="${_url_stripped##*:}"

echo "=== tmpl_gen setup ==="
echo "  Repo dir     : ${REPO_DIR}"
if [[ ${USE_CONDA} -eq 1 ]]; then
    echo "  Env mode     : conda"
    echo "  conda        : ${CONDA_BIN}"
    echo "  Env name     : ${CONDA_ENV_NAME}"
    echo "  Python (new) : ${PY_VERSION_TARGET}"
else
    echo "  Env mode     : venv (fallback)"
fi
echo "  Install deps : $([ ${RUN_INSTALL} -eq 1 ] && echo yes || echo skipped)"
echo "  Neo4j URL    : ${NEO4J_URL}"
echo "  Neo4j user   : ${NEO4J_USER}"
echo "  Database     : ${NEO4J_DB}"
echo

# ── Step 1: TCP preflight (no Python deps yet) ────────────────────────────────
if [[ ${RUN_PREFLIGHT} -eq 1 ]]; then
    echo "[1/4] TCP check to ${NEO4J_HOST}:${NEO4J_PORT}..."
    /usr/bin/python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(3)
try: s.connect(('${NEO4J_HOST}', int('${NEO4J_PORT}'))); print('  OK')
except Exception as e: print(f'  connection failed: {e}', file=sys.stderr); sys.exit(1)
finally: s.close()
" || { echo "ERROR: Neo4j not reachable. Start DBMS in Neo4j Desktop and retry." >&2; exit 1; }
else
    echo "[1/4] TCP preflight SKIPPED"
fi

# ── Step 2: create/activate conda env (or fall back to venv) ──────────────────
if [[ ${USE_CONDA} -eq 1 ]]; then
    CONDA_BASE="$("${CONDA_BIN}" info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if [[ ${RECREATE_ENV} -eq 1 ]] && "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
        echo "[2/4] Removing existing conda env '${CONDA_ENV_NAME}' (--recreate)..."
        "${CONDA_BIN}" env remove -n "${CONDA_ENV_NAME}" -y
    fi
    if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
        echo "[2/4] Reusing conda env '${CONDA_ENV_NAME}'"
    else
        echo "[2/4] Creating conda env '${CONDA_ENV_NAME}' (python=${PY_VERSION_TARGET})..."
        "${CONDA_BIN}" create -n "${CONDA_ENV_NAME}" "python=${PY_VERSION_TARGET}" -y
    fi
    conda activate "${CONDA_ENV_NAME}"
    PYTHON_BIN="$(command -v python)"
    PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')"
    echo "      active: ${PYTHON_BIN} (${PY_VERSION})"
else
    # venv fallback — detect a Python >= 3.10 and create tmpl_gen/venv
    _check_py() {
        local bin="$1"
        [[ -x "$bin" ]] || command -v "$bin" >/dev/null 2>&1 || return 1
        "$bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null
    }
    if [[ -z "${PYTHON_BIN}" ]]; then
        for _c in python3.13 python3.12 python3.11 python3.10 \
                  /opt/anaconda3/envs/python310/bin/python \
                  /opt/anaconda3/envs/python313/bin/python \
                  /opt/anaconda3/bin/python python3; do
            if _check_py "${_c}"; then PYTHON_BIN="${_c}"; break; fi
        done
    fi
    [[ -n "${PYTHON_BIN}" ]] && _check_py "${PYTHON_BIN}" || {
        echo "ERROR: no Python >= 3.10 found. Install one or set TMPL_GEN_PYTHON in .env" >&2; exit 1; }
    VENV_DIR="${REPO_DIR}/venv"
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "[2/4] Creating virtualenv at ${VENV_DIR} with ${PYTHON_BIN}..."
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    else
        echo "[2/4] Reusing virtualenv at ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="$(command -v python)"
    PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')"
    echo "      active: ${PYTHON_BIN} (${PY_VERSION})"
fi

# ── Step 3: install tmpl_gen (editable) + deps ────────────────────────────────
if [[ ${RUN_INSTALL} -eq 1 ]]; then
    echo "[3/4] Installing tmpl_gen (editable) + dependencies..."
    "${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null
    "${PYTHON_BIN}" -m pip install -e "${REPO_DIR}"
else
    echo "[3/4] Install SKIPPED (--skip-install)"
fi

# ── Step 4: Neo4j auth + DB online + has nodes ────────────────────────────────
if [[ ${RUN_PREFLIGHT} -eq 1 ]]; then
    echo "[4/4] Verifying Neo4j auth, DB '${NEO4J_DB}' online, and populated..."
    [[ -n "${NEO4J_PASSWORD:-}" ]] || { echo "ERROR: NEO4J_PASSWORD is empty (set it in .env)." >&2; exit 1; }
    export NEO4J_URL NEO4J_USER NEO4J_PASSWORD NEO4J_DB
    "${PYTHON_BIN}" - <<'PYEOF'
import os, sys
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
url, user, pw, db = (os.environ['NEO4J_URL'], os.environ['NEO4J_USER'],
                     os.environ['NEO4J_PASSWORD'], os.environ['NEO4J_DB'])
try:
    drv = GraphDatabase.driver(url, auth=(user, pw)); drv.verify_connectivity()
except AuthError as e:
    print(f"ERROR: auth failed for '{user}': {e}", file=sys.stderr); sys.exit(1)
except ServiceUnavailable as e:
    print(f"ERROR: unreachable at {url}: {e}", file=sys.stderr); sys.exit(1)
with drv.session(database='system') as s:
    rows = {r['name']: r for r in s.run('SHOW DATABASES').data()}
if db not in rows or (rows[db].get('currentStatus') or rows[db].get('requestedStatus')) != 'online':
    print(f"ERROR: database '{db}' not online. Existing: {sorted(rows)}", file=sys.stderr)
    drv.close(); sys.exit(1)
with drv.session(database=db) as s:
    n = s.run('MATCH (n) RETURN count(n) AS c').single()['c']
drv.close()
if n == 0:
    print(f"ERROR: database '{db}' has 0 nodes — run athena_cti_db/utils/setup.sh first.", file=sys.stderr); sys.exit(1)
print(f"  OK  auth=ok  db='{db}'  nodes={n:,}")
PYEOF
else
    echo "[4/4] Neo4j auth/DB preflight SKIPPED"
fi

echo
echo "=== Setup complete ==="
if [[ ${USE_CONDA} -eq 1 ]]; then
    echo "Activate the conda env in new shells with:"
    echo "  conda activate ${CONDA_ENV_NAME}"
else
    echo "Activate the venv in new shells with:"
    echo "  source ${REPO_DIR}/venv/bin/activate"
fi
echo "Next: generate triples with"
echo "  bash ${SCRIPT_DIR}/generate_triples.sh \\"
echo "       -t ${REPO_DIR}/templates/04202026/Sophia-CTI-Templates-04022026.txt \\"
echo "       -t ${REPO_DIR}/templates/04202026/Sophia-CTI-Templates-04022026-benchmark-addendum.txt \\"
echo "       -o /tmp/sft-0420-smoke -m 3"
