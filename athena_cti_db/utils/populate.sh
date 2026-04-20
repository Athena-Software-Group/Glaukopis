#!/bin/bash

# Populates the Athena CTI Neo4j database with all threat intelligence data.
# Uses whichever Python is active in the current shell (conda env, venv, or system Python).
#
# Usage:
#   ./populate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY_POPULATE="${REPO_DIR}/threat_framework/populate_neo4j_complete.py"

set -e

# ── Run population script ─────────────────────────────────────────────────────
echo "=== Populating Athena CTI Database ==="
echo

python "${PY_POPULATE}"

echo
echo "=== Population complete ==="
