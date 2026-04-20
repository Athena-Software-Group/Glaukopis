#!/bin/bash

# Installs all Python dependencies for the Athena CTI database population script.
# Run from the athena_cti_db/utils/ directory with your target environment active.
#
# Usage:
#   ./install.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -e

echo "=== Installing athena_cti_db dependencies ==="
pip install --upgrade pip
pip install -r "${REPO_DIR}/requirements.txt"

echo
echo "=== Installation complete ==="
