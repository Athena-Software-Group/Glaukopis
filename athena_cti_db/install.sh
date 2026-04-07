#!/bin/bash

# Installs all Python dependencies for the Athena CTI database population script.
# Run from the athena_cti_db/ directory with your target environment active.
#
# Usage:
#   ./install.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -e

echo "=== Installing athena_cti_db dependencies ==="
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo
echo "=== Installation complete ==="
