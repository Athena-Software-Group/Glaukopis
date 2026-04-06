#!/bin/bash

# Installs the tmpl_gen package and all its dependencies.
# Run from the tmpl_gen/ directory with your target environment active.
#
# Usage:
#   ./install.sh          # standard install
#   ./install.sh -e       # editable install (recommended for development)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -e

if [[ "$1" == "-e" ]]; then
    echo "=== Installing tmpl_gen in editable mode ==="
    pip install -e "${SCRIPT_DIR}"
else
    echo "=== Installing tmpl_gen ==="
    pip install "${SCRIPT_DIR}"
fi

echo
echo "=== Installation complete ==="
