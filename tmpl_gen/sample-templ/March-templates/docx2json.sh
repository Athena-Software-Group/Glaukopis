#!/bin/bash

# Extracts templates from .docx file where they are edited and nicely described and
# exports them to tmpl_gen format JSON file, ready to be used by iftgen.py for triples generation.

PY_SCRIPT="../../scripts/tmpl_docx2json.py"

if [ "$#" -lt 1 ]; then
    echo "Error: Exactly 1 argument is required"
    echo "Usage: $0 tmpl.docx [count_limit]" >&2
    exit 1
fi

# Word document with templates:
SOURCE_DOC="$1"

# Output file with templates and parameters  in the current directory, in JSON format:
TMPL_JSONFILE="${SOURCE_DOC##*/}"
TMPL_JSONFILE="${TMPL_JSONFILE%.*}.json"

# Set a generation count limit for each template. It can/will be overridden.
COUNT_LIMIT=${3:-10}


python3 ${PY_SCRIPT} -i ${SOURCE_DOC} -o ${TMPL_JSONFILE} --count_limit ${COUNT_LIMIT}
