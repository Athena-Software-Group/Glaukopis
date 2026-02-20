# Extracts templates from .docx file where they are edited and nicely described and
# exports them to tmpl_gen format JSON file, ready to be used by iftgen.py for triples generation.

PY_SCRIPT="../../scripts/tmpl_docx2json.py"

# Word document with templates:
SOURCE_DOC="templates-aligned-2026-02.docx"

# Output file with templates and parameters in JSON format:
TMPL_JSONFILE="templates-aligned-2026-02.json"

# Set a generation count limit for each template. It can/will be overridden.
COUNT_LIMIT=10

python3 $PY_SCRIPT -i $SOURCE_DOC -o $TMPL_JSONFILE --count_limit $COUNT_LIMIT
