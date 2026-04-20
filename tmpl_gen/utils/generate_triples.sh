#!/bin/bash

# End-to-end triple / Alpaca dataset generation for tmpl_gen.
# Default inputs: the two Sophia-CTI-Templates-04022026 files.
#   - Sophia-CTI-Templates-04022026.docx         (base)
#   - Sophia-CTI-Templates-04022026-benchmark-addendum.txt
#
# Pipeline:
#   1. Preflight  : TCP/auth check against Neo4j, verify DB has nodes loaded
#   2. docx2json  : each template file -> tmpl_gen JSON
#   3. merge      : concatenate all template JSONs into one
#   4. iftgen     : templates + Neo4j graph -> per-template triples
#   5. to_alpaca  : triples -> one Alpaca-format JSON
#
# Usage:
#   ./generate_triples.sh [OPTIONS]
#
# Options:
#   -t, --template FILE   Add a template file (repeatable). If omitted, uses
#                         the two default 04022026 files.
#   -o, --outdir DIR      Output root (default: tmpl_gen/utils/output)
#   -l, --count-limit N   per-template generation hint for docx2json (default: 10)
#   -m, --count-max N     max triples per template in iftgen (default: 2000)
#       --skip-preflight  Skip Neo4j preflight (not recommended)
#   -h, --help            Show this help
#
# Configuration precedence (highest wins):
#   1. CLI flags
#   2. Shell env vars (NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)
#   3. tmpl_gen/utils/.env
#   4. Built-in defaults

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Load .env (existing env wins) ─────────────────────────────────────────────
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
NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"
NEO4J_DB="${NEO4J_DB:-athena-cti-db}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TEMPLATES_DIR="${REPO_DIR}/templates"
DEFAULT_TEMPLATES=(
    "${TEMPLATES_DIR}/Sophia-CTI-Templates-04022026.docx"
    "${TEMPLATES_DIR}/Sophia-CTI-Templates-04022026-benchmark-addendum.txt"
)
OUTDIR="${SCRIPT_DIR}/output"
COUNT_LIMIT=10
COUNT_MAX=2000
SKIP_PREFLIGHT=0
TEMPLATES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--template)       TEMPLATES+=("$2"); shift 2 ;;
        -o|--outdir)         OUTDIR="$2"; shift 2 ;;
        -l|--count-limit)    COUNT_LIMIT="$2"; shift 2 ;;
        -m|--count-max)      COUNT_MAX="$2"; shift 2 ;;
        --skip-preflight)    SKIP_PREFLIGHT=1; shift ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; started=1; next} started{exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ ${#TEMPLATES[@]} -eq 0 ]] && TEMPLATES=("${DEFAULT_TEMPLATES[@]}")

for f in "${TEMPLATES[@]}"; do
    [[ -f "$f" ]] || { echo "ERROR: template file not found: $f" >&2; exit 1; }
done

GENCONF="${REPO_DIR}/data_generation/gencfg_default_neo4j.json"
[[ -f "${GENCONF}" ]] || { echo "ERROR: genconf not found: ${GENCONF}" >&2; exit 1; }

mkdir -p "${OUTDIR}"
TMPL_OUT="${OUTDIR}/templates"
TRIPLES_OUT="${OUTDIR}/triples"
MERGED_JSON="${OUTDIR}/templates-merged.json"
ALPACA_JSON="${OUTDIR}/ift_data.json"
RUNTIME_DBCONF="${OUTDIR}/_neo4j-runtime.json"
mkdir -p "${TMPL_OUT}"

echo "=== tmpl_gen triple generation ==="
echo "  Neo4j URL   : ${NEO4J_URL}"
echo "  Neo4j user  : ${NEO4J_USER}"
echo "  Database    : ${NEO4J_DB}"
echo "  Templates   : ${#TEMPLATES[@]}"; for f in "${TEMPLATES[@]}"; do echo "                ${f}"; done
echo "  Output dir  : ${OUTDIR}"
echo "  count_limit : ${COUNT_LIMIT}"
echo "  count_max   : ${COUNT_MAX}"
echo

# ── Generate runtime Neo4j config for iftgen --dbconf ─────────────────────────
python3 - "$RUNTIME_DBCONF" <<PYEOF
import json, os, sys
url = os.environ['NEO4J_URL']
# iftgen expects a bolt:// scheme; neo4j:// is equivalent for a single instance
if url.startswith('neo4j://'): url = 'bolt://' + url[len('neo4j://'):]
cfg = {"uri": url,
       "auth": [os.environ['NEO4J_USER'], os.environ.get('NEO4J_PASSWORD', '')],
       "db_name": os.environ['NEO4J_DB'],
       "nickname": "ASG-CTI"}
open(sys.argv[1], 'w').write(json.dumps(cfg, indent=2) + '\n')
PYEOF

# ── Preflight: TCP + auth + DB exists + has nodes ─────────────────────────────
if [[ ${SKIP_PREFLIGHT} -eq 0 ]]; then
    echo "[1/5] Neo4j preflight..."
    if [[ -z "${NEO4J_PASSWORD}" ]]; then
        echo "ERROR: NEO4J_PASSWORD is empty (set in .env or export it)." >&2; exit 1
    fi
    python3 - <<'PYEOF'
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
    echo "[1/5] preflight SKIPPED"
fi

# ── Step 2: template files -> tmpl_gen JSON ───────────────────────────────────
echo "[2/5] Converting ${#TEMPLATES[@]} template file(s) to JSON..."
PY_DOCX2JSON="${REPO_DIR}/scripts/tmpl_docx2json.py"
TMPL_JSONS=()
for f in "${TEMPLATES[@]}"; do
    base="$(basename "${f%.*}")"
    out_json="${TMPL_OUT}/${base}.json"
    echo "  ${f}"
    echo "    -> ${out_json}"
    python "${PY_DOCX2JSON}" -i "${f}" -o "${out_json}" --count_limit "${COUNT_LIMIT}"
    TMPL_JSONS+=("${out_json}")
done

# ── Step 3: merge template JSONs ──────────────────────────────────────────────
echo "[3/5] Merging template JSONs -> ${MERGED_JSON}"
python3 - "$MERGED_JSON" "${TMPL_JSONS[@]}" <<'PYEOF'
import json, sys
out = sys.argv[1]; merged = []
seen = set()
for src in sys.argv[2:]:
    for t in json.load(open(src)):
        key = t.get('shortname') or t.get('comment')
        if key in seen:
            print(f"  WARN duplicate shortname '{key}' in {src} — keeping first", file=sys.stderr); continue
        seen.add(key); merged.append(t)
open(out, 'w').write(json.dumps(merged, indent=2) + '\n')
print(f"  merged {len(merged)} templates")
PYEOF

# ── Step 4: iftgen (templates + graph -> triples) ─────────────────────────────
echo "[4/5] Generating triples (count_max=${COUNT_MAX})..."
rm -rf "${TRIPLES_OUT}"
python "${REPO_DIR}/scripts/iftgen.py" \
    --cmd generate \
    --genconf "${GENCONF}" \
    --dbconf "${RUNTIME_DBCONF}" \
    --tmpl "${MERGED_JSON}" \
    --results_dir "${TRIPLES_OUT}" \
    --count_max "${COUNT_MAX}"

# ── Step 5: to_alpaca ─────────────────────────────────────────────────────────
echo "[5/5] Converting triples -> Alpaca JSON: ${ALPACA_JSON}"
python "${REPO_DIR}/scripts/to_alpaca.py" \
    --results_dir "${TRIPLES_OUT}" \
    --output "${ALPACA_JSON}" \
    --count_max -1

echo
echo "=== Done ==="
echo "  Merged templates : ${MERGED_JSON}"
echo "  Triples dir      : ${TRIPLES_OUT}"
echo "  Alpaca dataset   : ${ALPACA_JSON}"
