# Run the iftgen.py script that generates triples from templates and the current CTI DB content:

# Python script that generates triples:
PY_IFTGEN="../../scripts/iftgen.py"

# generation configuration parameters:
GENCONF="gencfg_default_neo4j.json"

# neo4j connection and DB parameters: URL, USER, PASSWORD, DB:
NEO4JCONF="neo4j-local-config.json"

# source template JSON file, maybe generated from a Word DOCX file:
TMPL_JSON="templates-aligned-2026-02.json"

# output + results directory name:
RESULTS_DIR="results-dir"

echo "CAUTION: RESULTS DIRECTORY ${RESULTS_DIR} IS RECREATED WITH NEW FILES"
echo

# override max. number of triples generated from one template
#   (specified in the $GENCONF file):
COUNT_MAX=2000

rm -vfr $RESULTS_DIR

python3 $PY_IFTGEN --cmd generate --genconf $GENCONF \
        --dbconf $NEO4JCONF --tmpl $TMPL_JSON \
        --results_dir $RESULTS_DIR --count_max $COUNT_MAX
