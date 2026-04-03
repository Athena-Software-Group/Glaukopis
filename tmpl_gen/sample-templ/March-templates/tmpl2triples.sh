#!/bin/bash

# Run the iftgen.py script that generates triples from templates and the current CTI DB content:

if [ "$#" -lt 2 ]; then
    echo "Error: minimum 2 arguments required"  >&2
    echo "Usage: $0 tmpl.json results_dir [count_limit=2000]" >&2
    echo "CAUTION: results directory will be erased first."  >&2
    exit 1
fi


# Python script that generates triples:
PY_IFTGEN="../../scripts/iftgen.py"

# generation configuration parameters:
GENCONF="gencfg_default_neo4j.json"

# neo4j connection and DB parameters: URL, USER, PASSWORD, DB:
NEO4JCONF="neo4j-local-config.json"

# source template JSON file, maybe generated from a Word DOCX file:
TMPL_JSON="$1"

# output + results directory name:
RESULTS_DIR="$2"

echo "CAUTION: RESULTS DIRECTORY ${RESULTS_DIR} IS RECREATED WITH NEW FILES"
echo

# override max. number of triples generated from one template
#   (specified in the $GENCONF file):
COUNT_MAX="${3:-2000}"

rm -vfr $RESULTS_DIR

python3 $PY_IFTGEN --cmd generate --genconf $GENCONF \
        --dbconf $NEO4JCONF --tmpl $TMPL_JSON \
        --results_dir $RESULTS_DIR --count_max $COUNT_MAX
