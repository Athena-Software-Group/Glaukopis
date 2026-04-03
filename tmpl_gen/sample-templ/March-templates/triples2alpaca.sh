#!/bin/bash


# Converts IFT triples from multiple JSON files in a results directory
# that are in the templ_gen JSON format to the Alpaca JSON format
# with all triples in just one file.

if [ "$#" -lt 2 ]; then
    echo "Error: Exactly 2 arguments required"  >&2
    echo "Usage: $0 results_dir alpaca_json" >&2
    echo "CAUTION: results directory will be erased first."  >&2
    exit 1
fi




# script for conversion:
PY_SCRIPT="../../scripts/to_alpaca.py"

# directory with tmpl_gen generated triples (one .json file per template)
SOURCE_DIR="$1"

# output alpaca formatted json file: USE THIS FILE FOR FINE TUNING
ALPACA_JSON="$2"

# IMPORTANT:
# Uncomment next line to override triple instruction field with new one:
# OVERRIDE_INSTRUCTIONS="You are a CTI expert who gives precise and concise answers."
# Comment previous line to keep the existing instruction field.

# override max. number of triples generated from one template
# COUNT_MAX=10
# Uncomment next to take all triples generated:
COUNT_MAX=-1


if [[ -z ${OVERRIDE_INSTRUCTIONS} ]]; then    
    python3 $PY_SCRIPT --results_dir $SOURCE_DIR --output $ALPACA_JSON --count_max $COUNT_MAX
else
    echo "Override Instruction field with: ${OVERRIDE_INSTRUCTIONS}"
    python3 $PY_SCRIPT --results_dir $SOURCE_DIR --output $ALPACA_JSON \
	    --instruction "${OVERRIDE_INSTRUCTIONS}" --count_max $COUNT_MAX
fi
