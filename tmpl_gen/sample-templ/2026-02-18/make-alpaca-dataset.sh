# Converts IFT triples from multiple JSON files in a results directory
# that are in the templ_gen JSON format to the Alpaca JSON format
# with all triples in just one file.

# script for conversion:
PY_SCRIPT="../../scripts/to_alpaca.py"

# directory with tmpl_gen generated triples (one .json file per template)
SOURCE_DIR="results-dir"

# output alpaca formatted json file: USE THIS FILE FOR FINE TUNING
ALPACA_JSON="ift_data_alpaca.json"

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
