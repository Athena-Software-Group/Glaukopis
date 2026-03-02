# This script is used to generate test templates for the CTI DB schema.
# It parses an XLS file $SCHEMA_XLS with node and relationship details and generates
#      test templates in two JSON files, $TMPL_FILE and similar, but ending with "+props.json".
#      The first file has test templates for all nodes and relationships, using just
#      one property. 
#      The second file has test templates for all nodes, all their properties
#      and relationships, about 290 templates in total.

# This XLS file is edited by hand with schema informatin
#     from the ../docx/CTI-DB-schema-details.docx:
SCHEMA_XLS="../docs/cti-schema-target-2026-02.xlsx"

# test template JSON file:
TMPL_FILE="test-templates.json"

# script that generates the templates:
GEN_PY="./create-test-tmpl.py"

# this command generates TWO template files: test-templates.json and test-templates+props.json.
# the first has one template/node and one template/relationship
# the second file has one template/(node property) and one template/relationship

python3 $GEN_PY --xlsfile $SCHEMA_XLS --output $TMPL_FILE
