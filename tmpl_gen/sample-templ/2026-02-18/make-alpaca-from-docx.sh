# Generates IFT triples from templates written in a Word DOCX file.

# STEPS:
# 1. extract templates from source Word doc.
#    input: templates-aligned-2026-02.docx
#    output: templates-aligned-2026-02.json, the file with the templates in JSON, ready for iftgen.py
bash tmpl_docx2json.sh

# 2. generate triples from templates in tmpl_gen format + CTI DB to JSON
#   format (one file per template in results-dir directory):
bash make-triples.sh

# MUST DO: examine generation results in file results-dir/_results-report.json.
#          look for errors, with exception message for failed templates.
#          NOTE: many fail due to wrong/insufficient neo4j CTI DB schema and spec. errors

# 3. convert triples from results_dir to Alpaca format, in file ift_data_alpaca.json:
bash make-alpaca-dataset.sh
