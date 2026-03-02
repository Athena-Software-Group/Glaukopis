#
# Extracts the target CTI DB schema Graphviz graph from the template-stats.xls file
# that lists statistics for 02/2026 templates in file templates-aligned-2025-02.docx
#
# Must have Graphiz program dot installed. 

# XLS_IN="template-stats.xls"
XLS_IN="../../docs/cti-schema-target-2026-02.xlsx"

GV_OUT="CTI-schema-target-2026-02.gv"
PDF_OUT="${GV_OUT%.gv}.pdf"
PNG_OUT="${GV_OUT%.gv}.png"

# Generate a graphviz .gv file:
python3 ../../scripts/schemagraph.py $XLS_IN $GV_OUT

# convert to PDF:
dot -Tpdf $GV_OUT > $PDF_OUT

dot -Tpng $GV_OUT > $PNG_OUT
