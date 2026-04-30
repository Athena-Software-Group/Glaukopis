#!/usr/bin/env python3
"""Extract just AB.MCQ.4 and JS.MCQ.2 from v8_small.txt into a tiny
JSON manifest so we can re-run them in isolation with a meaningful count_max
and decide whether they're actually broken or just sample-size sensitive.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

SRC = Path("tmpl_gen/templates/04292026/Sophia-CTI-Templates-v8_small.txt")
OUT = Path("tmpl_gen/data_generation/_isolate_two.json")

txt = SRC.read_text()
# Convert .txt -> .json via tmpl_docx2json then prune
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tmpl_docx2json", "tmpl_gen/scripts/tmpl_docx2json.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class A:
    input = str(SRC)
    out = str(OUT.with_suffix(".full.json"))
    count_limit = 200

A_obj = A()
mod.extract_templates_from_txt(A_obj)
# the helper writes inside extract_templates(); call it directly:
templates = mod.extract_templates_from_txt(A_obj)
keep_ids = {"AB.MCQ.4", "JS.MCQ.2"}
keep = [t for t in templates if t.get("shortname") in keep_ids]
for t in keep:
    t["count_limit"] = 200
print(f"Selected {len(keep)} templates: {[t['shortname'] for t in keep]}")
OUT.write_text(json.dumps(keep, indent=4))
print(f"Wrote {OUT}")
