#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 12:54:16 2026

@author: icardei

Generates Graphviz graph from template stats XLS file.

Usage: python3 schemagraph.py template-stats.xls outputfile.gv

Use dotty to visualize the .gv file.

Use dot -Tpdf file.gv outputfile.pdf to export to PDF format.
"""

import pandas as pd
import sys
# import os.path


def make_gv(fn:str, fnout:str):
    df = pd.read_excel(fn, usecols=[2], header=None, skiprows=2)
    # print(df)
    nodes = []
    rels = []
    state = 0   # reading nodes
    for i in range(df.shape[0]):
        if state == 0:
            if type(df[2][i]) == float:
                state = 1
            else:
                nodes.append(df[2][i].strip())
        elif state == 1:
            if df[2][i] == "Relationships":
                state = 2
        else:
            rels.append(df[2][i].strip())

    with open(fnout, "w") as fout:
        fout.write("digraph schema_athena_cti {\n\
//    rankdir LR;\n\
    node [shape=box, style=filled];\n")
    
        for node in nodes:
            bgcolor = get_bgcolor(node)
            fout.write(f'"{node}" [color=navy, fontcolor=indigo, fillcolor={bgcolor}];\n')
            
        for rel in rels:
            print(rel)
            n1, rn, n2 = (s.strip() for s in rel.split(" - "))
            fout.write(f'"{n1}" -> "{n2}" [label="{rn}"];\n')
        fout.write("}")
    return df
    

def get_bgcolor(node:str) -> str:
    # col = "lightblue"
    col = "khaki1"
    coldct = {"cve": "tan1", "cwe": "wheat1", 
              "capec": "palegreen", "en": "lightblue", "kev": "lightpink"}
    for d,c in coldct.items():
        if node.startswith(f"{d}:"):
            col = c 
            break
    return col
    

if __name__ == "__main__":    
    # xlsfile = "template-stats.xls"
    if len(sys.argv) < 2:
        print("Generates Graphviz graph from template stats XLS file.\n\n")
        print("Usage: python3 schemagraph.py template-stats.xls outputfile.gv\n")
    else:
        xlsfile = sys.argv[1]
        foutname = sys.argv[2]
        # foutname = os.path.splitext(xlsfile)[0] + ".gv"
        df = make_gv(xlsfile, foutname)
