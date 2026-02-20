#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:36:46 2026

@author: icardei

Extracts and converts templates in jsonl format received 
from Salman on 10/17/2025.

"""

import os
import os.path
import sys
import json


print(sys.argv)

prefixmap = {"CAPEC":"A", "CISA": "S", "CVE": "V", "CWE": "W", 
             "EPSS": "S", "MITRE": "M", "MITRE_ENGAGE": "E"}

def convert_file(fn:str) -> str:
    """
    """
    # dom = os.path.splitext(fn)[0].split("_")[1]
    base = os.path.splitext(fn)[0]
    pos = base.find("_")
    dom = base[pos+1:]
    
    prefix = prefixmap[dom]
    with open(fn, "r") as fin:
        # skip initial lines with # at start:
        for line in fin:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                break
        
        lines = ["[", line]
        for line in fin:
            line = line.strip()
            # print(f"line [{line}]")
            if len(line) > 0 and line[0] == "#":
                break
            if len(line) > 0:
                if line == "}":    # IFT_CVE.jsonl file bug
                    line = "},"
                lines.append(line)
            
        # print(f"last line: [{lines[-1]}]\n")
        lines.append("]")
        if lines[-2][-1] == ",":
            lines[-2] = lines[-2][:-1]
        jsoncode = "".join(lines)
        
        # print("JSON CODE:")
        # print(jsoncode)
        
        jsonlst = json.loads(jsoncode)
        
        counter = 1
        gentmplst = list()
        
        for tmpl in jsonlst:
            
            # print("tmpl is ", tmpl, "--\n")
            
            gt = f"{prefix}.{counter} Instruction: {tmpl['instruction']}\n\n\
Question: {tmpl['input']}\n\nAnswer: {tmpl['output']}"    
            gentmplst.append(gt)
            counter += 1
            
        filetmpls = "\n\n\n".join(gentmplst)
        txt = f"{fn}\n\n{filetmpls}\n\n"
        return txt
        
    
def main():
    for fn in sys.argv[1:]:
        if fn.endswith(".jsonl"):
            txt = convert_file(fn)
            print(txt)

if __name__ == '__main__':
    main()
    
    