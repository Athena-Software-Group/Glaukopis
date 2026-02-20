#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 13:11:51 2026

@author: icardei


Extracts templates written in a Word docx file to a file in JSON tmpl_gen format.
The output JSON file can be used as input for iftgen.py to generate an IFT dataset.

Usage example:

    python3 tmpl_docx2json.py -i templates-aligned-2025-02.docx -o templates-aligned-2025-02.json --count_limit 3000

Here is an example for a template writte in a Word document:

"1.1 Instruction: You are a CTI expert that gives precise and concise answers. Question: What course of action should be taken to mitigate the attack pattern {ap:attack-pattern.name} ({ap.id})? Answer: To mitigate {ap.name}, organizations should implement the following course of action: {coa:ap.mitigates<course-of-action.name}. Description: {coa.description}.
Summary: {ap.mitigates<course-of-action|description}
Schema: course-of-action - mitigates - attack-pattern"

Format:
line 1: tmpl ID: xxx.yyy   string
line 1: Instruction: <instruction text> Question: <question text> Answer: <answer text>
OPTIONAL line 2: Summary: <summary template code> 
OPTIONAL next line: Schema: <relationship 1>[, <relationship 2>]*


NOTE:
    It only accepts as input DOCX files, NOT older .DOC files.


TODO: Generalize the input format, like .txt, .doc..,
"""

import docx
import json
import argparse
import re

def get_text_lines(filename:str) -> list[str]:
    """
    Returns the lines of text from a source file.
    NOTE: does not work wth .doc format.
    """
    if filename.endswith(".docx") or filename.endswith(".DOCX"): 
        doc = docx.Document(filename)
        lst_lines = list(doc.paragraphs)
    else:
        # treat like a text file:
        with open(filename, "r") as fin:
            lst_lines = list(fin)
    return lst_lines


def extract_templates(args):
    lst_para = get_text_lines(args.input)
    lst_tmpls = list()
    re_pattern = r"^(.*) Instruction: (.*) Question: (.*) Answer: (.*)$"
    
    i = 0
    while i < len(lst_para):
        line = lst_para[i].text.strip()
        i += 1
        # print(i, ":", line)
        if len(line) < 50:   # skip titles, headings, etc.
            print(f"--- Skip short line {i}: {line}\n")
            continue
        
        re_match = re.search(re_pattern, line)
        if not re_match:
            print(f"\n*** Skip no match line {i}: {line}\n")
            continue
        
        groups = list(re_match.groups()) 
        if len(groups) != 4:
            print(f"\n@@@ line {i} Template format error. Line:\n{line}\nGroups: {groups}\n")
            continue
        t_id, t_instr, t_q, t_a = groups
        t_text = f"Instruction: {t_instr}\n\nQuestion: {t_q}\n\nAnswer: {t_a}"
        tmpl = {"shortname": t_id, "comment": t_id, "text": t_text,
                "source_file": args.input, "source_line": i, 
                "count_limit": args.count_limit}
                
        lst_prefixes = ["Summary", "Schema"]
        for prefix in lst_prefixes:
            if i == len(lst_para):
                break
            line = lst_para[i].text.strip()
            ppref = f"{prefix}: "
            if line.startswith(ppref):
                tmpl[prefix.lower()] = line.split(ppref)[1]
                i += 1
        lst_tmpls.append(tmpl)
    
    with open(args.out, "w") as fout:
        json_txt = json.dumps(lst_tmpls, indent=4)
        fout.write(json_txt)
        
    print(f"\nWrote {len(lst_tmpls)} templates to file {args.out}")
    
    
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Extract templates from a Word doc/x file to a JSON file to be used as input to ifgen.
Usage example:

    python3 tmpl_docx2json.py -i templates-aligned-2025-02.docx -o templates-aligned-2025-02.json --count_limit 3000
"""
    )
    
    parser.add_argument(
         "--input",
         "-i",
         type=str,
         required=True,
#         default="",
         help="Word document input file (docx/doc format)"
    )
    
    parser.add_argument(
         "--count_limit",
         "-c",
         type=int,
         required=False,
         default=3000,
         help="Count limit parameter for triples generated from one template."
    )
    
    
    parser.add_argument(
         "--out",
         "-o",
         type=str,
         required=True,
#         default="",
         help="Output file, in JSON format."
    )
    
    args = parser.parse_args()

    extract_templates(args)


if __name__ == "__main__":
    main()
    
