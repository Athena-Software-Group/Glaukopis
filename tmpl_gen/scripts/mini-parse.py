#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 16:44:09 2025

@author: icardei

Parses athenabench/runs-mini output and extracts useful info.

"""

import json
import os
import sys
from pathlib import Path
import argparse
import re


def help_usage():
    return "TBD"


def parse_questions(args) -> tuple[str, list[str]]:
    """
    Reads input file and returns instruction and list with questions 

    Parameters
    ----------
    args : TYPE
        DESCRIPTION.

    Returns
    -------
    list[str]
        DESCRIPTION.

    """
    lst_quest_all = list()
    
    with open(args.input, "r") as fin:
        for line in fin:
            jsin = json.loads(line)
            
            # pattern = r"Instruction: (.*?)(\s)+Question: (.*)(\s)+Answer: (.*)(\s)*"
            pattern = r"(.*?)\n\nQuestion: (.*)"
            m = re.search(pattern, jsin["prompt"], flags=re.S)
            if m:
                instruction = m.group(1)
                question = m.group(2)
                lst_quest_all.append(question)
    return (instruction, lst_quest_all)


def task_extract_q(args):
    lst_jsin = []
    lst_pretty_all = list()
    lst_quest_all = list()
    
    with open(args.input, "r") as fin:
        for line in fin:
            jsin = json.loads(line)
            lst_jsin.append(jsin)
            str_jsin = json.dumps(jsin, indent=4)
            lst_pretty_all.append(str_jsin)
            
            # pattern = r"Instruction: (.*?)(\s)+Question: (.*)(\s)+Answer: (.*)(\s)*"
            pattern = r"(.*?)\n\nQuestion: (.*)"
            m = re.search(pattern, jsin["prompt"], flags=re.S)
            if m:
                instruction = m.group(1)
                question = m.group(2)
                lst_quest_all.append(question)

    lst_out = list()
    
    if args.output == "stdout":
        fout_json = sys.stdout
        fout_quests = sys.stdout
    else:
        fn_json = args.output + ".json"
        fout_json = open(fn_json, "w")
        fn_quests = f"{args.output}_questions.txt"
        fout_quests = open(fn_quests, "w")
    
    fout_json.write("[\n")
    for i in range(len(lst_pretty_all)):
        fout_json.write(lst_pretty_all[i])
        if i < len(lst_pretty_all) - 1:
            fout_json.write(",")
        fout_json.write("\n")
    fout_json.write("]\n")        
    fout_json.close()

    fout_quests.write("Instructions:\n")
    fout_quests.write(instruction + "\n\n")
    
    for i in range(len(lst_quest_all)):
        fout_quests.write(f"Question {i}.\n" )
        fout_quests.write(lst_quest_all[i])
        fout_quests.write("\n\n")
    fout_quests.close()
    

categories = [["ATT&CK"], ["CWE"], ["CVE"], ["CAPEC"], ["Android"]]

def categorize_q(args, qcategs:list[list[tuple[int, int, str]]]):
    """
    Categorize questions based on list of keywords indicating CTI areas.
    Prints out categories.
    """
    lst_cat_qs = [list() for cl in qcategs]
    lst_no_categ = list()
    lst_idx_mask_q = list()
    (instruction, lst_quests) = parse_questions(args)
    
    for (i, q) in enumerate(lst_quests):
        mask = 0 
        words = q.split()
        added = False
        for categ in range(len(qcategs)):
            if all( any(w.startswith(cw) for w in words) for cw in qcategs[categ]):
                lst_cat_qs[categ].append((i, q))
                added = True
                mask |= (1 << categ)
        if not added:
            lst_no_categ.append((i, 0, q))        
        lst_idx_mask_q.append((i, mask, q))
    lst_cat_qs.append(lst_no_categ)
    # return lst_cat_qs
    return lst_idx_mask_q
    

def task_categorize(args, qcategs):
    lst_cat_qs = categorize_q(args, qcategs)
    lst_ones = [[1 for (i,m,q) in lst_cat_qs if m & (1<<c) != 0] 
          for c in range(len(qcategs))]

    lst_hist = [len(l2) for l2 in lst_ones]
    lst_hist.append(sum(1 for (i,m,q) in lst_cat_qs if m == 0))

    for c in range(len(qcategs)):
        keywords = ", ".join(qcategs[c])
        print(f"{keywords}: {lst_hist[c]}   ", end="")
        
    print(f"none: {lst_hist[-1]}\n")
        
    if args.cat < 0 or args.cat > len(qcategs):
        print("ERORR: wrong value for cat parameter:", args.cat)
        return
    
    lst_to_print = [(i, m, q) for (i, m, q) in lst_cat_qs 
                        if (m & (1 << args.cat) != 0) or m == 0]
    
    for (i, m, q) in lst_to_print:
        # print(f"Question {i} {m}")
        print(f"Question {i}.")
        print(q, "\n\n")
        

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Parses athenabench/runs-mini output and extracts useful info.

Usage:
"""   + help_usage()
    )
        
    parser.add_argument(
         "--extract",
         "-x",
         action='store_true',
         help="Extract all questions.",
    )

    parser.add_argument(
         "--categorize",
         "-c",
         action='store_true',
         help="Categorize all questions.",
    )
        
    parser.add_argument(
         "--cat",
         type=int,
         required=False,
         default=0,
         help="Question category to print.",
    )
        
    parser.add_argument(
         "--input",
         "-i",
         type=str,
         required=True,
         help="Output JSON file name.",
    )
        

    parser.add_argument(
         "--output",
         "-o",
         type=str,
         required=False,
         default="stdout",
         help="Output JSON file name.",
    )

    if False:
        args = parser.parse_args()
    else:
        input_file = "../../../CTI-bench/athenabench/runs-mini/meta-llama/Llama-3.1-8B-Instruct/MCQ3k-scored.jsonl"
        # output_file = "stdout"
        output_file = "MCQmini"
        args = parser.parse_args(["-i", input_file, "-o", "stdout", "-c"])
    print(args)
    
    if args.extract:
        task_extract_questions(args)
        
    if args.categorize:
        task_categorize(args, categories)
        
        
if __name__ == "__main__":
    main()
    
    
    
