#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 20:50:27 2025

@author: icardei

Converts tmpl_gen generated text JSON files to Alpaca version.

"""

import os
import os.path
from pathlib import Path
import argparse
import json
import re
import dateutil

key_gen_lst = "generated_strings"

# Sentinel substitutions for literal JSON characters in TEXTSECTION bodies.
# See tmpl_gen/templates/04292026/Sophia-CTI-Templates-JSON-v8.txt section B
# for the rationale (the tmpl_parser TEXTSECTION rule reserves `{}[]`).
JSON_SENTINELS = (("<OBR>", "{"), ("<CBR>", "}"), ("<OBK>", "["), ("<CBK>", "]"))

def unescape_json_sentinels(s:str) -> str:
    for src, dst in JSON_SENTINELS:
        s = s.replace(src, dst)
    return s

def help_usage():
    return """
python3 to_alpaca.py --results_dir results_dir --output ift_data.json
        --from 2025-03-04T21:22:53.133Z --to 2025-12-05T11:22:53.133Z
"""

def has_tags(jsin:dict, tags_any:list[str], tags_all:list[str]) -> bool:
    """
    Checks if the triple in dict d complies with the tags in optional command args.

    Parameters
    ----------
    jsin : dict
        dict with JSON object for an input file.
    args : argparse.Namespace
        command line arguments object.

    Returns
    -------
    bool
        True if the template complies with the tags; False otherwise.

    """
    ttags = jsin["template_object"].get("tags", [])
    # print("ttags", ttags, " tags_all:", tags_all, " tags_any:", tags_any, "\n\n")
    if len(tags_all) > 0 and tags_all[0] != "" and not all(t in tags_all for t in ttags):
        return False
    
    if len(tags_any) > 0 and tags_any[0] != "" and not any(t in tags_any for t in ttags):
        return False    
    return True


def parse_triples(args:argparse.Namespace, jsin) -> list[dict[str, str]]:
    """
    Parse strings in lst_txt with format:
        Instruction: ....
        Question: ....
        Answer: ....
        
    to Alpaca format. See below.

    Parameters
    ----------
    args: argparse.Namespace
        command line parameters
        
    jsin: JSON object
    
    Returns
    -------
    list[dict[str, str]]
        list of dictionaries with {"instruction":..., "input":..., "output":...}

    """
    # Lazy quantifiers on Instruction/Question so the FIRST occurrence of
    # each section header acts as the delimiter. The greedy variant
    # (kept below for reference) matches up to the LAST occurrence of
    # `Question:` / `Answer:` in the text, which silently corrupts any
    # template whose body legitimately repeats those markers (e.g. an
    # AthenaBench-aligned RMS template whose assistant output ends with
    # a `Answer: M####, M####` final-line directive).
    # The Question:/Answer: separators are anchored to a preceding newline
    # because tmpl_gen always renders headers at the start of a line
    # (verified across v5/v7/v8 outputs: prev-context is always `\n\n`).
    # The unanchored `\s+Header:` form mis-fires on rendered text whose
    # CVE/product descriptions legitimately contain the substring
    # `Answer:` mid-sentence (e.g. CVE-2025-31810 -- PickPlugins "Question
    # Answer" plugin -- or CVE-2024-36229 -- Adobe AEM XSS description).
    # pattern = r"Instruction: (.*)\s+Question: (.*)\s+Answer: (.*)\s*"
    pattern = r"Instruction: (.*?)\n\s*Question: (.*?)\n\s*Answer: (.*)\s*"
    lst_out = list()
    shortname = jsin["template_object"].get("shortname", "")
    
    instruction_override = args.instruction
        
    # take up to count_max triples:    
    lst_txt = jsin[key_gen_lst] if args.count_max < 0 else jsin[key_gen_lst][:args.count_max]

    do_unescape = not getattr(args, "no_unescape_json_sentinels", False)
    for txt in lst_txt:
        m = re.search(pattern, txt, flags=re.S)
        if m:
            instr = m.group(1).strip() if instruction_override == "" else instruction_override
            d = {"instruction": instr, "input": m.group(2).strip(),
                 "output": m.group(3).strip()}
            if do_unescape:
                d["instruction"] = unescape_json_sentinels(d["instruction"])
                d["input"] = unescape_json_sentinels(d["input"])
                d["output"] = unescape_json_sentinels(d["output"])
            if shortname:
                d["shortname"] = shortname
            lst_out.append(d)
    
    print(f"Converted to Alpaca {len(lst_out)} triples.")
    return lst_out
            
            
def convert_to_alpaca(args:argparse.Namespace):
    # print(type(args))
    dir_path = Path(args.results_dir)
    if not dir_path.is_dir():
        print(f"Error: '{dir_path}' is not a directory or does not exist.")
        return 
    
    lst_tags_any = args.tags_any.split(",")
    lst_tags_all = args.tags_all.split(",")

    lst_all_triples = []
    for f in dir_path.iterdir():
        if f.is_file() and (not f.name.startswith("_")) and f.suffix.lower() == ".json":
            print("Processing:", f, end=": ")
            with f.open("r") as fin:
                jsin = json.load(fin)
                lst_triples = list()
                if has_tags(jsin, lst_tags_any, lst_tags_all):
                    lst_txt = jsin.get(key_gen_lst, list())
                    lst_triples = parse_triples(args, jsin)
                    lst_all_triples.extend(lst_triples)
                # print(f"converted {len(lst_triples)} triples.")
    print(f"Total converted {len(lst_all_triples)} triples.")
    with open(args.output, "w") as f:
        str_out = json.dumps(lst_all_triples, indent=4)
        f.write(str_out)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Converts JSON output files from tmpl_gen format to Aplaca format.

Usage:
"""   + help_usage()
    )
    
    parser.add_argument(
         "--results_dir",
         "-r",
         type=str,
         required=True,
         help="Directory with the tmpl_gen generated JSON files.",
    )
    
    parser.add_argument(
         "--output",
         "-o",
         type=str,
         required=True,
         help="Alpaca format output JSON file.",
    )

    parser.add_argument(
         "--instruction",
         "-i",
         type=str,
         required=False,
         default="",
         help="Override triple instruction with this string.",
    )

    parser.add_argument(
         "--count_max",
         "-M",
         type=int,
         required=False,
         default=-1,
         help="Select (and convert to Alpaca) up to count_max number of triples from the same template.",
    )

    parser.add_argument(
         "--from",
         type=str,
         default="",
         required=False,
         help="Start date [optional].",
    )

    parser.add_argument(
         "--to",
         type=str,
         default="",
         required=False,
         help="Start date [optional].",
    )

    parser.add_argument(
         "--tags_any",
         type=str,
         default="",
         required=False,
         help="Tags, any [optional].",
    )
    
    parser.add_argument(
         "--tags_all",
         type=str,
         default="",
         required=False,
         help="Tags, all required [optional].",
    )

    parser.add_argument(
         "--no_unescape_json_sentinels",
         action="store_true",
         default=False,
         help="Disable JSON sentinel post-processing (<OBR>/<CBR>/<OBK>/<CBK> -> {}[]).",
    )

    args = parser.parse_args()
    print(args)
    
    convert_to_alpaca(args)
    

if __name__ == "__main__":
    main()
    
    
    
