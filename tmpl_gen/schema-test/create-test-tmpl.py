#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:40:13 2026

@author: icardei

Creates templates that test the CTI schema from an XLS file with the target schema:
    nodes, properties, and relationships
"""

import os
import os.path
import re
import json
import argparse
import pandas as pd

def help_usage():
    return """
python3 create-test-templ.py --xlsfile schema-info.xls --output schematest-tmpls.json

"""

def create_templates(args:argparse.Namespace, with_props:bool):
    """
    """
    def clean_nodename(nn:str) -> str:
        """
        Remove optional source doc scope, like "cwe:" in "cwe:Weakness"
        
        If with_props is False, then it generates templates for nodes and
        relationships.
        Otherwise, it also generates templates for individual properties.
        """
        return re.sub(r"^.*?:", "", nn)
    
    count_limit = 10
    dct_schema = parse_xlsfile(args)
    all_json = list()
    sorted_nodes = sorted(dct_schema['nodes'].keys())
    n = 0
    
    # all non-ATT&CK node names include a document scope label, e.g. cwe:Weakness
    # that is not used by the CTI DB, nor the templates. It's just for keeping track in the
    # source XLS file and for creating the graph
    for nn in sorted_nodes:
        nnshort = clean_nodename(nn)
        for propname in sorted(dct_schema['nodes'][nn]):
            shortname = f"{nnshort}.{propname}"
            comment = f"Node property test for {shortname}"
            short_cmt = ""
            if nn in dct_schema['comments']:
                short_cmt = dct_schema['comments'][nn]
                comment += f" ; Comment: {short_cmt} "
                
            tmpl_txt = f"{comment}: {{{nnshort}.{propname}}}"
            tobj = {"shortname": shortname, "comment": comment, 
                    "text": tmpl_txt, "count_limit": count_limit}
            all_json.append(tobj)
            n += 1
            
            if not with_props:     # stop after the first property if we don't want a template / property
                break
            
    # sort edges by start node/rel/end node :
    for rel in sorted(dct_schema["relationships"]):
        startnd, relname, endnd = rel
        sn = clean_nodename(startnd)
        en = clean_nodename(endnd)
        
        shortname = f"{sn}.{relname}>{en}"
        defprop = dct_schema['nodes'][endnd][0]
        comment = f"Relationship test for {shortname}"
        if rel in dct_schema['comments']:
            comment += f"; Comment: {dct_schema['comments'][rel]}"
            
        tmpl_txt = f"{comment}: {{{sn}.{relname}>{en}.{defprop}}}"
        
        tobj = {"shortname": shortname, "comment": comment, 
                    "text": tmpl_txt, "count_limit": count_limit}
        all_json.append(tobj)
        n += 1

    nnodes = len(dct_schema['nodes'])
    nrels = len(dct_schema['relationships'])
    # print(all_json)
    if with_props:
        outfilename = os.path.splitext(args.output)[0] + "+props.json"
        print(f"Generate templates for {nnodes} nodes, properties, and {nrels} relationships")
    else:
        outfilename = args.output
        print(f"Generate templates for {nnodes} nodes and {nrels} relationships")
    with open(outfilename, "w") as fout:
        json.dump(all_json, fout, indent=4)
        print(f"Saved {n} test templates to file {outfilename}\n")


def parse_xlsfile(args:argparse.Namespace):
    """
    Parses the source XLS file and returns a dict 
    {"nodes": dict_nodeprops, "relationships": lst_rels, "comments": dict_comments}.

    Parameters
    ----------
    args : argparse.Namespace 
        command line args

    with_props : bool
        if True, gene

    Returns
    -------
    dict[str:[str|list|dict[str|list]]]
        {"nodes": dict_nodeprops, "relationships": lst_rels, "comments": dict_comments}.

    """
    def is_interesting(s):
        return type(s) == str and (s.isalnum() or "-" in s or ":" in s)
    
    df = pd.read_excel(args.xlsfile, usecols=[1, 2, 3], header=None, skiprows=2)
    # print(df)
    nodes = []
    rels = []
    comments = dict()
    nodeprops = dict()
    now_reading = "nothing"
    for i in range(df.shape[0]):
        celltxt = df[2][i]
        # print(celltxt, df[3][i])
        if now_reading == "nothing" and celltxt == "Nodes":
            now_reading = "nodes"
        elif now_reading == "nothing" and celltxt == "Relationships":
            now_reading = "relationships"
        elif now_reading == "nodes":
            if is_interesting(celltxt):
                nodename = celltxt.strip()
                nodes.append(nodename)
                proplst = sorted(p.strip() for p in df[3][i].strip().split(", "))
                nodeprops[nodename] = proplst
                if pd.notna(df[1][i]):
                    comments[nodename] = df[1][i]
            else:
                now_reading = "nothing"
        
        elif now_reading == "relationships":
            if is_interesting(celltxt):
                relstr = celltxt.strip()
                reltoks = tuple(tok.strip() for tok in relstr.split(" - "))
                if len(reltoks) != 3:
                    print("SKIP relationship cell", celltxt)
                    continue

                if pd.notna(df[1][i]):
                    comments[reltoks] = df[1][i]
                rels.append(reltoks)
            else:
                now_reading = "nothing"

    return {"nodes": nodeprops, "relationships": rels, "comments": comments}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Creates templates for testing the schema in a CTI DB.

Usage:
"""   + help_usage()
    )
    
    parser.add_argument(
         "--xlsfile",
         "-x",
         type=str,
         required=True,
         help="SourceXLS file with target schema info.",
    )
    
    parser.add_argument(
         "--output",
         "-o",
         type=str,
         required=True,
         help="JSON file with created templates.",
    )
    
    args = parser.parse_args()
    print(args)
    
    create_templates(args, with_props=False)

    create_templates(args, with_props=True)
    

if __name__ == "__main__":
    main()
    
    
    
