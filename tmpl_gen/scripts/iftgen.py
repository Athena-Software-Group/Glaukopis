#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:37:52 2025

@author: icardei

Changes:
    05/01/2026: added default support for non-empty property values with command line option
                --allow_nullprops (default False). Set to True to allow "" or null property values.

"""

import os
import os.path
import argparse
from tmpl_gen.tmpl_parser import TmplGenNeo4j, tool_tmplgen
from tmpl_gen.neo4j_utils import create_ATTACK_db, neo4j_extract_schema
from tmpl_gen.utils import GraphFormatterGraphviz


# this is used for testing and demo:
mitre_ent_attack_filename = "data/enterprise-attack.json"


def task_generate(args):
    if args.genconf == "":
        print("ERROR: genconf option is undefined")
    elif args.dbconf == "":
        print("ERROR: dbconf option is undefined")
    elif args.tmpl == "":
        print("ERROR: tmpl option is undefined")
    else:
        options = {
            "gen_conf_file": args.genconf,
            "templates_file": args.tmpl,
            "neo4j_conf_file": args.dbconf,
            "results_dir": args.results_dir,
            "count_max": args.count_max,
            "verbose": args.verbose,
            "allow_nullprops": args.allow_nullprops
            }
        
    tool_tmplgen(options)



def task_create_db(args):
    if args.dbconf == "":
        print("ERROR: neo4j configuration JSON file missing")
    else:
        print(f"Using source file {args.mitre_file}.")        
        create_ATTACK_db(args.dbconf, args.mitre_file)


def task_get_schema(args):
    if args.dbconf == "":
        print("ERROR: neo4j configuration JSON file missing")
    else:
        outfilename = "db_schema.json" if args.out == "" else args.out
        dct_schema = neo4j_extract_schema(args.dbconf, outfilename)
        
        # (basename, ext) = os.path.split(outfilename)
        nodirname = os.path.splitext(os.path.basename(outfilename))[0]
        gv_filename = os.path.splitext(os.path.basename(outfilename))[0] + ".gv"
         # = os.path.splitext(os.path.basename(outfilename))
        
        fmter = GraphFormatterGraphviz(dct_schema, nodirname)
        fmter.format_save(gv_filename)
        
        
def task_test(args):
    print("NOT IMPLEMENTED")


def task_fail(args):
    print("FAILURE: ", args)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Template-based generation tool for neo4j DB.
Example to generate strings from template file, allowing null/empty/N/A property values:

    python3 iftgen.py --cmd generate --genconf gencfg_default_neo4j.json \\
        --dbconf neo4j-TEST-config.json --tmpl sample-tmpl-attack.json \\
        --results_dir results-dir --allow_nullprops

Example to create test neo4j DB from MITRE ATT&CK Enterprise JSON file:
    python3 iftgen.py --cmd create_db --dbconf neo4j-TEST-config.json 
    
    NOTES:  First, create the DB named in the configuration file on the neo4j server instance.
            This script takes > 40 minutes long to execute, depending on hardware. 
    CAUTION: the neo4j DB will be WIPED OUT and recreated.

Example to extract DB schema to JSON file and to generate graph figure:
    python3 iftgen.py --cmd get_schema --dbconf neo4j-TEST-config.json --out schema.json
        
    
Use command line option --allow_nullprops to permit null, "N/A" or "" property values. 
Default it is False.
"""        
    )
    
    parser.add_argument(
         "--cmd",
         "-c",
         required=True,
         choices=["generate", "create_db", "get_schema", "test"],
         help="Command to execute.",
    )

    parser.add_argument(
         "--genconf",
         "-g",
         type=str,
         required=False,
         default="",
         help="Generation configuration file name (JSON format).",
    )
    
    parser.add_argument(
         "--dbconf",
         "-d",
         type=str,
         required=True,
         default="",
         help="DB configuration file name (JSON format).",
    )
    
    parser.add_argument(
         "--tmpl",
         "-t",
         type=str,
         required=True,
         help="Templates file name (JSON format).",
    )
    
    parser.add_argument(
         "--results_dir",
         "-r",
         type=str,
         required=False,
         default="results-dir",
         help="Directory name for storing results.",
    )

    parser.add_argument(
         "--mitre_file",
         "-m",
         type=str,
         required=False,
         default=mitre_ent_attack_filename,
         help="Source MITRE ATT&CK JSON file needed to create a neo4j DB. \n\
Only used by create_db command.\n\
CAUTION: The neo4j DB will be wiped out and recreated.",
    )

    parser.add_argument(
         "--out",
         "-o",
         type=str,
         required=False,
         default="schema-graph.json",
         help="Output file name for get_schema command.",
    )

    parser.add_argument(
         "--count_max",
         "-M",
         type=int,
         required=False,
         default=-1,
         help="Override count limit generation property: generate max. this many triples for each template.",
    )

    parser.add_argument(
         "--allow_nullprops",
         action='store_true',
         required=False,
         help='Include --allow_nullprop to allow null or empty/"" or "N/A" property values. Default it is False.',
    )

    parser.add_argument('--verbose', '-v', action='count', default=0,
        help="Increase verbosity level."
    )
    
    args = parser.parse_args()
    print(args)
    
    handlers = {
        "generate": task_generate,
        "create_db": task_create_db,
        "get_schema": task_get_schema,
        "test": task_test,
    }
    
    handlers.get(args.cmd, task_fail)(args)
    

if __name__ == "__main__":
    main()
    
    
    