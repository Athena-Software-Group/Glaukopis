#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 19:25:28 2026

@author: icardei

Reports schema differences using two CTI DB schema JSON files created by 
iftgen.py.

"""

import sys
import json

def readfile(filename:str) -> str:
    with open(filename, "r") as f:
        return f.read()

def find_node(nd:dict, node_lst:list[dict]):
    """
    Returns index in node_lst where node nd is located or -1 if not found.
    """
    for i in range(0, len(node_lst)):
        if node_lst[i]["type"] == nd["type"]:
            break
    else:
        return -1
    return i


def check_edges(nn1:str, lst_rel_n2:list[list[str]], sch2:dict) -> list[tuple[str, str, str]]:
    """
    Return all edges in schema 1 originating in nn1 that are missing in schema 2

    Parameters
    ----------
    nn1 : str
        DESCRIPTION.
    lst_rel_n2 : list[list[str]]
        DESCRIPTION.

    Returns
    -------
    list[tuple[str, str, str]]
        DESCRIPTION.

    """
    XXXXXXXXX
                       
def schemadiff(prefix:str, suffix:str, sch1, sch2):
    nodes1 = sch1['nodes']
    nodes2 = sch2['nodes']
    for n1 in nodes1:
        i2 = find_node(n1, nodes2)
        if i2 < 0:
            print(f"{prefix} node {n1['type']} not found in {suffix}")
            continue
        for prop1 in n1['properties']:
            if prop1 not in nodes2[i2]['properties']:
                print(f"{prefix} property {n1['type']}.{prop1} not found in {suffix}")

    for nn1, lst_rel_n2 in sch1["adj_lst"].items:
        missing_edges = check_edges(nn1, lst_rel_n2)
        if len(missing_edges) > 0:
            for n1, rel, n2 in missing_edges:
                print(f"{prefix} edge {n1} - {rel} - {n2} not found in {suffix}")


def main():
    print(sys.argv)
    if len(sys.argv) < 3:
        print("ERROR: insufficient arguments")
        print("Usage: python3 shemadiff.py schema1.json schema2.json")
    else:
        sch1 = json.loads(readfile(sys.argv[1]))
        sch2 = json.loads(readfile(sys.argv[2]))
        schemadiff(sch1, sch2)
        

if __name__ == "__main__":
    main()
