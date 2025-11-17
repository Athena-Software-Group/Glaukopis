#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 09:14:43 2025

@author: icardei
"""
import os
import requests
from pathlib import Path
import sys
import json
from typing import Sequence
from datetime import datetime, timezone
import dateutil
import xmltodict    # read XML files, convert to JSON 

__all__ = ["GraphFormatterGraphviz"]

def parse_datetime(s:str) -> datetime:
    if not s:
        return None
    return dateutil.parser.parse(s)

def format_datetime(dt:datetime) -> str:
    dt_utc = dt.astimezone(timezone.utc)
    sdt = dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return sdt

def readfile(filename:str) -> str:
    with open(filename, "r") as f:
        return f.read()

def writefile(filename:str, s:str):
    with open(filename, "w") as f:
        f.write(s)

def read_xml(filename:str) -> dict:
    with open(filename,  "r") as f:
        xml = f.read()
        d = xmltodict.parse(xml)
        return d

def download_web_file(url:str, local_filename:str=None) -> bool:
    """
    Download a document from a url to a local file, if it does not exist. 
    (Gemini code)
    
    Args:
        url (str): The URL of the file to download if local file does not exist.

        local_filename (str, optional): The local file name.
        
    Returns:
        True if file had to be downloaded. False, otherwise.
    """
    
    if local_filename is None:
        # Extract filename from the URL if not provided
        local_filename = Path(url).name.split('?')[0]
    # print(f"Attempting to open {local_filename}")
    # print(f"Saving as: {local_filename}")
    ret = False
    try:
        if not os.path.exists(local_filename):
            print(f"Download file from: {url}")
            print(f"Saving as: {local_filename}")
            ret = True
            
            # Use a stream to handle large files without loading them entirely into memory
            with requests.get(url, stream=True) as r:
                # Raise an HTTPError for bad responses (4xx or 5xx)
                r.raise_for_status() 
    
                # Get the total file size from the headers for progress tracking
                total_size = int(r.headers.get('content-length', 0))
                
                # Open the local file in binary write mode
                with open(local_filename, 'wb') as f:
                    downloaded_size = 0
                    
                    # Iterate over the response content in chunks
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Simple progress update (optional)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\rProgress: {progress:.2f}%", end='')
                
                print(f"\nDownload successful. File saved to {os.path.abspath(local_filename)}")
    except requests.exceptions.RequestException as e:
        print(f"open_web_file ERROR: error occurred during download: {e}")
        raise(e)
    return ret
    

def seq_find(seq, pred, default=None) -> bool:
    """
    Finds the first object from a sequence that satisfies a Boolean predicate.
    Returns default if not found.

    Parameters
    ----------
    seq : TYPE
        sequence.
    pred : TYPE
        pred: type -> bool.

    Returns
    -------
    sequence element type
        first matching or None.

    """
    return next(filter(pred, seq), default)


def seq_in(seq, pred) -> bool:
    return next(filter(pred, seq), None) != None


def xmltojson(filename:str) -> str: 
    pass    

def prettyfy_json_file(filename:str):
    jsobj = json.loads(readfile(filename))
    newfilename = filename    # .jsonl ?
    writefile(newfilename, json.dumps(jsobj, indent=4))


def UNUSED_enumerate_keys(obj) -> list[str]:
    def helper(x, path:list, acc:list) -> list:
        skeys = sorted(x.keys()) if isinstance(x, dict) else range(len(x))
        lst = []
        revpath = path if path == "" else path + "/" 
        for key in skeys:
            path2 = revpath + str(key)
            lst.append(path2)
            if isinstance(x[key], (list, tuple, dict)):
                lst += helper(x[key], path2, acc)
        return lst
    return helper(obj, "", [])
                
        

def dctnav(d, path:str|Sequence[str|int]):
    """
    Navigate a dictionary based on a path.
    Special characters:
        ? stands for the first entry matching the next key

    E.g.
    for x-mitre-tactic object: "external_references/?/external_id"

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.
    path : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    def helper(x, lst_path:list, n:int, results:list):        
        if n == len(lst_path) - 1:
            if type(x) == dict:
                if lst_path[n] in x:
                    results.append(x[lst_path[n]])
            elif type(x) in [list, tuple]:
                index = int(lst_path[n])
                results.append(x[index])
            else:
               raise ValueError(f"dctnav.helper: invalid type for x: {type(x)}") 
            return
        
        if n == len(lst_path):
            results.append(x)
            return
        if type(x) == dict:            
            if lst_path[n] in x:
                helper(x[lst_path[n]], lst_path, n + 1, results)
        elif type(x) in [list, tuple]:
            if lst_path[n] == "?":
                rlen = len(results)
                # find the first match and return:
                for y in x:
                    helper(y, lst_path, n + 1, results)
                    if len(results) > rlen:
                        return
            elif lst_path[n] == "*":
                # accumulate all results:
                for y in x:
                    helper(y, lst_path, n + 1, results)
            else:
                index = int(lst_path[n])
                helper(x[index], lst_path, n + 1, results)
        else:
            raise ValueError(f"dctnav.helper: invalid type for x: {type(x)}")
            
    # --------------            
    if type(path) == str:
        ks = path.split("/")
    else:
        ks = path
        
    results = list()
    helper(d, ks, 0, results)
    if len(results) == 0:
        return None
    if len(results) == 1:
        return results[0]
    return results


def ctobjnav(parsecfg:dict, obj:dict, path:str|Sequence[str|int], default="#throw"):
    """
    Find  entry in JSON dict based on navigation path.

    Parameters
    ----------
    parsecfg : dict
        DESCRIPTION.
    obj : dict
        DESCRIPTION.
    path : str|Sequence[str|int]
        DESCRIPTION.

    Returns
    -------
    None.

    """
    val = None
    if type(path) == str:
        mpath = parsecfg.get(path, path)
        val = dctnav(obj, mpath)
    elif type(path) in [list, tuple]:
        # map only the first part of the path:
        mpath_lst = [parsecfg.get(path[0], path[0])] + path[1:]
        val = dctnav(obj, mpath_lst)

    if type(val) == list and len(val) == 0 \
            or type(val) != list and val == None:
        if default == "#throw":
            raise ValueError(f"dctnav ERROR: could not find path '{path}' for object {(str(obj)+'...')[:500]}")
        else:
            return default
    return val
  
      
def dctfind(d, key:str, value=None) -> str:
    def find_helper(obj, path:list) -> list:
        if type(obj) == dict:
            if key in obj and \
                (value != None and obj[key] == value or value == None):
                    return path + [key]
            for (k, v) in obj.items():
                ret = find_helper(v, path + [k])
                if ret != None:
                    return ret
            return None
        
        elif type(obj) in [list, tuple]:
            for i in range(len(obj)):
                ret = find_helper(obj[i], path + [i])
                if ret != None:
                    return ret
            return None
        else:
            if key == None and obj == value:
                return path + [value]
            # raise ValueError(f"find_helper ERROR: invalid type for type(obj)={type(obj)}")
            return None
        
    retval = find_helper(d, list())
    if retval == None:
        return retval
    return "/".join(str(x) for x in retval)


parsecfg_attack = {
    "skip_nodes": ["x-mitre-collection"],
    "path_doc_node": "objects/0",
    "path_doc_version": "objects/0/x_mitre_version",
    "path_doc_name": "objects/0/name",
    "path_all_objects": "objects",
    "path_nodes": "objects",
    "path_rels": "objects",
    "path_obj_list": "objects/0/x_mitre_contents",
    "path_obj_ids": "objects/0/x_mitre_contents/*/object_ref",    
    "path_source_name": "external_references/?/source_name",
    "path_ext_id": "external_references/?/external_id",
    "id": "id",
    "type_node": "type",
    "deprecated": "x_mitre_deprecated",
    "type_node": "type",
    "type_rel": "relationship_type",       
    "name_relationship": "relationship",
    "srcid": "source_ref",
    "targetid": "target_ref",
    "datetime_properties": ["created", "modified"],
    "custom_relparser": 
        [
            {    # describes a relationship from node0 --> node1, identified through paths:
                "node0_type": "attack-pattern", 
                "node1_type": "x-mitre-tactic",
                "node1_id_path": "x_mitre_shortname",
                "node0_link_path": "kill_chain_phases/*/phase_name",
                "type_rel": "achieves",
                "doc":
"""
parsing the technique -> tactic relationship requires a custom step.

E.g.
x-mitre-tactic has prop "x_mitre_shortname": "credential-access"

attack-pattern name="Adversary-in-the-Middle" has prop:
            "kill_chain_phases": [
                {
                    "kill_chain_name": "mitre-attack",
                    "phase_name": "credential-access"
                },
                {
                    "kill_chain_name": "mitre-attack",
                    "phase_name": "collection"
                }
            ],
    
"""        
                    
            }
        ]
    }


    
    
def parse_attack(fn_attack:str, parsecfg:dict) -> dict:
    """
    Parses the ATT&CK JSON file and returns a dict with nodes 
    (including properties) and relationships with their properties.

    Parameters
    ----------
    fn_attack : str
        json file name.
    parsecfg : dict
        parsing configuration.

    Returns
    -------
    dict
        A dictionary with nodes and relationships.

    """
    # separator for detailed relationship names. E.g. course-of-action>detects>attack-pattern
    rel_sep = ">"           # can't use _ or - or space
    
    with open(fn_attack, "r") as f:
        attack_obj = json.load(f)
        
    cti_all_objs = ctobjnav(parsecfg, attack_obj, "path_all_objects")
    print(f"Found {len(cti_all_objs)} objects.")
    # cti_all_objids = ctobjnav(parsecfg, attack_obj, "path_obj_ids")
    print(f"Found {len(cti_all_objs)} CTI objects (nodes and relationships).")
    lst_rels = list()

    doc_version = ctobjnav(parsecfg, attack_obj, "path_doc_version")
    doc_name = ctobjnav(parsecfg, attack_obj, "path_doc_name")
    # TODO: update doc_source
    doc_source = doc_name
    graph = {"nodes": dict(), "edges": dict(), 
             "nodetypes": dict(), "edgetypes": dict()}
        
    node_types = set()
    
    for obj in cti_all_objs:
        if ctobjnav(parsecfg, obj, "type") == parsecfg["name_relationship"]:
            lst_rels.append(obj)
        else:    
            otype = obj[parsecfg["type_node"]]
            if otype in parsecfg.get("skip_nodes", []):
                continue
            oid = obj[parsecfg["id"]]
            obj["id"] = oid
            obj["type_node"] = otype            
            obj["public_id"] = ctobjnav(parsecfg, obj, parsecfg["path_ext_id"],
                                        oid)
            # assume it's a node:
            graph["nodes"][oid] = obj

            if otype not in graph["nodetypes"]:
                graph["nodetypes"][otype] = set()
            graph["nodetypes"][otype].add(oid)
            node_types.add(otype)
    
    # for cti_objid in cti_all_objids:
    #     # explicitly referred in the document:
    #     if cti_objid in self.cgraph.nodes_all():
    #         self.cgraph.node(cti_objid).set_cti_node()
        
    # print("\nNode types found:")
    # for ndtype in sorted(node_types):
    #     print("   " + ndtype)        
    # print()
    
    # self.cgraph.set_meta_node(CNode(self._parsecfg, cti_all_objs[0]))
    
    for rel in lst_rels:
        rid = rel[parsecfg["id"]]
        absrtype = rel[parsecfg["type_rel"]]     # abstract type, e.g. "uses" or "detects"
        srcid = ctobjnav(parsecfg, rel, "srcid")
        tgtid = ctobjnav(parsecfg, rel, "targetid")
        rel["srcid"] = srcid
        rel["targetid"] = tgtid
        rel["public_id"] = rid    #ctobjnav(parsecfg, rel, parsecfg["path_source_name"])

        if True:
            rel["type_rel"] = absrtype
        else:
            srctype = graph["nodes"][srcid]["type_node"]
            tgttype = graph["nodes"][tgtid]["type_node"]
            rel["type_rel"] = f"{srctype}{rel_sep}{absrtype}{rel_sep}{tgttype}"
        graph["edges"][rid] = rel
        if rel["type_rel"] not in graph["edgetypes"]:
            graph["edgetypes"][absrtype] = set()
            # graph["edgetypes"][rel["type_rel"]] = set()            
        graph["edgetypes"][absrtype].add(rel["id"])
        # graph["edgetypes"][rel["type_rel"]].add(rel["id"])
        
    parse_custom_rels(parsecfg, graph)
        
    print("Finished parsing the graph.")
    print(f"Found {len(graph['nodes'])} objects, {len(lst_rels)} relationships.")
    return graph


def parse_custom_rels(parsecfg:dict, dct_graph:dict):
    """
    Parse custom relationships.

    Parameters
    ----------
    parsecfg : dict
        DESCRIPTION.
    dct_graph : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if "custom_relparser" not in parsecfg:
        print("parse_custom_rels: no 'custom_relparser' dict in parsecfg. Return.")
        return
    
    for dct_relcfg in parsecfg["custom_relparser"]:
        ndtypes = [dct_relcfg[f"node{i}_type"] for i in [0, 1]]

        dct_nodes = [{ d["id"]:d for d in [dct_graph["nodes"][nodeid] 
                                                   for nodeid in dct_graph["nodetypes"][nt]]} 
                     for nt in ndtypes]

        # dct_node1s = {ctobjnav(parsecfg, d, parsecfg["custom_relparser"]["node1_id_path"]):d 
        #               for d in dct_nodes[1]}
        dct_node1s = dict()
        for d in dct_nodes[1].values():
            n0key = ctobjnav(parsecfg, d, dct_relcfg["node1_id_path"])
            dct_node1s[n0key] = d
        
        rel_count = 0
        rel_type = dct_relcfg["type_rel"]
        for dct_node0 in dct_nodes[0].values():
            # get props used as IDs 
            node1_ids = ctobjnav(parsecfg, dct_node0, dct_relcfg["node0_link_path"])
            if not isinstance(node1_ids, list):
                node1_ids = [node1_ids]
            for n1id in node1_ids:
                dct_node1 = dct_node1s[n1id]
                
                rel = { "id": f"{parsecfg['name_relationship']}-{rel_type}-{rel_count}"}
                rel_count += 1
                rel["srcid"] = dct_node0["id"]
                rel["public_id"] = rel["id"]
                rel["targetid"] = dct_node1["id"]
                rel["type"] = parsecfg["name_relationship"]
                rel["type_rel"] = rel_type
                rel[parsecfg["type_rel"]] = rel_type
                # rel["type_rel"] = f"{srctype}{rel_sep}{absrtype}{rel_sep}{tgttype}"
                dct_graph["edges"][rel["id"]] = rel
                if rel["type_rel"] not in dct_graph["edgetypes"]:
                    dct_graph["edgetypes"][rel["type_rel"]] = set()
                dct_graph["edgetypes"][rel["type_rel"]].add(rel["id"]) 

        print(f"Added {rel_count} relationships of type '{rel_type}'")

            
class GraphFormatter:
    """
    Base class for objects that generate string representations of graphs, e.g. HTML, Graphviz.
    """
    def __init__(self, graph: dict, name:str="default", 
                 nodefilter_predicate=None, node_key_fun=lambda nd: nd):
        self.node_count = 0
        self.edge_count = 0
        self.name = name
        self.graph = graph   # a dict
        self.nodes = [nd["type"] for nd in self.graph["nodes"]]
        self.nodefilter_predicate = nodefilter_predicate
        self.nodes_sorted = sorted(self.nodes, key=node_key_fun)
        
        edge_key_fun = lambda e: (e["from_node"][0], e["relationship"], e["to_node"][0])
        self.edges = [edge_key_fun(edge) for edge in self.graph["graph_structure"]]
        self.edges_sorted = sorted(self.edges)
        
    def format_header(self) -> str:
        raise ValueError("GraphRepresFormatter format_header ERROR: unimplemented")

    def format_footer(self) -> str:
        raise ValueError("GraphRepresFormatter format_footer ERROR: unimplemented")

    def format_node(self, node:str) -> str:
        raise ValueError("GraphRepresFormatter format_node ERROR: unimplemented")

    def format_edge(self, edge:str) -> str:
        raise ValueError("GraphRepresFormatter format_edge ERROR: unimplemented")

    def format(self) -> str:
        """
        Formats a graph to a string representation (e.g. HTML or Graphviz).
        Template Method.
        """
        str_lst = []
        nodes_used = set()
        str_lst.append(self.format_header())
        for node in self.nodes_sorted:            
            if self.nodefilter_predicate == None or self.nodefilter_predicate(node):
                str_node = self.format_node(node)
                str_lst.append(str_node)
                nodes_used.add(node)
                self.node_count += 1
            
        for edge in self.edges_sorted:
            if edge[0] in nodes_used and edge[2] in nodes_used:
                str_edge = self.format_edge(edge)
                str_lst.append(str_edge)
                self.edge_count += 1

        str_lst.append(self.format_footer())
        str_graph = "\n".join(str_lst)
        return str_graph

    def format_save(self, out_filename:str):
        s = self.format()
        with open(out_filename, "w") as fout:
            fout.write(s)
        print(f"\nGraph saved to file {out_filename}.\n")
        

class GraphFormatterGraphviz(GraphFormatter):
    """
    GraphViz formatter.
    """
    def __init__(self, graph:dict, name:str="default", nodefilter_predicate=None):
        super().__init__(graph, name=name, nodefilter_predicate=nodefilter_predicate)
        self.nodes_used = None
        
    def format_header(self) -> str:
        str_lst = []
        self.nodes_used = set()
        graph_name = self.name.replace("-", "_")
        graph_name = graph_name.replace("&", "_")
        graph_name = graph_name.replace(" ", "_")
        
        str_lst.append(f"digraph {graph_name} {{")
        str_lst.append("//    rankdir LR;")     # commented
        str_lst.append("    node [shape=box, style=filled, fillcolor=lightblue];")
        # str_lst.append("\n    // Nodes:")
        s = "\n".join(str_lst)
        return s

    def format_footer(self) -> str:
        return "}\n"

    def format_node(self, node:str) -> str:
        return f'    "{node}" [color=navy, fontcolor=indigo];'
    
    def format_edge(self, edge:tuple[str, str, str]) -> str:
        return f'    "{edge[0]}" -> "{edge[2]}" [label="{edge[1]}"];'
        
    
# if __name__ == "__main__":
#     fn_ent_attack = "data/enterprise-attack.json"
#     dct_attack = parse_attack(fn_ent_attack, parsecfg_attack)


