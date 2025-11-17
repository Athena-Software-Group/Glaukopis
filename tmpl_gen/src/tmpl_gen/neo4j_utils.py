#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Oct  2 17:42:51 2025

@author: icardei


neo4j references:
    Parameters: https://neo4j.com/docs/cypher-manual/current/syntax/parameters/

    CREATE: https://neo4j.com/docs/cypher-manual/current/clauses/create/
    
    neo4j STORAGE: https://neo4j.com/developer/kb/understanding-data-on-disk/

"""

import math
import json
import time
import datetime
import dateutil
from neo4j import GraphDatabase, Record

from .utils import parse_attack, parse_datetime, readfile
from .utils import ctobjnav, parsecfg_attack, download_web_file
# import priorityQ


__all__ = ["create_ATTACK_db", "neo4j_extract_schema", "Neo4jDriver"]

# Read neo4j connecion configuration from JSON file.
neo4j_asg_config_filename = "neo4j-asg-config.json"

neo4j_TEST_config_filename = "neo4j-TEST-config.json"
neo4j_NEW_config_filename = "neo4j-NEW-config.json"


mitre_ent_attack_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
mitre_ent_attack_filename = "enterprise-attack.json"

# --- Cypher Queries to retrieve schema information ---
# This query gets all distinct node labels and their properties by inspecting one node of each type.
# It uses UNWIND to flatten the list of labels for each node.
NODE_PROPERTIES_QUERY = """
MATCH (n)
WITH DISTINCT labels(n) AS labels
UNWIND labels AS label
MATCH (m)
WHERE label IN labels(m)
RETURN label, collect(DISTINCT keys(m)) AS properties
"""

# This query gets all distinct relationship types and their properties.
RELATIONSHIP_PROPERTIES_QUERY = """
MATCH ()-[r]-()
RETURN DISTINCT type(r) AS type, collect(DISTINCT keys(r)) AS properties
"""

# This query gets the full graph structure (from node label, relationship type, to node label)
# for relationships with properties.
RELATIONSHIP_STRUCTURE_QUERY = """
MATCH (n)-[r]->(m)
RETURN DISTINCT labels(n) AS from_node, type(r) AS relationship, labels(m) AS to_node
"""


class Neo4jDriver:
    def __init__(self, neo4j_conf_filename:str):
        cfg_dct = json.loads(readfile(neo4j_conf_filename))
        uri, auth, db_name = cfg_dct["uri"], cfg_dct["auth"], cfg_dct["db_name"] 
        self.driver = GraphDatabase.driver(uri, auth=tuple(auth))
        self.driver.verify_connectivity()
        print(f"Neo4jDriver: Connection to Neo4j '{db_name}' database successful.")
        self._db_name = db_name
        self._db_nickname = cfg_dct.get("nickname", self._db_name)
        
    def db_name(self):
        return self._db_name
    
    def db_nickname(self):
        return self._db_nickname
    
    def session(self):
        return self.driver.session(database=self.db_name())
    
    # THIS DOESN'T  WORK: Gives error ResultConsumedError
    #    since session exits at return and the record stream would be consumed thereafter....
    def run_query_FAILS(self, query:str):
        with self.session() as session:
            node_results = session.run(query)
            return node_results
    
    def run_query_collect(self, query:str, qry_params:dict=None) -> list[Record]:
        with self.session() as session:
            node_results = session.run(query, qry_params)
            return list(node_results)
    
    def __del__(self):
        self.close()
    
    def close(self):
        try:
            self.driver.close()
            print(f"Neo4jDriver: Connection to Neo4j '{self.db_name()}' database closed.")
        except Exception:
            pass
      
    def get_db_schema(self) -> dict:
        schema = {
            "nodes": [],
            "relationships": [],
            "graph_structure": [],
            "adj_lst": {}
        }

        try:
            with self.session() as session:
                # Get node labels and properties
                node_results = session.run(NODE_PROPERTIES_QUERY)
                for record in node_results:
                    schema["nodes"].append({
                        "type": record["label"],
                        "properties": list(set([prop for props in record["properties"] for prop in props]))
                    })
    
                # Get relationship types and properties
                rel_results = session.run(RELATIONSHIP_PROPERTIES_QUERY)
                for record in rel_results:
                    schema["relationships"].append({
                        "type": record["type"],
                        "properties": list(set([prop for props in record["properties"] for prop in props]))
                    })
                
                # Get graph structure
                structure_results = session.run(RELATIONSHIP_STRUCTURE_QUERY)
                for record in structure_results:
                     schema["graph_structure"].append({
                        "from_node": record["from_node"],
                        "relationship": record["relationship"],
                        "to_node": record["to_node"]
                    })
                
                # create a nice adjacency list:
                ad = schema["adj_lst"]
                for rel in schema["graph_structure"]:
                    if len(rel["from_node"]) > 1:
                        raise ValueError("neo4j_get_db_schema ERROR: len(rel['from_node']) > 1")
                    if len(rel["to_node"]) > 1:
                        raise ValueError("neo4j_get_db_schema ERROR: len(rel['from_node']) > 1")
                    if rel["from_node"][0] not in ad:
                        ad[rel["from_node"][0]] = list()
                    ad[rel["from_node"][0]].append((rel["relationship"], rel["to_node"][0]))
                    if rel["to_node"][0] not in ad:
                        ad[rel["to_node"][0]] = list()
        except Exception as e:
            msg = f"Neo4jDriver.get_db_schema ERROR: could not connect to Neo4j or execute query:\n{e}\n"
            print(msg)
            raise e
        # finally:
        #     driver.close()

        return schema
        
    
class DebugNeo4jDriver:
    """
    Mock-up class used for debugging.
    """
    def __init__(self, schemagraph:dict):
        self.schemagraph = schemagraph
        
    def get_db_schema(self) -> dict:
        schema = dict()
        
        lst_nodes = sorted([ nt for nt in self.schemagraph["adj_lst"].keys()])        
        schema["nodes"] = [ {"type":nt, "properties":["id", "name"]} for nt in lst_nodes]
        lst_rels = sorted(rel for al in self.schemagraph["adj_lst"].values() for (rel, tonode) in al)
        schema["relationships"] = [ {"type":re, "properties":["id", "name"]} for re in lst_rels]
        schema["adj_lst"] = self.schemagraph["adj_lst"]
        return schema
        
    def db_name(self):
        return "debug"
    
    def db_nickname(self):
        return self.db_name()
    
    def session(self):
        return None
    
    
    def run_query(self, query:str) -> dict:
        """
        Returns bogus results for a Cypher query with  RERTURN clause.

        Parameters
        ----------
        query : str
            query str.

        Returns
        -------
        dct_bogus_results : dict
            bogus results for a Cypher query with  RETURN clause.

        """
        result_set_size = 3
        ret_k = "RETURN "
        str_return_expr = query[query.find(ret_k) + len(ret_k):]
        q_exprs = [s.strip() for s in str_return_expr.split(", ")]
        # dct_bogus_results = [{k:f"value{j}-{i}" for (i, k) in enumerate(q_exprs)} 
        #                      for j in range(result_set_size)]
        dct_bogus_results = [{k:f"{k}{j}-{i}" for (i, k) in enumerate(q_exprs)} 
                             for j in range(result_set_size)]
        return dct_bogus_results
    
    def run_query_collect(self, query:str):
        return self.run_query(query)
    
    def close(self):
        pass
    
    
def neo4j_get_db_schema(cfg_dct:dict):
    """
    Connects to the Neo4j database, executes queries to retrieve schema information,
    and returns it as a dictionary.
    """
    uri, auth, db_name = cfg_dct["uri"], cfg_dct["auth"], cfg_dct["db_name"] 
    schema = {
        "nodes": [],
        "relationships": [],
        "graph_structure": []
    }

    try:
        driver = GraphDatabase.driver(uri, auth=tuple(auth))
        driver.verify_connectivity()
        print("Connection to Neo4j database successful.")

        with driver.session(database=db_name) as session:
            # Get node labels and properties
            node_results = session.run(NODE_PROPERTIES_QUERY)
            for record in node_results:
                schema["nodes"].append({
                    "label": record["label"],
                    "properties": list(set([prop for props in record["properties"] for prop in props]))
                })

            # Get relationship types and properties
            rel_results = session.run(RELATIONSHIP_PROPERTIES_QUERY)
            for record in rel_results:
                schema["relationships"].append({
                    "type": record["type"],
                    "properties": list(set([prop for props in record["properties"] for prop in props]))
                })
            
            # Get graph structure
            structure_results = session.run(RELATIONSHIP_STRUCTURE_QUERY)
            for record in structure_results:
                 schema["graph_structure"].append({
                    "from_node": record["from_node"],
                    "relationship": record["relationship"],
                    "to_node": record["to_node"]
                })
            
            # create a nice adjacency list:
            ad = schema["adj_lst"] = dict()
            for rel in schema["graph_structure"]:
                if len(rel["from_node"]) > 1:
                    raise ValueError("neo4j_get_db_schema ERROR: len(rel['from_node']) > 1")
                if len(rel["to_node"]) > 1:
                    raise ValueError("neo4j_get_db_schema ERROR: len(rel['from_node']) > 1")
                if rel["from_node"][0] not in ad:
                    ad[rel["from_node"][0]] = list()
                ad[rel["from_node"][0]].append((rel["relationship"], rel["to_node"][0]))
                if rel["to_node"][0] not in ad:
                    ad[rel["to_node"][0]] = list()
    except Exception as e:
        msg = f"Error: Could not connect to Neo4j or execute query. {e}"
        print(msg)
        raise e
    finally:
        driver.close()

    return schema


def save_schema_to_json(schema_data, filename="neo4j_schema.json"):
    """
    Saves a dictionary to a JSON file.
    """
    if not schema_data:
        print("ERROR: No schema data to save.")
        raise ValueError("save_schema_to_json ERROR: No schema data to save.")

    try:
        with open(filename, 'w') as f:
            json.dump(schema_data, f, indent=4)
        print(f"save_schema_to_json: Schema successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
        raise e


def convert_prop_cypher(parsecfg:dict, propname:str, propval) -> str:
    """
    Converts a JSON dictionary value to a string suitable for CREATE Cypher query.
    It converts to datetime(prop) or a str(list) or JSON format as needed.
    
    Parameters
    ----------
    parsecfg : dict
        parsing configuration.
    propname : str
        property name
    propval : str | list | dict
        the JSON property value.

    Returns
    -------
    str
        string suitable for CREATE Cypher query.

    """
    datetime_props = parsecfg.get("datetime_properties", [])
    val = propval
    match propval:
        case str() if propname in datetime_props:
            # val = f"datetime('{propval}')"
            # val = datetime.datetime(propval)
            val = dateutil.parser.parse(propval)
        case list() if not all(isinstance(p, str) for p in propval):
            val = str(propval)
        case dict():
            val = json.dumps(propval, indent=4)
        case _:
            pass
            # raise ValueError(f"convert_prop_cypher ERROR: invalid propval type {type(propval)}")
    return val                
    

def neo4j_clean_db(session):
    """
    Deletes all nodes and relationships from a database.

    Parameters
    ----------
    session : neo4j Session
        neo4j session object.

    Returns
    -------
    None.

    """
    # this will leave constraints and indexes:
    # read more at https://neo4j.com/docs/cypher-manual/current/clauses/delete/
    query = "MATCH (n) DETACH DELETE n"
    session.run(query)


def neo4j_safe_identif(name:str) -> str:
    """
    Surround neo4j node/rel./property identifier with backquotes if needed.

    Parameters
    ----------
    name : str
        identifier.

    Returns
    -------
    str
        properly formatted neo4j identifier.

    """
    return name if name.isidentifier() else '`' + name + "`" 


def neo4j_create_db_from_dict(parsecfg:dict, graph:dict, neo4j_conf:dict):
    """
    Loads a graph from dict form to a neo4j DB described by a config param.

    Parameters
    ----------
    parsecfg : dict
        souce document parsing configuration        
    graph : dict
        graph data.
    neo4j_conf : dict
        neo4j connection configuration.

    Returns
    -------
    None.

    """
    uri, auth, db_name = neo4j_conf["uri"], neo4j_conf["auth"], neo4j_conf["db_name"] 

    try:
        driver = GraphDatabase.driver(uri, auth=tuple(auth))
        driver.verify_connectivity()
        print("Connection to Neo4j database successful.")

        with driver.session(database=db_name) as session:
            neo4j_clean_db(session)
            dct_nodenames = dict()
            lst_objinfo = []
            dct_qryprops = dict()      # used to pass Cypher query properties
            prop_counter = 0           # generate unique prop names
            # create nodes:
            for d in graph["nodes"].values():
                lst_prop_toks = list()
                for (k, v) in d.items():
                    try:
                        propval = convert_prop_cypher(parsecfg, k, v)
                    except Exception as ex:
                        print("neo4j_create_db_from_dict EXCEPTION, when adding nodes:", k, v)
                        raise(ex)
                        
                    pval_name = f"p{prop_counter}"
                    prop_counter += 1
                    prop_name = neo4j_safe_identif(k)
                    s = f"{prop_name}: ${pval_name}"
                    lst_prop_toks.append(s)
                    dct_qryprops[pval_name] = propval
                    
                    # if len(lst_prop_toks) > 2:
                    #     break
                    
                str_props = ", ".join(lst_prop_toks)
                node_type_raw = d["type_node"]
                node_type = neo4j_safe_identif(node_type_raw)
                ndname = f"n{len(dct_nodenames)}"
                dct_nodenames[d["id"]] = ndname
                str_node = f"( {ndname}:{node_type} {{ {str_props} }})"
                lst_objinfo.append(str_node)
                
                # if len(lst_objinfo) >= 20:
                #     break

            lst_relinfo = list()
            for rel in graph["edges"].values():
                lst_prop_toks = list()
                for (k, v) in rel.items():
                    try:
                        propval = convert_prop_cypher(parsecfg, k, v)
                    except Exception as ex:
                        print(f"neo4j_create_db_from_dict EXCEPTION, when adding edges: k={k}, v={v}")
                        raise(ex)
                        
                    pval_name = f"p{prop_counter}"
                    prop_counter += 1
                    prop_name = neo4j_safe_identif(k)
                    s = f"{prop_name}: ${pval_name}"
                    lst_prop_toks.append(s)
                    dct_qryprops[pval_name] = propval                
                
                str_props = ", ".join(lst_prop_toks)
                rel_type_raw = rel[parsecfg["type_rel"]]
                rel_type = neo4j_safe_identif(rel_type_raw)
                src = dct_nodenames[rel["srcid"]]
                tgt = dct_nodenames[rel["targetid"]]
                str_rel = f"({src}) -[:{rel_type} {{ {str_props} }}]-> ({tgt})"
                lst_relinfo.append(str_rel)
                
                # if len(lst_relinfo) >= 100:
                #     break

            qry_nodeinfo = ",\n".join(lst_objinfo)
            qry_relinfo = ",\n".join(lst_relinfo)
            str_query = "CREATE " + qry_nodeinfo + ", " + qry_relinfo
            
            # print("QUERY:\n", str_query)
            # print("\n\nPROPS:\n", dct_qryprops)
            nr_results = session.run(str_query, dct_qryprops)
          
            print("LEN(QUERY)", len(str_query))  

            # str_q_counts ="MATCH (n), ()-[r]->() RETURN count(n), count(r)"
            # str_q_counts ="MATCH (n) RETURN count(n) AS nodeCount"
            str_q_counts ="""
                OPTIONAL MATCH (n)
                WITH count(n) AS nodeCount
                OPTIONAL MATCH ()-[r]->()
                RETURN nodeCount, count(r) AS relationshipCount
                """  
            result = session.run(str_q_counts)
            records = result.data()
            if records:
                node_count = records[0]["nodeCount"]
                rel_count = records[0]["relationshipCount"]
            else:
                node_count = 0
                rel_count = 0            
            print(f" *** Created {node_count} nodes and {rel_count} relationships. ***")
    except Exception as e:
        msg = f"neo4j_create_db_from_dict Error: Could not connect to Neo4j or execute query. {e}"
        print(msg)
        raise e
    finally:
        driver.close()



class SchemaGraph:
    """
    Graph for neo4j DB schema with methods for shortest path using 
    Dijkstra's algorithm and for computing the adjacency list.
    """
    def __init__(self, adj_dct:dict):
        self.adjdct = adj_dct
        self.nmap = { nn:i for (i, nn) in enumerate(adj_dct.keys()) }
        self.imap = { i:nn for (i, nn) in enumerate(adj_dct.keys()) }
        
        self.adjlist = self.make_adjlist(self.adjdct)
        self.cache = dict()    
        
    def make_adjlist(self, adj_dct) -> list[list[int]]:
        """
        Computes and returns an adjacency list that uses numbers (0...n-1) instead of
        strings

        Parameters
        ----------
        adj_dct : dict
            schema adjacency dict.

        Returns
        -------
        list[list[str]]
            adjacency list .

        """
        n = len(adj_dct)
        al = [list() for _ in range(n)]
        for nn, lst in adj_dct.items():
            al[self.nmap[nn]] = [ (rel, self.nmap[en]) for (rel, en) in lst]
            
        return al
        
    
    def shortest_path(self, start:str, end:str) -> list[str]:
        """
        Returns the shortest path from start node to end node in the schema graph's adjacency dictionary.
        Uses Dijkstra's algorithm.
        
        Returns a path a-[:p]-b-[:q]-c in format: [a, p, b, q, c], all strings.
        Caches results.
        """
        key = (start, end)
        if key not in self.cache:                
            intpath = self.dijkstra(self.nmap[start], self.nmap[end])
            path = []
            if len(intpath) > 0:
                for i in range(len(intpath) - 1):
                    sn = self.imap[intpath[i]]
                    rel = next(rn for (rn, v) in self.adjlist[intpath[i]] if v == intpath[i + 1])
                    path.append(sn)
                    path.append(rel)
                path.append(self.imap[intpath[-1]])
            self.cache[key] = path
        return self.cache[key]

        
    def dijkstra(self, start:int, end:int) -> list[str]:
        """
        Returns the shortest path from start node to end node in the schema graph's adjacency dictionary.
        Uses Dijkstra's algorithm.'
        """        
        n = len(self.adjlist)
        costs = [ math.inf ] * n       # dsq[u] is distance from source to u
        costs[start] = 0
        pred = [-1] * n          # predecessor node
        vxadded = [False] * n    # True if already added to the heap
        vxadded[start] = True 
        
        # use a simple list for storing (cost, dstnode) tuples
        minQ = [(0, start)]
        found = False
        while not found and len(minQ) > 0 :
            (ucost, u) = min(minQ)
            minQ.remove((ucost, u))
            vxadded[u] = True
            # print(" popped ", (ucost, u))
            if u == end:
                found = True
            else:
                for v in [en for (rel, en) in self.adjlist[u]]:
                    if not vxadded[v] :
                        d = costs[u] + 1
                        
                        if d < costs[v]:
                            if (costs[v], v) in minQ: 
                                minQ.remove((costs[v], v))
                            costs[v] = d
                            pred[v] = u
                            minQ.append((costs[v], v))
                            # print("  pushed ", (costs[v], v))
        path = list()
        if found:
            u = end
            while u != start :
                path.append(u)
                u = pred[u]
            path.append(u)
            path = list(reversed(path))
        return path            
        
        
def test_schema():
    neo4j_config_dct = json.loads(readfile(neo4j_TEST_config_filename))
        # schema_dict = get_db_schema(URI, AUTH, DB_NAME)
    schema_dict = neo4j_get_db_schema(neo4j_config_dct)
    
    if schema_dict:
        sch_fn = f'schema_{neo4j_config_dct["nickname"]}.json'
        save_schema_to_json(schema_dict, sch_fn)
        # print("Saved neo4j DB schema to file '{sch_fn}'.")
        
        schemagraph = SchemaGraph(schema_dict["adj_lst"])
        start, end = "malware", "x-mitre-tactic"
        # start, end = "campaign", 
        path1 = schemagraph.shortest_path(start, end)
        print(path1)

def create_ATTACK_db(neo4j_cfg_filename:str, mitre_filename:str=mitre_ent_attack_filename):
    """
    Clears and then populates (loads) the ATT&CK DB from MITRE .json file
    
    The neo4j DB must have been created already.
    """    
    # 1. load connection configuration from json file:
    time0 = time.time()    
    neo4j_cfg = json.loads(readfile(neo4j_cfg_filename))

    # 2. download ATT&CK URL to local file:
    time01 = time.time()
    download_web_file(mitre_ent_attack_url, mitre_ent_attack_filename)

    # 3. load ATT&CK objects from json file into a JSON object:  
    time1 = time.time()
    
    dct_attack = parse_attack(mitre_ent_attack_filename, parsecfg_attack)
    
    # 3. generate Cypher queries to insert nodes and relationships into neo4j DB.
    #    run query
    time2 = time.time()

    neo4j_create_db_from_dict(parsecfg_attack, dct_attack, neo4j_cfg)
    # this (takes >40 minutes)
    time3 = time.time()
    
    print("time to load neo4j cofiguraion JSON file:", time01 - time0)
    print("time to download ATT&CK file (or it was cached):", time1 - time01)
    print("time to load and parse ATT&CK to graph dict:", time2 - time1)
    print("time to clean and populate new neo4j DB from graph dict:", time3 - time2)

    schema_dict = neo4j_get_db_schema(neo4j_cfg)
    
    sch_fn = f'schema_{neo4j_cfg.get("nickname", "mitre")}.json'
    save_schema_to_json(schema_dict, sch_fn)


def neo4j_extract_schema(db_conf_filename:str, schema_filename:str):
    """
    Extracts the schema of a neo4j DB and saves it to a file.
    Returns the schema dictionary.
    
    Parameters
    ----------
    db_conf_filename : str
        neo4j JSON DB configuration file.
    schema_filename : str
        schema output file name.

    Returns
    -------
    the schema dictionary.

    """
    neo4jdriver = Neo4jDriver(db_conf_filename)
    schema_dict = neo4jdriver.get_db_schema()   
    save_schema_to_json(schema_dict, schema_filename)
    return schema_dict

    
if __name__ == "__main__":
    pass
