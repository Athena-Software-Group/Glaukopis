#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 12:39:18 2025

@author: icardei


TODO:
    - add edge relationships to grammar and text generation
    - explain how new var name binds to second node in var:n1.rel.n2
    - implement code for list comprehensions, RE: template 2.8
    - add support for scope, e.g. cwe#Weakness
    - MAYBE add support for JSON objects
"""

from __future__ import annotations
import os
import os.path
import sys
import json
import re
import random
import time
import datetime
from datetime import timezone
from dataclasses import dataclass
from collections.abc import Iterable

import lark
import lark.lexer
from lark import Lark, Tree
import neo4j
from neo4j import GraphDatabase
# from neo4j import time 

from .utils import parse_datetime, format_datetime, readfile, seq_in, seq_find
from .neo4j_utils import save_schema_to_json, SchemaGraph, neo4j_get_db_schema
from .neo4j_utils import Neo4jDriver, DebugNeo4jDriver, neo4j_safe_identif

# Optional dependency: BeautifulSoup is used by clean_cti_description() to
# strip HTML/XML markup out of freeform CTI text fields (description, notes,
# detection, etc.) before they are substituted into a template. If bs4 is
# missing we degrade to a regex-only HTML-tag strip so the build still runs.
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    BeautifulSoup = None
    _HAS_BS4 = False

# Freeform multi-line node properties whose substituted values are passed
# through clean_cti_description() in format_prop_val(). Short-string
# properties (name, mitre_id, cvss_*, etc.) are intentionally NOT in this
# set: they don't contain HTML and any whitespace collapse is wasted work.
# Keep this list in sync with the v10 manifest's <desc>...</desc> wrapping
# convention (Sophia-CTI-Templates-v10.txt v8.1 -> v10 delta (5)).
FREEFORM_PROPS = {
    "description", "descriptions", "extended_description", "notes", "detection",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n[ \t\n]*")


def clean_cti_description(raw_text: str) -> str:
    """
    Sanitise a freeform CTI text field prior to template substitution.
    Strips HTML/XML tags and collapses runs of blank lines so the trained
    model sees compact prose rather than markup-laden multi-page blobs.

    Idempotent and safe on already-clean text.
    """
    if not raw_text:
        return raw_text
    if _HAS_BS4:
        text = BeautifulSoup(raw_text, "html.parser").get_text(separator="\n")
    else:
        text = _HTML_TAG_RE.sub("", raw_text)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


# public identifiers:
__all__ = ["TmplParser", "tool_tmplgen", "clean_cti_description", "FREEFORM_PROPS"]

## These were used for testing. 
# Read neo4j connecion configuration from JSON file.
neo4j_asg_config_filename = "neo4j-asg-config.json"
neo4j_TEST_config_filename = "neo4j-TEST-config.json"
fn_ent_attack = "data/enterprise-attack.json"

# Template language grammar, EBNF:

def make_templ_grammar1():
    grammar = r"""
template:   topfield+

topfield:    qfield | TEXTSECTION | invsection | listsection | constraint

qfield:      "{" vardef? scope? cnameseq subscript? "}" 

cnameseq:    CNAME ("." CNAME)*
vardef:      VNAME ":"
scope:       CNAME "#"

constraint:  "{force" vardef? scope? cnameseq RELOP vardef? scope? cnameseq "}"

subscript:   "[" subscr_exp "]"

subscr_exp:    POSINTEGER | QUANTIFIER | (QUANTIFIER RELOP svalue)

invsection:  "<*" (INVCONTENT | qfield | constraint)* "*>"

listsection: "[" (TEXTSECTION | qfield)* "]"

svalue:       QSTRING | BOOL | NUMBER | OTHERVALUE

// added a space around " < " and " > " to avoid confusion  with <> used for 
//     edges:
RELOP:        "=" | "!=" | " < " | "<=" | ">=" | " > " | "~"
 
INVCONTENT:   /(?:[^*{]|\*(?!>))+/
QFTEXT:       /(?:[^}])+/
TEXTSECTION:  /(?:[^{<[\]]|<(?!\*))+/
                                          
DIGIT:        "0".."9"                 
POSINTEGER:   "0" | ("1".."9" DIGIT*)
    
CNAMELETTER:  "a".."z" | "A".."Z" | "_"
CNAMECHAR:    CNAMELETTER | "-"

// add support for inverse relationship notation using a "<" at their end:
// CNAME:        CNAMELETTER (CNAMECHAR | DIGIT)* ("<")?
// '<' is used for inverse relationships using original name, tool.revoked-by<malware...
// '>' is used to disambiguate multiple relationships with the same e.g. campaign.uses>malware vs. campaign.uses>tool
CNAME:        CNAMELETTER (CNAMECHAR | DIGIT | ("<")| (">"))*

// variable names should not use '-'
VNAME:        CNAMELETTER (CNAMELETTER | DIGIT)*

QUANTIFIER:   QUANTIF_ANY | QUANTIF_ALL | EXACTLY
QUANTIF_ANY:  "?"
QUANTIF_ALL:  "*"    
EXACTLY:      "_"

SUBSCR_OP:    "!=" | "="
//SVALUE:       QSTRING | BOOL | NUMBER | OTHERVALUE
BOOL:         "true" | "false"
NUMBER:       /[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?/
OTHERVALUE:   /[A-Za-z0-9:-]+/

QSTRING:      "\"" QSTRING_DQ "\"" | "'" QSTRING_SQ "'"
QSTRING_DQ:   /(?:[^"\\]|\\.)*/
QSTRING_SQ:   /(?:[^'\\]|\\.)*/

%import common.WS
%ignore WS                       
"""
    return grammar


def make_templ_grammar2():
    """
        Old version. NOT USED.
        Template grammar with vardef(varname, CNAME).
    """
    grammar = r"""
template:   topfield+

topfield:    qfield | TEXTSECTION | invsection | listsection

qfield:      "{" vardef? scope? cnameseq subscript? "}" 

cnameseq:    CNAME ("." CNAME)*

vardef:      VNAME ":" CNAME "."
scope:       CNAME "#" CNAME "."

subscript:   "[" subscr_exp "]"

subscr_exp:    QUANTIFIER | (QUANTIFIER SUBSCR_OP QSTRING)

invsection:  "<*" (INVCONTENT | qfield)* "*>"

listsection: "[" (TEXTSECTION | qfield)* "]"

INVCONTENT:   /(?:[^*{]|\*(?!>))+/
QFTEXT:       /(?:[^}])+/
TEXTSECTION:  /(?:[^{<[\]]|<(?!\*))+/
                     
                     
DIGIT:        "0".."9"                     
CNAMELETTER:  "a".."z" | "A".."Z" | "_"
CNAMECHAR:    CNAMELETTER | "-"
CNAME:        CNAMELETTER (CNAMECHAR|DIGIT)*

// variable names should not use '-'
VNAME:        CNAMELETTER (CNAMELETTER | DIGIT)*

QUANTIFIER:   QUANTIF_ANY | QUANTIF_ALL 
QUANTIF_ANY:  "?"
QUANTIF_ALL:  "*"    
SUBSCR_OP:    "!=" | "="
QSTRING:      "\"" QSTRING_DQ "\"" | "'" QSTRING_SQ "'"
QSTRING_DQ:   /(?:[^"\\]|\\.)*/
QSTRING_SQ:   /(?:[^'\\]|\\.)*/
//QSTRING_DQ:   /"(?:[^"\\]|\\.)*"/
//QSTRING_SQ:   /'(?:[^'\\]|\\.)*'/

//TEXTSECTION:  /(?:[^{<]|<(?!\*))+/
"""
    return grammar


    
def tool_tmplgen(options:dict):
    """
    Called from other scripts.
    Runs a generation session from template JSON file.
    """
    tmplgen = TmplGenNeo4j(options)

    lst_tmplobjs = tmplgen.load_templates(options["templates_file"])
    (count_gen, count_fail) = tmplgen.generate(lst_tmplobjs, do_print=False)
    
    print(f"Generated: {count_gen}  failed: {count_fail}")
    print(f"Results saved in directory {options['results_dir']}")



def tree2str(tree:Tree, indent=2, previndent=0):
    crtindent = previndent + indent
    indent_str = " " * crtindent
    if type(tree) == Tree:
        data_str = indent_str + repr(tree.data)
        # print("--", data_str)
        child_lst = [tree2str(child, indent, crtindent) for child in tree.children]
        ret_str = "\n".join([data_str] + child_lst)
    else:
        ret_str = indent_str + repr(tree)   # Token
    return ret_str


class TmplParseTransfDummy(lark.Transformer):
    def _concat(seq):
        return "".join(seq)
    
    def template(self, args):
        print("template: ", args)
        return "".join(args)
    
    def topfield(self, args):
        print("topfield: ", args)
        return args[0]
    
    def qfield(self, args):
        print("qfield: ", args)
        return TmplParseTransfDummy._concat(args)
    
    def cnameseq(self, args):
        print("cnameseq: ", args)
        return TmplParseTransfDummy._concat(args)
    
    def TEXTSECTION(self, arg):
        print("TEXTSECTION: ", arg)
        return arg
    
    def CNAME(self, arg):
        print("CNAME: ", arg)
        return arg
    
# ========   ========   ========   ========   ========   ========   
# Template parsing objects:

@dataclass
class ParseElement:
    pass    

@dataclass
class TextSection(ParseElement):
    text: str

@dataclass 
class Nodespec(ParseElement):  
    varname: str    
    ntype: str

@dataclass
class Vardef(ParseElement):
    varname: str

@dataclass
class Scope(ParseElement):
    name: str

@dataclass
class Relspec(ParseElement):
    relname: str

@dataclass
class InvRelspec(ParseElement):
    """ Inverse relationship specification"""
    relname: str

@dataclass
class Propspec(ParseElement):
    propname: str
    subscript: Subscript

@dataclass
class Subscript(ParseElement):
    """
    Used to describe filters.
    In the following, Value is a quoted string ('abc'), bool (true), datetime
        (2025-04-25T14:45:30Z), number (12.3).
        
    Notation for list property subscript:
        a.p[0] select a.p value with index 0
        a.p[?] select one of a.p values at random
        a.p[*] select all of a.p values at random, equivalent to just a.p
        a.p[?=Value] or a.p[?!=Value]  select only entries for which min. one of
            a.p[i] == Value or a.p[i] != Value, respectively
        a.p[*='Value'] or a.p[?=Value] select only entries for which all of
            a.p[i] == Value or a.p[i] != Value, respectively
        
        
    Notation for a non-list property subscript:
        a.p[_=Value] or a.p[?!=Value]  select only the entry for which 
            a.p == Value or a.p != Value, respectively
    """
    quantif: str
    op: str 
    val: object

@dataclass
class ConstrRelop(ParseElement):
    """
    Placed between adjacent Pathspecs to enforce a constraint 
    e.g. {force coa3:t1.mitigates<.id=coa4:t2.mitigates<.id}
    """
    relop: str

@dataclass
class InvsecStart(ParseElement):
    """
    Marker for start of invisible section. Nesting is supported.
    """
    pass
    
@dataclass
class InvsecStop(ParseElement):
    """
    Marker for end of invisible section. Nesting is supported.
    """
    pass
    

@dataclass
class Pathspec(ParseElement):
    """
    Represents an expression: varname.rel.rel.propname
    """
    lst: list[Nodespec | Relspec | InvRelspec | Propspec]
    

@dataclass
class UnboundQfield(ParseElement):
    """
    Used for storing qfield args (Tokens) until the top CNAME is resolvable by a bound variable.
    """
    token_lst: list[lark.Token]
    
# ------------------------------------------------------------    
    
class TmplParseTransf(lark.Transformer):
    """
    Parser Transform class that is visited during parsing uring parse tree
    subtrees as parameters.
    """
    def __init__(self, conf:dict, sch_dct:dict):
        """
        Constructor.

        Parameters
        ----------
        conf : dict
            parse configuration.
        sch_dct : dict
            schema graph

        Returns
        -------
        None.

        """
        self.conf = conf
        self.varname_cnt= dict()            # counter per node type
        self.nodespecs = dict()             # save node specs
        # self.explicit_varnames = set()    # variables defined in the template qfield
        self.sch_dct = sch_dct              # schema graph
        self.sch_adj_dct = sch_dct['adj_lst']    # adjacency list as a dict
    
    def varname_next(self, ntype:str) -> str:
        """
        Returns new variable name

        Parameters
        ----------
        ntype : str
            node type.

        Returns
        -------
        str
            new variable name.

        """
        
        if ntype not in self.varname_cnt:
            self.varname_cnt[ntype] = 1
        san_nt = self.sanitize_str(ntype)
        
        nm = f"_{san_nt}{self.varname_cnt[ntype]}"
        self.varname_cnt[ntype] += 1
        return nm
    
    def varname_default(self, ntype:str) -> str:
        """
        Returns default variable name.

        Parameters
        ----------
        ntype : str
            node type.

        Returns
        -------
        str
            default variable name.

        """
        san_nt = self.sanitize_str(ntype)
        nm = f"_{san_nt}0"
        return nm

    def sanitize_str(self, s:str) -> str:
        """
        Replace '-' with "_"; used for property names.
        """
        return s.replace("-", "_")
    
    def add_nodespec(self, ns:Nodespec):
        self.nodespecs[ns.varname] = ns

    def get_nodespec(self, varname:str) -> Nodespec | None:
        return self.nodespecs.get(varname, None) 
    
    def map_nodetype(self, nodetype:str) -> str:
        md = self.conf.get("nodetype_mappings", None)
        if md == None:
            return nodetype
        return md.get(nodetype, nodetype)
    
    def is_node_type(self, ntype:str) -> bool:  
        nt_real = self.map_nodetype(ntype)
        # the following does not work for nodes with no originating relationships:
        # return nt_real in self.sch_dct['adj_lst'] 
        # return nt_real in (nd["type"] for nd in self.sch_dct['nodes'])
        return seq_in(self.sch_dct['nodes'], lambda d: nt_real == d["type"])
    
    def is_property(self, ntype:str, propname:str) -> bool:
        return seq_in(self.sch_dct['nodes'], lambda d: ntype == d["type"] and propname in d["properties"])
    
    def map_property(self, nodetype:str, propname:str) -> str:
        """
        Map a property name using the gen configuration.
        It returns propname if not found in the config map dictionary.
        
        Example: self.conf["property_mappings"] looks like this:
            [{'id': ['public_id', ['course-of-action', 'malware', 'tool', 'x-mitre-tactic', 'attack-pattern', 
                            'x-mitre-data-component', 'intrusion-set', 
                            'campaign', 'x-mitre-data-source']]}]
            This maps 'id' to 'public_id' for the ATT&CK nodes.

        Parameters
        ----------
        nodetype : str
            node type.
        propname : str
            property to map.

        Returns
        -------
        str
            mapped property name (if found) or propname parameter

        """
        pm = self.conf.get("property_mappings", None)
        if pm == None:
            return propname
        dct = seq_find(pm, lambda d: propname in d, None)
        if dct == None:
            return propname     # property name not mapped for any node type        
        return dct[propname][0] if nodetype in dct[propname][1] else propname
    
    def is_nodename_resolvable(self, nname:str):
        """
        Returns True if nname is a node type or a bound variable name.
        """
        # return nname in self.explicit_varnames or nname in self.explicit_varnames
        mappedname = self.map_nodetype(nname)
        # return self.is_node_type(mappedname) or \
        #     seq_in(self.nodespecs.values(), lambda ns: ns.ntype == mappedname)
        return self.is_node_type(mappedname) or nname in self.nodespecs.keys()
     
                    
    def find_edge_end(self, nodetype:str, relname:str, throw=False) -> str:
        """
        Returns type of the node at the end of this relationship edge.
        Returns None if wrong start node type or wrong relationship name.
        """
        res = seq_find(self.sch_adj_dct.get(nodetype, []), lambda rn: rn[0] == relname, 
                        (None, None))[1] 
        
        if not res and throw:
            err = f"TmplParseTransf.find_edge_end ERROR: invalid relationship name {relname}"
            print(err)
            raise ValueError(err)
        return res
    
    def find_edge_start(self, nodetype:str, relname:str) -> str:
        """
        Returns types of the nodes at the start of inverse relationships with given name.
        If the list has length > 1 then relname is an ambiguous relationship and
        the caller must raise an exception. 
        E.g. attack-pattern has 4 "uses" incomign relationships.
        
        Returns [] if wrong end node type or wrong relationship name.
        """
        d = self.sch_adj_dct
        # return next((start for start, rel_ends in d.items() if (relname, nodetype) in rel_ends), None)
        lst = [start for (start, rel_ends) in d.items() if (relname, nodetype) in rel_ends]
        return lst
                    
    def parse_relstr(self, relstr:str) -> tuple[str, str, str]:
        """
        Parses rel<type into ("rel", "<", "type"). Last two can be "".
        """
        got_sep = False
        for i in range(len(relstr)):
            # print(f"i={i}, c={relstr[i]}")
            if relstr[i] in "<>":
                got_sep = True
                break         
        if got_sep:
            rel = relstr[:i]
            sep = relstr[i:i+1]
            othertype = relstr[i+1:]
        else:
            rel = relstr
            sep = ""
            othertype = ""
        return (rel, sep, othertype)


    def map_rel_TODELETE(self, ntype:str, relstr:str) -> tuple[bool, str, str]:
        is_inverse = False
        inv_dct = self.conf.get("inverse_relationships", dict())
        (relname, sep, othertype) = self.parse_relstr(relstr)
        
        # sep is "" if unspecified; othertype is "" if implicit
        
        # map type aliases:
        ntype = self.conf.get("nodetype_mappings", {}).get(ntype, ntype)
        othertype = self.conf.get("nodetype_mappings", {}).get(othertype, othertype)

        # is there a direct rel. 
        print(ntype, relname, sep, othertype)
        
        # 4 cases:
        sys.exit(1)        

    
    def map_rel(self, ntype:str, relstr:str) -> tuple[bool, str, str]:
        """
        Compute a direct Relspec or an InvRelspec depending on relationship name 
        and previous node type.
        
        It is a direct relationship if relname has no < and > and it is not in the 
        config. dictionary for inverse relationships.
        Otherwise, the format is:
            rel_dir<start_type, e.g. malware.uses<campaign.name
            rel_inv>start_type, e.g. malware.used-by<campaign.name
        
        The start type is optional in general, but REQUIRED when the start type 
        cannot be disambiguated, such in the situation when the end node type
        (malware) has two incoming "uses" relationships:
            campaign - [:uses] -> malware 
            intrusion-set - [:uses] -> malware 
        Without the start type disambiguation, the generation code will raise an exception.
        
        
        Parameters
        ----------
        relname : str
            relationshipp CNAME
        prev_type : str
            the node type parsed already

        Returns
        -------
        tuple with true/false, canonical relationship name, and the next type.

        """

        """
        OLD COMMENT:
        In case this is an inverse relationship, e.g. ntype==malware:
            e.g. malware.used-by or malware.used-by> or malware.used-by>campaign, OR
            e.g. malware.uses< or malware.uses<campaign
        it maps it to the canonical inverse relationship name: 
            (true, canonic_rel, start_node_type), e.g. (true, "uses", "campaign")
            
        Otherwise, it is a direct relationship and it returns:
            (false, , canonic_rel, end_node_type), e.g. (false, "uses", "attack-pattern")

        Parameters
        ----------
        ntype : str
            the node type of the current node
            
        relname : str
            relationship name. Could be direct or inverse rel. alias.

        Returns
        -------
        tuple[bool, str, str]
            (Relspec(canonical_rel), end_type) if this is a direct relationship, or 
            (InvRelspec(canonic_rel), start_type) if inverse rel.

        """
        is_inverse = False
        inv_dct = self.conf.get("inverse_relationships", dict())
        (relname, sep, othertype) = self.parse_relstr(relstr)
        
        # map type aliases:
        ntype = self.conf.get("nodetype_mappings", {}).get(ntype, ntype)
        othertype = self.conf.get("nodetype_mappings", {}).get(othertype, othertype)
        
        dir_rel = relname
        
        if relname in inv_dct:
            # relname is an inverse relationship:
            is_inverse = not is_inverse
            dir_rel = inv_dct[relname]
            
        # if sep == ""  or sep == ">":            
        #     return (is_inverse, dir_rel, othertype)

        if sep == "<":
            is_inverse = not is_inverse

        try:            
            # type checking:
            if is_inverse:
                # get list of node types for which (dir_rel, ntype) is in its adj. list:
                nt_lst = [nt for (nt, rt_lst) in self.sch_adj_dct.items() if (dir_rel, ntype) in rt_lst]
                if othertype == "":
                    if len(nt_lst) == 0:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: inverse case: no relationship found: [:{dir_rel}]->{ntype}")
                    elif len(nt_lst) > 1:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: inverse case: [:{dir_rel}]->{ntype} defined for types \
                                         {', '.join(nt_lst)} ; ambiguous")
                                         
                    othertype = nt_lst[0]
                else:
                    if othertype not in self.sch_adj_dct:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: inverse case: node type {othertype} is undefined")
                        
                    if (dir_rel, ntype) not in self.sch_adj_dct[othertype]:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: inverse case: {othertype}-[:{dir_rel}]->{ntype} is undefined")                        
                        
            else:
                # a direct relationship:
                if ntype not in self.sch_adj_dct:
                    raise ValueError(f"TmplParseTransf.map_rel ERROR: node type {ntype} is undefined")

                if othertype == "":
                    et_lst = [et for (r, et) in self.sch_adj_dct[ntype] if dir_rel == r]
                    if len(et_lst) == 0:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: no property or relationship found: {ntype}-[:{dir_rel}]->*")
                    elif len(et_lst) > 1:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: {ntype}-[:{dir_rel}]->* defined for types \
                                         {','.join(et_lst)}; ambiguous")
                                         
                    othertype = et_lst[0]                
                else:
                    if othertype not in self.sch_adj_dct:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: node type {othertype} is undefined")
                        
                    if (dir_rel, othertype) not in self.sch_adj_dct[ntype]:
                        raise ValueError(f"TmplParseTransf.map_rel ERROR: {ntype}-[:{dir_rel}]->{othertype} is undefined")                        
                                         
        except Exception as exc:
            err = f"TmplParseTransf.map_rel ERROR: got exception: {exc}  for type {ntype}, relstr: {relstr}"
            # print(err)
            raise ValueError(err)

        return (is_inverse, dir_rel, othertype)
                                    
        
    def make_relspec(self, prev_type:str, relstr:str) -> tuple[Relspec | InvRelspec, str]:
        """
        Compute a direct Relspec or an InvRelspec depending on relationship name 
        and previous node type.
        
        It is a direct relationship if relname has no < and > and it is not in the 
        config. dictionary for inverse relationships.
        Otherwise, the format is:
            rel_dir<start_type, e.g. malware.uses<campaign.name
            rel_inv>start_type, e.g. malware.used-by<campaign.name
        
        The start type is optional in general, but REQUIRED when the start type 
        cannot be disambiguated, such in the situation when the end node type
        (malware) has two incoming "uses" relationships:
            campaign - [:uses] -> malware 
            intrusion-set - [:uses] -> malware 
        Without the start type disambiguation, the generation code will raise an exception.
        
        
        Parameters
        ----------
        relname : str
            relationshipp CNAME
        prev_type : str
            the node type parsed already

        Returns
        -------
        tuple with Relspec or an InvRelspec with canonical relationship name, and the next type.

        """
        (is_inverse, can_rel, next_type) = self.map_rel(prev_type, relstr)
        sp = InvRelspec(can_rel) if is_inverse else Relspec(can_rel)
        return (sp, next_type)
            
    def make_varname(self, cnamelst:list, crtidx:int, ntype:str, vardef:Vardef) -> str:
        """
        Selects a var. name for a new Nodespec depending on the cname list and whether the current 
        cname is followed by a property name.
        
        If the current cname at crtidx is followed by a property or if there is no explicit property given,
        the selected varname is one from vardef.
        """
        is_last_rel = (crtidx == len(cnamelst) - 1) or self.is_property(ntype, self.map_property(ntype, cnamelst[crtidx + 1]))
        
        if is_last_rel:        
            if vardef != None:
                return vardef.varname
            else:
                vn = self.varname_default(ntype) if crtidx == 0 else self.varname_next(ntype)
                return vn
        
        return self.varname_next(ntype)
    
    
    # ===========  GRAMMAR nonterminal rule methods:  ================
    
    def template(self, args):
        # print("template: ", args)
        # flatten list argsL
        flat_args = [s for a in args for s in (a if type(a)==list else [a])]              
        dct_unbound = dict()
        lst_result = list()
        for i in range(len(flat_args)):
            spec = flat_args[i]
            match spec:
                case UnboundQfield():
                    # dct_unbound[varname] = spec
                    dct_unbound[i] = spec
                    lst_result.append(None)
                case Pathspec():
                    # dct_vars[varname] = ns
                    lst_result.append(spec)
                case TextSection():
                    lst_result.append(spec)
                case InvsecStart() | InvsecStop():
                    lst_result.append(spec)
                case ConstrRelop():
                    lst_result.append(spec)
                case x:
                    err = f"TmplParseTransf.template ERROR: unexpected list element '{x}'"
                    # print("\n", err)
                    raise ValueError(err)
        
        max_retries = 5
        j = 0
        while len(dct_unbound) > 0:
            still_unbound_vars = set()
            len0 = len(dct_unbound)
            for (index, unb) in list(dct_unbound.items()):
                ps = self.qfield_process(unb.token_lst)
                match ps:
                    case UnboundQfield([Vardef(), [varname, *_]]):
                        still_unbound_vars.add(varname)
                    case UnboundQfield([[varname, *_]]):
                        still_unbound_vars.add(varname)
                    case Pathspec([Nodespec(varname), *_]):
                        lst_result[index] = ps
                        dct_unbound.pop(index)                        
            len1 = len(dct_unbound)

            if len0 == len1:
                err = "TmplParseTransf.template ERROR: unbound variables (possible: from undefined node type): " + ", ".join(sorted(still_unbound_vars))
                # print("\n", err)
                raise ValueError(err)
                    
        if self.conf.get("verbose", 0) > 0:
            for pr in lst_result:
                print(pr, "\n---------")
        return lst_result
    
    def topfield(self, args):  # args is a list [Pathlist(....)]
        # print("topfield: ", args)
        return args[0]
    
    def qfield(self, args):
        if self.conf.get("verbose", 0) > 0:
            print("qfield: ", args)
        # args is [Vardef, [CNAME, ...]] or [[CNAME, ...]], optionally with a Subscript at the end
                
        index = 1 if isinstance(args[0], Vardef) else 0
        if self.is_nodename_resolvable(args[index][0]):
            return self.qfield_process(args)
        else:
            # spec0 is an unbound variable name. Save args and solve it later:
            uq = UnboundQfield(args)   
            return uq


    def qfield_process(self, args:list):
        """
        The first CNAME token is now resolvable, i.e. a node type or a bound variable.
        
        Processes a list of [Vardef, [Token, ...]] or [[Token, ...]].
        Returns a Pathspec if the first element is a Nodespec or a node type.
        Otherwise it returns an UnboundQfield with its args list since the first 
        token is an unbounded variable name.
        In that case, self.template() must come back later and try it until var is bound.
        """
        if self.conf.get("verbose", 0) > 0:
            print("qfield_process: ", args)
        # is_CNAME = lambda tok: type(tok) == lark.Token and tok.type == "CNAME"
                
        clst = list()      # use this for the Pathspec we return
        # spec0 = args[0]
        lst = args[0]
        # tidx = 0       # token index
        vardef = None   
        scope = None
        qf_index = 0
        
        if isinstance(args[qf_index], Vardef):
            vardef = args[qf_index] 
            if self.get_nodespec(vardef.varname) != None:
                err = f"TmplParseTransf.qfield_process ERROR: cannot redefine variable {vardef.varname}"
                # print(err)
                raise ValueError(err)
            qf_index += 1
            lst = args[qf_index]               

        if isinstance(args[qf_index], Scope):
            # TODO: check scope name
            scope = args[qf_index] 
            qf_index += 1
            lst = args[qf_index]               

        # TODO: add support for scope
        
        prev_cname = None
        prev_ns = None
        ns = None
        propname = None
        
        crtidx = 0
        while crtidx < len(lst):
            cname = lst[crtidx].value
            if prev_cname == None:
                # first iteration
                prev_cname = self.map_nodetype(cname)   # map if necessary (alias for node type)
                ns = self.get_nodespec(prev_cname)
                if ns == None:
                    # not bound. Is it a node type?
                    if self.is_node_type(prev_cname):        # delay Nodespec init
                        varname = self.make_varname(lst, crtidx, prev_cname, vardef)
                        ns = Nodespec(varname, ntype=prev_cname)
                        self.add_nodespec(ns)
                        clst.append(ns)
                        crtidx += 1
                        continue
                    else:
                        # not bound. Nothing to do yet. Come back later.
                        return UnboundQfield(args)
                clst.append(ns)
            else:
                propname = self.map_property(ns.ntype, cname)
                if self.is_property(ns.ntype, propname):
                    if crtidx != len(lst) - 1:
                        err = f"TmplParseTransf.qfield_process ERROR: property {cname} must be the last CNAME in a qfield"
                        print(err)
                        raise ValueError(err)                    
                else:
                    # cname is a relationship:
                    propname = None
                    relstr = cname                
                    (rs, next_type) = self.make_relspec(ns.ntype, relstr)            
                    clst.append(rs)
    
                    # new variable names are also used for intermediary/end nodes:
                    varname = self.make_varname(lst, crtidx, next_type, vardef)        
                    ns = Nodespec(varname, next_type)
                    self.add_nodespec(ns)
                    clst.append(ns)
            
            crtidx += 1

        if propname == None:
            propname = self.conf["default_propname"]

        subscript = args[-1] if type(args[-1]) == Subscript else None

        ps = Propspec(propname, subscript)
        clst.append(ps)

        pathspec = Pathspec(clst)
        return pathspec


    def constraint(self, args):
        """
        We model a constraint such as 
            {force coa3:t1.mitigates<.id=coa4:t2.mitigates<.id}
        as 
            [InvsecStart(), qfield("coa3:t1.mitigates<.id"), ConstrRelop('='), 
                 qfield("coa4:t2.mitigates<.id"), InvsecStop()]
            
        This allows us to use the template() code to also resolve the two qfields embedded 
        in the constraint non-terminal.
        
        NOTE: The constraint non-terminal does not result in generated text.
        
        The ConstrRelop("=") is used by process_terminal() to add a WHERE term to the Cypher query.

        """
        # print("constraint:", args)
        (i, relop_tok) = seq_find(enumerate(args), lambda t: type(t[1]) == lark.Token and t[1].type == "RELOP")
        qf_LHS = self.qfield(args[:i])
        qf_RHS = self.qfield(args[i+1:])

        lst = [InvsecStart(), qf_LHS, ConstrRelop(relop_tok.value), qf_RHS, InvsecStop()]
        return lst

    def invsection(self, args):
        # print("invsection: ", args)
        flat_args = [x for a in args for x in( a if type(a)==list else [a])]  
        lst = [InvsecStart()]
        lst.extend(flat_args)
        lst.append(InvsecStop())
        return lst
    
    def cnameseq(self, args):
        # print("cnameseq: ", args)
        return args
    
    def vardef(self, args):
        # print("vardef:", args)
        if self.is_node_type(self.map_nodetype(args[0])):
            err = f"TmplParseTransf.vardef ERROR: variable name {args[0]} is also a type or a type alias"
            print(err)
            raise ValueError(err)
        
        varname_safe = args[0].value.replace("-", "_")
        vs = Vardef(varname_safe)
        return vs
    
    def subscript(self, args):
        # print("subscript:", args)
        return args[0]
    
    def subscr_exp(self, args):
        # print("subscr_exp:", args)
        if len(args) == 1:
            quan = args[0].value
            oper = None
            val = None
            if args[0].type == "POSINTEGER":
                val = int(args[0].value)
                quan = "index"
        else:
            quan = args[0].value
            oper = args[1].value
            val = args[2]
        return Subscript(quan, oper, val)
    
    def svalue(self, args):
        # print("svalue:", args)
        tok = args[0]
        val = None
        if tok.type == "NUMBER":
            val = int(tok.value) if tok.value.isdigit() else float(tok.value)
        elif tok.type == "BOOL":
            val = True if tok.value == "true" else False
        elif tok.type == "QSTRING":
            val = tok.value[1:-1]    # skip '' or "" characters
        elif tok.type == "OTHERVALUE":
            val = parse_datetime(tok.value)
            # CAUTION: this throws dateutil.parser._parser.ParserError in case of error
        return val
    
    def scope(self, args):
        # TODO: NOT IMPLEMENTED YET
        # print("vardef:", args)
        raise RuntimeError("Scope terminal NOT IMPLEMENTED YET")
        # return Scope(args)
        
    def TEXTSECTION(self, arg):
        # print("TEXTSECTION: ", arg)
        return TextSection(arg.value)
    
    def INVCONTENT(self, arg):
        return TextSection(arg.value)
    
    def CNAME(self, arg):
        # print("CNAME: ", arg)
        return arg
    
    def VNAME(self, arg):
        # print("VNAME: ", arg)
        return arg
    
# -----------------------------------------------------------    

class TmplParser:
    """
    Class that parses one string to a list of ParseElements,
    such as Textsections and Pathspecs:
    [TextSection(text='this is c's property: '),    
    Pathspec(lst=[Nodespec(varname='bx', ntype='b'), 
                  Relspec(relname='q'), Nodespec(varname='_c2', ntype='c'), 
                  Propspec(propname='gamma')])]        
    """
    def __init__(self, grammar:str, parsecfg:dict, sch_dct:dict):
        self.grammar = grammar
        self.parsecfg = parsecfg
        self.schema_dct = sch_dct
        self.parser = Lark(grammar, start='template')

    def parse(self, text:str, 
              lark_trnsfrmr_cls:lark.Transformer=TmplParseTransf) -> list[ParseElement]:
        tree_transformer = lark_trnsfrmr_cls(self.parsecfg, self.schema_dct)

        tree = self.parser.parse(text)
        
        # print(tree)
        # print(tree2str(tree))
        lst_elems = tree_transformer.transform(tree)
        return lst_elems


class TmplGenNeo4j:
    def __init__(self, options:dict):
        self.options = options
        self.grammar = options.get("grammar", make_templ_grammar1())
        
        self.gencfg = options.get("gen_conf", json.loads(readfile(options["gen_conf_file"])))
        self.gencfg['verbose'] = self.options.get('verbose', self.gencfg.get('verbose', 0))
        self.neo4j_conf = options["neo4j_conf_file"]
        
        if options.get("debug", False):
            self.neo4j_driver = DebugNeo4jDriver(make_graph_test())
        else:
            self.neo4j_driver = Neo4jDriver(self.neo4j_conf)
                        
        # self.querygen = options["querygen"]
        (self.schema_dict, self.schemagraph) = self.get_schema()
        self.tmpl_parser = TmplParser(self.grammar, self.gencfg, self.schema_dict)
        
    def get_schema(self) -> tuple[dict, SchemaGraph()]:
        """
        Obtain the schema graph from the DB server and return G(V, E).

        Returns a tuple with
        -------
        schema_dict : dict
            schema dictionary with G(V, E).
        schemagraph : SchemaGraph
            same info as in schema_dict encoded with adjacency list.

        """
        schema_dict = self.neo4j_driver.get_db_schema()
        schemagraph = SchemaGraph(schema_dict["adj_lst"])
                
        # print("\n\n***********\n")
        # print(schema_dict)
        
        with open("_db_schema.json", "w") as fout:
            fout.write(json.dumps(schema_dict, indent=4))
        
        return (schema_dict, schemagraph)
        
    def format_prop_val(self, return_spec:str, nodespec:Nodespec, 
                        propspec:Propspec, propval:str|list|datetime.datetime) -> str:
        """
        Formats a property value returned from a Cypher query.
        Returns the resulting string.
        
        TODO: handle dict values encoded in JSON.
        """
        default_sep = ", "
        pv = propval
        if type(propval) == list:
            if propspec.subscript != None and propspec.subscript.quantif == "index":
                # like: technique.x-mitre-platforms[0]: return just one value from list:
                idx = propspec.subscript.val
                
                # TODO: what to do if idx >= len(propval) ??
                if idx >= len(propval):
                    pv = "N/A"
                    # raise ValueError(f"TmplGenNeo4j.format_prop_val ERROR: wrong index in {nodespec}.{propspec}")
                else:
                    pv = self.format_prop_val(return_spec, nodespec, propspec, propval[idx])
            else:
                # return the entire list formatted as a string:
                sep = self.gencfg.get("list_property_separator", default_sep)
                lst = [self.format_prop_val(return_spec, nodespec, propspec, val) for val in propval]
                pv = sep.join(lst)
        # elif type(propval) == datetime.datetime:
        elif type(propval) == neo4j.time.DateTime:
            dt = propval.to_native()
            dt_utc = dt.astimezone(timezone.utc)
            pv = dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        else:
            pv = str(propval)
            # Sanitise freeform CTI text fields (description, descriptions,
            # extended_description, notes, detection) before substitution.
            # See FREEFORM_PROPS / clean_cti_description() module-level docs.
            if propspec.propname in FREEFORM_PROPS:
                pv = clean_cti_description(pv)
        return pv
    
    def OLD_format_prop_val(self, propname:str, propval:str|list) -> str:
        """
        Formats a property value returned from a Cypher query.
        Returns the resulting string.
        """
        default_sep = ", "
        pv = propval
        if type(propval) == list:
            sep = self.gencfg.get("list_property_separator", default_sep)
            lst = [self.format_prop_val(propname, val) for val in propval]
            pv = sep.join(lst)
        else:
            pv = str(propval)
        return pv
    
 
    # Regexes for native MCQ option shuffling (see _shuffle_mcq_options).
    _MCQ_OPT_RE = re.compile(r"^([A-E])\)[ \t]+(.+?)[ \t]*$", re.M)
    _MCQ_ANS_RE = re.compile(r"Therefore,\s+([A-E])\.")
    # JSON-output MCQ answer marker: "answer": "X" (used by JS.MCQ.* v8 templates
    # whose Answer body wraps a JSON object in <OBR>..<CBR> sentinels).
    _MCQ_JSON_ANS_RE = re.compile(r'"answer"\s*:\s*"([A-E])"')

    @classmethod
    def _shuffle_mcq_options(cls, text:str) -> str:
        """
        For an MCQ-style rendered triple (5 option lines labelled A)..E) in the
        Question block and a final answer marker in the Answer), shuffle the
        options to a random A-E order and remap the answer letter to the new
        position of the originally correct option.

        Two answer-marker shapes are supported:
          * narrative form: "Therefore, <letter>."  (legacy AB.MCQ.* templates)
          * JSON form: '"answer": "<letter>"'       (v8 JS.MCQ.* templates)

        The option block is identified as the 5 A-E line-starts immediately
        preceding the answer marker. This tolerates incidental "A) ..." bullet
        markers inside CTI description text, which previously caused the
        global-match version to bail on ~50% of MCQ rows (leaving the template
        default answer letter in place and producing a position-A training
        bias).

        Returns the original text unchanged when:
          * no recognised answer marker is present, or
          * fewer than 5 A-E line-starts precede the marker, or
          * the 5 matches immediately before the marker are not A,B,C,D,E in order.
        """
        ans_match = cls._MCQ_ANS_RE.search(text)
        ans_re = cls._MCQ_ANS_RE
        ans_fmt = "Therefore, {}."
        if not ans_match:
            ans_match = cls._MCQ_JSON_ANS_RE.search(text)
            ans_re = cls._MCQ_JSON_ANS_RE
            ans_fmt = '"answer": "{}"'
        if not ans_match:
            return text

        head = text[:ans_match.start()]
        all_opts = list(cls._MCQ_OPT_RE.finditer(head))
        if len(all_opts) < 5:
            return text

        matches = all_opts[-5:]
        if [m.group(1) for m in matches] != list("ABCDE"):
            return text

        orig_correct_idx = ord(ans_match.group(1)) - ord('A')

        options = [m.group(2) for m in matches]
        perm = list(range(5))
        random.shuffle(perm)
        new_options = [options[perm[i]] for i in range(5)]
        new_correct_letter = chr(ord('A') + perm.index(orig_correct_idx))

        start, end = matches[0].start(), matches[-1].end()
        new_block = "\n".join(f"{chr(ord('A')+i)}) {new_options[i]}" for i in range(5))
        new_text = text[:start] + new_block + text[end:]
        new_text = ans_re.sub(ans_fmt.format(new_correct_letter), new_text, count=1)
        return new_text

    def process_template(self, tmplobj:dict) -> tuple[list[str], str]:
        """
        Process one template described by a JSON object (a dict) with properties:
            text, shortname?, count_limit?, coverage_limit?, order_by*,       (* are optional)

        Parameters
        ----------
        tmplobj : dict
            a template object - a dict.

        Returns
        -------
        tuple[list[str], str]
            (list with generated strings, string with Cypher query).

        """
        lst_prs_results = self.tmpl_parser.parse(tmplobj["text"])

        # lst_prs_results  is a list of TextSection alternating with Pathspec objects
        # a Pathspec objects embeds a list of [Nodespec, [Relspec, Nodespec]*, Propspec(prop)]
        # Each Pathspec will be replaced by nd.prop where nd is the target node on the
        # Pathspec chain and prop is the last (and only) property.
        
        # Per-template row count resolution. Priority (highest to lowest):
        #   1. gencfg.override_count_limit  - operator hard-override applied to
        #      all templates (rare; used to force-cap a whole run).
        #   2. tmplobj.count_limit          - per-template Count: directive
        #      from the .txt/.docx template. Authoritative author intent; used
        #      e.g. to cap MCQ templates so they do not dominate the dataset.
        #   3. CLI --count_max              - operator's broad cap for templates
        #      that do not declare a Count: of their own.
        #   4. gencfg.default_count_limit   - final fallback.
        #
        # The CLI --count_max additionally acts as a ceiling on whichever value
        # was selected above (1 or 2), so an operator can always tighten a run
        # without editing templates.
        override = self.gencfg.get("override_count_limit")
        tmpl_limit = tmplobj.get("count_limit")
        cli_limit = self.options.get("count_max", -1)
        default = self.gencfg.get("default_count_limit", 10)

        if override is not None:
            limit = override
        elif tmpl_limit is not None:
            limit = tmpl_limit
        elif cli_limit >= 0:
            limit = cli_limit
        else:
            limit = default

        if cli_limit >= 0:
            limit = min(limit, cli_limit)
            
        lst_match = list()
        lst_return = list()             # used to build the RETURN params       
        lst_text_struct = list()        # it has its structure of the template; fill it with record fields
        set_edge_qrs = set()            # MATCH 2-edge paths, e.g. "(a0:a) -[:r] -> (b0:b)"
        set_returns = set()          
        # above  sets are needed bc. neo4j prohibits repeating MATCH and RETURN parameters
        #   ... we need to check for them before adding to the query text        
        
        is_invisible_counter = 0        # if > 0 then don't include parse results in generated text  
        
        # TODO: for now, we do not take out properties in the invisible section 
        # to be returned by the MATCH query.
        # We exclude those returned properties + TextSections from generating the final string.
        
        for pr in lst_prs_results:
            # is_invisible_counter not used here; but we exercise the counter thing
            match pr:
                case Pathspec(lst):
                    str_edge_qry = self.gen_edge_info(pr)
                    # propspec = Propspec(f"{lst[-2].varname}.{lst[-1].propname}", None)
                    # TODO: CHANGE THIS:
                    nodespec, propspec = self.find_node_prop_specs(pr)
                    # propspec = Propspec(f"{lst[-2].varname}.{lst[-1].propname}", None)
                    return_spec = f"{nodespec.varname}.{propspec.propname}"
                    
                    # remember template structure
                    # lst_text_struct.append(propspec)
                    lst_text_struct.append((return_spec, nodespec, propspec))

                    # avoid repeating query parameters:
                    if str_edge_qry not in set_edge_qrs:
                        lst_match.append(str_edge_qry)
                        set_edge_qrs.add(str_edge_qry)

                    if return_spec not in set_returns:
                        lst_return.append(return_spec)
                        set_returns.add(return_spec)

                case TextSection(text):
                    # remember template structure
                    lst_text_struct.append(pr)
                    
                case ConstrRelop():
                    # needed for WHERE clause terms
                    lst_text_struct.append(pr)
                    
                case InvsecStart():
                    # must copy start/stop elements since they will control emission of text for result
                    lst_text_struct.append(pr)
                    is_invisible_counter += 1
                    
                case InvsecStop():
                    lst_text_struct.append(pr)
                    is_invisible_counter -= 1
                    
        str_match = ", ".join(lst_match)
        str_return = ", ".join(lst_return)
        
        # no WITH yet
        str_with = ""
        
        # generate WHERE expression for filter subscript operator on property value
        lst_filters = []
        
        str_modif_constraints = self.qry_make_modif_constraints(lst_prs_results)
        if str_modif_constraints: 
            lst_filters.append(str_modif_constraints)

        str_constraints = self.qry_make_constraints(lst_prs_results)
        if str_constraints: 
            lst_filters.append(str_constraints)
        
        str_filter = self.qry_make_filter(lst_prs_results)
        if str_filter: 
            lst_filters.append(str_filter)
        
        # # ensures "primary" nodes are distinct:
        # str_distinct = self.qry_make_distinct(lst_prs_results)
        # if str_distinct:
        #     lst_filters.append(str_distinct)
            
        # WHERE clause:
        str_where = ""
        if len(lst_filters) > 0:
            str_where = "WHERE " + " AND ".join(lst_filters)
        
        str_order = self.qry_make_order(tmplobj)

        # Optional primary-node pre-sampling: if the template declares
        # `sample: <varname>`, draw LIMIT random primary nodes first and
        # expand the rest of the pattern from that bounded set. This caps
        # Cartesian fan-out on high-out-degree nodes (e.g. intrusion-set in
        # AB.TAA, attack-pattern in AB.MCQ negative sampling) and prevents
        # the DISTINCT/ORDER-BY-rand() full-graph materialisation.
        sample_varname = tmplobj.get("sample", "").strip()
        str_sample_prefix = ""
        sample_grouping_active = False
        str_inner_return_vars = ""
        sv_safe = ""
        if sample_varname:
            sample_ntype = None
            for pr in lst_prs_results:
                if isinstance(pr, Pathspec):
                    for spec in pr.lst:
                        if isinstance(spec, Nodespec) and spec.varname == sample_varname:
                            sample_ntype = spec.ntype
                            break
                if sample_ntype:
                    break
            if sample_ntype:
                sv_safe = neo4j_safe_identif(sample_varname)
                st_safe = neo4j_safe_identif(sample_ntype)
                str_sample_prefix = (f"MATCH ({sv_safe}:{st_safe}) "
                                     f"WITH DISTINCT {sv_safe} ORDER BY rand() LIMIT {limit} ")
                # Per-primary grouping: collect all non-primary varnames that
                # appear in RETURN so we can yield one row per primary via a
                # CALL subquery with LIMIT 1. This ensures each of the LIMIT
                # primaries contributes one combination (deterministic first
                # match), giving LIMIT distinct subjects per template.
                other_vars = []
                seen = {sample_varname}
                for rs in lst_return:
                    vn = rs.split(".", 1)[0]
                    if vn not in seen:
                        other_vars.append(vn)
                        seen.add(vn)
                if other_vars:
                    sample_grouping_active = True
                    str_inner_return_vars = ", ".join(neo4j_safe_identif(v) for v in other_vars)
            else:
                print(f"  WARN: sample var '{sample_varname}' not found among parsed nodes; ignoring sample directive")

        # Primary form: sample over the full candidate space for diversity
        # (LIMIT after RETURN DISTINCT ... ORDER BY rand()).
        # Optional per-primary grouping (gated by gencfg "per_primary_grouping"):
        # when active, wrap the pattern expansion in a CALL subquery with LIMIT 1
        # per primary so each of the LIMIT primaries contributes exactly one row.
        # This is experimental because the planner's handling of the resulting
        # query is sensitive to graph shape and can be very slow on templates
        # with many free bindings of the same type; keep off by default.
        # Fallback form: apply LIMIT before RETURN to bound memory for queries
        # whose Cartesian fan-out would otherwise exceed dbms.memory.transaction.total.max.
        use_per_primary = bool(self.gencfg.get("per_primary_grouping", False))
        if sample_grouping_active and use_per_primary:
            match_query = (
                f"{str_sample_prefix}"
                f"CALL ({sv_safe}) {{ "
                f"MATCH {str_match}     {str_where}{str_with}     "
                f"RETURN {str_inner_return_vars}     LIMIT 1 }}     "
                f"RETURN DISTINCT {str_return}     {str_order}     LIMIT {limit}"
            )
            match_query_fallback = (
                f"{str_sample_prefix}MATCH {str_match}     {str_where}{str_with}"
                f"     LIMIT {limit}     RETURN DISTINCT {str_return}     {str_order}"
            )
        else:
            match_query = f"{str_sample_prefix}MATCH {str_match}     {str_where}{str_with}     RETURN DISTINCT {str_return}     {str_order}     LIMIT {limit}"
            match_query_fallback = f"{str_sample_prefix}MATCH {str_match}     {str_where}{str_with}     LIMIT {limit}     RETURN DISTINCT {str_return}     {str_order}"

        if self.gencfg.get("verbose", 0) > 0:
            print(f"\nMatch query:\n{match_query}\n")

        # a sequence of results.  the primary form is sampled for diversity but
        # can stall on heavy cartesians; gencfg "primary_query_timeout_s" (default
        # 90.0) bounds wall time and triggers the bounded fallback on timeout or
        # transaction-memory exhaustion.
        primary_timeout = self.gencfg.get("primary_query_timeout_s", 90.0)
        try:
            qry_results = self.neo4j_driver.run_query_collect(match_query, timeout=primary_timeout)
        except (neo4j.exceptions.TransientError, neo4j.exceptions.ClientError, neo4j.exceptions.DriverError) as e:
            msg = str(e)
            is_memory = "MemoryPool" in msg or "memory pool" in msg.lower()
            is_timeout = ("TransactionTimedOut" in msg or "transaction has been terminated"
                          in msg.lower() or "timed out" in msg.lower())
            if is_memory or is_timeout:
                reason = "transaction memory" if is_memory else f"timeout (>{primary_timeout:.0f}s)"
                print(f"  WARN: primary query hit {reason}; falling back to bounded form")
                if self.gencfg.get("verbose", 0) > 0:
                    print(f"\nFallback query:\n{match_query_fallback}\n")
                # Bound the fallback too: without a tx timeout it can only be
                # cleared by an external watchdog, which in turn surfaces as an
                # uncaught "Explicitly terminated by the user" ClientError that
                # poisons the whole template (e.g. Q.MSR.1, AB.MCQ.3).
                qry_results = self.neo4j_driver.run_query_collect(
                    match_query_fallback, timeout=primary_timeout)
            else:
                raise
        
        lst_gentext = list()      # stores all generated texts
        shuffle_mode = str(tmplobj.get("shuffle", "")).strip().lower()
        for record in qry_results:
            # record is a neo4j Record object that looks like a dictionary, 
            # key:value pairs, where key is nodevar and value is the value from the DB
            lst_vals = list()   # used to generate the template text        
            is_invisible_counter = 0
            
            for ts_or_prps in lst_text_struct:
            # for (i, pr) in enumerate(lst_prs_results):
                match ts_or_prps:
                    case (return_spec, nodespec, propspec):
                    # case Propspec(propname):
                        if is_invisible_counter == 0: 
                            # take record field from DB and format it according to its type:
                            prop_str = self.format_prop_val(return_spec, nodespec, propspec, record[return_spec])
                            lst_vals.append(prop_str)
                        
                    case TextSection(text):
                        # regular text section, use if not invisible:
                        if is_invisible_counter == 0: 
                            lst_vals.append(text)

                    case ConstrRelop():
                        pass

                    case InvsecStart():
                        # invisible sections can be nested.
                        is_invisible_counter += 1
                        
                    case InvsecStop():
                        is_invisible_counter -= 1

            gentext = "".join(lst_vals)
            if shuffle_mode == "mcq":
                gentext = self._shuffle_mcq_options(gentext)
            lst_gentext.append(gentext)
        return [lst_gentext, match_query]

        
    def gen_edge_info(self, ps:Pathspec) -> str:
        """
        Takes a Pathspec([[Nodespec,Relspec]*, Propspec]) and returns a string from it.
        E.g. Pathspec(lst=[Nodespec(varname='bx', ntype='b'), Relspec(relname='q'), 
                           Nodespec(varname='_c2', ntype='c'), Propspec(propname='gamma')])
           is converted to "(bx:b) - [:q] -> (_c2:c)"
           
           Pathspec(lst=[Nodespec(varname='bx', ntype='b'), Relspec(relname='p'), 
                         Nodespec(varname='_e2', ntype='e'), Relspec(relname='r'), 
                         Nodespec(varname='_c3', ntype='c'), Propspec(propname='gamma')]
        is converted to:
            "(bx:b) - [:p] -> (_e2:e), (_e2:e) -> [:r] - (_c3:c)"

        Parameters
        ----------
        ps : Pathspec
            Pathspec([[Nodespec,Relspec]+, Propspec]). [...]+  can repeat >=1 times

        Returns
        -------
        str
            Pathspec converted to string.

        """
        lst_all = list()
        # lst_edge = list()
        # rel = None
        for spec in ps.lst:
            match spec:
                case Nodespec(varname, ntype):
                    ntsafe = neo4j_safe_identif(ntype)
                    varname_safe = neo4j_safe_identif(varname)
                    node_str = f"({varname_safe}:{ntsafe})"
                    lst_all.append(node_str)
                case Relspec(relname):
                    relname_safe = neo4j_safe_identif(relname)
                    lst_all.append(f"-[:{relname_safe}]->")
                case InvRelspec(relname):
                    relname_safe = neo4j_safe_identif(relname)
                    lst_all.append(f"<-[:{relname_safe}]-")
                case _:   # skip property and other specs
                    pass
        ei = "".join(lst_all)
        return ei
                    

    def format_value(self, value:int|float|str|datetime.datetime) -> str:
        st = ""
        if type(value) == str:
            if "\"" in value:
                st = f"'{value}'"
            else:
                st = f'"{value}"'
        elif type(value) == datetime.datetime:
            sdt = format_datetime(value)
            st = f"datetime('{sdt}')"
        else:
            st = str(value)
        return st
        
    def make_subscript_term(self, nodespec:Nodespec, propspec:Propspec) -> str:
        """
        Creates a condition term string based on nodespec and its property spec: 
            property name and its subscript object.
        """
        # mapping operators from template language to cypher syntax
        vn = nodespec.varname
        dct_ops = {"!=": "<>"}
        term = ""
        def_ret = None
        s = propspec.subscript
        op = dct_ops.get(s.op, s.op)
        val = self.format_value(s.val)
        
        if s.quantif == "_":
            # for non-list property. E.g. t.version[_='2.1']
            if s.op == "~":   # case-insensitive substring check:
                term = f"(toLower({vn}.{propspec.propname}) CONTAINS toLower({val}))" 
            else:
                term = f"({vn}.{propspec.propname}{op}{val})" 
        elif s.quantif == "index":
            # case t.x_mitre_platforms[0]; nothing to do here
            term = def_ret
        elif s.quantif == "?":
            # for list property. E.g. t.version[?='2.1'] or t.version[?!='2.1'], at least one of values equal to
            if s.op == "=":
                term = f"({val} IN {vn}.{propspec.propname})" 
            elif s.op == "!=": # value not in list
                term = f"(NOT {val} IN {vn}.{propspec.propname})" 
            elif s.op == "~": # case inseneitive subbstring check
                propvar = f"_{propspec.propname}"
                term = f"ANY({propvar} IN {vn}.{propspec.propname} WHERE \
                    toLower({propvar}) CONTAINS toLower({val}))"                
            else:
                raise ValueError(f"TmplGenNeo4j.make_subscript_term ERROR: invalid subscript operator in {propspec}")
        elif s.quantif == "*":
            if s.op == "=":
                term = f"ALL(x IN {vn}.{propspec.propname} WHERE x={val})" 
            elif s.op == "!=":
                term = f"NONE(x IN {vn}.{propspec.propname} WHERE x={val})" 
            elif s.op == "~": # case inseneitive subbstring check
                propvar = f"_{propspec.propname}"
                term = f"ALL({propvar} IN {vn}.{propspec.propname} WHERE \
                    toLower({propvar}) CONTAINS toLower({val}))"                
            else:
                raise ValueError(f"TmplGenNeo4j.make_subscript_term ERROR: invalid subscript operator in {propspec}")
        else:
            raise ValueError(f"TmplGenNeo4j.make_subscript_term ERROR: invalid quantifier in subscript operator in {propspec}")
        return term


    def find_node_prop_specs(self, pathspec:Pathspec) -> tuple[Nodespec, Propspec]:
        propspec, nodespec = None, None
        for i in range(len(pathspec.lst) - 1, 0, -1):
            if type(pathspec.lst[i]) == Propspec:
                propspec, nodespec = pathspec.lst[i], pathspec.lst[i - 1]
                break;
        
        if propspec == None:
            err = f"TmplGenNeo4j.find_node_prop_specs ERROR: no Propspec for template with parse results {pathspec}"
            print(err)
            raise ValueError(err)
        return (nodespec, propspec)


    def qry_make_filter(self, lst_prs_results:list) -> str:
        """
        Creates a WHERE term for the subscript operators applied to the property values.

        Parameters
        ----------
        lst_prs_results : list
            list of parse results.

        Returns
        -------
        str
            WHERE conjunction term.

        """
        lst_terms = list()
        propspec = None
        for p_or_t in lst_prs_results:
            match p_or_t:
                case Pathspec():
                    (nodespec, propspec) = self.find_node_prop_specs(p_or_t)                    
                    if propspec.subscript != None:
                        # pass the Nodespec and the following Propspec:
                        term = self.make_subscript_term(nodespec, propspec)
                        if term:
                            lst_terms.append(term)
                case _:
                    pass
        where_filter = " AND ".join(lst_terms) if len(lst_terms) > 0 else ""
        return where_filter
    


    def qry_make_modif_constraints(self, lst_prs_results:list) -> str:
        """
        Creates WHERE conjunctive terms for 'modified' date constraints if present in the
        generation configuration dictionary.
        
        CAUTION: These contraints are not checked agaist other datetime constraints in subscript operations.
        
        Parameters
        ----------
        lst_prs_results : list
            list of parse results.

        Returns
        -------
        str
            WHERE conjunction term for modified datetime constraints.

        """
        modified_constr_key = "modified_constraints"
        modified_field = "modified"
        if modified_constr_key not in self.gencfg or \
                    not self.gencfg[modified_constr_key].get("enabled", False):
            return ""   # nothing to do
                
        lst_terms = list()
                 
        dct_mc = self.gencfg[modified_constr_key]
        dt_after = parse_datetime(dct_mc.get("after", None))
        dt_before = parse_datetime(dct_mc.get("before", None))
        apply_to = dct_mc.get("apply_to", None)
        apply_op = " OR " if apply_to == "any" else " AND "
        
        set_nodes = set()
        
        if apply_to == None:
            return ""     # unspecified constraint; nothing to do
        
        if apply_to not in ["any", "all", "target"]:
            raise ValueError(f"qry_make_modif_constraints ERROR: invalid 'apply_to' field in configuration: '{apply_to}'")
            
        for p_or_t in lst_prs_results:
            path_cond = ""
            match p_or_t:
                case Pathspec(lst):
                    if apply_to == "target":
                        (nodespec, propspec) = self.find_node_prop_specs(p_or_t)
                        if nodespec.varname not in set_nodes:
                            lst3 = list()
                            if dt_after:
                                lst3.append(f"{nodespec.varname}.{modified_field}>={self.format_value(dt_after)}")
                            if dt_before:
                                lst3.append(f"{nodespec.varname}.{modified_field}<={self.format_value(dt_before)}")
                            # set_nodes.add(nodespec.varname)
                            # lst_path.append(" AND ".join(lst3))
                            path_cond = " AND ".join(lst3)

                    else:
                        lst2 = list()
                        for spec in lst:
                            match spec:
                                case Nodespec(varname, ntype):
                                    lst3 = list()
                                    if varname not in set_nodes:
                                        if dt_after:
                                            lst3.append(f"{varname}.{modified_field}>={self.format_value(dt_after)}")
                                        if dt_before:
                                            lst3.append(f"{varname}.{modified_field}<={self.format_value(dt_before)}")
                                        # set_nodes.add(varname)
                                        node_cond = " AND ".join(lst3)
                                        if node_cond not in lst2:
                                            lst2.append(node_cond)   
                                case _:
                                    pass
                        if len(lst2) > 0: 
                            # lst_path.append(f"({apply_op.join(lst2)})")
                            path_cond  = f"({apply_op.join(lst2)})"
                            
                    if path_cond and path_cond not in lst_terms:
                    # if len(lst_path) > 0 and :
                        # path_term = " AND ".join(lst_path)
                        # lst_terms.extend(lst_path)
                        lst_terms.append(path_cond)

        
        # lst_terms.extend(lst_pairs)
        # where_filter = " AND ".join(lst_terms) if len(lst_terms) > 0 else ""
        where_filter = apply_op.join(lst_terms) if len(lst_terms) > 0 else ""
        return where_filter


    def qry_make_constraints(self, lst_prs_results:list) -> str:
        """
        Creates WHERE terms for (Pathspec, ConstrRelop, Pathspec) sequences present in 
        lst_prs_results and also for "primary" nodes to be distinct.
        Primary nodes are those at the start of each Pathspec.
        
        For each pair of nodes n, m of the same type  it generates "n <> m",
        to be added to the WHERE clause.
        E.g. for constraint: {force coa3:t1.mitigates<.id=coa4:t2.mitigates<.id} 
        it generates Cypher WHERE term coa3.id = coa4.id
        
        Example with distict primary nodes:
            "{technique.name} ... {technique.used-by>.tool}" two technique variables are different
            "{coa1:t1.mitigates<.name} {coa2:t2.mitigates<.name}" two mitigation variables are different
        
        The Constraint with ConstrRelop takes precedence.
        
        Parameters
        ----------
        lst_prs_results : list
            list of parse results.

        Returns
        -------
        str
            WHERE conjunction term for constraints.

        """
        set_var_pairs = set()
        lst_terms = list()
        
        # first, get Constraints:
        for (i, p_or_t) in enumerate(lst_prs_results):
            match p_or_t:
                case ConstrRelop(relop):
                    pslhs = lst_prs_results[i - 1]
                    psrhs = lst_prs_results[i + 1]
                    
                    lhs_nodespec, lhs_propspec = self.find_node_prop_specs(pslhs)
                    rhs_nodespec, rhs_propspec = self.find_node_prop_specs(psrhs)
                    
                    term = f"{lhs_nodespec.varname}.{lhs_propspec.propname} {relop} {rhs_nodespec.varname}.{rhs_propspec.propname}"
                    lst_terms.append(term)
                    vp = (lhs_nodespec.varname, rhs_nodespec.varname)
                    set_var_pairs.add((min(vp), max(vp)))
                    
        # add terms for distinct primary nodes:
        dct_node_vars = dict()
        for p_or_t in lst_prs_results:
            match p_or_t:
                case Pathspec([Nodespec(varname, ntype), *_]):
                    if ntype not in dct_node_vars:
                        dct_node_vars[ntype] = set()
                    dct_node_vars[ntype].add(varname)
                case _:
                    pass
        
        # print(dct_node_vars)
        make_pairs = lambda seq: [f"elementId({u}) <> elementId({v})" for u in seq for v in seq 
                                  if u < v and ((u, v) not in set_var_pairs)]

        lst_lst_pairs = [make_pairs(sv) for sv in dct_node_vars.values() if len(sv) > 1]
        lst_pairs = [x for w in lst_lst_pairs for x in w]

        lst_terms.extend(lst_pairs)
        where_filter = " AND ".join(lst_terms) if len(lst_terms) > 0 else ""
        return where_filter


    def qry_make_distinct(self, lst_prs_results:list) -> str:
        """
        Creates a WITH condition that enforces "primary" nodes are distinct.
        Primary nodes are those at the start of each Pathspec.
        
        For each pair of nodes n, m of the same type  it generates "n <> m",
        to be added to the WHERE clause.
        """
        # print(lst_prs_results)
        dct_node_vars = dict()
        for p_or_t in lst_prs_results:
            match p_or_t:
                case Pathspec([Nodespec(varname, ntype), *_]):
                    if ntype not in dct_node_vars:
                        dct_node_vars[ntype] = set()
                    dct_node_vars[ntype].add(varname)
                case _:
                    pass
        
        # print(dct_node_vars)
        # make_pairs = lambda seq: [f"{u} <> {v}" for u in seq for v in seq if u < v]
        make_pairs = lambda seq: [f"elementId({u}) <> elementId({v})" for u in seq for v in seq if u < v]
        lst_lst_pairs = [make_pairs(sv) for sv in dct_node_vars.values() if len(sv) > 1]
        lst_pairs = [x for w in lst_lst_pairs for x in w]
        if len(lst_pairs) > 0:
            return " AND ".join(lst_pairs)
        else:
            return ""
    
    def qry_make_order(self, tmplobj:dict) -> str:
        """
        Returns the ORDER BY clause.
        Uses tmplobj['order_by'] if present. Otherwise uses default in gene. config.
        """
        
        # priority ordering:
        str_ob = self.gencfg.get('override_order_by', tmplobj.get('order_by', 
                                         self.gencfg.get('default_order_by', "")))
        
        if str_ob.lower() == "random":
            str_ob = "rand()"     # cypher function to create a random number
        if str_ob.lower() == "none":
            return ""
        return "ORDER BY " + str_ob
        
    
    def load_templates(self, jsonfn:str) -> object:
        """
        Loads a template object from a JSON file and returns the object.
        Used to load template descriptions.
        
        Parameters
        ----------
        jsonfn : str
            JSON file.

        Returns
        -------
        object
            JSON object.

        """
        js = json.loads(readfile(jsonfn))
        return js
    
    
    def generate(self, tmpl_objs:Iterable[dict], do_print:bool=False):
        """
        Generate text based on sequence of template objects (JSON dicts?).
        Return (count_generated, count_failed) tuple.
        
        Results saved in directory self.genconf['results_dir'].
        Generation report JSON saved to file 
        'results-dir'/_results-report-<TIMESTAMP>.json
        
        Exceptions thrown by a call to self.process_template are caught and reported.
        
        Parameters:
            tmpl_objs : Iterable[dict]
                template object iterable to generate from
                
            do_print : bool
                print to stdout if True
                
        Returns 
            (count_generated, count_failed) : tuple
        """
        results_dir = self.options["results_dir"]
        
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
            if do_print or self.gencfg.get("verbose", 0) > 0:
                print(f"process_templates: created results directory {results_dir}")
        
        # save generation results to new JSON file:
        dct_results = {"gen_conf_file": self.options.get("gen_conf_file", "unknown"), 
                       "results_dir": results_dir,
                       "neo4j_conf_file": self.options["neo4j_conf_file"]}
        
        dct_results["templates_file"] = self.options.get("templates_file", "unknown")
        dct_results["timestamp"] = format_datetime(datetime.datetime.now(datetime.UTC))
        dct_results["gencfg"] = self.gencfg, 
                             
        lst_results = [] 
        gen_count = 0
        all_gen_count = 0    # total number of strings/IFT triples generated
        failed_count = 0
        t_totaltime_s = 0.0  # total time to generate
        resobj = None
        for (i, tmplobj) in enumerate(tmpl_objs):
            try:
                t_comment = tmplobj.get("comment", "")
                t_text = tmplobj["text"]
                t_shortname = tmplobj.get("shortname", "noname")
                
                t_filename = os.path.join(results_dir, f"t_{i:05d}_{t_shortname}.json")
                                
                t_results = dict()
                t_results["gen_conf_file"] = dct_results["gen_conf_file"]
                t_results["templates_file"] = self.options.get("templates_file", "unknown")
                t_results["results_dir"] = results_dir                
                t_results["template_object"] = tmplobj
                t_results["timestamp"] = format_datetime(datetime.datetime.now(datetime.UTC))
                
                t_results["template_index"] = i
                
                resobj = {
                    "template_index": i,
                    "template_object": tmplobj                    
                    }
                lst_results.append(resobj)
                
                if do_print or self.gencfg.get("verbose", 0) > 0:
                    print(f'\n\nTEMPLATE {i}.\n')
                    print(f'Comment: {t_comment}\n')
                    print(f'Text:\n{t_text}\n')
                
                tstart = time.time()                # report elapsed time
                
                # -----------------------------
                # GENERATE STRING FROM TEMPLATE:
                (lst_tmpl_text, query_str) = self.process_template(tmplobj)            
                # -----------------------------
                
                telapsed = time.time() - tstart
                t_totaltime_s += telapsed
                all_gen_count += len(lst_tmpl_text)
                gen_count += 1
                
                resobj["generated_count"] = len(lst_tmpl_text)
                resobj["generation_time"] = telapsed
                resobj["query"] = query_str
                
                t_results["generation_time"] = telapsed
                t_results["generated_count"] = len(lst_tmpl_text)
                t_results["query"] = query_str
                
                t_results["generated_strings"] = lst_tmpl_text
                                
                with open(t_filename, "w") as fout:
                    json.dump(t_results, fout, indent=4)
                    
                if do_print or self.gencfg.get("verbose", 0) > 0:
                    print(f"GENERATED SAMPLES: {len(lst_tmpl_text)} in {telapsed} s\n")
                    for (j, txt) in enumerate(lst_tmpl_text):                    
                        print(f"{j}.\n{txt}")
                        print("---------------------------\n")                                        
                    print("=======================================")
            except Exception as ex:
                if do_print or self.gencfg.get("verbose", 0) > 0:
                    print("Exception caught")
                    print(ex)
                resobj["exception"] = str(ex)
                resobj["generated_count"] = 0
                failed_count += 1
            finally:
                pass
                        
        # counts templates successfully processed, not strings generated
        dct_results["all_generated_count"] = all_gen_count
        dct_results["generation_time"] = t_totaltime_s
        dct_results["failed_count"] = failed_count
        dct_results["results"] = lst_results
        
        # results_filename = os.path.join(results_dir, f"_results-report-{dct_results['timestamp']}.json")
        results_filename = os.path.join(results_dir, f"_results-report.json")

        with open(results_filename, "w") as fres:
            json.dump(dct_results, fres, indent=4)
            if do_print or self.gencfg.get("verbose", 0) > 0:
                print(f"Processed: {gen_count}  Failed {failed_count}   All gen {all_gen_count}")
                print(f"Results saved to file {results_filename} and JSON files in {results_dir} directory.")
        
        return (gen_count, failed_count)
    
# -----------------------------------------------------------    



    
def make_templates_dummy():
    templates2 = [
    #     "abcd efgh {qf1} ijk {qf2} lmn",
    #     "ab<*inv section1 *> cd<*inv section1*>de",
    #     "<* inv section {qf1} inv text*>",
    #     "abcd <* inv1 {qf1} inv2 ijk {qf2} *> efg {qf3} hij {qf4} jk",
    #     "ab [ cde {qf1}{qf2} fgh] <*{qf3}ij*>",
    #     "ab[cde{qf1}fgh{qf2}ij]kl",
        # r"ab {var1:scope#cd.ef.gh[?]} {var1:scope#cd.ef.gh[*='abcd']} ",
        # r"{var1:scope#cd.ef.gh[*='abcd']}", 
        # r'{var1:scope#cd.ef.gh[*="abcd"]}', 
        # 'ab{cd.ef.gh}ijk', 
        # 'ab{qf1}ijk{qf2}kl', 
        # "{ax.p.beta}...{a.p.q.gamma}...{ax:a.p.beta}",
        "{ax.p.beta}...{a.p.q.gamma}...{ax:a.p.beta}",
        # "...{ax.p.beta}...{a.p.q.gamma}",
    ]   
    
    templates = [
        # "aaa{ax.p.beta}bbb{a.p.p.eps}ccc{ax:a.p.q.s.delta}ddd",
        # "aaa{ax.p.beta}bbb{a.p.p.eps}ccc{ax:a.p.q.s.delta}ddd",
        "aaa {ax.p.beta} bbb {a.p.p.eps} ccc {ax:a.alpha} ddd {cx.s.delta}\
            ee {cx:ax.p.p.r.gamma}",
        ]
    return templates    


def make_parsecfg_test():
    cfg = {
        }
    return cfg
    


def make_graph_test():
    g = {
        "adj_lst": {
            "a": [("p", "b")],
            "b": [("q", "c"), ("p", "e")],
            "c": [("s", "d")],
            "d": [],
            "e": [("r", "c")]
            }
        }
    return g


def test_parsing():
    tmpl_parser = Lark(make_templ_grammar2, start='template')
    
    templates = [
    #     "abcd efgh {qf1} ijk {qf2} lmn",
    #     "ab<*inv section1 *> cd<*inv section1*>de",
    #     "<* inv section {qf1} inv text*>",
    #     "abcd <* inv1 {qf1} inv2 ijk {qf2} *> efg {qf3} hij {qf4} jk",
    #     "ab [ cde {qf1}{qf2} fgh] <*{qf3}ij*>",
    #     "ab[cde{qf1}fgh{qf2}ij]kl",
        # r"ab {var1:scope#cd.ef.gh[?]} {var1:scope#cd.ef.gh[*='abcd']} ",
        r"{var1:scope#cd.ef.gh[*='abcd']}", 
        r'{var1:scope#cd.ef.gh[*="abcd"]}', 
        # "ab {v1:cd.ef.gh}",
    ]    

    
    for i, tmpltxt in enumerate(templates):
        print(f"\n{i+1:02} {tmpltxt}")
        tree = tmpl_parser.parse(tmpltxt)
        
        print("======================")
        # print(tree)
        print(tree2str(tree))
        print("\n======================")
        print(tree.pretty())
        # print(tree)
        


def test_parsing_neo4j():
    templates = make_templates_dummy()
    
    neo4j_db_file = neo4j_TEST_config_filename
    neo4j_config_dct = json.loads(readfile(neo4j_db_file))
    schema_dict = neo4j_get_db_schema(neo4j_config_dct)
    
    if not schema_dict:
        print("Failure loading neo4j DB schema from file", neo4j_db_file)
        return
    
    sch_fn = f'schema_{neo4j_config_dct["nickname"]}.json'
    save_schema_to_json(schema_dict, sch_fn)
    # print("Saved neo4j DB schema to file '{sch_fn}'.")
    
    schemagraph = SchemaGraph(schema_dict["adj_lst"])

    tmpl_grammar = make_templ_grammar2()
    tmpl_parser = Lark(tmpl_grammar, start='template')
    parsecfg = make_parsecfg_test()
    
    tree_transformer = TmplParseTransf(parsecfg, schemagraph)

    for i, tmpltxt in enumerate(templates):
        print(f"\n{i+1:02} {tmpltxt}")
        tree = tmpl_parser.parse(tmpltxt)
        
        print("======================")
        # print(tree)
        print(tree2str(tree))
        print("\n======================")
        x = tree_transformer.transform(tree)
        print(type(x), x)
        
def test_parsing_dummy():
    # templates = make_templates_dummy()
    templates = [
        # "aaa{ax.p.beta}bbb{a.p.p.eps}ccc{ax:a.p.q.s.delta}ddd",
        # "aaa{ax.p.beta}bbb{a.p.p.eps}ccc{ax:a.p.q.s.delta}ddd",
        # "aaa {ax.p.id} bbb {a.p.p.id} ccc {ax:a.id} ddd {cx.s.id}\
        #     ee {cx:ax.p.p.r.id}",
        "aaa {cx.s.id} bbb {ax.p.id} ccc {cx:ax.p.p.r.id} ddd {a.p.p.id} eee {ax:a.id} fff",
        ]

    
    # neo4j_db_file = neo4j_TEST_config_filename
    # neo4j_config_dct = json.loads(readfile(neo4j_db_file))
    # schema_dict = neo4j_get_db_schema(neo4j_config_dct)
    
    schemagraph = make_graph_test()
    # schemagraph = SchemaGraph(schema_dict["adj_lst"])

    tmpl_grammar = make_templ_grammar2()
    tmpl_parser = Lark(tmpl_grammar, start='template')
    parsecfg = make_parsecfg_test()
    
    tree_transformer = TmplParseTransf(parsecfg, schemagraph)

    for i, tmpltxt in enumerate(templates):
        print(f"\n{i+1:02} {tmpltxt}")
        tree = tmpl_parser.parse(tmpltxt)
        
        print("======================")
        # print(tree)
        print(tree2str(tree))
        print("\n======================")
        x = tree_transformer.transform(tree)
        print("\n========================================")
        print(type(x), x)
        
     
        
def test_templ_gen_dummy():
    options = {
        "grammar": make_templ_grammar1(),
        "gen_config": {
            "inverse_rel_sep": "<",  # don't change
            "default_count_limit": 3,
            "inverse_relationships": {
                "used_by": "uses",
                # TODO
                }
            },
        "neo4j_conf": neo4j_TEST_config_filename
        }
    
    tmplgen = TmplGenNeo4j(options)
    
    # templates = make_templates_dummy()
    templates = [
#        "aaa {ax.p.id} bbb {a.p.p.id} ccc {ax:a.id} ddd {cx.s.id} ee {cx:ax.p.p.r.id}",
        # "aaa {cx.s.id} bbb {ax.p.id} ccc {cx:ax.p.p.r.id} ddd {a.p.p.id} eee {ax:a.id} fff",
            
        # inverse relationships:
        "{c.q<.p.id}",
        "{ax:d.s<.r<.p<.p<.id} {b.q.r<.p<.q.id} {ax.p.q.r<.id}"
        # circular variable references:
        # "{cx:bx.p.r.id} {bx:cx.r.id}",
        
        # unbounded variables: ax, cx
        # "aaa {cx.s.id} bbb {ax.p.id} ccc {cx:ax.p.p.r.id} ddd {a.p.p.id} eee",
    ]

    for (i, tmpl) in enumerate(templates):
        try:
            dct_tmpl = {"text": tmpl}
            print(f"\nTEMPLATE {i}\n {tmpl}\n")
            lst_text = tmplgen.process_template(dct_tmpl)
            for txt in lst_text:
                print(txt, "\n")
            print("---------------------------\n")
        except Exception as ex:
            print("Exception caught")
            print(ex)
      
            
def make_default_gencfg():
    cfg = {
        "version": "0.0",       
        'default_propname': "name",     # used when a property name is missing in qfield
        "default_count_limit": 10,
        "default_order_by": "none",
        "list_property_separator": ", ",  # used to format list properties
        "inverse_relationships": {
            "used-by": "uses",
            "used_by": "uses",
            "attributed_by": "attributed-to",
            "revokes": "revoked-by",
            "detected-by": "detects",
            "detected_by": "detects",
            "mitigated-by": "mitigates",
            "mitigated_by": "mitigates",
            "has-subtechnique": "subtechnique-of",
            "has_subtechnique": "subtechnique-of",
            "achieved-by": "achieves",       # attack-pattern -[:achieves]-> x-mitre-tactic
            "achieved_by": "achieves",
            # "": "",
            },
        
        "property_mappings": [
                {
                "id": ["public_id", ["course-of-action", "malware", "tool",
                                     "x-mitre-tactic", "attack-pattern", "x-mitre-data-component",
                                     "intrusion-set", "campaign", "x-mitre-data-source"]],
                }
            ],
        "nodetype_mappings": {
            "technique": "attack-pattern",
            "mitigation": "course-of-action",
            "tactic": "x-mitre-tactic"
            },
        "modified_constraints": {
            "after": "2025-04-16T00:00:00.000Z",
            "before": "2025-04-16T23:59:59.000Z",
            "before_ALT": "2025-04-02T17:29:15.914Z",
            "apply_to": "any",
            "apply_to_ALT": "all",
            "apply_to_ALT": "target",
            "enabled": 0
            }
        }    
    return cfg
            
def test_templ_gen_neo4j():
    options = {
        "gen_config": make_default_gencfg(),
        "neo4j_conf": neo4j_TEST_config_filename
        }
        
    with open(fn:="gencfg_default_neo4j.json", "w") as f:
        print(f"Saved default generation configuration to {fn}")
        json.dump(options["gen_config"], f, indent=4)
    
    tmplgen = TmplGenNeo4j(options)
    
    lst_tmplobjs = tmplgen.load_templates("sample-tmpl-attack.json")
    for (i, tmplobj) in enumerate(lst_tmplobjs):
        if i != len(lst_tmplobjs) - 1 :
            continue
        try:
            tmp_comment = tmplobj.get("comment", "")
            tmp_text = tmplobj["text"]

            print(f'\n\nTEMPLATE {i}.\n')
            print(f'Comment: {tmp_comment}\n')
            print(f'Text:\n{tmp_text}\n')
            
            lst_tmpl_text = tmplgen.process_template(tmplobj)            
            
            print(f"GENERATED SAMPLES: {len(lst_tmpl_text)}\n")
            for (j, txt) in enumerate(lst_tmpl_text):
                print(f"{j}.\n{txt}")
                print("---------------------------\n")
            print("=======================================")
        except Exception as ex:
            print("Exception caught")
            print(ex)
        
def test_tmplgen():
    gen_cfg_file = "gencfg_default_neo4j.json"
    template_json_file = "sample-tmpl-attack.json"
    
    options = {
        "gen_conf_file": gen_cfg_file,
        "templates_file": template_json_file,
        "neo4j_conf_file": neo4j_TEST_config_filename,
        "results_dir": "results-dir"
        }
    
    tmplgen = TmplGenNeo4j(options)

    lst_tmplobjs = tmplgen.load_templates(options["templates_file"])
    (count_gen, count_fail) = tmplgen.generate(lst_tmplobjs, do_print=False)
    
    print(f"Generated: {count_gen}  Failed {count_fail}")
        
# test_parsing()
# test_tree_traversals()
# test_parsing_dummy()
# test_templ_gen_dummy()
# test_templ_gen_neo4j()
# test_tmplgen()