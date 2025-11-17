#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-level package for the Template-based dataset generation from graph DB.

Created on Fri Nov 14 19:17:31 2025

@author: icardei
"""

from .tmpl_parser import TmplParser, tool_tmplgen
from .neo4j_utils import neo4j_extract_schema, create_ATTACK_db

from ._version import __version__

# exported identifiers:
__all__ = ["__version__", "TmplParser", "tool_tmplgen", "neo4j_extract_schema", 
           "create_ATTACK_db"
           ]
