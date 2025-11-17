#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 19:40:33 2025

@author: icardei
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tmpl_gen")
except PackageNotFoundError:
    __version__ = "0.1.0"
    