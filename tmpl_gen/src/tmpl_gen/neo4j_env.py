#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j credentials from the environment or a local .env file.

The connection configs (neo4j-local-config.json) hold only the URI and the
database name. The user name and password come from NEO4J_USER and
NEO4J_PASSWORD, read from the process environment or, when unset there, from
a .env file in the working directory or at the repository root. .env is
gitignored; .env.example at the repository root lists the variables.

Dependency-free so that standalone build scripts can import it without the
tmpl_gen package (put this directory on sys.path and `import neo4j_env`).
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_dotenv(path=None):
    """Export KEY=VALUE lines from a .env file; variables already set win.
    Returns the file read, or None if there is none."""
    if path is None:
        for candidate in (Path.cwd() / ".env", _REPO_ROOT / ".env"):
            if candidate.is_file():
                path = candidate
                break
        else:
            return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        os.environ.setdefault(key, val)
    return path


def neo4j_auth(cfg_dct=None):
    """(user, password) for GraphDatabase.driver().

    Environment (or .env) first; a legacy "auth" pair in the connection config
    is the fallback so older local configs keep working."""
    load_dotenv()
    password = os.environ.get("NEO4J_PASSWORD")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    if password is None and cfg_dct and cfg_dct.get("auth"):
        cfg_user, password = cfg_dct["auth"]
        user = user or cfg_user
    if password is None:
        raise RuntimeError(
            "Neo4j credentials not set: define NEO4J_USER and NEO4J_PASSWORD in the "
            f"environment or in {_REPO_ROOT / '.env'} (see .env.example).")
    return (user or "neo4j", password)
