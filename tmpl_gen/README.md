# Sophia `tmpl_gen`: Text Generation from Graphical Templates

`tmpl_gen` is a Python module and command-line tool for generating structured text from graph-based templates. It supports template expressions that reference nodes, edges, and node properties stored in a graph database such as Neo4j.

Its primary objective is to generate Instruction Fine Tuning (IFT) triples for the ASG CTI LLM.
The included demo generates text from a Neo4j instance containing the MITRE ATT&CK dataset.

The library includes:

- A template parser and text generation code.
- Functions for traversing graph structures.  
- Neo4j access helpers.  
- General-purpose helper functions for template-based text generation

The command-line tool `iftgen.py` demonstrates end-to-end text generation against a Neo4j ATT&CK demo graph.

---

## Directory Structure

```
tmpl_gen/
├── pyproject.toml
│   └── README.md
├── scripts/
│   └── iftgen.py
├── docs/
│   ├── IFTDesign.doc
│   └── IFTDesign.pdf
└── src/
    └── tmpl_gen/
        ├── __init__.py
        ├── _version.py
        ├── neo4j_utils.py
        ├── priorityQ.py
        ├── tmpl_parser.py
        └── utils.py
```

---

## Installation

### 1. Install the Package and Dependencies 

Standard installation:

```bash
pip install .
```

Editable installation (recommended during development):

```bash
pip install -e .
```

This makes the `tmpl_gen` module importable from anywhere and ensures that code changes are reflected immediately.

A running neo4j server with the proper ASG CTI database is necessary for text generation to work.

---

## Using the Text Generation Tool `iftgen.py`
The script `iftgen.py` provides a demonstration of template-based generation using a Neo4j ATT&CK dataset.

Show help:

```bash
python scripts/iftgen.py --help
```

The normal flow to demo the software is the following:
1. Create a new and empty DB on the neo4j server; this is done using Neo4JDesktop or a
separate client.
2. Edit the neo4j configuration file as needed.
3. Use `iftgen.py` to populate the new neo4j DB with the MITRE ATT&CK graph info.
4. Use `iftgen.py` to generate text from templates.


The `iftgen.py` program supoorts these functions:

- generate text from templates:
```bash
    python3 iftgen.py --cmd generate --genconf gencfg_default_neo4j.json \\
        --dbconf neo4j-TEST-config.json --tmpl sample-tmpl-attack.json \\
        --results_dir ift_dataset-dir
```
- populate test CTI neo4j DB using the current MITRE ATT&CK Enterprise source document:
```bash
    python3 iftgen.py --cmd create_db --dbconf neo4j-TEST-config.json 
```
> [!IMPORTANT]
> First, create the DB named in the configuration file on the neo4j server instance.

> [!NOTE]
> This script takes > 40 minutes long to execute, depending on hardware. 

> [!CAUTION]
> The neo4j DB will be WIPED OUT and recreated.

- extract DB schema to JSON file and generate graph Graphviz figure:
```bash
    python3 iftgen.py --cmd get_schema --dbconf neo4j-TEST-config.json --out schema.json
```

The `--out` parameter indicates the schema output file. It lists the nodes, relationships,
and properties from the neo4j DB indicated in the neo4j configuration file. A Graphviz schema.gv file
is generated with the graph figure of the DB schema.


### Neo4j Configuration File

Have a running neo4j database instance up and running.

A neo4j DB configuration file called `neo4j-TEST-config.json` is provided in the `scripts` directory:

```python
{   
    "comment": "Connection configuration to the TEST neo4j service: GUI and bolt JSON/HTTP interface",
    "uri": "bolt://localhost:7687",
    "auth": ["neo4j", "neo4jneo4j"],    
    "db_name": "test-cti2",
    "nickname": "ASG-CTI"
}
```

The `auth` list has the DB username and password strings. 
Copy and edit this file to match your setup.

Ensure the neo4j DB named by field `db_name` exists and has the necessary graph data.


### The Parsing and Generation Configuration File

The template parsing and text generation code uses configuration parameters defined in file
`gencfg_default_neo4j.json` in the `scripts` directory.
The configuration is described in the [Template Generation Design Document](docs/IFT-Design.pdf).

This configuration file must be edited to fit the needs of the user:

- define node, relationship, and property mappings based on the CTI DB schema
- define generation parameters, such as text count limit, order, time interval filtering for
incremental generation/IFT


### The Template File

File `scripts/sample-tmpl-attack.json` has several templates defined in accordance to the
[Design Document](docs/IFT-Design.pdf).
The templates are not representative of the full capabilities of the software.
Some templates were written with intentional parsing errors in order to test the system.

---

## API Usage

Examine file `scripts/iftgen.py` for an example for text generation from templates in function `task_generate`.

Import function `tool_tmplgen`:

```python
from tmpl_gen.tmpl_parser import tool_tmplgen
```

Create a dict with the main generation parameters supplied by JSON files:

```python
    options = {
            "gen_conf_file": args.genconf,    # the generation config file: gencfg_default_neo4j.json
            "templates_file": args.tmpl,      # the templates file: sample-tmpl-attack.json
            "neo4j_conf_file": args.dbconf,   # neo4j config. file: neo4j-TEST-config.json
            "results_dir": args.results_dir,  # results directory where generated text is saved: e.g. results-dir
            "verbose": args.verbose           # False, or True to see nitty gritty details
            }
        
    tool_tmplgen(options)                     # call the generation function
```

Further examination of the `tool_tmplgen` function in file `src/tmpl_gen/tmpl_parser.py` reveals
how to use the `TmplGenNeo4j` class:

```python
def tool_tmplgen(options:dict):
    """
    Called from other scripts.
    Runs a generation session from template JSON file.
    """
    tmplgen = TmplGenNeo4j(options)

    lst_tmplobjs = tmplgen.load_templates(options["templates_file"])
    (count_gen, count_fail) = tmplgen.generate(lst_tmplobjs, do_print=False)
    
    print(f"Generated: {count_gen}  Failed {count_fail}")
```

---

## Results

For a template-based generation task all results are saved to the directory specified by the
`--results_dir` command line argument.

That directory has a file `_results-report.json` with a summary for each template:
- the original template text
- Cypher query string (used for testing parsing)
- count of successfully generated texts
-    or an error message, including exception information


For each template the resulting generated strings are saved to a separate JSON file with the format
indicated in the [Design Document](docs/IFT-Design.pdf).


---
## Status

This project is in an early development stage. Template syntax, parser behavior, and API structure may evolve.

