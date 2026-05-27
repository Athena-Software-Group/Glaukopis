# `tmpl_gen/data_generation/` — IFT Dataset Build Pipeline


## Author
Dr. Ionut Cardei and Mamoon Khan

## Upstream Dependency

This pipeline reads from the Neo4j graph populated by [`athena_cti_db`](../../athena_cti_db/). Stand up that database first (`athena_cti_db/utils/setup.sh`) and point `neo4j-local-config.json` at it before running any step here. The active template vintage is **v21** (`../templates/05182026/`), which feeds the v21 SFT chain targeting `Qwen2.5-32B-Instruct` — see [`../templates/05182026/README-21.md`](../templates/05182026/README-21.md) for the full build recipe (per-stage `count_limit` / `count_max` and build-dir conventions `_v21_{core,taa,cse}_build/`).

## Recommended entry point — `make_dataset.sh`

For all routine builds (including v21), use the single-script wrapper that runs steps 1–3 in sequence and accepts either `.docx` or `.json` input:

```bash
./make_dataset.sh <tmpl.docx|tmpl.json> <results_dir> <alpaca_out.json> [count_limit] [count_max]
```

The three step scripts below remain available for piecewise / debugging use.

## Run script files in this order: ##

### 1. `docx2json.sh` ###
-- converts a template source .docx file to a JSON file with just the templates.

Usage: ./docx2json.sh tmpl.docx [count_limit]

It will extract from the given source Word docx file all templates in format:
```
Id.x Instruction:... Question:... Answer:....
```
to a JSON file in the current directory with the same name as the Word file.

Example usage:

```bash
./docx2json.sh ../templates/05182026/Sophia-CTI-Templates-v21.txt
```

This will extract all templates from file
`../templates/05182026/Sophia-CTI-Templates-v21.txt` to a file `./Sophia-CTI-Templates-v21.json`.


### 2. `tmpl2triples.sh` ###
-- Uses the CTI DB to generates triples from a template JSON file.

Usage: ./tmpl2triples.sh tmpl.json results_dir [count_limit=2000]

It generates triples using the given template JSON file, placing the JSON
results file and triple files in the results_dir directory.
Optional argument count_limit sets a maximum limit for the number of triples generated from
the same template. Default is 2000.

Example usage:

```bash
./tmpl2triples.sh Sophia-CTI-Templates-v21.json results_dir 1500
```

### 3. `triples2alpaca.sh` ###
-- Converts the generated triples from a results directory to ONE JSON file in Alpaca format.

Usage: ./triples2alpaca.sh results_dir alpaca_json

Example usage:

```bash
./triples2alpaca.sh results_dir/ alpaca.json
```

NOTE: The generated Alpaca format was not tested yet with Llama Factory.
The script that does the actual conversion is `tmpl_gen/scripts/to_alpaca.py`.
It should be very easy to modify.

## Configuration Files ##

### 1. `gencfg_default_neo4j.json` ###
Triple configuration file. No need to change it yet.

### 2. `neo4j-local-config.json` ###
CTI DB connection parameters.
You must edit (or copy) this file with your CTI DB params.