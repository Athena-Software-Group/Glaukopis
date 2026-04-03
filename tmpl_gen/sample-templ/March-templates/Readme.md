# Directory with revised scripts as of 4/02/2026 #


## Author
Dr. Ionut Cardei


## Run cript files in this order: ##

### 1. `docx2json.sh` ###
-- converts a template source .docx file to a JSON file with just the templates.

Usage: ./docx2json.sh tmpl.docx [count_limit]

It will extract from the given source  Word docx file all templates in format:
```Id.x Instruction:... Question:... Answer:....
```
to a JSON file in the current directory with the same name as the Word file.

Example usage:

```bash
./docx2json.sh ../../March-2026/Sophia-CTI-Templates-04022026.docx
```

This will extract  all templates from file 
`../../March-2026/Sophia-CTI-Templates-04022026.docx` to a file `./Sophia-CTI-Templates-04022026.json`.


### 2. `tmpl2triples.sh` ###
-- Uses the CTI DB to generates triples from a template JSON file.

Usage: ./tmpl2triples.sh tmpl.json results_dir [count_limit=2000]

It generates triples using the given template JSON file, placing the JSON
results file and triple files in the results_dir directory.
Optional argument count_limit sets a maximum limit for the number of triples generated from
the same template. Default is 2000.

Example usage:

```bash
./tmpl2triples.sh Sophia-CTI-Templates-04022026.json results_dir 1000
```

### 3. `triples2alpaca.sh` ###
-- Converts the generated triples from a results directory to ONE JSON file in Alpaca format.

Usage: ./triples2alpaca.sh results_dir alpaca_json

Example usage:

```bash
./triples2alpaca.sh results_dir/ alpaca.json
```

NOTE: The generated Alpaca format was not tested yet with Llama Factory.
The script that does the actua conversion is `tmpl_gen/scripts/to_alpaca.py`.
It should be very easy to modify.

## Configuration Files ##

### 4. `gencfg_default_neo4j.json` ###
Triple configuration file. No need to chage it yet.

### 5. `neo4j-local-config.json` ###
CTI DB connection parameters.
You must edit (or copy) this file with your CTI DB params.