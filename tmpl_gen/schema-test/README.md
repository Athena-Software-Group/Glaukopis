
# CTI DB Schema Test Using IFT Triple Generation from Sophia tmpl_gen Templates

This document explains how to generate IFT templates from a schema summary XLS file and
how to test the CTI DB schema by generating triples from a template files.

## Installation:

Run this command from directory tmpl_gen/ to install the Sophia/tmpl_gen repository + dependencies, if not done yet:
```python
pip install -e .
```

## Prerequisites:

### Neo4j Database Connection Parameters
Edit file tmpl_gen/schema-test/neo4j-local-config.json with your neo4j DB parameters.

### Triple Generation Configuration
Edit file tmpl_gen/schema-test/gencfg_default_neo4j.json as needed, but with caution.

## CTI DB Schema Test Workflow

### Input
File docs/cti-schema-table-2026-02.xlsx : summarizes the TARGET (desired) schema info in table format from docs/CTI-DB-Schema-details.docx. This file is edited by hand.

### Output
File results_test_triples/_results-report.json : triple generation results from template file. For each template (each node/property and each relationship it lists the number of generated triples:

- a positive number in case of success: e.g. "generated_count": 10

- "generated_count": 0 in case of failure. An **exception** field explains the failure reason, e.g. invalid property/relationship

- files results_test_triples/t_nnnn_node.field.json for triples with successful node properties

- files results_test_triples/t_nnnn_startnode.relationship.endnode.json for triples with successful relationships.

*** USE THE **results_test_triples/_results-report.json** FILE TO DEBUG THE CTI DB populate_neo4j CODE.  ***

The "results" list contains one report object for each test template.

A **successful generation report** looks like the following. Notice the **"generated_count": 7** field indicating 7 triples were generated. Notice there is no **exception** field marking an error.
The template_object/shortname field, the template_object/comment and the template_object/text all describe the targetted property or relationship.
The **template_object/text** field is the actual template text from the test-templates.json file.

```json
        {
            "template_index": 260,
            "template_object": {
                "shortname": "Weakness.mitigated_by>Mitigation",
                "comment": "Relationship test for Weakness.mitigated_by>Mitigation",
                "text": "Relationship test for Weakness.mitigated_by>Mitigation: {Weakness.mitigated_by>Mitigation.description}",
                "count_limit": 10
            },
            "generated_count": 7,
            "generation_time": 0.0030431747436523438,
            "query": "MATCH (_Weakness1:Weakness)-[:mitigated_by]->(_Mitigation1:Mitigation)          LIMIT 10     RETURN DISTINCT _Mitigation1.description     ORDER BY rand()"
        },

```

A **failed generation report** looks like the following. Notice the  **"generated_count": 0** field indicating that no triples were generated. Notice the **exception** field marking an error cause.
Identify the failing node/property/relationship using the template shortname or comment or text fields.

```json
        {
            "template_index": 9,
            "template_object": {
                "shortname": "attack-pattern.x_mitre_attack_spec_version",
                "comment": "Node property test for attack-pattern.x_mitre_attack_spec_version",
                "text": "Node property test for attack-pattern.x_mitre_attack_spec_version: {attack-pattern.x_mitre_atta
ck_spec_version}",
                "count_limit": 10
            },
            "exception": "Error trying to process rule \"qfield\":\n\nTmplParseTransf.map_rel ERROR: got exception: TmplParseTransf.map_rel ERROR: no property or relationship found: attack-pattern-[:x_mitre_attack_spec_version]->*  for type attack-pattern, relstr: x_mitre_attack_spec_version",
            "generated_count": 0
        },
```


### Generate Templates from Target Schema 
Script tmpl_gen/schema-test/make-test-templates.sh extracts the target schema information and generates two JSON files with templates:

```bash
./make-test-templates.sh
```

The terminal output for ths program looks like this:

```bash
(ctidb2) icardei@zapada:~/ic/projects/ASG/work/fine-tuning/template-gen/tmpl_gen/schema-test$ ./make-test-templates.sh
Namespace(xlsfile='../docs/cti-schema-target-2026-02.xlsx', output='test-templates.json')
Generate templates for 25 nodes and 51 relationships
Saved 76 test templates to file test-templates.json

Generate templates for 25 nodes, properties, and 51 relationships
Saved 290 test templates to file test-templates+props.json
```

The program creates these two files:

1.  test-templates.json : templates files with one template per node and one per relationship. This file has about 76 templates for the 02/2026 target schema. It is useful to check if all nodes and relationships from the target schema exist in the CTI BD schema. This is a smaller file and it is recommended for high level schema check for nodes and relationships only.

2.  test-templates+props.json : templates files with one template per *node property* and one per relationship. This file has about 290 templates for the 02/2026 target schema. It is useful to check if all nodes *AND properties*, and relationships from the target schema exist in the CTI BD schema. In case a node is missing from the neo4j DB, all templates referring its properties/relationships will cause errors. 


### Test CTI DB Schema: Generate Triples from Templates
Script tmpl_gen/schema-test/test-CTI-schema.sh uses program tmpl_gen/scripts/iftgen.py to generate IFT triples from one of the template test files: test-templates.json or test-templates+props.json, depending if you want to check the schema just for nodes and relationships or you want to check also for all node properties.


```bash
./schema-test/test-CTI-schema.sh 
```

The shell output from this program reads like this:
```bash
(ctidb2) icardei@zapada:~/ic/projects/ASG/work/fine-tuning/template-gen/tmpl_gen/schema-test$ ./test-CTI-schema.sh 
**** CAUTION: results directory results_test_triples will be erased and recreated
Namespace(cmd='generate', genconf='gencfg_default_neo4j.json', dbconf='neo4j-local-config.json', tmpl='test-templates+props.json', results_dir='results_test_triples', mitre_file='data/enterprise-attack.json', out='', count_max=10, verbose=0)
Neo4jDriver: Connection to Neo4j 'athena-threat-db' database successful.
Generated: 155  failed: 135
Results saved in directory results_test_triples
Neo4jDriver: Connection to Neo4j 'athena-threat-db' database closed.

===========================================================
Triple generation results in file results_test_triples/_results-report.json
Examine the results dictionary and look for entries with generated_count==0 or with exception fields.

All triples generated are in the results_test_triples directory/

```

This bash script creates the following files:
1. results_test_triples: a directory with all results files

2. results_test_triples/_results-report.json : the MAIN results file

3+. results_test_triples/t-*.json files : one file per successfully generation from node/property or relationship template



