# Sophia CTI Template Vintages

This directory holds the historical succession of Sophia CTI template sets used to build the Athena IFT corpus. Vintages are organised by date-stamped subdirectories (e.g. `05182026/` for the v21 build); the root-level `Sophia-CTI-Templates-03222026*.txt` files are the original hand-crafted and schema-aligned starting points the vintages were derived from.

The current shipping vintage is **v21** under [`05182026/`](05182026/) — a strict-reproducibility fork of v18.1 that also carries the Qwen2.5-32B port and the 32B-tuned Stage-4 Recalibrate recipe (`v21-recal-32b`, Total 65.0 / Weighted 62.9). See [`05182026/README-21.md`](05182026/README-21.md) for the full build recipe (per-stage `count_limit` / `count_max`, byte-identical inputs vs. v18.1, the off-plan Recalibrate stage that ships, and the Qwen3-30B-A3B-Thinking-2507 MoE port).

The expanded set is grouped by template IDs (`M`, `A`, `W`, `V`, `S`, `P`, `E`, `X`) and designed to align with the IFT syntax and graph traversal behavior implemented in this repository.

## Category, Source, and Design Strategy

| Template Set | Subject Category | Underlying Source(s) | Design Strategy |
|---|---|---|---|
| `M.*` | ATT&CK core technique, tactic, campaign, intrusion-set, malware/tool usage | MITRE ATT&CK (`attack-pattern`, `x-mitre-tactic`, `campaign`, `intrusion-set`, `malware`, `tool`, `course-of-action`, detection strategy/analytic/data component nodes) | Build strong ATT&CK-only baselines first: describe technique behavior, tactic mapping, mitigations, actor/campaign context, and detection pipeline (`detects` -> `implemented_by` -> `requires_data`). |
| `A.*` | CAPEC attack patterns and their mappings | CAPEC (`CAPEC`) with bridges to CWE (`Weakness`), CVE (`CVE`), and ATT&CK (`attack-pattern`) | Emphasize attack-pattern semantics (execution, severity, prerequisites), then add taxonomy alignment via `exploits` and `map_ap`; include CAPEC hierarchy relations (`ChildOf`, `PeerOf`, `CanPrecede`). |
| `W.*` | CWE weakness analysis and defensive handling | CWE (`Weakness`) with `Detection_Method`, `Mitigation`, `CVE`, `KEV`, `CAPEC` | Use weakness-centric reasoning: root cause, introduction phase, detection, mitigation, observed CVEs, KEV linkage, and CAPEC/ATT&CK tie-in through `related_attack_pattern`. |
| `V.*` | CVE vulnerability interpretation and triage | CVE (`CVE`) with CWE (`problemType`), CAPEC (`impacts`), KEV (`known_exploit`), EPSS (`scores`) | Start from CVE details (description and CVSS fields), then progressively correlate to exploit likelihood and downstream defensive context (CWE/CAPEC/ATT&CK and KEV urgency). |
| `S.*` | KEV operational remediation and compliance urgency | CISA KEV (`KEV`) with linked CVE/CWE/CAPEC/ATT&CK/EPSS context | Keep KEV action-oriented: patch urgency, due-date pressure, ransomware indicator, required actions; enrich KEV entries by traversing back into CVE/CWE and forward into ATT&CK mappings. |
| `P.*` | EPSS exploit likelihood prioritization | FIRST EPSS (`EPSS`) with linked CVE, KEV, CWE, CAPEC, ATT&CK | Treat EPSS as likelihood signal, not impact score: pair EPSS with CVSS/KEV and graph context to produce actionable risk ranking templates instead of EPSS-only scoring prompts. |
| `E.*` | MITRE ENGAGE active defense model | MITRE ENGAGE-style entities (`activity`, `approach`, `goal`, `attack_technique`, `attack_tactic`, `vulnerability`) with ATT&CK mapping (`maps_ap`, `maps_ack`) | Capture defender-driven planning logic: activity-goal-approach chains, tactic/technique mapping, and alignment with ATT&CK semantics for operations and deception workflows. |
| `X.*` | Multi-source advanced correlation | Cross-source joins across ATT&CK, CAPEC, CWE, CVE, KEV, EPSS, ENGAGE | Build end-to-end correlation chains and add hard consistency constraints (`{force ...}`) to enforce same-entity joins (e.g., CVE->CWE and CAPEC->CWE equivalence) for richer training patterns. |

## Syntax and Modeling Conventions Used

- Template format follows IFT triples with `Instruction:`, `Question:`, and `Answer:`.
- Graph placeholders follow parser syntax from `tmpl_gen/src/tmpl_gen/tmpl_parser.py`, including:
  - Node/property access: `{var:NodeType.property}`
  - Relationship traversal: `{var1.rel>TargetType.property}`
  - Inverse traversal: `{var1.rel<SourceType.property}`
  - Constraints: `{force left.path=right.path}`
  - Invisible sections: `<* ... *>`
- Property/type alias behavior follows generation config mappings (for example `id` mappings for ATT&CK-related nodes).

## Repository Artifacts Used to Design the Expanded Set

- `tmpl_gen/docs/IFT-Design.pdf`
- `tmpl_gen/src/tmpl_gen/tmpl_parser.py`
- `tmpl_gen/schema-test/test-templates.json`
- `tmpl_gen/schema-test/test-templates+props.json`
- `tmpl_gen/schema-test/gencfg_default_neo4j.json`

These files were used to keep category coverage, node/relationship naming, and template syntax consistent with the current repo implementation.

## Pinned Benchmark Notes

`MASTER_RESULTS.md` (this directory) holds the most recent AthenaBench / CyberSOCEval / CyberMetric / MMLU-Pro numbers across the v7..v21 vintages and the authoritative architecture summary. Read it first when restoring context. The most recent ship checkpoint (`v21-recal-32b`, Qwen2.5-32B-Instruct) and the 32B-tuned Stage-4 recipe rationale are documented in [`05182026/README-21.md`](05182026/README-21.md).
