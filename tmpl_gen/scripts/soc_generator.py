#!/usr/bin/env python3
"""SOC.*.GEN.* dataset generator for v12 (tmpl_gen/templates/05052026/
v12_plan.txt §5.1 SOC recovery + §11 saturation guardrail). Complements
the template-driven SOC.IR.* / SOC.MAL.* / SOC.SIGMA.* / SOC.TRIAGE.*
families which bind to SigmaHQ-rule and malware-family Cypher seeds
that saturate at ~5K rows in v12. The generator supplies the structural
and procedural SOC knowledge those seeds do not exercise.

Families:
  SOC.SIGMA.GEN.1  ~1,500  Sigma YAML schema, logsource taxonomy,
                            detection-field semantics, modifiers,
                            condition expressions, severity levels
  SOC.MAL.GEN.1    ~1,500  ~45 malware families with category /
                            primary capability / characteristic ATT&CK
                            technique / target platform set
  SOC.IR.GEN.1     ~1,000  NIST SP 800-61 phase x scenario actions
                            (ransomware, BEC, ATO, exfil, web shell,
                            lateral movement, DDoS), 800-86 volatility
                            order, common forensic artefacts
  SOC.TRIAGE.GEN.1 ~1,000  alert severity rubric, TP/FP signals,
                            enrichment fields, escalation matrix

Knowledge tables encoded in tmpl_gen/scripts/soc_data/{sigma_rules,
malware,ir_playbook,triage}.py as structured Python literals -- no
external data, no model-authored text. Each (fact, paraphrase) is
reproducible from --seed; option order is shuffled per row.

Usage:
  python tmpl_gen/scripts/soc_generator.py \\
      --output _v12_build/soc_seed.json \\
      --report _v12_build/soc_report.json \\
      --target-sigma 1500 --target-mal 1500 \\
      --target-ir 1000 --target-triage 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Allow `from soc_data.* import ...` regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from soc_data import ir_playbook as soc_ir          # noqa: E402
from soc_data import malware as soc_mal             # noqa: E402
from soc_data import sigma_rules as soc_sigma       # noqa: E402
from soc_data import triage as soc_triage           # noqa: E402

# Verbatim instruction strings. Each family uses an SOC-analyst persona
# matching the template-driven SOC.* prompts so the trained model treats
# generator + template rows interchangeably.
INSTR_SIGMA = (
    "You are a detection engineer at an enterprise Security Operations "
    "Center answering a question about Sigma-rule structure and "
    "semantics. Reason about each option in turn against the SigmaHQ "
    "specification, and then select the single best answer from A, B, "
    "C, D, or E."
)
INSTR_MAL = (
    "You are a malware analyst at an enterprise Security Operations "
    "Center answering a question about a known malware family. Reason "
    "about each option in turn against MITRE ATT&CK Software entries and "
    "vendor reporting, and then select the single best answer from A, B, "
    "C, D, or E."
)
INSTR_IR = (
    "You are an incident responder at an enterprise Security Operations "
    "Center answering an incident-response procedure question. Reason "
    "about each option in turn against NIST SP 800-61 Rev 2 and SP "
    "800-86, and then select the single best answer from A, B, C, D, "
    "or E."
)
INSTR_TRIAGE = (
    "You are a Tier-1 SOC analyst at an enterprise Security Operations "
    "Center answering an alert-triage question. Reason about each option "
    "in turn against the standard triage rubric (severity, true/false "
    "positive signals, enrichment context, and escalation criteria), "
    "and then select the single best answer from A, B, C, D, or E."
)


def make_mcq(rng: random.Random, question: str, correct: str,
             distractors: list[str], rationale: str,
             shortname: str, instruction: str) -> dict:
    """Build a single Alpaca MCQ row with shuffled A-E options.

    Identical contract to cm_generator.make_mcq and mcq_generator.make_mcq
    so the trained model sees a uniform MCQ surface across CM.*, MCQ.EXT.*,
    SOC.*.GEN.*, and templates.
    """
    pool = [correct] + [d for d in distractors if d != correct][:4]
    fillers = ["None of the above", "All of the above",
               "Both A and B", "Both B and C", "Cannot be determined"]
    fi = 0
    while len(pool) < 5:
        if fillers[fi] not in pool:
            pool.append(fillers[fi])
        fi += 1
    rng.shuffle(pool)
    letter = "ABCDE"[pool.index(correct)]
    options = "\n".join(f"{l}) {o}" for l, o in zip("ABCDE", pool))
    return {
        "instruction": instruction,
        "input": f"{question}\n{options}",
        "output": f"{rationale} Therefore, {letter}.",
        "shortname": shortname,
    }


FAMILIES = [
    ("SOC.SIGMA.GEN.1",  INSTR_SIGMA,  soc_sigma.generate,  "target_sigma"),
    ("SOC.MAL.GEN.1",    INSTR_MAL,    soc_mal.generate,    "target_mal"),
    ("SOC.IR.GEN.1",     INSTR_IR,     soc_ir.generate,     "target_ir"),
    ("SOC.TRIAGE.GEN.1", INSTR_TRIAGE, soc_triage.generate, "target_triage"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--target-sigma", type=int, default=1500)
    p.add_argument("--target-mal", type=int, default=1500)
    p.add_argument("--target-ir", type=int, default=1000)
    p.add_argument("--target-triage", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for shortname, instr, gen, tgt_attr in FAMILIES:
        target = getattr(args, tgt_attr)
        sub = gen(rng, target, instr, shortname, make_mcq)
        rows.extend(sub)
        counts[shortname] = len(sub)
        print(f"[soc_gen] {shortname:20s} target={target:>5d} "
              f"actual={len(sub):>5d}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    if args.report:
        letters = Counter(r["output"].rstrip(".").rsplit(", ", 1)[-1]
                          for r in rows)
        args.report.write_text(json.dumps({
            "total_rows": len(rows),
            "per_family": counts,
            "answer_letter_distribution": dict(letters),
            "seed": args.seed,
        }, indent=2))
    print(f"[soc_gen] wrote {len(rows):,} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
