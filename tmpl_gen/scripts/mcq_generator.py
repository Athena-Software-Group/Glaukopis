#!/usr/bin/env python3
"""MCQ.EXT.* dataset generator for v12 (tmpl_gen/templates/05052026/
v12_plan.txt §5.1 MCQ recovery + saturation guardrail). Complements
the template-driven AB.MCQ.* / JS.MCQ.* families which are bound to
the MITRE ATT&CK Cypher graph and saturate at ~3K rows of distinct
(anchor, correct-answer) tuples in v12. The generator covers the
relationship edges those templates do NOT traverse, plus general
application/network/OS security topics outside the MITRE ontology
and outside the CM crypto/access/compliance/governance scope.

Families:
  AB.MCQ.EXT.MITRE.1    ~1,500  technique<->platform, sub-tech<->parent,
                                software<->category, defense-evaded
                                control, kill-chain phase mapping
  AB.MCQ.EXT.SEC.1      ~1,500  OWASP Top-10 2021, OWASP API Top-10 2023,
                                web vuln classes (SQLi/XSS/CSRF/SSRF/...),
                                network protocols, OS/endpoint primitives

Knowledge tables are encoded in tmpl_gen/scripts/mcq_data/{mitre_extras,
security_concepts}.py as structured Python literals -- no external data,
no model-authored text. Each (fact, paraphrase) is reproducible from
--seed; option order is shuffled per row.

Usage:
  python tmpl_gen/scripts/mcq_generator.py \\
      --output _v12_build/mcq_seed.json \\
      --report _v12_build/mcq_report.json \\
      --target-mitre 1500 --target-sec 1500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Allow `from mcq_data.* import ...` regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcq_data import mitre_extras as mcq_mitre        # noqa: E402
from mcq_data import security_concepts as mcq_sec     # noqa: E402

# Verbatim instruction strings. The MITRE-extras family mirrors the
# AB.MCQ.* prompt so the model treats generator + template rows
# interchangeably; the security-concepts family uses an
# application-security flavour that overlaps neither CM nor templates.
INSTR_MITRE = (
    "You are a cybersecurity analyst at an enterprise Security Operations "
    "Center answering a Cyber Threat Intelligence multiple-choice question. "
    "Consult MITRE ATT&CK, reason about each option in turn, and then select "
    "the single best answer from A, B, C, D, or E."
)
INSTR_SEC = (
    "You are an application-security analyst at an enterprise Security "
    "Operations Center answering a security-fundamentals multiple-choice "
    "question covering OWASP, web vulnerabilities, network protocols, and "
    "OS/endpoint hardening. Reason about each option in turn against the "
    "relevant standard or primitive, and then select the single best "
    "answer from A, B, C, D, or E."
)


def make_mcq(rng: random.Random, question: str, correct: str,
             distractors: list[str], rationale: str,
             shortname: str, instruction: str) -> dict:
    """Build a single Alpaca MCQ row with shuffled A-E options.

    Identical contract to cm_generator.make_mcq so the trained model
    sees a uniform MCQ surface across CM.*, MCQ.EXT.*, and templates.
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
    ("AB.MCQ.EXT.MITRE.1", INSTR_MITRE, mcq_mitre.generate, "target_mitre"),
    ("AB.MCQ.EXT.SEC.1",   INSTR_SEC,   mcq_sec.generate,   "target_sec"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--target-mitre", type=int, default=1500)
    p.add_argument("--target-sec", type=int, default=1500)
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
        print(f"[mcq_gen] {shortname:25s} target={target:>5d} "
              f"actual={len(sub):>5d}")

    for r in rows:
        r["source"] = "athena-cti-db-internal"
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
    print(f"[mcq_gen] wrote {len(rows):,} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
