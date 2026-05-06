#!/usr/bin/env python3
"""CM.* dataset generator for v12 (tmpl_gen/templates/05052026/
v12_plan.txt §5.2). Bypasses the Neo4j-template path because the
underlying knowledge (cryptography fundamentals, access-control models,
compliance frameworks, governance/risk methodology) lives outside the
CTI graph. Emits ~6,000 deterministic Alpaca rows across four families:

  CM.CRYPTO.1     ~1,500  algorithms, modes, key sizes, attacks, standards
  CM.ACCESS.1     ~1,500  DAC/MAC/RBAC/ABAC, BLP/Biba/CW/BN, OAuth/OIDC,
                          Kerberos, SAML, JWT, MFA/SSO
  CM.COMPLIANCE.1 ~2,000  NIST CSF 2.0, ISO 27001:2022, HIPAA, PCI-DSS v4,
                          NIST SP 800-53, GDPR, SOC 2
  CM.GOV.1        ~1,000  risk (ALE/ARO/SLE), frameworks (COBIT/ITIL/COSO/
                          ISO 31000/NIST RMF), policy hierarchy, roles,
                          three lines of defence, BIA/RTO/RPO

Knowledge tables are encoded in tmpl_gen/scripts/cm_data/{crypto,access,
compliance,governance}.py as structured Python literals -- no external
data fetched, no model-authored text consumed. Each (fact, pattern) pair
is reproducible from --seed; option order is shuffled per row.

Usage:
  python tmpl_gen/scripts/cm_generator.py \\
      --output _v12_build/cm_seed.json \\
      --report _v12_build/cm_report.json \\
      --target-crypto 1500 --target-access 1500 \\
      --target-compliance 2000 --target-gov 1000 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Allow `from cm_data.* import ...` regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cm_data import access as cm_access  # noqa: E402
from cm_data import compliance as cm_compliance  # noqa: E402
from cm_data import crypto as cm_crypto  # noqa: E402
from cm_data import governance as cm_governance  # noqa: E402

# Verbatim instruction strings from the v12 manifest. Mirrors what
# docx2json would have emitted had these templates been Cypher-backed,
# so the model sees the same prompt prefix at training and inference.
INSTR_CRYPTO = (
    "You are a cybersecurity analyst at an enterprise Security Operations "
    "Center answering a cryptography fundamentals multiple-choice question. "
    "Reason about each option in turn against the relevant cryptographic "
    "primitive, protocol, or attack pattern, and then select the single "
    "best answer from A, B, C, D, or E."
)
INSTR_ACCESS = (
    "You are a security architect at an enterprise Security Operations "
    "Center answering an access-control models multiple-choice question. "
    "Reason about each option in turn against the relevant access-control "
    "model (DAC, MAC, RBAC, ABAC), authentication mechanism, or "
    "authorisation framework, and then select the single best answer "
    "from A, B, C, D, or E."
)
INSTR_COMPLIANCE = (
    "You are a compliance analyst at an enterprise Security Operations "
    "Center answering a security-compliance multiple-choice question "
    "covering NIST CSF, ISO 27001, HIPAA, and PCI-DSS. Reason about each "
    "option in turn against the cited standard's controls or requirements, "
    "and then select the single best answer from A, B, C, D, or E."
)
INSTR_GOV = (
    "You are a security governance specialist at an enterprise Security "
    "Operations Center answering a governance and risk-management "
    "multiple-choice question. Reason about each option in turn against "
    "the relevant governance framework, risk-management methodology, or "
    "organisational control, and then select the single best answer from "
    "A, B, C, D, or E."
)


def make_mcq(rng: random.Random, question: str, correct: str,
             distractors: list[str], rationale: str,
             shortname: str, instruction: str) -> dict:
    """Build a single Alpaca MCQ row with shuffled A-E options.

    Output format mirrors v11 AB.MCQ.* rows: rationale text terminated
    by 'Therefore, X.' so the existing benchmark post-processor (last
    A-E match per line) extracts the answer letter unchanged.
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
    ("CM.CRYPTO.1",     INSTR_CRYPTO,     cm_crypto.generate,     "target_crypto"),
    ("CM.ACCESS.1",     INSTR_ACCESS,     cm_access.generate,     "target_access"),
    ("CM.COMPLIANCE.1", INSTR_COMPLIANCE, cm_compliance.generate, "target_compliance"),
    ("CM.GOV.1",        INSTR_GOV,        cm_governance.generate, "target_gov"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--target-crypto", type=int, default=1500)
    p.add_argument("--target-access", type=int, default=1500)
    p.add_argument("--target-compliance", type=int, default=2000)
    p.add_argument("--target-gov", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for shortname, instr, gen, tgt_attr in FAMILIES:
        target = getattr(args, tgt_attr.replace("target_", "target-").replace("-", "_"))
        sub = gen(rng, target, instr, shortname, make_mcq)
        rows.extend(sub)
        counts[shortname] = len(sub)
        print(f"[cm_gen] {shortname:20s} target={target:>5d} "
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
    print(f"[cm_gen] wrote {len(rows):,} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
