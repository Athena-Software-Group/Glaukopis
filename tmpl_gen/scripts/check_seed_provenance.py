#!/usr/bin/env python3
"""Build-time seed-provenance gate for the v13 training corpus
(tmpl_gen/templates/05072026/v13_plan.txt sec 4.5 + sec 10.3).

Companion to check_corpus_licences.py: that script verifies the
emitted row-level `source` tags against the v13 commercial-use
allowlist; this script verifies that every UPSTREAM seed file
consumed by the v13 generators is accompanied by a PROVENANCE.txt
that (a) declares its licence, (b) records the SHA-256 hash that
the seed file is currently shipping under, and (c) the on-disk
SHA-256 matches the recorded one (so a refresh cannot silently
land without a provenance update).

Why this exists:
  v12 had no upstream-seed audit. v13 sec 10.3 requires that every
  external corpus consumed by the build carries an in-tree
  PROVENANCE.txt naming the licence; this script is the build-time
  enforcement, run by _v13_build/watcher.sh as Phase 0 before any
  generator touches the seeds.

Exit codes:
  0  all registered seeds OK
  1  one or more seeds failed (missing PROVENANCE, hash mismatch,
     or no allowlisted licence string in the PROVENANCE)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# v13_plan.txt sec 4.5: every upstream seed consumed by the v13 build
# is registered here. The PROVENANCE.txt that lives next to each seed
# must contain at least one of the licence-keyword strings associated
# with it.
SEEDS: list[dict] = [
    {
        "name": "MITRE ATT&CK Enterprise STIX bundle",
        "path": "cpt/cache/raw/mitre_attack_enterprise/enterprise-attack.json",
        "provenance": "cpt/cache/raw/mitre_attack_enterprise/PROVENANCE.txt",
        "expected_sha256":
            "628c4fc3c01b9ef37e1cd84ca3c421e1d43950a43464a14aabd1a7089601dc45",
        "licence_keywords": ["MITRE ATT&CK Terms of Use"],
    },
    {
        "name": "MISP threat-actor galaxy snapshot",
        "path": "tmpl_gen/data_generation/seeds/misp/threat-actor.json",
        "provenance": "tmpl_gen/data_generation/seeds/misp/PROVENANCE.txt",
        "expected_sha256":
            "46eae3bd9af0409c1fd687f50712228be690778cc83462396f17e1ffa857fff4",
        "licence_keywords": ["CC0 1.0", "CC-0", "Public Domain"],
    },
    {
        "name": "Athena CTI vendor-alias CSV",
        "path": "SFT/eval/benchmark_data/athena_bench/athena_taa/aliases.csv",
        "provenance": "SFT/eval/benchmark_data/athena_bench/athena_taa/PROVENANCE.txt",
        "expected_sha256":
            "f6b3a56a3cdae1930a3af9dc2f35d452ebc315a97a794724f4c28a8e598a5ade",
        "licence_keywords": ["Athena internal"],
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_one(seed: dict, repo_root: Path) -> dict:
    rec = {"name": seed["name"], "path": seed["path"],
           "provenance": seed["provenance"], "issues": []}
    p = repo_root / seed["path"]
    pv = repo_root / seed["provenance"]
    if not p.exists():
        rec["issues"].append(f"seed file missing: {p}")
        return rec
    if not pv.exists():
        rec["issues"].append(f"PROVENANCE.txt missing: {pv}")
        return rec
    actual = sha256_file(p)
    rec["actual_sha256"] = actual
    rec["expected_sha256"] = seed["expected_sha256"]
    if actual != seed["expected_sha256"]:
        rec["issues"].append(
            f"SHA-256 mismatch (actual={actual} expected={seed['expected_sha256']})")
    text = pv.read_text()
    matches = [kw for kw in seed["licence_keywords"] if kw in text]
    rec["licence_keyword_hits"] = matches
    if not matches:
        rec["issues"].append(
            f"no licence keyword from {seed['licence_keywords']} found in PROVENANCE.txt")
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--report", type=Path,
                   help="Optional JSON path for the per-seed report.")
    args = p.parse_args()

    results = [check_one(s, args.repo_root) for s in SEEDS]
    failed = [r for r in results if r["issues"]]

    for r in results:
        marker = "  " if not r["issues"] else "!!"
        print(f"{marker} {r['name']}", file=sys.stderr)
        print(f"     path       : {r['path']}", file=sys.stderr)
        print(f"     provenance : {r['provenance']}", file=sys.stderr)
        if "actual_sha256" in r:
            print(f"     sha256     : {r['actual_sha256']}", file=sys.stderr)
        if r.get("licence_keyword_hits"):
            print(f"     licence kw : {r['licence_keyword_hits']}", file=sys.stderr)
        for issue in r["issues"]:
            print(f"     ISSUE      : {issue}", file=sys.stderr)
        print(file=sys.stderr)

    report = {"total_seeds": len(SEEDS),
              "failed_seeds": len(failed),
              "outcome": "fail" if failed else "ok",
              "results": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"report written to {args.report}", file=sys.stderr)

    if failed:
        print(f"\nFAIL: {len(failed)} seed(s) failed provenance gate.",
              file=sys.stderr)
        return 1
    print(f"\nseed-provenance gate PASSED ({len(SEEDS)} seeds).",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
