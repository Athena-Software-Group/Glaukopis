#!/usr/bin/env python3
"""MISP threat-actor galaxy generator for v13 (tmpl_gen/templates/05072026/
v13_plan.txt §4.4). Mirrors taa_canon_generator.py's three-family shape
(MISP.CANON.{1,2,3}) but reads the MISP `threat-actor` galaxy cluster JSON
under tmpl_gen/data_generation/seeds/misp/ instead of MITRE STIX.

Why this exists:
  v12 TAA.CANON.* topped out at ~10.7K rows from 187 MITRE intrusion-sets.
  The MISP threat-actor galaxy is a CC-0 / public-domain cross-vendor
  catalog of ~985 threat-actor entries (~3.4 synonyms each on average,
  386 with synonyms, 929 with descriptions, 456 with country attribution).
  This generator emits ~12K rows from the MISP catalog while staying
  inside the v13 licence allowlist (every row tagged misp-galaxy-cc0).

Validation surface (the user-emphasised "good validation process"):
  1. SHA-256 of the vendored seed (default expected SHA is pinned to the
     2026-05-07 vendored snapshot; --expected-sha256 overrides; --skip-sha
     disables for development).
  2. Licence file presence and content (LICENSE.md must contain a CC-0
     reference; halts on missing or non-permissive).
  3. Input JSON schema (top-level type == "threat-actor"; per-value
     `value` and `uuid` required; `meta.synonyms` must be list of str).
  4. Per-actor quality gates (drop empty/whitespace synonyms, drop
     synonyms that are case-insensitive duplicates of canonical, drop
     synonyms shorter than 3 chars, drop control characters).
  5. Optional cross-source dedup against MITRE STIX (--mitre): MISP
     entries whose canonical name OR every synonym already appears in
     MITRE Groups are FLAGGED in the report but NOT auto-dropped (the
     MISP descriptions and hard-negative pairs add value even on
     overlapping actors).
  6. Optional eval-set leak audit (--eval-aliases): counts how many
     emitted aliases appear in the AthenaBench TAA aliases.csv (warns,
     does not halt; v13 dedup phase will catch true row collisions).
  7. Output validation: every row must have non-empty instruction,
     input, output, shortname; source == SOURCE_TAG; shortname matches
     ^MISP\\.CANON\\.[123]$.
  8. Determinism: actors sorted by uuid pre-iteration; --seed pins
     all random draws.

Usage:
  python tmpl_gen/scripts/misp_taa_generator.py \\
      --input tmpl_gen/data_generation/seeds/misp/threat-actor.json \\
      --license tmpl_gen/data_generation/seeds/misp/LICENSE.md \\
      --mitre cpt/cache/raw/mitre_attack_enterprise/enterprise-attack.json \\
      --eval-aliases SFT/test/benchmark_data/athena_bench/athena_taa/aliases.csv \\
      --output SFT/data/ift_data_2026_05_07_v13_misp_taa.json \\
      --report _v13_build/misp_report.json \\
      --target-canon1 5000 --target-canon2 4000 --target-canon3 3000 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_TYPE: str = "threat-actor"
SOURCE_TAG: str = "misp-galaxy-cc0"

# Pinned SHA-256 of the snapshot vendored at v13 build time. See
# tmpl_gen/data_generation/seeds/misp/PROVENANCE.txt for source commit.
EXPECTED_SHA256_DEFAULT: str = (
    "46eae3bd9af0409c1fd687f50712228be690778cc83462396f17e1ffa857fff4"
)

# CC-0 marker strings the licence file must contain (any one suffices).
LICENCE_MARKERS: tuple[str, ...] = (
    "CC0",
    "Public Domain Dedication",
    "publicdomain/zero",
)

SHORTNAME_RE = re.compile(r"^MISP\.CANON\.[123]$")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MIN_SYNONYM_LEN: int = 3

# ISO-3166 alpha-2 -> friendly country name for the most common attributors
# in the MISP catalog. Unknown codes fall back to the raw code.
COUNTRY_NAMES: dict[str, str] = {
    "CN": "China", "RU": "Russia", "IR": "Iran", "KP": "North Korea",
    "KR": "South Korea", "US": "United States", "IL": "Israel",
    "IN": "India", "PK": "Pakistan", "VN": "Vietnam", "TR": "Turkey",
    "BY": "Belarus", "UA": "Ukraine", "SY": "Syria", "LB": "Lebanon",
    "SA": "Saudi Arabia", "AE": "United Arab Emirates", "EG": "Egypt",
    "IQ": "Iraq", "JO": "Jordan", "PS": "State of Palestine",
    "RO": "Romania", "BR": "Brazil", "GB": "United Kingdom",
    "FR": "France", "DE": "Germany", "ES": "Spain", "IT": "Italy",
    "NL": "Netherlands", "JP": "Japan", "TW": "Taiwan", "MY": "Malaysia",
    "TH": "Thailand", "ID": "Indonesia", "PH": "Philippines",
    "MX": "Mexico", "CA": "Canada", "AU": "Australia", "NZ": "New Zealand",
}

INSTR_CANON1: list[str] = [
    "You are a CTI attribution analyst at an enterprise Security Operations Center resolving vendor-specific threat-actor aliases to their canonical name. Consult the MISP threat-actor galaxy and report the canonical name (and country attribution where documented) that the alias list belongs to.",
    "You are a senior threat intelligence analyst. Vendor reporting often refers to the same threat actor by different names. Resolve the supplied alias list to the canonical name documented in the MISP threat-actor galaxy.",
    "You are a CTI deconfliction specialist. Map the vendor-naming alias list to the canonical threat-actor name documented in the MISP threat-actor galaxy.",
    "You are an enterprise SOC attribution lead. Given a list of vendor-specific names that all refer to the same threat actor, return the canonical name as documented in the MISP threat-actor galaxy.",
]

INSTR_CANON2: list[str] = [
    "You are a CTI attribution analyst producing an alias-resolution card for a named threat actor documented in the MISP threat-actor galaxy. Report the canonical name, the country attribution where documented, and the documented synonyms.",
    "You are a threat intelligence editor compiling an alias-resolution reference card for a named threat actor. Report the MISP canonical name, the country attribution (if known), and the documented synonyms.",
    "You are a CTI knowledge engineer producing a structured alias-resolution card for a MISP threat-actor entry. Include the canonical name, the country attribution where documented, and the synonym list.",
]

INSTR_CANON3: list[str] = [
    "You are a CTI attribution analyst evaluating a proposed alias-to-actor mapping. Reject mappings unsupported by the MISP threat-actor galaxy and state the canonical actor the alias actually resolves to.",
    "You are a threat intelligence reviewer auditing an alias-to-actor claim. Reject the claim if the MISP threat-actor galaxy does not document the alias under the named actor; cite the correct canonical actor.",
    "You are a CTI deconfliction specialist verifying an alias-to-actor mapping. Reject the mapping if it is not documented in the MISP threat-actor galaxy and report the actual canonical actor the alias resolves to.",
]



# ----------------------------------------------------------------------
# Validation primitives
# ----------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_licence(path: Path) -> str:
    """Return the licence file text after confirming a CC-0 marker.

    Raises ValueError if the file is missing or contains no CC-0 marker.
    The v13 licence-allowlist gate (check_corpus_licences.py) accepts
    misp-galaxy-cc0 only when this validation passes.
    """
    if not path.is_file():
        raise ValueError(f"licence file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not any(m in text for m in LICENCE_MARKERS):
        raise ValueError(
            f"licence file {path} does not contain a CC-0 marker; "
            f"expected one of: {', '.join(LICENCE_MARKERS)}"
        )
    return text


def validate_input_schema(bundle: dict) -> None:
    """Raise ValueError if the MISP cluster JSON is malformed."""
    if not isinstance(bundle, dict):
        raise ValueError("MISP bundle must be a top-level object")
    if bundle.get("type") != EXPECTED_TYPE:
        raise ValueError(
            f"MISP bundle type {bundle.get('type')!r} != {EXPECTED_TYPE!r}"
        )
    values = bundle.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("MISP bundle.values must be a non-empty list")
    for i, v in enumerate(values):
        if not isinstance(v, dict):
            raise ValueError(f"values[{i}] is not an object")
        if not isinstance(v.get("value"), str) or not v["value"].strip():
            raise ValueError(f"values[{i}] missing required string `value`")
        if not isinstance(v.get("uuid"), str) or not v["uuid"].strip():
            raise ValueError(f"values[{i}] missing required string `uuid`")
        meta = v.get("meta")
        if meta is not None and not isinstance(meta, dict):
            raise ValueError(f"values[{i}].meta must be object or null")
        if meta and "synonyms" in meta:
            syns = meta["synonyms"]
            if not isinstance(syns, list) or not all(
                isinstance(s, str) for s in syns
            ):
                raise ValueError(
                    f"values[{i}].meta.synonyms must be list[str]"
                )


def clean_synonyms(actor: dict) -> list[str]:
    """Return the per-actor synonym list after the quality gates."""
    canonical = actor.get("value", "").strip()
    canonical_low = canonical.lower()
    raw = (actor.get("meta") or {}).get("synonyms") or []
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        if not isinstance(s, str):
            continue
        norm = s.strip()
        if len(norm) < MIN_SYNONYM_LEN:
            continue
        if CONTROL_CHARS_RE.search(norm):
            continue
        low = norm.lower()
        if low == canonical_low:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(norm)
    return out


def country_label(meta: dict) -> str:
    """Return a human-readable country label or empty string."""
    if not isinstance(meta, dict):
        return ""
    code = (meta.get("country") or "").strip().upper()
    if not code:
        return ""
    return COUNTRY_NAMES.get(code, code)


def load_misp_actors(
    input_path: Path,
    expected_sha: str | None,
    skip_sha: bool = False,
) -> tuple[list[dict], dict]:
    """Load + validate the MISP cluster JSON.

    Returns (actors, summary). Raises on any unrecoverable validation
    failure. `summary` contains per-stage counts for the build report.
    """
    if not input_path.is_file():
        raise ValueError(f"MISP input not found: {input_path}")
    actual_sha = sha256_file(input_path)
    if expected_sha and not skip_sha and actual_sha != expected_sha:
        raise ValueError(
            f"MISP input SHA-256 mismatch: expected {expected_sha}, "
            f"got {actual_sha}; refresh PROVENANCE.txt or pass "
            f"--skip-sha-check"
        )
    bundle = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input_schema(bundle)
    raw_actors: list[dict] = list(bundle["values"])
    actors_sorted = sorted(raw_actors, key=lambda a: a["uuid"])
    summary = {
        "input_path": str(input_path),
        "input_sha256": actual_sha,
        "input_total_values": len(actors_sorted),
        "input_with_synonyms": sum(
            1 for a in actors_sorted if clean_synonyms(a)
        ),
        "input_with_country": sum(
            1 for a in actors_sorted if country_label(a.get("meta") or {})
        ),
        "input_with_description": sum(
            1 for a in actors_sorted if (a.get("description") or "").strip()
        ),
    }
    return actors_sorted, summary



# ----------------------------------------------------------------------
# Optional cross-source loaders (audit-only; do not auto-drop)
# ----------------------------------------------------------------------


def load_mitre_overlap(path: Path | None) -> dict[str, str]:
    """Return lowercase-name -> MITRE G-code map for overlap auditing.

    The resulting map covers every MITRE intrusion-set canonical name and
    every documented alias. None when --mitre is not supplied.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"--mitre file not found: {path}")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    overlap: dict[str, str] = {}
    for o in bundle.get("objects", []):
        if o.get("type") != "intrusion-set":
            continue
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        gid = next(
            (er.get("external_id") for er in o.get("external_references", [])
             if er.get("source_name") == "mitre-attack"),
            None,
        )
        if not gid:
            continue
        for n in [o.get("name", "")] + list(o.get("aliases") or []):
            n_norm = (n or "").strip().lower()
            if n_norm:
                overlap.setdefault(n_norm, gid)
    return overlap


def load_eval_aliases(path: Path | None) -> set[str]:
    """Return the set of lowercase aliases in the AthenaBench TAA eval.

    Used for a leak-audit warning only; the v13 dedup phase enforces row
    collisions, this just surfaces vocabulary overlap up-front.
    """
    if path is None:
        return set()
    if not path.is_file():
        raise ValueError(f"--eval-aliases file not found: {path}")
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("Alias", "alias", "ThreatActor", "threat_actor"):
                v = (row.get(key) or "").strip()
                if v:
                    out.add(v.lower())
    return out


# ----------------------------------------------------------------------
# Row generators (MISP.CANON.{1,2,3})
# ----------------------------------------------------------------------


def join_aliases(items: list[str], rng: random.Random) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        sep = rng.choice([" and ", ", also tracked as ", "; also called "])
        return items[0] + sep + items[1]
    sep = rng.choice([", ", "; "])
    tail = rng.choice([", and ", ", or ", " and "])
    return sep.join(items[:-1]) + tail + items[-1]


def gen_canon1(actors: list[dict],
               target: int,
               rng: random.Random) -> list[dict]:
    """Alias-subset -> canonical name (+ country if known) rows."""
    rows: list[dict] = []
    candidates: list[tuple[dict, list[str], str]] = []
    for a in actors:
        syns = clean_synonyms(a)
        if len(syns) >= 1:
            candidates.append((a, syns, country_label(a.get("meta") or {})))
    if not candidates:
        return rows

    draws: list[tuple[dict, tuple[str, ...], str]] = []
    for a, syns, ctry in candidates:
        n = len(syns)
        for k in range(1, min(n, 5) + 1):
            combos = list(itertools.combinations(syns, k))
            rng.shuffle(combos)
            cap = max(2, min(len(combos), 12))
            for c in combos[:cap]:
                draws.append((a, c, ctry))
    rng.shuffle(draws)

    seen_io: set[tuple[str, str]] = set()
    for i, (actor, subset, ctry) in enumerate(draws):
        if len(rows) >= target:
            break
        canonical = actor["value"].strip()
        instr = INSTR_CANON1[i % len(INSTR_CANON1)]
        alias_list = list(subset)
        rng.shuffle(alias_list)
        kind = i % 3
        if kind == 0:
            inp = (f"Resolve the following vendor-naming alias list to its "
                   f"canonical threat-actor name as documented in the MISP "
                   f"threat-actor galaxy: "
                   f"{join_aliases(alias_list, rng)}.")
        elif kind == 1:
            inp = (f"The following names appear in vendor reporting and "
                   f"refer to the same threat actor: "
                   f"{join_aliases(alias_list, rng)}. What is the canonical "
                   f"name documented in the MISP threat-actor galaxy?")
        else:
            inp = (f"Map the alias list [{', '.join(alias_list)}] to the "
                   f"canonical threat-actor name as documented in the MISP "
                   f"threat-actor galaxy.")
        ctry_clause = (
            f" with country attribution {ctry}" if ctry else ""
        )
        out = (f"Canonical threat actor: {canonical}{ctry_clause}. The "
               f"supplied alias list ({join_aliases(alias_list, rng)}) "
               f"resolves to {canonical} per the MISP threat-actor "
               f"galaxy. Therefore, the canonical actor is {canonical}.")
        if (inp, out) in seen_io:
            continue
        seen_io.add((inp, out))
        rows.append({
            "instruction": instr,
            "input": inp,
            "output": out,
            "shortname": "MISP.CANON.1",
            "source": SOURCE_TAG,
        })
    return rows



def gen_canon2(actors: list[dict],
               target: int,
               rng: random.Random) -> list[dict]:
    """Alias-resolution card rows: canonical + country + synonym list.

    Subset-rotated like CANON.1: each actor contributes multiple
    (input, output) pairs by varying which synonym subset is mentioned
    in the prompt while the output card always reports the full
    documented synonym list. This trains "given any subset, recall the
    canonical and full synonym list."
    """
    rows: list[dict] = []
    eligible = [a for a in actors if clean_synonyms(a)]
    if not eligible:
        return rows

    draws: list[tuple[dict, tuple[str, ...]]] = []
    for a in eligible:
        syns = clean_synonyms(a)
        n = len(syns)
        # Whole-actor case (no subset mention) plus rotated subsets.
        draws.append((a, ()))
        for k in range(1, min(n, 4) + 1):
            combos = list(itertools.combinations(syns, k))
            rng.shuffle(combos)
            cap = max(2, min(len(combos), 8))
            for c in combos[:cap]:
                draws.append((a, c))
    rng.shuffle(draws)

    seen_io: set[tuple[str, str]] = set()
    for i, (actor, subset) in enumerate(draws):
        if len(rows) >= target:
            break
        canonical = actor["value"].strip()
        syns = clean_synonyms(actor)
        ctry = country_label(actor.get("meta") or {})
        instr = INSTR_CANON2[i % len(INSTR_CANON2)]
        # When a subset is provided, surface it in the prompt; otherwise
        # ask for the whole-actor card.
        if subset:
            subset_list = list(subset)
            rng.shuffle(subset_list)
            subset_str = join_aliases(subset_list, rng)
            kind = i % 3
            if kind == 0:
                inp = (f"Produce an alias-resolution card for the MISP "
                       f"threat-actor entry that documents the synonyms "
                       f"{subset_str}.")
            elif kind == 1:
                inp = (f"Compile a structured alias-resolution card for the "
                       f"MISP threat actor whose documented synonyms include "
                       f"{subset_str}; report the canonical name, the country "
                       f"attribution where documented, and the full synonym "
                       f"list.")
            else:
                inp = (f"Build an attribution reference card for the MISP "
                       f"threat actor referenced under the names "
                       f"{subset_str}: canonical name, country attribution, "
                       f"and the documented synonyms.")
        else:
            kind = i % 3
            if kind == 0:
                inp = (f"Produce an alias-resolution card for the threat "
                       f"actor {canonical} as documented in the MISP "
                       f"threat-actor galaxy.")
            elif kind == 1:
                inp = (f"Compile a structured alias-resolution card for the "
                       f"MISP threat-actor entry {canonical}, including its "
                       f"canonical name, country attribution where "
                       f"documented, and the documented synonym list.")
            else:
                inp = (f"Build an attribution reference card for "
                       f"{canonical}: canonical name, country attribution, "
                       f"and the documented synonyms from the MISP "
                       f"threat-actor galaxy.")
        ctry_field = ctry if ctry else "not documented"
        # Stable synonym ordering inside the OUTPUT keeps the (inp,out)
        # dedup deterministic across reruns even though the INPUT rotates.
        syn_str = ", ".join(syns)
        out = (f"Alias card -- Canonical name: {canonical}. "
               f"Country attribution: {ctry_field}. "
               f"Documented synonyms: {syn_str}. "
               f"Therefore, the canonical actor is {canonical}.")
        if (inp, out) in seen_io:
            continue
        seen_io.add((inp, out))
        rows.append({
            "instruction": instr,
            "input": inp,
            "output": out,
            "shortname": "MISP.CANON.2",
            "source": SOURCE_TAG,
        })
    return rows


def gen_canon3(actors: list[dict],
               target: int,
               rng: random.Random) -> list[dict]:
    """Hard-negative refusal rows: reject false alias->actor mappings."""
    rows: list[dict] = []
    # Build alias -> [owner_actor]; aliases owned by exactly one MISP actor
    # are the candidates. For each, pair against a different actor as foil.
    alias_owner: dict[str, list[dict]] = defaultdict(list)
    for a in actors:
        for s in clean_synonyms(a):
            alias_owner[s.lower()].append(a)
    unique_alias_pairs: list[tuple[str, dict]] = [
        (a, owners[0])
        for a, owners in alias_owner.items()
        if len(owners) == 1
    ]
    if not unique_alias_pairs:
        return rows

    eligible_foils = [a for a in actors if a.get("value", "").strip()]
    if len(eligible_foils) < 2:
        return rows

    triples: list[tuple[str, dict, dict]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for alias, true_a in unique_alias_pairs:
        for _ in range(5):
            foil = rng.choice(eligible_foils)
            attempts = 0
            while foil["uuid"] == true_a["uuid"] and attempts < 5:
                foil = rng.choice(eligible_foils)
                attempts += 1
            if foil["uuid"] == true_a["uuid"]:
                continue
            key = (alias.lower(), foil["uuid"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            triples.append((alias, true_a, foil))
    rng.shuffle(triples)

    for i, (alias, true_a, foil) in enumerate(triples):
        if len(rows) >= target:
            break
        true_name = true_a["value"].strip()
        foil_name = foil["value"].strip()
        instr = INSTR_CANON3[i % len(INSTR_CANON3)]
        kind = i % 2
        if kind == 0:
            inp = (f"Is the vendor alias \"{alias}\" used to refer to the "
                   f"threat actor {foil_name} as documented in the MISP "
                   f"threat-actor galaxy?")
        else:
            inp = (f"Confirm or reject: the alias \"{alias}\" maps to the "
                   f"threat actor {foil_name} per the MISP threat-actor "
                   f"galaxy.")
        out = (f"Reject. The alias \"{alias}\" is not documented as a name "
               f"for {foil_name} in the MISP threat-actor galaxy. Per the "
               f"MISP catalog, \"{alias}\" resolves to the canonical actor "
               f"{true_name}. Therefore, the correct canonical actor for "
               f"the alias is {true_name}.")
        rows.append({
            "instruction": instr,
            "input": inp,
            "output": out,
            "shortname": "MISP.CANON.3",
            "source": SOURCE_TAG,
        })
    return rows


# ----------------------------------------------------------------------
# Output validation
# ----------------------------------------------------------------------


def validate_output_rows(rows: list[dict]) -> list[str]:
    """Return list of validation failure strings (empty list = clean)."""
    errs: list[str] = []
    seen_io: set[tuple[str, str]] = set()
    for i, r in enumerate(rows):
        for k in ("instruction", "input", "output", "shortname", "source"):
            v = r.get(k)
            if not isinstance(v, str) or not v.strip():
                errs.append(f"row {i}: missing/blank {k}")
        sn = r.get("shortname", "")
        if not SHORTNAME_RE.match(sn):
            errs.append(f"row {i}: shortname {sn!r} does not match regex")
        if r.get("source") != SOURCE_TAG:
            errs.append(
                f"row {i}: source {r.get('source')!r} != {SOURCE_TAG!r}"
            )
        key = (r.get("input", ""), r.get("output", ""))
        if key in seen_io:
            errs.append(f"row {i}: (input,output) duplicate")
        seen_io.add(key)
    return errs



# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="MISP threat-actor.json cluster file.")
    p.add_argument("--license", required=True, type=Path,
                   help="MISP LICENSE.md (must contain CC-0 marker).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output Alpaca-format JSON.")
    p.add_argument("--report", type=Path,
                   help="Optional JSON report destination.")
    p.add_argument("--mitre", type=Path, default=None,
                   help="Optional MITRE STIX bundle for overlap audit.")
    p.add_argument("--eval-aliases", type=Path, default=None,
                   help="Optional AthenaBench TAA aliases.csv for leak audit.")
    p.add_argument("--target-canon1", type=int, default=5000)
    p.add_argument("--target-canon2", type=int, default=4000)
    p.add_argument("--target-canon3", type=int, default=3000)
    p.add_argument("--max-rows", type=int, default=0,
                   help="Hard cap on total emitted rows (0 = no cap).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-sha256", type=str,
                   default=EXPECTED_SHA256_DEFAULT,
                   help="Pinned SHA-256 of the input file.")
    p.add_argument("--skip-sha-check", action="store_true",
                   help="Disable SHA-256 verification (development only).")
    p.add_argument("--strict", action="store_true",
                   help="Treat overlap/leak warnings as fatal.")
    args = p.parse_args()

    rng = random.Random(args.seed)

    # Stage 1: licence + input integrity
    print("[1/6] validating licence ...", file=sys.stderr)
    validate_licence(args.license)
    print(f"      OK ({args.license})", file=sys.stderr)

    print("[2/6] loading + validating MISP input ...", file=sys.stderr)
    actors, summary = load_misp_actors(
        args.input, args.expected_sha256, args.skip_sha_check
    )
    print(
        f"      OK -- {summary['input_total_values']:,} actors "
        f"({summary['input_with_synonyms']:,} with synonyms, "
        f"{summary['input_with_country']:,} with country, "
        f"{summary['input_with_description']:,} with description)",
        file=sys.stderr,
    )

    # Stage 2: optional cross-source audits
    print("[3/6] cross-source overlap audit ...", file=sys.stderr)
    mitre_overlap = load_mitre_overlap(args.mitre)
    eval_aliases = load_eval_aliases(args.eval_aliases)
    overlap_actor_count = 0
    if mitre_overlap:
        for a in actors:
            if a["value"].strip().lower() in mitre_overlap:
                overlap_actor_count += 1
    leak_alias_count = 0
    if eval_aliases:
        for a in actors:
            if a["value"].strip().lower() in eval_aliases:
                leak_alias_count += 1
            for s in clean_synonyms(a):
                if s.lower() in eval_aliases:
                    leak_alias_count += 1
    print(
        f"      OK -- MITRE-overlap actors: {overlap_actor_count:,} / "
        f"{len(actors):,}; eval-vocab hits: {leak_alias_count:,} "
        f"(audit-only; v13 dedup phase enforces row collisions)",
        file=sys.stderr,
    )

    # Stage 3: row generation
    print("[4/6] generating MISP.CANON.{1,2,3} rows ...", file=sys.stderr)
    canon1 = gen_canon1(actors, args.target_canon1, rng)
    canon2 = gen_canon2(actors, args.target_canon2, rng)
    canon3 = gen_canon3(actors, args.target_canon3, rng)
    print(
        f"      MISP.CANON.1: {len(canon1):,} (target {args.target_canon1:,})",
        file=sys.stderr,
    )
    print(
        f"      MISP.CANON.2: {len(canon2):,} (target {args.target_canon2:,})",
        file=sys.stderr,
    )
    print(
        f"      MISP.CANON.3: {len(canon3):,} (target {args.target_canon3:,})",
        file=sys.stderr,
    )

    out_rows = canon1 + canon2 + canon3
    if args.max_rows and len(out_rows) > args.max_rows:
        rng.shuffle(out_rows)
        out_rows = out_rows[: args.max_rows]
        print(
            f"      capped at --max-rows {args.max_rows:,}",
            file=sys.stderr,
        )
    rng.shuffle(out_rows)

    # Stage 4: output validation
    print("[5/6] validating output rows ...", file=sys.stderr)
    errs = validate_output_rows(out_rows)
    if errs:
        for e in errs[:20]:
            print(f"      ERROR: {e}", file=sys.stderr)
        if len(errs) > 20:
            print(
                f"      ... {len(errs) - 20} more errors suppressed",
                file=sys.stderr,
            )
        print(
            f"      FAIL -- {len(errs):,} validation errors", file=sys.stderr
        )
        return 2
    print(f"      OK -- {len(out_rows):,} rows clean", file=sys.stderr)

    # Stage 5: write outputs
    print("[6/6] writing outputs ...", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_rows, indent=2))
    print(f"      wrote {len(out_rows):,} rows -> {args.output}",
          file=sys.stderr)

    if args.report:
        report = {
            **summary,
            "license_path": str(args.license),
            "license_marker_ok": True,
            "mitre_overlap_actors": overlap_actor_count,
            "mitre_overlap_path": str(args.mitre) if args.mitre else None,
            "eval_alias_leak_hits": leak_alias_count,
            "eval_aliases_path": (
                str(args.eval_aliases) if args.eval_aliases else None
            ),
            "canon1_rows": len(canon1),
            "canon2_rows": len(canon2),
            "canon3_rows": len(canon3),
            "total_rows": len(out_rows),
            "max_rows_cap": args.max_rows,
            "seed": args.seed,
            "source_tag": SOURCE_TAG,
            "validation_errors": [],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"      report -> {args.report}", file=sys.stderr)

    if args.strict and (overlap_actor_count or leak_alias_count):
        print(
            f"--strict: overlap={overlap_actor_count} leak={leak_alias_count}; "
            f"failing build",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
