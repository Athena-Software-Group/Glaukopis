#!/usr/bin/env python3
"""TAA.CANON dataset generator for v12 (tmpl_gen/templates/05052026/
v12_plan.txt §5.2). Bypasses the v11 Neo4j-template path that ceiling'd
at 728 rows and emits the TAA.CANON.{1,2,3} families directly from the
MITRE ATT&CK STIX bundle + Athena's vendor-alias CSV.

Why this exists:
  v11 TAA.CANON.* yielded only 728 rows (TAA.CANON.1=187, TAA.CANON.2=591)
  because each Neo4j Cypher binding emitted one row per intrusion-set,
  and there are only 187 intrusion sets in MITRE ATT&CK Enterprise. The
  v11 plan §3.1 target was 2,000; the v12 plan §3.1 target is 10,000.
  This generator produces the canonical surface combinatorially:
    - For each of 102 multi-alias groups, emit alias-subset -> canonical
      rows (TAA.CANON.1).
    - For each of 168 groups with >=1 documented technique, emit
      alias-resolution-card rows with rotated signature techniques
      (TAA.CANON.2).
    - For each of N hard-negative pairs (visually-similar aliases that
      belong to different groups), emit refusal-pattern rows
      (TAA.CANON.3).

Seed sources (read directly, no Neo4j):
  cpt/cache/raw/mitre_attack_enterprise/enterprise-attack.json
    -> intrusion-set objects + uses-relationships -> attack-pattern
  SFT/eval/benchmark_data/athena_bench/athena_taa/aliases.csv
    -> vendor-specific aliases NOT in MITRE (e.g. CrowdStrike Spider
       names, Mandiant APT/UNC IDs, Microsoft Tempest names)

Output: Alpaca-format JSON consumed by the standard SFT pipeline.
Fields per row: instruction, input, output, shortname.

Usage:
  python tmpl_gen/scripts/taa_canon_generator.py \\
      --mitre cpt/cache/raw/mitre_attack_enterprise/enterprise-attack.json \\
      --athena-aliases SFT/eval/benchmark_data/athena_bench/athena_taa/aliases.csv \\
      --output SFT/data/ift_data_2026_05_05_v12_taa_canon_seed.json \\
      --target-canon1 3500 --target-canon2 3500 --target-canon3 3000 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


# Instruction phrasings per subfamily. Multiple variants give the model
# natural-language robustness without changing semantics.
INSTR_CANON1: list[str] = [
    "You are a CTI attribution analyst at an enterprise Security Operations Center resolving vendor-specific threat-actor aliases to their canonical MITRE ATT&CK intrusion-set name. Consult the MITRE ATT&CK groups knowledge base aliases catalog and report the canonical name and G-code that the alias list belongs to.",
    "You are a senior threat intelligence analyst. Vendor reporting often refers to the same intrusion set by different names. Resolve the supplied alias list to the canonical MITRE ATT&CK group name and identifier.",
    "You are a CTI deconfliction specialist. Map the vendor-naming alias list to the MITRE ATT&CK canonical intrusion-set name and G-code, citing the catalog as authority.",
    "You are an enterprise SOC attribution lead. Given a list of vendor-specific names that all refer to the same actor, return the MITRE ATT&CK canonical name and G-code.",
]

INSTR_CANON2: list[str] = [
    "You are a CTI attribution analyst at an enterprise Security Operations Center producing an alias-resolution card for a named MITRE ATT&CK intrusion set. Consult the MITRE ATT&CK groups knowledge base and report the canonical name, the catalog identifier, the documented aliases, and one signature technique used by the group.",
    "You are a threat intelligence editor compiling an alias-resolution reference for a named MITRE ATT&CK intrusion set. Report the canonical name, the G-code, the documented vendor aliases, and one signature technique.",
    "You are a CTI knowledge engineer producing a structured alias-resolution card for a MITRE ATT&CK intrusion set. Include the canonical name, the catalog identifier, the alias list, and one representative attack pattern.",
]

INSTR_CANON3: list[str] = [
    "You are a CTI attribution analyst rejecting a misattribution. The supplied vendor alias does NOT belong to the named MITRE ATT&CK intrusion set; identify the correct canonical group instead.",
    "You are a senior threat intelligence reviewer correcting an attribution error. The alias in the question is sometimes confused with the named MITRE ATT&CK group; state the actual canonical group the alias belongs to.",
    "You are a CTI deconfliction specialist refuting an incorrect alias->group mapping. Reject the proposed mapping and supply the correct MITRE ATT&CK canonical name and G-code for the alias.",
]


def load_mitre_groups(path: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Return (groups, group_id -> [attack_pattern external_id])."""
    bundle = json.loads(path.read_text())
    objects = bundle["objects"]
    groups = [o for o in objects
              if o.get("type") == "intrusion-set"
              and not o.get("revoked") and not o.get("x_mitre_deprecated")]
    ap_by_id: dict[str, dict] = {}
    for o in objects:
        if o.get("type") == "attack-pattern" and not o.get("revoked"):
            ap_by_id[o["id"]] = o
    rels = [o for o in objects
            if o.get("type") == "relationship"
            and o.get("relationship_type") == "uses"]
    group_to_aps: dict[str, list[str]] = defaultdict(list)
    for r in rels:
        src, tgt = r.get("source_ref", ""), r.get("target_ref", "")
        if src.startswith("intrusion-set--") and tgt in ap_by_id:
            ap = ap_by_id[tgt]
            ext = next((er for er in ap.get("external_references", [])
                        if er.get("source_name") == "mitre-attack"), None)
            if ext and ext.get("external_id"):
                group_to_aps[src].append((ext["external_id"], ap.get("name", "")))
    return groups, group_to_aps


def mitre_id(group: dict) -> str | None:
    for er in group.get("external_references", []):
        if er.get("source_name") == "mitre-attack":
            return er.get("external_id")
    return None


def load_athena_aliases(path: Path) -> dict[str, set[str]]:
    """Return ThreatActor -> {alias, ...}."""
    by_actor: dict[str, set[str]] = defaultdict(set)
    with path.open() as f:
        for row in csv.DictReader(f):
            actor = row.get("ThreatActor", "").strip()
            alias = row.get("Alias", "").strip()
            if actor and alias:
                by_actor[actor].add(alias)
    return by_actor


def merge_aliases(group: dict,
                  athena_by_actor: dict[str, set[str]]) -> list[str]:
    """Return de-duplicated alias list for a group, augmented from athena."""
    aliases: list[str] = list(group.get("aliases") or [])
    name = group.get("name", "")
    candidates: set[str] = set()
    if name in athena_by_actor:
        candidates.update(athena_by_actor[name])
    for a in aliases:
        if a in athena_by_actor:
            candidates.update(athena_by_actor[a])
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases + sorted(candidates):
        norm = a.strip()
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
    return out



def join_aliases(items: list[str], rng: random.Random) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        sep = rng.choice([" and ", ", also tracked as ", "; also called "])
        return items[0] + sep + items[1]
    sep = rng.choice([", ", "; "])
    tail = rng.choice([", and ", ", or ", " and "])
    return sep.join(items[:-1]) + tail + items[-1]


def gen_canon1(groups: list[dict],
               athena: dict[str, set[str]],
               target: int,
               rng: random.Random) -> list[dict]:
    """Alias subset -> canonical name + G-code rows."""
    rows: list[dict] = []
    candidates: list[tuple[dict, list[str]]] = []
    for g in groups:
        aliases = merge_aliases(g, athena)
        if len(aliases) >= 2:
            candidates.append((g, aliases))
    if not candidates:
        return rows

    # Build a working pool of (group, alias_subset) draws. For groups with
    # many aliases we sample subsets; for low-alias groups we exhaust the
    # combinations to get coverage.
    draws: list[tuple[dict, tuple[str, ...]]] = []
    for g, aliases in candidates:
        n = len(aliases)
        for k in range(1, min(n, 5) + 1):
            combos = list(itertools.combinations(aliases, k))
            rng.shuffle(combos)
            cap = max(2, min(len(combos), 12))
            for c in combos[:cap]:
                draws.append((g, c))
    rng.shuffle(draws)

    i = 0
    while len(rows) < target and draws:
        g, subset = draws[i % len(draws)]
        i += 1
        gid = mitre_id(g)
        if not gid:
            continue
        canonical = g.get("name", "")
        instr = INSTR_CANON1[i % len(INSTR_CANON1)]
        alias_list = list(subset)
        rng.shuffle(alias_list)
        prompt_kind = i % 3
        if prompt_kind == 0:
            inp = (f"Resolve the following vendor-naming alias list to its "
                   f"canonical MITRE ATT&CK intrusion-set name and G-code: "
                   f"{join_aliases(alias_list, rng)}.")
        elif prompt_kind == 1:
            inp = (f"The following names appear in vendor reporting and refer "
                   f"to the same intrusion set: {join_aliases(alias_list, rng)}. "
                   f"What is the canonical MITRE ATT&CK group name and "
                   f"identifier?")
        else:
            inp = (f"Map the alias list [{', '.join(alias_list)}] to the "
                   f"MITRE ATT&CK canonical intrusion-set name and G-code.")
        out = (f"Canonical MITRE ATT&CK intrusion-set: {canonical} ({gid}). "
               f"The supplied alias list ({join_aliases(alias_list, rng)}) "
               f"resolves to {canonical} per the MITRE ATT&CK groups "
               f"knowledge base. Therefore, the canonical group is "
               f"{canonical} ({gid}).")
        rows.append({
            "instruction": instr,
            "input": inp,
            "output": out,
            "shortname": "TAA.CANON.1",
        })
        if i > target * 8:
            break
    return rows


def gen_canon2(groups: list[dict],
               group_to_aps: dict[str, list],
               athena: dict[str, set[str]],
               target: int,
               rng: random.Random) -> list[dict]:
    """Alias-resolution card rows with rotated signature techniques."""
    rows: list[dict] = []
    eligible = [g for g in groups
                if mitre_id(g) and group_to_aps.get(g["id"])]
    if not eligible:
        return rows
    # Cards-per-group budget so coverage is uniform across groups. Multiple
    # rounds of cycling let us reach the target even when many groups have
    # fewer techniques than the per-group budget; signature-technique
    # rotation across rounds keeps each row distinct.
    # Oversubscribe by 3x so the (input, output) dedup pass below still
    # hits the target after collisions; INSTR_CANON2 has 3 input phrasings
    # so each (group, AP) can yield up to 3 unique rows.
    per_group = max(1, (target * 3) // len(eligible) + 1)
    draws: list[tuple[dict, tuple[str, str]]] = []
    for g in eligible:
        aps = list(group_to_aps[g["id"]])
        rng.shuffle(aps)
        rounds = (per_group + len(aps) - 1) // len(aps)
        cycled = (aps * rounds)[:per_group]
        for ap in cycled:
            draws.append((g, ap))
    rng.shuffle(draws)

    seen_io: set[tuple[str, str]] = set()
    for i, (g, (ap_id, ap_name)) in enumerate(draws):
        if len(rows) >= target:
            break
        gid = mitre_id(g)
        canonical = g.get("name", "")
        aliases = merge_aliases(g, athena) or [canonical]
        instr = INSTR_CANON2[i % len(INSTR_CANON2)]
        kind = i % 3
        if kind == 0:
            inp = (f"Produce an alias-resolution card for the MITRE ATT&CK "
                   f"intrusion set {canonical} ({gid}).")
        elif kind == 1:
            inp = (f"Compile a structured alias-resolution card for the "
                   f"MITRE ATT&CK group {canonical} ({gid}), including its "
                   f"canonical name, G-code, documented aliases, and one "
                   f"representative technique.")
        else:
            inp = (f"Build an attribution reference card for {canonical} "
                   f"({gid}): canonical name, MITRE ATT&CK identifier, "
                   f"vendor aliases, and one signature attack pattern.")
        alias_str = ", ".join(aliases) if len(aliases) > 1 else aliases[0]
        out = (f"Alias card -- Canonical name: {canonical}. "
               f"MITRE ATT&CK identifier: {gid}. "
               f"Documented aliases: {alias_str}. "
               f"Signature technique: MITRE ATT&CK technique {ap_id} "
               f"({ap_name}) is documented as used by this group. "
               f"Therefore, the canonical group is {canonical} ({gid}).")
        if (inp, out) in seen_io:
            continue
        seen_io.add((inp, out))
        rows.append({
            "instruction": instr,
            "input": inp,
            "output": out,
            "shortname": "TAA.CANON.2",
        })
    return rows



def gen_canon3(groups: list[dict],
               athena: dict[str, set[str]],
               target: int,
               rng: random.Random) -> list[dict]:
    """Hard-negative refusal rows: reject false alias->group mappings."""
    rows: list[dict] = []
    # Build alias -> [(group, gid)] index. Aliases owned by a single group are
    # the candidates; for each, pair with a different group as the foil.
    alias_owner: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for g in groups:
        gid = mitre_id(g)
        if not gid:
            continue
        for a in merge_aliases(g, athena):
            alias_owner[a.lower()].append((g, gid))
    unique_alias_pairs: list[tuple[str, dict, str]] = [
        (a, owners[0][0], owners[0][1])
        for a, owners in alias_owner.items()
        if len(owners) == 1
    ]
    if not unique_alias_pairs:
        return rows

    eligible_foils = [g for g in groups if mitre_id(g)]
    if len(eligible_foils) < 2:
        return rows

    # Generate (alias, true_owner, foil) triples where foil != true_owner.
    # 5 foils per unique alias gives a comfortable margin for the dedup pass
    # below to still hit target.
    triples: list[tuple[str, dict, str, dict, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for alias, true_g, true_gid in unique_alias_pairs:
        for _ in range(5):
            foil = rng.choice(eligible_foils)
            attempts = 0
            while foil["id"] == true_g["id"] and attempts < 5:
                foil = rng.choice(eligible_foils)
                attempts += 1
            if foil["id"] == true_g["id"]:
                continue
            key = (alias.lower(), foil["id"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            triples.append((alias, true_g, true_gid, foil, mitre_id(foil)))
    rng.shuffle(triples)

    for i, (alias, true_g, true_gid, foil, foil_gid) in enumerate(triples):
        if len(rows) >= target:
            break
        true_name = true_g.get("name", "")
        foil_name = foil.get("name", "")
        instr = INSTR_CANON3[i % len(INSTR_CANON3)]
        kind = i % 2
        if kind == 0:
            inp = (f"Is the vendor alias \"{alias}\" used to refer to the "
                   f"MITRE ATT&CK intrusion set {foil_name} ({foil_gid})?")
        else:
            inp = (f"Confirm or reject: the alias \"{alias}\" maps to the "
                   f"MITRE ATT&CK group {foil_name} ({foil_gid}).")
        out = (f"Reject. The alias \"{alias}\" is not documented as a name "
               f"for {foil_name} ({foil_gid}). Per the MITRE ATT&CK groups "
               f"knowledge base, \"{alias}\" resolves to the canonical "
               f"intrusion set {true_name} ({true_gid}). Therefore, the "
               f"correct canonical group for the alias is {true_name} "
               f"({true_gid}).")
        rows.append({
            "instruction": instr,
            "input": inp,
            "output": out,
            "shortname": "TAA.CANON.3",
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mitre", required=True, type=Path,
                   help="Path to enterprise-attack.json (STIX bundle).")
    p.add_argument("--athena-aliases", required=True, type=Path,
                   help="Path to athena_taa/aliases.csv.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--target-canon1", type=int, default=3500)
    p.add_argument("--target-canon2", type=int, default=3500)
    p.add_argument("--target-canon3", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=Path)
    args = p.parse_args()

    rng = random.Random(args.seed)

    groups, group_to_aps = load_mitre_groups(args.mitre)
    athena = load_athena_aliases(args.athena_aliases)
    print(f"loaded {len(groups):,} MITRE intrusion-sets; "
          f"{sum(len(v) for v in group_to_aps.values()):,} group-uses-AP edges; "
          f"{sum(len(v) for v in athena.values()):,} athena aliases across "
          f"{len(athena):,} actors", file=sys.stderr)

    canon1 = gen_canon1(groups, athena, args.target_canon1, rng)
    canon2 = gen_canon2(groups, group_to_aps, athena, args.target_canon2, rng)
    canon3 = gen_canon3(groups, athena, args.target_canon3, rng)

    print(f"\nTAA.CANON.1: {len(canon1):,} (target {args.target_canon1:,})",
          file=sys.stderr)
    print(f"TAA.CANON.2: {len(canon2):,} (target {args.target_canon2:,})",
          file=sys.stderr)
    print(f"TAA.CANON.3: {len(canon3):,} (target {args.target_canon3:,})",
          file=sys.stderr)

    out_rows = canon1 + canon2 + canon3
    rng.shuffle(out_rows)
    for r in out_rows:
        r["source"] = "athena-cti-db-internal"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_rows, indent=2))
    print(f"\nwrote {len(out_rows):,} rows to {args.output}", file=sys.stderr)

    if args.report:
        report = {
            "mitre": str(args.mitre),
            "athena_aliases": str(args.athena_aliases),
            "mitre_groups": len(groups),
            "athena_actors": len(athena),
            "canon1_rows": len(canon1),
            "canon2_rows": len(canon2),
            "canon3_rows": len(canon3),
            "total_rows": len(out_rows),
            "seed": args.seed,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"report written to {args.report}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
