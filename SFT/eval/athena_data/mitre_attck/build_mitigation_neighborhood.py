"""Build the RMS mitigation neighborhood artifact (Option A).

For each MITRE ATT&CK technique T, the artifact records:
  - strict     : mitigations directly bound to T via "mitigates"
  - plausible  : mitigations bound to T's parent, siblings, or children
                 (sub-technique relations) but not to T itself
The evaluator uses this to award credit for predicted mitigations that
address a neighbouring technique even when they are not in the gold set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import requests

from pipelines.data_loader import load_yaml

MITRE_SRC_NAMES = ("mitre-attack", "mitre-mobile-attack", "mitre-ics-attack")


def ensure_attack_bundle(url: str, dest: Path) -> Path:
    logger.info("Downloading ATT&CK bundle from %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    dest.write_bytes(r.content)
    logger.info("Bundle saved to %s", dest)
    return dest


def external_id(obj: Dict[str, Any]) -> str:
    for ref in obj.get("external_references", []) or []:
        if ref.get("source_name") in MITRE_SRC_NAMES and ref.get("external_id"):
            return ref["external_id"]
    return ""

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "athena_data/config.yaml"
DEFAULT_OUTPUT = "benchmark_data/athena_bench/athena_rms/mitigation_neighborhood.json"


def _build_indexes(objs: List[Dict[str, Any]]):
    attack_patterns: Dict[str, Dict[str, Any]] = {}
    mitigations: Dict[str, Dict[str, Any]] = {}
    for o in objs:
        t = o.get("type")
        if t not in ("attack-pattern", "course-of-action"):
            continue
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        eid = external_id(o)
        if not eid:
            continue
        if t == "attack-pattern" and eid.startswith("T"):
            attack_patterns[o["id"]] = o
        elif t == "course-of-action" and eid.startswith("M"):
            mitigations[o["id"]] = o

    tech_to_mits: Dict[str, Set[str]] = {}
    parent_of: Dict[str, str] = {}
    children_of: Dict[str, Set[str]] = {}

    for o in objs:
        if o.get("type") != "relationship":
            continue
        rt = o.get("relationship_type")
        src = o.get("source_ref")
        tgt = o.get("target_ref")
        if rt == "mitigates" and src in mitigations and tgt in attack_patterns:
            mid = external_id(mitigations[src])
            tid = external_id(attack_patterns[tgt])
            if mid and tid:
                tech_to_mits.setdefault(tid, set()).add(mid)
        elif rt == "subtechnique-of" and src in attack_patterns and tgt in attack_patterns:
            child_id = external_id(attack_patterns[src])
            parent_id = external_id(attack_patterns[tgt])
            if child_id and parent_id:
                parent_of[child_id] = parent_id
                children_of.setdefault(parent_id, set()).add(child_id)

    return attack_patterns, tech_to_mits, parent_of, children_of


def _neighbors(tid: str, parent_of: Dict[str, str], children_of: Dict[str, Set[str]]) -> Set[str]:
    neigh: Set[str] = set()
    parent = parent_of.get(tid)
    if parent:
        neigh.add(parent)
        neigh.update(children_of.get(parent, set()))
    neigh.update(children_of.get(tid, set()))
    neigh.discard(tid)
    return neigh


def build_neighborhood(bundle_path: Path) -> Dict[str, Dict[str, List[str]]]:
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objs = data.get("objects", [])
    attack_patterns, tech_to_mits, parent_of, children_of = _build_indexes(objs)

    out: Dict[str, Dict[str, List[str]]] = {}
    for ap in attack_patterns.values():
        tid = external_id(ap)
        if not tid:
            continue
        strict = tech_to_mits.get(tid, set())
        plaus: Set[str] = set()
        for n_tid in _neighbors(tid, parent_of, children_of):
            plaus.update(tech_to_mits.get(n_tid, set()))
        plaus -= strict
        out[tid] = {
            "strict": sorted(strict),
            "plausible": sorted(plaus),
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--attack-bundle",
        default=None,
        help="Path to enterprise-attack.json. If omitted, downloaded per config.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    mitre_cfg = cfg.get("MITRE_ATTCK", {})

    if args.attack_bundle:
        bundle_path = Path(args.attack_bundle)
        if not bundle_path.exists():
            raise FileNotFoundError(bundle_path)
    else:
        bundle_path = Path(mitre_cfg.get("cache_path"))
        if not bundle_path.exists():
            ensure_attack_bundle(mitre_cfg["attack_url"], bundle_path)

    logger.info("Loading bundle %s", bundle_path)
    neigh = build_neighborhood(bundle_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(neigh, indent=2, sort_keys=True), encoding="utf-8")

    n_strict = sum(1 for v in neigh.values() if v["strict"])
    n_plaus = sum(1 for v in neigh.values() if v["plausible"])
    logger.info(
        "Wrote %d techniques (%d with strict mits, %d with plausible neighbours) -> %s",
        len(neigh), n_strict, n_plaus, out_path,
    )


if __name__ == "__main__":
    main()
