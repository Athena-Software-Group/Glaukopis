"""Post-processing registry for MCQ task preparation."""
from __future__ import annotations

from typing import Callable, Dict, Optional

from .cisa import parse_cisa
from .cwe_capec import parse_capec, parse_cwe
from .mitre import parse_mitre_attack

Parser = Callable[[str], Optional[dict]]


PARSER_REGISTRY: Dict[str, Parser] = {
    "mitre_attack": parse_mitre_attack,
    "cwe_catalog": parse_cwe,
    "capec_catalog": parse_capec,
    "cisa_ics": parse_cisa,
    "cisa_csa": parse_cisa,
}


def get_parser(source_type: str) -> Optional[Parser]:
    return PARSER_REGISTRY.get(source_type)


