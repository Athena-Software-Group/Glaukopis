"""Source registry exposing URL collectors."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..models import UrlRecord
from .cisa import collect_cisa_csa, collect_cisa_ics
from .collect_mitre_urls import collect_mitre_urls
from .cwe import collect_capec_urls, collect_cwe_urls

CollectorFn = Callable[..., List[UrlRecord]]


SOURCE_REGISTRY: Dict[str, CollectorFn] = {
    "mitre_attack": collect_mitre_urls,
    "cwe": collect_cwe_urls,
    "capec": collect_capec_urls,
    "cisa_ics": collect_cisa_ics,
    "cisa_csa": collect_cisa_csa,
}


def available_sources() -> List[str]:
    return sorted(SOURCE_REGISTRY.keys())


def get_collector(name: str) -> CollectorFn:
    if name not in SOURCE_REGISTRY:
        raise KeyError(f"Unknown source '{name}'. Known values: {', '.join(available_sources())}")
    return SOURCE_REGISTRY[name]


