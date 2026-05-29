"""Collectors for CISA advisories via sitemap extraction."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable, List, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

from ..models import UrlRecord
from .crawl_utils import canonicalize, make_session

BASE = "https://www.cisa.gov"
SITEMAP_URL = "https://www.cisa.gov/sitemaps/default/sitemap.xml"

ICS_PREFIX = "/news-events/ics-advisories/"
CSA_PREFIX = "/news-events/cybersecurity-advisories/"


def _fetch_sitemap(session: requests.Session, timeout: float) -> ET.Element:
    response = session.get(SITEMAP_URL, timeout=timeout)
    response.raise_for_status()
    return ET.fromstring(response.content)


def _collect_from_sitemap(
    *,
    prefix: str,
    source_type: str,
    timeout: float,
    limit: Optional[int] = None,
) -> List[UrlRecord]:
    session = make_session()
    root = _fetch_sitemap(session, timeout)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    host = urlparse(BASE).netloc.lower()

    records: "OrderedDict[str, UrlRecord]" = OrderedDict()

    for loc in root.findall("sm:url/sm:loc", namespace):
        url_text = loc.text.strip() if loc.text else ""
        if not url_text:
            continue
        parsed = urlparse(url_text)
        if parsed.netloc.lower() != host:
            continue
        if not parsed.path.startswith(prefix):
            continue
        canonical = canonicalize(url_text, "https")
        if canonical in records:
            continue
        metadata = {"source": "sitemap"}
        records[canonical] = UrlRecord.build(
            canonical,
            source_type=source_type,
            source_link=SITEMAP_URL,
            metadata=metadata,
        )
        if limit is not None and len(records) >= limit:
            break

    return list(records.values())


def collect_cisa_ics(
    limit: Optional[int] = None,
    *,
    timeout: float = 30.0,
) -> List[UrlRecord]:
    return _collect_from_sitemap(
        prefix=ICS_PREFIX,
        source_type="cisa_ics",
        timeout=timeout,
        limit=limit,
    )


def collect_cisa_csa(
    limit: Optional[int] = None,
    *,
    timeout: float = 30.0,
) -> List[UrlRecord]:
    return _collect_from_sitemap(
        prefix=CSA_PREFIX,
        source_type="cisa_csa",
        timeout=timeout,
        limit=limit,
    )


