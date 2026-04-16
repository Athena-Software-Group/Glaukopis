"""Collectors for CWE authoritative sources."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import bs4
import requests

from ..models import UrlRecord
from .crawl_utils import allowed_by_robots, canonicalize, fetch_robots, make_session

BASE = "https://cwe.mitre.org"
SEED_PAGES = [
    "https://cwe.mitre.org/data/definitions/1000.html",
    "https://cwe.mitre.org/data/definitions/699.html",
    "https://cwe.mitre.org/data/definitions/1194.html",
]


def _extract_definition_links(html: str, page_url: str) -> Iterator[Tuple[str, str]]:
    soup = bs4.BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or "data/definitions/" not in href or not href.endswith(".html"):
            continue
        full_url = canonicalize(urljoin(page_url, href), "https")
        label = anchor.get_text(strip=True)
        yield full_url, label


def _add_record(
    records: "OrderedDict[str, UrlRecord]",
    url: str,
    *,
    label: Optional[str],
    source_page: str,
    disallows,
) -> None:
    path = urlparse(url).path or "/"
    if not allowed_by_robots(path, disallows):
        return
    if url in records:
        existing = records[url]
        if label and label and "label" not in existing.metadata:
            existing.metadata["label"] = label
        existing.metadata.setdefault("source_pages", [])
        if source_page not in existing.metadata["source_pages"]:
            existing.metadata["source_pages"].append(source_page)
        return
    metadata: dict = {}
    if label:
        metadata["label"] = label
    metadata["source_pages"] = [source_page]
    records[url] = UrlRecord.build(
        url,
        source_type="cwe_catalog",
        source_link=BASE,
        metadata=metadata,
    )


def collect_cwe_urls(
    limit: Optional[int] = None,
    *,
    timeout: float = 30.0,
) -> List[UrlRecord]:
    session = make_session("athena-ctibench/cwe-collector (+https://github.com/athena-cti/cti-bench)")
    disallows = fetch_robots(BASE, session)
    records: "OrderedDict[str, UrlRecord]" = OrderedDict()

    for page in SEED_PAGES:
        try:
            response = session.get(page, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            continue
        for url, label in _extract_definition_links(response.text, page):
            _add_record(records, url, label=label, source_page=page, disallows=disallows)
            if limit is not None and len(records) >= limit:
                return list(records.values())

    return list(records.values())


def collect_capec_urls(
    limit: Optional[int] = None,
    *,
    timeout: float = 30.0,
) -> List[UrlRecord]:
    session = make_session("athena-ctibench/capec-collector (+https://github.com/athena-cti/cti-bench)")
    base = "https://capec.mitre.org"
    disallows = fetch_robots(base, session)
    seeds = [
        "https://capec.mitre.org/data/definitions/1000.html",
        "https://capec.mitre.org/data/definitions/2000.html",
        "https://capec.mitre.org/data/definitions/3000.html",
    ]
    records: "OrderedDict[str, UrlRecord]" = OrderedDict()

    for page in seeds:
        try:
            response = session.get(page, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            continue
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or "data/definitions/" not in href or not href.endswith(".html"):
                continue
            candidate = canonicalize(urljoin(page, href), "https")
            parsed = urlparse(candidate)
            if parsed.netloc.lower() != urlparse(base).netloc.lower():
                continue
            if not allowed_by_robots(parsed.path or "/", disallows):
                continue
            if candidate in records:
                rec = records[candidate]
                rec.metadata.setdefault("source_pages", [])
                if page not in rec.metadata["source_pages"]:
                    rec.metadata["source_pages"].append(page)
                continue
            metadata: dict = {}
            label = anchor.get_text(strip=True)
            if label:
                metadata["label"] = label
            metadata["source_pages"] = [page]
            records[candidate] = UrlRecord.build(
                candidate,
                source_type="capec_catalog",
                source_link=base,
                metadata=metadata,
            )
            if limit is not None and len(records) >= limit:
                return list(records.values())

    return list(records.values())


