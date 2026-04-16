"""MITRE ATT&CK URL collection utilities."""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from typing import Dict, Iterable, Optional
from urllib.parse import urljoin, urlparse

import bs4
import requests

from ..models import UrlRecord
from .crawl_utils import allowed_by_robots, canonicalize, fetch_robots, make_session

BASE = "https://attack.mitre.org"
HOST = urlparse(BASE).netloc.lower()

ROOT_PATHS = [
    "/",
    "/techniques/enterprise/",
    "/techniques/mobile/",
    "/techniques/ics/",
    "/datasources/",
    "/mitigations/enterprise/",
    "/mitigations/mobile/",
    "/mitigations/ics/",
    "/assets/",
    "/groups/",
    "/software/",
    "/campaigns/",
]

ALLOWED_PREFIXES = (
    "/techniques/",
    "/datasources/",
    "/mitigations/",
    "/assets/",
    "/groups/",
    "/software/",
    "/campaigns/",
)

EXCLUDED_PATHS = set(ROOT_PATHS) | {"/techniques/", "/datasources/", "/mitigations/", "/assets/", "/groups/", "/software/", "/campaigns/"}

TECHNIQUE_ID = re.compile(r"^/techniques/[A-Z]\d{4}(?:/\d{3})?/?$")
DATA_ID = re.compile(r"^/datasources/[A-Z]{2}\d{4}(?:/\d{3})?/?$")
MITIGATION_ID = re.compile(r"^/mitigations/M\d{4}/?$")
GROUP_ID = re.compile(r"^/groups/G\d{4}/?$")
SOFTWARE_ID = re.compile(r"^/software/S\d{4}/?$")
CAMPAIGN_ID = re.compile(r"^/campaigns/C\d{4}/?$")
ASSET_PATH = re.compile(r"^/assets/.+")

PATTERNS = (
    TECHNIQUE_ID,
    DATA_ID,
    MITIGATION_ID,
    GROUP_ID,
    SOFTWARE_ID,
    CAMPAIGN_ID,
    ASSET_PATH,
)


def _build_record(url: str, *, label: Optional[str], metadata: Optional[Dict[str, str]], published: Optional[str]) -> UrlRecord:
    data: Dict[str, str] = metadata.copy() if metadata else {}
    if label and label.strip():
        data.setdefault("label", label.strip())
    return UrlRecord.build(
        url,
        source_type="mitre_attack",
        source_link=BASE,
        published=published,
        metadata=data or None,
    )


def _extract_metadata(response: requests.Response) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    title = None
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
    except Exception:
        title = None
    if title:
        metadata["title"] = title
    last_modified = response.headers.get("last-modified")
    if last_modified:
        metadata["last_modified"] = last_modified
    return metadata


def _matches_patterns(path: str) -> bool:
    for pattern in PATTERNS:
        if pattern.match(path):
            return True
    return False


def _should_include(path: str) -> bool:
    if path in EXCLUDED_PATHS:
        return False
    if not any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        return False
    return _matches_patterns(path) or path.startswith("/assets/")


def collect_mitre_urls(
    *,
    max_pages: int = 200_000,
    sleep_s: float = 0.0,
    timeout: float = 30.0,
) -> list[UrlRecord]:
    session = make_session("athena-ctibench/mitre-targeted (+https://github.com/athena-cti/cti-bench)")
    disallows = fetch_robots(BASE, session)
    records: "OrderedDict[str, UrlRecord]" = OrderedDict()

    def add(url: str, *, label: Optional[str] = None, metadata: Optional[Dict[str, str]] = None, published: Optional[str] = None) -> None:
        url = canonicalize(url, "https")
        path = urlparse(url).path or "/"
        if not allowed_by_robots(path, disallows):
            return
        record = _build_record(url, label=label, metadata=metadata, published=published)
        records[url] = record

    for root_path in ROOT_PATHS:
        full_url = canonicalize(urljoin(BASE, root_path), "https")
        try:
            response = session.get(full_url, timeout=timeout)
        except Exception:
            continue
        if response.status_code != 200:
            continue
        metadata = _extract_metadata(response)
        published = metadata.pop("last_modified", None)
        add(full_url, metadata=metadata, published=published)

        try:
            soup = bs4.BeautifulSoup(response.text, "html.parser")
        except Exception:
            soup = None
        if soup:
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()
                if not href:
                    continue
                candidate = canonicalize(urljoin(full_url, href), "https")
                parsed = urlparse(candidate)
                if parsed.netloc.lower() != HOST:
                    continue
                path = parsed.path or "/"
                if not _should_include(path):
                    continue
                label = anchor.get_text(strip=True)
                add(candidate, label=label)
        if sleep_s:
            time.sleep(sleep_s)

    results = list(records.values())
    if max_pages and len(results) > max_pages:
        return results[:max_pages]
    return results


