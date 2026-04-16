"""High level orchestration helpers for athena_scrape."""
from __future__ import annotations

import bs4

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm

from .markdown_renderers import render_markdown
NAV_SELECTORS = [
    "nav",
    "header",
    "footer",
    ".breadcrumbs",
    ".breadcrumb",
    "#breadcrumbs",
    ".usa-breadcrumb",
    ".site-alert",
]

CONTENT_SELECTORS = {
    "capec_catalog": [
        "td#Contentpane",
        "table.MainPane",
        "table#MainPane",
    ],
    "cwe_catalog": [
        "div#CWEDefinition",
        "td#Contentpane",
    ],
    "mitre_attack": [
        "div#v-tabContent",
    ],
    "cisa_ics": [
        "main.c-main",
        "main",
    ],
    "cisa_csa": [
        "main.c-main",
        "main",
    ],
}

EXCLUDE_PREFIXES = {
    "capec_catalog": [
        "CAPEC -",
        "Common Attack Pattern Enumeration",
        "Home",
        "About",
        "Community",
        "Search",
        "CAPEC on",
        "CAPEC List Quick Access",
        "News",
        "Cookie Notice",
    ],
    "cwe_catalog": [
        "CWE -",
        "Common Weakness Enumeration",
        "Home",
        "Search",
        "Cookie Notice",
    ],
    "mitre_attack": [
        "Home",
        "Techniques",
        "Enterprise",
        "Mobile",
        "ICS",
        "Defenses",
        "Data Sources",
        "Mitigations",
        "Assets",
        "CTI",
        "Groups",
        "Software",
        "Campaigns",
        "Resources",
        "Get Started",
        "Learn More about ATT&CK",
        "ATT&CK Data & Tools",
        "FAQ",
        "Engage with ATT&CK",
        "Version History",
        "Updates",
        "Legal & Branding",
        "Benefactors",
        "Blog",
        "Search",
        "ATT&CKcon",
        "Tickets are available now",
    ],
    "cisa_ics": [
        "Share this page:",
        "Subscribe:",
        "Was this page helpful?",
    ],
    "cisa_csa": [
        "Share this page:",
        "Subscribe:",
        "Was this page helpful?",
    ],
}

EXCLUDE_LINES = {
    "mitre_attack": {
        "Matrices",
        "Tactics",
        "Enterprise",
        "Mobile",
        "ICS",
        "Defenses",
        "Data Sources",
        "Mitigations",
        "Assets",
        "CTI",
        "Groups",
        "Software",
        "Campaigns",
        "Resources",
        "Get Started",
        "Learn More about ATT&CK",
        "ATT&CKcon 6.0 is coming October 14-15 in McLean, VA and live online.",
        "Tickets are available now",
        "Search",
        "!",
    },
    "cisa_ics": {"Share this page:", "Subscribe:", "Was this page helpful?"},
    "cisa_csa": {"Share this page:", "Subscribe:", "Was this page helpful?"},
}

def _select_content_node(source_type: str, soup: bs4.BeautifulSoup) -> bs4.element.Tag | bs4.BeautifulSoup:
    selectors = CONTENT_SELECTORS.get(source_type, [])
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node
    return soup

def _strip_navigation(soup_fragment: bs4.BeautifulSoup) -> bs4.BeautifulSoup:
    for selector in NAV_SELECTORS:
        for elem in soup_fragment.select(selector):
            elem.decompose()
    return soup_fragment

def _trim_text(source_type: str, text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    def find_start(predicate):
        for idx, line in enumerate(lines):
            if predicate(line):
                return idx
        return 0

    if source_type == "capec_catalog":
        start = find_start(lambda l: "CAPEC-" in l or "CAPEC View" in l)
    elif source_type == "cwe_catalog":
        start = find_start(lambda l: "CWE-" in l)
    elif source_type == "mitre_attack":
        start = find_start(
            lambda l: any(part.startswith("T") and part[1:].replace(".", "").isdigit() for part in l.split())
        )
    elif source_type.startswith("cisa_"):
        start = find_start(lambda l: any(keyword in l for keyword in ("Advisory", "Alert", "Summary")))
    else:
        start = 0

    lines = lines[start:]
    prefixes = EXCLUDE_PREFIXES.get(source_type, [])
    filtered = [line for line in lines if not any(line.startswith(prefix) for prefix in prefixes)]
    excluded_lines = EXCLUDE_LINES.get(source_type, set())
    filtered = [line for line in filtered if line not in excluded_lines]
    return "\n".join(filtered).strip()




from .fetchers import fetch_url_content, make_session
from .io_utils import (
    read_content_records,
    read_url_records_csv,
    write_content_records,
    write_processed_records,
    write_url_records_csv,
)
from .models import ContentRecord, ProcessedRecord, UrlRecord
from .processors import get_parser
from .sources import available_sources, get_collector


def collect_urls(
    *,
    sources: Sequence[str] | None,
    output: Path | None = None,
    mitre_max_pages: int = 200_000,
    mitre_sleep: float = 0.2,
    mitre_timeout: float = 30.0,
    generic_limit: int | None = None,
) -> List[UrlRecord]:
    selected = list(sources) if sources else available_sources()
    all_records: Dict[str, UrlRecord] = {}
    for name in selected:
        collector = get_collector(name)
        kwargs = {}
        if name == "mitre_attack":
            kwargs = {
                "max_pages": mitre_max_pages,
                "sleep_s": mitre_sleep,
                "timeout": mitre_timeout,
            }
        else:
            if generic_limit is not None:
                kwargs = {"limit": generic_limit}
        try:
            records = collector(**kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to collect from {name}: {exc}")
            continue
        for record in records:
            all_records[record.url] = record
    path = write_url_records_csv(list(all_records.values()), output)
    return read_url_records_csv(path)


def scrape_urls(
    *,
    url_csv: Path,
    output: Path | None = None,
    limit: int | None = None,
    timeout: float = 30.0,
) -> List[ContentRecord]:
    records = read_url_records_csv(url_csv)
    if limit is not None:
        records = records[:limit]
    session = make_session()
    fetched: List[ContentRecord] = []
    for record in records:
        fetched.append(fetch_url_content(record, session=session, timeout=timeout))
    out_path = write_content_records(fetched, output)
    return read_content_records(out_path)


def process_content(
    *,
    raw_path: Path,
    output_root: Path | None = None,
) -> Dict[str, List[ProcessedRecord]]:
    contents = read_content_records(raw_path)
    buckets: Dict[str, List[ProcessedRecord]] = defaultdict(list)
    for item in contents:
        parser = get_parser(item.source_type)
        if parser is None:
            continue
        payload = parser(item.content)
        if not payload:
            continue
        record = ProcessedRecord(
            url_id=item.url_id,
            url=item.url,
            source_type=item.source_type,
            payload=payload,
            processed_at=item.fetched_at,
        )
        buckets[item.source_type].append(record)

    for source_type, records in buckets.items():
        write_processed_records(source_type, records, root=output_root)
    return buckets


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = bs4.BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return soup.get_text("\n", strip=True)


def build_mcq_corpus(
    *,
    url_csv: Path,
    raw_root: Path,
    processed_root: Path,
    limit: int | None = None,
    timeout: float = 30.0,
) -> Dict[str, int]:
    records = read_url_records_csv(url_csv)
    if limit is not None:
        records = records[:limit]
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    session = make_session()
    processed = 0
    skipped = 0

    for record in tqdm(records, desc="Fetching MCQ content", unit="url"):
        try:
            content = fetch_url_content(record, session=session, timeout=timeout)
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Failed to fetch {record.url}: {exc}")
            continue

        if content.status != 200 or not content.content:
            skipped += 1
            print(f"[WARN] Skipping {record.url} (HTTP {content.status})")
            continue

        soup = bs4.BeautifulSoup(content.content, "html.parser")
        primary = _select_content_node(record.source_type, soup)
        working = bs4.BeautifulSoup(str(primary), "html.parser")
        _strip_navigation(working)

        text_content = _trim_text(record.source_type, working.get_text("\n", strip=True))
        if not text_content:
            text_content = working.get_text("\n", strip=True)

        raw_dir = raw_root / record.source_type
        processed_dir = processed_root / record.source_type
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_path = raw_dir / f"{record.url_id}.txt"
        processed_path = processed_dir / f"{record.url_id}.txt"

        raw_path.write_text(text_content + "\n", encoding="utf-8")

        markdown = render_markdown(record.source_type, str(working), text_content)
        if not markdown.endswith("\n"):
            markdown += "\n"
        processed_path.write_text(markdown, encoding="utf-8")

        processed += 1

    return {"processed": processed, "skipped": skipped}

