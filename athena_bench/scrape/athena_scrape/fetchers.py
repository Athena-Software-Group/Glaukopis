"""HTTP fetching helpers for athena_scrape."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import bs4
import requests
from requests.adapters import HTTPAdapter, Retry

from .models import ContentRecord, UrlRecord

TEXTUAL_CONTENT = re.compile(r"text/(html|plain|xml|markdown)", re.I)


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.headers.update({"User-Agent": "athena-ctibench/scraper"})
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def _html_to_text(html: str) -> str:
    soup = bs4.BeautifulSoup(html, "html.parser")
    # Drop script/style blocks to keep content clean.
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return soup.get_text("\n", strip=True)


def fetch_url_content(
    record: UrlRecord,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 30.0,
) -> ContentRecord:
    sess = session or make_session()
    response = sess.get(record.url, timeout=timeout, allow_redirects=True)
    content_type = response.headers.get("content-type", "")
    if TEXTUAL_CONTENT.search(content_type):
        content = _html_to_text(response.text)
    else:
        content = response.text if isinstance(response.text, str) else ""
    return ContentRecord(
        url_id=record.url_id,
        url=record.url,
        source_type=record.source_type,
        fetched_at=datetime.now(tz=timezone.utc).isoformat(),
        status=response.status_code,
        content_type=content_type,
        content=content,
        metadata={"encoding": response.encoding},
    )




