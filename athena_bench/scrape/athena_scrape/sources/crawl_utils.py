"""Shared crawling utilities for athena_scrape sources."""
from __future__ import annotations

import re
import time
from collections import deque
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse

import bs4
import requests
from requests.adapters import HTTPAdapter, Retry

from ..models import UrlRecord

SKIP_EXT_DEFAULT = re.compile(
    r".*\.(png|jpe?g|gif|svg|ico|pdf|zip|gz|tgz|bz2|xz|7z|mp4|webm|csv|json|xml|rss)$",
    re.I,
)

_DEFAULT_UA = (
    "Mozilla/5.0 (compatible; athena-ctibench/1.0; "
    "+https://github.com/athena-cti/cti-bench)"
)


def make_session(user_agent: str | None = None) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    session.headers.update({"User-Agent": user_agent or _DEFAULT_UA})
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def fetch_robots(base: str, session: requests.Session) -> List[str]:
    parsed = urlparse(base)
    robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    disallows: List[str] = []
    try:
        response = session.get(robots_url, timeout=15)
        response.raise_for_status()
    except Exception:
        return disallows
    active = False
    for raw_line in response.text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("user-agent:"):
            agent = line.split(":", 1)[1].strip().strip('"')
            active = agent in {"*"}
        elif active and low.startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            if path:
                disallows.append(path)
    return disallows


def allowed_by_robots(path: str, disallows: Iterable[str]) -> bool:
    return not any(path.startswith(rule) for rule in disallows)


def canonicalize(url: str, prefer_scheme: str) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme or prefer_scheme
    netloc = parsed.netloc.lower()
    if scheme == "http" and prefer_scheme == "https" and netloc:
        scheme = "https"
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    normalized = parsed._replace(scheme=scheme, netloc=netloc, path=path)
    return urlunparse(normalized)


def default_metadata_extractor(url: str, response: requests.Response) -> Tuple[dict, Optional[str]]:
    metadata: dict = {}
    last_modified = response.headers.get("last-modified")
    if last_modified:
        metadata["last_modified"] = last_modified
    published: Optional[str] = None
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
    except Exception:
        return metadata, published

    title = soup.find("title")
    if title:
        metadata["title"] = title.get_text(strip=True)

    meta_candidates = [
        {"property": "article:published_time"},
        {"property": "og:updated_time"},
        {"name": "pubdate"},
        {"name": "publishdate"},
        {"name": "date"},
        {"name": "dcterms.date"},
    ]
    for attrs in meta_candidates:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            published = tag["content"].strip()
            break

    if not published:
        time_tag = soup.find("time")
        if time_tag:
            published = time_tag.get("datetime") or time_tag.get_text(strip=True)

    return metadata, published


def crawl_site(
    *,
    base: str,
    seeds: Sequence[str],
    source_type: str,
    source_link: Optional[str] = None,
    max_pages: int = 5000,
    sleep_s: float = 0.2,
    timeout: float = 30.0,
    allowed_path: Optional[Callable[[str], bool]] = None,
    skip_ext: Optional[re.Pattern[str]] = None,
    metadata_extractor: Optional[Callable[[str, requests.Response], Tuple[dict, Optional[str]]]] = None,
) -> List[UrlRecord]:
    parsed_base = urlparse(base)
    scheme = parsed_base.scheme or "https"
    host = parsed_base.netloc.lower()
    allowed_hosts = {host}
    if host.startswith("www."):
        allowed_hosts.add(host[4:])
    else:
        allowed_hosts.add(f"www.{host}")

    session = make_session(f"athena-ctibench/{source_type}-collector (+https://github.com/athena-cti/cti-bench)")
    disallows = fetch_robots(base, session)
    skip_ext = skip_ext or SKIP_EXT_DEFAULT
    metadata_extractor = metadata_extractor or default_metadata_extractor

    queue: deque[str] = deque()
    for seed in seeds:
        queue.append(canonicalize(urljoin(base, seed), scheme))

    seen: dict[str, UrlRecord] = {}

    while queue and len(seen) < max_pages:
        url = queue.popleft()
        url = canonicalize(url, scheme)
        if url in seen:
            continue

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc.lower() not in allowed_hosts:
            continue
        if skip_ext.match(parsed.path.lower()):
            continue
        if allowed_path and not allowed_path(parsed.path):
            continue
        if not allowed_by_robots(parsed.path, disallows):
            continue

        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
        except Exception:
            if sleep_s:
                time.sleep(sleep_s)
            continue

        status = response.status_code
        if status != 200:
            if sleep_s:
                time.sleep(sleep_s)
            continue

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            if sleep_s:
                time.sleep(sleep_s)
            continue

        metadata, published = metadata_extractor(url, response)
        record = UrlRecord.build(
            url,
            source_type=source_type,
            source_link=source_link or base,
            published=published,
            metadata={k: v for k, v in metadata.items() if v},
        )
        seen[url] = record

        try:
            soup = bs4.BeautifulSoup(response.text, "html.parser")
        except Exception:
            soup = None

        if soup:
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()
                if not href or href.startswith("mailto:") or href.startswith("javascript:"):
                    continue
                next_url = canonicalize(urljoin(url, href), scheme)
                next_parsed = urlparse(next_url)
                if next_parsed.netloc.lower() not in allowed_hosts:
                    continue
                if skip_ext.match(next_parsed.path.lower()):
                    continue
                if allowed_path and not allowed_path(next_parsed.path):
                    continue
                if not allowed_by_robots(next_parsed.path, disallows):
                    continue
                if next_url not in seen:
                    queue.append(next_url)

        if sleep_s:
            time.sleep(sleep_s)

    return list(seen.values())


