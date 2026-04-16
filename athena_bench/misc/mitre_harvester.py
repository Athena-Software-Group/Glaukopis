#!/usr/bin/env python3
"""
Harvest all internal URLs from https://attack.mitre.org, politely.

- Scopes to attack.mitre.org only
- Obeys robots.txt Disallow rules
- Skips non-HTML/asset URLs (png, pdf, etc.)
- Retries with backoff; configurable sleep between requests
- Writes deduped list to mitre_urls.txt (and a JSONL with status/ctype)

Usage:
  python harvest_attack_urls.py --out mitre_urls.txt --max 200000 --sleep 0.2
"""

import argparse
import re
import time
import json
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

BASE = "https://attack.mitre.org"
SEEDS = [
    "/", "/techniques/", "/techniques/enterprise/", "/techniques/mobile/", "/techniques/ics/",
    "/groups/", "/software/", "/mitigations/", "/datasources/", "/campaigns/", "/matrices/", "/matrices/enterprise/"
]

SKIP_EXT = re.compile(
    r".*\.(png|jpg|jpeg|gif|svg|ico|pdf|zip|gz|tgz|bz2|xz|7z|mp4|webm|csv|json|xml)$",
    re.I,
)

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    s.headers.update({"User-Agent": "athena-ctibench-url-harvester/0.1 (+research)"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def fetch_robots(session: requests.Session) -> list[str]:
    """Return Disallow paths for User-agent: * (simple parser)."""
    disallows = []
    try:
        r = session.get(urljoin(BASE, "/robots.txt"), timeout=20)
        txt = r.text
        active = False
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("user-agent:"):
                ua = line.split(":", 1)[1].strip()
                active = (ua == "*" or ua.strip('"') == "*")
            elif active and low.startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    disallows.append(path)
    except Exception:
        # Fallback if robots fails to load; be conservative
        disallows = ["/previous/", "/versions/"]
    return disallows

def allowed_by_robots(path: str, disallows: list[str]) -> bool:
    for d in disallows:
        if path.startswith(d):
            return False
    return True

def same_origin(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in {"http", "https"} and p.netloc == urlparse(BASE).netloc

def is_html_url(url: str) -> bool:
    return not SKIP_EXT.match(url)

def harvest(out_txt: str, out_jsonl: str, max_pages: int, sleep_s: float, timeout: float):
    sess = make_session()
    disallows = fetch_robots(sess)

    seen: set[str] = set()
    q: deque[str] = deque()

    # seed queue
    for p in SEEDS:
        q.append(urljoin(BASE, p))

    with open(out_jsonl, "w", encoding="utf-8") as meta_out:
        while q and len(seen) < max_pages:
            url = q.popleft()
            url, _ = urldefrag(url)  # drop fragments
            if url in seen:
                continue
            if not same_origin(url) or not is_html_url(url):
                continue
            if not allowed_by_robots(urlparse(url).path, disallows):
                continue

            try:
                r = sess.get(url, timeout=timeout, allow_redirects=True)
                ctype = r.headers.get("content-type", "")
                status = r.status_code

                meta_out.write(json.dumps({
                    "url": url, "status": status, "content_type": ctype
                }) + "\n")

                # Only parse HTML pages
                if status == 200 and "text/html" in ctype:
                    seen.add(url)
                    soup = BeautifulSoup(r.text, "lxml")
                    for a in soup.find_all("a", href=True):
                        href = a["href"].strip()
                        if not href:
                            continue
                        nxt = urljoin(url, href)
                        nxt, _ = urldefrag(nxt)
                        if nxt not in seen and same_origin(nxt) and is_html_url(nxt):
                            # obey robots for the discovered path
                            if allowed_by_robots(urlparse(nxt).path, disallows):
                                q.append(nxt)
            except Exception:
                # swallow & move on, but slow down a touch
                pass
            finally:
                if sleep_s > 0:
                    time.sleep(sleep_s)

    # Write the final deduped list in stable order (sorted)
    urls_sorted = sorted(seen)
    with open(out_txt, "w", encoding="utf-8") as fout:
        for u in urls_sorted:
            fout.write(u + "\n")

    print(f"Discovered {len(urls_sorted)} HTML pages. Wrote:")
    print(f" - {out_txt}")
    print(f" - {out_jsonl}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mitre_urls.txt", help="Output text file (one URL per line).")
    ap.add_argument("--meta", default="mitre_urls_meta.jsonl", help="Per-URL status/ctype log.")
    ap.add_argument("--max", type=int, default=50000, help="Safety cap on pages to visit.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Delay between requests (sec).")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout.")
    args = ap.parse_args()

    harvest(args.out, args.meta, args.max, args.sleep, args.timeout)

if __name__ == "__main__":
    main()
