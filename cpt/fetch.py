#!/usr/bin/env python3
"""Per-source fetchers for the CPT corpus build.

Dispatches on the `fetcher` key of each entry in sources.yaml:
  http_json  - GET a single URL (handles .gz transparently); write to raw/
  http_zip   - GET + unzip; write extracted members to raw/
  git        - shallow clone a repo (subdir optional); leave as tree on disk
  stix       - alias for http_json (STIX bundles are JSON)
  rss        - feedparser on an RSS/Atom URL; fetch each <link> to raw/
  sitemap    - parse an XML sitemap; fetch each <loc> to raw/
  manual     - read cache/manual/<source>/urls.txt; fetch each line to raw/

All fetchers are idempotent: if the expected output file already exists
on disk, they skip the download. Use --force to refetch. Per-source raw
output lands in cache/raw/<source-name>/.

This module is importable (build_corpus.py dispatches through
fetch_source) and runnable (`python cpt/fetch.py --source <name>`).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
RAW_DIR = CACHE_DIR / "raw"
MANUAL_DIR = CACHE_DIR / "manual"

USER_AGENT = "Glaukopis-CPT-Corpus/0.1 (+https://github.com/Athena-Software-Group/Glaukopis)"
TIMEOUT = 60


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _safe_filename(url: str) -> str:
    """Derive a stable, filesystem-safe filename from a URL."""
    parsed = urlparse(url)
    tail = Path(parsed.path).name or "index"
    if not tail or tail == "/":
        tail = "index"
    # Guard against absurdly long URLs / query strings.
    if len(tail) > 128:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        tail = f"{tail[:64]}_{h}"
    return tail


def _write_stream(dst: Path, content: bytes, decompress_gz: bool = True) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if decompress_gz and dst.suffix == ".gz":
        dst = dst.with_suffix("")  # drop .gz
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
            dst.write_bytes(gz.read())
    else:
        dst.write_bytes(content)
    return dst


# ---------- fetcher implementations ----------

def fetch_http_json(name: str, spec: dict[str, Any], force: bool) -> list[Path]:
    url = spec["url"]
    out_dir = RAW_DIR / name
    fname = _safe_filename(url)
    dst = out_dir / fname
    # If gzipped, the final on-disk name drops .gz
    if dst.suffix == ".gz":
        final = dst.with_suffix("")
    else:
        final = dst
    if final.exists() and not force:
        return [final]
    print(f"[fetch:{name}] GET {url}")
    r = _session().get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return [_write_stream(dst, r.content, decompress_gz=True)]


def fetch_http_zip(name: str, spec: dict[str, Any], force: bool) -> list[Path]:
    url = spec["url"]
    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".fetched"
    if marker.exists() and not force:
        return sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != ".fetched")
    print(f"[fetch:{name}] GET (zip) {url}")
    r = _session().get(url, timeout=TIMEOUT)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out_dir)
    marker.write_text("ok")
    return sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != ".fetched")


def fetch_git(name: str, spec: dict[str, Any], force: bool) -> list[Path]:
    url = spec["url"]
    out_dir = RAW_DIR / name
    repo_dir = out_dir / "repo"
    if repo_dir.exists() and not force:
        return [repo_dir]
    if repo_dir.exists() and force:
        shutil.rmtree(repo_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fetch:{name}] git clone --depth=1 {url}")
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", url, str(repo_dir)],
        check=True,
    )
    return [repo_dir]


def _fetch_url_list(name: str, urls: list[str], force: bool) -> list[Path]:
    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = _session()
    results: list[Path] = []
    for url in urls:
        fname = _safe_filename(url)
        dst = out_dir / fname
        if dst.exists() and not force:
            results.append(dst)
            continue
        try:
            print(f"[fetch:{name}] GET {url}")
            r = sess.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            dst.write_bytes(r.content)
            results.append(dst)
        except Exception as e:  # noqa: BLE001
            print(f"[fetch:{name}] FAIL {url}: {e}", file=sys.stderr)
    return results


def fetch_rss(name: str, spec: dict[str, Any], force: bool) -> list[Path]:
    import feedparser  # local import: heavy dep

    url = spec["url"]
    print(f"[fetch:{name}] RSS {url}")
    feed = feedparser.parse(url)
    urls = [e.link for e in feed.entries if getattr(e, "link", None)]
    return _fetch_url_list(name, urls, force)


def fetch_sitemap(name: str, spec: dict[str, Any], force: bool) -> list[Path]:
    from xml.etree import ElementTree as ET

    url = spec["url"]
    print(f"[fetch:{name}] sitemap {url}")
    r = _session().get(url, timeout=TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    # Sitemaps: <urlset><url><loc>...</loc></url></urlset>
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [el.text.strip() for el in root.findall(".//sm:loc", ns) if el.text]
    return _fetch_url_list(name, urls, force)


def fetch_manual(name: str, spec: dict[str, Any], force: bool) -> list[Path]:
    urls_file = MANUAL_DIR / name / "urls.txt"
    if not urls_file.exists():
        print(f"[fetch:{name}] manual URL list missing: {urls_file}", file=sys.stderr)
        print(f"[fetch:{name}] create the file (one URL per line) and rerun", file=sys.stderr)
        return []
    urls = [ln.strip() for ln in urls_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    return _fetch_url_list(name, urls, force)


DISPATCH = {
    "http_json": fetch_http_json,
    "stix": fetch_http_json,  # alias
    "http_zip": fetch_http_zip,
    "git": fetch_git,
    "rss": fetch_rss,
    "sitemap": fetch_sitemap,
    "manual": fetch_manual,
}


def fetch_source(name: str, spec: dict[str, Any], force: bool = False) -> list[Path]:
    fetcher = spec.get("fetcher")
    if fetcher not in DISPATCH:
        raise ValueError(f"Unknown fetcher '{fetcher}' for source '{name}'")
    return DISPATCH[fetcher](name, spec, force)


def load_sources(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch CTI corpus sources declared in sources.yaml")
    p.add_argument("--config", default=str(SCRIPT_DIR / "sources.yaml"))
    p.add_argument("--source", default="", help="Fetch only this named source (default: all enabled)")
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = p.parse_args(argv)

    cfg = load_sources(Path(args.config))
    sources = cfg.get("sources", {})
    targets = [args.source] if args.source else [n for n, s in sources.items() if s.get("enabled", False)]

    for name in targets:
        spec = sources.get(name)
        if spec is None:
            print(f"[fetch] unknown source: {name}", file=sys.stderr)
            return 2
        if not spec.get("enabled", False) and not args.source:
            continue
        try:
            paths = fetch_source(name, spec, force=args.force)
            print(f"[fetch:{name}] ok, {len(paths)} file(s)")
        except Exception as e:  # noqa: BLE001
            print(f"[fetch:{name}] ERROR: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
