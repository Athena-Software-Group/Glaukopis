#!/usr/bin/env python
"""Fetch CyberSOCEval data into ``SFT/test/benchmark_data/cybersoceval/``.

Sources (cloned/downloaded once, then converted in place):

  * ``meta-llama/PurpleLlama``        - ``questions.json`` (malware) and
                                        ``report_questions.json`` (threat-intel).
  * ``CrowdStrike/CyberSOCEval_data`` - ``data/hybrid-analysis/`` JSON sandbox
                                        reports + ``data/crowdstrike-reports/``
                                        bundled PDFs.

Layout produced (read by ``benchmarks/cybersoceval_*.py``)::

    benchmark_data/cybersoceval/
      malware_analysis/
        questions.jsonl
        hybrid-analysis/<attack>/<sha256>
      threat_intel_reasoning/
        report_questions.jsonl
        crowdstrike-reports/<report_id>.pdf
        pdfs/<report_id>.pdf            (downloaded for non-CrowdStrike sources)
        <report_id>.txt                 (extracted text, one per question report)

Idempotent: re-running skips already-cloned repos, already-downloaded PDFs,
and already-extracted text files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

PURPLELLAMA_RAW = (
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/"
    "CybersecurityBenchmarks/datasets/crwd_meta"
)
CYBERSOCEVAL_REPO = "https://github.com/CrowdStrike/CyberSOCEval_data.git"


def _git_clone_or_pull(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists; pulling latest")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] {url} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def _curl_json(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url} -> {dest}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def _json_to_jsonl(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"[skip] {dst} already present")
        return
    print(f"[conv] {src} -> {dst}")
    data = json.loads(src.read_text(encoding="utf-8"))
    with dst.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        print(f"[link] {dst} -> {src}")
    except OSError:
        print(f"[copy] {src} -> {dst}")
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


# Per-attempt backoff for retryable PDF fetch failures. Tuned for
# web.archive.org's burst limiter, which closes the TCP connection after
# ~10-15 rapid requests and stays unresponsive for ~30-60s before letting
# new connections through. Five attempts with 5/15/60/180s waits gives a
# ~4 minute worst-case ceiling per URL while comfortably outlasting a
# typical IA throttling event.
_RETRY_BACKOFF_SECS = (5, 15, 60, 180)
# 4xx codes that mean "the resource is gone / forbidden" -- retrying is
# pointless and just wastes wall time. Anything else (network errors, 429,
# 5xx) is treated as transient and retried.
_TERMINAL_HTTP_CODES = frozenset({400, 401, 403, 404, 410})


def _download_pdf(
    url: str,
    dst: Path,
    retries: int = len(_RETRY_BACKOFF_SECS),
    pacing_delay: float = 0.3,
) -> Optional[Path]:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Inter-URL pacing: cheap insurance against IA's burst limiter
    # (~10-15 rapid requests then it stops accepting connections). Applied
    # before the first attempt so the call site doesn't have to sleep
    # between successful downloads. Skipped when the file is already on
    # disk via the early return above.
    if pacing_delay > 0:
        time.sleep(pacing_delay)
    total_attempts = retries + 1
    for attempt in range(total_attempts):
        try:
            print(f"[pdf ] ({attempt + 1}/{total_attempts}) {url}")
            r = requests.get(url, allow_redirects=True, timeout=60)
            if r.status_code == 200:
                dst.write_bytes(r.content)
                return dst
            print(f"  -> HTTP {r.status_code}")
            if r.status_code in _TERMINAL_HTTP_CODES:
                # No point retrying a 404; bail immediately.
                return None
            # Honor server-provided Retry-After when present (seconds or
            # HTTP-date). Falls through to the backoff schedule when the
            # header is missing or unparseable.
            wait_override = _parse_retry_after(r.headers.get("Retry-After"))
        except requests.RequestException as e:
            print(f"  -> {e}")
            wait_override = None
        if attempt < retries:
            wait = wait_override if wait_override is not None else _RETRY_BACKOFF_SECS[attempt]
            print(f"  -> backing off {wait}s before retry")
            time.sleep(wait)
    return None


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Best-effort parse of a Retry-After header. Returns seconds-to-wait,
    or None when the header is absent or unparseable."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    # HTTP-date form (rare for this use case); fall back to None rather
    # than pulling in email.utils for a one-shot parse.
    return None


def _pdf_to_text(pdf: Path, txt: Path) -> bool:
    if txt.exists():
        return True
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[ERR ] pypdf not installed. Run: pip install pypdf", file=sys.stderr)
        return False
    try:
        reader = PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        txt.write_text(text, encoding="utf-8")
        return True
    except Exception as e:
        print(f"[ERR ] {pdf} -> {txt}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="SFT/test/benchmark_data/cybersoceval")
    ap.add_argument("--cache-dir", default="SFT/test/benchmark_data/cybersoceval/_cyberSOCEval_data")
    ap.add_argument(
        "--max-retries", type=int, default=len(_RETRY_BACKOFF_SECS),
        help="Per-PDF retry budget on transient failures (default: %(default)s; "
             "backoff schedule: 5/15/60/180s).",
    )
    ap.add_argument(
        "--pacing-delay", type=float, default=0.3,
        help="Sleep between successive PDF downloads in seconds (default: %(default)s). "
             "web.archive.org's burst limiter trips at ~10-15 rapid requests; the "
             "default pace keeps us comfortably under that ceiling.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    cache = Path(args.cache_dir)
    malware_dir = out / "malware_analysis"
    ti_dir = out / "threat_intel_reasoning"
    malware_dir.mkdir(parents=True, exist_ok=True)
    ti_dir.mkdir(parents=True, exist_ok=True)

    _git_clone_or_pull(CYBERSOCEVAL_REPO, cache)
    _symlink_or_copy(cache / "data" / "hybrid-analysis", malware_dir / "hybrid-analysis")
    _symlink_or_copy(cache / "data" / "crowdstrike-reports", ti_dir / "crowdstrike-reports")

    mq_json = malware_dir / "questions.json"
    tq_json = ti_dir / "report_questions.json"
    _curl_json(f"{PURPLELLAMA_RAW}/malware_analysis/questions.json", mq_json)
    _curl_json(f"{PURPLELLAMA_RAW}/threat_intel_reasoning/report_questions.json", tq_json)
    _json_to_jsonl(mq_json, malware_dir / "questions.jsonl")
    _json_to_jsonl(tq_json, ti_dir / "report_questions.jsonl")

    pdfs_dir = ti_dir / "pdfs"
    questions = json.loads(tq_json.read_text(encoding="utf-8"))
    seen: set[str] = set()
    missing: list[str] = []
    for entry in questions:
        rid = entry["report_id"]
        if rid in seen:
            continue
        seen.add(rid)
        if entry.get("source") == "CrowdStrike":
            pdf = ti_dir / "crowdstrike-reports" / f"{rid}.pdf"
        else:
            pdf = _download_pdf(
                entry["url_source"],
                pdfs_dir / f"{rid}.pdf",
                retries=args.max_retries,
                pacing_delay=args.pacing_delay,
            )
        if pdf is None or not pdf.exists():
            missing.append(rid)
            continue
        if not _pdf_to_text(pdf, ti_dir / f"{rid}.txt"):
            missing.append(rid)

    print(f"\n[done] reports total={len(seen)} missing={len(missing)}")
    if missing:
        print("[done] missing report_ids:", ", ".join(sorted(missing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
