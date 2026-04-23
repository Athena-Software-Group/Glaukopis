#!/usr/bin/env python3
"""Post-parse processing: quality filter, benchmark-leak filter, dedupe, stats.

Input: one or more JSONL files produced by parse.py, each line
    {"text": str, "meta": {...}}
Output: a single merged+filtered JSONL in cpt/corpus/<name>.jsonl, plus
    cpt/cache/leak_report.json (counts of docs dropped per filter).

The three filters in order:

  1. quality_filter  - minimum length, maximum length, ascii ratio,
                       language heuristic (english-heavy tokens)
  2. leak_filter     - exact-id + 13-gram minhash against benchmark TSVs
  3. near_dup_filter - within-corpus dedupe via MinHashLSH (Jaccard 0.8)

Quality gates are conservative; leak filter is strict (bias toward false
positives since contamination invalidates eval). All dropped docs are
counted in leak_report.json for audit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ---------- quality ----------

_ID_PATTERNS = re.compile(
    r"CVE-\d{4}-\d{4,7}|CWE-\d+|CAPEC-\d+|T\d{4}(?:\.\d+)?|G\d{4}|S\d{4}|M\d{4}",
    re.IGNORECASE,
)


def extract_ids(text: str) -> set[str]:
    return {m.group(0).upper() for m in _ID_PATTERNS.finditer(text)}


def quality_ok(text: str, min_chars: int = 200, max_chars: int = 200_000) -> bool:
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    # cheap language heuristic: at least 60% ascii letters / whitespace / punctuation
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / n < 0.85:
        return False
    # word-like token density
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 40:
        return False
    return True


# ---------- minhash + LSH ----------

def _shingles(text: str, n: int) -> list[bytes]:
    toks = re.findall(r"[A-Za-z0-9_.-]+", text.lower())
    if len(toks) < n:
        return []
    return [" ".join(toks[i : i + n]).encode("utf-8") for i in range(len(toks) - n + 1)]


def _minhash(text: str, n: int, num_perm: int = 128):
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(text, n):
        mh.update(sh)
    return mh


# ---------- benchmark leak index ----------

def build_leak_index(config: dict[str, Any]) -> tuple[set[str], Any]:
    """Build (exact_ids, lsh_index) from the benchmark test splits.

    exact_ids: set of uppercased CVE/CWE/CAPEC/technique ids seen in any
        test row. Any corpus doc containing *one of these* verbatim is
        dropped (strong prior against contamination).
    lsh_index: datasketch MinHashLSH holding a minhash per test row for
        near-dup Jaccard filtering.
    """
    from datasketch import MinHashLSH

    lp = config.get("leak_protection", {}) or {}
    threshold = float(lp.get("jaccard_threshold", 0.3))
    n = int(lp.get("ngram", 13))

    rows: list[str] = []
    for rel in lp.get("benchmarks", []) or []:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"[leak] benchmark file missing (skipped): {path}", file=sys.stderr)
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    rows.append(" ".join(c for c in row if c))
        except Exception as e:  # noqa: BLE001
            print(f"[leak] read {path}: {e}", file=sys.stderr)
    for rel in lp.get("athena_dirs", []) or []:
        d = REPO_ROOT / rel
        if not d.exists():
            continue
        for p in d.rglob("*.jsonl"):
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        rows.append(json.dumps(obj, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                continue

    exact_ids: set[str] = set()
    for r in rows:
        exact_ids.update(extract_ids(r))

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, r in enumerate(rows):
        mh = _minhash(r, n=n)
        if len(mh.digest()) == 0:
            continue
        try:
            lsh.insert(f"bench_{i}", mh)
        except ValueError:
            # duplicate key; ignore
            pass
    print(f"[leak] indexed {len(rows)} benchmark rows, {len(exact_ids)} exact ids")
    return exact_ids, lsh


def leak_ok(doc: dict, exact_ids: set[str], lsh, ngram: int) -> bool:
    text = doc.get("text", "")
    ids = extract_ids(text)
    if ids & exact_ids:
        return False
    mh = _minhash(text, n=ngram)
    if len(mh.digest()) == 0:
        return True
    return not lsh.query(mh)


# ---------- main processor ----------

def read_jsonl(paths: list[Path]) -> Iterator[dict]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:  # noqa: BLE001
                    continue


def process(
    inputs: list[Path],
    out_file: Path,
    config: dict[str, Any],
    report_file: Path,
) -> dict[str, int]:
    from datasketch import MinHashLSH

    lp = config.get("leak_protection", {}) or {}
    ngram = int(lp.get("ngram", 13))
    near_dup_threshold = float(lp.get("near_dup_threshold", 0.8))

    exact_ids, leak_lsh = build_leak_index(config)

    counters = Counter(
        total=0, dropped_quality=0, dropped_leak_exact=0, dropped_leak_near=0,
        dropped_near_dup=0, kept=0,
    )
    per_source: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    near_dup_lsh = MinHashLSH(threshold=near_dup_threshold, num_perm=128)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as out:
        for doc in read_jsonl(inputs):
            counters["total"] += 1
            text = doc.get("text", "")
            source = (doc.get("meta") or {}).get("source", "unknown")
            if not quality_ok(text):
                counters["dropped_quality"] += 1
                continue
            # exact-id leak
            ids = extract_ids(text)
            if ids & exact_ids:
                counters["dropped_leak_exact"] += 1
                continue
            # near-dup leak (benchmark)
            mh = _minhash(text, n=ngram)
            if len(mh.digest()) > 0 and leak_lsh.query(mh):
                counters["dropped_leak_near"] += 1
                continue
            # content hash (exact dupe) fast path
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                counters["dropped_near_dup"] += 1
                continue
            # near-dup within corpus
            key = f"d_{counters['kept']}"
            if len(mh.digest()) > 0 and near_dup_lsh.query(mh):
                counters["dropped_near_dup"] += 1
                continue
            if len(mh.digest()) > 0:
                try:
                    near_dup_lsh.insert(key, mh)
                except ValueError:
                    pass
            seen_hashes.add(h)
            counters["kept"] += 1
            per_source[source] += 1
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    report = {
        "counters": dict(counters),
        "per_source_kept": dict(per_source),
        "output": str(out_file),
    }
    report_file.write_text(json.dumps(report, indent=2))
    return report


def approx_tokens(jsonl_path: Path, bytes_per_token: float = 4.0) -> int:
    """Quick token-count estimate without loading a tokenizer (bytes / 4).

    For an accurate count, run with the Llama tokenizer; this estimate is
    within ~15% of ground truth for English-heavy technical text and is
    adequate for compute planning.
    """
    total_bytes = sum(len(line.encode("utf-8")) for line in jsonl_path.open("r", encoding="utf-8"))
    return int(total_bytes / bytes_per_token)
