#!/usr/bin/env python3
"""Detect n-gram leakage between v8 SFT corpus rows and held-out eval sets.

Builds an n-gram fingerprint set from each held-out eval JSON/JSONL file
(athena_bench, cti_bench, cybermetricdataset, cybersoceval), then scans the
candidate Alpaca-format SFT corpus for rows whose ``input`` or ``output``
share at least --hit-threshold distinct n-grams with any single eval row.
Exits non-zero (fails loud) when any row exceeds the threshold so the v8
generation pipeline aborts before training contaminates the eval signal.

Usage:
  python dedup_against_evals.py \\
      --input  SFT/data/ift_data_2026_04_29_json_v8.json \\
      --eval-dir SFT/test/benchmark_data \\
      --n 13 --hit-threshold 1 \\
      --report SFT/data/ift_data_2026_04_29_json_v8.dedup.json

Run with --self-test for an inline check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokens(s: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(s or "")]


def ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _eval_text_from_record(rec: dict) -> str:
    parts = []
    for k in ("question", "Question", "prompt", "report", "context",
              "input", "instruction"):
        v = rec.get(k)
        if isinstance(v, str):
            parts.append(v)
    ans = rec.get("answers")
    if isinstance(ans, dict):
        parts.extend(str(v) for v in ans.values())
    elif isinstance(ans, list):
        parts.extend(str(v) for v in ans)
    return " \n ".join(parts)


def load_eval_records(path: Path) -> list[str]:
    text = path.read_text(errors="replace")
    if path.suffix == ".jsonl":
        recs = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        obj = json.loads(text)
        if isinstance(obj, dict):
            recs = obj.get("questions") or obj.get("data") or obj.get("rows") or [obj]
        elif isinstance(obj, list):
            recs = obj
        else:
            recs = []
    return [_eval_text_from_record(r) for r in recs if isinstance(r, dict)]


def build_eval_index(eval_files: Iterable[Path], n: int) -> dict[tuple[str, ...], list[str]]:
    """Map ngram -> [eval_path:row_idx, ...] for membership lookup."""
    idx: dict[tuple[str, ...], list[str]] = {}
    for f in eval_files:
        try:
            for i, txt in enumerate(load_eval_records(f)):
                key = f"{f.name}:{i}"
                for g in ngrams(tokens(txt), n):
                    idx.setdefault(g, []).append(key)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  warn: skip {f}: {e}", file=sys.stderr)
    return idx


def scan_corpus(rows: list[dict], idx: dict[tuple[str, ...], list[str]],
                n: int, hit_threshold: int) -> list[dict]:
    hits = []
    for ri, r in enumerate(rows):
        text = " \n ".join(str(r.get(k, "")) for k in ("instruction", "input", "output"))
        per_eval: dict[str, int] = {}
        for g in ngrams(tokens(text), n):
            for k in idx.get(g, ()):
                per_eval[k] = per_eval.get(k, 0) + 1
        worst = [(k, c) for k, c in per_eval.items() if c >= hit_threshold]
        if worst:
            worst.sort(key=lambda kc: -kc[1])
            hits.append({"row_index": ri, "shortname": r.get("shortname", ""),
                         "matches": worst[:5]})
    return hits


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", help="Candidate Alpaca JSON to scan.")
    p.add_argument("--eval-dir", default="SFT/test/benchmark_data")
    p.add_argument("--eval-glob", action="append", default=None,
                   help="Glob (relative to --eval-dir) restricting eval files. Repeatable.")
    p.add_argument("--n", type=int, default=13)
    p.add_argument("--hit-threshold", type=int, default=1,
                   help="Min shared n-grams between corpus row and a single eval row to flag.")
    p.add_argument("--report", help="Optional JSON path to write per-row hits.")
    p.add_argument("--max-fail", type=int, default=0,
                   help="Allow up to this many flagged rows before exit 1.")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return _self_test()
    if not args.input:
        p.error("--input is required (unless --self-test)")

    eval_root = Path(args.eval_dir)
    globs = args.eval_glob or ["**/*.jsonl", "**/*.json"]
    eval_files = sorted({f for g in globs for f in eval_root.glob(g)})
    print(f"indexing {len(eval_files)} eval files (n={args.n}) ...")
    idx = build_eval_index(eval_files, args.n)
    print(f"  {len(idx)} unique {args.n}-grams in eval index")

    rows = json.loads(Path(args.input).read_text())
    hits = scan_corpus(rows, idx, args.n, args.hit_threshold)
    print(f"scanned {len(rows)} corpus rows -> {len(hits)} flagged "
          f"(threshold={args.hit_threshold})")

    if args.report:
        Path(args.report).write_text(json.dumps(hits, indent=2))
    if len(hits) > args.max_fail:
        for h in hits[:10]:
            print("  hit:", h)
        sys.exit(1)


def _self_test():
    eval_text = "the quick brown fox jumps over the lazy dog repeatedly today"
    rows = [{"input": "irrelevant content here that does not overlap entirely",
             "output": "totally distinct"},
            {"input": eval_text, "output": ""}]
    idx = {g: ["eval:0"] for g in ngrams(tokens(eval_text), 5)}
    hits = scan_corpus(rows, idx, 5, hit_threshold=1)
    assert any(h["row_index"] == 1 for h in hits), hits
    assert not any(h["row_index"] == 0 for h in hits), hits
    print("self-test ok")


if __name__ == "__main__":
    main()
