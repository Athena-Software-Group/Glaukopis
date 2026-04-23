#!/usr/bin/env python3
"""End-to-end driver: fetch all enabled sources, parse, process, emit corpus.

Output layout:
  cpt/cache/raw/<source>/*          raw fetched artifacts
  cpt/cache/parsed/<source>.jsonl   per-source parsed docs
  cpt/corpus/<name>.jsonl           merged+filtered corpus
  cpt/cache/leak_report.json        drop counts per filter
  cpt/cache/build_report.json       per-source token / doc stats

Usage:
  python cpt/build_corpus.py --out cpt/corpus --name cti_corpus_v1
  python cpt/build_corpus.py --source mitre_attack_enterprise --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
PARSED_DIR = CACHE_DIR / "parsed"
DEFAULT_CORPUS_DIR = SCRIPT_DIR / "corpus"

sys.path.insert(0, str(SCRIPT_DIR))
import fetch  # noqa: E402
import parse  # noqa: E402
import process  # noqa: E402


def build(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    sources = cfg.get("sources", {})

    if args.source:
        requested = [s.strip() for s in args.source.split(",") if s.strip()]
        unknown = [s for s in requested if s not in sources]
        if unknown:
            print(f"[build] unknown source(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        targets = {n: sources[n] for n in requested}
    else:
        targets = {n: s for n, s in sources.items() if s.get("enabled", False)}

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_files: list[Path] = []
    per_source_docs: dict[str, int] = {}

    for name, spec in targets.items():
        print(f"\n=== {name} ===")
        try:
            raw_paths = fetch.fetch_source(name, spec, force=args.force)
        except Exception as e:  # noqa: BLE001
            print(f"[build:{name}] fetch ERROR: {e}", file=sys.stderr)
            continue
        if not raw_paths:
            print(f"[build:{name}] no raw files; skipping parse")
            continue

        parsed_path = PARSED_DIR / f"{name}.jsonl"
        # Per-source leak policy stamped onto each doc's meta so process.py
        # can vary behavior without re-reading sources.yaml (cve-family: true,
        # structural taxonomies: false, inherit top-level default otherwise).
        drop_policy = spec.get("drop_on_exact_id", None)
        count = 0
        dropped_since = 0
        with parsed_path.open("w", encoding="utf-8") as out:
            for doc in parse.parse_source(name, spec, raw_paths):
                if drop_policy is not None:
                    doc.setdefault("meta", {})["drop_on_exact_id"] = bool(drop_policy)
                # Recency cutoff: skip docs whose dated content almost certainly
                # already exists in the base model's pretrain. Only dated docs
                # are eligible for filtering; undated (HTML/Sigma) pass through.
                if args.since:
                    d = (doc.get("meta") or {}).get("date", "")
                    if d and d < args.since:
                        dropped_since += 1
                        continue
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
        per_source_docs[name] = count
        parsed_files.append(parsed_path)
        extra = f" (dropped {dropped_since} pre-{args.since})" if args.since else ""
        print(f"[build:{name}] parsed {count} docs -> {parsed_path}{extra}")

    if args.fetch_parse_only:
        print("\n[build] --fetch-parse-only: stopping before processing")
        return 0

    if not parsed_files:
        print("[build] no parsed files; nothing to process", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = out_dir / f"{args.name}.jsonl"
    leak_report = CACHE_DIR / "leak_report.json"

    print("\n=== processing (quality + leak + dedupe) ===")
    report = process.process(parsed_files, corpus_file, cfg, leak_report)
    approx_tok = process.approx_tokens(corpus_file)

    build_report = {
        "corpus": str(corpus_file),
        "per_source_parsed": per_source_docs,
        "process_counters": report["counters"],
        "per_source_kept": report["per_source_kept"],
        "approx_tokens": approx_tok,
        "approx_mb": round(corpus_file.stat().st_size / 1_000_000, 2),
    }
    (CACHE_DIR / "build_report.json").write_text(json.dumps(build_report, indent=2))

    print("\n=== build summary ===")
    print(json.dumps(build_report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the CTI CPT corpus end-to-end.")
    p.add_argument("--config", default=str(SCRIPT_DIR / "sources.yaml"))
    p.add_argument("--out", default=str(DEFAULT_CORPUS_DIR), help="Output dir for the final corpus JSONL")
    p.add_argument("--name", default="cti_corpus_v1", help="Corpus name (filename stem)")
    p.add_argument("--source", default="",
                   help="Build only the named source(s); comma-separated for multiple")
    p.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    p.add_argument("--fetch-parse-only", action="store_true",
                   help="Stop after parse (skip dedupe/leak filter); useful for incremental builds")
    p.add_argument("--since", default="",
                   help="Drop dated docs older than YYYY-MM-DD (reduces overlap with "
                        "the base model's pretrain; undated docs are unaffected). "
                        "E.g. '2023-10-01' for Llama-3.1 cutoff")
    args = p.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
