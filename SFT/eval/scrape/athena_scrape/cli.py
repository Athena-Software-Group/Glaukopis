"""Command line interface for athena_scrape."""
from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import RAW_CONTENT_ROOT, RAW_URL_ROOT
from .pipeline import build_mcq_corpus, collect_urls, process_content, scrape_urls
from .mcq_planner import build_mcq_plan
from .sources import available_sources


def _add_collect_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("collect", help="Collect URLs and metadata from CTI sources.")
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=available_sources(),
        help="Subset of sources to collect. Defaults to all registered sources.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_URL_ROOT / "all_urls.csv",
        help="Where to write the consolidated CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit applied to non-MITRE sources.",
    )
    parser.add_argument("--mitre-max-pages", type=int, default=200_000, help="Max MITRE pages to traverse.")
    parser.add_argument("--mitre-sleep", type=float, default=0.2, help="Delay between MITRE requests.")
    parser.add_argument("--mitre-timeout", type=float, default=30.0, help="Timeout for MITRE requests.")
    return parser


def _add_scrape_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("scrape", help="Fetch raw content for collected URLs.")
    parser.add_argument(
        "--url-csv",
        type=Path,
        default=RAW_URL_ROOT / "all_urls.csv",
        help="Path to the CSV built by the collect command.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_CONTENT_ROOT / "content.jsonl",
        help="Where to store the scraped content JSONL.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of URLs to scrape.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds).")
    parser.add_argument(
        "--confirm-scrape",
        action="store_true",
        help="Acknowledge that network scraping will be performed.",
    )
    return parser


def _add_process_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("process", help="Post-process raw content for the MCQ task.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_CONTENT_ROOT / "content.jsonl",
        help="Raw content JSONL produced by the scrape command.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional root directory for processed outputs (defaults to data/processed/mcq).",
    )
    return parser



def _add_build_corpus_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("build-corpus", help="Download raw text and markdown summaries for MCQ data.")
    parser.add_argument("--url-csv", type=Path, default=RAW_URL_ROOT / "all_urls.csv", help="CSV produced by the collect command.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mcq_data"), help="Directory for raw plain text outputs.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mcq"), help="Directory for markdown outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on processed URLs.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds).")
    return parser


def _add_plan_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("plan-mcq", help="Plan MCQ question counts per URL.")
    parser.add_argument("--url-csv", type=Path, default=RAW_URL_ROOT / "all_urls.csv", help="CSV produced by the collect command.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mcq"), help="Directory containing processed MCQ text files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mcq_data"), help="Directory containing raw MCQ text files.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/mcq_plan.tsv"), help="Where to write the MCQ plan TSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling URLs.")
    return parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for preparing MCQ task corpora.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_collect_parser(subparsers)
    _add_scrape_parser(subparsers)
    _add_process_parser(subparsers)
    _add_build_corpus_parser(subparsers)
    _add_plan_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        records = collect_urls(
            sources=args.sources,
            output=args.out,
            mitre_max_pages=args.mitre_max_pages,
            mitre_sleep=args.mitre_sleep,
            mitre_timeout=args.mitre_timeout,
            generic_limit=args.limit,
        )
        print(f"Collected {len(records)} unique URLs across {len(args.sources or available_sources())} sources.")
        print(f"Saved metadata to {args.out}")
        return 0

    if args.command == "scrape":
        if not args.confirm_scrape:
            parser.error("Refusing to scrape without --confirm-scrape acknowledgement.")
        records = scrape_urls(
            url_csv=args.url_csv,
            output=args.out,
            limit=args.limit,
            timeout=args.timeout,
        )
        print(f"Fetched {len(records)} pages. Raw content written to {args.out}")
        return 0

    if args.command == "process":
        buckets = process_content(raw_path=args.raw, output_root=args.out)
        total = sum(len(items) for items in buckets.values())
        print(f"Processed {total} records across {len(buckets)} source buckets.")
        return 0

    if args.command == "build-corpus":
        stats = build_mcq_corpus(
            url_csv=args.url_csv,
            raw_root=args.raw_dir,
            processed_root=args.processed_dir,
            limit=args.limit,
            timeout=args.timeout,
        )
        print(
            f"Wrote raw text files to {args.raw_dir} and markdown summaries to {args.processed_dir}."
        )
        print(
            f"Processed {stats['processed']} URLs; skipped {stats['skipped']} due to fetch issues."
        )
        return 0

    if args.command == "plan-mcq":
        stats = build_mcq_plan(
            url_csv=args.url_csv,
            processed_root=args.processed_dir,
            raw_root=args.raw_dir,
            output_csv=args.out,
            random_seed=args.seed,
        )
        print(
            f"Planned MCQ generation for {stats['total_urls']} URLs (total questions: {stats['total_questions']})."
        )
        for key, value in stats.items():
            if key not in {"total_urls", "total_questions"}:
                print(f"  {key}: {value}")
        print(f"Plan written to {args.out}")
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())





