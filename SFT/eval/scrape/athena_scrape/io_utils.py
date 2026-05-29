"""Utility helpers for reading and writing athena_scrape artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from .models import ContentRecord, ProcessedRecord, UrlRecord


RAW_URL_ROOT = Path("data/raw/urls")
RAW_CONTENT_ROOT = Path("data/raw/content")
PROCESSED_ROOT = Path("data/processed/mcq")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_url_records_csv(records: Sequence[UrlRecord], path: Path | None = None) -> Path:
    path = path or RAW_URL_ROOT / "all_urls.csv"
    _ensure_parent(path)
    fieldnames = [
        "url_id",
        "url",
        "source_type",
        "source_link",
        "collected_at",
        "published",
        "metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "url_id": record.url_id,
                    "url": record.url,
                    "source_type": record.source_type,
                    "source_link": record.source_link,
                    "collected_at": record.collected_at,
                    "published": record.published or "",
                    "metadata": json.dumps(record.metadata, ensure_ascii=False),
                }
            )
    return path


def read_url_records_csv(path: Path) -> List[UrlRecord]:
    records: List[UrlRecord] = []
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            metadata = json.loads(row.get("metadata", "{}")) if row.get("metadata") else {}
            records.append(
                UrlRecord(
                    url_id=row["url_id"],
                    url=row["url"],
                    source_type=row["source_type"],
                    source_link=row["source_link"],
                    collected_at=row["collected_at"],
                    published=row.get("published") or None,
                    metadata=metadata,
                )
            )
    return records


def write_content_records(records: Iterable[ContentRecord], path: Path | None = None) -> Path:
    path = path or RAW_CONTENT_ROOT / "content.jsonl"
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(
                json.dumps(
                    {
                        "url_id": record.url_id,
                        "url": record.url,
                        "source_type": record.source_type,
                        "fetched_at": record.fetched_at,
                        "status": record.status,
                        "content_type": record.content_type,
                        "content": record.content,
                        "metadata": record.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def read_content_records(path: Path) -> List[ContentRecord]:
    records: List[ContentRecord] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            raw = json.loads(line)
            records.append(
                ContentRecord(
                    url_id=raw["url_id"],
                    url=raw["url"],
                    source_type=raw["source_type"],
                    fetched_at=raw["fetched_at"],
                    status=int(raw["status"]),
                    content_type=raw.get("content_type", ""),
                    content=raw.get("content", ""),
                    metadata=raw.get("metadata", {}),
                )
            )
    return records


def write_processed_records(
    source_type: str,
    records: Iterable[ProcessedRecord],
    root: Path | None = None,
) -> Path:
    root = root or PROCESSED_ROOT
    path = root / f"{source_type}.jsonl"
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(
                json.dumps(
                    {
                        "url_id": record.url_id,
                        "url": record.url,
                        "source_type": record.source_type,
                        "processed_at": record.processed_at,
                        "payload": record.payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


