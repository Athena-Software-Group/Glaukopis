"""Utilities for planning MCQ question generation."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(slots=True)
class MCQEntry:
    url_id: str
    url: str
    source_type: str
    raw_path: Path
    processed_path: Path
    char_count: int
    question_count: int = 0


SUPPORTED_SOURCES = {
    "mitre_attack",
    "capec_catalog",
    "cwe_catalog",
    "cisa_ics",
    "cisa_csa",
}


def _load_url_metadata(csv_path: Path) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url_id = row["url_id"]
            data[url_id] = row
    return data


def _build_entries(
    url_meta: Dict[str, dict],
    processed_root: Path,
    raw_root: Path,
    extra_sources: Sequence[str] | None = None,
) -> Dict[str, List[MCQEntry]]:
    processed_root = processed_root.resolve()
    raw_root = raw_root.resolve()
    entries: Dict[str, List[MCQEntry]] = {source: [] for source in SUPPORTED_SOURCES}
    if extra_sources:
        for source in extra_sources:
            entries.setdefault(source, [])

    for source_dir in processed_root.iterdir():
        if not source_dir.is_dir():
            continue
        source_type = source_dir.name
        entry_bucket = entries.setdefault(source_type, [])
        for processed_file in source_dir.glob("*.txt"):
            url_id = processed_file.stem
            meta = url_meta.get(url_id)
            url = meta.get("url") if meta else ""
            raw_file = raw_root / source_type / f"{url_id}.txt"
            try:
                text = processed_file.read_text(encoding="utf-8")
            except Exception:
                continue
            char_count = len(text)
            entry_bucket.append(
                MCQEntry(
                    url_id=url_id,
                    url=url,
                    source_type=source_type,
                    raw_path=raw_file,
                    processed_path=processed_file,
                    char_count=char_count,
                )
            )
    return entries


def _assign_three_tiers(entries: List[MCQEntry], low_ratio: float, high_ratio: float, values: Tuple[int, int, int]) -> None:
    if not entries:
        return
    entries_sorted = sorted(entries, key=lambda e: e.char_count)
    n = len(entries_sorted)
    low_count = max(1, int(n * low_ratio)) if n > 1 else n
    high_count = max(1, int(n * high_ratio)) if n > 1 else 0
    if low_count + high_count > n:
        high_count = max(0, n - low_count)
    mid_count = n - low_count - high_count

    for idx, entry in enumerate(entries_sorted):
        if idx < low_count:
            entry.question_count = values[0]
        elif idx >= n - high_count:
            entry.question_count = values[2]
        else:
            entry.question_count = values[1]

    if mid_count <= 0 and high_count == 0:
        # When all entries fall into the low tier, ensure at least the first receives the mid value.
        entries_sorted[0].question_count = values[1]


def _assign_two_tiers(entries: List[MCQEntry], split_ratio: float, values: Tuple[int, int]) -> None:
    if not entries:
        return
    entries_sorted = sorted(entries, key=lambda e: e.char_count)
    n = len(entries_sorted)
    low_count = int(n * split_ratio)
    for idx, entry in enumerate(entries_sorted):
        entry.question_count = values[0] if idx < low_count else values[1]
    if n == 1:
        entries_sorted[0].question_count = values[0]


def _ensure_counts(entries: Iterable[MCQEntry], default: int = 1) -> None:
    for entry in entries:
        if entry.question_count <= 0:
            entry.question_count = default


def build_mcq_plan(
    *,
    url_csv: Path,
    processed_root: Path,
    raw_root: Path,
    output_csv: Path,
    random_seed: int = 42,
) -> Dict[str, int]:
    url_meta = _load_url_metadata(url_csv)
    entries_by_source = _build_entries(url_meta, processed_root, raw_root)

    rng = random.Random(random_seed)

    selected: List[MCQEntry] = []
    summary: Dict[str, int] = {}

    # MITRE ATT&CK
    mitre_entries = entries_by_source.get("mitre_attack", [])
    mitre_sample = mitre_entries if len(mitre_entries) <= 1500 else rng.sample(mitre_entries, 1500)
    _assign_three_tiers(mitre_sample, 0.25, 0.25, (1, 2, 3))
    selected.extend(mitre_sample)
    summary["mitre_attack"] = sum(e.question_count for e in mitre_sample)

    # CAPEC + CWE combined sample of 500 URLs
    capec_entries = entries_by_source.get("capec_catalog", [])
    cwe_entries = entries_by_source.get("cwe_catalog", [])
    combined_capec_cwe = capec_entries + cwe_entries
    capec_cwe_sample = (
        combined_capec_cwe
        if len(combined_capec_cwe) <= 500
        else rng.sample(combined_capec_cwe, 500)
    )
    _assign_two_tiers(capec_cwe_sample, 0.5, (1, 2))
    selected.extend(capec_cwe_sample)
    summary["capec_catalog"] = sum(e.question_count for e in capec_cwe_sample if e.source_type == "capec_catalog")
    summary["cwe_catalog"] = sum(e.question_count for e in capec_cwe_sample if e.source_type == "cwe_catalog")

    # CISA advisories
    cisa_entries = entries_by_source.get("cisa_ics", []) + entries_by_source.get("cisa_csa", [])
    cisa_sample = cisa_entries if len(cisa_entries) <= 500 else rng.sample(cisa_entries, 500)
    _assign_two_tiers(cisa_sample, 0.5, (1, 2))
    selected.extend(cisa_sample)
    summary["cisa"] = sum(e.question_count for e in cisa_sample)

    # Others – any remaining directories
    reserved_sources = {"mitre_attack", "capec_catalog", "cwe_catalog", "cisa_ics", "cisa_csa"}
    others_entries: List[MCQEntry] = []
    for source, entries in entries_by_source.items():
        if source not in reserved_sources:
            others_entries.extend(entries)
    others_sample = others_entries  # use all available
    if others_sample:
        _assign_three_tiers(others_sample, 0.25, 0.25, (3, 4, 5))
        selected.extend(others_sample)
        summary["others"] = sum(e.question_count for e in others_sample)

    _ensure_counts(selected)

    processed_root = processed_root.resolve()
    raw_root = raw_root.resolve()
    data_root = processed_root.parents[2] if len(processed_root.parents) >= 3 else processed_root

    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "url_id",
                "url",
                "source_type",
                "processed_path",
                "raw_path",
                "char_count",
                "question_count",
            ]
        )
        for entry in selected:
            writer.writerow(
                [
                    entry.url_id,
                    entry.url,
                    entry.source_type,
                    str(entry.processed_path.relative_to(processed_root)),
                    str((Path('data') / entry.raw_path.relative_to(data_root)).as_posix()) if entry.raw_path.exists() else "",
                    entry.char_count,
                    entry.question_count,
                ]
            )

    summary["total_questions"] = sum(e.question_count for e in selected)
    summary["total_urls"] = len(selected)
    return summary


