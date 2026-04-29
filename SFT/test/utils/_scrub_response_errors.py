#!/usr/bin/env python3
"""Drop error rows from a benchmark response file in-place.

Used by run_benchmark.sh --retry-errors to surgically remove rows whose
model response is an error sentinel (e.g. "Error", "Error: ...") so the
existing per-benchmark resume logic re-processes them on the next run.

Schema by file extension:
    .jsonl  athena-*, cybersoceval-*    -- "response" field
    .tsv    cti-bench mcq/rcm/vsp/ate/taa -- "Raw_Response" column
    .csv    cybermetric                 -- "raw_response" column (empty
                                          string indicates a failed call;
                                          cybermetric.py swallows the
                                          exception and writes '' rather
                                          than a sentinel string)

Exits 0 on success (even when nothing was scrubbed). Prints a one-line
summary to stdout: "[scrub] <path>: removed N / kept M".
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


_ERROR_PREFIXES = ("Error", "error generating response", "ERROR_")


def _is_error_response(value: str) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    if s == "Error":
        return True
    return s.startswith("Error:") or s.startswith("Error generating response")


def _scrub_jsonl(path: Path) -> tuple[int, int]:
    kept_lines: list[str] = []
    removed = 0
    kept = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                # Preserve unparseable lines untouched (sanitize_jsonl
                # in the benchmark will handle them on next resume).
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                kept += 1
                continue
            resp = rec.get("response", "")
            if _is_error_response(resp):
                removed += 1
                continue
            kept_lines.append(line if line.endswith("\n") else line + "\n")
            kept += 1
    if removed:
        path.write_text("".join(kept_lines), encoding="utf-8")
    return removed, kept


def _scrub_tsv(path: Path) -> tuple[int, int]:
    rows: list[list[str]] = []
    removed = 0
    kept = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        rows.append(header)
        try:
            err_col = header.index("Raw_Response")
        except ValueError:
            return 0, len(list(reader))
        for row in reader:
            if err_col < len(row) and _is_error_response(row[err_col]):
                removed += 1
                continue
            rows.append(row)
            kept += 1
    if removed:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(rows)
    return removed, kept


def _scrub_csv_cybermetric(path: Path) -> tuple[int, int]:
    rows: list[list[str]] = []
    removed = 0
    kept = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        rows.append(header)
        try:
            raw_col = header.index("raw_response")
        except ValueError:
            return 0, len(list(reader))
        for row in reader:
            raw = row[raw_col].strip() if raw_col < len(row) else ""
            # cybermetric.py writes raw_response='' when the underlying
            # model call raises; treat that as an error to retry.
            if not raw or _is_error_response(raw):
                removed += 1
                continue
            rows.append(row)
            kept += 1
    if removed:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    return removed, kept


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: _scrub_response_errors.py <file> [<file> ...]", file=sys.stderr)
        return 2
    overall = 0
    for raw_path in argv[1:]:
        path = Path(raw_path)
        if not path.exists():
            print(f"[scrub] {path}: missing (skip)")
            continue
        ext = path.suffix.lower()
        if ext == ".jsonl":
            removed, kept = _scrub_jsonl(path)
        elif ext == ".tsv":
            removed, kept = _scrub_tsv(path)
        elif ext == ".csv":
            removed, kept = _scrub_csv_cybermetric(path)
        else:
            print(f"[scrub] {path}: unsupported extension '{ext}' (skip)")
            continue
        print(f"[scrub] {path}: removed {removed} / kept {kept}")
        overall += removed
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
