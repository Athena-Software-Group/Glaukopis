#!/usr/bin/env python
"""Check that a finished scoring run is complete and was asked the same questions as the published run.

The scorer logs a failed suite as a warning and still exits 0, so a clean exit proves nothing.

Usage: check_scores.py ALIAS [--no-mmlu] [--no-compare]
       check_scores.py --build-manifest PUBLISHED_DIR PUBLISHED_ALIAS OUT_JSON

Every response and scored file of the published Recal-32B run must exist for ALIAS with the same row count,
summary_model.json must report worst_exit 0, and at most 1% of the rows in a file may have an empty response.
Unless --no-compare is given, each item's id, prompt and answer key must also match the published run exactly.
Reads expected_rows.json and published_items.json from this script's folder. Exits 0 only if all of that holds.
"""
import csv
import hashlib
import json
import os
import sys

csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Glaukopis/SFT/eval/responses")
MAX_EMPTY = 0.01


def read(path):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def empties(rows):
    n = 0
    for r in rows:
        key = "response" if "response" in r else "raw_response" if "raw_response" in r else None
        if key and not (r[key] or "").strip():
            n += 1
    return n


def item_key(rel, r):
    if rel.endswith(".jsonl"):
        return [r.get("id"), r.get("prompt"), r.get("answer")]
    if rel.startswith("mmlu-pro/"):
        return [r.get("question_id"), r.get("question"), r.get("ground_truth")]
    return [r.get("question"), r.get("answer")]


def digests(rel, rows):
    return sorted(hashlib.sha256(json.dumps(item_key(rel, r), sort_keys=True).encode()).hexdigest() for r in rows)


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def build_manifest(published, alias, out):
    manifest = {}
    for rel in load("expected_rows.json"):
        if "_response." in rel:
            manifest[rel] = digests(rel, read(os.path.join(published, rel.replace("{alias}", alias))))
    with open(out, "w") as f:
        json.dump(manifest, f)
    print(len(manifest), "files,", sum(len(v) for v in manifest.values()), "items")


def main():
    if sys.argv[1] == "--build-manifest":
        build_manifest(*sys.argv[2:5])
        return
    alias = sys.argv[1]
    no_mmlu = "--no-mmlu" in sys.argv[2:]
    compare = "--no-compare" not in sys.argv[2:]
    base = os.path.join(ROOT, alias)
    published = load("published_items.json") if compare else {}
    problems, checked, rows_total, compared = [], 0, 0, 0
    for rel, want in sorted(load("expected_rows.json").items()):
        if no_mmlu and rel.startswith("mmlu-pro/"):
            continue
        path = os.path.join(base, rel.replace("{alias}", alias))
        if not os.path.exists(path):
            problems.append(f"missing {rel}")
            continue
        rows = read(path)
        got, empty = len(rows), empties(rows)
        checked += 1
        rows_total += got
        if got != want:
            problems.append(f"{rel}: {got} rows, expected {want}")
        if got and empty / got > MAX_EMPTY:
            problems.append(f"{rel}: {empty} of {got} responses are empty")
        if rel in published:
            mine, theirs = digests(rel, rows), published[rel]
            different = len(theirs) - sum(1 for a, b in zip(mine, theirs) if a == b) if len(mine) == len(theirs) else -1
            if mine != theirs:
                problems.append(f"{rel}: questions differ from the published run ({different} sorted positions, -1 = counts differ)")
            compared += got
    summary = os.path.join(base, "summary_model.json")
    elapsed = None
    if not os.path.exists(summary):
        problems.append("missing summary_model.json")
    else:
        with open(summary) as f:
            s = json.load(f)
        elapsed = s.get("total_elapsed_sec")
        if s.get("worst_exit") != 0:
            problems.append(f"worst_exit={s.get('worst_exit')}")
    if problems:
        print(f"FAIL {alias}: " + "; ".join(problems), flush=True)
        sys.exit(2)
    print(f"OK {alias}: {checked} files, {rows_total} rows, {compared} items identical to published, elapsed_sec={elapsed}", flush=True)


if __name__ == "__main__":
    main()
