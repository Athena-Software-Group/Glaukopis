#!/usr/bin/env python3
"""Aggregate per-suite summary_*.json files for a single model into one table.

Invoked by run_foundation_8b_baselines.sh (and any other multi-suite
orchestrator) after all sub-sweeps complete. Reads every
``responses/<display_name>/summary_*.json`` produced by run_benchmark.sh's
end-of-sweep step and emits a combined Markdown table covering every
task across every suite that ran for the model.

Usage:
    _print_model_summary.py <display_name> [--responses-root DIR]
                                            [--out-md PATH] [--out-json PATH]

Exits 0 even when no per-suite summaries are found (prints a notice and
returns). The file is intentionally read-only with respect to the
per-suite artifacts so re-running it is always safe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt_elapsed(sec: int) -> str:
    h, rem = divmod(int(sec or 0), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


_PCT_KEYS = {
    "accuracy", "avg_score", "parse_error_pct",
    "plausible_accuracy", "combined_accuracy",
    "f1", "plausible_f1", "combined_f1",
    "MCQ", "MCQ3K",
}
_HIDDEN_KEYS = {
    "correct_mc_count", "incorrect_mc_count", "response_parsing_error_count",
}


def _fmt_metrics(d: dict | None) -> str:
    if not d:
        return "-"
    parts = []
    for k, v in d.items():
        if isinstance(v, dict) or k in _HIDDEN_KEYS:
            continue
        if isinstance(v, float):
            if k in _PCT_KEYS:
                parts.append(f"{k}: {v:.2f}%")
            else:
                parts.append(f"{k}: {v:.4f}")
        else:
            parts.append(f"{k}: {v}")
    return ", ".join(parts) if parts else "-"


def _suite_label(summary: dict) -> str:
    suite = summary.get("suite") or "?"
    stem = summary.get("cybermetric_stem") or ""
    if suite in ("cybermetric", "all") and stem:
        return f"{suite} ({stem})"
    return suite


def _dedupe_key(s: dict) -> tuple:
    """Group key for collapsing redundant per-suite summaries.

    Two summary files describing the same suite + same task set + same
    row counts are treated as duplicates (typical case: a stale pre-
    namespacing `summary_cybermetric_<rows>_v<V>.json` from before the
    cybermetric size was added to the filename, plus a fresh
    `summary_cybermetric_<size>_<rows>_v<V>.json` from the latest run).
    Different cybermetric sizes produce different task row counts, so
    they get distinct keys and both survive.
    """
    suite = _suite_label(s)
    tasks = tuple(sorted(
        (t.get("task", ""), int(t.get("rows", 0) or 0))
        for t in (s.get("tasks") or [])
    ))
    return (suite, tasks)


def _collect(model_dir: Path) -> list[dict]:
    if not model_dir.is_dir():
        return []
    raw: list[tuple[Path, dict]] = []
    for path in sorted(model_dir.glob("summary_*.json")):
        # Skip the model-wide aggregate this script writes; otherwise a
        # second invocation would feed its own previous output back in.
        if path.name == "summary_model.json":
            continue
        try:
            raw.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as e:
            print(f"[model-summary] WARN: skipping {path}: {e}", file=sys.stderr)

    # Dedupe by (suite, tasks-fingerprint), keep the entry with the latest
    # `finished` ISO timestamp; fall back to file mtime when missing/equal.
    best: dict[tuple, tuple[str, float, Path, dict]] = {}
    for path, s in raw:
        key = _dedupe_key(s)
        finished = str(s.get("finished") or "")
        mtime = path.stat().st_mtime
        cand = (finished, mtime, path, s)
        prev = best.get(key)
        if prev is None or cand[:2] > prev[:2]:
            best[key] = cand

    # Warn when we dropped a duplicate so a stale file isn't silently masked.
    seen_paths = {c[2] for c in best.values()}
    for path, s in raw:
        if path not in seen_paths:
            print(
                f"[model-summary] note: ignoring stale duplicate "
                f"{path.name} (superseded by a newer run for "
                f"suite='{_suite_label(s)}')",
                file=sys.stderr,
            )

    return [c[3] for c in best.values()]


def _build_md(display_name: str, summaries: list[dict]) -> tuple[str, dict]:
    total_elapsed = sum(int(s.get("elapsed_sec") or 0) for s in summaries)
    overall_exit = max((int(s.get("overall_exit") or 0) for s in summaries), default=0)
    suites = sorted({_suite_label(s) for s in summaries})

    head = [
        f"# Model summary: `{display_name}`",
        "",
        f"- suites    : {', '.join(suites) if suites else '-'}",
        f"- runs      : {len(summaries)}",
        f"- elapsed   : {_fmt_elapsed(total_elapsed)} (sum of per-suite wall-clock)",
        f"- worst exit: {overall_exit}",
        "",
        "| Suite | Task | Rows | Elapsed | Exit | Metrics |",
        "|---|---|---:|---:|---:|---|",
    ]
    flat_rows: list[dict] = []
    for s in summaries:
        suite = _suite_label(s)
        for t in s.get("tasks", []) or []:
            head.append(
                f"| {suite} | {t.get('task','')} | {t.get('rows',0)} "
                f"| {_fmt_elapsed(int(t.get('elapsed_sec') or 0))} "
                f"| {t.get('exit','?')} | {_fmt_metrics(t.get('metrics'))} |"
            )
            flat_rows.append({"suite": suite, **t})
    md = "\n".join(head) + "\n"
    payload = {
        "display_name": display_name,
        "suites": suites,
        "total_elapsed_sec": total_elapsed,
        "worst_exit": overall_exit,
        "tasks": flat_rows,
        "source_summaries": [s.get("suite", "?") for s in summaries],
    }
    return md, payload


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("display_name")
    p.add_argument("--responses-root", default="responses")
    p.add_argument("--out-md", default=None)
    p.add_argument("--out-json", default=None)
    args = p.parse_args(argv)

    model_dir = Path(args.responses_root) / args.display_name
    summaries = _collect(model_dir)
    if not summaries:
        print(f"[model-summary] no summary_*.json under {model_dir}; nothing to aggregate")
        return 0

    md, payload = _build_md(args.display_name, summaries)
    print("=== Model-wide benchmark summary ===")
    print(md)

    out_md = Path(args.out_md or model_dir / "summary_model.md")
    out_json = Path(args.out_json or model_dir / "summary_model.json")
    try:
        out_md.write_text(md, encoding="utf-8")
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_md}")
        print(f"wrote {out_json}")
    except OSError as e:
        print(f"[model-summary] WARN: failed to write artifacts: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
