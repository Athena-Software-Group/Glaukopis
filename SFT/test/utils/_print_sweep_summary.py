#!/usr/bin/env python3
"""Render an end-of-sweep summary table for run_benchmark.sh.

Consumes per-task data exported by the parent shell via RB_* env vars,
prints a Markdown table to stdout (so it lands in the tee'd sweep log),
and persists the same data as JSON + Markdown in the model's response
directory.

Not meant to be run directly; invoked by run_benchmark.sh at end-of-sweep.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

US = "\x1f"  # ASCII unit separator used by the shell to join array elements


def _split(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw:
        return []
    return raw.split(US)


def _fmt_elapsed(sec: int) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _parse_metrics(raw: str) -> dict | None:
    """inference.py prints e.g. `{'accuracy': '78.42%'}` -- literal_eval it."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        val = ast.literal_eval(raw)
        return val if isinstance(val, dict) else {"value": val}
    except (ValueError, SyntaxError):
        return {"raw": raw}


# Headline keys we know are percentages (formatted with a trailing %). All
# other floats get a plain :.2f. Percentages are emitted as 18.78% rather
# than 18.7793 so the table cell is human-scannable.
_PCT_KEYS = {
    "accuracy", "avg_score", "parse_error_pct",
    "plausible_accuracy", "combined_accuracy",
    "f1", "plausible_f1", "combined_f1",
    "MCQ", "MCQ3K",
}

# Keys we drop from the table cell because they're either redundant with a
# percentage we already show or just noise in the headline (the per-slice
# breakdowns get their own sub-tables under the main table).
_HIDDEN_KEYS = {
    "correct_mc_count",
    "incorrect_mc_count",
    "response_parsing_error_count",
}


def _fmt_scalar(k: str, v) -> str:
    if isinstance(v, float):
        if k in _PCT_KEYS:
            return f"{k}: {v:.2f}%"
        return f"{k}: {v:.4f}"
    return f"{k}: {v}"


def _fmt_metrics(d: dict | None) -> str:
    """Render the headline metrics cell -- scalars only.

    Nested dict values (e.g. per_topic, per_source) are rendered as
    separate Markdown sub-tables by ``_render_slice_tables``.
    """
    if not d:
        return "-"
    parts = []
    for k, v in d.items():
        if isinstance(v, dict):
            continue
        if k in _HIDDEN_KEYS:
            continue
        parts.append(_fmt_scalar(k, v))
    return ", ".join(parts) if parts else "-"


def _render_slice_tables(rows_data: list[dict]) -> str:
    """Emit per-task per-slice Markdown sub-tables.

    For each task whose metrics dict contains nested per-* dicts, render
    one sub-table per slice key showing N / Jaccard / Strict acc / Parse
    err per slice value, sorted by N descending so the largest slices
    surface first. Returns "" when no task has any nested per-* metrics.
    """
    blocks: list[str] = []
    for r in rows_data:
        m = r.get("metrics") or {}
        per_keys = [k for k, v in m.items() if isinstance(v, dict) and k.startswith("per_")]
        if not per_keys:
            continue
        blocks.append(f"### `{r['task']}` per-slice breakdown")
        for pk in per_keys:
            slice_label = pk[len("per_"):]
            slice_dict = m[pk]
            ordered = sorted(
                slice_dict.items(),
                key=lambda kv: -(int(kv[1].get("answered", 0)) + int(kv[1].get("parse_errors", 0))),
            )
            blocks.append(f"\n**per_{slice_label}** (sorted by N desc)")
            blocks.append(
                f"| {slice_label} | N | Jaccard | Strict acc | Parse err |"
            )
            blocks.append("|---|---:|---:|---:|---:|")
            for name, vals in ordered:
                answered = int(vals.get("answered", 0))
                parse_err = int(vals.get("parse_errors", 0))
                n = answered + parse_err
                jacc = float(vals.get("avg_score", 0.0))
                acc = float(vals.get("correct_mc_pct", 0.0))
                blocks.append(
                    f"| {name} | {n} | {jacc:.2f}% | {acc:.2f}% | {parse_err} |"
                )
        blocks.append("")
    return "\n".join(blocks)


def main() -> int:
    tasks = _split("RB_RES_TASKS")
    if not tasks:
        print("[summary] no per-task results captured; skipping summary")
        return 0

    elapsed = _split("RB_RES_ELAPSED")
    exits = _split("RB_RES_EXIT")
    metrics_raw = _split("RB_RES_METRICS")
    rows = _split("RB_RES_ROWS")
    started = _split("RB_RES_STARTED")
    finished = _split("RB_RES_FINISHED")

    rows_data = []
    for i, task in enumerate(tasks):
        rows_data.append({
            "task": task,
            "rows": int(rows[i]) if i < len(rows) and rows[i].isdigit() else 0,
            "elapsed_sec": int(elapsed[i]) if i < len(elapsed) and elapsed[i].isdigit() else 0,
            "exit": int(exits[i]) if i < len(exits) and exits[i].lstrip("-").isdigit() else -1,
            "metrics": _parse_metrics(metrics_raw[i] if i < len(metrics_raw) else ""),
            "started": started[i] if i < len(started) else "",
            "finished": finished[i] if i < len(finished) else "",
        })

    summary = {
        "model": os.environ.get("RB_MODEL", ""),
        "display_name": os.environ.get("RB_DISPLAY", ""),
        "suite": os.environ.get("RB_SUITE", ""),
        "version": int(os.environ.get("RB_VERSION", "1") or 1),
        "rows_filter": os.environ.get("RB_ROWS_STR", "all"),
        "batch": int(os.environ["RB_BATCH"]) if os.environ.get("RB_BATCH") else None,
        "tasks_requested": os.environ.get("RB_TASKS_REQUESTED", "").split(),
        "cybermetric_stem": os.environ.get("RB_CYBERMETRIC_STEM", ""),
        "env": os.environ.get("RB_ENV_NAME", ""),
        "started": os.environ.get("RB_STARTED", ""),
        "finished": os.environ.get("RB_FINISHED", ""),
        "elapsed_sec": int(os.environ.get("RB_ELAPSED", "0") or 0),
        "overall_exit": int(os.environ.get("RB_OVERALL_EXIT", "0") or 0),
        "log_file": os.environ.get("RB_LOG_FILE", ""),
        "tasks": rows_data,
    }

    # --- Markdown table -----------------------------------------------------
    header_lines = [
        f"## Sweep summary: `{summary['model']}`",
        "",
        f"- display name : `{summary['display_name']}`",
        f"- suite        : {summary['suite'] or '-'}",
        f"- version      : {summary['version']}",
        f"- rows filter  : {summary['rows_filter']}",
        f"- batch        : {summary['batch'] if summary['batch'] is not None else '-'}",
        f"- env          : {summary['env'] or '-'}",
    ]
    if summary["suite"] in ("cybermetric", "all") and summary["cybermetric_stem"]:
        header_lines.append(f"- cybermetric  : {summary['cybermetric_stem']}")
    header_lines += [
        f"- started      : {summary['started']}",
        f"- finished     : {summary['finished']}",
        f"- elapsed      : {_fmt_elapsed(summary['elapsed_sec'])}",
        f"- overall exit : {summary['overall_exit']}",
    ]
    header = "\n".join(header_lines) + "\n"

    table_rows = [
        "| Task | Rows | Elapsed | Exit | Metrics |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows_data:
        table_rows.append(
            f"| {r['task']} | {r['rows']} | {_fmt_elapsed(r['elapsed_sec'])} "
            f"| {r['exit']} | {_fmt_metrics(r['metrics'])} |"
        )
    md = header + "\n" + "\n".join(table_rows) + "\n"
    slice_md = _render_slice_tables(rows_data)
    if slice_md:
        md += "\n" + slice_md + "\n"

    # --- stdout (goes to the tee'd log too) ---------------------------------
    print("=== Sweep results ===")
    print(md)

    # --- write artifacts next to the responses ------------------------------
    try:
        Path(os.environ["RB_SUMMARY_JSON"]).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        Path(os.environ["RB_SUMMARY_MD"]).write_text(md, encoding="utf-8")
        print(f"wrote {os.environ['RB_SUMMARY_JSON']}")
        print(f"wrote {os.environ['RB_SUMMARY_MD']}")
    except OSError as e:
        print(f"[summary] WARN: failed to write summary files: {e}", file=sys.stderr)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
