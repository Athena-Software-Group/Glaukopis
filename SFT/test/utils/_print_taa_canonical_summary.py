#!/usr/bin/env python3
"""Print a single TAA-Canonical comparison table across N models.

Walks ``responses/<display_name>/summary_*.json`` for each display name
on argv, isolates the ``athena-taa-canonical`` task entry (if present),
and emits a Markdown table with the canonical headline metrics
(accuracy, plausible_accuracy, combined_accuracy, f1, plausible_f1,
combined_f1, parse_error_pct) per model.

Invoked by run_taa_canonical_baselines.sh at end-of-sweep. Read-only
with respect to the per-suite summary files; safe to re-run.

Usage (must run from SFT/test/ so the default responses-root resolves):
    cd /root/Glaukopis/SFT/test
    python utils/_print_taa_canonical_summary.py <display_1> [<display_2> ...] \\
        [--responses-root DIR] [--out-md PATH] [--out-json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_HEADLINE_KEYS = (
    "accuracy",
    "plausible_accuracy",
    "combined_accuracy",
    "f1",
    "plausible_f1",
    "combined_f1",
    "parse_error_pct",
)


def _fmt_pct(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{float(v):.2f}%"
    s = str(v).strip()
    return s if s else "-"


def _find_canonical_row(model_dir: Path) -> tuple[dict, Path] | tuple[None, None]:
    """Return (task_dict, source_summary_path) for athena-taa-canonical.

    Iterates every summary_*.json under model_dir (skips summary_model.json
    written by _print_model_summary.py to avoid double-counting), and picks
    the entry with the latest `finished` ISO timestamp when a model has
    multiple summaries that include the task (typical after a re-bench).
    Returns (None, None) when no summary contains the task.
    """
    if not model_dir.is_dir():
        return None, None
    best: tuple[str, float, dict, Path] | None = None
    for path in sorted(model_dir.glob("summary_*.json")):
        if path.name == "summary_model.json":
            continue
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        finished = str(s.get("finished") or "")
        mtime = path.stat().st_mtime
        for t in (s.get("tasks") or []):
            if t.get("task") == "athena-taa-canonical":
                cand = (finished, mtime, t, path)
                if best is None or cand[:2] > best[:2]:
                    best = cand
    if best is None:
        return None, None
    return best[2], best[3]


def _row(display: str, task: dict | None, source: Path | None) -> str:
    if task is None:
        cells = [display, "-", "-"] + ["-"] * len(_HEADLINE_KEYS) + ["(no canonical row found)"]
        return "| " + " | ".join(cells) + " |"
    metrics = task.get("metrics") or {}
    cells = [
        display,
        str(task.get("rows", "-")),
        str(task.get("exit", "-")),
    ]
    for k in _HEADLINE_KEYS:
        cells.append(_fmt_pct(metrics.get(k)))
    cells.append(source.name if source else "-")
    return "| " + " | ".join(cells) + " |"


def _build(display_names: list[str], responses_root: Path) -> tuple[str, list[dict]]:
    header = ["Model", "Rows", "Exit"] + list(_HEADLINE_KEYS) + ["Source"]
    align = ["---"] + ["---:"] * (len(header) - 2) + ["---"]
    lines = [
        "## TAA Canonical (athena-taa-canonical) -- per-model comparison",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
    ]
    payload: list[dict] = []
    for d in display_names:
        model_dir = responses_root / d
        task, src = _find_canonical_row(model_dir)
        lines.append(_row(d, task, src))
        payload.append({
            "display_name": d,
            "found": task is not None,
            "source_summary": (src.name if src else None),
            "rows": task.get("rows") if task else None,
            "exit": task.get("exit") if task else None,
            "metrics": (task.get("metrics") if task else None),
        })
    return "\n".join(lines) + "\n", payload


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("display_names", nargs="+")
    p.add_argument("--responses-root", default="responses")
    p.add_argument("--out-md", default=None)
    p.add_argument("--out-json", default=None)
    args = p.parse_args(argv)

    responses_root = Path(args.responses_root)
    md, payload = _build(args.display_names, responses_root)
    print(md)

    if args.out_md:
        try:
            Path(args.out_md).write_text(md, encoding="utf-8")
            print(f"wrote {args.out_md}")
        except OSError as e:
            print(f"[taa-canonical-summary] WARN: failed to write {args.out_md}: {e}", file=sys.stderr)
    if args.out_json:
        try:
            Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {args.out_json}")
        except OSError as e:
            print(f"[taa-canonical-summary] WARN: failed to write {args.out_json}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
