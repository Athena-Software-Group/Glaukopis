#!/usr/bin/env python3
"""SFT run registry: record training metadata + benchmark results per run.

Storage: SFT/research/runs.jsonl, one JSON object per run keyed by run_id.
The CLI is a tiny upsert layer on top of that file -- no schema migrations,
no external deps. Designed to be run both on the training box (to register
a finished SFT run) and on whatever host has the SFT/eval sweep output
(to attach benchmark metrics). All files referenced are kept as absolute
paths so entries are portable across hosts only by the run_id.

Commands:
    register   snapshot SFT/training meta from an output dir
    attach     merge SFT/eval summary JSON(s) into a run
    show       pretty-print the registry as an ASCII table
    dump       print a single run as JSON (pretty)
    path       print the registry path

All commands accept --registry PATH to override the default location.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "SFT" / "research" / "runs.jsonl"
BENCH_RESPONSES_DIR = REPO_ROOT / "SFT" / "test" / "responses"


# ---------- registry I/O ------------------------------------------------------

def _load(path: Path) -> dict[str, dict[str, Any]]:
    """Read the JSONL registry into a run_id -> record dict."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[registry] skipping malformed line {i}: {e}", file=sys.stderr)
            continue
        rid = rec.get("run_id")
        if rid:
            out[rid] = rec
    return out


def _write(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rid in sorted(records):
            f.write(json.dumps(records[rid], separators=(",", ":")) + "\n")
    tmp.replace(path)


def _upsert(path: Path, rec: dict[str, Any]) -> None:
    records = _load(path)
    rid = rec["run_id"]
    # Shallow merge: preserve existing benchmarks if the new record doesn't
    # carry them (register -> attach flow writes different top-level keys).
    existing = records.get(rid, {})
    if "benchmarks" in existing and "benchmarks" not in rec:
        rec["benchmarks"] = existing["benchmarks"]
    records[rid] = rec
    _write(path, records)


# ---------- helpers -----------------------------------------------------------

def _fmt_elapsed(sec: int) -> str:
    h, rem = divmod(int(sec or 0), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _detect_gpus(explicit_num: int | None, explicit_name: str | None) -> dict[str, Any]:
    """Best-effort GPU detection via nvidia-smi; caller overrides win."""
    info: dict[str, Any] = {"num_gpus": explicit_num, "gpu_name": explicit_name}
    if shutil.which("nvidia-smi") is None:
        return {k: v for k, v in info.items() if v is not None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, FileNotFoundError):
        return {k: v for k, v in info.items() if v is not None}
    lines = [l.strip() for l in out if l.strip()]
    if not lines:
        return {k: v for k, v in info.items() if v is not None}
    if info["num_gpus"] is None:
        info["num_gpus"] = len(lines)
    if info["gpu_name"] is None:
        info["gpu_name"] = lines[0].split(",")[0].strip()
    mem = lines[0].split(",")[-1].strip() if "," in lines[0] else ""
    if mem:
        info["gpu_memory"] = mem
    return info


def _categories_from_dataset(dataset_path: Path) -> dict[str, int] | None:
    """Count triples by category via the `shortname` convention (AB.<CAT>.<N>).

    Returns None when the dataset file is missing or the schema doesn't
    carry a shortname field; caller can still register the run without
    category stats in that case.
    """
    if not dataset_path.exists():
        return None
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list):
        return None
    counts: collections.Counter[str] = collections.Counter()
    for row in data:
        if not isinstance(row, dict):
            continue
        sn = row.get("shortname") or ""
        parts = sn.split(".")
        cat = parts[1] if len(parts) >= 2 else "(uncategorized)"
        counts[cat] += 1
    return dict(counts) if counts else None



def _parse_train_log(log_path: Path) -> dict[str, Any]:
    """Extract start/end timestamps, elapsed, trainable param count, and
    a best-effort final loss from an llamafactory train.log.

    Missing fields are simply omitted; this is a nice-to-have enrichment,
    not a hard requirement for registering a run.
    """
    out: dict[str, Any] = {}
    if not log_path.exists():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")

    m = re.search(r"started\s*:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", text)
    if m:
        out["started"] = m.group(1)
    m = re.search(r"finished:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", text)
    if m:
        out["finished"] = m.group(1)
    if "started" in out and "finished" in out:
        try:
            s = datetime.strptime(out["started"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            f = datetime.strptime(out["finished"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            out["elapsed_sec"] = int((f - s).total_seconds())
            out["elapsed_human"] = _fmt_elapsed(out["elapsed_sec"])
        except ValueError:
            pass

    m = re.search(r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)", text)
    if m:
        out["trainable_params"] = int(m.group(1).replace(",", ""))
        out["total_params"] = int(m.group(2).replace(",", ""))

    losses = re.findall(r"['\"]loss['\"]:\s*([0-9.]+)", text)
    if losses:
        try:
            out["final_train_loss"] = float(losses[-1])
        except ValueError:
            pass
    evals = re.findall(r"['\"]eval_loss['\"]:\s*([0-9.]+)", text)
    if evals:
        try:
            out["best_eval_loss"] = min(float(x) for x in evals)
        except ValueError:
            pass
    return out


def _derive_run_id(repo_id: str | None, output_dir: Path) -> str:
    """Prefer the HF repo basename, fall back to the output-dir name."""
    if repo_id:
        return repo_id.split("/", 1)[-1]
    return output_dir.name


def _extract_flag(extra: str, flag: str) -> Any:
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", extra)
    if not m:
        return None
    val = m.group(1)
    try:
        return float(val) if "." in val or "e" in val.lower() else int(val)
    except ValueError:
        return val


# ---------- commands ----------------------------------------------------------

def cmd_register(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    config_file = output_dir / "train_config.json"
    if not config_file.exists():
        print(f"[register] train_config.json not found at {config_file}", file=sys.stderr)
        return 2
    cfg = json.loads(config_file.read_text(encoding="utf-8"))

    repo_id = args.repo_id or cfg.get("push_to_hf") or ""
    run_id = args.run_id or _derive_run_id(repo_id or None, output_dir)

    if args.dataset_file:
        dataset_file = Path(args.dataset_file).resolve()
    else:
        dataset_file = REPO_ROOT / "SFT" / "data" / f"{cfg.get('dataset', '')}.json"
    categories = _categories_from_dataset(dataset_file)

    extra = cfg.get("extra_args", "") or ""
    lora_rank = _extract_flag(extra, "--lora_rank")
    lora_alpha = _extract_flag(extra, "--lora_alpha")
    lora_dropout = _extract_flag(extra, "--lora_dropout")

    hardware = _detect_gpus(args.num_gpus, args.gpu_name)
    timing = _parse_train_log(output_dir / "train.log")

    per_device_bs = int(cfg.get("per_device_train_batch_size") or 0)
    grad_accum = int(cfg.get("gradient_accumulation_steps") or 0)
    effective_batch = per_device_bs * grad_accum * (hardware.get("num_gpus") or 1)

    rec = {
        "run_id": run_id,
        "repo_id": repo_id,
        "output_dir": str(output_dir),
        "git_sha": cfg.get("git_sha", ""),
        "git_status": cfg.get("git_status", ""),
        "training": {
            "model": cfg.get("model_name_or_path", ""),
            "dataset": cfg.get("dataset", ""),
            "template": cfg.get("template", ""),
            "finetuning_type": cfg.get("finetuning_type", ""),
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "epochs": cfg.get("num_train_epochs"),
            "learning_rate": cfg.get("learning_rate"),
            "per_device_batch_size": per_device_bs,
            "grad_accum": grad_accum,
            "effective_batch": effective_batch,
            "cutoff_len": cfg.get("cutoff_len"),
            "max_samples": cfg.get("max_samples"),
            "report_to": cfg.get("report_to", ""),
            "extra_args": extra,
        },
        "dataset_stats": {
            "file": str(dataset_file),
            "total_rows": sum(categories.values()) if categories else None,
            "categories": categories,
        },
        "hardware": hardware,
        "timing": timing,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    registry = Path(args.registry)
    _upsert(registry, rec)
    print(f"[register] wrote {run_id} to {registry}")
    return 0



def _summarize_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten per-task metric dicts into a compact {task: metrics} map and
    pick a single headline number per task for the show table.

    The convention in SFT/eval is that each task's metrics dict has one
    obvious top-level accuracy-like value (`accuracy`, `macro_f1`, etc.).
    We keep the full dict and also project a scalar headline for display.
    """
    per_task: dict[str, Any] = {}
    headline: dict[str, float] = {}
    for t in tasks:
        name = t.get("task", "")
        metrics = t.get("metrics") or {}
        per_task[name] = metrics
        for key in ("accuracy", "macro_f1", "f1", "exact_match", "score"):
            v = metrics.get(key)
            if v is None:
                continue
            if isinstance(v, str) and v.endswith("%"):
                try:
                    headline[name] = float(v.rstrip("%")) / 100.0
                except ValueError:
                    continue
            elif isinstance(v, (int, float)):
                headline[name] = float(v)
            break
    return {"per_task": per_task, "headline": headline}


def cmd_attach(args: argparse.Namespace) -> int:
    registry = Path(args.registry)
    records = _load(registry)
    if args.run_id not in records:
        print(f"[attach] run_id '{args.run_id}' not in {registry}", file=sys.stderr)
        return 2
    rec = records[args.run_id]
    bench = rec.setdefault("benchmarks", {})

    summary_paths: list[Path] = [Path(p).resolve() for p in args.summary_json]
    if not summary_paths:
        # Auto-discover: SFT/eval/responses/<display>/summary_<suite>_*.json
        display = args.display or args.run_id
        pattern_dir = BENCH_RESPONSES_DIR / display
        if pattern_dir.exists():
            summary_paths = sorted(pattern_dir.glob("summary_*.json"))
    if not summary_paths:
        print("[attach] no summary JSON found (pass --summary-json or --display)", file=sys.stderr)
        return 2

    attached: list[str] = []
    for sp in summary_paths:
        if not sp.exists():
            print(f"[attach] missing: {sp}", file=sys.stderr)
            continue
        data = json.loads(sp.read_text(encoding="utf-8"))
        suite = data.get("suite") or "athena"
        bench[suite] = {
            "summary_file": str(sp),
            "display_name": data.get("display_name", ""),
            "version": data.get("version"),
            "rows_filter": data.get("rows_filter", ""),
            "batch": data.get("batch"),
            "tasks_requested": data.get("tasks_requested", []),
            "cybermetric_stem": data.get("cybermetric_stem", ""),
            "started": data.get("started", ""),
            "finished": data.get("finished", ""),
            "elapsed_sec": data.get("elapsed_sec", 0),
            "overall_exit": data.get("overall_exit", 0),
            "tasks": _summarize_tasks(data.get("tasks", [])),
            "attached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        attached.append(suite)

    records[args.run_id] = rec
    _write(registry, records)
    print(f"[attach] {args.run_id}: attached suite(s) {', '.join(attached)}")
    return 0


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def cmd_show(args: argparse.Namespace) -> int:
    registry = Path(args.registry)
    records = _load(registry)
    if not records:
        print(f"[show] registry is empty: {registry}")
        return 0

    runs = sorted(records.values(), key=lambda r: r.get("recorded_at", ""))

    cols = [
        ("run_id", 42),
        ("ft", 6),
        ("r/a", 7),
        ("ep", 3),
        ("bs", 4),
        ("rows", 6),
        ("loss", 6),
        ("elapsed", 8),
        ("athena", 8),
        ("ctibench", 9),
        ("cybermetric", 12),
    ]
    header = " | ".join(f"{n:<{w}}" for n, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    print(header)
    print(sep)

    for r in runs:
        tr = r.get("training", {}) or {}
        ds = r.get("dataset_stats", {}) or {}
        tm = r.get("timing", {}) or {}
        bm = r.get("benchmarks", {}) or {}

        def _suite_avg(name: str) -> str:
            s = bm.get(name) or {}
            head = ((s.get("tasks") or {}).get("headline")) or {}
            if not head:
                return "-"
            return _fmt_pct(sum(head.values()) / len(head))

        lora = f"{tr.get('lora_rank','-')}/{tr.get('lora_alpha','-')}"
        row = [
            (r.get("run_id", "")[:42], 42),
            ((tr.get("finetuning_type") or "-")[:6], 6),
            (lora[:7], 7),
            (str(tr.get("epochs") or "-")[:3], 3),
            (str(tr.get("effective_batch") or "-")[:4], 4),
            (str(ds.get("total_rows") or "-")[:6], 6),
            (f"{tm.get('final_train_loss'):.3f}" if tm.get("final_train_loss") is not None else "-", 6),
            ((tm.get("elapsed_human") or "-")[:8], 8),
            (_suite_avg("athena"), 8),
            (_suite_avg("ctibench"), 9),
            (_suite_avg("cybermetric"), 12),
        ]
        print(" | ".join(f"{v:<{w}}" for v, w in row))
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    registry = Path(args.registry)
    records = _load(registry)
    if args.run_id not in records:
        print(f"[dump] run_id '{args.run_id}' not in {registry}", file=sys.stderr)
        return 2
    print(json.dumps(records[args.run_id], indent=2, sort_keys=True))
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    print(str(Path(args.registry)))
    return 0


# ---------- argparse wiring ---------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="registry",
        description="SFT run registry: snapshot training meta + attach benchmark results.",
    )
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                   help=f"JSONL registry path (default: {DEFAULT_REGISTRY})")
    sub = p.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="Snapshot a finished SFT run into the registry")
    reg.add_argument("--output-dir", required=True,
                     help="LLaMA-Factory output_dir containing train_config.json and train.log")
    reg.add_argument("--repo-id", default="", help="HF repo id (owner/name); defaults to push_to_hf from config")
    reg.add_argument("--run-id", default="", help="Override run_id (default: HF repo basename or output_dir name)")
    reg.add_argument("--dataset-file", default="",
                     help="Path to the training dataset JSON (default: SFT/data/<cfg.dataset>.json)")
    reg.add_argument("--num-gpus", type=int, default=None, help="Override GPU count (bypass nvidia-smi)")
    reg.add_argument("--gpu-name", default=None, help="Override GPU name string")
    reg.set_defaults(func=cmd_register)

    att = sub.add_parser("attach", help="Attach SFT/eval summary JSON(s) to a run")
    att.add_argument("run_id", help="Registered run_id")
    att.add_argument("--summary-json", action="append", default=[],
                     help="Path to a summary_<suite>_*.json (repeatable)")
    att.add_argument("--display", default="",
                     help="Display name under SFT/eval/responses/ for auto-discovery")
    att.set_defaults(func=cmd_attach)

    show = sub.add_parser("show", help="Print the registry as an ASCII table")
    show.set_defaults(func=cmd_show)

    dump = sub.add_parser("dump", help="Print a single run as pretty JSON")
    dump.add_argument("run_id")
    dump.set_defaults(func=cmd_dump)

    pth = sub.add_parser("path", help="Print the registry file path")
    pth.set_defaults(func=cmd_path)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

