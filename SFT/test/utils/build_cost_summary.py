#!/usr/bin/env python3
"""Aggregate per-benchmark cost into responses/cost_summary.{csv,tsv}.

Two cost bases combined in one table:

  - API-billed models: per-task cost from responses/api_usage_checkpoint.json
    (populated by inference.py:save_checkpoint after each billable task).
  - SFT vLLM models : per-task wallclock from responses/<safe>/summary_*.json
    (written by run_benchmark.sh -> _print_sweep_summary.py), converted to
    USD via a flat GPU-hour assumption (default: 2xH100 @ $2.50/hr).

Override the GPU billing via env vars: GPU_RATE_USD_PER_HR, GPU_COUNT.

SFT row selection: --include regex (default ``-v21(-|$)``) is matched
against each SFT alias and only matches are kept; --exclude (default
empty) drops further matches after that. Pass ``--include .`` to keep
every SFT row.

Run from SFT/test/ so the relative paths resolve.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

FAMILY_OF_TASK = {
    "athena-mcq": "athena", "athena-rcm": "athena", "athena-vsp": "athena",
    "athena-ate": "athena", "athena-taa": "athena",
    "athena-taa-canonical": "athena", "athena-rms": "athena",
    "cybersoceval-malware": "cybersoceval", "cybersoceval-ti": "cybersoceval",
    "mmlu-pro": "mmlu_pro",
}
FAMS = ["athena", "cybersoceval", "cybermetric", "mmlu_pro"]

# Match anything under the SFT response tree; the alias is the dir name.
SFT_DIR_RE = re.compile(r"(athena-cti-sft-|asg-ai_athena-cti-sft-)")

# Default: keep only v21 stages (any model family, any sub-stage suffix).
DEFAULT_INCLUDE = r"-v21(-|$)"
DEFAULT_EXCLUDE = r""


def family(task: str) -> str | None:
    if task.startswith("cybermetric"):
        return "cybermetric"
    return FAMILY_OF_TASK.get(task)


def aggregate_api(ckpt_path: str) -> dict:
    agg: dict = defaultdict(lambda: {"cost": 0.0, "in": 0, "out": 0})
    if not os.path.exists(ckpt_path):
        return agg
    for d in json.load(open(ckpt_path)):
        fam = family(d.get("task", ""))
        if fam is None:
            continue
        cost = float(d.get("total_cost", 0) or 0)
        in_t = int(d.get("input_tokens", 0) or 0)
        out_t = int(d.get("output_tokens", 0) or 0)
        if cost == 0 and in_t == 0 and out_t == 0:
            continue
        k = (d["model_name"], fam)
        agg[k]["cost"] += cost
        agg[k]["in"] += in_t
        agg[k]["out"] += out_t
    return agg


def aggregate_sft(responses_dir: str, include_re: re.Pattern,
                  exclude_re: re.Pattern | None) -> dict:
    agg: dict = defaultdict(lambda: {"sec": 0})
    for model_dir in sorted(glob.glob(os.path.join(responses_dir, "*"))):
        safe = os.path.basename(model_dir)
        if not SFT_DIR_RE.search(safe):
            continue
        if not include_re.search(safe):
            continue
        if exclude_re is not None and exclude_re.search(safe):
            continue
        for sj in glob.glob(os.path.join(model_dir, "summary_*.json")):
            try:
                s = json.load(open(sj))
            except (OSError, ValueError):
                continue
            for trec in s.get("tasks", []):
                t = trec.get("task", "")
                base = "cybermetric" if t.startswith("cybermetric") else t
                fam = family(base)
                if fam is None:
                    continue
                agg[(safe, fam)]["sec"] += int(trec.get("elapsed_sec", 0) or 0)
    return agg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--responses-dir", default="responses")
    p.add_argument("--out-csv", default="responses/cost_summary.csv")
    p.add_argument("--out-tsv", default="responses/cost_summary.tsv")
    p.add_argument("--include", default=DEFAULT_INCLUDE,
                   help=f"Regex (re.search) SFT aliases must match to be "
                        f"kept. Default: {DEFAULT_INCLUDE!r} (v21 stages "
                        f"only). Use '.' to keep every SFT row.")
    p.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                   help="Regex (re.search) applied after --include to drop "
                        "further SFT rows. Default: empty (no extra drops).")
    args = p.parse_args()

    gpu_rate = float(os.environ.get("GPU_RATE_USD_PER_HR", "2.50"))
    gpu_count = int(os.environ.get("GPU_COUNT", "2"))
    basis_sft = f"{gpu_count}xH100 @ ${gpu_rate:.2f}/hr"
    basis_api = "API tokens"
    include_re = re.compile(args.include)
    exclude_re = re.compile(args.exclude) if args.exclude else None

    api = aggregate_api(os.path.join(args.responses_dir, "api_usage_checkpoint.json"))
    sft = aggregate_sft(args.responses_dir, include_re, exclude_re)

    hdr = ["model"] + [f"{f}_cost_usd" for f in FAMS] + [
        "total_cost_usd", "total_input_tok", "total_output_tok",
        "usd_per_1k_tok", "wallclock_hours", "gpu_hours_billed", "billing_basis"
    ]
    rows = []
    for m in sorted({k[0] for k in api}):
        r = [m]; tc = ti = to = 0
        for f in FAMS:
            k = (m, f)
            if k in api:
                r.append(f"{api[k]['cost']:.4f}")
                tc += api[k]["cost"]; ti += api[k]["in"]; to += api[k]["out"]
            else:
                r.append("")
        tok = ti + to
        r += [f"{tc:.4f}", ti, to,
              f"{tc/tok*1000:.6f}" if tok else "", "", "", basis_api]
        rows.append(r)
    for m in sorted({k[0] for k in sft}):
        r = [m]; tc = 0.0; total_sec = 0
        for f in FAMS:
            k = (m, f)
            if k in sft:
                sec = sft[k]["sec"]
                cost = (sec / 3600.0) * gpu_count * gpu_rate
                r.append(f"{cost:.4f}")
                tc += cost; total_sec += sec
            else:
                r.append("")
        wh = total_sec / 3600.0
        r += [f"{tc:.4f}", "", "", "",
              f"{wh:.3f}", f"{wh*gpu_count:.3f}", basis_sft]
        rows.append(r)

    for path, delim in [(args.out_tsv, "\t"), (args.out_csv, ",")]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter=delim, lineterminator="\n")
            w.writerow(hdr); w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
