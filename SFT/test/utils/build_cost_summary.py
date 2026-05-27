#!/usr/bin/env python3
"""Aggregate per-benchmark cost into responses/cost_summary.{csv,tsv}.

Two cost bases combined in one table:

  - API-billed models: per-task cost from responses/api_usage_checkpoint.json
    (populated by inference.py:save_checkpoint after each billable task).
  - SFT vLLM models : per-task wallclock from responses/<safe>/summary_*.json
    (written by run_benchmark.sh -> _print_sweep_summary.py), converted to
    USD via a flat GPU-hour assumption (default: 2xH100 @ $2.50/hr).

Override the GPU billing via env vars: GPU_RATE_USD_PER_HR, GPU_COUNT.

Local row selection: --include regex (default keeps v21 SFT stages +
five named upstream base models) is matched against each response
directory basename; only matches are kept. --exclude (default empty)
drops further matches after that. Pass ``--include .`` to keep every
local row that has at least one summary_*.json.

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

# Default keeps the v21 SFT chain (any model family, any sub-stage
# suffix) plus the upstream base-model response dirs explicitly tracked
# for cost-comparison baselines. Matched case-insensitively (re.I) so
# both HF-id-form safe dirs (e.g. ``Qwen_Qwen2.5-14B-Instruct``, written
# when the bench was invoked with the raw HF id) and alias-form safe
# dirs (e.g. ``qwen2.5-14b-vllm``, written by the cost-revalidation
# chain via the ``-vllm`` aliases in pipelines/models.py) land in the
# summary. Add to this list by passing --include with an alternation;
# pass --include '.' to keep every dir with a summary_*.json file.
DEFAULT_INCLUDE = (
    # v21 SFT chain (any model family, any sub-stage suffix)
    r"-v21(-|$)"
    # HF-id-form baseline safe dirs ('<org>_<repo>')
    r"|Foundation-Sec-8B-Instruct"
    r"|Meta-Llama-3\.1-8B-Instruct"
    r"|Qwen2\.5-14B-Instruct"
    r"|Qwen2\.5-32B-Instruct"
    r"|Qwen3-30B-A3B-Thinking-2507"
    r"|Qwen3-30B-A3B-Instruct-2507"
    # Alias-form baseline safe dirs ('<alias>-vllm')
    r"|^foundation-8b-instruct-vllm$"
    r"|^llama-3-8b-vllm$"
    r"|^qwen2\.5-14b-vllm$"
    r"|^qwen2\.5-32b-vllm$"
    r"|^qwen3-30b-a3b-thinking-2507(-no-think)?-vllm$"
    r"|^qwen3-30b-a3b-instruct-2507-vllm$"
    r"|^qwen3-32b(-no-think)?-vllm$"
)
DEFAULT_INCLUDE_FLAGS = re.IGNORECASE
# Same-model_key collisions (e.g. alias-form 'qwen2.5-32b-vllm' vs HF-
# id-form 'Qwen_Qwen2.5-32B-Instruct', or HF-id-as-alias 'asg-ai_...'
# vs '-vllm' alias for the same SFT checkpoint) are resolved by the
# dedup-by-max-cost pass in main(), so no per-dir exclude is needed
# here by default.
DEFAULT_EXCLUDE = r""

# Reasoning level used per API model_name for the runs captured in
# api_usage_checkpoint.json. The checkpoint records the alias but NOT
# the reasoning_effort flag, so this table is the authoritative source
# for that column. Add a new entry when a new reasoning-mode sweep is
# committed; leave a model out to display blank.
REASONING_EFFORT = {
    "gpt5.2":         "medium",   # default; token ratio 0.53 out:in
    "gpt5.5":         "medium",   # default; token ratio 0.69 out:in
    "gemini-3-flash": "default",  # 32x out:in -- thinking tokens billed
                                  # at output rate, level not exposed
}

# Canonical display name (HF repo id or vendor model id) per API
# model_name. The checkpoint records the bench-internal alias (e.g.
# 'gpt5.2', 'deepseek-v4-pro-hf'); this table maps it back to the name
# used in vendor pricing pages / HF model cards for the leaderboard.
# Local vLLM rows derive their key by rule (see derive_model_key); add
# entries here only for API aliases or for SFT aliases whose canonical
# name doesn't follow the default '-vllm'/'-no-think-vllm' strip rule.
API_MODEL_KEY = {
    "gpt4":                       "gpt-4-turbo-2024-04-09",
    "gpt5":                       "gpt-5",
    "gpt5.2":                     "gpt-5.2",
    "gpt5.5":                     "gpt-5.5",
    "gpt5.5-pro":                 "gpt-5.5-pro",
    "gemini-2.5-flash":           "gemini-2.5-flash",
    "gemini-2.5-pro":             "gemini-2.5-pro",
    "gemini-3-pro":               "gemini-3-pro-preview",
    "gemini-3-flash":             "gemini-3-flash-preview",
    "gemini-3.1-pro":             "gemini-3.1-pro-preview",
    "deepseek-v3.1-terminus-hf":  "deepseek-ai/DeepSeek-V3.1-Terminus",
    "deepseek-v3.2-exp-hf":       "deepseek-ai/DeepSeek-V3.2-Exp",
    "deepseek-v4-pro-hf":         "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-v4-flash-hf":       "deepseek-ai/DeepSeek-V4-Flash",
}

# Override table for safe-dir basenames whose canonical name cannot be
# derived from the default rules below. Empty by design; add an entry
# only if a new alias breaks the rule (e.g. a vendor whose alias prefix
# is not 'athena-cti-' and whose safe-dir doesn't contain '_').
SFT_MODEL_KEY_OVERRIDES: dict[str, str] = {}

# Maps an alias-form baseline safe dir (after '-vllm' / '-no-think-vllm'
# suffix strip) to the canonical HF repo id. The same HF id is also the
# model_key resolved from the HF-id-form safe dir of the same baseline
# (e.g. 'Qwen_Qwen2.5-32B-Instruct' -> 'Qwen/Qwen2.5-32B-Instruct'), so
# both safe-dir flavours collide on a single model_key and the dedup
# pass in main() keeps the higher-cost row. Mirrors model_mapping in
# pipelines/models.py; add a new entry when a new '-vllm' baseline alias
# is added there.
BASELINE_ALIAS_KEY = {
    "qwen2.5-14b":                    "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b":                    "Qwen/Qwen2.5-32B-Instruct",
    "qwen3-30b-a3b-thinking-2507":    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3-30b-a3b-instruct-2507":    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-32b":                      "Qwen/Qwen3-32B",
    "llama-3-8b":                     "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "foundation-8b-instruct":         "fdtn-ai/Foundation-Sec-8B-Instruct",
    "foundation-8b":                  "fdtn-ai/Foundation-Sec-8B",
}


def derive_model_key(name: str, basis: str) -> str:
    """Return canonical HF-repo / vendor-id display name for a row.

    API rows: look up the bench-internal alias in API_MODEL_KEY.
    Local vLLM rows: strip '-no-think-vllm' or '-vllm' suffix, then
        - lookup in BASELINE_ALIAS_KEY (covers the upstream baselines)
        - 'athena-cti-*'       -> 'asg-ai/<stripped>'
        - 'asg-ai_X' (from a HF-id-as-alias run, no '-vllm' suffix)
                              -> 'asg-ai/X'
        - 'X_Y' (HF-id-safe: '<org>_<repo>' with no '-vllm')
                              -> 'X/Y'
        - everything else      -> '' (blank; add to SFT_MODEL_KEY_OVERRIDES)
    """
    if name in SFT_MODEL_KEY_OVERRIDES:
        return SFT_MODEL_KEY_OVERRIDES[name]
    if basis == "API tokens":
        return API_MODEL_KEY.get(name, "")
    a = name
    if a.endswith("-no-think-vllm"):
        a = a[: -len("-no-think-vllm")]
    elif a.endswith("-vllm"):
        a = a[: -len("-vllm")]
    if a in BASELINE_ALIAS_KEY:
        return BASELINE_ALIAS_KEY[a]
    if a.startswith("athena-cti-"):
        return "asg-ai/" + a
    if a.startswith("asg-ai_"):
        return "asg-ai/" + a[len("asg-ai_"):]
    if "_" in a:
        head, _, tail = a.partition("_")
        return f"{head}/{tail}"
    return ""


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
                  exclude_re: re.Pattern | None,
                  api_model_names: set[str]) -> dict:
    agg: dict = defaultdict(lambda: {"sec": 0})
    for model_dir in sorted(glob.glob(os.path.join(responses_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        safe = os.path.basename(model_dir)
        summaries = glob.glob(os.path.join(model_dir, "summary_*.json"))
        if not summaries:
            continue
        # Guard against double-counting: if the same key is already
        # billed via the API checkpoint, skip the local summary path.
        if safe in api_model_names:
            continue
        if not include_re.search(safe):
            continue
        if exclude_re is not None and exclude_re.search(safe):
            continue
        for sj in summaries:
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
    # Default include uses DEFAULT_INCLUDE_FLAGS (re.IGNORECASE) so the
    # alias-form ('qwen2.5-14b-vllm') and HF-id-form ('Qwen_Qwen2.5-14B-
    # Instruct') safe dirs both match the same baseline patterns. A
    # user-supplied --include is honoured verbatim with no flags so
    # explicit case-sensitive overrides remain possible.
    inc_flags = DEFAULT_INCLUDE_FLAGS if args.include == DEFAULT_INCLUDE else 0
    include_re = re.compile(args.include, inc_flags)
    exclude_re = re.compile(args.exclude) if args.exclude else None

    api = aggregate_api(os.path.join(args.responses_dir, "api_usage_checkpoint.json"))
    api_model_names = {k[0] for k in api}
    sft = aggregate_sft(args.responses_dir, include_re, exclude_re, api_model_names)

    hdr = ["model_key", "model"] + [f"{f}_cost_usd" for f in FAMS] + [
        "total_input_tok", "total_output_tok", "usd_per_1k_tok",
        "wallclock_hours", "gpu_hours_billed", "billing_basis",
        "reasoning_effort", "total_cost_usd",
    ]
    # (total_cost_float, row) pairs so we can sort by cost desc across both
    # API and SFT rows before serialising.
    scored: list[tuple[float, list]] = []
    for m in sorted({k[0] for k in api}):
        r = [derive_model_key(m, basis_api), m]; tc = 0.0; ti = to = 0
        for f in FAMS:
            k = (m, f)
            if k in api:
                r.append(f"{api[k]['cost']:.4f}")
                tc += api[k]["cost"]; ti += api[k]["in"]; to += api[k]["out"]
            else:
                r.append("")
        tok = ti + to
        r += [ti, to, f"{tc/tok*1000:.6f}" if tok else "",
              "", "", basis_api, REASONING_EFFORT.get(m, ""), f"{tc:.4f}"]
        scored.append((tc, r))
    for m in sorted({k[0] for k in sft}):
        r = [derive_model_key(m, basis_sft), m]; tc = 0.0; total_sec = 0
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
        r += ["", "", "",
              f"{wh:.3f}", f"{wh*gpu_count:.3f}", basis_sft, "", f"{tc:.4f}"]
        scored.append((tc, r))
    # Dedup by model_key (column 0): when an upstream baseline lands
    # twice -- once via the alias-form safe dir ('qwen2.5-32b-vllm')
    # and once via the HF-id-form safe dir ('Qwen_Qwen2.5-32B-Instruct')
    # -- both rows now collide on the same model_key (via
    # BASELINE_ALIAS_KEY) and the higher-cost entry is kept on the
    # assumption it reflects the more complete / fresher sweep. Rows
    # with an empty model_key are passed through untouched so unknown-
    # canonical-name rows never get silently merged with each other.
    by_key: dict[str, tuple[float, list]] = {}
    passthrough: list[tuple[float, list]] = []
    for tc, r in scored:
        mk = r[0]
        if not mk:
            passthrough.append((tc, r))
            continue
        prev = by_key.get(mk)
        if prev is None or tc > prev[0]:
            by_key[mk] = (tc, r)
    deduped = list(by_key.values()) + passthrough
    rows = [r for _, r in sorted(deduped, key=lambda x: x[0], reverse=True)]

    for path, delim in [(args.out_tsv, "\t"), (args.out_csv, ",")]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter=delim, lineterminator="\n")
            w.writerow(hdr); w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
