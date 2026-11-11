#!/usr/bin/env python3
"""Numbers for the AAIML camera-ready rewrite in which the single-pass model is the headline (11 Nov 2026).

Per-item results only; nothing is copied by hand. Writes data/paper_numbers.json and prints a summary.
  base     Qwen2.5-32B-Instruct, May run `qwen2.5-32b-vllm` (the higher of the two May base runs; per-item files in
           the alkidemos archive)
  1-pass   single pass over all shards (G1), November, SFT/eval/responses
  layered  Core, +TAA, CSE, Recal-32B, May per-item files (SFT/test/responses)
Single-checkpoint intervals: 95% percentile bootstrap over items. Paired differences resample the same items for
both checkpoints. Same seed and item keys as bootstrap_new_runs.py / equivalence_ports.py.
Usage: python3 paper_numbers.py [R]
"""
import json
import os
import sys

import numpy as np

import bootstrap_new_runs as B
import equivalence_ports as E

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
C = "athena-cti-sft-qwen25-32b-v21-"
CKS = {
    "Base": ("qwen2.5-32b-vllm", [E.MAY_ROOT]),
    "Base (other May run)": ("Qwen_Qwen2.5-32B-Instruct", [E.MAY_ROOT]),
    "1-pass": (C + "single-stage-vllm", [B.EVAL_ROOT]),
    "Core": (C + "core-vllm", [B.TEST_ROOT]),
    "+TAA": (C + "taa-vllm", [B.TEST_ROOT]),
    "CSE": (C + "cse-vllm", [B.TEST_ROOT]),
    "Recal-32B": (C + "recal-32b-vllm", [B.TEST_ROOT]),
}
SHOW = ["athena-mcq", "athena-ate", "athena-rcm", "athena-rms", "athena-vsp", "athena-taa", "athena-taa-canonical",
        "cybersoceval-malware", "cybersoceval-ti", "cm2000", "cm10000", "mmlu"]
AGG = ["T_bal", "A_bal", "C", "Y", "B_bal", "ten"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper_numbers.json")


def single(items_by_task):
    """Point estimates and 95% bootstrap intervals for one checkpoint."""
    rng = np.random.default_rng(E.SEED)
    pt, bs = {}, {}
    for t in B.TASKS:
        items = items_by_task[t]
        keys = sorted(items, key=lambda k: (str(type(k)), k))
        p = E.prep(t, items, keys)
        n = len(keys)
        pt[t] = E.stats(t, p, np.arange(n, dtype=np.int32)[None, :])
        bs[t] = E.stats(t, p, rng.integers(0, n, size=(R, n), dtype=np.int32))
    out = {t: {"v": float(pt[t]["acc"][0])} for t in B.TASKS}
    rp, rb = E.rollup(pt), E.rollup(bs)
    for m in AGG:
        lo, hi = E.interval(rb[m], 95)
        out[m] = {"v": float(rp[m][0]), "lo95": lo, "hi95": hi}
    for t in B.TASKS:
        lo, hi = E.interval(bs[t]["acc"], 95)
        out[t].update(lo95=lo, hi95=hi)
    out["vsp_failed"] = int(len(items_by_task["athena-vsp"]) - E.prep("athena-vsp", items_by_task["athena-vsp"], list(items_by_task["athena-vsp"]))["ok"].sum())
    return out


def main():
    loaded = {}
    for lab, (alias, roots) in CKS.items():
        data, miss = E.load(alias, roots)
        assert not miss, (lab, miss)
        loaded[lab] = data
    res = {"R": R, "checkpoints": {k: v[0] for k, v in CKS.items()}, "single": {}, "paired": {}}
    for lab, data in loaded.items():
        res["single"][lab] = single(data)
    for a, b in [("Base", "1-pass"), ("Base", "Recal-32B"), ("Recal-32B", "1-pass"), ("Core", "1-pass")]:
        res["paired"][f"{b} - {a}"] = E.compare(loaded[a], loaded[b], R)["d"]

    def f(x):
        return f"{x:6.2f}"
    print("checkpoint            " + " ".join(f"{t[-8:]:>8s}" for t in SHOW + AGG))
    for lab, s in res["single"].items():
        print(f"{lab:21s} " + " ".join(f"{s[t]['v']:8.2f}" for t in SHOW + AGG))
    for lab in ("1-pass", "Base", "Recal-32B"):
        s = res["single"][lab]
        print(lab, "A_bal", f(s["A_bal"]["v"]), f"[{s['A_bal']['lo95']:.2f}, {s['A_bal']['hi95']:.2f}]",
              "B_bal", f(s["B_bal"]["v"]), f"[{s['B_bal']['lo95']:.2f}, {s['B_bal']['hi95']:.2f}]", "vsp_failed", s["vsp_failed"])
    for k, d in res["paired"].items():
        print(k, {m: (round(d[m]["d"], 2), round(d[m]["lo95"], 2), round(d[m]["hi95"], 2)) for m in ("A_bal", "B_bal", "C", "Y", "mmlu", "athena-ate", "athena-rms", "athena-vsp", "athena-taa-canonical") if m in d})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
