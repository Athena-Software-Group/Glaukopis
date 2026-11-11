#!/usr/bin/env python3
"""Per-layer, per-model breakdown of the layered development schedule (AAIML camera-ready, 11 Nov 2026).

For every base model: the base, each retained layer checkpoint (Core, +TAA, CSE, Layer 4), and the single pass, with
the per-task scores and aggregates of the paper. Point estimates from per-item files; paired 95% intervals for each
layer transition and for single pass minus each layer. Nothing is copied by hand.

Sources: layered checkpoints are the May 2026 per-item files (alkidemos archive, or SFT/test/responses for the 32B,
which are the same files); single passes are November (SFT/eval/responses). A checkpoint with no per-item files is
reported as missing rather than filled in.
Usage: python3 layer_breakdown.py [R]   (writes data/layer_breakdown.json)
"""
import glob
import json
import os
import sys

import numpy as np

import bootstrap_new_runs as B
import equivalence_ports as E



def _one_full_run(d, rel, roots=B.ROOTS):
    """As bootstrap_new_runs.one, but a full '_all_' result file wins over a smoke-test file in the same folder
    (the Llama +TAA folder also holds a two-question MMLU-Pro run)."""
    for root in roots:
        hits = sorted(glob.glob(f"{root}/{d}/{rel}"))
        if hits:
            full = [h for h in hits if "_all_" in os.path.basename(h)]
            return (full or hits)[0]
    raise FileNotFoundError(f"{d}/{rel}")


B.one = _one_full_run
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
C = "athena-cti-sft-"
MAY, NOV, TEST = [E.MAY_ROOT], [B.EVAL_ROOT], [B.TEST_ROOT]
Q3 = C + "qwen3-30b-a3b-thinking-2507-v21-"
# model -> ordered list of (row label, alias, roots). "L4 Nov" is the November re-score of the layered comparator, the
# value paired with the single pass in equivalence_ports.py; the Layer-4 row itself stays the May development score.
# The 14B Layer-3 checkpoint (no May per-item files kept) and every Qwen3 layer (scored in May only with thinking off)
# come from the November scoring with thinking on, like the Qwen3 base and single pass.
MODELS = {
    "Qwen2.5-32B": [
        ("Base", "qwen2.5-32b-vllm", MAY),
        ("Core", C + "qwen25-32b-v21-core-vllm", TEST),
        ("+TAA", C + "qwen25-32b-v21-taa-vllm", TEST),
        ("CSE", C + "qwen25-32b-v21-cse-vllm", TEST),
        ("L4", C + "qwen25-32b-v21-recal-32b-vllm", TEST),
        ("L4 Nov", C + "qwen25-32b-v21-recal-32b-vllm", NOV),
        ("Single", C + "qwen25-32b-v21-single-stage-vllm", NOV),
    ],
    "Qwen2.5-14B": [
        ("Base", "qwen2.5-14b-vllm", MAY),
        ("Core", C + "qwen25-14b-v21-core-vllm", MAY),
        ("+TAA", C + "qwen25-14b-v21-taa-vllm", MAY),
        ("CSE", C + "qwen25-14b-v21-cse-vllm", NOV),
        ("L4", C + "qwen25-14b-v21-recalibrate-vllm", MAY),
        ("L4 Nov", C + "qwen25-14b-v21-recalibrate-vllm", NOV),
        ("Single", C + "qwen25-14b-v21-single-stage-vllm", NOV),
    ],
    "Qwen3-30B-A3B-Thinking": [
        ("Base", "qwen3-30b-a3b-thinking-2507-vllm", MAY),
        ("Core", Q3 + "core-vllm", NOV),
        ("+TAA", Q3 + "taa-vllm", NOV),
        ("CSE", Q3 + "cse-vllm", NOV),
        ("L4", Q3 + "recal-32b-vllm", NOV),
        ("Single", Q3 + "single-stage-vllm", NOV),
    ],
    "Foundation-Sec-8B": [
        ("Base", "foundation-8b-instruct-vllm", MAY),
        ("Core", C + "foundation-8b-v21-core-vllm", MAY),
        ("+TAA", C + "foundation-8b-v21-taa-vllm", MAY),
        ("CSE", C + "foundation-8b-v21-cse-vllm", MAY),
        ("L4", C + "foundation-8b-v21-recalibrate-vllm", MAY),
        ("L4 Nov", C + "foundation-8b-v21-recalibrate-vllm", NOV),
        ("Single", C + "foundation-8b-v21-single-stage-vllm", NOV),
    ],
    "Llama-3.1-8B": [
        ("Base", "llama-3-8b-vllm", MAY),
        ("Core", C + "llama31-8b-v21-core-vllm", MAY),
        ("+TAA", C + "llama31-8b-v21-taa-vllm", MAY),
        ("CSE", C + "llama31-8b-v21-cse-vllm", MAY),
        ("L4", C + "llama31-8b-v21-recalibrate-vllm", MAY),
        ("L4 Nov", C + "llama31-8b-v21-recalibrate-vllm", NOV),
        ("Single", C + "llama31-8b-v21-single-stage-vllm", NOV),
    ],
}
# Alternative base runs, printed for reference only.
ALT_BASES = {
    "Qwen2.5-14B": [("Qwen_Qwen2.5-14B-Instruct", MAY), ("Qwen-2.5-14b", MAY), ("Qwen-2.5-14b", NOV)],
    "Foundation-Sec-8B": [("fdtn-ai_Foundation-Sec-8B-Instruct", MAY)],
    "Llama-3.1-8B": [("meta-llama_Meta-Llama-3.1-8B-Instruct", MAY)],
}
AGG = ["A_bal", "T_bal", "C", "Y", "B_bal", "ten"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "layer_breakdown.json")


def point(items):
    pt = {}
    for t in B.TASKS:
        keys = sorted(items[t], key=lambda k: (str(type(k)), k))
        p = E.prep(t, items[t], keys)
        pt[t] = E.stats(t, p, np.arange(len(keys), dtype=np.int32)[None, :])
    r = E.rollup(pt)
    out = {t: float(pt[t]["acc"][0]) for t in B.TASKS}
    out["rms_f1"] = float(pt["athena-rms"]["f1"][0])
    out.update({m: float(r[m][0]) for m in AGG})
    return out


def paired(a, b):
    """b minus a, 95% paired item bootstrap, on B_bal, A_bal, C, Y and MMLU-Pro."""
    res = E.compare(a, b, R)["d"]
    want = {"B_bal": "B_bal", "A_bal": "A_bal", "C": "C", "Y": "Y", "mmlu": "mmlu", "taa_can": "athena-taa-canonical",
            "vsp": "athena-vsp", "malware": "cybersoceval-malware", "ti": "cybersoceval-ti"}
    return {k: {f: res[name][f] for f in ("d", "lo95", "hi95", "lo90", "hi90")} for k, name in want.items() if name in res}


def main():
    result = {}
    for model, rows in MODELS.items():
        print(f"\n=== {model}")
        data, entry = {}, {"rows": {}, "missing": {}, "transitions": {}, "single_minus": {}}
        for lab, alias, roots in rows:
            items, missing = E.load(alias, roots)
            if missing:
                entry["missing"][lab] = {"alias": alias, "tasks": missing}
                print(f"  {lab:7s} {alias}: MISSING {len(missing)} tasks ({', '.join(missing[:4])}...)")
                continue
            data[lab] = items
            p = point(items)
            entry["rows"][lab] = {"alias": alias, **p}
            print(f"  {lab:7s} A {p['A_bal']:5.1f} TAAcan {p['athena-taa-canonical']:5.1f} VSP {p['athena-vsp']:5.1f} "
                  f"C {p['C']:5.1f} Y {p['Y']:5.1f} B {p['B_bal']:5.1f} ten {p['ten']:5.1f} MMLU {p['mmlu']:5.1f}")
        order = [lab for lab, _, _ in rows if lab in data and lab not in ("Single", "L4 Nov")]
        for x, y in zip(order, order[1:]):
            entry["transitions"][f"{y} - {x}"] = paired(data[x], data[y])
        if "Single" in data:
            for x in order:
                entry["single_minus"][x] = paired(data[x], data["Single"])
        for k, v in entry["transitions"].items():
            print(f"  {k:14s} B {v['B_bal']['d']:+5.1f} [{v['B_bal']['lo95']:+.1f},{v['B_bal']['hi95']:+.1f}]  "
                  f"MMLU {v['mmlu']['d']:+5.1f}  TAAcan {v['taa_can']['d']:+5.1f}  C {v['C']['d']:+5.1f}")
        for k, v in entry["single_minus"].items():
            print(f"  Single - {k:6s} B {v['B_bal']['d']:+5.1f} [{v['B_bal']['lo95']:+.1f},{v['B_bal']['hi95']:+.1f}]  "
                  f"MMLU {v['mmlu']['d']:+5.1f}")
        for alias, roots in ALT_BASES.get(model, []):
            try:
                items, missing = E.load(alias, roots)
            except (TypeError, KeyError) as e:  # older result formats
                print(f"  alt base {alias}: unreadable ({e})")
                continue
            if not missing:
                p = point(items)
                tag = "May" if roots == MAY else "Nov"
                print(f"  alt base {alias} ({tag}): B {p['B_bal']:5.1f} A {p['A_bal']:5.1f} MMLU {p['mmlu']:5.1f}")
        result[model] = entry
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
