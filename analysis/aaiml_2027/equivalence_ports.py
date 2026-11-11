#!/usr/bin/env python3
"""Single-pass versus layered fine-tuning: the pre-set equivalence test, one row per model.

The rule was fixed in RUN_PLAN_JARVIS.md before the four cross-model runs were trained:

    A model counts as equivalent if the 90% paired interval of (single pass minus layered)
    lies entirely inside +/-1.5 on B_bal AND inside +/-2 on MMLU-Pro (TOST at the 5% level).

All models are reported whichever way they land, and the claim covers only those that pass.
The margins were chosen after the Qwen2.5-32B result was known, so that row is shown for
reference and only the four other models are a clean test. Ten-test and per-task differences
are reported alongside but do not enter the decision. Intervals come from resampling test
items, so they do not include training variability (seed SD measured on 32B: B_bal 0.26,
MMLU-Pro 1.74).

Both sides of every comparison are read from SFT/eval/responses, i.e. scored on one serving
stack. A second block compares each re-scored layered checkpoint against its May per-item
results, as a check that the serving stack did not move the numbers. A third block is the
hardware control: the layered chain retrained on 2xH200, against the single pass on the same
hardware and against the published 8xH100 layered checkpoint. The pre-set rule is not applied
to that block.

Scoring functions and aggregates are imported from bootstrap_new_runs.py and the resampling
below is checked against them at the point estimate.

Usage: python equivalence_ports.py [R]   (default R=2000; writes data/equivalence_ports.json)
"""
import glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap_new_runs as B

C = "athena-cti-sft-"
Q3 = C + "qwen3-30b-a3b-thinking-2507-v21-"
# (label, single-pass alias, layered alias, comparator name, margins fixed before the result was seen)
MODELS = [
    ("Qwen2.5-32B", C + "qwen25-32b-v21-single-stage-vllm", C + "qwen25-32b-v21-recal-32b-vllm", "Recal-32B", False),
    ("Qwen2.5-14B", C + "qwen25-14b-v21-single-stage-vllm", C + "qwen25-14b-v21-recalibrate-vllm", "Recalibrate", True),
    ("Qwen3-30B-A3B-Thinking", Q3 + "single-stage-vllm", Q3 + "cse-vllm", "CSE", True),
    ("Foundation-Sec-8B", C + "foundation-8b-v21-single-stage-vllm", C + "foundation-8b-v21-recalibrate-vllm", "Recalibrate", True),
    ("Llama-3.1-8B", C + "llama31-8b-v21-single-stage-vllm", C + "llama31-8b-v21-recalibrate-vllm", "Recalibrate", True),
]
# Hardware control: (label, single pass, layered chain retrained on the single pass's hardware, published layered checkpoint
# re-scored, that hardware).
CONTROLS = [
    ("Foundation-Sec-8B", C + "foundation-8b-v21-single-stage-vllm", C + "foundation-8b-v21-2xh200-recalibrate-vllm", C + "foundation-8b-v21-recalibrate-vllm", "2xH200"),
    ("Llama-3.1-8B", C + "llama31-8b-v21-single-stage-vllm", C + "llama31-8b-v21-2xh200-recalibrate-vllm", C + "llama31-8b-v21-recalibrate-vllm", "2xH200"),
    ("Qwen2.5-14B", C + "qwen25-14b-v21-single-stage-vllm", C + "qwen25-14b-v21-4xh200-recalibrate-vllm", C + "qwen25-14b-v21-recalibrate-vllm", "4xH200"),
    ("Qwen2.5-32B", C + "qwen25-32b-v21-single-stage-vllm", C + "qwen25-32b-v21-4xh200-recal-32b-vllm", C + "qwen25-32b-v21-recal-32b-vllm", "4xH200"),
]
MARGINS = {"B_bal": 1.5, "mmlu": 2.0}
SEED = 20260906  # same seed as bootstrap_new_runs.py
MAY_ROOT = f"{B.R_ROOT}/doc/AAIML/alkidemos_archive_20261109/SFT_test/responses"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "equivalence_ports.json")
AGGS = ("B_bal", "A_bal", "T_bal", "C", "Y", "ten", "B_failed_as_wrong")


def prep(task, items, keys):
    v = [items[k] for k in keys]
    p = {"score": np.array([0.0 if x["score"] is None else x["score"] for x in v], float),
         "ok": np.array([x["success"] for x in v], float)}
    if task == "athena-rms":
        p["f1"] = np.array([x["f1"] for x in v], float)
    return p


def stats(task, p, idx):
    """Vectorised bootstrap_new_runs.metric: idx is resamples x items; each value has one entry per resample."""
    n = idx.shape[1]
    s, ok = p["score"][idx], p["ok"][idx]
    if task == "athena-vsp":
        nok = np.maximum(ok.sum(1), 1)
        return {"acc": 100 * (1 - (s * ok).sum(1) / nok / B.VSP_DENOM)}
    if task.startswith("cybersoceval"):
        nok = np.maximum(ok.sum(1), 1)
        return {"acc": 100 * ((s == 1.0) & (ok == 1)).sum(1) / nok, "jaccard": 100 * (s * ok).sum(1) / nok,
                "acc_all": 100 * ((s == 1.0) & (ok == 1)).sum(1) / n, "jaccard_all": 100 * (s * ok).sum(1) / n}
    out = {"acc": 100 * s.sum(1) / n}
    if task == "athena-rms":
        out["f1"] = 100 * p["f1"][idx].sum(1) / n
    return out


def ten_test(m):
    """The paper's ten-test mean (column 'Weighted Avg Total Score' of the results workbook): RMS uses plain F1."""
    return (m["athena-mcq"]["acc"] + m["athena-ate"]["acc"] + m["athena-rcm"]["acc"] + m["athena-rms"]["f1"]
            + m["athena-vsp"]["acc"] + (m["athena-taa"]["acc"] + m["athena-taa-canonical"]["acc"]) / 2
            + m["cybersoceval-malware"]["acc"] + m["cybersoceval-ti"]["acc"] + m["cm2000"]["acc"] + m["cm10000"]["acc"]) / 10


def rollup(m):
    a = B.aggregates(m)
    a["ten"] = ten_test(m)
    # Sensitivity only: the harness marks empty answers and over-long prompts as failed and drops them from CyberSOCEval's
    # denominator, so a model that answers [] on hard items is not charged for them. Here those items count as wrong.
    c_all = (m["cybersoceval-malware"]["acc_all"] + m["cybersoceval-malware"]["jaccard_all"] + m["cybersoceval-ti"]["acc_all"]) / 3
    a["B_failed_as_wrong"] = (a["A_bal"] + c_all + a["Y"]) / 3
    return a


def interval(d, level):
    s = np.sort(d)
    return tuple(float(x) for x in (B.ci(s.tolist(), (100 - level) / 2, 100 - (100 - level) / 2)))


def load(alias, roots):
    data, missing = {}, []
    for t in B.TASKS:
        try:
            data[t] = B.load_task(alias, t, roots)
        except FileNotFoundError:
            missing.append(t)
    return data, missing


def warn_duplicates(alias, roots):
    for root in roots:
        for rel in ["mmlu-pro/*.csv", "*/*_scored.jsonl", "cybermetric/*_response.csv"]:
            by_dir = {}
            for f in glob.glob(f"{root}/{alias}/{rel}"):
                key = os.path.dirname(f) if "cybermetric" not in rel else f.split("CyberMetric-")[1][:5]
                by_dir.setdefault(key, []).append(f)
            for k, fs in by_dir.items():
                if len(fs) > 1:
                    print(f"  WARNING {alias}: {len(fs)} result files for {os.path.basename(str(k))}; using the first, {os.path.basename(fs[0])}")


def compare(a, b, R, tasks=None):
    """Paired bootstrap of b minus a over the tasks both have. a, b: dict task -> items."""
    tasks = tasks or [t for t in B.TASKS if t in a and t in b]
    rng = np.random.default_rng(SEED)
    keys = {t: sorted(set(a[t]) & set(b[t]), key=lambda k: (str(type(k)), k)) for t in tasks}
    pa = {t: prep(t, a[t], keys[t]) for t in tasks}
    pb = {t: prep(t, b[t], keys[t]) for t in tasks}
    pt_a, pt_b, bs_a, bs_b = {}, {}, {}, {}
    for t in tasks:
        n = len(keys[t])
        whole = np.arange(n, dtype=np.int32)[None, :]
        pt_a[t], pt_b[t] = stats(t, pa[t], whole), stats(t, pb[t], whole)
        for side, items in (("a", a[t]), ("b", b[t])):  # the resampling must agree with the published metric
            ref = B.metric(t, [items[k] for k in keys[t]])["acc"]
            got = float((pt_a if side == "a" else pt_b)[t]["acc"][0])
            assert abs(ref - got) < 1e-9, (t, side, ref, got)
        idx = rng.integers(0, n, size=(R, n), dtype=np.int32)
        bs_a[t], bs_b[t] = stats(t, pa[t], idx), stats(t, pb[t], idx)
    res = {"n": {t: len(keys[t]) for t in tasks}, "n_a": {t: len(a[t]) for t in tasks}, "n_b": {t: len(b[t]) for t in tasks}, "d": {},
           # failed requests: CyberSOCEval and VSP leave them out of the denominator, so a different count changes what is averaged
           "failed_a": {t: int(len(keys[t]) - pa[t]["ok"].sum()) for t in tasks}, "failed_b": {t: int(len(keys[t]) - pb[t]["ok"].sum()) for t in tasks}}

    def add(name, xa, xb, da, db):
        d = db - da
        lo90, hi90 = interval(d, 90)
        lo95, hi95 = interval(d, 95)
        res["d"][name] = {"a": float(xa), "b": float(xb), "d": float(xb - xa), "lo90": lo90, "hi90": hi90, "lo95": lo95, "hi95": hi95}

    for t in tasks:
        add(t, pt_a[t]["acc"][0], pt_b[t]["acc"][0], bs_a[t]["acc"], bs_b[t]["acc"])
    if len(tasks) == len(B.TASKS):
        ra, rb, sa, sb = rollup(pt_a), rollup(pt_b), rollup(bs_a), rollup(bs_b)
        for m in AGGS:
            add(m, ra[m][0], rb[m][0], sa[m], sb[m])
    return res


def show(res, names):
    for name in names:
        if name in res["d"]:
            r = res["d"][name]
            print(f"    {name:22s} {r['a']:7.2f} {r['b']:7.2f}  {r['d']:+6.2f}  90% [{r['lo90']:+6.2f},{r['hi90']:+6.2f}]  95% [{r['lo95']:+6.2f},{r['hi95']:+6.2f}]")


def report_failures(res):
    diff = {t: (res["failed_a"][t], res["failed_b"][t]) for t in res["n"] if res["failed_a"][t] or res["failed_b"][t]}
    if diff:
        print("    failed requests (first side, second side): " + ", ".join(f"{t} {x}/{y}" for t, (x, y) in diff.items()))


def verdict(res):
    out = {}
    for m, margin in MARGINS.items():
        r = res["d"][m]
        out[m] = {"margin": margin, "inside": bool(-margin < r["lo90"] and r["hi90"] < margin)}
    out["equivalent"] = all(v["inside"] for k, v in out.items() if k in MARGINS)
    return out


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = {"R": R, "rule": "90% paired interval of single minus layered inside +/-1.5 B_bal and +/-2 MMLU-Pro", "models": {}, "anchors": {}, "hardware_controls": {}}
    eval_roots = [B.EVAL_ROOT]
    print("Each row: layered, single pass, difference (single minus layered).\n")
    for label, single, layered, comp, clean in MODELS:
        warn_duplicates(single, eval_roots); warn_duplicates(layered, eval_roots)
        a, miss_a = load(layered, eval_roots)
        b, miss_b = load(single, eval_roots)
        print(f"=== {label}: single pass vs {comp}" + ("" if clean else "  (margins set after this result; reference only)"))
        if miss_a or miss_b:
            print(f"    not ready: layered is missing {miss_a or 'nothing'}; single pass is missing {miss_b or 'nothing'}\n")
            out["models"][label] = {"status": "pending", "missing_layered": miss_a, "missing_single": miss_b}
            continue
        res = compare(a, b, R)
        v = verdict(res)
        short = [t for t in B.TASKS if res["n"][t] < 0.995 * max(res["n_a"][t], res["n_b"][t])]
        if short:
            print(f"    NOTE fewer than 99.5% of items in common for: {short}")
        report_failures(res)
        show(res, ["B_bal", "mmlu"])
        print(f"    decision: B_bal inside +/-1.5: {v['B_bal']['inside']}; MMLU-Pro inside +/-2: {v['mmlu']['inside']}; equivalent under the rule: {v['equivalent']}")
        print("    reported alongside, not used for the decision:")
        show(res, ["ten", "B_failed_as_wrong", "A_bal", "T_bal", "C", "Y"] + [t for t in B.TASKS if t != "mmlu"])
        print()
        out["models"][label] = {"status": "scored", "single": single, "layered": layered, "comparator": comp,
                                "margins_preset": clean, "verdict": v, "result": res}

    print("=== Anchor: May per-item results vs the re-scored layered checkpoint (re-score minus May) ===")
    for label, _, layered, comp, _ in MODELS:
        a, _ = load(layered, [MAY_ROOT])
        b, _ = load(layered, eval_roots)
        common = [t for t in B.TASKS if t in a and t in b]
        if not common:
            print(f"  {label} {comp}: nothing to compare yet")
            continue
        res = compare(a, b, R, tasks=common)
        names = (["B_bal", "ten"] if "B_bal" in res["d"] else []) + common
        drift = [n for n in names if not (res["d"][n]["lo95"] <= 0 <= res["d"][n]["hi95"])]
        print(f"  {label} {comp}: {len(common)} of {len(B.TASKS)} tests; 95% interval excludes zero for: {drift or 'none'}")
        report_failures(res)
        show(res, ["B_bal", "ten", "mmlu"])
        out["anchors"][label] = res

    print("\n=== Hardware control: layered chain retrained on the single pass's hardware (all checkpoints scored on one stack) ===")
    names = ["B_bal", "mmlu", "ten", "A_bal", "C", "Y"]
    for label, single, retrained, published, hw_name in CONTROLS:
        loaded = [load(a, eval_roots) for a in (retrained, single, published)]
        if any(miss for _, miss in loaded):
            print(f"  {label}: not ready")
            continue
        (h, _), (s1, _), (p1, _) = loaded
        same_hw, hw = compare(h, s1, R), compare(p1, h, R)
        print(f"  {label}: layered on {hw_name}, single pass on {hw_name}, single minus layered")
        report_failures(same_hw)
        show(same_hw, names)
        print(f"  {label}: published layered (May), layered on {hw_name}, retrained minus published")
        report_failures(hw)
        show(hw, names)
        out["hardware_controls"][label] = {"hardware": hw_name, "single_minus_layered_same_hardware": same_hw, "retrained_minus_published": hw}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
