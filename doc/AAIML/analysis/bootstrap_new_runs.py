#!/usr/bin/env python3
"""Paired item bootstrap for the AAIML camera-ready, extended to the reviewer-requested controls.

Same scoring, aggregates, resampling scheme, and RNG seed as
doc/v1.5 (Computers & Security)/analysis/bootstrap_v21_32b.py (which produced the paper's paired
intervals), so with only the five published checkpoints present it reproduces those numbers.
Each additional checkpoint below is included automatically once its scored files exist under
SFT/test/responses/<alias>/ or SFT/eval/responses/<alias>/ (i.e. after its serve_and_bench run):

  1-pass         matched single-pass mixture over all four shards      (reviewer comment 1)
  1-pass lr5e-6  its learning-rate sensitivity run                     (optional)
  SkipTAA-CSE    Core -> CSE, attribution layer omitted                 (reviewer comment 1)
  SkipTAA-Recal  Core -> CSE -> Recal-32B, attribution layer omitted    (reviewer comment 1)
  Recal-32B sN   Stage-4 seed replicates (released run = seed 42)       (reviewer comment 5)

Paired differences are later-minus-earlier on the items every included checkpoint answered.
Seed replicates are also summarised as mean and sample SD of the point estimates across
{released run, seed replicates}; that SD is training variability, which item-bootstrap intervals
do not measure.

Usage: python bootstrap_new_runs.py [R]   (default R=2000; writes data/bootstrap_new_runs.json)
"""
import csv, glob, json, math, os, random, statistics, sys
from collections import OrderedDict

R_ROOT = os.environ.get("GLAUKOPIS_ROOT", "/Users/pietro/code/Glaukopis")
# Published per-item files were extracted under SFT/test/responses; the current pipeline
# (SFT/eval/utils/run_foundation_8b_baselines.sh) writes SFT/eval/responses/<alias>/. Search both.
ROOTS = [f"{R_ROOT}/SFT/test/responses", f"{R_ROOT}/SFT/eval/responses"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bootstrap_new_runs.json")
P = "athena-cti-sft-qwen25-32b-v21-"
PUBLISHED = [("Core", P + "core-vllm"), ("+TAA", P + "taa-vllm"), ("CSE", P + "cse-vllm"),
             ("Recal-14B", P + "recalibrate-vllm"), ("Recal-32B", P + "recal-32b-vllm")]
CANDIDATES = [("1-pass", P + "single-stage-vllm"), ("1-pass lr5e-6", P + "single-stage-lr5e-06-vllm"),
              ("SkipTAA-CSE", P + "skiptaa-cse-vllm"), ("SkipTAA-Recal", P + "skiptaa-recal-32b-vllm"),
              ("Recal-32B s1", P + "recal-32b-seed1-vllm"), ("Recal-32B s2", P + "recal-32b-seed2-vllm"),
              ("Recal-32B s3", P + "recal-32b-seed3-vllm")]
PAIRS = [("Core", "+TAA"), ("+TAA", "CSE"), ("CSE", "Recal-32B"), ("CSE", "Recal-14B"), ("Recal-14B", "Recal-32B"),
         ("Core", "Recal-32B"), ("+TAA", "Recal-32B"),
         # reviewer-requested controls: positive = the layered chain scores higher
         ("1-pass", "Recal-32B"), ("1-pass lr5e-6", "Recal-32B"), ("1-pass", "CSE"),
         ("SkipTAA-CSE", "CSE"), ("SkipTAA-Recal", "Recal-32B"), ("Core", "1-pass"), ("Core", "SkipTAA-Recal")]
SEED_GROUP = ["Recal-32B", "Recal-32B s1", "Recal-32B s2", "Recal-32B s3"]
VSP_DENOM = 7.7
TASKS = ["athena-mcq", "athena-ate", "athena-rcm", "athena-rms", "athena-vsp", "athena-taa", "athena-taa-canonical",
         "cybersoceval-malware", "cybersoceval-ti", "cm2000", "cm10000", "mmlu"]
AGGS = ("A_bal", "B_bal", "T_bal", "C", "Y")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def one(d, rel):
    for root in ROOTS:
        hits = glob.glob(f"{root}/{d}/{rel}")
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{d}/{rel}")


def load_task(d, task):
    if task in ("cm2000", "cm10000"):
        stem = "CyberMetric-2000-v1" if task == "cm2000" else "CyberMetric-10000-v1"
        items, seen = {}, {}
        with open(one(d, f"cybermetric/cybermetric_{stem}_all_v1_*_response.csv"), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                q = (r["question"] or "").strip(); c = seen.get(q, 0); seen[q] = c + 1
                ok = (r["answer"] or "").strip() == (r["answer_after_post_processing"] or "").strip()
                items[(q, c)] = dict(score=1.0 if ok else 0.0, success=True)
        return items
    if task == "mmlu":
        items = {}
        with open(one(d, "mmlu-pro/*.csv"), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ok = (r["ground_truth"] or "").strip() == (r["prediction"] or "").strip()
                items[r["question_id"]] = dict(score=1.0 if ok else 0.0, success=True)
        return items
    items = {}
    for r in load_jsonl(one(d, f"{task}/*_scored.jsonl")):
        s = r["score"]; suc = bool(r.get("success", True))
        if task in ("athena-taa", "athena-taa-canonical"):
            val = dict(score=float(s["combined"]), correct=s["correct"], plausible=s["plausible"], success=suc)
        elif task == "athena-rms":
            val = dict(score=float(s["combined_f1"]), f1=float(s["f1"]), success=suc)
        elif task == "athena-vsp":
            val = dict(score=float(s) if suc else None, success=suc)
        else:
            val = dict(score=float(s), success=suc)
        items[r.get("id")] = val
    return items


def metric(task, vals):
    n = len(vals)
    if task == "athena-vsp":
        ok = [v["score"] for v in vals if v["success"]]
        return {"acc": 100 * (1 - (sum(ok) / len(ok) if ok else 0.0) / VSP_DENOM), "n_fail": n - len(ok)}
    if task.startswith("cybersoceval"):
        ok = [v for v in vals if v["success"]]
        return {"acc": 100 * sum(1 for v in ok if v["score"] == 1.0) / len(ok) if ok else 0.0,
                "jaccard": 100 * sum(v["score"] for v in ok) / len(ok) if ok else 0.0}
    if task in ("athena-taa", "athena-taa-canonical"):
        return {"acc": 100 * sum(v["score"] for v in vals) / n}
    return {"acc": 100 * sum(v["score"] for v in vals) / n}


def aggregates(m):
    T = (m["athena-taa"]["acc"] + m["athena-taa-canonical"]["acc"]) / 2
    A = (m["athena-mcq"]["acc"] + m["athena-ate"]["acc"] + m["athena-rcm"]["acc"] + m["athena-rms"]["acc"]
         + m["athena-vsp"]["acc"] + T) / 6
    C = (m["cybersoceval-malware"]["acc"] + m["cybersoceval-malware"]["jaccard"] + m["cybersoceval-ti"]["acc"]) / 3
    Y = (m["cm2000"]["acc"] + m["cm10000"]["acc"]) / 2
    return {"T_bal": T, "A_bal": A, "C": C, "Y": Y, "B_bal": (A + C + Y) / 3}


def ci(vals, lo=2.5, hi=97.5):
    s = sorted(vals); n = len(s)
    return s[int(math.floor(lo / 100 * (n - 1)))], s[int(math.ceil(hi / 100 * (n - 1)))]


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    CK = OrderedDict(PUBLISHED); data = {}
    for lab, d in PUBLISHED:
        data[lab] = {t: load_task(d, t) for t in TASKS}
    for lab, d in CANDIDATES:
        try:
            data[lab] = {t: load_task(d, t) for t in TASKS}; CK[lab] = d
            print(f"included {lab} ({d})")
        except FileNotFoundError as e:
            print(f"skipped  {lab}: no scored files yet ({e})")
    keys = {}
    for t in TASKS:
        ks = set(data["Core"][t])
        for ck in CK:
            ks &= set(data[ck][t])
        keys[t] = sorted(ks, key=lambda k: (str(type(k)), k))
        print(f"{t}: n_common={len(keys[t])} per-checkpoint n={[len(data[ck][t]) for ck in CK]}")

    point = {ck: {t: metric(t, [data[ck][t][k] for k in keys[t]]) for t in TASKS} for ck in CK}
    for ck in CK:
        point[ck]["agg"] = aggregates(point[ck])
    random.seed(20260906)  # same seed as bootstrap_v21_32b.py
    boot = {ck: {t: [] for t in TASKS + ["agg"]} for ck in CK}
    for _ in range(R):
        idx = {t: [random.randrange(len(keys[t])) for _ in range(len(keys[t]))] for t in TASKS}
        for ck in CK:
            m = {}
            for t in TASKS:
                ks = keys[t]; items = data[ck][t]
                m[t] = metric(t, [items[ks[i]] for i in idx[t]]); boot[ck][t].append(m[t])
            boot[ck]["agg"].append(aggregates(m))

    out = {"R": R, "checkpoints": dict(CK), "n": {t: len(keys[t]) for t in TASKS}, "point": point, "paired": {}}
    print("\n=== Point estimates (common items) ===")
    for ck in CK:
        a = point[ck]["agg"]
        print(f"{ck:15s} " + "  ".join(f"{k}={a[k]:.2f}" for k in AGGS))
    print("\n=== Paired differences (later minus earlier), 95% CI ===")
    for a, b in PAIRS:
        if a not in CK or b not in CK:
            continue
        res = {}
        for m in AGGS:
            d = [y[m] - x[m] for x, y in zip(boot[a]["agg"], boot[b]["agg"])]; lo, hi = ci(d)
            res[m] = {"d": point[b]["agg"][m] - point[a]["agg"][m], "lo": lo, "hi": hi}
        for t in TASKS:
            d = [y["acc"] - x["acc"] for x, y in zip(boot[a][t], boot[b][t])]; lo, hi = ci(d)
            res[t] = {"d": point[b][t]["acc"] - point[a][t]["acc"], "lo": lo, "hi": hi}
        out["paired"][f"{b} - {a}"] = res
        print(f"{b} - {a}: " + " | ".join(f"{m}={res[m]['d']:+.2f}[{res[m]['lo']:+.2f},{res[m]['hi']:+.2f}]" for m in AGGS))

    seeds = [ck for ck in SEED_GROUP if ck in CK]
    if len(seeds) > 1:
        sv = {}
        for m in AGGS:
            v = [point[ck]["agg"][m] for ck in seeds]
            sv[m] = {"mean": statistics.mean(v), "sd": statistics.stdev(v), "min": min(v), "max": max(v)}
        for t in TASKS:
            v = [point[ck][t]["acc"] for ck in seeds]
            sv[t] = {"mean": statistics.mean(v), "sd": statistics.stdev(v), "min": min(v), "max": max(v)}
        out["seed_variability"] = {"runs": seeds, "stats": sv}
        print(f"\n=== Seed variability over {len(seeds)} Stage-4 runs (sample SD of point estimates) ===")
        print("  ".join(f"{m}={sv[m]['mean']:.2f}±{sv[m]['sd']:.2f}" for m in AGGS))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
