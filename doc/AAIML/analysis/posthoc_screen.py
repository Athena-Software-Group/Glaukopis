#!/usr/bin/env python3
"""Post-hoc verbatim 13-gram screen for the AAIML camera-ready (reviewer comment 2).

The build-time screen (tmpl_gen/scripts/dedup_against_evals.py) did not index CyberSOCEval,
whose files were fetched only at evaluation time, and it never scanned the general instruction
data mixed into Stage 1A. This script closes both gaps after the fact, with the build-time
screen's own tokenizer and 13-gram definition:
  A. the three v21 Sophia shards (Core, TAA, CSE) against the CyberSOCEval question files;
  B. the general data actually trained in Stage 1A (first 240,000 Tulu-3 SFT rows and the 999
     Alpaca demo rows) against AthenaBench, CyberMetric, and CyberSOCEval.
For each scanned row the maximum number of distinct shared 13-grams with any single evaluation
item is bucketed as in the paper (1, 2-4, 5-12, 13-24, 25-49, >=50; >=50 is the build-time
removal threshold). Nothing is removed; the models were trained before this screen.

CyberSOCEval items are indexed on question text plus answer options (the build-time reader saw
neither the threat-intelligence `question_text` field nor any options). AthenaBench and
CyberMetric items use the build-time reader unchanged.

Usage: python posthoc_screen.py --cybersoceval-dir DIR --tulu P0 P1   (needs pyarrow)
"""
import argparse, collections, json, multiprocessing as mp, os, sys
from pathlib import Path

R = os.environ.get("GLAUKOPIS_ROOT", "/Users/pietro/code/Glaukopis")
sys.path.insert(0, f"{R}/tmpl_gen/scripts")
from dedup_against_evals import load_eval_records, ngrams, tokens  # noqa: E402

N = 13
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "posthoc_screen.json")
BANDS = [(1, 1, "1"), (2, 4, "2-4"), (5, 12, "5-12"), (13, 24, "13-24"), (25, 49, "25-49"), (50, 10**9, ">=50")]

def soc_records(path):
    recs = json.load(open(path, encoding="utf-8"))
    out = []
    for r in recs:
        q = r.get("question") or r.get("question_text") or ""
        opts = r.get("options") or []
        opts = list(opts.values()) if isinstance(opts, dict) else opts
        out.append(" \n ".join([q] + [str(o) for o in opts]))
    return out

def build_index(sources):
    idx = collections.defaultdict(list); sizes = {}
    for bench, files, reader in sources:
        n_items = 0
        for f in files:
            for i, txt in enumerate(reader(f)):
                n_items += 1
                for g in ngrams(tokens(txt), N):
                    idx[g].append(f"{bench}:{Path(f).name}:{i}")
        sizes[bench] = n_items
    return dict(idx), sizes

_IDX = None
def _scan(texts):
    res = []
    for t in texts:
        per = collections.Counter()
        for g in ngrams(tokens(t), N):
            for k in _IDX.get(g, ()):
                per[k] += 1
        if per:
            k, c = per.most_common(1)[0]
            res.append((c, k.split(":", 1)[0]))
        else:
            res.append((0, None))
    return res

def scan(texts, idx, workers=os.cpu_count() or 4):
    global _IDX
    _IDX = idx
    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    with mp.get_context("fork").Pool(workers) as pool:
        out = [r for part in pool.map(_scan, chunks) for r in part]
    summary = {"scanned": len(texts), "flagged": 0, "bands": {b: 0 for _, _, b in BANDS}, "by_benchmark_flagged": collections.Counter()}
    for c, bench in out:
        if c == 0:
            continue
        summary["flagged"] += 1; summary["by_benchmark_flagged"][bench] += 1
        for lo, hi, b in BANDS:
            if lo <= c <= hi:
                summary["bands"][b] += 1
    summary["by_benchmark_flagged"] = dict(summary["by_benchmark_flagged"])
    return summary

def sophia_texts(name):
    rows = json.load(open(f"{R}/SFT/data/{name}.json", encoding="utf-8"))
    return [" \n ".join(str(r.get(k, "")) for k in ("instruction", "input", "output")) for r in rows]

def tulu_texts(paths, limit=240000):
    import pyarrow.parquet as pq
    out = []
    for p in paths:
        for b in pq.ParquetFile(p).iter_batches(columns=["messages"], batch_size=4096):
            for msgs in b.column(0).to_pylist():
                if len(out) >= limit:
                    return out
                out.append(" \n ".join(m["content"] for m in msgs))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cybersoceval-dir", required=True)
    ap.add_argument("--tulu", nargs="+", required=True)
    a = ap.parse_args()
    bd = Path(f"{R}/SFT/eval/benchmark_data")
    soc = sorted(Path(a.cybersoceval_dir).glob("*.json"))
    athena = sorted({*bd.glob("athena_bench/**/*.jsonl"), *bd.glob("athena_bench/**/*.json")})
    cm = sorted({*bd.glob("cybermetricdataset/**/*.json"), *bd.glob("cybermetricdataset/**/*.jsonl")})
    res = {"n": N, "bands_note": ">=50 was the build-time removal threshold; nothing was removed here"}

    idx_soc, sz = build_index([("CyberSOCEval", soc, soc_records)])
    res["A_sophia_vs_cybersoceval"] = {"index_items": sz, "index_13grams": len(idx_soc)}
    for lab, name in [("Core (train)", "ift_data_2026_05_18_v21_core_train"), ("TAA", "ift_data_2026_05_18_v21_taa"),
                      ("CSE", "ift_data_2026_05_18_v21_cse")]:
        res["A_sophia_vs_cybersoceval"][lab] = scan(sophia_texts(name), idx_soc)
        print("A", lab, res["A_sophia_vs_cybersoceval"][lab])

    idx_all, sz = build_index([("AthenaBench", athena, load_eval_records), ("CyberMetric", cm, load_eval_records),
                               ("CyberSOCEval", soc, soc_records)])
    res["B_general_vs_all"] = {"index_items": sz, "index_13grams": len(idx_all),
                               "index_files": {"AthenaBench": [p.name for p in athena], "CyberMetric": [p.name for p in cm]}}
    alp = json.load(open(f"{R}/SFT/data/alpaca_en_demo.json", encoding="utf-8"))
    res["B_general_vs_all"]["Alpaca demo (999)"] = scan(
        [" \n ".join(str(r.get(k, "")) for k in ("instruction", "input", "output")) for r in alp], idx_all)
    res["B_general_vs_all"]["Tulu-3 SFT (first 240,000)"] = scan(tulu_texts(a.tulu), idx_all)
    for k in ("Alpaca demo (999)", "Tulu-3 SFT (first 240,000)"):
        print("B", k, res["B_general_vs_all"][k])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", os.path.normpath(OUT))

if __name__ == "__main__":
    main()
