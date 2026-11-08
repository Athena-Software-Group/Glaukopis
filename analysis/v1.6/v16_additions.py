#!/usr/bin/env python3
"""Reproduce every number that is new in manuscript v1.6 relative to v1.5.

Run from anywhere; set GLAUKOPIS_ROOT if the repository is not at the default path.
Reads only existing artifacts (no training, no inference):
  - SFT/data/ift_data_2026_05_18_v21_{taa,cse}.json           (training shards)
  - SFT/eval/benchmark_data/athena_bench/athena_taa_canonical/athena-cti-taa-canonical.jsonl
  - SFT/test/responses/athena-cti-sft-qwen25-32b-v21-*-vllm/athena-vsp/*_scored.jsonl
  - SFT/eval/Glaukopis Results.xlsx   (sheets 'Final wMMLUPro-v21', 'Glaukopis (TEST)', 'MMLU Pro', 'Cost')

Outputs (printed and written to data/v16_additions.json):
  1. Table 12 (shard output/input profile)            -> shard_profile
  2. Table 14 metric-level accuracy and error direction -> vsp_metric_level, vsp_direction
  3. Section 8.7 MMLU-Pro per-subject deltas           -> mmlu_subject_delta
  4. Table 7 single-stage reference run                 -> single_stage
  5. Table 17 token / wall-clock columns                -> efficiency
"""
import collections, glob, json, os, re, sys

R = os.environ.get("GLAUKOPIS_ROOT", "/Users/pietro/code/Glaukopis")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "v16_additions.json")
res = {}

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ---------------------------------------------------------------- 1. shard profile
G = re.compile(r"\bG\d{4}\b"); T = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
taac = jl(f"{R}/SFT/eval/benchmark_data/athena_bench/athena_taa_canonical/athena-cti-taa-canonical.jsonl")
groups = sorted({r["answer"].strip() for r in taac})
groups_l = [g.lower() for g in groups if len(g) >= 4]
prof = {}
for shard in ["taa", "cse"]:
    rows = json.load(open(f"{R}/SFT/data/ift_data_2026_05_18_v21_{shard}.json", encoding="utf-8"))
    n = len(rows); c = collections.Counter(); lens = []
    for r in rows:
        out = str(r.get("output", "")); ol = out.lower()
        inp = (str(r.get("instruction", "")) + " " + str(r.get("input", ""))).lower()
        c["out_gcode"] += bool(G.search(out))
        c["out_group"] += any(g in ol for g in groups_l)
        c["out_tcode"] += bool(T.search(out))
        c["in_group"] += any(g in inp for g in groups_l)
        lens.append(len(out.split()))
    lens.sort()
    prof[shard] = {"rows": n, **{k: v for k, v in c.items()}, "median_output_words": lens[n // 2]}
res["shard_profile"] = prof
print("Shard profile:", json.dumps(prof, indent=1))

# ---------------------------------------------------------------- 2. VSP metric-level accuracy and direction
VEC = re.compile(r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]")
def parse(v):
    m = VEC.search(v or "")
    return dict(p.split(":") for p in m.group(0).split("/")[1:]) if m else None
CK = [("Core", "core"), ("+TAA", "taa"), ("CSE", "cse"), ("Recal-14B", "recalibrate"), ("Recal-32B", "recal-32b")]
METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
vsp = {}; direction = {}
for lab, d in CK:
    p = glob.glob(f"{R}/SFT/test/responses/athena-cti-sft-qwen25-32b-v21-{d}-vllm/athena-vsp/*_scored.jsonl")[0]
    rows = jl(p); n = 0; unparsed = 0; exact = 0; correct = 0
    mism = {k: 0 for k in METRICS}; conf = {k: collections.Counter() for k in ["AV", "PR", "S"]}
    for r in rows:
        g = parse(r["answer"]); pr = parse(r["response"])
        if g is None: continue
        n += 1
        if pr is None: unparsed += 1; continue
        m = sum(g[k] == pr[k] for k in METRICS); correct += m; exact += (m == 8)
        for k in METRICS:
            if g[k] != pr[k]:
                mism[k] += 1
                if k in conf: conf[k][f"{g[k]}->{pr[k]}"] += 1
    vsp[lab] = {"n": n, "unparsed": unparsed, "exact": exact,
                "metric_level_acc_pct": round(100 * correct / (8 * n), 1), "mismatches": mism}
    direction[lab] = {k: dict(v.most_common(4)) for k, v in conf.items()}
res["vsp_metric_level"] = vsp; res["vsp_direction"] = direction
print("VSP:", json.dumps(vsp, indent=1)); print("VSP direction:", json.dumps(direction, indent=1))

# ---------------------------------------------------------------- 3-5. workbook-derived tables
try:
    import openpyxl
except ImportError:
    print("openpyxl not available; skipping workbook-derived tables", file=sys.stderr); openpyxl = None
if openpyxl:
    wb = openpyxl.load_workbook(f"{R}/SFT/eval/Glaukopis Results.xlsx", data_only=True)
    # MMLU-Pro subjects: rows keyed by model name in column B; subject columns C..P, overall in column R
    ws = wb["MMLU Pro"]; hdr = [str(c.value).strip() if c.value else "" for c in ws[1]]
    rows = {str(r[1]).strip(): r for r in ws.iter_rows(min_row=2, values_only=True) if r[1]}
    base = rows["Qwen_Qwen2.5-32B-Instruct"]; fin = rows["athena-cti-sft-qwen25-32b-v21-recal-32b-vllm"]
    subj = {hdr[i]: round(float(base[i]) - float(fin[i]), 1) for i in range(2, 16) if base[i] not in (None, "") and fin[i] not in (None, "")}
    res["mmlu_subject_delta_base_minus_final"] = subj; print("MMLU-Pro subject drop (base - final):", subj)
    # Single-stage reference run (Glaukopis (TEST) sheet) and chain final (Final wMMLUPro-v21 sheet)
    def row(sheet, name):
        for r in wb[sheet].iter_rows(values_only=True):
            if r and r[0] and str(r[0]).strip() == name: return r
    v7 = row("Glaukopis (TEST)", "asg-ai/athena-cti-sft-qwen25-32b-abaligned-v7")
    # columns in the TEST sheet: 2 CKT, 3 ATE, 4 RCM, 7 RMS combined, 8 VSP, 11 TAA classic combined, 17-19 CSE (mal acc, mal avg, ti acc), 22-23 CM tiers
    ss = dict(CKT=v7[2], ATE=v7[3], RCM=v7[4], RMS=v7[7], VSP=v7[8], TAA_classic=v7[11],
              C=round((v7[17] + v7[18] + v7[19]) / 3, 1), Y=round((v7[22] + v7[23]) / 2, 1))
    ss["A_classic"] = round(sum([ss[k] for k in ["CKT", "ATE", "RCM", "RMS", "VSP", "TAA_classic"]]) / 6, 1)
    ss["B_classic"] = round((ss["A_classic"] + ss["C"] + ss["Y"]) / 3, 1)
    res["single_stage_v7_32b"] = ss; print("Single-stage v7 32B:", ss)
    # Efficiency columns (Cost sheet): input/output tokens for hosted rows
    eff = {}
    for r in wb["Cost"].iter_rows(min_row=2, values_only=True):
        if r and r[0]: eff[str(r[0])] = {"input_tok": r[6], "output_tok": r[7], "wallclock_h": r[9], "total_usd": r[13] if len(r) > 13 else None}
    res["efficiency"] = eff

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("written", OUT)
