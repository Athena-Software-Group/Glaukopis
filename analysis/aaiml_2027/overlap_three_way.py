#!/usr/bin/env python3
"""Three-way overlap split for the AAIML camera-ready overlap table (reviewer comment 4).

For each Benchmark-A (AthenaBench) mapping task, every item is assigned to exactly one group:
  none     -- the item's identifying entity never appears in a training row;
  entity   -- the entity appears in some training row, but never together with the complete gold answer;
  answer   -- the entity and the complete gold answer co-occur in one training row.
Items the scorer excludes (unparseable VSP vectors) are counted separately. Mean item scores are
reported per group for every retained 32B checkpoint.

Matching rules are those of doc/v1.5 (Computers & Security)/analysis/overlap_subsets.py, extended
with complete-answer matching for VSP (CVE identifier and the exact gold CVSS vector in one row).
Reads only existing artifacts: the v21 training shards, the AthenaBench task files, and the
per-item scored files under SFT/test/responses/. Writes data/overlap_three_way.json.
"""
import collections, glob, json, os, re

R = os.environ.get("GLAUKOPIS_ROOT", "/Users/pietro/code/Glaukopis")
BD = f"{R}/SFT/eval/benchmark_data/athena_bench"
RESP = f"{R}/SFT/test/responses"          # May per-item files of the layered checkpoints
RESP_NOV = f"{R}/SFT/eval/responses"     # November runs (single pass)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "overlap_three_way.json")

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

CVE = re.compile(r"CVE-\d{4}-\d{4,}"); CWE = re.compile(r"CWE-\d+")
TID = re.compile(r"T\d{4}(?:\.\d{3})?"); MID = re.compile(r"M\d{4}")
VEC = re.compile(r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]")

def word_in(tl, nl):
    return re.search(r"(?:^|[^a-z0-9])" + re.escape(nl) + r"(?:$|[^a-z0-9])", tl) is not None

rcm = jl(f"{BD}/athena-cti-rcm.jsonl"); rms = jl(f"{BD}/athena-cti-rms.jsonl"); vsp = jl(f"{BD}/athena-cti-vsp.jsonl")
taac = jl(f"{BD}/athena_taa_canonical/athena-cti-taa-canonical.jsonl")
ate = jl(f"{BD}/athena-cti-ate.jsonl")

def gold_vec(r):
    m = VEC.search(r.get("answer", "") or r.get("cvss_vector", "") or "")
    return m.group(0).replace("CVSS:3.0/", "CVSS:3.1/") if m else None

items = {  # task -> list of (prompt, entity, answer-key)
    "athena-rcm": [(r["prompt"], r["cve_id"], r["answer"].strip()) for r in rcm],
    "athena-vsp": [(r["prompt"], r["cve_id"], gold_vec(r)) for r in vsp],
    "athena-rms": [(r["prompt"], r["technique_id"], tuple(sorted(set(MID.findall(r["answer"]))))) for r in rms],
    "athena-taa-canonical": [(r["prompt"], (r.get("alias_used") or r.get("alias") or "").strip(), r["answer"].strip()) for r in taac],
}
ent = {t: {e for _, e, _ in v} for t, v in items.items()}
seen_entity = collections.defaultdict(set); seen_answer = collections.defaultdict(set)
alias_l = {a.lower(): a for a in ent["athena-taa-canonical"] if a and len(a) >= 3}
taa_answer = {a: c for _, a, c in items["athena-taa-canonical"]}
rms_by_tid = collections.defaultdict(set)
for _, t, ms in items["athena-rms"]:
    rms_by_tid[t].add(ms)
vsp_gold = {e: g for _, e, g in items["athena-vsp"]}
ate_tid_rows = collections.Counter()  # ATE: training rows naming each technique id (for frequency tertiles)
rcm_gold = collections.defaultdict(set)
for _, c, w in items["athena-rcm"]:
    rcm_gold[c].add(w)

for shard in ["core", "taa", "cse"]:
    for r in json.load(open(f"{R}/SFT/data/ift_data_2026_05_18_v21_{shard}.json", encoding="utf-8")):
        txt = " ".join(str(r.get(k, "")) for k in ("instruction", "input", "output")); tl = txt.lower()
        cves = set(CVE.findall(txt))
        for c in cves & ent["athena-rcm"]:
            seen_entity["athena-rcm"].add(c)
            for w in set(CWE.findall(txt)) & rcm_gold[c]:
                seen_answer["athena-rcm"].add((c, w))
        if cves & ent["athena-vsp"]:
            vecs = {v.replace("CVSS:3.0/", "CVSS:3.1/") for v in VEC.findall(txt)}
            for c in cves & ent["athena-vsp"]:
                seen_entity["athena-vsp"].add(c)
                if vsp_gold.get(c) in vecs:
                    seen_answer["athena-vsp"].add((c, vsp_gold[c]))
        for t in set(TID.findall(txt)):
            ate_tid_rows[t] += 1
        tids = set(TID.findall(txt)) & ent["athena-rms"]
        if tids:
            mids = set(MID.findall(txt))
            for t in tids:
                seen_entity["athena-rms"].add(t)
                for ms in rms_by_tid[t]:
                    if set(ms) <= mids:
                        seen_answer["athena-rms"].add((t, ms))
        for al, a in alias_l.items():
            if al in tl and word_in(tl, al):
                seen_entity["athena-taa-canonical"].add(a)
                if word_in(tl, taa_answer[a].lower()):
                    seen_answer["athena-taa-canonical"].add((a, taa_answer[a]))

group = {}
for task, lst in items.items():
    for p, e, a in lst:
        group[(task, p)] = "answer" if (e, a) in seen_answer[task] else ("entity" if e in seen_entity[task] else "none")

# ATE has no answer-exposure split; items are grouped by the training frequency of the gold technique (tertiles).
cnts = sorted(ate_tid_rows[r["technique_id"]] for r in ate); q1 = cnts[len(cnts) // 3]; q2 = cnts[2 * len(cnts) // 3]
for r in ate:
    n = ate_tid_rows[r["technique_id"]]
    group[("athena-ate", r["prompt"])] = "low" if n <= q1 else ("mid" if n <= q2 else "high")
items["athena-ate"] = [(r["prompt"], r["technique_id"], None) for r in ate]

CK = [("Core", "core"), ("+TAA", "taa"), ("CSE", "cse"), ("Recal-14B", "recalibrate"), ("Recal-32B", "recal-32b"), ("1-pass", "single-stage")]
res = {"definitions": __doc__.strip().split("\n\n")[1], "items": {}, "scores": {}}
for task, lst in items.items():
    res["items"][task] = dict(collections.Counter(group[(task, p)] for p, _, _ in lst))
for task in items:
    for lab, d in CK:
        path = (glob.glob(f"{RESP}/athena-cti-sft-qwen25-32b-v21-{d}-vllm/{task}/*_scored.jsonl")
                or glob.glob(f"{RESP_NOV}/athena-cti-sft-qwen25-32b-v21-{d}-vllm/{task}/*_scored.jsonl"))[0]
        agg = collections.defaultdict(list); excluded = collections.Counter(); unmatched = 0
        for r in jl(path):
            g = group.get((task, r.get("prompt")))
            if g is None:
                unmatched += 1; continue
            sc = r["score"]
            if task == "athena-rms": v = sc["combined_f1"]
            elif task == "athena-taa-canonical": v = sc["combined"]
            elif task == "athena-vsp":
                if not r.get("success", True):
                    excluded[g] += 1; continue
                v = 1 - float(sc) / 7.7
            else: v = float(sc)
            agg[g].append(v)
        res["scores"][f"{task}|{lab}"] = {
            **{g: {"n": len(v), "mean": round(100 * sum(v) / len(v), 1)} for g, v in agg.items()},
            "excluded_unparseable": dict(excluded), "unmatched_prompts": unmatched}

print(json.dumps(res["items"], indent=1))
for k, v in res["scores"].items():
    if k.endswith("Recal-32B") or k.endswith("|+TAA") or k.endswith("|1-pass"):
        print(k, v)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", os.path.normpath(OUT))
