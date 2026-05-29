#!/usr/bin/env python
"""ID-level leakage check: scan eval files for CVE-xxxx-xxxx, T\\d{4}, M\\d{4},
CWE-\\d+, CAPEC-\\d+ identifiers; then for each training row, count how many
of those eval IDs appear, classifying severity:

  HIGH:  same CVE-id appears in both train_input and eval_input (full leakage)
  MED:   same technique/mitigation ID appears in both train_OUTPUT and eval ANSWER
  LOW:   ID appears in train_input but eval question is about a DIFFERENT thing
"""
import json, re, pathlib, sys
from collections import defaultdict, Counter

ID_RE = re.compile(r"\b(?:CVE-\d{4}-\d{4,7}|CWE-\d+|CAPEC-\d+|T\d{4}(?:\.\d{3})?|M\d{4}|G\d{4}|S\d{4})\b")

def ids_in(s: str) -> set[str]:
    return set(ID_RE.findall(s or ""))

train_path = sys.argv[1] if len(sys.argv) > 1 else "SFT/data/ift_data_2026_04_30_v81.subsampled.json"
eval_dir = pathlib.Path("SFT/eval/benchmark_data")

# 1. Index eval IDs per file (input-IDs and answer-IDs separately)
eval_input_ids = defaultdict(set)   # file -> set of IDs in question/desc
eval_answer_ids = defaultdict(set)  # file -> set of IDs in answer
all_eval_input = set()
all_eval_answer = set()
for ef in eval_dir.rglob("*.jsonl"):
    with open(ef) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            qtxt = " ".join(str(rec.get(k, "")) for k in
                            ("question", "Question", "prompt", "report",
                             "context", "input", "instruction", "description",
                             "cve_id"))
            atxt = " ".join(str(rec.get(k, "")) for k in
                            ("answer", "answers", "ground_truth", "label", "output"))
            eval_input_ids[ef.name] |= ids_in(qtxt)
            eval_answer_ids[ef.name] |= ids_in(atxt)
            all_eval_input |= ids_in(qtxt)
            all_eval_answer |= ids_in(atxt)
for ef in eval_dir.rglob("*.json"):
    if ef.suffix == ".jsonl": continue
    try:
        rec = json.load(open(ef))
    except Exception:
        continue
    if isinstance(rec, list):
        items = rec
    elif isinstance(rec, dict) and "data" in rec:
        items = rec["data"]
    else:
        items = [rec]
    for r in items:
        if not isinstance(r, dict): continue
        qtxt = " ".join(str(r.get(k, "")) for k in
                        ("question", "Question", "prompt", "report",
                         "context", "input", "instruction", "description",
                         "cve_id"))
        atxt = " ".join(str(r.get(k, "")) for k in
                        ("answer", "answers", "ground_truth", "label", "output"))
        eval_input_ids[ef.name] |= ids_in(qtxt)
        eval_answer_ids[ef.name] |= ids_in(atxt)
        all_eval_input |= ids_in(qtxt)
        all_eval_answer |= ids_in(atxt)

print(f"eval index: {len(eval_input_ids)} files")
print(f"  total unique eval input-IDs : {len(all_eval_input)}")
print(f"  total unique eval answer-IDs: {len(all_eval_answer)}")
print()

# Break down eval IDs by type
def by_type(s):
    c = Counter()
    for i in s:
        if i.startswith("CVE"): c["CVE"] += 1
        elif i.startswith("CWE"): c["CWE"] += 1
        elif i.startswith("CAPEC"): c["CAPEC"] += 1
        elif i.startswith("T"): c["TECH"] += 1
        elif i.startswith("M"): c["MIT"] += 1
        elif i.startswith("G"): c["GROUP"] += 1
        elif i.startswith("S"): c["SOFTWARE"] += 1
    return c

print(f"  eval input-ID types : {dict(by_type(all_eval_input))}")
print(f"  eval answer-ID types: {dict(by_type(all_eval_answer))}")
print()

# 2. Scan training data
data = json.load(open(train_path))
print(f"scanning {len(data)} training rows ...")

high_risk = []  # train input contains an ID that's also in eval ANSWER set (output-leak)
input_id_overlap = Counter()  # IDs present in both train_input and eval_input
for i, row in enumerate(data):
    in_ids = ids_in(row.get("input", "")) | ids_in(row.get("instruction", ""))
    out_ids = ids_in(row.get("output", ""))
    # severity HIGH: train OUTPUT (the answer) contains an ID that's in eval ANSWER set
    leaked = out_ids & all_eval_answer
    if leaked:
        # but only flag if the train INPUT also touches the eval question domain
        # (which it does for catalog drills by design)
        high_risk.append((i, row.get("shortname"), sorted(leaked)[:5]))
    for x in in_ids & all_eval_input:
        input_id_overlap[x] += 1

print(f"  rows where train.OUTPUT contains an eval-answer-ID: {len(high_risk)}")
print(f"  unique IDs appearing in both train.input and eval.input: {len(input_id_overlap)}")
print()
print("  top 15 most-shared IDs (train.input ∩ eval.input):")
for k, v in input_id_overlap.most_common(15):
    print(f"    {k:24s} {v:6d} train rows")
print()

# Group HIGH-risk hits by template + answer
print("HIGH severity (train.output ID in eval.answer set) by template + ID:")
by_template = Counter()
by_id = Counter()
for ridx, sn, ids in high_risk:
    by_template[sn] += 1
    for x in ids:
        by_id[x] += 1
for k, v in by_template.most_common(20):
    print(f"  {k:24s} {v:6d}")
print()
print("  top 15 leaked IDs:")
for k, v in by_id.most_common(15):
    print(f"    {k:24s} {v:6d}")
