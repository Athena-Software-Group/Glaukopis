#!/usr/bin/env python3
"""Regenerate the numbers used in glaukopis_cs_v1.5.tex from primary artifacts.

Sources (read-only):
  * SFT/eval/Glaukopis Results.xlsx           cached cell values (formulas were audited separately)
  * SFT/eval/responses/cost_summary.csv       tracked cost ledger
  * SFT/test/responses/<ckpt>/summary_*_all_v1.json   per-suite run manifests (local, untracked)
  * _v21_{core,taa,cse}_build/{watcher_status,dedup_report}.json   build reports

Writes CSV tables into ../data/ and prints them. No model inference.
"""
from __future__ import annotations
import csv, json, glob, os, sys
from pathlib import Path
import openpyxl

ROOT = Path("/Users/pietro/code/Glaukopis")
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
GPU_RATE, GPU_COUNT = 2.50, 2

def write(name, header, rows):
    p = OUT / name
    with p.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"\n### {name}"); print(",".join(header))
    for r in rows: print(",".join(str(x) for x in r))

# ---------------------------------------------------------------- aggregates
def T(cl, can): return (cl + can) / 2
def A(k, e, r, m, v, t): return (k + e + r + m + v + t) / 6
def C3(mal_acc, mal_jac, ti_acc): return (mal_acc + mal_jac + ti_acc) / 3
def C4(mal_acc, mal_jac, ti_acc, ti_jac): return (mal_acc + mal_jac + ti_acc + ti_jac) / 4
def B(a, c, y): return (a + c + y) / 3

wb = openpyxl.load_workbook(ROOT / "SFT/eval/Glaukopis Results.xlsx", data_only=True)
ws = wb["Final wMMLUPro-v21"]
def cell(r, col): return ws[f"{col}{r}"].value
rows_of_interest = {
    5: "GPT-5.5", 6: "Gemini-3-Flash-Preview", 7: "DeepSeek-V4-Pro", 8: "GPT-5.2 (medium)",
    9: "DeepSeek-V3.2-Exp", 10: "Gemini-2.5-Flash", 12: "Recal-32B", 13: "CSE", 14: "Recal-14B",
    15: "Core", 16: "+TAA", 17: "Qwen2.5-32B base",
    19: "MoE cse", 24: "MoE base", 26: "14B taa", 27: "14B recalibrate", 28: "14B core", 29: "14B cse", 30: "14B base",
    32: "Llama core", 33: "Llama recalibrate", 34: "Llama taa", 35: "Llama cse", 36: "Llama base",
    38: "Foundation core", 39: "Foundation recalibrate", 40: "Foundation base", 41: "Foundation cse", 42: "Foundation taa",
}
agg_rows = []
for r, name in rows_of_interest.items():
    k, e, rc = cell(r, "G"), cell(r, "H"), cell(r, "I")
    rms_strict, rms_comb = cell(r, "J"), cell(r, "L")
    v = cell(r, "M"); taa_cl = cell(r, "P"); taa_can = cell(r, "S")
    mal_acc, mal_jac, ti_acc, ti_jac = cell(r, "W"), cell(r, "X"), cell(r, "Y"), cell(r, "Z")
    cm2, cm10 = cell(r, "AB"), cell(r, "AC"); mmlu = cell(r, "AE")
    t = T(taa_cl, taa_can); a = A(k, e, rc, rms_comb, v, t)
    c3 = C3(mal_acc, mal_jac, ti_acc); c4 = C4(mal_acc, mal_jac, ti_acc, ti_jac); y = (cm2 + cm10) / 2
    b3 = B(a, c3, y); b4 = B(a, c4, y)
    ten = (k + e + rc + rms_strict + v + t + mal_acc + ti_acc + cm2 + cm10) / 10
    nine = (mmlu + k + e + rc + rms_comb + v + t + c3 + y) / 9
    ab_sheet = cell(r, "V"); e_sheet = cell(r, "E")
    agg_rows.append([name, k, e, rc, rms_comb, v, taa_cl, taa_can, round(t, 3), round(a, 3), round(c3, 3), round(c4, 3), round(y, 3),
                     round(b3, 3), round(b4, 3), round(ten, 3), round(nine, 3), mmlu,
                     round(ab_sheet, 3) if isinstance(ab_sheet, (int, float)) else ab_sheet,
                     round(e_sheet, 3) if isinstance(e_sheet, (int, float)) else e_sheet])
write("aggregates_uniform_convention.csv",
      ["model", "CKT", "ATE", "RCM", "RMS_comb", "VSP", "TAA_classic", "TAA_canonical", "T_bal", "A_bal", "C3", "C4", "Y",
       "B_bal_C3", "B_bal_C4", "ten_test_mean", "nine_component_mean", "MMLU_Pro", "sheet_AB_AvgII_asis", "sheet_AvgCTIBench_asis"], agg_rows)

# ordering check: does the C4 variant change the ordering of the reported rows?
rep = [r for r in agg_rows if r[0] in ("GPT-5.5", "Gemini-3-Flash-Preview", "DeepSeek-V4-Pro", "GPT-5.2 (medium)", "DeepSeek-V3.2-Exp",
                                        "Gemini-2.5-Flash", "Recal-32B", "Qwen2.5-32B base", "CSE", "Recal-14B", "Core", "+TAA")]
o3 = [r[0] for r in sorted(rep, key=lambda r: -r[13])]; o4 = [r[0] for r in sorted(rep, key=lambda r: -r[14])]
print("\nordering under C3:", o3); print("ordering under C4:", o4); print("same ordering:", o3 == o4)
print("max |B_C4 - B_C3| over reported rows:", max(abs(r[14] - r[13]) for r in rep))
base = next(r for r in rep if r[0] == "Qwen2.5-32B base"); ship = next(r for r in rep if r[0] == "Recal-32B")
print("gain C3:", round(ship[13] - base[13], 3), "gain C4:", round(ship[14] - base[14], 3))

# ---------------------------------------------------------------- run-record costs
CK = {"Core": "athena-cti-sft-qwen25-32b-v21-core-vllm", "+TAA": "athena-cti-sft-qwen25-32b-v21-taa-vllm",
      "CSE": "athena-cti-sft-qwen25-32b-v21-cse-vllm", "Recal-14B": "athena-cti-sft-qwen25-32b-v21-recalibrate-vllm",
      "Recal-32B": "athena-cti-sft-qwen25-32b-v21-recal-32b-vllm"}
ledger = {r["model"]: r for r in csv.DictReader(open(ROOT / "SFT/eval/responses/cost_summary.csv"))}
cost_rows = []
for lab, d in CK.items():
    fam = {"athena": 0, "cybersoceval": 0, "cybermetric": 0, "mmlu_pro": 0}; dates = []
    for f in sorted(glob.glob(str(ROOT / "SFT/test/responses" / d / "summary_*_all_v1.json"))):
        s = json.load(open(f)); dates.append((s["started"], s["finished"]))
        for t in s["tasks"]:
            k = t["task"]
            fam["cybermetric" if k.startswith("cybermetric") else "athena" if k.startswith("athena") else "cybersoceval" if k.startswith("cybersoceval") else "mmlu_pro"] += t["elapsed_sec"]
    usd = {k: v / 3600 * GPU_COUNT * GPU_RATE for k, v in fam.items()}
    tot = sum(usd.values()); cti = usd["athena"] + usd["cybersoceval"] + usd["cybermetric"]
    led = ledger[d]
    cost_rows.append([lab, sum(fam.values()), round(usd["athena"], 4), round(usd["cybersoceval"], 4), round(usd["cybermetric"], 4), round(usd["mmlu_pro"], 4),
                      round(cti, 4), round(tot, 4), led["total_cost_usd"], round(float(led["total_cost_usd"]) / tot, 3), min(x[0] for x in dates), max(x[1] for x in dates)])
write("local_cost_run_records.csv", ["checkpoint", "elapsed_sec_total", "athena_usd", "cybersoceval_usd", "cybermetric_usd", "mmlu_pro_usd",
                                     "cti_only_usd", "total_usd", "ledger_total_usd", "ledger_over_runrecord", "first_started", "last_finished"], cost_rows)

# ---------------------------------------------------------------- build reports
br = []
for s in ("core", "taa", "cse"):
    st = json.load(open(ROOT / f"_v21_{s}_build/watcher_status.json"))["detail"]
    d = json.load(open(ROOT / f"_v21_{s}_build/dedup_report.json"))
    mx = [max(c for _, c in h["matches"]) for h in d]
    hist = {"1": sum(1 for m in mx if m == 1), "2-4": sum(1 for m in mx if 2 <= m <= 4), "5-12": sum(1 for m in mx if 5 <= m <= 12),
            "13-24": sum(1 for m in mx if 13 <= m <= 24), "25-49": sum(1 for m in mx if 25 <= m <= 49), ">=50": sum(1 for m in mx if m >= 50)}
    br.append([s, st.get("balanced_rows"), len(d), hist[">=50"], st.get("clean_rows"), st.get("train_rows"), st.get("val_rows"),
               st.get("phase_a_rows", ""), st.get("phase_b_rows", ""), json.dumps(hist)])
write("build_reports_v21.csv", ["shard", "rows_scanned", "flagged", "removed", "clean_rows", "train_rows", "val_rows", "phase_a_rows", "phase_b_rows", "flag_histogram"], br)
print("\nDone. Tables written to", OUT)
