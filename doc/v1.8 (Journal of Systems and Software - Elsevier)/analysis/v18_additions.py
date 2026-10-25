#!/usr/bin/env python3
"""Reproduce every number that is new in manuscript v1.8 (JSS) relative to v1.7 (IJIS).

Run from anywhere; set GLAUKOPIS_ROOT if the repository is not at the default path.
Reads only existing artifacts (no training, no inference):
  - SFT/data/ift_data_2026_04_26_combined_v7.json, SFT/data/alpaca_en_demo.json
  - SFT/data/ift_data_2026_05_18_v21_{core_a_kb_mcq_taa_soc_cm_ms_yn,core_b_rms_ate_vsp_rcm,taa,cse}.json
  - SFT/eval/Glaukopis Results.xlsx  (sheets 'Glaukopis Baseline (TEST II))', 'Glaukopis Baseline (v20)',
    'Glaukopis (TEST)')
Needs openpyxl for the workbook part.

Outputs (printed and written to data/v18_additions.json):
  1. Table "single-stage vs chain configuration": rows, row-passes and characters per run  -> budget
  2. Table "32B single-stage reference": v7 single-stage, chain final, and both base evaluations -> ref32
  3. Table "14B design history": single-pass and phased 14B releases                          -> hist14
"""
import json, os, sys

R =os.environ.get("GLAUKOPIS_ROOT", "/Users/pietro/code/Glaukopis")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "v18_additions.json")
res = {}

# ---------------------------------------------------------------- 1. training budgets
def shard(name):
    rows = json.load(open(f"{R}/SFT/data/{name}.json", encoding="utf-8"))
    chars = sum(len(str(r.get("instruction", ""))) + len(str(r.get("input", ""))) + len(str(r.get("output", "")))
                for r in rows)
    return len(rows), chars

v7_rows, v7_chars = shard("ift_data_2026_04_26_combined_v7")
alp_rows, _ = shard("alpaca_en_demo")
v21 = {k: shard(f"ift_data_2026_05_18_v21_{k}") for k in
       ["core_a_kb_mcq_taa_soc_cm_ms_yn", "core_b_rms_ate_vsp_rcm", "taa", "cse"]}
v21_rows = sum(n for n, _ in v21.values()); v21_chars = sum(c for _, c in v21.values())
TULU_CAP = 240000          # --max-samples 240000 on tulu_3_sft_mixture in Stage 1A (run_sft_qwen25_32b_v21_core.sh)
STAGE4_ROWS = 6000         # 3,600 per shard at max probability 0.60 (run_sft_qwen25_32b_v21_recal_32b.sh)
res["budget"] = {
    "v7_single_stage": {"cti_rows": v7_rows, "epochs": 3, "general_rows": alp_rows,
                        "cti_row_passes": 3 * v7_rows, "total_row_passes": 3 * (v7_rows + alp_rows),
                        "cti_mchar_processed": round(3 * v7_chars / 1e6, 1)},
    "v21_chain": {"cti_rows_by_shard": {k: n for k, (n, _) in v21.items()}, "cti_rows": v21_rows,
                  "general_rows": TULU_CAP + alp_rows, "stage4_rows": STAGE4_ROWS,
                  "cti_row_passes": v21_rows + STAGE4_ROWS,
                  "total_row_passes": v21_rows + TULU_CAP + alp_rows + STAGE4_ROWS,
                  "cti_mchar_processed_stages1to3": round(v21_chars / 1e6, 1)},
    "matched_control_planned": {"cti_row_passes": v21_rows, "total_row_passes": v21_rows + TULU_CAP + alp_rows,
                                "cti_mchar_processed": round(v21_chars / 1e6, 1)},
}
print("Budget:", json.dumps(res["budget"], indent=1))

# ---------------------------------------------------------------- 2-3. workbook rows
try:
    import openpyxl
except ImportError:
    sys.exit("openpyxl is required for the workbook part")
wb = openpyxl.load_workbook(f"{R}/SFT/eval/Glaukopis Results.xlsx", data_only=True)

# Column layout shared by the three result sheets (0-based): 2 CKT, 3 ATE, 4 RCM, 7 RMS combined,
# 8 VSP, 11 TAA Classic combined, 17-19 CyberSOCEval (malware strict, malware Jaccard, TI strict), 24 CyberMetric mean.
def row(sheet, name):
    for r in wb[sheet].iter_rows(values_only=True):
        if r[0] and str(r[0]).strip() == name:
            return r
    return None

def scores(r):
    K, E, Rc, M, V, T = (round(float(r[i]), 1) for i in (2, 3, 4, 7, 8, 11))
    A = round((K + E + Rc + M + V + T) / 6, 1)
    C = round((float(r[17]) + float(r[18]) + float(r[19])) / 3, 1)   # project CSE composite (Section on aggregation)
    Y = round(float(r[24]), 1)
    return {"CKT": K, "ATE": E, "RCM": Rc, "RMS": M, "VSP": V, "TAA_Cl": T, "A_Cl": A, "C": C, "Y": Y,
            "B_Cl": round((A + C + Y) / 3, 1)}

S2, S20, ST = "Glaukopis Baseline (TEST II))", "Glaukopis Baseline (v20)", "Glaukopis (TEST)"
res["ref32"] = {
    "base_eval_1 (TEST II sheet)": scores(row(S2, "Qwen/Qwen2.5-32B-Instruct")),
    "base_eval_2 (TEST sheet)": scores(row(ST, "Qwen/Qwen2.5-32B-Instruct")),
    "v7_single_stage": scores(row(ST, "asg-ai/athena-cti-sft-qwen25-32b-abaligned-v7")),
}
b1, b2 = res["ref32"]["base_eval_1 (TEST II sheet)"], res["ref32"]["base_eval_2 (TEST sheet)"]
res["ref32"]["base_drift_max_task"] = round(max(abs(b1[k] - b2[k]) for k in ["CKT", "ATE", "RCM", "RMS", "VSP", "TAA_Cl"]), 1)
res["ref32"]["base_drift_aggregates"] = {k: round(abs(b1[k] - b2[k]), 1) for k in ["A_Cl", "C", "Y", "B_Cl"]}
print("32B reference:", json.dumps(res["ref32"], indent=1))

HIST14 = [  # label, shape, sheet, workbook model name
    ("Base", "--", S2, "Qwen/Qwen2.5-14B-Instruct"),
    ("v7", "single pass, 3 epochs, 4,096 packed", S2, "asg-ai/athena-cti-sft-qwen25-14b-abaligned-v7"),
    ("v10", "single pass, 1 epoch, 8,192 packed", S2, "asg-ai/athena-cti-sft-qwen25-14b-abaligned-v10"),
    ("v11", "single pass, 1 epoch, 8,192 packed", S2, "asg-ai/athena-cti-sft-qwen25-14b-v11"),
    ("v9", "two phases", S2, "asg-ai/athena-cti-sft-qwen25-14b-abaligned-v9"),
    ("v12", "three phases", S2, "asg-ai/athena-cti-sft-qwen25-14b-v12"),
    ("v21", "four stages (Recalibrate)", S20, "asg-ai/athena-cti-sft-qwen25-14b-v21-recalibrate"),
]
res["hist14"] = [{"release": lab, "shape": shp, **scores(row(sh, nm))} for lab, shp, sh, nm in HIST14]
print("14B history:", json.dumps(res["hist14"], indent=1))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", os.path.normpath(OUT))
