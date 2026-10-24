#!/usr/bin/env python3
"""Recompute v1.4 tables from published-draft numbers; NOT a new evaluation.
Input stage scores are transcribed from the attached v1.3 Table V.
Cost components are transcribed from the repository cost_summary.csv at
c4d3a2f45e0a08672c87c2b46c3c4be109440c38. They are expenditure-ledger
entries, not independently validated matched single-pass costs.
Only the package's data/ directory is written. No model/project is accessed.
"""
from pathlib import Path
import csv, json
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
rows = [
 ('Base',52.9,18.2,59.5,7.6,78.8,48.,2.,42.6,90.2,69.),
 ('Core',72.9,67.2,69.8,63.8,83.1,46.5,18.4,31.1,89.6,59.2),
 ('+TAA',75.4,63.4,69.,62.9,83.5,47.,36.8,29.8,89.4,58.2),
 ('CSE',72.5,63.4,67.7,63.7,78.9,53.5,3.4,42.,88.8,54.2),
 ('Recal-32B',75.5,65.8,67.1,62.5,77.8,50.5,3.4,42.6,89.8,57.4),
]
fields=['checkpoint','CKT','ATE','RCM','RMS','VSP','TAA_classic','TAA_canonical','CSE_reported','CyberMetric_reported','MMLU_Pro']
stages=[]
for row in rows:
 d=dict(zip(fields,row)); vals=row[1:]
 d['AB_classic_recomputed']=(sum(vals[:5])+vals[5])/6
 d['AB_blended_recomputed']=(sum(vals[:5])+(vals[5]+vals[6])/2)/6
 d['CTI_benchmark_blended_recomputed']=(d['AB_blended_recomputed']+d['CSE_reported']+d['CyberMetric_reported'])/3
 stages.append(d)
with (DATA/'stage_scores_and_reaggregation.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
costs=[
 ('Gemini-3-Flash-Preview',5.7202,78.4613,64.4625,1005.4883,1154.1323),
 ('GPT-5.5',202.6194,79.1854,38.7299,187.5974,508.1321),
 ('GPT-5.2',30.4439,23.2834,4.0444,151.8979,209.6696),
 ('Gemini-2.5-Flash',56.6752,12.2585,19.0787,78.2252,166.2376),
 ('DeepSeek-V4-Pro',0.3222,27.9262,9.8686,88.2809,126.3979),
 ('DeepSeek-V3.1-Terminus',2.4458,3.7016,0.4794,7.7003,14.3272),
 ('DeepSeek-V3.2-Exp',0.0062,3.6839,0.4501,7.0992,11.2394),
 ('Qwen2.5-32B base',2.5361,0.6139,0.2306,7.2861,10.6667),
 ('v21-core',1.0667,0.6500,0.2333,0.9333,2.8833),
 ('v21-taa',0.9833,0.6528,0.2278,0.8639,2.7278),
 ('v21-cse',0.7333,0.5861,0.2278,0.6278,2.1750),
 ('v21-recalibrate (14B recipe)',1.0361,0.5889,0.2361,0.6472,2.5083),
 ('v21-recal-32b',1.2194,0.5944,0.2333,0.7861,2.8333),
]
ship_cti=sum(costs[-1][1:4]); ship_total=costs[-1][-1]
cout=[]
for m,ab,cse,cm,mmlu,total in costs:
 cti=ab+cse+cm
 cout.append(dict(model=m,AthenaBench_usd=ab,CyberSOCEval_usd=cse,CyberMetric_usd=cm,MMLU_Pro_usd=mmlu,recorded_total_usd=total,derived_CTI_only_usd=cti,CTI_ledger_ratio=cti/ship_cti,full_ledger_ratio=total/ship_total,MMLU_share_percent=100*mmlu/total))
with (DATA/'cost_ledger_reaggregation.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cout[0]));w.writeheader();w.writerows(cout)
legacy=[
 ('GPT-5.5',72.9,73.5,76.9,77.,48.3,93.3,88.3),
 ('DeepSeek-V4-Pro',70.4,67.9,69.6,67.2,52.2,91.8,79.1),
 ('Gemini-3-Flash-Preview',69.8,68.4,71.7,69.3,48.1,92.1,89.2),
 ('GPT-5.2',67.5,65.,67.,63.8,46.8,91.8,81.3),
 ('DeepSeek-V3.2-Exp',66.,62.2,63.9,59.,47.2,91.7,82.5),
 ('v21-recal-32b',66.3,62.9,62.8,66.5,42.6,89.8,57.4),
 ('v21-cse',65.8,62.4,62.2,66.6,42.,88.8,54.2),
 ('v21-recalibrate',65.9,62.4,62.2,65.9,42.4,89.2,54.6),
 ('Gemini-2.5-Flash',63.4,57.9,59.8,53.1,46.2,91.,82.1),
 ('v21-core',62.6,61.6,63.2,67.2,31.1,89.6,59.2),
 ('v21-taa',62.,61.7,63.7,66.9,29.8,89.4,58.2),
 ('Qwen2.5-32B base',59.,49.6,49.3,44.2,42.6,90.2,69.),
]
lf=['model','legacy_CTI_bench','legacy_CTI_test_UNRECONCILED','legacy_total_UNRECONCILED','legacy_AB_classic_consistent','CSE','CM','MMLU']
with (DATA/'legacy_headline_table_UNRECONCILED.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(lf);w.writerows(legacy)
audit={
 'notice':'All calculations are post hoc arithmetic from existing rounded summaries, not new model runs. Cost ratios are ledger ratios, not proven per-pass comparisons.',
 'derived_ship_CTI_mean':stages[-1]['CTI_benchmark_blended_recomputed'],
 'derived_base_CTI_mean':stages[0]['CTI_benchmark_blended_recomputed'],
 'derived_CTI_gain':stages[-1]['CTI_benchmark_blended_recomputed']-stages[0]['CTI_benchmark_blended_recomputed'],
 'gain_attribution':{'AB_component':(stages[-1]['AB_blended_recomputed']-stages[0]['AB_blended_recomputed'])/3,'CSE_component':0.0,'CM_component':(89.8-90.2)/3},
 'legacy_total_formula_check':{'base_claimed':49.3,'base_implied_by_legacy_10_test_mean':(10*49.6+69)/11,'ship_claimed':62.8,'ship_implied_by_legacy_10_test_mean':(10*62.9+57.4)/11},
 'distinct_13gram_drop50_shortrow_bound':'A compared evaluation record with fewer than 62 word tokens has at most 49 distinct 13-grams. A complete duplicate can therefore fail the drop=50 threshold (assuming unique record keys).',
}
(DATA/'audit_calculations.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
for d in cout[:8]: print(d['model'],round(d['derived_CTI_only_usd'],4),round(d['CTI_ledger_ratio'],2))
