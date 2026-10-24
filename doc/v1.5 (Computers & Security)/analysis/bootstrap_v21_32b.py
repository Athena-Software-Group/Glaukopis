#!/usr/bin/env python3
"""Paired item-level bootstrap over the five local Qwen2.5-32B v21 checkpoints.
Reads scored artifacts under SFT/test/responses. No inference. Writes JSON + prints summary."""
import json, csv, glob, os, sys, random, math
from collections import OrderedDict
ROOT="/Users/pietro/code/Glaukopis/SFT/test/responses"
CK=OrderedDict([("Core","athena-cti-sft-qwen25-32b-v21-core-vllm"),("+TAA","athena-cti-sft-qwen25-32b-v21-taa-vllm"),
    ("CSE","athena-cti-sft-qwen25-32b-v21-cse-vllm"),("Recal-14B","athena-cti-sft-qwen25-32b-v21-recalibrate-vllm"),
    ("Recal-32B","athena-cti-sft-qwen25-32b-v21-recal-32b-vllm")])
VSP_DENOM=7.7
def load_jsonl(p):
    with open(p,encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def load_task(ck, task):
    d=CK[ck]
    if task in ("cm2000","cm10000"):
        stem="CyberMetric-2000-v1" if task=="cm2000" else "CyberMetric-10000-v1"
        p=glob.glob(f"{ROOT}/{d}/cybermetric/cybermetric_{stem}_all_v1_*_response.csv")[0]
        items={}; seen={}
        with open(p,encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                q=(r["question"] or "").strip(); c=seen.get(q,0); seen[q]=c+1
                items[(q,c)]=dict(score=1.0 if (r["answer"] or "").strip()==(r["answer_after_post_processing"] or "").strip() else 0.0, success=True)
        return items
    if task=="mmlu":
        p=glob.glob(f"{ROOT}/{d}/mmlu-pro/*.csv")[0]
        items={}
        with open(p,encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                items[r["question_id"]]=dict(score=1.0 if (r["ground_truth"] or "").strip()==(r["prediction"] or "").strip() else 0.0, success=True)
        return items
    p=glob.glob(f"{ROOT}/{d}/{task}/*_scored.jsonl")[0]
    items={}
    for r in load_jsonl(p):
        key=r.get("id")
        s=r["score"]; suc=bool(r.get("success",True))
        if task=="athena-taa" or task=="athena-taa-canonical":
            val=dict(score=float(s["combined"]),correct=s["correct"],plausible=s["plausible"],success=suc,pred=r.get("prediction",""),resp=r.get("response",""),answer=r.get("answer",""),mitre_id=r.get("mitre_id",""))
        elif task=="athena-rms":
            val=dict(score=float(s["combined_f1"]),f1=float(s["f1"]),success=suc)
        elif task=="athena-vsp":
            val=dict(score=float(s) if suc else None,success=suc)
        elif task.startswith("cybersoceval"):
            val=dict(score=float(s),success=suc)
        else:
            val=dict(score=float(s),success=suc)
        items[key]=val
    return items
def metric(task, vals):
    """vals: list of item dicts (already resampled). Returns dict of metrics (percent)."""
    n=len(vals)
    if task=="athena-vsp":
        ok=[v["score"] for v in vals if v["success"]]
        mad=sum(ok)/len(ok) if ok else 0.0
        return {"acc":100*(1-mad/VSP_DENOM),"n_fail":n-len(ok)}
    if task.startswith("cybersoceval"):
        ok=[v for v in vals if v["success"]]
        strict=100*sum(1 for v in ok if v["score"]==1.0)/len(ok) if ok else 0.0
        jac=100*sum(v["score"] for v in ok)/len(ok) if ok else 0.0
        strict_pen=100*sum(1 for v in ok if v["score"]==1.0)/n
        jac_pen=100*sum(v["score"] for v in ok)/n
        return {"acc":strict,"jaccard":jac,"acc_pen":strict_pen,"jaccard_pen":jac_pen,"parse_err_pct":100*(n-len(ok))/n}
    if task in ("athena-taa","athena-taa-canonical"):
        return {"acc":100*sum(v["score"] for v in vals)/n,"strict":100*sum(v["correct"] for v in vals)/n,"plausible":100*sum(v["plausible"] for v in vals)/n}
    if task=="athena-rms":
        return {"acc":100*sum(v["score"] for v in vals)/n,"strict_f1":100*sum(v["f1"] for v in vals)/n}
    return {"acc":100*sum(v["score"] for v in vals)/n}
TASKS=["athena-mcq","athena-ate","athena-rcm","athena-rms","athena-vsp","athena-taa","athena-taa-canonical","cybersoceval-malware","cybersoceval-ti","cm2000","cm10000","mmlu"]
data={ck:{t:load_task(ck,t) for t in TASKS} for ck in CK}
# common keys per task
keys={}
for t in TASKS:
    ks=set(data["Core"][t].keys())
    for ck in CK: ks&=set(data[ck][t].keys())
    keys[t]=sorted(ks, key=lambda k: (str(type(k)),k))
    print(f"{t}: n_common={len(keys[t])} per-ck n={[len(data[ck][t]) for ck in CK]}")
def aggregates(m):
    """m: dict task->metrics for one checkpoint. Returns A_bal,B_bal, C(cse composite), Y(cm)"""
    T=(m["athena-taa"]["acc"]+m["athena-taa-canonical"]["acc"])/2
    A=(m["athena-mcq"]["acc"]+m["athena-ate"]["acc"]+m["athena-rcm"]["acc"]+m["athena-rms"]["acc"]+m["athena-vsp"]["acc"]+T)/6
    C=(m["cybersoceval-malware"]["acc"]+m["cybersoceval-malware"]["jaccard"]+m["cybersoceval-ti"]["acc"])/3
    C4=(m["cybersoceval-malware"]["acc"]+m["cybersoceval-malware"]["jaccard"]+m["cybersoceval-ti"]["acc"]+m["cybersoceval-ti"]["jaccard"])/4
    Y=(m["cm2000"]["acc"]+m["cm10000"]["acc"])/2
    B=(A+C+Y)/3
    Cpen=(m["cybersoceval-malware"]["acc_pen"]+m["cybersoceval-malware"]["jaccard_pen"]+m["cybersoceval-ti"]["acc_pen"])/3
    return {"T_bal":T,"A_bal":A,"C":C,"C4":C4,"Cpen":Cpen,"Y":Y,"B_bal":B,"B_bal_C4":(A+C4+Y)/3,"B_bal_Cpen":(A+Cpen+Y)/3}
point={ck:{t:metric(t,[data[ck][t][k] for k in keys[t]]) for t in TASKS} for ck in CK}
for ck in CK: point[ck]["agg"]=aggregates(point[ck])
random.seed(20260906)
R=int(sys.argv[1]) if len(sys.argv)>1 else 2000
boot={ck:{t:[] for t in TASKS+["agg"]} for ck in CK}
for r in range(R):
    idx={t:[random.randrange(len(keys[t])) for _ in range(len(keys[t]))] for t in TASKS}
    for ck in CK:
        m={}
        for t in TASKS:
            ks=keys[t]; items=data[ck][t]
            m[t]=metric(t,[items[ks[i]] for i in idx[t]])
            boot[ck][t].append(m[t])
        boot[ck]["agg"].append(aggregates(m))
def ci(vals,lo=2.5,hi=97.5):
    s=sorted(vals); n=len(s)
    return (s[int(math.floor(lo/100*(n-1)))], s[int(math.ceil(hi/100*(n-1)))])
out={"R":R,"n":{t:len(keys[t]) for t in TASKS},"point":point,"ci":{},"paired":{}}
print("\n=== Point estimates and 95% bootstrap CIs (item resampling; single training run) ===")
for ck in CK:
    out["ci"][ck]={}
    line=[]
    for t in TASKS:
        v=[b["acc"] for b in boot[ck][t]]; c=ci(v); out["ci"][ck][t]={"acc":point[ck][t]["acc"],"lo":c[0],"hi":c[1]}
        line.append(f"{t}={point[ck][t]['acc']:.1f}[{c[0]:.1f},{c[1]:.1f}]")
    for a in ("A_bal","B_bal","C","Y","T_bal","B_bal_C4","Cpen","B_bal_Cpen"):
        v=[b[a] for b in boot[ck]["agg"]]; c=ci(v); out["ci"][ck][a]={"acc":point[ck]["agg"][a],"lo":c[0],"hi":c[1]}
        line.append(f"{a}={point[ck]['agg'][a]:.2f}[{c[0]:.2f},{c[1]:.2f}]")
    print(ck+": "+" | ".join(line))
    extra={t:{k:v for k,v in point[ck][t].items() if k!="acc"} for t in TASKS if len(point[ck][t])>1}
    print("   extras:",json.dumps({k:{kk:round(vv,2) for kk,vv in v.items()} for k,v in extra.items()}))
print("\n=== Paired differences (later minus earlier), 95% CI ===")
pairs=[("Core","+TAA"),("+TAA","CSE"),("CSE","Recal-32B"),("CSE","Recal-14B"),("Recal-14B","Recal-32B"),("Core","Recal-32B"),("+TAA","Recal-32B")]
for a,b in pairs:
    out["paired"][f"{b}-{a}"]={}
    line=[]
    for t in TASKS+["agg"]:
        if t=="agg":
            for m in ("A_bal","B_bal","T_bal","C","Y"):
                d=[y[m]-x[m] for x,y in zip(boot[a]["agg"],boot[b]["agg"])]; c=ci(d); pt=point[b]["agg"][m]-point[a]["agg"][m]
                out["paired"][f"{b}-{a}"][m]={"d":pt,"lo":c[0],"hi":c[1]}
                line.append(f"{m}={pt:+.2f}[{c[0]:+.2f},{c[1]:+.2f}]")
        else:
            d=[y["acc"]-x["acc"] for x,y in zip(boot[a][t],boot[b][t])]; c=ci(d); pt=point[b][t]["acc"]-point[a][t]["acc"]
            out["paired"][f"{b}-{a}"][t]={"d":pt,"lo":c[0],"hi":c[1]}
            line.append(f"{t}={pt:+.1f}[{c[0]:+.1f},{c[1]:+.1f}]")
    print(f"{b} - {a}: "+" | ".join(line))
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"bootstrap_results.json"),"w"),indent=1)
print("\nwrote bootstrap_results.json")
