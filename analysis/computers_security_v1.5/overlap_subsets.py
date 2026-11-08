#!/usr/bin/env python3
"""Accuracy on evaluation items whose gold fact co-occurs in a training row vs. items whose fact does not."""
import json,re,glob,hashlib,collections,time
R="/Users/pietro/code/Glaukopis"; BD=f"{R}/SFT/eval/benchmark_data/athena_bench"; RESP=f"{R}/SFT/test/responses"
def jl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
rcm=jl(f"{BD}/athena-cti-rcm.jsonl"); rms=jl(f"{BD}/athena-cti-rms.jsonl"); vsp=jl(f"{BD}/athena-cti-vsp.jsonl")
taac=jl(f"{BD}/athena_taa_canonical/athena-cti-taa-canonical.jsonl"); ate=jl(f"{BD}/athena-cti-ate.jsonl")
CVE=re.compile(r"CVE-\d{4}-\d{4,}"); CWE=re.compile(r"CWE-\d+"); TID=re.compile(r"T\d{4}(?:\.\d{3})?"); MID=re.compile(r"M\d{4}")
rcm_pairs={(r["cve_id"],r["answer"].strip()):r["prompt"] for r in rcm}
rms_pairs={(r["technique_id"],tuple(sorted(set(MID.findall(r["answer"]))))):r["prompt"] for r in rms}
rms_tid_prompt={r["prompt"]:r["technique_id"] for r in rms}
rcm_cve_prompt={r["prompt"]:r["cve_id"] for r in rcm}; vsp_cve_prompt={r["prompt"]:r["cve_id"] for r in vsp}
taac_pairs={((r.get("alias_used") or r.get("alias") or "").strip(),r["answer"].strip()):r["prompt"] for r in taac}
ate_prompt_tid={r["prompt"]:r["technique_id"] for r in ate}
def word_in(tl,nl): return re.search(r"(?:^|[^a-z0-9])"+re.escape(nl)+r"(?:$|[^a-z0-9])",tl) is not None
hit={"rcm_pair":set(),"rcm_cve":set(),"vsp_cve":set(),"rms_pair":set(),"taac_pair":set(),"taac_alias":set(),"ate_tid_rows":collections.Counter()}
rcm_cves=set(rcm_cve_prompt.values()); vsp_cves=set(vsp_cve_prompt.values()); rms_tids=set(rms_tid_prompt.values())
alias_l={a.lower():(a,c) for (a,c) in taac_pairs if a}
for shard in ["core","taa","cse"]:
    rows=json.load(open(f"{R}/SFT/data/ift_data_2026_05_18_v21_{shard}.json",encoding="utf-8"))
    for r in rows:
        txt=" ".join(str(r.get(k,"")) for k in ("instruction","input","output")); tl=txt.lower()
        cves=set(CVE.findall(txt))
        if cves:
            for c in cves&rcm_cves:
                hit["rcm_cve"].add(c)
                for w in set(CWE.findall(txt)):
                    if (c,w) in rcm_pairs: hit["rcm_pair"].add((c,w))
            for c in cves&vsp_cves: hit["vsp_cve"].add(c)
        tids=set(TID.findall(txt))
        if tids:
            mids=set(MID.findall(txt))
            for tid in tids&rms_tids:
                for (t,ms) in rms_pairs:
                    if t==tid and set(ms)<=mids: hit["rms_pair"].add((t,ms))
            for t in tids: hit["ate_tid_rows"][t]+=1
        for al,(a,c) in alias_l.items():
            if len(al)>=3 and al in tl and word_in(tl,al):
                hit["taac_alias"].add(a)
                if word_in(tl,c.lower()): hit["taac_pair"].add((a,c))
    print(shard,"done", {k:(len(v) if not isinstance(v,collections.Counter) else len(v)) for k,v in hit.items()})
flag={}  # prompt -> (task, overlap flag)
for (c,w),p in rcm_pairs.items(): flag[("athena-rcm",p)]=("pair" if (c,w) in hit["rcm_pair"] else ("cve_only" if c in hit["rcm_cve"] else "none"))
for (t,ms),p in rms_pairs.items(): flag[("athena-rms",p)]=("pair" if (t,ms) in hit["rms_pair"] else "none")
for p,c in vsp_cve_prompt.items(): flag[("athena-vsp",p)]=("cve" if c in hit["vsp_cve"] else "none")
for (a,c),p in taac_pairs.items(): flag[("athena-taa-canonical",p)]=("pair" if (a,c) in hit["taac_pair"] else ("alias_only" if a in hit["taac_alias"] else "none"))
# ATE: frequency of the gold technique id in training rows (tertiles)
cnts=[hit["ate_tid_rows"][t] for t in ate_prompt_tid.values()]; s=sorted(cnts); q1=s[len(s)//3]; q2=s[2*len(s)//3]
for p,t in ate_prompt_tid.items():
    n=hit["ate_tid_rows"][t]; flag[("athena-ate",p)]=("low" if n<=q1 else "mid" if n<=q2 else "high")
print("ATE technique-id training-row-count tertile cut points:",q1,q2,"min/max",s[0],s[-1])
CK=[("Core","core"),("+TAA","taa"),("CSE","cse"),("Recal-14B","recalibrate"),("Recal-32B","recal-32b")]
out={}
for task in ["athena-rcm","athena-rms","athena-vsp","athena-taa-canonical","athena-ate"]:
    print(f"\n=== {task}: score by overlap subset (mean item score; RMS=combined F1, VSP=1-|d|/7.7, TAA=combined) ===")
    for lab,d in CK:
        p=glob.glob(f"{RESP}/athena-cti-sft-qwen25-32b-v21-{d}-vllm/{task}/*_scored.jsonl")[0]
        agg=collections.defaultdict(list); miss=0
        for r in jl(p):
            f=flag.get((task,r.get("prompt")))
            if f is None: miss+=1; continue
            sc=r["score"]
            if task=="athena-rms": v=sc["combined_f1"]
            elif task=="athena-taa-canonical": v=sc["combined"]
            elif task=="athena-vsp":
                if not r.get("success",True): continue
                v=1-float(sc)/7.7
            else: v=float(sc)
            agg[f].append(v)
        line=" | ".join(f"{k}: n={len(v)} mean={100*sum(v)/len(v):.1f}" for k,v in sorted(agg.items()))
        out[(task,lab)]={k:(len(v),100*sum(v)/len(v)) for k,v in agg.items()}
        print(f"  {lab}: {line} (unmatched prompts: {miss})")
json.dump({f"{t}|{l}":v for (t,l),v in out.items()},open("/private/tmp/claude-502/-Users-pietro-code-Glaukopis/c87ef551-ce64-4e86-aac5-38202593b78c/scratchpad/analysis/overlap_subsets.json","w"),indent=1)
