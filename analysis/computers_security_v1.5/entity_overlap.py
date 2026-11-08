#!/usr/bin/env python3
"""Entity/fact overlap between v21 training shards and AthenaBench evaluation items (read-only)."""
import json,re,collections,sys,time
R="/Users/pietro/code/Glaukopis"
BD=f"{R}/SFT/eval/benchmark_data/athena_bench"
def jl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
ate=jl(f"{BD}/athena-cti-ate.jsonl"); rms=jl(f"{BD}/athena-cti-rms.jsonl"); rcm=jl(f"{BD}/athena-cti-rcm.jsonl"); vsp=jl(f"{BD}/athena-cti-vsp.jsonl")
taac=jl(f"{BD}/athena_taa_canonical/athena-cti-taa-canonical.jsonl"); taa=jl(f"{BD}/athena_taa/athena-cti-taa.jsonl")
ev={}
ev["rcm_cves"]={r["cve_id"] for r in rcm}; ev["rcm_pairs"]={(r["cve_id"],r["answer"].strip()) for r in rcm}; ev["rcm_cwes"]={r["answer"].strip() for r in rcm}
ev["vsp_cves"]={r["cve_id"] for r in vsp}; ev["vsp_pairs"]={(r["cve_id"],r["answer"].strip()) for r in vsp}
ev["ate_tids"]={r["technique_id"] for r in ate}; ev["rms_tids"]={r["technique_id"] for r in rms}
ev["rms_pairs"]={(r["technique_id"],tuple(sorted(re.findall(r"M\d{4}",r["answer"])))) for r in rms}
ev["taac_aliases"]={(r.get("alias_used") or r.get("alias") or "").strip() for r in taac}; ev["taac_pairs"]={((r.get("alias_used") or r.get("alias") or "").strip(),r["answer"].strip()) for r in taac}
ev["taac_answers"]={r["answer"].strip() for r in taac}
ev["taa_answers"]={r["answer"].strip() for r in taa}
print("eval sets:",{k:len(v) for k,v in ev.items()}); print("taac sample:",list(ev["taac_pairs"])[:3])
CVE=re.compile(r"CVE-\d{4}-\d{4,}"); CWE=re.compile(r"CWE-\d+"); TID=re.compile(r"T\d{4}(?:\.\d{3})?"); MID=re.compile(r"M\d{4}"); VEC=re.compile(r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]")
alias_l={a.lower():a for a in ev["taac_aliases"] if a}; canon_l={c.lower():c for c in ev["taac_answers"]}
taa_l={c.lower():c for c in ev["taa_answers"]}
def word_in(text_l, name_l): return re.search(r"(?:^|[^a-z0-9])"+re.escape(name_l)+r"(?:$|[^a-z0-9])",text_l) is not None
res={}
for shard in ["core","taa","cse"]:
    t0=time.time()
    rows=json.load(open(f"{R}/SFT/data/ift_data_2026_05_18_v21_{shard}.json",encoding="utf-8"))
    seen_cve=set(); seen_cwe=set(); seen_tid=set(); seen_mid=set()
    rcm_pair_hits=set(); vsp_pair_hits=set(); rms_pair_hits=set(); taac_alias_hits=set(); taac_pair_hits=set(); taa_ans_hits=set(); canon_hits=set()
    rows_with_eval_cve=0
    for r in rows:
        txt=" ".join(str(r.get(k,"")) for k in ("instruction","input","output"))
        cves=set(CVE.findall(txt)); 
        if cves:
            seen_cve|=cves
            ec=cves&(ev["rcm_cves"]|ev["vsp_cves"])
            if ec:
                rows_with_eval_cve+=1
                cwes=set(CWE.findall(txt)); vecs=set(VEC.findall(txt))
                for c in ec:
                    for w in cwes:
                        if (c,w) in ev["rcm_pairs"]: rcm_pair_hits.add((c,w))
                    for v in vecs:
                        if (c,v) in ev["vsp_pairs"]: vsp_pair_hits.add((c,v))
        tids=set(TID.findall(txt))
        if tids:
            seen_tid|=tids
            mids=set(MID.findall(txt))
            for tid in tids&ev["rms_tids"]:
                for (t,ms) in ev["rms_pairs"]:
                    if t==tid and set(ms)<=mids: rms_pair_hits.add((t,ms))
            seen_mid|=mids
        seen_cwe|=set(CWE.findall(txt))
        tl=txt.lower()
        for al,a in alias_l.items():
            if len(al)>=3 and al in tl and word_in(tl,al):
                taac_alias_hits.add(a)
                for (aa,cn) in ev["taac_pairs"]:
                    if aa==a and word_in(tl,cn.lower()): taac_pair_hits.add((aa,cn))
        for cl,c in canon_l.items():
            if len(cl)>=3 and cl in tl and word_in(tl,cl): canon_hits.add(c)
        for cl,c in taa_l.items():
            if len(cl)>=3 and cl in tl and word_in(tl,cl): taa_ans_hits.add(c)
    d={
     "rows":len(rows),
     "rcm: eval CVE ids appearing in shard":f"{len(ev['rcm_cves']&seen_cve)}/{len(ev['rcm_cves'])}",
     "rcm: eval (CVE,CWE) gold pairs co-occurring in one row":f"{len(rcm_pair_hits)}/{len(ev['rcm_pairs'])}",
     "rcm: eval gold CWE ids appearing anywhere":f"{len(ev['rcm_cwes']&seen_cwe)}/{len(ev['rcm_cwes'])}",
     "vsp: eval CVE ids appearing":f"{len(ev['vsp_cves']&seen_cve)}/{len(ev['vsp_cves'])}",
     "vsp: eval (CVE,vector) gold pairs co-occurring":f"{len(vsp_pair_hits)}/{len(ev['vsp_pairs'])}",
     "ate: eval technique ids appearing":f"{len(ev['ate_tids']&seen_tid)}/{len(ev['ate_tids'])}",
     "rms: eval technique ids appearing":f"{len(ev['rms_tids']&seen_tid)}/{len(ev['rms_tids'])}",
     "rms: eval (technique, full gold mitigation set) co-occurring in one row":f"{len(rms_pair_hits)}/{len(ev['rms_pairs'])}",
     "taa-canonical: eval alias strings appearing":f"{len(taac_alias_hits)}/{len(ev['taac_aliases'])}",
     "taa-canonical: (alias, canonical) pairs co-occurring in one row":f"{len(taac_pair_hits)}/{len(ev['taac_pairs'])}",
     "taa-canonical: gold canonical names appearing":f"{len(canon_hits)}/{len(ev['taac_answers'])}",
     "taa-classic: gold actor names appearing":f"{len(taa_ans_hits)}/{len(ev['taa_answers'])}",
     "rows mentioning any eval CVE id":rows_with_eval_cve,
     "distinct CVE ids in shard":len(seen_cve),"distinct technique ids in shard":len(seen_tid),
     "secs":round(time.time()-t0,1)}
    res[shard]=d
    print(f"\n=== {shard} ===")
    for k,v in d.items(): print(f"  {k}: {v}")
json.dump(res,open("/private/tmp/claude-502/-Users-pietro-code-Glaukopis/c87ef551-ce64-4e86-aac5-38202593b78c/scratchpad/analysis/entity_overlap.json","w"),indent=1)
