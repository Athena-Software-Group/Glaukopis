import json
import collections

r = json.load(open("/tmp/sft-0424-smoke/triples/_results-report.json"))
results = r["results"]

print(f"Total templates: {len(results)}")
print(f"Generated: {r['all_generated_count']}, failed: {r['failed_count']}")
print()

# Bucketize by reason
buckets = collections.defaultdict(list)
for entry in results:
    sn = entry["template_object"].get("shortname", "?")
    excs = entry.get("exception") or []
    n = entry.get("generated_count", 0)
    if n > 0 and not excs:
        buckets["OK"].append((sn, n, ""))
        continue
    msg = excs if isinstance(excs, str) else str(excs)
    if "cpe_matches" in msg:
        cause = "missing prop CVE.cpe_matches"
    elif "SigmaRule" in msg:
        cause = "missing label SigmaRule"
    elif "ExploitDBEntry" in msg:
        cause = "missing label ExploitDBEntry"
    elif "GithubPoC" in msg:
        cause = "missing label GithubPoC"
    elif "has_weaponized_exploit" in msg:
        cause = "missing rel has_weaponized_exploit"
    elif "has_poc" in msg:
        cause = "missing rel has_poc"
    else:
        cause = "other"
    buckets[cause].append((sn, n, msg.strip().splitlines()[-1][:140]))

for cause, items in sorted(buckets.items(), key=lambda x: (-len(x[1]), x[0])):
    print(f"--- [{len(items):>2}] {cause}")
    for sn, n, msg in items:
        print(f"     {sn:18s} count={n}  {msg if cause.startswith('other') else ''}")
