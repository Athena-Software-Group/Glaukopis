#!/usr/bin/env python3
"""Diagnose athena-taa parsing/resolution issues against ground truth.

Mirrors the production pipeline (athena_taa_answer -> _resolve_actor_in_text
-> threat_actor_connection) and distinguishes three failure modes:

  1. parser_miss  : correct actor name/alias appears somewhere in the
                    response but the last-line-only parser dropped it
                    (model wrote the answer earlier in its reasoning).
  2. resolver_miss: last line contains the correct alias but
                    _resolve_actor_in_text picked a different (shorter or
                    earlier) key that resolves to the wrong cluster.
  3. model_wrong  : the response genuinely names a different actor
                    (unrecoverable by parser changes).

Reports correct / plausible / incorrect / parse_failed, and the recoverable
upper bound if the parser scanned the full response instead of just the
last line. Stdlib-only; no conda env activation required.

Usage (run on the inference host where responses live):

  cd ~/Glaukopis/SFT/test
  python utils/diagnose_taa.py \\
      --response-file responses/<model_dir>/athena-taa/<file>_response.jsonl \\
      --alias-csv  benchmark_data/athena_bench/athena_taa/aliases.csv \\
      --related-csv benchmark_data/athena_bench/athena_taa/related_groups.csv \\
      --out-mismatches /tmp/taa_mismatches.jsonl
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list:
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: bad JSON at {path}:{i}: {e}", file=sys.stderr)
    return out


def load_alias_dict(path: Path) -> dict:
    alias: dict = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = row["ThreatActor"].strip().lower()
            v = row["Alias"].strip().lower()
            alias.setdefault(k, []).append(v)
            alias.setdefault(v, []).append(k)
    return alias


def load_related_dict(path: Path) -> dict:
    rel: dict = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = row["ThreatActor"].strip().lower()
            v = row["RelatedGroup"].strip().lower()
            rel.setdefault(k, []).append(v)
            rel.setdefault(v, []).append(k)
    return rel


def current_parse_taa(text: str) -> str:
    """Mirror athena_cti_postprocessing.athena_taa_answer: last non-empty line."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    raw = lines[-1].strip().strip('"\'')
    return re.sub(r"\s+", " ", raw)


def resolve_in_text(text: str, alias_dict: dict) -> str:
    """Mirror _resolve_actor_in_text: longest alias-dict key as whole word."""
    t = (text or "").strip().lower()
    if not t or t in alias_dict:
        return t
    candidates = [
        k for k in alias_dict
        if k and len(k) >= 3
        and re.search(r"(?:^|[^a-z0-9])" + re.escape(k) + r"(?:$|[^a-z0-9])", t)
    ]
    if not candidates:
        return t
    candidates.sort(key=lambda k: (-len(k), t.find(k)))
    return candidates[0]


def fallback_scan(full_text: str, alias_dict: dict) -> list:
    """Scan the ENTIRE response for alias-dict keys; return [(key, pos)]."""
    t = (full_text or "").lower()
    hits = []
    seen = set()
    for k in alias_dict:
        if not k or len(k) < 3 or k in seen:
            continue
        m = re.search(r"(?:^|[^a-z0-9])" + re.escape(k) + r"(?:$|[^a-z0-9])", t)
        if m:
            hits.append((k, m.start()))
            seen.add(k)
    hits.sort(key=lambda kv: (-len(kv[0]), kv[1]))
    return hits


def bfs_connected(src: str, dst: str, graph: dict) -> bool:
    if not src or not dst:
        return False
    visited, queue = set(), [src]
    while queue:
        cur = queue.pop(0)
        if cur == dst:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        for nxt in graph.get(cur, []):
            if nxt not in visited:
                queue.append(nxt)
    return False


def classify(gt: str, resolved: str, alias_dict: dict, related_dict: dict) -> str:
    """Return 'C', 'P', 'I' per production scorer semantics."""
    gt_k = (gt or "").strip().lower()
    r_k = (resolved or "").strip().lower()
    if not r_k:
        return "I"
    if bfs_connected(gt_k, r_k, alias_dict):
        return "C"
    combined = {k: (alias_dict.get(k, []) + related_dict.get(k, [])) for k in set(alias_dict) | set(related_dict)}
    if bfs_connected(gt_k, r_k, combined):
        return "P"
    return "I"



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response-file", required=True, type=Path,
                    help="path to *_response.jsonl from SFT/test/inference.py")
    ap.add_argument("--alias-csv", type=Path,
                    default=Path("benchmark_data/athena_bench/athena_taa/aliases.csv"))
    ap.add_argument("--related-csv", type=Path,
                    default=Path("benchmark_data/athena_bench/athena_taa/related_groups.csv"))
    ap.add_argument("--out-mismatches", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=15,
                    help="how many non-correct examples to print (default: 15)")
    args = ap.parse_args()

    if not args.response_file.is_file():
        sys.exit(f"response file not found: {args.response_file}")
    alias_dict = load_alias_dict(args.alias_csv)
    related_dict = load_related_dict(args.related_csv)

    responses = load_jsonl(args.response_file)
    rows_out = []
    counters = Counter()
    recovery_by = Counter()

    for r in responses:
        rid = r.get("id")
        resp = r.get("response") or ""
        gt = (r.get("answer") or "").strip()

        last_line = current_parse_taa(resp)
        resolved_last = resolve_in_text(last_line, alias_dict)
        status_letter = classify(gt, resolved_last, alias_dict, related_dict)

        if not gt:
            status = "gt_missing"
        elif not last_line.strip():
            status = "parse_failed"
        elif status_letter == "C":
            status = "correct"
        elif status_letter == "P":
            status = "plausible"
        else:
            status = "incorrect"

        # Heuristic: full-response scan. Does any longer alias-dict key
        # appear elsewhere in the response and resolve to GT's cluster?
        recovery = None
        if status in ("incorrect", "plausible", "parse_failed") and gt:
            for cand, _pos in fallback_scan(resp, alias_dict):
                cand_status = classify(gt, cand, alias_dict, related_dict)
                if cand_status == "C" and status != "correct":
                    recovery = ("full_scan_correct", cand)
                    break
                if cand_status == "P" and status == "incorrect":
                    recovery = ("full_scan_plausible", cand)
                    # keep searching for a 'C' win

        counters[status] += 1
        if recovery:
            recovery_by[recovery[0]] += 1

        rows_out.append({
            "id": rid,
            "gt": gt,
            "last_line": last_line,
            "resolved": resolved_last,
            "status": status,
            "recovery": recovery,
            "response_tail": (resp or "").strip()[-300:],
        })

    total = len(rows_out) or 1
    correct = counters["correct"]
    plausible = counters["plausible"]
    incorrect = counters["incorrect"]
    parse_failed = counters["parse_failed"]
    gt_missing = counters["gt_missing"]
    scorable = correct + plausible + incorrect + parse_failed
    recoverable = sum(1 for r in rows_out if r.get("recovery"))
    c_gains = sum(1 for r in rows_out if r.get("recovery") and r["recovery"][0] == "full_scan_correct")

    print(f"=== TAA diagnostic: {args.response_file.name} ===")
    print(f"  total rows             : {total}")
    print(f"  correct (alias cluster): {correct:>5}  ({100*correct/total:6.2f} %)")
    print(f"  plausible (related)    : {plausible:>5}  ({100*plausible/total:6.2f} %)")
    print(f"  incorrect              : {incorrect:>5}  ({100*incorrect/total:6.2f} %)")
    print(f"  parse_failed           : {parse_failed:>5}  ({100*parse_failed/total:6.2f} %)")
    print(f"  gt_missing             : {gt_missing:>5}  ({100*gt_missing/total:6.2f} %)")
    print()
    acc = 100 * correct / scorable if scorable else 0
    upper_c = 100 * (correct + c_gains) / scorable if scorable else 0
    print(f"  accuracy (scorable)    : {acc:6.2f} %  [{correct}/{scorable}]")
    print(f"  recoverable (full-scan): {recoverable} (of which {c_gains} -> C)")
    print(f"  upper-bound accuracy   : {upper_c:6.2f} %  (if parser scanned full response)")
    if recovery_by:
        print(f"  recovery breakdown     : {dict(recovery_by)}")
    print()

    if args.sample:
        print(f"=== Up to {args.sample} non-correct examples ===")
        shown = 0
        for r in rows_out:
            if r["status"] == "correct":
                continue
            tail = r["response_tail"].replace("\n", " \u21b5 ")
            rec = r["recovery"] or "-"
            print(f"  id={r['id']:>4}  gt={r['gt']!r}  resolved={r['resolved']!r}  "
                  f"status={r['status']}  recovery={rec}")
            print(f"    last_line: {r['last_line'][:160]!r}")
            print(f"    tail: ...{tail}")
            shown += 1
            if shown >= args.sample:
                break

    if args.out_mismatches:
        args.out_mismatches.parent.mkdir(parents=True, exist_ok=True)
        with args.out_mismatches.open("w", encoding="utf-8") as f:
            for r in rows_out:
                if r["status"] != "correct":
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        noncorrect = sum(1 for r in rows_out if r["status"] != "correct")
        print(f"\n[out] {args.out_mismatches}  ({noncorrect} non-correct rows)")


if __name__ == "__main__":
    main()
