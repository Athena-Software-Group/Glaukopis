#!/usr/bin/env python3
"""Diagnose athena-mcq parsing issues against ground truth.

Reads a *_response.jsonl written by athena_bench/inference.py and
re-evaluates every row against three questions:

  1. Did GT normalization succeed (row has a non-empty A-E letter)?
  2. Did the current parser extract a letter at all (not 'X')?
  3. If the prediction is wrong, does the correct letter appear in
     the raw response text under any of several fallback heuristics
     (trailing "Therefore, X.", "answer is X", bare-letter line,
      parenthesised "(X)", verbatim option-text match)?

Output distinguishes "model-wrong" (the response confidently picks
a different letter) from "parser-miss" (the response contains the
right answer but the regex dropped it). Stdlib-only; no conda env
activation required.

Usage (run on the inference host where responses live):

  cd ~/Glaukopis/athena_bench
  python utils/diagnose_mcq.py \\
      --response-file responses/pworth1971_athena-cti-sft-llama31-8b-abaligned/athena-mcq/athena-mcq_all_v1_pworth1971_athena-cti-sft-llama31-8b-abaligned_response.jsonl \\
      --benchmark-file benchmark_data/athena_bench/athena-cti-mcq-3k.jsonl \\
      --out-mismatches /tmp/mcq_mismatches.jsonl
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

_LETTERS = ("A", "B", "C", "D", "E")
_LETTER_SET = set(_LETTERS)
_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|prediction|output|result)\s*[:\-–—]?\s*",
    re.IGNORECASE,
)


def current_parse_mcq(text: str) -> str:
    """Mirror athena_cti_postprocessing.extract_answer('athena-mcq').

    Takes the LAST \\b[A-E]\\b match on each line (bottom-up over lines).
    """
    if not text:
        return ""

    def last(line: str):
        hits = re.findall(r"\b([A-E])\b", line, re.IGNORECASE)
        return hits[-1].upper() if hits else None

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        raw = lines[i]
        line = _PREFIX_RE.sub("", raw).strip()
        m = last(line)
        if m:
            return m
        if re.search(r"\banswer\b", raw, re.IGNORECASE):
            for nb in (i + 1, i - 1):
                if 0 <= nb < len(lines):
                    n = _PREFIX_RE.sub("", lines[nb]).strip()
                    m = last(n)
                    if m:
                        return m
    return ""


def heuristic_letters(response: str, options: dict) -> list:
    """Return list of (heuristic_name, letter) candidates found in response."""
    out = []
    if not response:
        return out
    tail = response.strip()
    m = re.search(r"Therefore[, ]+\(?([A-E])\)?\s*\.?\s*$", tail, re.IGNORECASE)
    if m:
        out.append(("therefore_trailing", m.group(1).upper()))
    m = re.search(
        r"(?:answer|choice|option)\s+(?:is|:|=)\s*\(?([A-E])\b",
        response, re.IGNORECASE,
    )
    if m:
        out.append(("answer_is_X", m.group(1).upper()))
    for ln in reversed([l.strip() for l in response.splitlines() if l.strip()]):
        m = re.fullmatch(r"\(?([A-E])\)?\.?", ln, re.IGNORECASE)
        if m:
            out.append(("bare_letter_line", m.group(1).upper()))
            break
    parens = list(re.finditer(r"\(([A-E])\)", response, re.IGNORECASE))
    if parens:
        out.append(("paren_letter_last", parens[-1].group(1).upper()))
    if options:
        hits = []
        for L, opt in options.items():
            opt = (opt or "").strip()
            if len(opt) >= 8 and opt.lower() in response.lower():
                hits.append((len(opt), L))
        if hits:
            hits.sort(reverse=True)
            out.append(("option_text_match", hits[0][1]))
    return out


def normalize_gt(row: dict) -> str:
    """Mirrors athena_mcq._normalize_mcq_gt."""
    raw = (row.get("GT") or row.get("answer") or row.get("correct_answer") or "").strip()
    if not raw:
        return ""
    if raw.upper() in _LETTER_SET:
        return raw.upper()
    if len(raw) < 3:
        return ""
    matches = []
    for L in _LETTERS:
        opt = row.get(f"option_{L.lower()}") or ""
        if re.search(rf"\b{re.escape(raw)}\b", opt, re.IGNORECASE):
            matches.append(L)
    return matches[0] if len(matches) == 1 else ""


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


def fmt_row(r: dict) -> str:
    tail = r["response_tail"].replace("\n", " \u21b5 ")
    return (
        f"  id={r['id']:>4}  gt={r['gt'] or '-'}  "
        f"stored_pred={r['stored_pred'] or '-'}  reparsed={r['reparsed_pred'] or '-'}  "
        f"status={r['status']}  fixable_by={r['fixable_by'] or '-'}\n"
        f"    heuristics={r['heuristics']}\n"
        f"    tail: ...{tail}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response-file", required=True, type=Path,
                    help="path to *_response.jsonl from athena_bench/inference.py")
    ap.add_argument("--benchmark-file", type=Path, default=None,
                    help="benchmark jsonl (optional; enables option-text match heuristic "
                         "and recovery of GT when the response file's 'answer' field is empty)")
    ap.add_argument("--out-mismatches", type=Path, default=None,
                    help="write all non-correct rows to this JSONL for manual review")
    ap.add_argument("--sample", type=int, default=15,
                    help="how many mismatch examples to print to stdout (default: 15)")
    args = ap.parse_args()

    if not args.response_file.is_file():
        sys.exit(f"response file not found: {args.response_file}")

    responses = load_jsonl(args.response_file)
    bench_by_id = {}
    if args.benchmark_file and args.benchmark_file.is_file():
        for idx, row in enumerate(load_jsonl(args.benchmark_file)):
            bench_by_id[idx] = row

    rows_out = []
    counters = Counter()
    gt_dist = Counter()
    pred_dist = Counter()
    recovery_by_heuristic = Counter()

    for r in responses:
        rid = r.get("id")
        resp = r.get("response") or ""
        stored_pred = (r.get("prediction") or "").upper()
        stored_gt = (r.get("answer") or "").upper()
        reparsed = current_parse_mcq(resp)

        gt = stored_gt if stored_gt in _LETTER_SET else ""
        if not gt and rid in bench_by_id:
            gt = normalize_gt(bench_by_id[rid])

        options = {}
        if rid in bench_by_id:
            brow = bench_by_id[rid]
            for L in _LETTERS:
                options[L] = brow.get(f"option_{L.lower()}") or brow.get(f"option_{L}") or ""

        heuristics = heuristic_letters(resp, options) if resp and resp != "Error" else []

        if not gt:
            status = "gt_missing"
        elif not reparsed:
            status = "parse_failed"
        elif reparsed == gt:
            status = "correct"
        else:
            status = "wrong"

        counters[status] += 1
        if gt:
            gt_dist[gt] += 1
        if reparsed:
            pred_dist[reparsed] += 1

        fixable = None
        if status in ("wrong", "parse_failed") and gt:
            for hname, letter in heuristics:
                if letter == gt:
                    fixable = hname
                    recovery_by_heuristic[hname] += 1
                    break

        rows_out.append({
            "id": rid,
            "gt": gt,
            "stored_pred": stored_pred,
            "reparsed_pred": reparsed,
            "status": status,
            "fixable_by": fixable,
            "heuristics": heuristics,
            "response_tail": (resp or "").strip()[-300:],
        })

    total = len(rows_out) or 1
    correct = counters["correct"]
    wrong = counters["wrong"]
    parse_failed = counters["parse_failed"]
    gt_missing = counters["gt_missing"]
    recoverable = sum(1 for r in rows_out if r.get("fixable_by"))

    print(f"=== MCQ diagnostic: {args.response_file.name} ===")
    print(f"  total rows            : {total}")
    print(f"  correct               : {correct:>5}  ({100 * correct / total:6.2f} %)")
    print(f"  wrong (model-chose-other): {wrong:>5}  ({100 * wrong / total:6.2f} %)")
    print(f"  parse_failed (pred='') : {parse_failed:>5}  ({100 * parse_failed / total:6.2f} %)")
    print(f"  gt_missing             : {gt_missing:>5}  ({100 * gt_missing / total:6.2f} %)  "
          "(excluded from accuracy denominator)")
    print()
    scorable = correct + wrong + parse_failed
    acc = 100 * correct / scorable if scorable else 0
    upper = 100 * (correct + recoverable) / scorable if scorable else 0
    print(f"  accuracy (scorable)    : {acc:6.2f} %  [{correct}/{scorable}]")
    print(f"  recoverable via heuristics: {recoverable}  "
          f"-> upper-bound accuracy {upper:6.2f} %")
    if recovery_by_heuristic:
        print(f"  recovery breakdown     : {dict(recovery_by_heuristic)}")
    print()
    print("  letter distribution (A  B  C  D  E):")
    print("    GT      : " + "  ".join(f"{gt_dist.get(L, 0):>5}" for L in _LETTERS))
    print("    Parsed  : " + "  ".join(f"{pred_dist.get(L, 0):>5}" for L in _LETTERS))
    print()

    if args.sample:
        shown = 0
        print(f"=== Up to {args.sample} wrong / parse_failed examples ===")
        for r in rows_out:
            if shown >= args.sample:
                break
            if r["status"] in ("wrong", "parse_failed"):
                print(fmt_row(r))
                print()
                shown += 1

    if args.out_mismatches:
        args.out_mismatches.parent.mkdir(parents=True, exist_ok=True)
        with args.out_mismatches.open("w", encoding="utf-8") as f:
            for r in rows_out:
                if r["status"] != "correct":
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[out] {args.out_mismatches}  ({total - correct} non-correct rows)")


if __name__ == "__main__":
    main()
