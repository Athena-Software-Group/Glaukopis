#!/usr/bin/env python3
"""Diagnose athena-{rcm,ate,rms,taa} results that scored near zero.

Re-runs the **production** extractor (pipelines.post_processing.athena_cti)
and the **production** scorer (pipelines.evaluation.athena_cti_eval) on
every row of a response.jsonl and classifies each row as one of:

  correct          - production extractor matched and production scorer
                     awards credit (TAA: alias/related match; RMS: F1>0;
                     ATE: T-ID equal after subtechnique flattening; RCM:
                     CWE equal).
  wrong_answer     - extractor produced a non-empty string but scorer
                     awards zero (knowledge failure, not a parser bug).
  extraction_miss  - extractor returned empty (format failure).

Separately surfaces **parser_false_zero** candidates: rows where the GT
token appears verbatim in the raw response but the extractor returned
empty. These are the rows worth inspecting to decide whether the
extractor needs loosening vs. the model simply failing to commit to an
answer in the expected shape.

Because this imports the production classes directly, the `correct`
count here must match what `run_benchmark.sh` already produced; any
drift signals a bug in this script, not in the parser.

Usage (on the inference host, run from SFT/eval/):

  # Single task:
  python utils/diagnose_zero_tasks.py \\
      --response-file responses/meta-llama_Llama-3.1-8B/athena-ate/athena-ate_all_v1_meta-llama_Llama-3.1-8B_response.jsonl \\
      --task athena-ate --sample 5

  # All four zero-scoring tasks for a given model at once:
  python utils/diagnose_zero_tasks.py \\
      --model-dir responses/meta-llama_Llama-3.1-8B --sample 3
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
from collections import Counter

# Resolve SFT/eval/ as the working import root so the production modules
# (pipelines.*) resolve regardless of where this script is invoked from.
HERE = Path(__file__).resolve().parent
SFT_TEST_ROOT = HERE.parent
sys.path.insert(0, str(SFT_TEST_ROOT))

from pipelines.post_processing.athena_cti import athena_cti_postprocessing  # noqa: E402
from pipelines.evaluation.athena_cti_eval import ATHENAEvaluate  # noqa: E402

SUPPORTED_TASKS = ("athena-rcm", "athena-ate", "athena-rms", "athena-taa")

# Token patterns used only for the "does GT appear verbatim in response"
# parser-miss heuristic. The production extractor itself is imported
# above; these are diagnostic-only.
GT_PRESENCE_PATTERN = {
    "athena-rcm": re.compile(r"CWE-\d+", re.IGNORECASE),
    "athena-ate": re.compile(r"T\d{4}(?:\.\d{3})?", re.IGNORECASE),
    "athena-rms": re.compile(r"M\d{4}", re.IGNORECASE),
}


def _make_evaluator() -> ATHENAEvaluate:
    """Instantiate the production evaluator with TAA alias CSVs resolved
    relative to SFT/eval/ (the usual cwd for run_benchmark.sh)."""
    try:
        return ATHENAEvaluate(predictions_dir=str(SFT_TEST_ROOT / "responses"))
    except FileNotFoundError:
        # TAA CSVs absent; fall back to empty dicts so RCM/ATE/RMS still
        # score and TAA degrades to string equality.
        ev = ATHENAEvaluate.__new__(ATHENAEvaluate)
        ev.pred_dir = SFT_TEST_ROOT / "responses"
        ev.alias_dict = {}
        ev.related_dict = {}
        ev.processor = athena_cti_postprocessing()
        return ev


_POST = athena_cti_postprocessing()
_EVAL = _make_evaluator()


def gt_appears_in(task: str, response: str, gt: str) -> bool:
    """True when the GT token appears somewhere in the raw response.

    For id-shaped tasks (RCM/ATE/RMS) uses a case-insensitive substring
    test anchored to the task's id regex so 'Not CWE-79 but ...' counts
    as GT-present if GT=='CWE-79'. For TAA falls back to a lowercase
    substring check on the actor name.
    """
    if not gt:
        return False
    if task in ("athena-rcm", "athena-ate", "athena-rms"):
        # Pull all ids of the right shape out of the response and
        # compare on the same-shape normalization that the evaluator uses.
        shape = GT_PRESENCE_PATTERN[task]
        hits = {m.group(0).upper() for m in shape.finditer(response or "")}
        if task == "athena-ate":
            hits = {h.split(".")[0] for h in hits}
            gt = gt.split(".")[0]
        return gt.upper() in hits
    return gt.strip().lower() in (response or "").lower()


def score_row(task: str, pred: str, ans: str, record=None) -> float:
    """Call the production scorer. Returns a numeric score (0/1 for
    exact-match tasks, strict F1 for RMS, combined 0/0.5/1 for TAA)."""
    res, _ok = _EVAL.score_record(
        task, pred or "", ans or "", _EVAL.alias_dict, _EVAL.related_dict, record=record
    )
    if isinstance(res, dict):
        # TAA: {correct, plausible, combined}; RMS: {f1, plausible_f1, combined_f1, ...}
        if "correct" in res:
            return float(res.get("correct", 0.0))
        return float(res.get("f1", 0.0))
    return float(res)


def classify(task: str, rows, sample_n: int):
    c = Counter()
    samples = {"extraction_miss": [], "wrong_answer": [], "parser_false_zero": []}
    for r in rows:
        resp = r.get("response", "") or ""
        gt = str(r.get("answer", "") or r.get("gt", "") or "")
        pred = _POST.extract_answer(task, resp)
        if not pred:
            bucket = "parser_false_zero" if gt_appears_in(task, resp, gt) else "extraction_miss"
            c[bucket] += 1
            if len(samples[bucket]) < sample_n:
                samples[bucket].append({"id": r.get("id"), "gt": gt, "response": resp[:800]})
            continue
        score = score_row(task, pred, gt, record=r)
        if score >= 1.0 - 1e-9 or (task == "athena-rms" and score > 0):
            c["correct"] += 1
        else:
            c["wrong_answer"] += 1
            if len(samples["wrong_answer"]) < sample_n:
                samples["wrong_answer"].append({
                    "id": r.get("id"), "pred": pred, "gt": gt,
                    "score": round(score, 3), "response": resp[:500],
                })
    return c, samples


def verdict(task: str, n: int, c: Counter) -> str:
    false_zero = c.get("parser_false_zero", 0)
    miss = c.get("extraction_miss", 0)
    if n == 0:
        return "n/a (empty file)"
    fz_pct = 100 * false_zero / n
    miss_pct = 100 * miss / n
    if false_zero == 0 and miss_pct < 5:
        return "PARSER OK (no GT-in-response rows missed; extraction_miss < 5%)"
    if false_zero == 0:
        return f"PARSER OK, MODEL FORMAT WEAK ({miss_pct:.1f}% empty extractions, but GT never present in those)"
    if fz_pct >= 5:
        return f"PARSER SUSPECT ({fz_pct:.1f}% of rows have GT in response but extractor missed)"
    return f"PARSER MOSTLY OK ({false_zero} false-zero rows, {fz_pct:.1f}% of total)"


def report_one(path: Path, task: str, sample_n: int):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print(f"[{task}] {path.name}: empty file"); return
    c, samples = classify(task, rows, sample_n)
    n = len(rows)
    correct = c.get("correct", 0)
    wrong = c.get("wrong_answer", 0)
    miss = c.get("extraction_miss", 0)
    fz = c.get("parser_false_zero", 0)
    print(f"\n=== {task} :: {path.name} ===")
    print(f"  rows              : {n}")
    print(f"  correct           : {correct:>5} ({100*correct/n:5.1f}%)  (production scorer > 0)")
    print(f"  wrong_answer      : {wrong:>5} ({100*wrong/n:5.1f}%)  (extractor hit, scorer 0)")
    print(f"  extraction_miss   : {miss:>5} ({100*miss/n:5.1f}%)  (extractor empty, GT absent)")
    print(f"  parser_false_zero : {fz:>5} ({100*fz/n:5.1f}%)  (extractor empty, GT present in response)")
    print(f"  verdict           : {verdict(task, n, c)}")
    for bucket, items in samples.items():
        if not items:
            continue
        print(f"\n  --- sample[{bucket}] ---")
        for s in items:
            extras = f"  pred={s.get('pred','')!r}" if 'pred' in s else ""
            score = f"  score={s.get('score')}" if 'score' in s else ""
            print(f"  id={s.get('id')}  gt={s.get('gt')!r}{extras}{score}")
            print(f"  response: {s['response']!r}")
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--response-file", help="single *_response.jsonl")
    g.add_argument("--model-dir", help="responses/<model> to scan all zero-scoring tasks")
    ap.add_argument("--task", choices=list(SUPPORTED_TASKS), help="required with --response-file")
    ap.add_argument("--sample", type=int, default=3, help="rows to show per bucket (default 3)")
    args = ap.parse_args()
    if args.response_file:
        if not args.task:
            sys.exit("--task required with --response-file")
        report_one(Path(args.response_file), args.task, args.sample)
    else:
        base = Path(args.model_dir)
        for task in SUPPORTED_TASKS:
            d = base / task
            if not d.is_dir():
                continue
            files = sorted(d.glob(f"{task}_all_v*_response.jsonl"))
            if not files:
                print(f"[{task}] no response files under {d}"); continue
            report_one(files[-1], task, args.sample)
