"""Evaluator for CyberSOCEval (CrowdStrike + Meta) tasks.

Mirrors the upstream ``process_results`` pipeline from PurpleLlama's
``crwd_meta`` benchmarks: per-row Jaccard similarity between the model's
predicted set of MCQ letters and the ground-truth set, then aggregated
into ``avg_score`` / ``correct_mc_pct`` / ``response_parsing_error_count``
overall and per-slice (topic / attack / difficulty for malware; source /
report for threat-intel).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

from pipelines.data_loader import load_jsonl
from pipelines.post_processing.cybersoceval import cybersoceval_postprocessing


def _split_letters(canon: str) -> List[str]:
    return [s for s in (canon or "").split(",") if s]


def jaccard_similarity(pred: Iterable[str], gold: Iterable[str]) -> float:
    a, b = set(pred), set(gold)
    union = len(a | b)
    return float(len(a & b)) / union if union else 0.0


def _empty_bucket() -> Dict[str, float]:
    return {
        "avg_score": 0.0,
        "total_score": 0.0,
        "correct_mc_count": 0,
        "incorrect_mc_count": 0,
        "response_parsing_error_count": 0,
        "correct_mc_pct": 0.0,
    }


def _update_bucket(bucket: Dict[str, float], pred: str, score: float, parsed: bool) -> None:
    if not parsed:
        bucket["response_parsing_error_count"] += 1
    elif score == 1.0:
        bucket["correct_mc_count"] += 1
        bucket["total_score"] += score
    else:
        bucket["incorrect_mc_count"] += 1
        bucket["total_score"] += score
    answered = bucket["correct_mc_count"] + bucket["incorrect_mc_count"]
    if answered > 0:
        bucket["correct_mc_pct"] = bucket["correct_mc_count"] / answered
        bucket["avg_score"] = bucket["total_score"] / answered


# Slice keys per task. The malware benchmark exposes topic / attack /
# difficulty; threat-intel exposes source / report_id.
_SLICE_KEYS: Dict[str, Tuple[str, ...]] = {
    "cybersoceval-malware": ("topic", "attack", "difficulty"),
    "cybersoceval-ti": ("source", "report_id"),
}


class CYBERSOCEVALEvaluate:
    def __init__(self) -> None:
        self.processor = cybersoceval_postprocessing()

    def score_row(self, pred_letters: List[str], gold_letters: List[str]) -> Tuple[float, bool]:
        """Return (jaccard_score, was_parseable)."""
        if not pred_letters and not gold_letters:
            return 0.0, False
        if not pred_letters:
            return 0.0, False
        return jaccard_similarity(pred_letters, gold_letters), True

    def evaluate_file(self, task: str, preds_path: Path, out_path: Path) -> Dict[str, object]:
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[Cache] Using existing scored file: {out_path}")
            records = load_jsonl(str(out_path))
        else:
            print(f"[Scoring] Computing scores from predictions: {preds_path}")
            raw = load_jsonl(str(preds_path))
            records = []
            for rec in tqdm(raw, total=len(raw), desc=str(preds_path)):
                response = rec.get("response", "") or ""
                pred_canon = self.processor.extract_answer(task, response)
                gold_canon = rec.get("answer", "") or ""
                pred_set = _split_letters(pred_canon)
                gold_set = _split_letters(gold_canon)
                score, parsed = self.score_row(pred_set, gold_set)
                records.append({
                    **rec,
                    "prediction": pred_canon,
                    "score": score,
                    "success": parsed,
                })
            with out_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        overall = _empty_bucket()
        slice_keys = _SLICE_KEYS.get(task, ())
        slices: Dict[str, Dict[str, Dict[str, float]]] = {k: {} for k in slice_keys}

        for rec in records:
            pred = rec.get("prediction", "")
            score = float(rec.get("score", 0.0))
            parsed = bool(rec.get("success", False))
            _update_bucket(overall, pred, score, parsed)
            for k in slice_keys:
                v = rec.get(k, "") or "(unknown)"
                bucket = slices[k].setdefault(str(v), _empty_bucket())
                _update_bucket(bucket, pred, score, parsed)

        total = len(records)
        parse_err = overall["response_parsing_error_count"]
        metrics: Dict[str, object] = {
            "accuracy": (overall["correct_mc_pct"] or 0.0) * 100.0,
            "avg_score": (overall["avg_score"] or 0.0) * 100.0,
            "correct_mc_count": overall["correct_mc_count"],
            "incorrect_mc_count": overall["incorrect_mc_count"],
            "response_parsing_error_count": parse_err,
            "parse_error_pct": (parse_err / total * 100.0) if total else 0.0,
            "total": total,
        }
        for k in slice_keys:
            metrics[f"per_{k}"] = {
                key: {
                    "avg_score": round((v["avg_score"] or 0.0) * 100.0, 2),
                    "correct_mc_pct": round((v["correct_mc_pct"] or 0.0) * 100.0, 2),
                    "answered": v["correct_mc_count"] + v["incorrect_mc_count"],
                    "parse_errors": v["response_parsing_error_count"],
                }
                for key, v in sorted(slices[k].items())
            }
        return metrics
