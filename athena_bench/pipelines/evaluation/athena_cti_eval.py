"""Evaluate model predictions for benchmark tasks (class version)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from cvss import CVSS3
from tqdm import tqdm

from pipelines.data_loader import load_jsonl, load_yaml
from pipelines.post_processing.athena_cti import athena_cti_postprocessing


class ATHENAEvaluate:
    def __init__(
        self,
        predictions_dir: str = "responses",
        alias_csv: str = "benchmark_data/athena_bench/athena_taa/aliases.csv",
        related_csv: str = "benchmark_data/athena_bench/athena_taa/related_groups.csv"
    ):
        self.pred_dir = Path(predictions_dir)
        self.alias_dict = self.load_alias_dict(alias_csv)
        self.related_dict = self.load_related_dict(related_csv)
        self.processor = athena_cti_postprocessing()

    # -----------------------------------------------------------------------
    # TAA helpers

    def load_alias_dict(self, path: str) -> Dict[str, List[str]]:
        alias: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row["ThreatActor"].strip().lower()
                v = row["Alias"].strip().lower()
                alias.setdefault(k, []).append(v)
                alias.setdefault(v, []).append(k)
        return alias

    def load_related_dict(self, path: str) -> Dict[str, List[str]]:
        related: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row["ThreatActor"].strip().lower()
                v = row["RelatedGroup"].strip().lower()
                related.setdefault(k, []).append(v)
                related.setdefault(v, []).append(k)
        return related

    def is_alias_connected(self, actor1: str, actor2: str, alias_dict: Dict[str, List[str]]) -> bool:
        visited = set()
        queue = [actor1]
        while queue:
            cur = queue.pop(0)
            if cur == actor2:
                return True
            visited.add(cur)
            for nxt in alias_dict.get(cur, []):
                if nxt not in visited:
                    queue.append(nxt)
        return False

    def is_related_connected(
        self,
        actor1: str,
        actor2: str,
        alias_dict: Dict[str, List[str]],
        related_dict: Dict[str, List[str]],
    ) -> bool:
        visited = set()
        queue = [actor1]
        while queue:
            cur = queue.pop(0)
            if cur == actor2:
                return True
            visited.add(cur)
            neighbours = alias_dict.get(cur, []) + related_dict.get(cur, [])
            for nxt in neighbours:
                if nxt not in visited:
                    queue.append(nxt)
        return False

    def threat_actor_connection(
        self,
        actor1: str,
        actor2: str,
        alias_dict: Dict[str, List[str]],
        related_dict: Dict[str, List[str]],
    ) -> str:
        actor1 = actor1.strip().lower()
        actor2 = actor2.strip().lower()
        if self.is_alias_connected(actor1, actor2, alias_dict):
            return "C"
        if self.is_related_connected(actor1, actor2, alias_dict, related_dict):
            return "P"
        return "I"

    def score_taa(
        self,
        pred: str,
        ans: str,
        alias_dict: Dict[str, List[str]],
        related_dict: Dict[str, List[str]],
    ) -> Tuple[Dict[str, float], bool]:
        res = self.threat_actor_connection(ans, pred, alias_dict, related_dict)
        score = {
            "correct": 1 if res == "C" else 0,
            "plausible": 1 if res in {"C", "P"} else 0,
            "combined": 1.0 if res == "C" else 0.5 if res == "P" else 0.0,
        }
        return score, True

    # -----------------------------------------------------------------------
    # Task scoring

    def score_record(self, task: str, pred: str, ans: str, alias_dict, related_dict):
        task = task.lower()
        if task == "athena-rcm":
            return (1 if pred.strip().lower() == ans.strip().lower() else 0, True)
        if task == "athena-vsp":
            try:
                p = CVSS3(pred.strip()).scores()[0]
                a = CVSS3(ans.strip()).scores()[0]
                return (abs(p - a), True)
            except Exception:
                return (0.0, False)
        if task == "athena-taa":
            return self.score_taa(pred, ans, alias_dict, related_dict)
        if task == "athena-ate":
            p = pred.strip().split(".")[0].upper()
            a = ans.strip().split(".")[0].upper()
            return (1 if p and p == a else 0, True)
        if task == "athena-rms":
            p_ids = set(re.findall(r"M\d{4}", pred.upper()))
            a_ids = set(re.findall(r"M\d{4}", ans.upper()))
            tp = len(p_ids & a_ids)
            fp = len(p_ids - a_ids)
            fn = len(a_ids - p_ids)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            return (f1, True)
        # default MCQ style
        return (1 if pred.strip().lower() == ans.strip().lower() else 0, True)

    # -----------------------------------------------------------------------
    # Main evaluation logic
    def format_percentage_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        formatted: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and (
                "accuracy" in key.lower() or key.lower() == "f1"
            ):
                formatted[key] = value * 100.0
            else:
                formatted[key] = value
        return formatted

    def evaluate_file(self, task: str, preds_path: Path, out_path: Path, vsp_denominator: float = 7.7) -> Dict[str, float]:
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[Cache] Using existing scored file: {out_path}")
            records = load_jsonl(str(out_path))
        else:
            print(f"[Scoring] Computing scores from predictions: {preds_path}")
            raw_records = load_jsonl(str(preds_path))
            records = []
            sum_score = 0.0
            sum_correct = 0
            sum_plausible = 0
            sum_combined = 0.0
            count_success = 0

            iterator = enumerate(raw_records)
            iterator = tqdm(iterator, total=len(raw_records), desc=str(preds_path))

            for _, rec in iterator:
                response = rec.get("response", "")
                pred = self.processor.extract_answer(task, response)
                ans = rec.get("answer", "")
                score, success = self.score_record(task, pred, ans, self.alias_dict, self.related_dict)

                records.append({**rec, "score": score, "success": success})

                if success:
                    count_success += 1
                    if isinstance(score, dict):
                        sum_correct += score.get("correct", 0)
                        sum_plausible += score.get("plausible", 0)
                        sum_combined += score.get("combined", 0.0)
                        iterator.set_postfix(
                            {
                                "avg_correct": f"{sum_correct / count_success:.3f}",
                                "avg_plausible": f"{sum_plausible / count_success:.3f}",
                                "avg_combined": f"{sum_combined / count_success:.3f}",
                            }
                        )
                    else:
                        sum_score += float(score)
                        iterator.set_postfix({"avg_score": f"{sum_score / count_success:.3f}"})

            with out_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        task_up = task.lower()

        if task_up == "athena-taa":
            corr = [r["score"]["correct"] for r in records if r["success"]]
            plaus = [r["score"]["plausible"] for r in records if r["success"]]
            comb = [r["score"]["combined"] for r in records if r["success"]]
            metrics = {
                "accuracy": sum(corr) / len(corr) if corr else 0.0,
                "plausible_accuracy": sum(plaus) / len(plaus) if plaus else 0.0,
                "combined_accuracy": sum(comb) / len(comb) if comb else 0.0,
            }
            return self.format_percentage_metrics(metrics)

        scores = [r["score"] for r in records if r["success"]]

        if task_up == "athena-rms":
            metrics = {"f1": sum(scores) / len(scores) if scores else 0.0}
            return self.format_percentage_metrics(metrics)

        if task_up == "athena-vsp":
            mad = sum(scores) / len(scores) if scores else 0.0
            denom = vsp_denominator if vsp_denominator else 1.0
            if denom == 0:
                denom = 1.0
            accuracy = 1 - (mad / denom)
            metrics = {"MAD": mad, "accuracy": accuracy}
            return self.format_percentage_metrics(metrics)

        if task_up == "cvss":
            metrics = {"mean_absolute_deviation": sum(scores) / len(scores) if scores else 0.0}
            return metrics

        if task_up == "athena-mcq":
            metrics = {"accuracy": sum(scores) / len(scores) if scores else 0.0}
            return self.format_percentage_metrics(metrics)

        metrics = {"accuracy": sum(scores) / len(scores) if scores else 0.0}
        return self.format_percentage_metrics(metrics)

    # -----------------------------------------------------------------------
    def evaluate_folder(self, model_name: str, mini: bool = False) -> Dict[str, Dict]:
        tasks = [
            "athena-mcq",
            "athena-ate",
            "athena-rcm",
            "athena-rms",
            "athena-vsp",
            "athena-taa",
        ]

        results = {}
        avg_display = {}

        mcq3k_accuracy = None           # single source of truth
        other_task_values: List[float] = []

        full_runs_dir = self.pred_dir / model_name
        runs_mini_dir = Path("runs_mini")

        print(f"\nEvaluating Athena tasks for model: {model_name}")
        print(f"Mini mode: {mini}\n")

        if mini:
            runs_mini_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            task_dir = full_runs_dir / task
            response_files = list(task_dir.glob("*_response.jsonl")) if task_dir.exists() else []

            if not response_files:
                print(f"[Skip] No response file found for task: {task}")
                continue

            full_preds_path = response_files[0]
            metrics = None

            # ---------------- MINI MODE ----------------
            if mini:
                out_dir = runs_mini_dir / model_name / task
                out_dir.mkdir(parents=True, exist_ok=True)

                if task.startswith("athena-mcq"):
                    n_full = sum(1 for _ in load_jsonl(str(full_preds_path)))
                    mini_datasets = (
                        ["athena-cti-mcq.jsonl", "athena-cti-mcq-3k.jsonl"]
                        if n_full > 3000
                        else ["athena-cti-mcq-3k.jsonl"]
                    )
                else:
                    mini_datasets = [f"athena-cti-{task.split('-')[-1]}.jsonl"]

                for mini_ds_name in mini_datasets:
                    mini_ds = Path(
                        "/Users/aiman/Desktop/athena-cti-bench/benchmark_data_mini"
                    ) / mini_ds_name

                    if not mini_ds.exists():
                        print(f"[Skip] Mini dataset missing for {task} ({mini_ds_name})")
                        continue

                    scored_path = out_dir / f"{Path(mini_ds_name).stem}_scored.jsonl"

                    if scored_path.exists() and scored_path.stat().st_size > 0:
                        metrics = self.evaluate_file(task, scored_path, scored_path)
                    else:
                        def phash(s: str | None) -> str:
                            return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

                        mini_records = load_jsonl(str(mini_ds))
                        mini_map = {
                            r.get("prompt_hash") or phash(r.get("prompt")): r.get("answer", "")
                            for r in mini_records
                        }
                        mini_order = list(mini_map.keys())

                        full_preds = load_jsonl(str(full_preds_path))
                        filtered = []
                        for rec in full_preds:
                            h = rec.get("prompt_hash") or phash(rec.get("prompt"))
                            if h in mini_map:
                                new_rec = dict(rec)
                                new_rec["answer"] = mini_map[h]
                                filtered.append(new_rec)

                        if not filtered:
                            print(f"[Warning] No predictions matched mini dataset for {task}")
                            continue

                        order_index = {h: i for i, h in enumerate(mini_order)}
                        filtered.sort(
                            key=lambda r: order_index.get(
                                r.get("prompt_hash") or phash(r.get("prompt")), 10**9
                            )
                        )

                        tmp_path = out_dir / f"{Path(mini_ds_name).stem}_response.jsonl"
                        with tmp_path.open("w", encoding="utf-8") as f:
                            for r in filtered:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")

                        metrics = self.evaluate_file(task, tmp_path, scored_path)
                        tmp_path.unlink(missing_ok=True)

                    value = metrics.get("accuracy") or metrics.get("f1")

                    if task.startswith("athena-mcq"):
                        if mini_ds_name == "athena-cti-mcq.jsonl":
                            avg_display["MCQ"] = value
                        else:
                            avg_display["MCQ3K"] = value
                            if mcq3k_accuracy is None:
                                mcq3k_accuracy = value
                    else:
                        avg_display[task.upper()] = value
                        other_task_values.append(value)

                    results[f"{task}_{mini_ds_name}"] = metrics
                    print(f"{model_name} {task} ({mini_ds_name}): {metrics}")

            # ---------------- FULL MODE ----------------
            else:
                scored_path = full_preds_path.with_name(
                    full_preds_path.stem.replace("_response", "") + "_scored.jsonl"
                )

                metrics = (
                    self.evaluate_file(task, scored_path, scored_path)
                    if scored_path.exists() and scored_path.stat().st_size > 0
                    else self.evaluate_file(task, full_preds_path, scored_path)
                )

                if metrics is None:
                    continue

                value = metrics.get("accuracy") or metrics.get("f1")

                if task.startswith("athena-mcq"):
                    n_samples = sum(1 for _ in load_jsonl(str(full_preds_path)))
                    if n_samples <= 3000:
                        avg_display["MCQ3K"] = value
                        mcq3k_accuracy = value
                    else:
                        avg_display["MCQ"] = value
                else:
                    avg_display[task.upper()] = value
                    other_task_values.append(value)

                results[task] = metrics
                print(f"{model_name} {task}: {metrics}")

        # ---------------- FINAL COMBINED ----------------
        combined_entries = []
        if mcq3k_accuracy is not None:
            combined_entries.append(mcq3k_accuracy)
        combined_entries.extend(other_task_values)

        if combined_entries:
            combined = sum(combined_entries) / len(combined_entries)
            results["combined"] = combined

            print(f"\n{model_name} Athena overall score: {combined:.4f}")
            headers = "\t".join(avg_display.keys()) + "\tCombined"
            values = "\t".join(f"{v:.4f}" for v in avg_display.values()) + f"\t{combined:.4f}"
            print(headers)
            print(values)

        return results
