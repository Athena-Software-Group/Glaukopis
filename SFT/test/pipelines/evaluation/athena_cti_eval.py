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


RMS_NEIGHBORHOOD_PATH = "benchmark_data/athena_bench/athena_rms/mitigation_neighborhood.json"
RMS_SOURCE_PATH = "benchmark_data/athena_bench/athena-cti-rms.jsonl"


class ATHENAEvaluate:
    def __init__(
        self,
        predictions_dir: str = "responses",
        alias_csv: str = "benchmark_data/athena_bench/athena_taa/aliases.csv",
        related_csv: str = "benchmark_data/athena_bench/athena_taa/related_groups.csv",
        rms_neighborhood_path: str = RMS_NEIGHBORHOOD_PATH,
        rms_source_path: str = RMS_SOURCE_PATH,
    ):
        self.pred_dir = Path(predictions_dir)
        self.alias_dict = self.load_alias_dict(alias_csv)
        self.related_dict = self.load_related_dict(related_csv)
        self.processor = athena_cti_postprocessing()
        self.rms_neighborhood = self._load_rms_neighborhood(rms_neighborhood_path)
        self._rms_source_path = rms_source_path
        self._rms_prompt_hash_to_tid: Dict[str, str] | None = None

    @staticmethod
    def _load_rms_neighborhood(path: str) -> Dict[str, Dict[str, List[str]]]:
        p = Path(path)
        if not p.exists():
            print(
                f"[Warn] RMS mitigation neighborhood not found at {p}; "
                f"plausible/combined F1 will fall back to strict F1. "
                f"Run python -m athena_data.mitre_attck.build_mitigation_neighborhood to generate it."
            )
            return {}
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _rms_tid_for_prompt_hash(self, prompt_hash: str | None) -> str | None:
        if not prompt_hash:
            return None
        if self._rms_prompt_hash_to_tid is None:
            src = Path(self._rms_source_path)
            mapping: Dict[str, str] = {}
            if src.exists():
                for row in load_jsonl(str(src)):
                    h = row.get("prompt_hash")
                    tid = row.get("technique_id")
                    if h and tid:
                        mapping[h] = tid
            self._rms_prompt_hash_to_tid = mapping
        return self._rms_prompt_hash_to_tid.get(prompt_hash)

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

    def _resolve_actor_in_text(
        self, pred: str, alias_dict: Dict[str, List[str]]
    ) -> str:
        """Normalise a TAA prediction to a bare alias-dict key.

        The TAA extractor (r"(.+)") returns the whole trained conclusion
        sentence, e.g. "therefore, the adversary is apt29.". The alias BFS
        expects a bare key ("apt29"), so if *pred* is not itself a key we
        search it for the longest alias-dict key occurring as a whole word
        and return that; if nothing matches, fall back to *pred* unchanged.
        """
        t = pred.strip().lower()
        if not t or t in alias_dict:
            return t
        candidates = []
        for k in alias_dict:
            if not k or len(k) < 3:
                continue
            if re.search(r"(?:^|[^a-z0-9])" + re.escape(k) + r"(?:$|[^a-z0-9])", t):
                candidates.append(k)
        if not candidates:
            return t
        candidates.sort(key=lambda k: (-len(k), t.find(k)))
        return candidates[0]

    def threat_actor_connection(
        self,
        actor1: str,
        actor2: str,
        alias_dict: Dict[str, List[str]],
        related_dict: Dict[str, List[str]],
    ) -> str:
        actor1 = actor1.strip().lower()
        actor2 = self._resolve_actor_in_text(actor2, alias_dict)
        if not actor2:
            return "I"
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

    @staticmethod
    def _f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def score_rms(self, pred: str, ans: str, technique_id: str | None) -> Tuple[Dict[str, float], bool]:
        """Strict / plausible / combined F1 over predicted MITRE mitigation IDs.

        Strict   : exact set-F1 over gold mitigation IDs (legacy behaviour).
        Plausible: predictions that hit gold OR a neighbour-technique
                   mitigation count toward precision; recall denominator
                   stays the strict gold so missing the actual answer is
                   still penalised.
        Combined : strict matches contribute 1.0, plausible-only matches
                   contribute 0.5 toward precision; recall is strict.
        """
        p_ids = set(re.findall(r"M\d{4}", pred.upper()))
        a_ids = set(re.findall(r"M\d{4}", ans.upper()))

        tp = len(p_ids & a_ids)
        fp = len(p_ids - a_ids)
        fn = len(a_ids - p_ids)
        strict_p = tp / (tp + fp) if (tp + fp) else 0.0
        strict_r = tp / (tp + fn) if (tp + fn) else 0.0
        strict_f1 = self._f1(strict_p, strict_r)

        neigh = self.rms_neighborhood.get(technique_id or "", {})
        plausible_set = set(neigh.get("plausible", []))

        if plausible_set:
            ext_gold = a_ids | plausible_set
            tp_p = len(p_ids & ext_gold)
            plaus_p = tp_p / len(p_ids) if p_ids else 0.0
            plaus_f1 = self._f1(plaus_p, strict_r)

            strict_hits = len(p_ids & a_ids)
            plaus_only = len(p_ids & plausible_set) - len(p_ids & a_ids & plausible_set)
            comb_tp = strict_hits + 0.5 * plaus_only
            comb_p = comb_tp / len(p_ids) if p_ids else 0.0
            comb_f1 = self._f1(comb_p, strict_r)
        else:
            plaus_p = strict_p
            plaus_f1 = strict_f1
            comb_p = strict_p
            comb_f1 = strict_f1

        score = {
            "f1": strict_f1,
            "precision": strict_p,
            "recall": strict_r,
            "plausible_f1": plaus_f1,
            "plausible_precision": plaus_p,
            "combined_f1": comb_f1,
            "combined_precision": comb_p,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_pred": len(p_ids),
            "n_gold": len(a_ids),
            "n_plausible_neighbors": len(plausible_set),
        }
        return score, True

    def score_record(self, task: str, pred: str, ans: str, alias_dict, related_dict, record=None):
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
            tid = (record or {}).get("technique_id") if isinstance(record, dict) else None
            if not tid and isinstance(record, dict):
                tid = self._rms_tid_for_prompt_hash(record.get("prompt_hash"))
            return self.score_rms(pred, ans, tid)
        # default MCQ style
        return (1 if pred.strip().lower() == ans.strip().lower() else 0, True)

    # -----------------------------------------------------------------------
    # Main evaluation logic
    def format_percentage_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        formatted: Dict[str, float] = {}
        for key, value in metrics.items():
            kl = key.lower()
            is_pct = "accuracy" in kl or kl == "f1" or kl.endswith("_f1")
            if isinstance(value, (int, float)) and is_pct:
                formatted[key] = value * 100.0
            else:
                formatted[key] = value
        return formatted

    def evaluate_file(self, task: str, preds_path: Path, out_path: Path, vsp_denominator: float = 7.7) -> Dict[str, float]:
        # Cache freshness: the scored file is only reusable when it is at least
        # as recent as the predictions file. Otherwise (e.g. after
        # `run_benchmark.sh --retry-errors` scrubbed error rows and the next
        # run regenerated them) the cached scores reflect a stale snapshot of
        # the responses and silently re-emit broken metrics.
        cache_fresh = (
            out_path.exists()
            and out_path.stat().st_size > 0
            and preds_path.exists()
            and out_path.stat().st_mtime >= preds_path.stat().st_mtime
        )
        if cache_fresh:
            print(f"[Cache] Using existing scored file: {out_path}")
            records = load_jsonl(str(out_path))
        else:
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[Cache] Stale (predictions newer); re-scoring: {out_path}")
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

            sum_f1 = 0.0
            sum_plaus_f1 = 0.0
            sum_comb_f1 = 0.0

            for _, rec in iterator:
                response = rec.get("response", "")
                prompt = rec.get("prompt", "") or ""
                pred = self.processor.extract_answer(task, response, prompt=prompt)
                ans = rec.get("answer", "")
                score, success = self.score_record(
                    task, pred, ans, self.alias_dict, self.related_dict, record=rec
                )

                records.append({**rec, "score": score, "success": success})

                if success:
                    count_success += 1
                    if isinstance(score, dict) and "correct" in score:
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
                    elif isinstance(score, dict) and "f1" in score:
                        sum_f1 += score.get("f1", 0.0)
                        sum_plaus_f1 += score.get("plausible_f1", 0.0)
                        sum_comb_f1 += score.get("combined_f1", 0.0)
                        iterator.set_postfix(
                            {
                                "avg_f1": f"{sum_f1 / count_success:.3f}",
                                "avg_plaus_f1": f"{sum_plaus_f1 / count_success:.3f}",
                                "avg_comb_f1": f"{sum_comb_f1 / count_success:.3f}",
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
            if scores and isinstance(scores[0], dict):
                n = len(scores)
                f1 = sum(s.get("f1", 0.0) for s in scores) / n
                plaus_f1 = sum(s.get("plausible_f1", 0.0) for s in scores) / n
                comb_f1 = sum(s.get("combined_f1", 0.0) for s in scores) / n
                covered = sum(1 for s in scores if s.get("n_plausible_neighbors", 0) > 0)
                metrics = {
                    "f1": f1,
                    "plausible_f1": plaus_f1,
                    "combined_f1": comb_f1,
                    "n_scored": n,
                    "n_with_neighborhood": covered,
                }
            else:
                # Legacy scored files where score is a bare float (strict F1).
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
