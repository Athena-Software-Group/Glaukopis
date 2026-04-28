"""CyberSOCEval - Threat Intel Reasoning (CrowdStrike / CISA / NSA / IC3 PDFs).

Text-only modality: each question carries a ``report_id`` whose extracted
plain text lives at ``<reports_dir>/<report_id>.txt`` (produced by
``utils/fetch_cybersoceval_data.py``). The image-modality path from the
upstream benchmark is intentionally unsupported here; the framework's
``get_single_prediction`` interface is text-only.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from benchmarks.base import Benchmark
from pipelines.data_loader import load_json_or_jsonl, sanitize_jsonl
from pipelines.evaluation.cybersoceval_eval import CYBERSOCEVALEvaluate
from pipelines.models import get_single_prediction, model_mapping
from pipelines.post_processing.cybersoceval import (
    canonicalize_answer_list,
    cybersoceval_postprocessing,
)


def _build_prompt(context: str, question: str, options) -> str:
    """Verbatim port of ThreatIntelReasoningBenchmark._create_prompt (text mode)."""
    suffix = (
        "\n\n        Given the context, answer the following question: "
        f"{question}. Options: {options}.\n"
        "        You need to return the list of correct answers. it is possible that "
        "there are multiple correct answers. or a single correct answer.\n"
        "        Respond in a JSON with the following structure:\n"
        "        {correct_answers: string[] //The list of the letters corresponding "
        "to the correct answers, just the letters}\n"
        "        surround the JSON response with <json_object></json_object> tags and "
        "make sure to return the JSON response only.\n"
        "        Example response:\n"
        "        <json_object>{\"correct_answers\": [\"A\", \"C\", \"D\"]}</json_object>.\n"
        "\n        "
    )
    return f"\n            {context}\n            " + suffix


class CYBERSOCEVALTI(Benchmark):
    """Threat-intel MCQ over long-context CrowdStrike / CISA / NSA / IC3 reports."""

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.task = "cybersoceval-ti"
        self.version = version
        self.data_file = data_path or (
            "benchmark_data/cybersoceval/threat_intel_reasoning/report_questions.jsonl"
        )
        self.reports_dir = Path(self.data_file).parent
        self.display_model_name = model_mapping.get(model_name, model_name).replace("/", "_")
        rows_str = str(num_rows) if num_rows else "all"
        self.model_folder = os.path.join("responses", self.display_model_name, self.task)
        os.makedirs(self.model_folder, exist_ok=True)
        self.response_file = os.path.join(
            self.model_folder,
            f"{self.task}_{rows_str}_v{self.version}_{self.display_model_name}_response.jsonl",
        )
        self.postprocessor = cybersoceval_postprocessing()
        self.eval = CYBERSOCEVALEvaluate()

    def _load_context(self, report_id: str) -> str:
        path = self.reports_dir / f"{report_id}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Report text not found: {path}. Run utils/fetch_cybersoceval_data.py first."
            )
        return path.read_text(encoding="utf-8", errors="ignore")

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        records = load_json_or_jsonl(self.data_file, num_rows=self.num_rows)
        # Drop any partial trailing line left by a previously-killed run
        # (ENOSPC, SIGTERM, OOM) before we open the file in append mode.
        # Without this, the next f.write would concatenate onto the partial
        # line and produce a corrupt mega-record that breaks scoring.
        sanitize_jsonl(self.response_file)
        processed_ids: set = set()
        if os.path.exists(self.response_file):
            with open(self.response_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        else:
            Path(self.response_file).touch()
        remaining = [(idx, row) for idx, row in enumerate(records) if idx not in processed_ids]
        if not remaining:
            print("All records already processed. Nothing to do.")
            return
        print(f"Processing {len(remaining)} remaining records...")

        def process_record(idx_row):
            idx, row = idx_row
            try:
                context = self._load_context(row["report_id"])
                prompt = _build_prompt(context, row["question_text"], row["options"])
                response = get_single_prediction(
                    prompt, self.model_name, task=self.task,
                    cleanup_after=cleanup, use_web_search=use_web_search,
                )
                prediction = self.postprocessor.extract_answer(self.task, response)
            except Exception as e:
                prompt, response, prediction = "", f"Error: {e}", ""
                print(f"Error processing id {idx}: {e}")
            return {
                "id": idx,
                "prompt": prompt,
                "response": response,
                "prediction": prediction,
                "answer": canonicalize_answer_list(row.get("correct_answer", [])),
                "report_id": row.get("report_id", ""),
                "source": row.get("source", ""),
                "url_source": row.get("url_source", ""),
                "question_id": row.get("question_id", ""),
                "options": row.get("options", []),
                "question_text": row.get("question_text", ""),
            }

        # fsync after each row makes the file durable against crashes:
        # if the process dies, every flushed line is a complete JSON record
        # (sanitize_jsonl on the next run would still be a safety net).
        with open(self.response_file, "a", encoding="utf-8") as f:
            if batch and batch > 1:
                with ThreadPoolExecutor(max_workers=batch) as ex:
                    futures = [ex.submit(process_record, r) for r in remaining]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                        rec = fut.result()
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
            else:
                for rec in tqdm(remaining, desc="Generating responses"):
                    out = process_record(rec)
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

    def evaluate_cybersoceval_ti(self):
        scored = Path(self.model_folder) / (
            f"{self.task}_{self.num_rows or 'all'}_v{self.version}_{self.display_model_name}_scored.jsonl"
        )
        return self.eval.evaluate_file(self.task, Path(self.response_file), scored)
