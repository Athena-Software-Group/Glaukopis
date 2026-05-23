from pathlib import Path
from benchmarks.base import Benchmark
from pipelines.post_processing.athena_cti import athena_cti_postprocessing
from pipelines.evaluation.athena_cti_eval import ATHENAEvaluate
from pipelines.data_loader import load_json_or_jsonl
from pipelines.models import get_single_prediction, alias_to_safe_name
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
import os

class ATHENATAA(Benchmark):
    """Benchmark class for CTI Threat Actor Association (TAA) with thread pooling + safe resume."""

    def __init__(self, model_name, num_rows=None, data_path=None,version=1):
        super().__init__(model_name, num_rows)
        self.task = 'athena-taa'
        self.version = version
        self.data_file = data_path if data_path else 'benchmark_data/athena_bench/athena_taa/athena-cti-taa.jsonl'
        self.display_model_name = alias_to_safe_name(model_name)
        rows_str = str(num_rows) if num_rows else "all"

        self.model_folder = os.path.join("responses", self.display_model_name, self.task)
        os.makedirs(self.model_folder, exist_ok=True)

        self.response_file = os.path.join(
            self.model_folder, f'{self.task}_{rows_str}_v{self.version}_{self.display_model_name}_response.jsonl'
        )

        self.postprocessor = athena_cti_postprocessing()
        self.eval = ATHENAEvaluate()

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        """Generate predictions with thread pooling and safe resume using IDs."""
        records = load_json_or_jsonl(self.data_file, num_rows=self.num_rows)

        # --- Resume logic using IDs ---
        processed_ids = set()
        if os.path.exists(self.response_file):
            with open(self.response_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        processed_ids.add(rec['id'])
                    except json.JSONDecodeError:
                        continue  # skip broken lines
        else:
            Path(self.response_file).touch()

        # Filter only records that were not processed
        remaining_records = [(idx, row) for idx, row in enumerate(records) if idx not in processed_ids]

        if not remaining_records:
            print("All records already processed. Nothing to do.")
            return

        print(f"Processing {len(remaining_records)} remaining records...")

        # --- Worker function ---
        def process_record(idx_row):
            idx, row = idx_row
            prompt = row.get('Prompt') or row.get('prompt') or ""
            gt_value = row.get('GT') or row.get('answer') or ""
            try:
                response = get_single_prediction(
                    prompt,
                    self.model_name,
                    task=self.task,
                    cleanup_after=cleanup,
                    use_web_search=use_web_search
                )
                prediction = self.postprocessor.extract_answer(self.task, response) or 'X'
            except Exception as e:
                response, prediction = 'Error', 'X'
                print(f"Error processing id {idx}: {e}")
            return {
                "id": idx,
                "prompt": prompt,
                "response": response,
                "prediction": prediction,
                "answer": gt_value
            }

        # --- Process and append safely ---
        os.makedirs(os.path.dirname(self.response_file), exist_ok=True)
        with open(self.response_file, 'a', encoding='utf-8') as f:
            if batch and batch > 1:
                with ThreadPoolExecutor(max_workers=batch) as executor:
                    futures = [executor.submit(process_record, rec) for rec in remaining_records]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                        rec = future.result()
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
            else:
                for rec in tqdm(remaining_records, desc="Generating responses"):
                    result = process_record(rec)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

    def evaluate_athena_taa(self):
        scored_file = Path(self.model_folder) / f"{self.task}_{self.num_rows or 'all'}_v{self.version}_{self.display_model_name}_scored.jsonl"
        return self.eval.evaluate_file(self.task, Path(self.response_file), scored_file)
