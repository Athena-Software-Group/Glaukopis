import os
import pandas as pd
from benchmarks.base import Benchmark
from pipelines.post_processing.cti import cti_postprocessing
from pipelines.data_loader import load_data
from pipelines.models import get_single_prediction, get_cached_model, model_mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class CTIMCQ(Benchmark):
    """Benchmark class for CTI Multiple Choice Questions (MCQ) with thread-safe resume and threaded writing."""

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.task = 'mcq'
        self.version = version
        self.data_file = data_path if data_path else 'benchmark_data/cti_bench/cti-mcq.tsv'
        self.cli_model_name = model_name
        self.display_model_name = model_mapping.get(model_name, model_name).replace('/', '_')
        rows_str = str(num_rows) if num_rows is not None else "all"
        self.model_folder = os.path.join("responses", self.display_model_name, self.task)
        os.makedirs(self.model_folder, exist_ok=True)
        self.response_file = os.path.join(
            self.model_folder, f'{self.task}_{rows_str}_v{self.version}_{self.display_model_name}_response.tsv'
        )
        self.postprocessor = cti_postprocessing()
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        """Generate and save MCQ responses safely with threads and resume."""
        data = load_data(self.data_file)

        # Ensure GT column exists
        if 'Ground Truth' not in data.columns and 'GT' not in data.columns:
            raise ValueError(f"Ground Truth column not found. Found columns: {list(data.columns)}")
        if 'GT' in data.columns:
            data = data.rename(columns={'GT': 'Ground Truth'})

        if self.num_rows is not None:
            data = data.iloc[:self.num_rows]

        # --- Resume logic using row ids ---
        processed_ids = set()
        if os.path.exists(self.response_file):
            existing_data = pd.read_csv(self.response_file, sep='\t')
            if 'id' in existing_data.columns:
                processed_ids = set(existing_data['id'].tolist())
        else:
            # Create header row only (not pre-populated IDs)
            header_df = pd.DataFrame(columns=['id', 'Ground Truth', 'Prompt', 'Raw_Response', self.cli_model_name])
            header_df.to_csv(self.response_file, sep='\t', index=False)

        # Prepare rows to process
        remaining_rows = [(idx, row) for idx, row in data.iterrows() if idx not in processed_ids]

        if not remaining_rows:
            print("All rows already processed. Nothing to do.")
            return

        print(f"Processing {len(remaining_rows)} remaining rows...")

        # Pre-load model once before processing to catch loading errors early
        # and ensure the model is cached for all subsequent calls
        print("Pre-loading model to ensure it's cached...")
        get_cached_model(self.model_name)
        print("Model ready for inference.")

        def process_row(idx_row):
            idx, row = idx_row
            prompt = row['Prompt']
            gt_value = row['Ground Truth']
            try:
                raw_response = get_single_prediction(prompt, self.model_name, cleanup_after=cleanup)
                formatted_pred = self.postprocessor.format_mcq(raw_response)
                prediction = formatted_pred if formatted_pred in ['A', 'B', 'C', 'D'] else 'A'
                print(f"Row {idx+1}: {prediction}")
            except Exception as e:
                raw_response = "Error: " + str(e)
                prediction = 'A'
                print(f"Row {idx+1}: Error - {e}")
            return idx, gt_value, prompt, raw_response, prediction

        # --- Thread-safe append using as_completed ---
        with open(self.response_file, 'a', encoding='utf-8') as f:
            if batch and batch > 1:
                with ThreadPoolExecutor(max_workers=batch) as executor:
                    futures = [executor.submit(process_row, r) for r in remaining_rows]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                        idx, gt_value, prompt, raw_response, prediction = future.result()
                        raw_escaped = raw_response.replace('\n', '\\n').replace('\t', '\\t')
                        line = f"{idx}\t{gt_value}\t{prompt}\t{raw_escaped}\t{prediction}\n"
                        f.write(line)
                        f.flush()
            else:
                # Sequential fallback
                for idx_row in tqdm(remaining_rows, desc="Generating responses"):
                    idx, gt_value, prompt, raw_response, prediction = process_row(idx_row)
                    raw_escaped = raw_response.replace('\n', '\\n').replace('\t', '\\t')
                    line = f"{idx}\t{gt_value}\t{prompt}\t{raw_escaped}\t{prediction}\n"
                    f.write(line)
                    f.flush()

    def compute_mcq_accuracy(self):
        """Compute MCQ accuracy for the generated responses."""
        return self.eval.compute_mcq_accuracy(self.response_file, self.cli_model_name)
