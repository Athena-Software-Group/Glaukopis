import os
import pandas as pd
from benchmarks.base import Benchmark
from pipelines.data_loader import load_csv, save_responses
from pipelines.models import get_single_prediction, alias_to_safe_name
from concurrent.futures import ThreadPoolExecutor, as_completed # Added as_completed
from pathlib import Path
#from pipelines.evaluation.urlhaus_acc_f1 import compute_accuracy_f1
from tqdm import tqdm
import datetime

class URLHAUS(Benchmark):
    """Benchmark for URL IOC task with resume and thread safety"""

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.task = 'urlhaus'
        self.version = version
        self.data_file = data_path if data_path else 'benchmark_data/urlhaus/urls_benchmark2.csv'
        self.display_model_name = alias_to_safe_name(model_name)
        rows_str = str(num_rows) if num_rows else "all"
        
        # --- Create model-specific folder ---
        self.model_folder = os.path.join("responses", self.display_model_name)
        os.makedirs(self.model_folder, exist_ok=True)
        
        today = datetime.datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
        self.response_file = os.path.join(self.model_folder, f'{self.task}_{rows_str}_v{self.version}_{self.display_model_name}_response.csv')
        
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        data = load_csv(self.data_file)

        # Shuffle dataset deterministically and reset index to ensure a fresh, consistent index (0 to N-1)
        data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

        if self.num_rows:
            data = data.iloc[:self.num_rows]

        # --- Resume Setup: Use index-based filtering for thread safety ---
        processed_ids = set()
        if os.path.exists(self.response_file):
            try:
                # Read existing data, treating the first column (original index) as the index
                existing = pd.read_csv(self.response_file, index_col=0)
                processed_ids = set(existing.index.tolist())
                print(f"Resuming. Found {len(processed_ids)} completed rows.")
            except Exception as e:
                print(f"Warning: Could not read existing response file ({e}). Starting fresh.")
                processed_ids = set()

        # Create header if file is new or empty
        if not os.path.exists(self.response_file) or len(processed_ids) == 0:
            # Create the file header, setting 'idx' as the index label
            pd.DataFrame(columns=['url', 'reference', 'model_prediction']).to_csv(
                self.response_file, index=True, index_label='idx'
            )
            processed_ids = set()

        remaining_records_indexed = [
            (idx, row['url'], row['reference'])
            for idx, row in data.iterrows()
            if idx not in processed_ids
        ]

        if not remaining_records_indexed:
            print("All rows already processed. Skipping.")
            return pd.read_csv(self.response_file, index_col=0).to_dict(orient="records")

        print(f"Processing {len(remaining_records_indexed)} remaining rows...")

        def process_record(idx_url_ref):
            """Processes a single example and returns its data dictionary including the original index."""
            idx, url, reference = idx_url_ref # idx is the global/shuffled index
            prompt = (
                "Classify the following URL as either 'benign' or 'malicious'. "
                "Do not provide any explanation or extra text, return only one word.\n"
                f"URL: {url}"
            )
            try:
                pred = get_single_prediction(prompt, self.model_name, task=self.task, cleanup_after=cleanup)
                prediction = pred.strip()
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                prediction = "ERROR_MALICIOUS" # Fallback prediction on error

            return {
                'idx': idx, # CRUCIAL: Include the index
                'url': url,
                'reference': reference,
                'model_prediction': prediction
            }

        Path(self.response_file).touch()

        # --- Thread-safe append (out-of-order execution) ---
        with open(self.response_file, 'a', encoding='utf-8') as f:
            if batch and batch > 1:
                # Thread pool for concurrency
                with ThreadPoolExecutor(max_workers=batch) as executor:
                    futures = [executor.submit(process_record, r) 
                            for r in remaining_records_indexed]
                    
                    # Use as_completed to process results as they finish (out of order)
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating URLhaus responses"):
                        row = future.result()
                        # Use set_index('idx') to write the row with its index, without header
                        pd.DataFrame([row]).set_index('idx').to_csv(f, mode='a', header=False)
                        f.flush() # Ensure data is written to disk immediately
            else:
                # Sequential fallback
                for rec in tqdm(remaining_records_indexed, desc="Generating URLhaus responses"):
                    row = process_record(rec)
                    pd.DataFrame([row]).set_index('idx').to_csv(f, mode='a', header=False)
                    f.flush() # Ensure data is written to disk immediately

        # Final read for memory storage and return
        final_results = pd.read_csv(self.response_file, index_col=0).to_dict(orient="records")
        print(f"Responses saved to {self.response_file}")
        return final_results


    def compute_accuracy(self):
        # Ensure responses are generated before evaluation
        self.generate_responses()
        return self.eval.urlhaus_acc_f1(self.response_file)
