import os
import re
import pandas as pd
from datasets import load_dataset
from benchmarks.base import Benchmark
from pipelines.data_loader import save_responses
from pipelines.models import get_single_prediction, model_mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
#from pipelines.evaluation.mmlu_acc import compute_mmlu_accuracy
from tqdm import tqdm


class MMLU(Benchmark):
    """Benchmark class for MMLU multiple-choice evaluation with resume support."""

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.task = "mmlu"
        self.display_model_name = model_mapping.get(model_name, model_name).replace("/", "_")
        rows_str = str(num_rows) if num_rows else "all"

        # Save responses under responses/<model_name>/mmlu
        self.model_folder = os.path.join("responses", self.display_model_name, "mmlu")
        os.makedirs(self.model_folder, exist_ok=True)
        self.response_file = os.path.join(
            self.model_folder, f"{self.task}_{rows_str}_{self.display_model_name}_response.csv"
        )
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def load_samples(self):
        """Load MMLU test set from HuggingFace directly."""
        print("Loading MMLU dataset (test split)...")
        data = load_dataset("cais/mmlu", "all", split="test")

        if self.num_rows:
            data = data.select(range(self.num_rows))

        samples = []
        for ex in data:
            q = ex["question"]
            choices = ex["choices"]  # list of 4 strings
            ref_idx = ex["answer"]   # integer (0–3)
            ref_letter = ["A", "B", "C", "D"][ref_idx]

            prompt = (
                f"{q}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n\n"
                "Reply with just the letter (A, B, C, or D) for the correct answer."
            )

            samples.append({
                "question": q,
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
                "prompt": prompt,
                "ground_truth": ref_letter,
            })
        return samples

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        """
        Generate predictions and save them incrementally with resume support + optional thread pooling.
        Uses original dataset index ('idx') for robust resume logic.
        """
        samples = self.load_samples()

        # --- Resume Setup ---
        processed_ids = set()

        if os.path.exists(self.response_file):
            try:
                # Read existing data, treating the first column (original index) as the index
                existing_df = pd.read_csv(self.response_file, index_col=0)
                processed_ids = set(existing_df.index.tolist())
                print(f"Resuming. Found {len(processed_ids)} completed rows.")
            except Exception as e:
                # If reading fails, start fresh
                print(f"Warning: Could not read existing response file ({e}). Starting fresh.")
                processed_ids = set()

        # Create header if file is new or empty
        if not os.path.exists(self.response_file) or len(processed_ids) == 0:
            # Create the file header, setting 'idx' as the index label
            pd.DataFrame(columns=["question", "ground_truth", "prediction"])\
                .to_csv(self.response_file, index=True, index_label='idx')
            processed_ids = set() # Reset in case of fresh start due to error

        # Filter samples based on processed IDs (original indices)
        remaining_samples_indexed = [
            (idx, sample) for idx, sample in enumerate(samples) if idx not in processed_ids
        ]

        if not remaining_samples_indexed:
            print("All rows already processed. Skipping.")
            return pd.read_csv(self.response_file, index_col=0).to_dict(orient="records")

        print(f"Processing {len(remaining_samples_indexed)} remaining rows...")

        def process_sample(idx_sample):
            """Processes a single example and returns its data dictionary including the original index."""
            idx, sample = idx_sample
            try:
                response_text = get_single_prediction(
                    sample["prompt"],
                    self.model_name,
                    task="mmlu",
                    cleanup_after=cleanup,
                    use_web_search=use_web_search,
                )
                predicted = self._parse_choice(response_text)
            except Exception as e:
                print(f"Error on original index {idx}: {e}")
                predicted = "NOT_FOUND"

            return {
                "idx": idx, # CRUCIAL: Original dataset index for resume safety
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "prediction": predicted,
            }
        
        with open(self.response_file, 'a', encoding='utf-8') as f:
            if batch and batch > 1:
                with ThreadPoolExecutor(max_workers=batch) as executor:
                    futures = [
                        executor.submit(process_sample, r)
                        for r in remaining_samples_indexed
                    ]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating MMLU responses"):
                        res = future.result()
                        pd.DataFrame([res]).set_index('idx').to_csv(f, mode='a', header=False)
                        f.flush()
            else:
                # Sequential fallback
                for rec in tqdm(remaining_samples_indexed, desc="Generating MMLU responses"):
                    res = process_sample(rec)
                    pd.DataFrame([res]).set_index('idx').to_csv(f, mode='a', header=False)
                    f.flush()

        # Final read for memory storage
        final_results = pd.read_csv(self.response_file, index_col=0).to_dict(orient="records")
        return final_results



    def evaluate_mmlu(self):
        """Compute MMLU benchmark accuracy using centralized evaluation function."""
        self.generate_responses()
        accuracy = self.eval.compute_mmlu_accuracy(self.response_file)
        return accuracy

    @staticmethod
    def _parse_choice(text: str) -> str:
        """Extract first A/B/C/D from model output."""
        match = re.search(r"[ABCD]", str(text).upper())
        return match.group(0) if match else "NOT_FOUND"