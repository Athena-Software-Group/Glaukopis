import os
import pandas as pd
from benchmarks.base import Benchmark
from pipelines.data_loader import load_json_or_jsonl, save_responses
from pipelines.models import get_single_prediction, alias_to_safe_name
from pipelines.post_processing.cti import cti_postprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class CVE(Benchmark):
    """Benchmark class for CVE identification with thread-safe resume support."""

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.task = 'cve'
        self.version = version
        self.data_file = data_path if data_path else 'benchmark_data/cve/100_sample_CVEs.jsonl'
        self.display_model_name = alias_to_safe_name(model_name)
        rows_str = str(num_rows) if num_rows else "all"

        self.model_folder = os.path.join("responses", self.display_model_name, self.task)
        os.makedirs(self.model_folder, exist_ok=True)

        self.response_file = os.path.join(
            self.model_folder, f'{self.task}_{rows_str}_v{self.version}_{self.display_model_name}_response.tsv'
        )

        self.postprocessor = cti_postprocessing()
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def load_samples(self):
        """Load CVE samples from JSONL."""
        raw_samples = load_json_or_jsonl(self.data_file, num_rows=self.num_rows)
        samples = []
        for data in raw_samples:
            cve_id = data['cve']['id']
            description = next((d['value'] for d in data['cve']['descriptions'] if d['lang']=='en'), None)
            if description:
                samples.append({'cve_id': cve_id, 'description': description})
        return samples

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        """Generate CVE predictions with safe resume and thread pooling."""
        samples = self.load_samples()

        # --- Resume logic using row index ---
        processed_ids = set()
        if os.path.exists(self.response_file):
            existing_data = pd.read_csv(self.response_file, sep='\t')
            processed_ids = set(existing_data.index.tolist())
        else:
            df = pd.DataFrame(columns=['description', 'ground_truth', 'predicted_cve'])
            save_responses(df, self.response_file, append=False)

        remaining_samples = [(idx, s) for idx, s in enumerate(samples) if idx not in processed_ids]

        if not remaining_samples:
            print("All samples already processed. Nothing to do.")
            return

        print(f"Processing {len(remaining_samples)} remaining samples...")

        def process_sample(idx_sample):
            idx, sample = idx_sample
            try:
                prompt = (
                    f"You are an expert in CVE classification. "
                    f"Which CVE does the following description map to: {sample['description']}"
                )
                response_text = get_single_prediction(
                    prompt,
                    self.model_name,
                    task='cve',
                    cleanup_after=cleanup,
                    use_web_search=use_web_search
                )
                predicted_cve, is_valid = self.postprocessor.extract_cve_id(response_text)
                if not is_valid:
                    predicted_cve = "NOT_FOUND"
            except Exception as e:
                print(f"Error at row {idx+1}: {e}")
                response_text = ""
                predicted_cve = "NOT_FOUND"

            return {
                'description': sample['description'],
                'ground_truth': sample['cve_id'],
                'predicted_cve': predicted_cve
            }

        # --- Thread-safe append ---
        if batch and batch > 1:
            with ThreadPoolExecutor(max_workers=batch) as executor:
                futures = {executor.submit(process_sample, r): r for r in remaining_samples}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                    result = future.result()
                    save_responses(pd.DataFrame([result]), self.response_file, append=True)
        else:
            for idx_sample in tqdm(remaining_samples, desc="Generating responses"):
                result = process_sample(idx_sample)
                save_responses(pd.DataFrame([result]), self.response_file, append=True)

        final_results = pd.read_csv(self.response_file).to_dict(orient='records')
        return final_results

    def evaluate_cve(self):
        """Compute CVE benchmark accuracy using centralized evaluation function."""
        # Ensure responses are generated
        self.generate_responses()
        accuracy = self.eval.compute_cve_accuracy(self.response_file)
        return accuracy
