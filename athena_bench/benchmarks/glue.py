import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
from benchmarks.base import Benchmark
from pipelines.models import get_single_prediction, model_mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

#from pipelines.evaluation.glue_metrics import evaluate_glue_csv
from tqdm import tqdm

class GLUE(Benchmark):
    """Benchmark class for all GLUE tasks using LLM models."""

    TASKS = {
        "cola": {
            "hf": ("glue", "cola"),
            "split": "validation",
            "prompt": lambda ex: f"Is this sentence grammatically acceptable? Return only the answer: acceptable or unacceptable.\n\"{ex['sentence']}\"",
            "parse": lambda output: 1 if output.lower() == "acceptable" else 0,
            "evaluate_metric": ("glue", "cola")
        },
        "sst2": {
            "hf": ("glue", "sst2"),
            "split": "validation",
            "prompt": lambda ex: f"Classify sentiment as negative or positive. Return only the answer: negative or positive.\n\"{ex['sentence']}\"",
            "parse": lambda output: 1 if output.lower() == "positive" else 0,
            "evaluate_metric": ("glue", "sst2")
        },
        "mrpc": {
            "hf": ("glue", "mrpc"),
            "split": "validation",
            "prompt": lambda ex: f"Are these two sentences paraphrases? Return only the answer: yes or no.\nSentence1: \"{ex['sentence1']}\"\nSentence2: \"{ex['sentence2']}\"",
            "parse": lambda output: 1 if output.lower() == "yes" else 0,
            "evaluate_metric": ("glue", "mrpc")
        },
        "qqp": {
            "hf": ("glue", "qqp"),
            "split": "validation",
            "prompt": lambda ex: f"Are these two questions duplicates? Return only the answer: yes or no.\nQuestion1: \"{ex['question1']}\"\nQuestion2: \"{ex['question2']}\"",
            "parse": lambda output: 1 if output.lower() == "yes" else 0,
            "evaluate_metric": ("glue", "qqp")
        },
        "mnli_matched": {
            "hf": ("glue", "mnli"),
            "split": "validation_matched",
            "prompt": lambda ex: f"Given the premise and hypothesis, what is their relationship? Return only the answer: entailment, neutral, or contradiction.\nPremise: \"{ex['premise']}\"\nHypothesis: \"{ex['hypothesis']}\"",
            "parse": lambda output: 0 if output.lower() == "entailment" else 1 if output.lower() == "neutral" else 2,
            "evaluate_metric": ("glue", "mnli_matched")
        },
        "mnli_mismatched": {
            "hf": ("glue", "mnli"),
            "split": "validation_mismatched",
            "prompt": lambda ex: f"Given the premise and hypothesis, what is their relationship? Return only the answer: entailment, neutral, or contradiction.\nPremise: \"{ex['premise']}\"\nHypothesis: \"{ex['hypothesis']}\"",
            "parse": lambda output: 0 if output.lower() == "entailment" else 1 if output.lower() == "neutral" else 2,
            "evaluate_metric": ("glue", "mnli_mismatched")
        },
        "qnli": {
            "hf": ("glue", "qnli"),
            "split": "validation",
            "prompt": lambda ex: f"Does the sentence contain the answer to the question? Return only the answer: entailment or not_entailment.\nQuestion: \"{ex['question']}\"\nSentence: \"{ex['sentence']}\"",
            "parse": lambda output: 0 if output.lower() == "entailment" else 1,
            "evaluate_metric": ("glue", "qnli")
        },
        "rte": {
            "hf": ("glue", "rte"),
            "split": "validation",
            "prompt": lambda ex: f"Does sentence1 entail sentence2? Return only the answer: entailment or not_entailment.\nSentence1: \"{ex['sentence1']}\"\nSentence2: \"{ex['sentence2']}\"",
            "parse": lambda output: 0 if output.lower() == "entailment" else 1,
            "evaluate_metric": ("glue", "rte")
        },
        "wnli": {
            "hf": ("glue", "wnli"),
            "split": "validation",
            "prompt": lambda ex: f"Does sentence1 entail sentence2? Return only the answer: entailment or not_entailment.\nSentence1: \"{ex['sentence1']}\"\nSentence2: \"{ex['sentence2']}\"",
            "parse": lambda output: 1 if output.lower() == "entailment" else 0,
            "evaluate_metric": ("glue", "wnli")
        },
        "stsb": {
            "hf": ("glue", "stsb"),
            "split": "validation",
            "prompt": lambda ex: f"""Rate the semantic similarity between these sentences on a scale from 0 to 5, where 0 is completely dissimilar and 5 is identical. Return only the number.\nSentence1: \"{ex['sentence1']}\"\nSentence2: \"{ex['sentence2']}\"\nSimilarity:""",
            "parse": lambda output: float(output.strip()) if output.strip().replace(".", "", 1).isdigit() else 0.0,
            "evaluate_metric": ("glue", "stsb")
        }
    }

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.tasks = self.TASKS
        self.display_model_name = model_mapping.get(model_name, model_name).replace("/", "_")

        # Main model folder
        self.model_folder = os.path.join("responses", self.display_model_name)
        os.makedirs(self.model_folder, exist_ok=True)

        # Dedicated glue subfolder
        self.glue_folder = os.path.join(self.model_folder, "glue")
        os.makedirs(self.glue_folder, exist_ok=True)

        self.all_results = {}
        self.response_file = {}  # store file paths per task
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        """Generate and save predictions for all GLUE tasks with resume logic + thread pooling (optional)."""
        for task_name, task_config in self.tasks.items():
            print(f"\nProcessing task: {task_name}")
            from datetime import datetime
            response_file = os.path.join(
                self.glue_folder,
                f"glue_{task_name}_{self.num_rows or 'all'}_{self.display_model_name}_response.csv"
            )
            self.response_file[task_name] = response_file

            dataset = load_dataset(*task_config["hf"], split=task_config["split"])
            if self.num_rows:
                dataset = dataset.select(range(self.num_rows))
            processed_ids = set()
            if os.path.exists(response_file):
                try:
                    existing_df = pd.read_csv(response_file, sep=',', index_col=0)
                    processed_ids = set(existing_df.index.tolist())
                    print(f"Resuming. Found {len(processed_ids)} completed rows.")
                except Exception as e:
                    # If reading fails (e.g., corrupted file), start fresh or skip.
                    print(f"Warning: Could not read existing response file ({e}). Starting fresh.")
                    processed_ids = set()
            
            if not os.path.exists(response_file) or len(processed_ids) == 0:
                pd.DataFrame(columns=["input_1", "input_2", "raw_response", "prediction", "reference"])\
                    .to_csv(response_file, sep=',', index=True, index_label='idx')
                processed_ids = set()


            # Filter based on the true indices
            remaining_dataset = [(idx, ex) for idx, ex in enumerate(dataset) if idx not in processed_ids]
            
            if not remaining_dataset:
                print("All rows already processed. Skipping.")
                # CHANGE: Use sep=',' instead of sep='\t'
                self.all_results[task_name] = pd.read_csv(response_file, sep=',', index_col=0)
                continue

            print(f"Processing {len(remaining_dataset)} remaining rows...")

            def process_example(idx_ex):
                """Processes a single example and returns its data dictionary."""
                idx, ex = idx_ex # idx is the original dataset index
                try:
                    raw_output = get_single_prediction(
                        task_config["prompt"](ex),
                        self.model_name,
                        task="glue",
                        cleanup_after=cleanup
                    )
                    pred = task_config["parse"](raw_output)
                except Exception:
                    raw_output = "error"
                    pred = 0 if task_name != "stsb" else 0.0

                if task_name in ["cola", "sst2"]:
                    in1, in2 = ex["sentence"], ""
                elif task_name in ["mrpc", "rte", "wnli", "stsb"]:
                    in1, in2 = ex["sentence1"], ex["sentence2"]
                elif task_name == "qqp":
                    in1, in2 = ex["question1"], ex["question2"]
                elif task_name == "qnli":
                    in1, in2 = ex["question"], ex["sentence"]
                elif task_name in ["mnli_matched", "mnli_mismatched"]:
                    in1, in2 = ex["premise"], ex["hypothesis"]
                return {
                    "idx": idx, # Original dataset index
                    "input_1": in1,
                    "input_2": in2,
                    "raw_response": raw_output,
                    "prediction": pred,
                    "reference": ex["label"]
                }

            # --- Thread-safe append (out-of-order execution) ---
            
            if batch and batch > 1:
                with ThreadPoolExecutor(max_workers=batch) as executor:
                    futures = [executor.submit(process_example, r) for r in remaining_dataset]
                    
                    # Open file for appending (file handle 'f' used by main thread only)
                    with open(response_file, 'a', encoding='utf-8') as f:
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {task_name} responses"):
                            res = future.result()
                            pd.DataFrame([res]).set_index('idx').to_csv(f, sep=',', mode='a', header=False)
            else:
                # Sequential fallback
                with open(response_file, 'a', encoding='utf-8') as f:
                    for rec in tqdm(remaining_dataset, desc=f"Generating {task_name} responses"):
                        res = process_example(rec)
                        pd.DataFrame([res]).set_index('idx').to_csv(f, sep=',', mode='a', header=False)

            self.all_results[task_name] = pd.read_csv(response_file, sep=',', index_col=0)
            print(f"Responses for task '{task_name}' saved to: {response_file}")

    def evaluate_glue(self):
        """Evaluate all generated responses using automatic subtask detection from filenames."""
        self.metrics = {}

        for task_name, csv_file in self.response_file.items():
            result = self.eval.evaluate_glue_csv(csv_file)
            subtask = result["subtask"]
            metrics = result["metrics"]

            # Store metrics keyed by subtask
            self.metrics[subtask] = metrics

        return self.metrics
