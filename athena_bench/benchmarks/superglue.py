import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from benchmarks.base import Benchmark
from pipelines.models import get_single_prediction, model_mapping
from concurrent.futures import ThreadPoolExecutor, as_completed # Import as_completed
from pathlib import Path
#from pipelines.evaluation.superglue_metrics import evaluate_superglue_csv  # your custom evaluatio
from tqdm import tqdm

class SUPERGLUE(Benchmark):
    """Benchmark class for all SuperGLUE tasks using LLM models."""

    TASKS = {
        "boolq": {
            "hf": ("super_glue", "boolq"),
            "split": "validation",
            "prompt": lambda ex: f"Is the answer to the question true or false based on the passage? Return only: true or false.\nQuestion: {ex['question']}\nPassage: {ex['passage']}",
            "parse": lambda output: 1 if output.lower() == "true" else 0,
            "evaluate_metric": ("accuracy",)
        },
        "cb": {
            "hf": ("super_glue", "cb"),
            "split": "validation",
            "prompt": lambda ex: f"What is the relationship between the premise and hypothesis? Return only: entailment, contradiction, or neutral.\nPremise: {ex['premise']}\nHypothesis: {ex['hypothesis']}",
            "parse": lambda output: 0 if output.lower() == "entailment" else 1 if output.lower() == "contradiction" else 2,
            "evaluate_metric": ("super_glue", "cb")
        },
        "copa": {
            "hf": ("super_glue", "copa"),
            "split": "validation",
            "prompt": lambda ex: f"Given the premise and question, which choice is correct? Return only: choice1 or choice2.\nPremise: {ex['premise']}\nQuestion: {ex['question']}\nChoice1: {ex['choice1']}\nChoice2: {ex['choice2']}",
            "parse": lambda output: 0 if output.lower() == "choice1" else 1,
            "evaluate_metric": ("accuracy",)
        },
        "record": {
            "hf": ("super_glue", "record"),
            "split": "validation",
            "prompt": lambda ex: f"Fill in the blank in the query with the correct entity based on the passage. Return only the entity name.\nPassage: {ex['passage']}\nQuery: {ex['query']}",
            # NOTE: For ReCoRD, ex["idx"] is the internal ID, not the HuggingFace dataset index.
            # We use ex["idx"] here for parsing/evaluation, but hf_idx for resume.
            "parse": lambda output, internal_idx: {"idx": internal_idx, "prediction_text": output.strip()},
            "evaluate_metric": ("super_glue", "record")
        },
        "rte": {
            "hf": ("super_glue", "rte"),
            "split": "validation",
            "prompt": lambda ex: f"Does the premise entail the hypothesis? Return only: entailment or not_entailment.\nPremise: {ex['premise']}\nHypothesis: {ex['hypothesis']}",
            "parse": lambda output: 0 if output.lower() == "entailment" else 1,
            "evaluate_metric": ("accuracy",)
        },
        "wic": {
            "hf": ("super_glue", "wic"),
            "split": "validation",
            "prompt": lambda ex: f"Does the word '{ex['word']}' have the same meaning in both sentences? Return only: true or false.\nSentence1: {ex['sentence1']}\nSentence2: {ex['sentence2']}",
            "parse": lambda output: 1 if output.lower() == "true" else 0,
            "evaluate_metric": ("accuracy",)
        },
        "wsc": {
            "hf": ("super_glue", "wsc"),
            "split": "validation",
            "prompt": lambda ex: f"Does '{ex['span2_text']}' refer to '{ex['span1_text']}' in the sentence? Return only: true or false.\nSentence Triad: {ex['text']}",
            "parse": lambda output: 1 if output.lower() == "true" else 0,
            "evaluate_metric": ("accuracy",)
        },
        "wsc.fixed": {
            "hf": ("super_glue", "wsc.fixed"),
            "split": "validation",
            "prompt": lambda ex: f"Does '{ex['span2_text']}' refer to '{ex['span1_text']}' in the sentence? Return only: true or false.\nSentence: {ex['text']}",
            "parse": lambda output: 1 if output.lower() == "true" else 0,
            "evaluate_metric": ("accuracy",)
        }
    }

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.tasks = self.TASKS
        self.display_model_name = model_mapping.get(model_name, model_name).replace("/", "_")

        # Main model folder
        self.model_folder = os.path.join("responses", self.display_model_name)
        os.makedirs(self.model_folder, exist_ok=True)

        # Dedicated superglue subfolder
        self.superglue_folder = os.path.join(self.model_folder, "superglue")
        os.makedirs(self.superglue_folder, exist_ok=True)

        self.all_results = {}
        self.response_file = {}
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        """Generate and save predictions for all SuperGLUE tasks with resume logic + optional thread pooling."""
        for task_name, task_config in self.tasks.items():
            print(f"\nProcessing task: {task_name}")
            
            response_file = os.path.join(
                self.superglue_folder,
                f"superglue_{task_name}_{self.num_rows or 'all'}_{self.display_model_name}_response.csv"
            )

            self.response_file[task_name] = response_file

            dataset = load_dataset(*task_config["hf"], split=task_config["split"])
            if self.num_rows:
                dataset = dataset.select(range(self.num_rows))

            # --- Resume Setup: Use index-based filtering for thread safety ---
            processed_ids = set()
            if os.path.exists(response_file):
                try:
                    # Read existing data, treating the first column (original index) as the index
                    existing_df = pd.read_csv(response_file, index_col=0)
                    processed_ids = set(existing_df.index.tolist())
                    print(f"Resuming. Found {len(processed_ids)} completed rows.")
                except Exception as e:
                    print(f"Warning: Could not read existing response file ({e}). Starting fresh.")
                    processed_ids = set()

            # Create header if file is new or empty
            if not os.path.exists(response_file) or len(processed_ids) == 0:
                # Create the file header, setting 'idx' as the index label
                pd.DataFrame(columns=["input_1", "input_2", "raw_response", "prediction", "reference"])\
                    .to_csv(response_file, index=True, index_label='idx')
                processed_ids = set()

            # Filter samples based on processed IDs (original HuggingFace dataset indices)
            remaining_records_indexed = [
                (hf_idx, ex) for hf_idx, ex in enumerate(dataset) if hf_idx not in processed_ids
            ]

            if not remaining_records_indexed:
                print("All rows already processed. Skipping.")
                self.all_results[task_name] = pd.read_csv(response_file, index_col=0)
                continue

            print(f"Processing {len(remaining_records_indexed)} remaining rows...")

            def process_record(hf_idx_ex):
                """Processes a single example and returns its data dictionary including the original index."""
                hf_idx, ex = hf_idx_ex # hf_idx is the original dataset index
                
                # Default values in case of error
                raw_output = "error"
                pred = 0
                reference = ex.get("label", None)
                in1, in2 = "", ""

                try:
                    raw_output = get_single_prediction(
                        task_config["prompt"](ex),
                        self.model_name,
                        task="superglue",
                        cleanup_after=cleanup
                    )
                    
                    if task_name == "record":
                        # Record requires the internal 'idx' for parsing
                        pred = task_config["parse"](raw_output, ex["idx"])
                        reference = {"idx": ex["idx"], "answers": ex["answers"]}
                        in1, in2 = ex["passage"], ex["query"]
                    else:
                        pred = task_config["parse"](raw_output)
                        reference = ex["label"]
                        if task_name == "boolq":
                            in1, in2 = ex["question"], ex["passage"]
                        elif task_name in ["cb", "rte"]:
                            in1, in2 = ex["premise"], ex["hypothesis"]
                        elif task_name == "copa":
                            in1, in2 = ex["premise"], f"{ex['question']} | Choice1: {ex['choice1']} | Choice2: {ex['choice2']}"
                        elif task_name == "wic":
                            in1, in2 = ex["sentence1"], ex["sentence2"]
                        elif task_name in ["wsc", "wsc.fixed"]:
                            in1, in2 = ex["text"], f"{ex['span1_text']} | {ex['span2_text']}"
                except Exception as e:
                    # Only print specific error if not a standard task name logic failure
                    print(f"Error processing record at index {hf_idx} for task {task_name}: {e}")
                    if task_name == "record":
                        pred = {"idx": ex.get("idx", 0), "prediction_text": ""}
                        reference = {"idx": ex.get("idx", 0), "answers": ex.get("answers", [])}
                    else:
                        pred = 0


                return {
                    "idx": hf_idx, # CRUCIAL: Original dataset index for resume safety
                    "input_1": in1,
                    "input_2": in2,
                    "raw_response": raw_output,
                    "prediction": pred,
                    "reference": reference
                }

            # --- Thread-safe append (out-of-order execution) ---
            with open(response_file, 'a', encoding='utf-8') as f:
                if batch and batch > 1:
                    # Thread pool for concurrency
                    with ThreadPoolExecutor(max_workers=batch) as executor:
                        futures = [executor.submit(process_record, r) 
                                for r in remaining_records_indexed]
                        
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {task_name} responses"):
                            row = future.result()
                            # Use set_index('idx') to write the results using the global index
                            pd.DataFrame([row]).set_index('idx').to_csv(f, mode='a', header=False)
                            f.flush()
                else:
                    # Sequential fallback
                    for rec in tqdm(remaining_records_indexed, desc=f"Generating {task_name} responses"):
                        row = process_record(rec)
                        pd.DataFrame([row]).set_index('idx').to_csv(f, mode='a', header=False)
                        f.flush() # Ensure data is written to disk immediately

            # Final read for memory storage
            self.all_results[task_name] = pd.read_csv(response_file, index_col=0)
            print(f"Responses for task '{task_name}' saved to: {response_file}")


    def evaluate_superglue(self):
        """Evaluate all generated responses using automatic subtask detection from filenames."""
        self.metrics = {}
        for task_name, csv_file in self.response_file.items():
            result = self.eval.evaluate_superglue_csv(csv_file)
            subtask = result["subtask"]
            metrics = result["metrics"]
            self.metrics[subtask] = metrics

        return self.metrics
