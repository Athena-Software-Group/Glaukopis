import os
import re
import pandas as pd
from datasets import load_dataset
from benchmarks.base import Benchmark
from pipelines.models import get_single_prediction
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


_CHOICE_LETTERS = "ABCDEFGHIJ"


class MMLUPRO(Benchmark):
    """Benchmark class for MMLU-Pro (TIGER-Lab/MMLU-Pro) zero-shot CoT evaluation.

    Mirrors the resume-by-idx + ThreadPoolExecutor pattern used by MMLU
    (benchmarks/mmlu.py). Prompt format and three-tier answer extraction
    follow the upstream evaluate_from_api.py / compute_accuracy.py from
    https://github.com/TIGER-AI-Lab/MMLU-Pro so per-model scores stay
    comparable to the published leaderboard.
    """

    def __init__(self, model_name, num_rows=None, data_path=None, version=1):
        super().__init__(model_name, num_rows)
        self.task = "mmlu-pro"
        self.version = version
        # data_path retained for CLI signature parity with other benchmarks
        # (inference.py passes args.data_path positionally); MMLU-Pro is
        # always loaded from the HF hub so the value is unused.
        self.data_path = data_path
        # Index the response cache by the alias verbatim (not the HF repo id)
        # so different aliases pointing to the same HF repo get separate
        # cache dirs. Required for the matched-conditions A/B on models like
        # Qwen3-30B-A3B-Thinking-2507 where '...-vllm' (thinking-on, 8192
        # floor, optionally --reasoning-parser qwen3) and '...-no-think-vllm'
        # (enable_thinking=False, per-task caps) produce meaningfully
        # different MMLU-Pro scores despite resolving to the same HF repo;
        # the previous mapping.get(...).replace("/", "_") convention would
        # collide them on resume and silently re-score the older run's CSV.
        # Other benchmarks (athena*/cybermetric/cybersoceval) keep the
        # HF-repo-id convention; only MMLU-Pro pays the disambiguation cost
        # because it's currently the only suite where the two inference
        # paths produce a meaningful score divergence in practice. The
        # matching --mode overwrite path lives in run_benchmark.sh's
        # resolve_resp_file (mmlu-pro case echoes the SAFE_NAME-keyed path).
        self.display_model_name = model_name.replace("/", "_")
        rows_str = str(num_rows) if num_rows else "all"

        self.model_folder = os.path.join("responses", self.display_model_name, self.task)
        os.makedirs(self.model_folder, exist_ok=True)
        self.response_file = os.path.join(
            self.model_folder,
            f"{self.task}_{rows_str}_v{self.version}_{self.display_model_name}_response.csv",
        )
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def load_samples(self):
        """Load MMLU-Pro test split from HuggingFace."""
        print("Loading MMLU-Pro dataset (test split)...")
        data = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

        if self.num_rows:
            data = data.select(range(self.num_rows))

        samples = []
        for ex in data:
            options = [opt for opt in ex["options"] if opt != "N/A"]
            opt_lines = "\n".join(f"{_CHOICE_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
            prompt = (
                f"The following is a multiple choice question (with answers) about {ex['category']}. "
                "Think step by step and then output the answer in the format of "
                "\"The answer is (X)\" at the end.\n\n"
                f"Question: {ex['question']}\n"
                f"Options:\n{opt_lines}\n"
                "Answer: Let's think step by step."
            )
            samples.append({
                "question_id": ex["question_id"],
                "question": ex["question"],
                "category": ex["category"],
                "prompt": prompt,
                "ground_truth": ex["answer"],
            })
        return samples

    def generate_responses(self, cleanup=False, use_web_search=False, batch=None):
        samples = self.load_samples()

        processed_ids = set()
        if os.path.exists(self.response_file):
            try:
                existing_df = pd.read_csv(self.response_file, index_col=0)
                processed_ids = set(existing_df.index.tolist())
                print(f"Resuming. Found {len(processed_ids)} completed rows.")
            except Exception as e:
                print(f"Warning: Could not read existing response file ({e}). Starting fresh.")
                processed_ids = set()

        if not os.path.exists(self.response_file) or len(processed_ids) == 0:
            pd.DataFrame(columns=["question_id", "category", "question",
                                  "ground_truth", "prediction", "raw_response"])\
                .to_csv(self.response_file, index=True, index_label="idx")
            processed_ids = set()

        remaining = [(idx, s) for idx, s in enumerate(samples) if idx not in processed_ids]
        if not remaining:
            print("All rows already processed. Skipping.")
            return pd.read_csv(self.response_file, index_col=0).to_dict(orient="records")

        print(f"Processing {len(remaining)} remaining rows...")

        def process_sample(idx_sample):
            idx, sample = idx_sample
            raw_response = ""
            try:
                raw_response = get_single_prediction(
                    sample["prompt"],
                    self.model_name,
                    task=self.task,
                    cleanup_after=cleanup,
                    use_web_search=use_web_search,
                )
                predicted = self._extract_answer(raw_response)
            except Exception as e:
                print(f"Error on original index {idx}: {e}")
                predicted = "NOT_FOUND"
            return {
                "idx": idx,
                "question_id": sample["question_id"],
                "category": sample["category"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "prediction": predicted if predicted else "NOT_FOUND",
                "raw_response": raw_response,
            }

        with open(self.response_file, "a", encoding="utf-8") as f:
            if batch and batch > 1:
                with ThreadPoolExecutor(max_workers=batch) as executor:
                    futures = [executor.submit(process_sample, r) for r in remaining]
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc="Generating MMLU-Pro responses"):
                        res = future.result()
                        pd.DataFrame([res]).set_index("idx").to_csv(f, mode="a", header=False)
                        f.flush()
            else:
                for rec in tqdm(remaining, desc="Generating MMLU-Pro responses"):
                    res = process_sample(rec)
                    pd.DataFrame([res]).set_index("idx").to_csv(f, mode="a", header=False)
                    f.flush()

        return pd.read_csv(self.response_file, index_col=0).to_dict(orient="records")

    def evaluate_mmlu_pro(self):
        return self.eval.compute_mmlu_pro_accuracy(self.response_file)

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Three-tier extraction matching upstream MMLU-Pro compute_accuracy.py."""
        if text is None:
            return "NOT_FOUND"
        s = str(text).replace("**", "")
        m = re.search(r"answer is \(?([A-J])\)?", s)
        if m:
            return m.group(1)
        m = re.search(r".*[aA]nswer:\s*\(?([A-J])\)?", s)
        if m:
            return m.group(1)
        m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", s, re.DOTALL)
        return m.group(0) if m else "NOT_FOUND"
