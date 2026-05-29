import os
import json
import pandas as pd
from tqdm import tqdm
from benchmarks.base import Benchmark
from pipelines.models import get_single_prediction, alias_to_safe_name
from pipelines.post_processing.cti import cti_postprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

class CYBERMETRIC(Benchmark):
    """Benchmark class for CyberMetric MCQ evaluation."""

    def __init__(self, model_name, num_rows=None, data_path=None,version=1):
        super().__init__(model_name, num_rows)
        self.task = 'cybermetric'
        self.version = 1
        self.data_file = data_path if data_path else 'benchmark_data/cybermetricdataset/CyberMetric-80-v1.json'
        json_filename = os.path.splitext(os.path.basename(self.data_file))[0]  # strip path and extension
        self.display_model_name = alias_to_safe_name(model_name)
        rows_str = str(num_rows) if num_rows else "all"

        self.model_folder = os.path.join("responses", self.display_model_name, self.task)
        os.makedirs(self.model_folder, exist_ok=True)

        self.response_file = os.path.join(
        self.model_folder,
        f'{self.task}_{json_filename}_{rows_str}_v{self.version}_{self.display_model_name}_response.csv'
    )

        self.postprocessor = cti_postprocessing()
        from pipelines.evaluation.cti_eval import CTIEvaluate
        self.eval = CTIEvaluate()

    def load_questions(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('questions', [])

    def generate_responses(self, cleanup=False, use_web_search=False, batch=4):
        questions = self.load_questions()

        # Resume by matching completed question text rather than row count.
        # The batched path appends rows in completion order (as_completed),
        # so len(existing) does not correspond to "questions 0..N-1 done".
        if os.path.exists(self.response_file):
            existing = pd.read_csv(self.response_file)
            done = set(existing['question'].astype(str))
            remaining_questions = [q for q in questions if str(q['question']) not in done]
            print(f"Resuming: {len(done)} done, {len(remaining_questions)} remaining")
        else:
            pd.DataFrame(columns=['question', 'answer', 'raw_response', 'answer_after_post_processing'])\
            .to_csv(self.response_file, index=False, encoding='utf-8')
            remaining_questions = questions

        def process_question(idx_item):
            idx, item = idx_item
            answers = item.get('answers', {})
            options = ', '.join([f"{k}) {answers.get(k, '')}" for k in ['A', 'B', 'C', 'D'] if k in answers])

            prompt = (
                "You are a helpful AI assistant.\n"
                "Instructions:\n"
                "a. Carefully read the question.\n"
                "b. Choose the correct answer (A, B, C, or D) only.\n"
                "c. Do NOT include any explanation or additional text in the response.\n"
                f"Question: {item['question']}\n"
                f"Options: {options}\n\n"
                "Only output the line: ANSWER: <letter>"
            )

            raw_response = ''
            try:
                raw_response = get_single_prediction(
                    prompt,
                    self.model_name,
                    task=self.task,
                    cleanup_after=cleanup,
                    use_web_search=use_web_search
                )
            except Exception as e:
                print(f"Model call failed for question {idx+1}: {e}")

            try:
                answer, _ = self.postprocessor.format_cybermetric(raw_response)
            except Exception as e:
                print(f"Postprocessing failed for question {idx+1}: {e}")
                answer = 'X'

            return {
                'question': item['question'],
                'answer': item['solution'],
                'raw_response': raw_response,
                'answer_after_post_processing': answer
            }

        if batch and batch > 1:
            # as_completed + tqdm mirrors the pattern used by every other
            # benchmark in this dir (cti_mcq.py, athena_mcq.py, cti_taa.py).
            # Rows arrive out of input order but each row carries its own
            # question + GT so downstream scoring is unaffected.
            with ThreadPoolExecutor(max_workers=batch) as executor:
                futures = [
                    executor.submit(process_question, idx_item)
                    for idx_item in enumerate(remaining_questions)
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                    result = future.result()
                    row_df = pd.DataFrame([result])
                    row_df.to_csv(self.response_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            # Fallback to sequential processing
            for idx, item in tqdm(list(enumerate(remaining_questions)),
                                  desc="Generating responses"):
                result = process_question((idx, item))
                row_df = pd.DataFrame([result])
                row_df.to_csv(self.response_file, mode='a', header=False, index=False, encoding='utf-8')


    def evaluate_cybermetric(self):
        return self.eval.compute_cybermetric_accuracy(self.response_file)