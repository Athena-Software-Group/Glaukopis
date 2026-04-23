import os
import re
import pandas as pd
from evaluate import load
from benchmarks.glue import GLUE
from benchmarks.superglue import SUPERGLUE
from pipelines.data_loader import load_pickle_file
from sklearn.metrics import accuracy_score, f1_score
from cvss import CVSS3


class CTIEvaluate:
    """Unified evaluation class for multiple CTI and NLP benchmarks."""

    # ---------- ATE ----------
    def compute_ate_f1(self, response_file, model_name):
        data = pd.read_csv(response_file, sep='\t')
        if 'GT' in data.columns:
            gt_col = 'GT'
        elif 'Ground Truth' in data.columns:
            gt_col = 'Ground Truth'
        else:
            raise ValueError(f"TSV file must contain a ground truth column named 'GT' or 'Ground Truth'")
        if model_name not in data.columns:
            raise ValueError(f"TSV file must contain '{model_name}' column")

        gt_strings = data[gt_col].astype(str)
        pred_strings = data[model_name].astype(str)

        def parse_technique_ids(s):
            if pd.isna(s) or s.strip() == '' or s.strip().upper() == 'X':
                return set()
            return set(re.findall(r'T\d{4}', s.upper()))

        total_f1, valid_rows = 0, 0
        for i in range(len(data)):
            gt_ids = parse_technique_ids(gt_strings.iloc[i])
            pred_ids = parse_technique_ids(pred_strings.iloc[i])
            if not gt_ids:
                continue
            valid_rows += 1
            tp = len(gt_ids & pred_ids)
            fp = len(pred_ids - gt_ids)
            fn = len(gt_ids - pred_ids)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            total_f1 += f1

        return total_f1 / valid_rows if valid_rows > 0 else 0.0

    # ---------- CVE ----------
    def compute_cve_accuracy(self, response_file, prediction_col='predicted_cve', gt_col='ground_truth'):
        df = pd.read_csv(response_file)
        if df.empty:
            print("No data found in response file.")
            return 0.0
        correct = (df[prediction_col].astype(str).str.upper() == df[gt_col].astype(str).str.upper()).sum()
        return correct / len(df) * 100

    # ---------- GLUE ----------
    def evaluate_glue_csv(self, csv_file_path: str):
        df = pd.read_csv(csv_file_path)
        basename = os.path.basename(csv_file_path)
        if not basename.startswith("glue_"):
            raise ValueError(f"Invalid GLUE response file name: {basename}")

        subtask_with_extra = basename[len("glue_"):]
        subtask = next((k for k in GLUE.TASKS if subtask_with_extra.startswith(k)), None)
        if not subtask:
            raise ValueError(f"Unknown subtask in CSV {csv_file_path}")

        task_config = GLUE.TASKS[subtask]
        metric = load(*task_config["evaluate_metric"])
        results = metric.compute(
            predictions=df["prediction"].tolist(),
            references=df["reference"].tolist(),
        )
        return {"subtask": subtask, "metrics": results}

    def evaluate_glue_folder(self, response_folder: str):
        results = {}
        for fname in os.listdir(response_folder):
            if fname.endswith(".csv") and fname.startswith("glue_"):
                try:
                    result = self.evaluate_glue_csv(os.path.join(response_folder, fname))
                    results[result["subtask"]] = result["metrics"]
                except Exception as e:
                    print(f"Error evaluating {fname}: {e}")
        return results

    # ---------- MCQ ----------
    def compute_mcq_accuracy(self, fname, col):
        df = pd.read_csv(fname, sep='\t')
        # Support both legacy 'GT' and newer 'Ground Truth' column names
        if 'GT' in df.columns:
            gt_col = 'GT'
        elif 'Ground Truth' in df.columns:
            gt_col = 'Ground Truth'
        else:
            raise ValueError("TSV file must contain a ground truth column named 'GT' or 'Ground Truth'")

        correct = total = 0
        for idx, row in df.iterrows():
            pred = str(row[col]).strip().upper()
            gt = str(row[gt_col]).strip().upper()
            if pred in ['A', 'B', 'C', 'D', 'X']:
                total += 1
            else:
                print(f'Invalid response at row {idx+1}')
            if pred == gt:
                correct += 1
        return correct / total * 100 if total > 0 else 0.0

    # ---------- MMLU ----------
    def compute_mmlu_accuracy(self, csv_file: str):
        df = pd.read_csv(csv_file)
        correct = sum(str(r["prediction"]).strip().upper() == str(r["ground_truth"]).strip().upper()
                      for _, r in df.iterrows())
        return correct / len(df) if len(df) > 0 else 0.0

    # ---------- RCM ----------
    def compute_rcm_accuracy(self, fname, col):
        df = pd.read_csv(fname, sep='\t')
        if 'GT' in df.columns:
            gt_col = 'GT'
        elif 'Ground Truth' in df.columns:
            gt_col = 'Ground Truth'
        else:
            raise ValueError("TSV file must contain a ground truth column named 'GT' or 'Ground Truth'")
        correct = total = 0
        for idx, row in df.iterrows():
            pred, gt = str(row[col]).upper(), str(row[gt_col]).upper()
            if pred.startswith('CWE-'):
                total += 1
            else:
                print(f'Invalid response at row {idx+1}')
            if pred == gt:
                correct += 1
        return correct / total * 100 if total > 0 else 0.0

    # ---------- SuperGLUE ----------
    def evaluate_superglue_csv(self, csv_file_path: str):
        df = pd.read_csv(csv_file_path)
        basename = os.path.basename(csv_file_path)
        if not basename.startswith("superglue_"):
            raise ValueError(f"Invalid SuperGLUE response file name: {basename}")

        subtask_with_extra = basename[len("superglue_"):]
        subtask = next((k for k in SUPERGLUE.TASKS if subtask_with_extra.startswith(k)), None)
        if not subtask:
            raise ValueError(f"Unknown subtask in CSV {csv_file_path}")

        if subtask == "record":
            predictions = df["prediction"].apply(eval).tolist()
            references = df["reference"].apply(eval).tolist()
        else:
            predictions, references = df["prediction"].tolist(), df["reference"].tolist()

        task_config = SUPERGLUE.TASKS[subtask]
        metric = load(*task_config["evaluate_metric"])
        metrics = metric.compute(predictions=predictions, references=references)
        return {"subtask": subtask, "metrics": metrics}

    def evaluate_superglue_folder(self, response_folder: str):
        results = {}
        for fname in os.listdir(response_folder):
            if fname.endswith(".csv") and fname.startswith("superglue_"):
                try:
                    result = self.evaluate_superglue_csv(os.path.join(response_folder, fname))
                    results[result["subtask"]] = result["metrics"]
                except Exception as e:
                    print(f"Error evaluating {fname}: {e}")
        return results

    # ---------- TAA ----------
    def threat_actor_connection(self, actor1, actor2, alias_dict, related_dict):
        actor1, actor2 = actor1.strip().lower(), actor2.strip().lower()
        alias_dict = {k.strip().lower(): [v.strip().lower() for v in val] for k, val in alias_dict.items()}
        for actor in list(alias_dict):
            for alias in alias_dict[actor]:
                alias_dict.setdefault(alias, []).append(actor)

        related_dict = {k.strip().lower(): [v.strip().lower() for v in val] for k, val in related_dict.items()}
        for actor in list(related_dict):
            for rel in related_dict[actor]:
                related_dict.setdefault(rel, []).append(actor)

        if self.is_alias_connected(actor1, actor2, alias_dict):
            return "C"
        if self.is_related_connected(actor1, actor2, alias_dict, related_dict):
            return "P"
        return "I"

    def is_alias_connected(self, a1, a2, alias_dict):
        visited, queue = set(), [a1]
        while queue:
            cur = queue.pop(0)
            visited.add(cur)
            for alias in alias_dict.get(cur, []):
                if alias == a2:
                    return True
                if alias not in visited:
                    queue.append(alias)
        return False

    def is_related_connected(self, a1, a2, alias_dict, related_dict):
        visited, queue = set(), [a1]
        while queue:
            cur = queue.pop(0)
            visited.add(cur)
            for alias in alias_dict.get(cur, []):
                if alias == a2:
                    return True
                if alias not in visited:
                    queue.append(alias)
            for rel in related_dict.get(cur, []):
                if rel == a2:
                    return True
                if rel not in visited:
                    queue.append(rel)
        return False

    def compute_taa_accuracy(self, fname, col):
        df = pd.read_csv(fname, sep='\t')
        if 'GT' in df.columns:
            gt_col = 'GT'
        elif 'Ground Truth' in df.columns:
            gt_col = 'Ground Truth'
        else:
            raise ValueError("TSV file must contain a ground truth column named 'GT' or 'Ground Truth'")
        correct = plausible = total = 0
        alias_dict = load_pickle_file('benchmark_data/cti_bench/cti_taa/alias_dict.pickle')
        related_dict = load_pickle_file('benchmark_data/cti_bench/cti_taa/related_dict.pickle')
        for _, row in df.iterrows():
            pred, gt = str(row[col]).lower().strip(), str(row[gt_col]).lower().strip()
            res = self.threat_actor_connection(gt, pred, alias_dict, related_dict)
            if res == 'C':
                correct += 1
            elif res == 'P':
                plausible += 1
            total += 1
        return correct/total*100, (correct+plausible)/total*100

    # ---------- IOC (URL classification) ----------
    def urlhaus_acc_f1(self, response_file):
        df = pd.read_csv(response_file)
        y_true = df['reference'].astype(str).str.lower()
        y_pred = df['model_prediction'].astype(str).str.lower()
        valid_labels = ["benign", "malicious"]
        y_pred = y_pred.apply(lambda x: x if x in valid_labels else "malicious")
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        return acc, f1_macro

    # ---------- VSP ----------
    def get_cvss_score(self, vector):
        c = CVSS3(vector)
        cvss_score = c.scores()[0]
        return cvss_score

    def compute_vsp_mad(self, fname, col):
        cvss_prefix = 'CVSS:3.0/'   # should be empty string if the model responds with the prefix
        df = pd.read_csv(fname, sep='\t')
        if 'GT' in df.columns:
            gt_col = 'GT'
        elif 'Ground Truth' in df.columns:
            gt_col = 'Ground Truth'
        else:
            raise ValueError("TSV file must contain a ground truth column named 'GT' or 'Ground Truth'")
        error = 0
        total = 0
        for idx, row in df.iterrows():
            pred = str(row[col]).upper()
            gt = str(row[gt_col]).upper()
            try:
                pred_vector = cvss_prefix + pred
                pred_score = self.get_cvss_score(pred_vector)
                gt_score = self.get_cvss_score(gt)
                error += abs(pred_score-gt_score)
            except Exception as e:
                print('Invalid response at row {}'.format(idx+1))
                print(e)
                continue
            total += 1
                
        return error/total if total > 0 else float('inf')
    
    # ---------------------- Cybermetric Accuracy -------------------------- #
    def compute_cybermetric_accuracy(self, response_file, answer_col='answer_after_post_processing', gt_col='answer'):
        """
        Compute accuracy for CyberMetric MCQ tasks.

        Args:
            response_file (str): CSV file path containing columns: 'question', 'answer', 'raw_response', 'answer_after_post_processing'.
            answer_col (str): Column name containing post-processed answers.
            gt_col (str): Column name containing ground truth answers.

        Returns:
            float: Accuracy in percentage (0-100).
        """
        df = pd.read_csv(response_file)
        if df.empty:
            print("No data found in response file.")
            return 0.0

        # Compare answers case-insensitively
        correct = (df[answer_col].astype(str).str.upper() == df[gt_col].astype(str).str.upper()).sum()
        accuracy = correct / len(df) * 100
        return accuracy
