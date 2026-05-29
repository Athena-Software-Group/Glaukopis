import argparse
import pandas as pd
from pipelines.models import model_mapping
from pathlib import Path
import os
from pipelines.evaluation.cti_eval import CTIEvaluate
from pipelines.evaluation.athena_cti_eval import ATHENAEvaluate

def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmark results.')
    parser.add_argument('--task', choices=['mcq', 'rcm', 'vsp', 'taa', 'ate', 'urlhaus', 'cve',
                                           'glue', 'superglue', 'mmlu', 'mmlu-pro', 'athena', 'athena-ate','athena-rcm',
                                           'athena-rms','athena-taa','athena-taa-canonical',
                                           'athena-vsp','athena-mcq','cybermetric'],
                        help='Task to evaluate')
    parser.add_argument('--model', required=True,
                        help='Model name key (e.g., gpt-3.5-turbo, gpt-4-turbo, gemini-1.5-pro)')
    parser.add_argument('--response_file', help='Path to the response TSV, CSV, or JSONL file')
    parser.add_argument('--mini', action='store_true', default=False,
                        help='Evaluate only on mini datasets (for Athena tasks)')
    args = parser.parse_args()

    # Validate model key and get value (responses folder uses value name)
    if args.model not in model_mapping:
        raise ValueError(f"Model key '{args.model}' not found in model_mapping. Available keys: {list(model_mapping.keys())}")
    display_model_name = model_mapping[args.model].replace('/', '_')  # folder name in responses/
    model_column = args.model  # column name in response files

    eval = CTIEvaluate()
    athena_eval = ATHENAEvaluate()

    # --- Athena tasks handling ---
    if args.task == 'athena':
        # Call evaluate_folder for all Athena tasks
        print(f"Evaluating all Athena tasks for model: {display_model_name}")
        results = athena_eval.evaluate_folder(display_model_name, mini=args.mini)
        print("ATHENA Evaluation Results:")
        for task, metrics in results.items():
            print(f"{task}: {metrics}")
        return

    # --- Individual Athena-xxx tasks ---
    elif args.task.startswith('athena-'):
        if not args.response_file:
            raise ValueError("Please provide --response_file for Athena tasks.")
        preds_path = Path(args.response_file)
        out_path = preds_path.with_name(preds_path.stem + "_scored.jsonl") if not preds_path.name.endswith('_scored.jsonl') else preds_path
        metrics = athena_eval.evaluate_file(args.task, preds_path, out_path)
        print(f"ATHENA-{args.task.split('-')[1].upper()} Metrics: {metrics}")
        return

    # --- Non-Athena tasks ---
    print(f"Evaluating task: {args.task}")
    print(f"Response file: {args.response_file}")
    print(f"Model column: {model_column}")

    if args.task == 'mcq':
        data = pd.read_csv(args.response_file, sep='\t')
        for index, row in data.iterrows():
            pred = row[model_column]
            console_pred = pred[0] if pred and isinstance(pred, str) else 'A'
            if 'Prompt' in data.columns and 'Raw_Response' in data.columns:
                prompt_preview = row['Prompt'][:30] + "..." if len(row['Prompt']) > 30 else row['Prompt']
                raw_response_preview = row['Raw_Response'][:40] + "..." if len(row['Raw_Response']) > 40 else row['Raw_Response']
                print(f"Row {index+1}: {prompt_preview} | Raw: {raw_response_preview} -> {console_pred}")
            elif 'Prompt' in data.columns:
                prompt_preview = row['Prompt'][:50] + "..." if len(row['Prompt']) > 50 else row['Prompt']
                print(f"Row {index+1}: {prompt_preview} -> {console_pred}")
            else:
                print(f"Row {index+1}: {console_pred}")
        result = eval.compute_mcq_accuracy(args.response_file, model_column)
        print(f"MCQ Accuracy: {result}")

    elif args.task == 'rcm':
        data = pd.read_csv(args.response_file, sep='\t')
        for index, row in data.iterrows():
            pred = row[model_column]
            print(f"Row {index+1}: {pred}")
        result = eval.compute_rcm_accuracy(args.response_file, model_column)
        print(f"RCM Accuracy: {result}")

    elif args.task == 'vsp':
        data = pd.read_csv(args.response_file, sep='\t')
        for index, row in data.iterrows():
            pred = row[model_column]
            print(f"Row {index+1}: {pred}")
        result = eval.compute_vsp_mad(args.response_file, model_column)
        print(f"VSP Mean Absolute Deviation: {float(result)}")

    elif args.task == 'taa':
        data = pd.read_csv(args.response_file, sep='\t')
        for index, row in data.iterrows():
            pred = row[model_column]
            print(f"Row {index+1}: {pred}")
        result = eval.compute_taa_accuracy(args.response_file, model_column)
        print(f"TAA Accuracy: {result}")

    elif args.task == 'ate':
        data = pd.read_csv(args.response_file, sep='\t')
        for index, row in data.iterrows():
            pred = row[model_column]
            print(f"Row {index+1}: {pred}")
        result = eval.compute_ate_f1(args.response_file, model_column)
        print(f"ATE F1 Score: {result:.4f}")

    elif args.task == 'urlhaus':
        data = pd.read_csv(args.response_file)
        for index, row in data.iterrows():
            pred = row['model_prediction']
            print(f"Row {index+1}: True={row['reference']} | Pred={pred}")
        acc, f1 = eval.urlhaus_acc_f1(args.response_file)
        print(f"IOC Benchmark -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

    elif args.task == 'cve':
        data = pd.read_csv(args.response_file)
        for index, row in data.iterrows():
            pred = row['predicted_cve']
            gt = row['ground_truth']
            print(f"Row {index+1}: True={gt} | Pred={pred}")
        accuracy = eval.compute_cve_accuracy(args.response_file)
        print(f"CVE Benchmark Accuracy: {accuracy:.4f}")

    elif args.task == 'glue':
        if os.path.isdir(args.response_file):
            print(f"Evaluating GLUE results in folder: {args.response_file}")
            metrics = eval.evaluate_glue_folder(args.response_file)
        else:
            print(f"Evaluating GLUE results from file: {args.response_file}")
            metrics = eval.evaluate_glue_csv(args.response_file)
        print(f"GLUE Benchmark Results: {metrics}")

    elif args.task == 'superglue':
        if os.path.isdir(args.response_file):
            print(f"Evaluating SUPERGLUE results in folder: {args.response_file}")
            metrics = eval.evaluate_superglue_folder(args.response_file)
        else:
            print(f"Evaluating SUPERGLUE results from file: {args.response_file}")
            metrics = eval.evaluate_superglue_csv(args.response_file)
        print(f"SUPERGLUE Benchmark Results: {metrics}")

    elif args.task == 'mmlu':
        accuracy = eval.compute_mmlu_accuracy(args.response_file)
        print(f"MMLU Benchmark Accuracy: {accuracy:.4f}")

    elif args.task == 'mmlu-pro':
        result = eval.compute_mmlu_pro_accuracy(args.response_file)
        print(f"MMLU-Pro Benchmark Result: {result}")

    elif args.task == 'cybermetric':
        accuracy = eval.compute_cybermetric_accuracy(args.response_file)
        print(f"Cybermetric Benchmark Result: {accuracy}")

if __name__ == "__main__":
    main()
