import argparse
from benchmarks import CTIMCQ, CTIRCM, CTIVSP, CTIATE, CTITAA, URLHAUS, CVE, GLUE, SUPERGLUE, MMLU, ATHENAATE, ATHENARCM, ATHENARMS, ATHENATAA, ATHENAVSP, CYBERMETRIC, ATHENAMCQ, CYBERSOCEVALMALWARE, CYBERSOCEVALTI
from pipelines.models import model_mapping, cleanup_model_cache, get_cached_model
from pipelines.api_usage import get_totals, save_checkpoint, restore_checkpoint
import os

# Headline keys we know are percentages (rendered with a trailing %).
_PCT_KEYS = {
    "accuracy", "avg_score", "parse_error_pct",
    "plausible_accuracy", "combined_accuracy",
    "f1", "plausible_f1", "combined_f1",
    "correct_accuracy",
}


def _print_pretty_result(task: str, model: str, result) -> None:
    """Pretty-print the per-task evaluator result before the canonical line.

    The canonical ``Evaluation result for <task> with <model>: <repr>`` line
    must remain a single-line literal-eval-able dict (parser contract with
    run_benchmark.sh + _print_sweep_summary.py). This helper emits a
    multi-line readable summary first so live-streamed sweep logs are
    easy to scan, especially for tasks like cybersoceval-* whose metrics
    dict carries nested per-slice breakdowns.
    """
    if not isinstance(result, dict):
        return
    print(f"--- Pretty result: {task} ({model}) ---")
    nested = []
    for k, v in result.items():
        if isinstance(v, dict):
            nested.append((k, v))
            continue
        if isinstance(v, float):
            if k in _PCT_KEYS:
                print(f"  {k:32s} {v:>8.2f}%")
            else:
                print(f"  {k:32s} {v:>10.4f}")
        else:
            print(f"  {k:32s} {v}")
    for slice_key, slice_dict in nested:
        if not slice_dict:
            continue
        print(f"  {slice_key}:")
        rows = sorted(
            slice_dict.items(),
            key=lambda kv: -(int(kv[1].get("answered", 0)) + int(kv[1].get("parse_errors", 0)))
            if isinstance(kv[1], dict) else 0,
        )
        name_w = min(28, max((len(str(n)) for n, _ in rows), default=4))
        print(f"    {'name':<{name_w}}  {'N':>5}  {'Jaccard':>8}  {'Strict':>7}  {'PE':>4}")
        for name, vals in rows:
            if not isinstance(vals, dict):
                continue
            answered = int(vals.get("answered", 0))
            pe = int(vals.get("parse_errors", 0))
            n = answered + pe
            jacc = float(vals.get("avg_score", 0.0))
            acc = float(vals.get("correct_mc_pct", 0.0))
            print(f"    {str(name):<{name_w}}  {n:>5}  {jacc:>7.2f}%  {acc:>6.2f}%  {pe:>4}")
    print(f"--- end pretty result ---")


def main():
    parser = argparse.ArgumentParser(description="Run inference for CTI tasks.")
    parser.add_argument("task", choices=["mcq", "rcm", "vsp", "ate", "taa", "urlhaus", "cve", "glue", "superglue", "mmlu",
                                         "athena-ate", "athena-rcm", "athena-rms", "athena-taa", "athena-vsp","athena-mcq", "cybermetric",
                                         "cybersoceval-malware", "cybersoceval-ti"],
                        help="Task to evaluate (mcq, rcm, vsp, ate, taa)")
    parser.add_argument("subtask", nargs="?", default=None, help="Optional GLUE or SUPERGLUE subtask (e.g., cola, sst2)")
    parser.add_argument("model_name", help="Model name (e.g., gpt-3.5-turbo, gemini, llama-3-8b)")
    parser.add_argument("--athena-cti-lnd", dest="athena_cti_lnd", action="store_true", help="Enable web search preview tool (only for CVE task with GPT-5 or Gemini)")
    parser.add_argument("--batch", type=int, default=None, help="Number of concurrent workers (only valid for GPT or Gemini models)")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows to process (default: all)")
    parser.add_argument("--version", type=int, default=1, help="Run version number (default=1). Use higher numbers for fresh runs")
    parser.add_argument("--data_path", default=None, help="Path to the input TSV file (default: data/cti-<task>.tsv)")
    parser.add_argument("--cleanup", action="store_true", help="Force cleanup of model from memory after each inference")
    parser.add_argument("--reasoning_effort", choices=["none", "low", "medium", "high", "xhigh"], default=None,
                        help="Reasoning effort for GPT-5.2 (none/low/medium/high/xhigh)")

    args = parser.parse_args()

    # Create reverse mapping from values to keys
    reverse_model_mapping = {v: k for k, v in model_mapping.items()}

    # Convert model display name to key
    if args.model_name in model_mapping:
        model_key = args.model_name
    elif args.model_name in reverse_model_mapping:
        model_key = reverse_model_mapping[args.model_name]
    else:
        model_key = args.model_name
        print(f"Warning: Model {args.model_name} not found in mapping, using as-is")

    # validate batch flag
    is_gpt_or_gemini = any(k in model_key for k in ["gpt", "gemini"])
    is_hf_inference = model_key.endswith("-hf")
    is_vllm = model_key.endswith("-vllm")
    if args.batch is not None:
        if not (is_gpt_or_gemini or is_hf_inference or is_vllm):
            raise ValueError("--batch flag is only supported for GPT, Gemini, HF Inference (*-hf), and vLLM (*-vllm) models")
        if args.batch <= 0:
            raise ValueError("--batch must be a positive integer")
        
    # Restore previous totals
    restore_checkpoint(args.task, model_key,version=args.version)

    task_classes = {
        'mcq': CTIMCQ,
        'rcm': CTIRCM,
        'vsp': CTIVSP,
        'ate': CTIATE,
        'taa': CTITAA,
        'urlhaus': URLHAUS,
        'cve': CVE,
        'glue': GLUE,
        'superglue': SUPERGLUE,
        'mmlu': MMLU,
        'athena-ate': ATHENAATE,
        'athena-rcm': ATHENARCM,
        'athena-rms': ATHENARMS,
        'athena-taa': ATHENATAA,
        'athena-vsp': ATHENAVSP,
        'athena-mcq': ATHENAMCQ,
        'cybermetric': CYBERMETRIC,
        'cybersoceval-malware': CYBERSOCEVALMALWARE,
        'cybersoceval-ti': CYBERSOCEVALTI,
    }

    if args.task not in task_classes:
        raise ValueError(f"Unknown task: {args.task}")

    print(f"Model {args.model_name} initializing and generating results...")
    
    # Instantiate benchmark class
    benchmark = task_classes[args.task](model_key, args.rows, args.data_path, version=args.version)

    if args.task in ["glue", "superglue"]:
        if args.subtask:
            if args.subtask not in benchmark.TASKS:
                raise ValueError(f"Unknown {args.task.capitalize()} subtask: {args.subtask}")
            benchmark.tasks = {args.subtask: benchmark.TASKS[args.subtask]}
            print(f"Running {args.task.capitalize()} benchmark only for subtask: {args.subtask}")
        else:
            print(f"Running {args.task.capitalize()} benchmark for all subtasks")

    # Set reasoning effort on the model instance and update output paths
    # (applies to the OpenAI responses-API reasoning family: gpt5.2, gpt5.5, gpt5.5-pro)
    from pipelines.models import REASONING_MODELS
    if args.reasoning_effort and model_key in REASONING_MODELS:
        model_instance = get_cached_model(model_key)
        model_instance.reasoning_effort = args.reasoning_effort

        old_display = benchmark.display_model_name
        new_display = f"{old_display}-{args.reasoning_effort}"
        benchmark.display_model_name = new_display
        benchmark.model_folder = benchmark.model_folder.replace(old_display, new_display)
        os.makedirs(benchmark.model_folder, exist_ok=True)
        if isinstance(benchmark.response_file, str):
            benchmark.response_file = benchmark.response_file.replace(old_display, new_display)

    # Pass web search flag only for CVE + supported models
    use_web_search = args.task == "cve" and args.athena_cti_lnd and model_key in ["gpt5", "gemini-2.5-flash"]

    try:
        # Pass both cleanup and use_web_search to generate_responses
        benchmark.generate_responses(cleanup=args.cleanup, use_web_search=use_web_search,batch=args.batch if (is_gpt_or_gemini or is_hf_inference or is_vllm) else None)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving checkpoint...")
        save_checkpoint(args.task, model_key, version=args.version)
        raise

    print("Results completed")

    print("Evaluating results...")
    if args.task == 'mcq':
        result = benchmark.compute_mcq_accuracy()
    elif args.task == 'rcm':
        result = benchmark.compute_rcm_accuracy()
    elif args.task == 'vsp':
        result = benchmark.compute_vsp_mad()
    elif args.task == 'ate':
        result = benchmark.compute_ate_f1()
    elif args.task == 'urlhaus':
        acc, f1 = benchmark.compute_accuracy()
        result = {'accuracy': acc, 'f1_score': f1}
    elif args.task == 'cve':
        result = benchmark.evaluate_cve()
    elif args.task == 'glue':
        result = benchmark.evaluate_glue()
    elif args.task == 'superglue':
        result = benchmark.evaluate_superglue()
    elif args.task == 'mmlu':
        result = benchmark.evaluate_mmlu()
    elif args.task == 'athena-ate':
        result = benchmark.evaluate_athena_ate()
    elif args.task == 'athena-rcm':
        result = benchmark.evaluate_athena_rcm()
    elif args.task == 'athena-rms':
        result = benchmark.evaluate_athena_rms()
    elif args.task == 'athena-taa':
        result = benchmark.evaluate_athena_taa()
    elif args.task == 'athena-vsp':
        result = benchmark.evaluate_athena_vsp()
    elif args.task == 'athena-mcq':
        result = benchmark.evaluate_athena_mcq()
    elif args.task == 'cybermetric':
        result = benchmark.evaluate_cybermetric()
    elif args.task == 'cybersoceval-malware':
        result = benchmark.evaluate_cybersoceval_malware()
    elif args.task == 'cybersoceval-ti':
        result = benchmark.evaluate_cybersoceval_ti()
    else:  # taa
        correct_acc, plausible_acc = benchmark.compute_taa_accuracy()
        result = {'correct_accuracy': correct_acc, 'plausible_accuracy': plausible_acc}

    _print_pretty_result(args.task, args.model_name, result)
    # Canonical single-line dict repr below is the parser contract that
    # run_benchmark.sh greps to populate RES_METRICS; do not split or
    # reformat without also updating the grep + ast.literal_eval path in
    # _print_sweep_summary.py.
    print(f"Evaluation result for {args.task} with {args.model_name}: {result}")
    # Only for API-based GPT or Gemini models, show cumulative usage and cost
    # Exclude gpt-oss models as they are Hugging Face models that run locally
    if (any(k in model_key for k in ["gpt", "gemini"]) and not model_key.startswith("gpt-oss")):
        totals = get_totals()
        print("=== API Usage Totals ===")
        print(f"Input tokens: {totals['input_tokens']}")
        print(f"Output tokens: {totals['output_tokens']}")
        print(f"Total tokens: {totals['total_tokens']}")
        print(f"Grounding calls: {totals['grounding_calls']}")
        print(f"Input cost (USD): {totals['input_cost']:.4f}")
        print(f"Output cost (USD): {totals['output_cost']:.4f}")
        print(f"Total cost (USD): {totals['total_cost']:.4f}")

    # Clean up model cache at the end to free memory
    if not args.cleanup:
        print("Cleaning up model cache...")
        cleanup_model_cache()


if __name__ == "__main__":
    main()