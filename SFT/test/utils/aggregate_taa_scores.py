"""Aggregate Athena-TAA scores (accuracy / plausible / combined) across every
model that has produced a response file under SFT/test/responses/.

Walks responses/<model>/athena-taa/*.jsonl, picks the largest response file
per model directory (handles canonical, pre-canonical, and scraped naming
conventions side-by-side), runs ATHENAEvaluate.evaluate_file once per model,
and prints a single sorted table. Also drops a Markdown + CSV artifact at
SFT/test/responses/_summary_taa.{md,csv} so the result is shareable.

Usage (must run from SFT/test/ so the alias_csv / related_csv relative paths
resolve inside ATHENAEvaluate.__init__):

    cd /home/Glaukopis/SFT/test
    python utils/aggregate_taa_scores.py
    # or:
    python utils/aggregate_taa_scores.py --responses-dir responses --min-rows 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make 'pipelines.*' importable when run from SFT/test/.
HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
sys.path.insert(0, str(TEST_DIR))

from pipelines.evaluation.athena_cti_eval import ATHENAEvaluate  # noqa: E402


def count_jsonl_rows(path: Path) -> int:
    """Number of non-empty lines (response files don't have headers)."""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def discover_taa_files(responses_dir: Path) -> List[Tuple[str, Path, int]]:
    """For each model dir under responses/, return (model_dir_name, best_file, row_count).

    'Best file' = the response *.jsonl with the most rows (ignores _scored.jsonl,
    summary files, and zero-row files)."""
    out: List[Tuple[str, Path, int]] = []
    for model_dir in sorted(responses_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        taa_dir = model_dir / "athena-taa"
        if not taa_dir.is_dir():
            continue
        candidates = []
        for p in taa_dir.glob("*.jsonl"):
            name = p.name
            if name.endswith("_scored.jsonl"):
                continue
            if name.startswith("summary_"):
                continue
            try:
                n = count_jsonl_rows(p)
            except Exception:
                n = 0
            if n > 0:
                candidates.append((p, n))
        if not candidates:
            continue
        candidates.sort(key=lambda t: t[1], reverse=True)
        best_file, best_n = candidates[0]
        out.append((model_dir.name, best_file, best_n))
    return out


def evaluate_one(evalr: ATHENAEvaluate, preds_path: Path) -> Dict[str, float]:
    scored_path = preds_path.with_name(preds_path.stem + "_scored.jsonl")
    return evalr.evaluate_file("athena-taa", preds_path, scored_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses-dir", default="responses",
                        help="Path to the responses/ root (relative to SFT/test/, default: responses)")
    parser.add_argument("--min-rows", type=int, default=1,
                        help="Skip response files with fewer than this many rows (default: 1)")
    parser.add_argument("--out-stem", default="responses/_summary_taa",
                        help="Stem for the .md and .csv artifacts (default: responses/_summary_taa)")
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir).resolve()
    if not responses_dir.is_dir():
        sys.stderr.write(f"responses dir not found: {responses_dir}\n")
        return 2

    found = discover_taa_files(responses_dir)
    found = [t for t in found if t[2] >= args.min_rows]
    if not found:
        print("No athena-taa response files discovered.")
        return 1

    print(f"Discovered {len(found)} model(s) with athena-taa response files.\n")
    evalr = ATHENAEvaluate()

    rows: List[Dict[str, object]] = []
    for model, preds_path, n_rows in found:
        try:
            metrics = evaluate_one(evalr, preds_path)
        except Exception as e:
            print(f"  [WARN] {model}: evaluate_file failed: {e}")
            continue
        rows.append({
            "model": model,
            "rows": n_rows,
            "accuracy": metrics.get("accuracy", 0.0),
            "plausible_accuracy": metrics.get("plausible_accuracy", 0.0),
            "combined_accuracy": metrics.get("combined_accuracy", 0.0),
            "response_file": str(preds_path.relative_to(responses_dir.parent)),
        })

    rows.sort(key=lambda r: r["combined_accuracy"], reverse=True)

    header = f"{'model':50s}  {'rows':>5s}  {'acc':>7s}  {'plaus':>7s}  {'comb':>7s}"
    print("\n" + header); print("-" * len(header))
    for r in rows:
        print(f"{r['model'][:50]:50s}  {r['rows']:>5d}  "
              f"{r['accuracy']:>6.2f}%  {r['plausible_accuracy']:>6.2f}%  {r['combined_accuracy']:>6.2f}%")

    out_md = (responses_dir.parent / f"{args.out_stem}.md").resolve()
    out_csv = (responses_dir.parent / f"{args.out_stem}.csv").resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| Model | Rows | Accuracy | Plausible | Combined |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| `{r['model']}` | {r['rows']} | {r['accuracy']:.2f}% | "
                    f"{r['plausible_accuracy']:.2f}% | {r['combined_accuracy']:.2f}% |\n")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "rows", "accuracy",
                                          "plausible_accuracy", "combined_accuracy",
                                          "response_file"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote: {out_md}\nWrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
