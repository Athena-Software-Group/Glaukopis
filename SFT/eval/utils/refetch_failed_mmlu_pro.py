#!/usr/bin/env python3
"""Re-run only the failed rows of an MMLU-Pro response CSV in place.

Targets rows whose `raw_response` was written as a sentinel error string
(matches the prefix emitted by pipelines.models.get_single_prediction:
"Error generating response:") or whose `raw_response` is empty. These
arise from transient provider outages (Gemini 503 UNAVAILABLE windows,
HF Router 502/504 storms) that the inner retry layers in GeminiModel /
HFInferenceModel can't always absorb. The base benchmark resume logic
treats any row already present in the CSV as "done" and skips it, so
without a targeted refetch the failed predictions silently depress
final accuracy.

Prompts are rebuilt by reusing MMLUPRO.load_samples() so the prompt
template stays in sync with the live benchmark (any future change to
the upstream prompt format is picked up automatically). Rows are
matched by `idx` (the original dataset index, also the CSV's index
column). Successfully refetched rows are written back in place; rows
that fail again are left unchanged so they remain visible for the
next pass.

Usage:
    cd SFT/eval
    python -m utils.refetch_failed_mmlu_pro \\
        --model gemini-2.5-flash \\
        [--response-file responses/<alias>/mmlu-pro/...csv] \\
        [--batch 8]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# Allow running as a module from SFT/eval (matches inference.py CWD convention).
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_ROOT = os.path.dirname(_HERE)
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

from benchmarks.mmlu_pro import MMLUPRO  # noqa: E402
from pipelines.models import get_single_prediction, alias_to_safe_name  # noqa: E402


_ERROR_PREFIX = "Error generating response"


def _is_failed_row(raw_response) -> bool:
    if raw_response is None:
        return True
    s = str(raw_response).strip()
    if not s or s.lower() == "nan":
        return True
    return s.startswith(_ERROR_PREFIX)


def _default_response_file(model_name: str) -> str:
    alias = alias_to_safe_name(model_name)
    return os.path.join("responses", alias, "mmlu-pro",
                        f"mmlu-pro_all_v1_{alias}_response.csv")


def refetch(model_name: str, response_file: str, batch: int) -> None:
    if not os.path.exists(response_file):
        raise FileNotFoundError(f"response file not found: {response_file}")

    df = pd.read_csv(response_file, index_col=0)
    failed_mask = df["raw_response"].apply(_is_failed_row)
    failed_idxs = df.index[failed_mask].tolist()
    total = len(df)
    print(f"[refetch] {response_file}: {len(failed_idxs)} / {total} failed rows")
    if not failed_idxs:
        return

    # Rebuild prompts via the live benchmark loader so any prompt-template
    # change in MMLUPRO is mirrored here without duplication.
    bench = MMLUPRO(model_name=model_name)
    samples = bench.load_samples()
    samples_by_idx = {idx: s for idx, s in enumerate(samples)}

    missing = [i for i in failed_idxs if i not in samples_by_idx]
    if missing:
        print(f"[refetch] WARNING: {len(missing)} failed idx not present in "
              f"loaded dataset (e.g. {missing[:5]}); skipping those")

    targets = [(i, samples_by_idx[i]) for i in failed_idxs if i in samples_by_idx]

    def _refetch_one(item):
        idx, sample = item
        raw = get_single_prediction(
            sample["prompt"],
            model_name,
            task="mmlu-pro",
        )
        predicted = MMLUPRO._extract_answer(raw)
        return idx, raw, predicted

    recovered = 0
    still_failed = 0
    if batch and batch > 1:
        with ThreadPoolExecutor(max_workers=batch) as ex:
            futures = [ex.submit(_refetch_one, t) for t in targets]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="Refetching MMLU-Pro rows"):
                idx, raw, predicted = f.result()
                if _is_failed_row(raw):
                    still_failed += 1
                    continue
                df.at[idx, "raw_response"] = raw
                df.at[idx, "prediction"] = predicted if predicted else "NOT_FOUND"
                recovered += 1
    else:
        for t in tqdm(targets, desc="Refetching MMLU-Pro rows"):
            idx, raw, predicted = _refetch_one(t)
            if _is_failed_row(raw):
                still_failed += 1
                continue
            df.at[idx, "raw_response"] = raw
            df.at[idx, "prediction"] = predicted if predicted else "NOT_FOUND"
            recovered += 1

    df.to_csv(response_file, index=True, index_label="idx")
    print(f"[refetch] recovered {recovered} / {len(targets)} "
          f"(still failing: {still_failed}); CSV updated in place")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="Model alias key (must exist in pipelines.models.model_mapping)")
    p.add_argument("--response-file", default=None,
                   help="Override path; defaults to responses/<alias>/mmlu-pro/"
                        "mmlu-pro_all_v1_<alias>_response.csv")
    p.add_argument("--batch", type=int, default=1,
                   help="Concurrent refetch workers (matches run_benchmark.sh --batch)")
    args = p.parse_args()

    response_file = args.response_file or _default_response_file(args.model)
    refetch(args.model, response_file, args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
