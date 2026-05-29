import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

from pipelines.data_loader import load_yaml


def jsonl_read(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def jsonl_write(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_prompt_hash(prompt: str) -> str:
    data = (prompt or "").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def ensure_prompt_hash(records: List[dict]) -> Tuple[List[dict], int]:
    updated = 0
    out: List[dict] = []
    for rec in records:
        if "prompt_hash" in rec and rec.get("prompt_hash"):
            out.append(rec)
            continue
        prompt = rec.get("prompt", "")
        new_rec = dict(rec)
        new_rec["prompt_hash"] = compute_prompt_hash(prompt)
        out.append(new_rec)
        updated += 1
    return out, updated


def verify_unique_prompt_hash(records: List[dict], task: str, src: Path) -> None:
    seen: Dict[str, int] = {}
    dups: List[Tuple[str, int, int]] = []  # (hash, first_idx, dup_idx)
    for idx, rec in enumerate(records):
        h = rec.get("prompt_hash")
        if not h:
            continue
        if h in seen:
            dups.append((h, seen[h], idx))
        else:
            seen[h] = idx
    if dups:
        details = "; ".join([f"{h} at {a},{b}" for h, a, b in dups[:10]])
        raise AssertionError(
            f"Duplicate prompt_hash detected in task {task} ({src}). Examples: {details}. "
            "Hashes are computed from 'prompt'; duplicate prompts within a task are not allowed."
        )


def parse_task_frac(overrides: List[str]) -> Dict[str, float]:
    """Parse CLI task fraction overrides provided as TASK=VALUE strings."""
    parsed: Dict[str, float] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid --task-frac override '{item}'. Expected format TASK=FRACTION."
            )
        task, value = item.split("=", 1)
        task = task.strip().upper()
        try:
            parsed[task] = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid fraction value in --task-frac override '{item}'."
            ) from exc
    return parsed


def sample_indices(n: int, frac: float, fixed_size: int | None, seed: int) -> List[int]:
    import random

    if fixed_size is not None:
        k = min(fixed_size, n)
    else:
        k = max(1, int(math.ceil(frac * n)))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def main():
    parser = argparse.ArgumentParser(description="Create mini benchmark subsets and add prompt_hash to originals.")
    parser.add_argument("--config", default="athena_data/tasks_config.yaml", help="Path to eval config for task files")
    parser.add_argument("--out-dir", default="benchmark_data_mini", help="Output directory for mini subsets")
    parser.add_argument("--frac", type=float, default=0.10, help="Sampling fraction for non-TAA tasks (default 0.10)")
    parser.add_argument("--taa-size", type=int, default=50, help="Fixed sample size for TAA (default 25)")
    parser.add_argument("--seed", type=int, default=14, help="Random seed")
    parser.add_argument(
        "--task-frac",
        action="append",
        default=[],
        help="Override sampling fraction for a task (format TASK=FRACTION).",
    )
    parser.add_argument("--update-originals", action="store_true", help="Write prompt_hash back into original benchmark files")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tasks: Dict[str, str] = cfg.get("tasks", {})
    frac_overrides = {"RMS": 0.2, "ATE": 0.2}
    frac_overrides.update(parse_task_frac(args.task_frac))
    out_dir = Path(args.out_dir)

    for task, rel_path in tasks.items():
        src = Path(rel_path)
        if not src.exists():
            print(f"[mini] Skip {task}: missing dataset {src}")
            continue

        print(f"[mini] Processing {task}: {src}")
        records = jsonl_read(src)
        records_h, n_added = ensure_prompt_hash(records)
        # Enforce uniqueness within each task file
        verify_unique_prompt_hash(records_h, task, src)

        if args.update_originals and n_added > 0:
            tmp_path = src.with_suffix(".with_hash.tmp.jsonl")
            jsonl_write(tmp_path, records_h)
            os.replace(tmp_path, src)
            print(f"[mini] Wrote prompt_hash into {src} (added {n_added})")
        else:
            print(f"[mini] prompt_hash present or update skipped (added {n_added})")

        # Determine sample size
        task_upper = task.upper()
        fixed_size = args.taa_size if task_upper == "TAA" else None
        frac = frac_overrides.get(task_upper, args.frac)
        idx = sample_indices(len(records_h), frac, fixed_size, seed=args.seed)

        mini = [records_h[i] for i in idx]
        out_path = out_dir / Path(rel_path).name
        jsonl_write(out_path, mini)
        print(
            f"[mini] Wrote subset for {task}: {out_path} (n={len(mini)} of {len(records_h)})"
        )


if __name__ == "__main__":
    main()