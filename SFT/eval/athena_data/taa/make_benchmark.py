import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime   # ← add this at the top with imports

from pipelines.data_loader import load_yaml
from athena_data.common.utils import setup_logger


# ---------- helpers ----------

def load_jsonl(path: Path):
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]], logger) -> None:
    """Write a list of dictionaries to *path* in JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(rows)} rows -> {path}")



def normalize_ts(row: Dict[str, Any]) -> str:
    """Return the timestamp field from *row* if present."""
    return row.get("time_stamp") or row.get("timestamp") or ""


def normalize_desc(row: Dict[str, Any]) -> str:
    """Normalize a text description from *row*."""
    desc = row.get("anonymized_threat_actions") or row.get("description") or ""
    return " ".join(str(desc).split())


# ---------- task builder ----------

def build_taa_tasks(records: List[Dict[str, Any]], prompt_template: str, k: int) -> List[Dict[str, Any]]:
    """Construct benchmark task entries from raw TAA records."""
    out = []
    for r in records:
        desc = normalize_desc(r)
        if not desc:
            continue
        out.append({
            "url": r.get("url", ""),
            "timestamp": normalize_ts(r),
            "description": desc,
            "prompt": prompt_template.format(desc),
            "answer": r.get("ground_truth", ""),
        })
    if k > 0:
        out = out[:k]
    return out


# ---------- main ----------

def main():
    """CLI entry point for building the TAA benchmark dataset."""
    parser = argparse.ArgumentParser(description="Build TAA benchmark JSONL")
    parser.add_argument("--config", default="athena_data/config.yaml")
    args = parser.parse_args()

    logger = setup_logger("task-taa")
    cfg = load_yaml(args.config)

    # COMMON
    prompts_path = cfg["COMMON"]["task_prompt_file"]
    logger.info("=== Benchmark Task Builder (task-taa) ===")
    logger.info(f"Prompts file: {prompts_path}")

    # TAA section
    taa_cfg = cfg.get("TAA", {})
    taa_in = Path(taa_cfg.get("processed_path", "data/processed/taa/athena-cti-taa.jsonl"))
    taa_out = Path(taa_cfg.get("output_path", "benchmark_data/athena_bench/athena-cti-taa.jsonl"))
    date_tag = datetime.now().strftime("%Y%m%d")
    taa_out = taa_out.with_name(f"{taa_out.stem}_{date_tag}{taa_out.suffix}")
    taa_k = int(taa_cfg.get("num_questions", 100))

    # Load prompts
    prompts = load_yaml(prompts_path)
    taa_prompt = prompts["TAA"]

    # Load input records
    records = list(load_jsonl(taa_in))
    logger.info(f"Loaded {len(records)} raw TAA records from {taa_in}")

    # Build tasks
    tasks = build_taa_tasks(records, taa_prompt, taa_k)
    write_jsonl(taa_out, tasks, logger)

    # Summary
    logger.info("--- Summary ---")
    logger.info(f"TAA -> wrote {len(tasks)} to {taa_out}")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
