import argparse
import json
import re
import time
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

from pipelines.data_loader import load_api_key, load_yaml


def read_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt[:max_chars]


def call_openai_responses(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    reasoning_effort: str,
    verbosity: str,
    timeout_s: int,
    max_retries: int,
    backoff_base: float = 2.0,
) -> str:
    combined_input = f"[SYSTEM]\nYou are a cybersecurity assistant. Return only TSV rows.\n\n[USER]\n{prompt}"
    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=combined_input,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": verbosity},
                timeout=timeout_s,
            )
            out = (getattr(resp, "output_text", None) or "").strip()
            return out
        except Exception as e:
            transient = any(x in str(e) for x in (
                "timed out", "Timeout", "429", "Rate limit", "502", "503", "504",
                "Temporary", "Connection reset", "RemoteDisconnected"
            ))
            if attempt < max_retries and transient:
                time.sleep(backoff_base ** attempt)
                continue
            return f"ERROR: {e}"


def _strip_label(s: str) -> str:
    return re.sub(r"^[A-E][\).]\s*", "", s or "").strip()


def _format_mcq_block(question: str, a: str, b: str, c: str, d: str, e: str) -> str:
    a = _strip_label(a)
    b = _strip_label(b)
    c = _strip_label(c)
    d = _strip_label(d)
    e = _strip_label(e)
    return (
        f"Question: {question}\n"
        f"A) {a}\n"
        f"B) {b}\n"
        f"C) {c}\n"
        f"D) {d}\n"
        f"E) {e}"
    )


def hydrate_prompts_only(jsonl_path: Path, mcq_eval_tmpl: str) -> int:
    updated = 0
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with jsonl_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                fout.write(line + "\n")
                continue
            if not rec.get("prompt"):
                q = rec.get("question", "")
                a = rec.get("option_a", "")
                b = rec.get("option_b", "")
                c = rec.get("option_c", "")
                d = rec.get("option_d", "")
                e = rec.get("option_e", "")
                if all([q, a, b, c, d, e]):
                    formatted = _format_mcq_block(q, a, b, c, d, e)
                    rec["prompt"] = mcq_eval_tmpl.format(formatted)
                    if not rec.get("answer") and rec.get("correct_answer"):
                        ans = str(rec.get("correct_answer", "")).strip().upper()[:1]
                        if ans in {"A", "B", "C", "D", "E"}:
                            rec["answer"] = ans
                    updated += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(jsonl_path)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCQ TSV from plan using OpenAI")
    parser.add_argument("--config", default="athena_data/config.yaml")
    parser.add_argument("--hydrate-only", action="store_true", help="Insert prompts into existing JSONL and exit")
    parser.add_argument("--rebuild", action="store_true", help="Force regenerate via API even if JSONL exists")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    prompts = load_yaml(cfg["COMMON"]["task_prompt_file"])
    mcq_cfg = cfg.get("MCQ", {})

    plan_path = Path(mcq_cfg.get("plan_path", "data/processed/mcq/mcq_plan.tsv"))
    processed_root = Path(mcq_cfg.get("processed_root", "data/processed/mcq"))
    out_path = Path(mcq_cfg.get("output_path", "benchmark_data/athena_bench/athena-cti-mcq.jsonl"))
     # Add date suffix (YYYYMMDD)
    date_tag = datetime.now().strftime("%Y%m%d")
    base, ext = out_path.stem, out_path.suffix  # Split name and extension safely
    out_path = out_path.with_name(f"{base}_{date_tag}{ext}")


    model = str(mcq_cfg.get("model", "gpt-5"))
    reasoning_effort = str(mcq_cfg.get("reasoning_effort", "medium"))
    verbosity = str(mcq_cfg.get("verbosity", "low"))
    timeout_s = int(mcq_cfg.get("openai_timeout_s", 120))
    max_retries = int(mcq_cfg.get("openai_max_retries", 4))
    sleep_between = float(mcq_cfg.get("sleep_between_calls", 1.0))
    text_max_chars = int(mcq_cfg.get("text_max_chars", 12000))

    api_key = load_api_key("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Load prompt template
    mcq_prompt_tmpl = prompts["MCQ"]
    mcq_eval_tmpl = prompts.get("MCQ_EVAL", "{}")

    # If JSONL exists and not forced rebuild, hydrate prompts and exit
    if (args.hydrate_only or (out_path.exists() and not args.rebuild)):
        print(f"Using existing MCQ JSONL at {out_path}: inserting prompts where missing...")
        count = hydrate_prompts_only(out_path, mcq_eval_tmpl)
        print(f"Inserted prompts for {count} rows. Done.")
        return

    # Read plan
    df = pd.read_csv(plan_path, sep="\t")
    required_cols = {"processed_path", "question_count"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        raise ValueError(f"MCQ plan file is missing required columns: {missing}")

    print("Generating MCQs via API and writing JSONL...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_target = int(df["question_count"].fillna(0).sum())
    # Truncate/create output file upfront so results are appended per-row
    with out_path.open("w", encoding="utf-8") as _f:
        pass
    rows_written = 0
    pbar = tqdm(df.itertuples(index=False), total=len(df), desc="MCQ")
    for row in pbar:
        processed_rel = str(getattr(row, "processed_path"))
        qcount = int(getattr(row, "question_count", 0) or 0)
        if qcount <= 0:
            continue
        # Resolve content file under processed_root
        content_path = processed_root / processed_rel
        # Normalize any backslashes in TSV paths
        content_path = Path(str(content_path))
        text = read_text(content_path, text_max_chars)
        if not text:
            continue
        prompt = mcq_prompt_tmpl.format(qcount, text)
        resp = call_openai_responses(
            client,
            model,
            prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        if not resp or resp.startswith("ERROR:"):
            continue
        # Parse TSV lines into JSON objects and append to JSONL
        lines = [ln.strip("\r\n ") for ln in resp.splitlines() if ln.strip()]
        if lines:
            with out_path.open("a", encoding="utf-8") as f:
                for ln in lines:
                    parts = ln.split("\t")
                    if len(parts) != 8:
                        continue
                    question, a_raw, b_raw, c_raw, d_raw, e_raw, correct, expl = parts
                    a = _strip_label(a_raw)
                    b = _strip_label(b_raw)
                    c = _strip_label(c_raw)
                    d = _strip_label(d_raw)
                    e = _strip_label(e_raw)
                    formatted = _format_mcq_block(question, a, b, c, d, e)

                    rec = {
                        "url_id": getattr(row, "url_id", ""),
                        "url": getattr(row, "url", ""),
                        "source_type": getattr(row, "source_type", ""),
                        "processed_path": processed_rel,
                        "raw_path": getattr(row, "raw_path", ""),
                        "char_count": int(getattr(row, "char_count", 0) or 0),
                        "question_count_planned": qcount,
                        "question": question,
                        "option_a": a,
                        "option_b": b,
                        "option_c": c,
                        "option_d": d,
                        "option_e": e,
                        "correct_answer": correct,
                        "explanation": expl,
                        "prompt": mcq_eval_tmpl.format(formatted),
                        "answer": correct,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            rows_written += len(lines)
        time.sleep(sleep_between)

    # Simple progress summary printed to console
    print(f"MCQ target questions (sum of plan): {total_target}")
    print(f"MCQ rows written: {rows_written} -> {out_path}")


if __name__ == "__main__":
    main()
