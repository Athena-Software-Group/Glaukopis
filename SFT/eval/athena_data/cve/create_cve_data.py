import os
import re
import json
import glob
import argparse
import random
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from athena_data.common.utils import setup_logger
from pipelines.data_loader import load_yaml, parse_date, within_inclusive

def normalize_desc(text: Any) -> str:
    if isinstance(text, dict):
        text = text.get("value") or text.get("description") or ""
    text = text or ""
    return " ".join(str(text).split())

CWE_PAT = re.compile(r"\bCWE-\d+[A-Za-z0-9-]*\b")

def normalize_cwes(raw: Any) -> List[str]:
    out = []
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, dict):
                v = x.get("value") or x.get("id") or ""
                out.extend(CWE_PAT.findall(str(v)))
            else:
                out.extend(CWE_PAT.findall(str(x)))
    elif isinstance(raw, dict):
        v = raw.get("value") or raw.get("id") or ""
        out.extend(CWE_PAT.findall(str(v)))
    elif isinstance(raw, str):
        out.extend(CWE_PAT.findall(raw))
    # unique, keep order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def choose_latest(existing: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    p1 = parse_date(existing.get("published_date"))
    p2 = parse_date(candidate.get("published_date"))
    if p1 and p2:
        if p2 > p1:
            return candidate
        if p2 < p1:
            return existing
    m1 = parse_date(existing.get("last_modified_date"))
    m2 = parse_date(candidate.get("last_modified_date"))
    if m1 and m2:
        return candidate if m2 >= m1 else existing
    return candidate

def pick_cvss_fields(rec: Dict[str, Any], version: str, nvd_only: bool) -> Optional[Tuple[str, Optional[float], str, str]]:
    version_key = "v31" if version.lower() in ("v31", "3.1", "cvss3.1") else "v40"
    nvd_vec = rec.get(f"nvd_cvss_{version_key}_vector") or ""
    nvd_score = rec.get(f"nvd_cvss_{version_key}_score")
    nvd_sev = rec.get(f"nvd_cvss_{version_key}_severity") or ""
    cna_vec = rec.get(f"cna_cvss_{version_key}_vector") or ""
    cna_score = rec.get(f"cna_cvss_{version_key}_score")
    cna_sev = rec.get(f"cna_cvss_{version_key}_severity") or ""
    if nvd_only:
        if nvd_vec:
            return nvd_vec, nvd_score, nvd_sev, "NVD"
        return None
    if nvd_vec:
        return nvd_vec, nvd_score, nvd_sev, "NVD"
    if cna_vec:
        return cna_vec, cna_score, cna_sev, "CNA"
    return None

def load_all_nvd_records(nvd_dir: str, start: datetime, end: datetime, logger) -> List[Dict[str, Any]]:
    import itertools
    paths = sorted(glob.glob(os.path.join(nvd_dir, "*.jsonl")))
    logger.info(f"Scanning {len(paths)} JSONL files in {nvd_dir}")
    records, total_lines, parsed_lines = [], 0, 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    parsed_lines += 1
                except Exception:
                    logger.debug(f"Skipping malformed line in {path}")
                    continue
                if within_inclusive(obj.get("published_date"), start, end):
                    records.append(obj)
    logger.info(f"Total lines read: {total_lines} | Parsed: {parsed_lines} | Within date window: {len(records)}")
    return records

# ---------- NEW: stats & bins ----------

def word_count(s: str) -> int:
    return len((s or "").split())

def log_bins(counts: List[int], logger, label: str):
    # bins: <10, 10–19, 20–29, 30–39, 40–49, ≥50
    b = [0, 0, 0, 0, 0, 0]
    for c in counts:
        if c < 10: b[0] += 1
        elif c < 20: b[1] += 1
        elif c < 30: b[2] += 1
        elif c < 40: b[3] += 1
        elif c < 50: b[4] += 1
        else: b[5] += 1
    logger.info(f"{label} bins (words): <10={b[0]}, 10-19={b[1]}, 20-29={b[2]}, 30-39={b[3]}, 40-49={b[4]}, >=50={b[5]}")

def log_desc_stats(records: List[Dict[str, Any]], logger, label: str):
    counts = []
    for r in records:
        desc = normalize_desc(r.get("description"))
        if desc:
            counts.append(word_count(desc))
    n = len(counts)
    if n == 0:
        logger.info(f"{label} stats: n=0 (no descriptions).")
        return
    mean_val = statistics.mean(counts)
    stdev_val = statistics.stdev(counts) if n > 1 else 0.0
    logger.info(f"{label} stats: n={n}, mean={mean_val:.2f}, stdev={stdev_val:.2f}")
    log_bins(counts, logger, label)

def filter_by_min_words(records: List[Dict[str, Any]], min_words: int, logger, label: str) -> List[Dict[str, Any]]:
    before = len(records)
    out = []
    for r in records:
        if word_count(normalize_desc(r.get("description"))) >= min_words:
            out.append(r)
    removed = before - len(out)
    logger.info(f"{label} min_words={min_words}: kept {len(out)} / {before} (removed {removed})")
    # log post-filter stats
    log_desc_stats(out, logger, f"{label} post-filter")
    return out

# ---------- existing filters ----------

def dedupe_by_description(records: List[Dict[str, Any]], logger) -> List[Dict[str, Any]]:
    before = len(records)
    by_desc: Dict[str, Dict[str, Any]] = {}
    for r in records:
        desc_key = normalize_desc(r.get("description"))
        if not desc_key:
            continue
        if desc_key not in by_desc:
            by_desc[desc_key] = r
        else:
            by_desc[desc_key] = choose_latest(by_desc[desc_key], r)
    out = list(by_desc.values())
    removed = before - len(out)
    logger.info(f"Deduplicated by description: kept {len(out)} / {before} (removed {removed})")
    return out

def filter_single_cwe(records: List[Dict[str, Any]], logger) -> List[Dict[str, Any]]:
    before = len(records)
    out = []
    for r in records:
        cwes = normalize_cwes(r.get("cwe_ids"))
        if len(cwes) == 1:
            r["_one_cwe"] = cwes[0]
            out.append(r)
    logger.info(f"Single-CWE filter: kept {len(out)} / {before}")
    return out

# ---------- task builders ----------

def build_rcm_tasks(records: List[Dict[str, Any]], prompt_template: str, k: int, logger, seed: int = 1337) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = records[:]
    if len(pool) > k:
        pool = rng.sample(pool, k)
    tasks = []
    for r in pool:
        desc = normalize_desc(r.get("description"))
        tasks.append({
            "cve_id": r.get("cve_id"),
            "description": desc,
            "prompt": prompt_template.format(desc),
            "answer": r.get("_one_cwe"),
        })
    logger.info(f"RCM tasks sampled: {len(tasks)} (requested {k}) | seed={seed}")
    return tasks

def build_cvss_tasks(records: List[Dict[str, Any]], prompt_template: str, k: int,
                     cvss_version: str, use_nvd_only: bool, logger,
                     seed: int = 1337) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    eligible, nvd_pick, cna_pick, none_pick = [], 0, 0, 0
    for r in records:
        picked = pick_cvss_fields(r, cvss_version, use_nvd_only)
        if not picked:
            none_pick += 1
            continue
        vec, score, sev, src = picked
        if src == "NVD":
            nvd_pick += 1
        else:
            cna_pick += 1
        desc = normalize_desc(r.get("description"))
        eligible.append({
            "cve_id": r.get("cve_id"),
            "description": desc,
            "prompt": prompt_template.format(desc),
            "answer": vec,
            "vector_score": score,
            "severity": sev,
        })
    logger.info(f"CVSS eligibility (version={cvss_version}, nvd_only={use_nvd_only}): "
                f"NVD={nvd_pick}, CNA={cna_pick}, None={none_pick}, Eligible total={len(eligible)}")
    if len(eligible) > k:
        eligible = rng.sample(eligible, k)
    logger.info(f"CVSS tasks sampled: {len(eligible)} (requested {k}) | seed={seed}")
    return eligible

def write_jsonl(path: str, rows: List[Dict[str, Any]], logger) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(rows)} rows -> {path}")

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Build RCM & CVSS benchmark JSONL with min description filter + bins")
    parser.add_argument("--config", default="athena_data/config.yaml")
    args = parser.parse_args()

    logger = setup_logger("task-cve")
    cfg = load_yaml(args.config)

    # COMMON
    prompts_path = cfg["COMMON"]["task_prompt_file"]
    nvd_dir = cfg["COMMON"]["nvd_data_dir"]
    start = datetime.fromisoformat(str(cfg["COMMON"]["nvd_start_date"]))
    end = datetime.fromisoformat(str(cfg["COMMON"]["nvd_end_date"]))
    total_days = (end - start).days + 1

    logger.info("=== Benchmark Task Builder (task-cve) ===")
    logger.info(f"Date window: {start.date()} .. {end.date()} (inclusive) | total_days={total_days}")
    logger.info(f"NVD directory: {nvd_dir}")
    logger.info(f"Prompts file: {prompts_path}")

    # Sections + defaults
    rcm_cfg = cfg.get("RCM", {})
    cvss_cfg = cfg.get("CVSS", {})
    rcm_min_words = int(rcm_cfg.get("min_description_words", 20))
    cvss_min_words = int(cvss_cfg.get("min_description_words", 30))

    # Load prompts
    prompts = load_yaml(prompts_path)
    rcm_prompt = prompts["RCM"]
    cvss_prompt = prompts["CVSS"]

    # Load & initial counts
    all_recs = load_all_nvd_records(nvd_dir, start, end, logger)
    logger.info(f"Total CVE records loaded within window: {len(all_recs)}")

    # Single-CWE filter
    single_cwe = filter_single_cwe(all_recs, logger)

    # Deduplicate by description
    deduped = dedupe_by_description(single_cwe, logger)

    # Log stats/bins after dedupe (pre min-words) so you can inspect distribution
    log_desc_stats(deduped, logger, "After dedupe (pre min-words)")

    # Apply min words per task (separate pools)
    rcm_pool = filter_by_min_words(deduped, rcm_min_words, logger, "RCM")
    cvss_pool = filter_by_min_words(deduped, cvss_min_words, logger, "CVSS")

    # Add date suffix (YYYYMMDD)
    date_tag = datetime.now().strftime("%Y%m%d")

    # Build RCM
    rcm_k = int(rcm_cfg.get("num_questions", 0))
    rcm_out = Path(rcm_cfg.get("output_path", "benchmark_data/athena_bench/athena-cti-rcm.jsonl"))
    rcm_out = rcm_out.with_name(f"{rcm_out.stem}_{date_tag}{rcm_out.suffix}")
    rcm_seed = int(rcm_cfg.get("seed", 1337))
    rcm_tasks = build_rcm_tasks(rcm_pool, rcm_prompt, rcm_k, logger, seed=rcm_seed)
    write_jsonl(rcm_out, rcm_tasks, logger)

    # Build CVSS (VSP)
    cvss_k = int(cvss_cfg.get("num_questions", 0))
    cvss_out = Path(cvss_cfg.get("output_path", "benchmark_data/athena_bench/athena-cti-vsp.jsonl"))
    cvss_out = cvss_out.with_name(f"{cvss_out.stem}_{date_tag}{cvss_out.suffix}")
    cvss_version = str(cvss_cfg.get("cvss_version", "v31"))
    use_nvd_only = bool(cvss_cfg.get("use_nvd_score_only", True))
    cvss_seed = int(cvss_cfg.get("seed", 1337))
    cvss_tasks = build_cvss_tasks(cvss_pool, cvss_prompt, cvss_k, cvss_version, use_nvd_only, logger, seed=cvss_seed)
    write_jsonl(cvss_out, cvss_tasks, logger)

    # Summary
    logger.info("--- Summary ---")
    logger.info(f"Loaded records: {len(all_recs)}")
    logger.info(f"After single-CWE: {len(single_cwe)}")
    logger.info(f"After dedupe (pre min-words): {len(deduped)}")
    logger.info(f"RCM pool after min_words={rcm_min_words}: {len(rcm_pool)}")
    logger.info(f"CVSS pool after min_words={cvss_min_words}: {len(cvss_pool)}")
    logger.info(f"RCM  -> wrote {len(rcm_tasks)} to {rcm_out}")
    logger.info(f"CVSS -> wrote {len(cvss_tasks)} to {cvss_out}")
    logger.info("=== Done ===")

if __name__ == "__main__":
    main()
