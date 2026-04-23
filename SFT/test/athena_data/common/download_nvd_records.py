import json
import time
import argparse
from datetime import datetime, timedelta

import os
import requests
import yaml

from pipelines.data_loader import load_api_key
from athena_data.common.utils import setup_logger

# -------------------------
# Config & API key helpers
# -------------------------

def load_config(path='config.yaml'):
    """Return configuration dictionary loaded from YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# -------------------------
# HTTP fetch (JSON API 2.0)
# -------------------------

def fetch_cves_json(start_dt: datetime, end_dt: datetime, api_key: str, logger, delay_s: float):
    """
    Fetch all CVEs from NVD JSON API between start_dt (inclusive) and end_dt (exclusive).
    Handles pagination. Sleeps between page requests to respect rate limits.
    """
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
        "pubStartDate": start_dt.isoformat() + "Z",
        "pubEndDate":   end_dt.isoformat() + "Z",
        "resultsPerPage": 2000,
        "startIndex": 0,
    }
    headers = {"apiKey": api_key}
    all_items = []

    while True:
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        all_items.extend(vulns)

        total = data.get("totalResults", 0)
        got = len(vulns)
        logger.debug(f"Fetched page at startIndex={params['startIndex']}, got {got}, total={total}")

        # Pagination exit
        if params["startIndex"] + got >= total:
            break

        params["startIndex"] += got

        # polite delay between page requests
        time.sleep(delay_s)

    return all_items

# -------------------------
# Parsing helpers
# -------------------------

def is_nvd_source(source_str: str) -> bool:
    """Classify metric source as NVD if it contains nist/nvd domain; otherwise treat as CNA."""
    if not source_str:
        return False
    s = source_str.lower()
    return ("nist.gov" in s) or ("@nist" in s) or ("nvd" in s)

def pick_better(existing, candidate):
    """
    Prefer Primary over Secondary. If existing is None, take candidate.
    Each metric dict must include 'type' with values like 'Primary' or 'Secondary'.
    """
    if existing is None:
        return candidate
    if (existing.get("type", "").lower() != "primary") and (candidate.get("type", "").lower() == "primary"):
        return candidate
    # otherwise keep existing
    return existing

def extract_version_metrics(metric_list):
    """
    From a list like cvssMetricV31 or cvssMetricV40, pick best NVD and best CNA entries.
    Return dictionaries with vector/score/severity and keep source/type for decision making.
    """
    best = {"nvd": None, "cna": None}
    for m in metric_list or []:
        src = m.get("source", "")
        typ = m.get("type", "")
        cvss = m.get("cvssData", {}) or {}
        entry = {
            "type": typ,
            "source": src,
            "vector": cvss.get("vectorString", "") or "",
            "score": cvss.get("baseScore"),
            "severity": cvss.get("baseSeverity", "") or "",
        }
        key = "nvd" if is_nvd_source(src) else "cna"
        best[key] = pick_better(best[key], entry)
    # Reduce to simple dicts (or blanks)
    def to_simple(e):
        if not e:
            return {"vector": "", "score": None, "severity": ""}
        return {"vector": e["vector"], "score": e["score"], "severity": e["severity"]}
    return to_simple(best["nvd"]), to_simple(best["cna"])

def get_english_description(cve_obj):
    for d in cve_obj.get("descriptions", []):
        if d.get("lang") == "en":
            return d.get("value", "")
    # fallback to first description if any
    if cve_obj.get("descriptions"):
        return cve_obj["descriptions"][0].get("value", "")
    return ""

def normalize_cwes(cve_obj):
    out = []
    for w in cve_obj.get("weaknesses", []):
        for d in w.get("description", []):
            val = d.get("value")
            if not val:
                continue
            # keep only canonical tokens like CWE-284, NVD-CWE-noinfo, etc.
            # avoid embedding dicts or language tags
            out.append(str(val))
    return out

def parse_cve_json(item, logger):
    cve = item.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        logger.warning(f"Skipping entry without CVE id: {item!r}")
        return None

    description = get_english_description(cve)
    cwe_ids = normalize_cwes(cve)

    metrics = cve.get("metrics", {})  # IMPORTANT: metrics is inside cve (not top-level)!
    # Extract v4.0 and v3.1 separately, classifying NVD vs CNA and preferring Primary
    nvd_v4, cna_v4 = extract_version_metrics(metrics.get("cvssMetricV40"))
    nvd_v31, cna_v31 = extract_version_metrics(metrics.get("cvssMetricV31"))

    return {
        "cve_id": cve_id,
        "published_date": cve.get("published"),
        "last_modified_date": cve.get("lastModified"),
        "description": description,
        "cwe_ids": cwe_ids,

        # NVD metrics
        "nvd_cvss_v4_vector": nvd_v4["vector"],
        "nvd_cvss_v4_score": nvd_v4["score"],
        "nvd_cvss_v4_severity": nvd_v4["severity"],
        "nvd_cvss_v31_vector": nvd_v31["vector"],
        "nvd_cvss_v31_score": nvd_v31["score"],
        "nvd_cvss_v31_severity": nvd_v31["severity"],

        # CNA metrics
        "cna_cvss_v4_vector": cna_v4["vector"],
        "cna_cvss_v4_score": cna_v4["score"],
        "cna_cvss_v4_severity": cna_v4["severity"],
        "cna_cvss_v31_vector": cna_v31["vector"],
        "cna_cvss_v31_score": cna_v31["score"],
        "cna_cvss_v31_severity": cna_v31["severity"],
    }

# -------------------------
# Output & dates
# -------------------------

def write_jsonl(records, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def resolve_dates(cfg):
    defaults = {"nvd_start_date": "2025-01-01", "nvd_end_date": "2025-07-31"}
    common = cfg.get("COMMON", {})
    start_val = common.get("nvd_start_date", defaults["nvd_start_date"])
    end_val   = common.get("nvd_end_date", defaults["nvd_end_date"])
    # Make sure they are strings for fromisoformat
    if not isinstance(start_val, str):
        start_val = str(start_val)
    if not isinstance(end_val, str):
        end_val = str(end_val)
    return datetime.fromisoformat(start_val), datetime.fromisoformat(end_val)

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch CVE data from NVD API (JSON v2.0)")
    parser.add_argument("--config", type=str, default="athena_data/config.yaml")
    args = parser.parse_args()

    logger = setup_logger("nvd")
    cfg = load_config(args.config)
    api_key = load_api_key("NVD_API_KEY")

    common = cfg.get("COMMON", {})
    output_dir = common.get("nvd_data_dir", "data/processed/nvd/")
    os.makedirs(output_dir, exist_ok=True)

    # Delay tuning: with API key you can go faster; keep conservative by default.
    # Override with env var NVD_DELAY_S if you want.
    has_key = bool(api_key)
    delay_s = float(os.getenv("NVD_DELAY_S", 1.0 if has_key else 6.0))

    start, end = resolve_dates(cfg)
    current = start
    while current <= end:
        next_day = current + timedelta(days=1)
        outfile = os.path.join(output_dir, f"nvd_data_{current.date()}.jsonl")

        if os.path.exists(outfile):
            logger.info(f"Skipping {current.date()} (already exists)")
        else:
            logger.info(f"Fetching CVEs for {current.date()} …")
            try:
                items = fetch_cves_json(current, next_day, api_key, logger, delay_s)
                records = []
                for it in items:
                    rec = parse_cve_json(it, logger)
                    if rec:
                        records.append(rec)
                logger.info(f"  Processed {len(records)} valid CVEs out of {len(items)} fetched")
                if records:
                    write_jsonl(records, outfile)
                    logger.info(f"  Saved {len(records)} CVEs to {outfile}")
                else:
                    logger.info(f"  No valid CVE records to write for {current.date()}")
            except Exception as e:
                logger.error(f"  Error fetching {current.date()}: {e}")

        # polite delay between day-requests
        time.sleep(delay_s)
        current = next_day

if __name__ == "__main__":
    main()
