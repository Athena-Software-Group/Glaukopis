import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from openai import OpenAI

from pipelines.data_loader import (
    load_api_key,
    load_yaml,
    parse_date,
    within_inclusive,
)


MIN_NEWSPAPER_CHARS = 200
HTML_TIMEOUT = 30
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
SYSTEM_PROMPT_ANON = (
    "You are a cybersecurity assistant. The following report text is about a specific threat actor.\n"
    "Extract only factual details about their activities as a bullet-point list.\n"
    "Rules:\n"
    "- Replace the primary threat actor's name with 'they'.\n"
    "- If any other threat actor is mentioned, OMIT the entire sentence containing that name.\n"
    "- Facts can include: attack tactics, techniques and procedures (TTPs), targets, motivations, industries, origin, date, etc.\n"
    "- Do not reference the original source, URL, or include any actor names other than the anonymized 'they'.\n"
    "- Output only the bullet list, no extra commentary."
)

USER_PROMPT_TMPL = "Report:\n{report_text}"

# ---------
# Helpers
# ---------

def word_count(text: str) -> int:
    return len((text or "").split())

def extract_article_text(url: str) -> Optional[str]:
    # 1) newspaper3k
    try:
        art = Article(url, keep_article_html=False, fetch_images=False)
        art.download()
        art.parse()
        text = (art.text or "").strip()
        if len(text) >= MIN_NEWSPAPER_CHARS:
            return text
    except Exception:
        pass
    # 2) requests + BeautifulSoup
    try:
        resp = requests.get(url, timeout=25, headers=HEADERS)
        if resp.status_code != 200 or not resp.text:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        # try common content selectors
        candidates = []
        for selector in ["article", "main", "div#content", "div.post", "div.article", "section", "div.entry-content"]:
            for node in soup.select(selector):
                txt = node.get_text(separator="\n", strip=True)
                if txt and len(txt) >= MIN_NEWSPAPER_CHARS:
                    candidates.append(txt)
        if not candidates:
            page = soup.get_text(separator="\n", strip=True)
            return page if len(page) >= MIN_NEWSPAPER_CHARS else None
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    except Exception:
        return None
    



# ---------------------------
# OpenAI
# ---------------------------
def openai_gpt5_generate(
    client, 
    model,
    system_prompt,
    user_prompt,
    *,
    reasoning_effort,   # minimal|low|medium|high
    verbosity,              # low|medium|high
    timeout_s,                # request timeout
    max_retries,                # retry count on timeouts/5xx/429
    backoff_base,           # exponential backoff base
) -> str:
    """
    GPT-5 Responses API with retries/backoff.
    Returns text or 'ERROR: ...' on failure after retries.
    """
    combined_input = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,                           # "gpt-5", "gpt-5-mini", "gpt-5-nano"
                input=combined_input,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": verbosity},
                timeout=timeout_s,
            )
            out = (getattr(resp, "output_text", None) or "").strip()
            return out if out else "ERROR: empty response"
        except Exception as e:
            err = str(e)
            transient = any(x in err for x in (
                "timed out", "Timeout", "429", "Rate limit", "502", "503", "504",
                "Temporary", "Connection reset", "RemoteDisconnected"
            ))
            if attempt < max_retries and transient:
                sleep_s = backoff_base ** attempt + (0.1 * attempt)
                time.sleep(sleep_s)
                continue
            return f"ERROR: {err}"
        
# ---------------------------
# Main builder
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Build TAA benchmark JSONL dataset")
    parser.add_argument("--config", default= "athena_data/config.yaml")
    args = parser.parse_args()
    
    cfg = load_yaml(args.config)
    taa_cfg = cfg.get("TAA", {})

    start_date = parse_date(taa_cfg.get("start_date"))
    end_date   = parse_date(taa_cfg.get("end_date"))

    if not (start_date and end_date):
        raise ValueError("TAA.start_date and TAA.end_date are required in the YAML (ISO-8601 recommended).")

    min_words   = int(taa_cfg.get("min_description_words", 30))
    processed_path = str(taa_cfg.get("processed_path", "data/processed/taa/athena-cti-taa.jsonl"))
    urls_csv    = str(taa_cfg.get("urls_csv", "urls.csv"))
    api_key = load_api_key("OPENAI_API_KEY")
    TEXT_MAX_CHARS = int(taa_cfg.get("text_max_chars", 12000))
    model            = str(taa_cfg.get("model", "gpt-5"))         
    reasoning_effort = str(taa_cfg.get("reasoning_effort", "minimal"))
    verbosity        = str(taa_cfg.get("verbosity", "low"))
    sleep_sec        = float(taa_cfg.get("sleep_between_calls", 1.5))
    timeout_s       =  int(taa_cfg.get("openai_timeout_s", 120))
    max_retries     =int(taa_cfg.get("openai_max_retries", 4))



    client = OpenAI(api_key=api_key)

    # ---- Input URLs ----
    df = pd.read_csv(urls_csv)
    if not set(["URL", "Timestamp"]).issubset(df.columns):
        raise ValueError("Input URLs CSV must contain at least: 'URL', 'Timestamp'. Optional: 'GT'.")
    
    rows = []
    for i, r in df.iterrows():
        url = str(r.get("URL", "")).strip()
        ts  = r.get("Timestamp", "")
        gt  = r.get("GT", "")
        print('-----')
        print(ts)

        if not url:
            continue
        if not within_inclusive(str(ts), start_date, end_date):
            continue

        print(f"[{i}] Fetch: {url}")
        text = extract_article_text(url)
        if not text:
            print("   - skipped: extraction failed")
            continue

        if word_count(text) < min_words:
            print(f"   - skipped: word_count<{min_words}")
            continue

        # Trim for model context
        text_trimmed = text[:TEXT_MAX_CHARS]
        user_prompt = USER_PROMPT_TMPL.format(report_text=text_trimmed)

        # Call OpenAI
        anon = openai_gpt5_generate(
        client=client,
        model=model,
        system_prompt=SYSTEM_PROMPT_ANON,
        user_prompt=user_prompt,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_base = 2.0  
)
        if anon.startswith("ERROR:"):
            print(f"   - OpenAI error: {anon[:120]}")
            continue

        ts_iso = parse_date(str(ts))
        ts_out = ts_iso.isoformat() if ts_iso else str(ts)

        rows.append({
            "url": url,
            "time_stamp": ts_out,
            "ground_truth": gt,
            "anonymized_threat_actions": anon
        })

        time.sleep(sleep_sec)

    # ---- Write JSONL ----
    out_path = Path(processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    main() 