import os
import pandas as pd
import argparse
from dotenv import load_dotenv
from tqdm import tqdm  
from datetime import datetime, timezone
from pipelines.models import get_single_prediction, model_mapping

# Load .env keys
load_dotenv()

# Paths
DATA_DIR = "benchmark_data/urlhaus"
INPUT_CSV = os.path.join(DATA_DIR, "urlhaus_full_ioc.csv")
date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
OUTPUT_CSV = os.path.join(DATA_DIR, f"urls_benchmark_{date_tag}.csv")


def call_model(malicious_url: str, model_name: str) -> list:
    """Generate a benign URL from a malicious URL and return both (malicious always present)."""
    try:
        prompt = (
            "Given this malicious URL, generate a similar-looking URL that is benign and safe.\n"
            "Return ONLY the URL (no explanation, no extra text).\n\n"
            f"Malicious URL: {malicious_url}"
        )

        if model_name not in model_mapping:
            raise ValueError(f"Model '{model_name}' not found in model_mapping")
        
        #model_key = model_mapping[model_name]

        benign_url = get_single_prediction(prompt, model_name).strip()
        result = [
            {"url": malicious_url, "reference": "malicious"},
            {"url": benign_url, "reference": "benign"} if benign_url else None,
        ]
        return [r for r in result if r]
    except Exception as e:
        print(f"Error generating benign for {malicious_url}: {e}")
        return [{"url": malicious_url, "reference": "malicious"}]


def ensure_output_header(output_csv):
    """Ensure output exists and has header; return set of existing URLs (if any)."""
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        existing_df = pd.read_csv(output_csv)
        existing_urls = set(existing_df['url'].astype(str).tolist())
        return existing_urls
    else:
        pd.DataFrame(columns=["url", "reference"]).to_csv(output_csv, index=False)
        return set()


def append_rows(output_csv, rows_df):
    """Append rows_df to output_csv without header."""
    rows_df.to_csv(output_csv, index=False, mode="a", header=False)


def main(input_csv, output_csv, model_name, num_iocs=None, include_all=False):
    if not os.path.exists(input_csv):
        print(f"Input CSV not found: {input_csv}")
        return
    
    base, ext = os.path.splitext(output_csv)
    output_csv = f"{base}_{date_tag}{ext}"
    print(f"Output CSV will be saved as: {output_csv}")

    df_all = pd.read_csv(input_csv)

    # Shuffle
    df_shuffled = df_all.sample(frac=1.0, random_state=None).reset_index(drop=True)

    # Limit to num_iocs
    df_to_process = df_shuffled.head(num_iocs).reset_index(drop=True) if num_iocs else df_shuffled

    existing_urls = ensure_output_header(output_csv)
    malicious_count, benign_count = 0, 0

    if include_all:
        df_all_mal = pd.DataFrame({
            "url": df_all['url'].astype(str).tolist(),
            "reference": ["malicious"] * len(df_all)
        }).drop_duplicates(subset=['url'])

        if existing_urls:
            df_all_mal = df_all_mal[~df_all_mal['url'].astype(str).isin(existing_urls)]

        if not df_all_mal.empty:
            append_rows(output_csv, df_all_mal)
            existing_urls.update(df_all_mal['url'].astype(str).tolist())
            malicious_count += len(df_all_mal)
            print(f"Wrote {len(df_all_mal)} malicious rows (all input) to {output_csv}")

    # Progress bar
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Processing IOCs", unit="url"):
        url_val = str(row['url'])
        res = call_model(url_val, model_name)

        if include_all:
            res_filtered = [r for r in res if r.get('reference') != 'malicious']
        else:
            res_filtered = res

        if res_filtered:
            df_out = pd.DataFrame(res_filtered)
            append_rows(output_csv, df_out)
            malicious_count += (df_out['reference'] == 'malicious').sum()
            benign_count += (df_out['reference'] == 'benign').sum()

    print(f"\n✅ Completed. Saved results to {output_csv}")
    print(f"Summary -> malicious : {malicious_count}, benign urls generated: {benign_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benign IOCs using LLM.")
    parser.add_argument("--input_csv", default=INPUT_CSV, help="Input CSV with malicious IOCs")
    parser.add_argument("--output_csv", default=OUTPUT_CSV, help="Output CSV with benign IOCs added")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt5)")
    parser.add_argument("--num_iocs",type=int,default=100, help="Number of malicious IOCs to process (after shuffle)")
    parser.add_argument("--include_all", action="store_true", help="Include all malicious rows from input in output before generating benigns")
    args = parser.parse_args()

    main(args.input_csv, args.output_csv, args.model, args.num_iocs, args.include_all)
