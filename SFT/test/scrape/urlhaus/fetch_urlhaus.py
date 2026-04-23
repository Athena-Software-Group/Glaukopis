import os
import json
import requests
from datetime import datetime
import csv

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
URLHAUS_DIR = os.path.join(DATA_DIR, "urlhaus")
os.makedirs(URLHAUS_DIR, exist_ok=True)

# --- URLhaus endpoint ---
API_URL = "https://urlhaus.abuse.ch/downloads/json_recent/"

# --- Functions ---
def fetch_urlhaus_recent():
    """Fetch recent URLhaus data."""
    try:
        resp = requests.get(API_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Save raw JSON
        json_path = os.path.join(URLHAUS_DIR, "urlhaus_full.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved raw JSON to {json_path}")
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URLhaus data: {e}")
        return None

def save_urls_csv(data):
    """Save CSV containing only URLs and reference='malicious'."""
    if not data:
        print("No data to save.")
        return None

    # Add timestamp to filename to avoid overwriting
    date_tag = datetime.now().strftime("%Y%m%d")
    filename = f"urlhaus_full_ioc_{date_tag}.csv"
    path = os.path.join(URLHAUS_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["url", "reference"])

        for key, entries in data.items():
            for entry in entries:
                url = entry.get("url")
                if url:
                    writer.writerow([url, "malicious"])

    print(f"Saved CSV with malicious URLs to {path}")
    return path

# --- Main ---
if __name__ == "__main__":
    data = fetch_urlhaus_recent()
    if data:
        save_urls_csv(data)
