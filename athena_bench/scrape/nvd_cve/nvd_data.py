#!/usr/bin/env python3

import os
import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberDataCollector:
    def __init__(self, output_dir: str):
        """Initialize the data collector with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.nvd_cve_url = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
        self.timeout = 30
        self.max_retries = 3
        
        # Rate limiting: unauthenticated max 5 requests per 30 seconds
        self.requests_per_period = 5
        self.period_seconds = 30
        self.request_times = []

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CyberLLMInstruct-DataCollector/1.0'
        })

    def _check_rate_limit(self):
        """Enforce rate limiting for unauthenticated requests."""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < self.period_seconds]
        if len(self.request_times) >= self.requests_per_period:
            sleep_time = self.period_seconds - (current_time - self.request_times[0])
            logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            self.request_times = self.request_times[1:]
        self.request_times.append(time.time())

    def _make_request(self, url, params=None):
        """Make GET request with retry and rate limiting."""
        retry_count = 0
        while retry_count < self.max_retries:
            self._check_rate_limit()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                retry_count += 1
                logger.warning(f"Request failed ({retry_count}/{self.max_retries}): {e}")
                time.sleep(5 * retry_count)
        logger.error(f"Failed to fetch data from {url} after {self.max_retries} retries")
        return None

    def fetch_recent_cve_data(self, days: int = 90):
        """Fetch all CVE data modified in the last `days` days using pagination."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        params = {
            'lastModStartDate': start_date.isoformat(timespec='seconds') + 'Z',
            'lastModEndDate': end_date.isoformat(timespec='seconds') + 'Z',
            'startIndex': 0,
            'resultsPerPage': 2000  # max allowed per NVD API
        }

        all_cves = []
        while True:
            logger.info(f"Fetching CVEs starting at index {params['startIndex']}...")
            data = self._make_request(self.nvd_cve_url, params=params)
            if not data or 'vulnerabilities' not in data or len(data['vulnerabilities']) == 0:
                break

            all_cves.extend(data['vulnerabilities'])

            # Check if there are more results
            total_results = data.get('totalResults', 0)
            if params['startIndex'] + params['resultsPerPage'] >= total_results:
                break

            params['startIndex'] += params['resultsPerPage']

        logger.info(f"Total CVEs fetched: {len(all_cves)}")
        return all_cves

    from datetime import datetime, timezone

    def save_data_jsonl(self, data, filename: str = "nvd_cve.jsonl"):
        """Save CVE data in JSONL format (one JSON object per line) with UTC date tag."""
        try:
            # Use timezone-aware UTC datetime
            date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{date_tag}{ext}"

            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.info(f"Data saved in JSONL format to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False

def main():
    # Determine the data folder at the same level as scrape
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "cve"   # <-- save under data/cve
    
    # Create folder if it does not exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    collector = CyberDataCollector(output_dir=data_dir)
    
    logger.info("Fetching CVE data for the past 90 days...")
    data = collector.fetch_recent_cve_data(days=90)
    
    if data:
        collector.save_data_jsonl(data)
    else:
        logger.warning("No data fetched.")


if __name__ == "__main__":
    main()
