from __future__ import annotations
import pandas as pd
import pickle
import os
import json
from typing import Dict, List, Optional
import yaml
from dotenv import load_dotenv
from datetime import datetime, timezone

def load_data(file_path, num_rows=None):
    """Load data from TSV or CSV file (auto-detect by extension)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found.")
    
    if file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)
    
    if num_rows:
        df = df.iloc[:num_rows]
    return df

def load_csv(file_path):
    """Load data from a CSV file (comma-separated)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file {file_path} not found.")
    return pd.read_csv(file_path)

def load_pickle_file(file_path):
    """Load data from a pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file {file_path} not found.")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_json_or_jsonl(file_path, num_rows=None):
    """
    Load data from JSON or JSONL file.
    
    Args:
        file_path (str): Path to JSON or JSONL file
        num_rows (int, optional): Limit number of rows
    Returns:
        list: List of loaded objects
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    data = []
    if file_path.endswith(".jsonl"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
                if num_rows and len(data) >= num_rows:
                    break
    elif file_path.endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if num_rows:
                data = data[:num_rows]
    else:
        raise ValueError("Unsupported file type. Use .json or .jsonl")
    
    return data

def save_responses(data, output_file, append=True, sep=None):
    """
    Save responses to CSV/TSV file.
    
    Args:
        data (pd.DataFrame): Data to save
        output_file (str): Path to file
        append (bool): Append to file if it exists
        sep (str): Separator (default: auto-detect from extension)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Auto-detect separator if not provided
    if sep is None:
        if output_file.endswith(".csv"):
            sep = ","
        elif output_file.endswith(".tsv"):
            sep = "\t"
        else:
            sep = ","  # default to CSV

    # If appending
    if append and os.path.exists(output_file):
        data.to_csv(output_file, sep=sep, index=False, mode="a", header=False)
    else:
        data.to_csv(output_file, sep=sep, index=False)

def load_yaml(path: str) -> Dict:
    """Return the YAML content of *path* as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[Dict]:
    """Return a list of JSON objects loaded from a JSONL file.

    Malformed lines (e.g., partial writes from a crashed run) are skipped
    with a warning so that one bad row does not abort scoring on the
    remaining good rows. Use ``sanitize_jsonl`` to drop them from disk.
    """
    data: List[Dict] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad += 1
                print(f"[load_jsonl] WARN: skipping malformed line {i} in {path}: {e.msg}")
    if bad:
        print(f"[load_jsonl] WARN: skipped {bad} malformed line(s) in {path}")
    return data


def sanitize_jsonl(path: str) -> int:
    """Drop malformed/partial lines from a JSONL file, rewriting it in place.

    Intended to be called at the start of a resume-aware generator so that
    any partial trailing line from a previously-killed run (e.g., ENOSPC,
    SIGTERM) is removed before append-mode writes resume. Without this,
    the next ``open(..., "a")`` write would concatenate onto the partial
    line, producing one giant corrupt record visible to ``load_jsonl``.

    Returns the number of bad lines dropped (0 if the file is clean or
    does not exist).
    """
    if not os.path.exists(path):
        return 0
    good: List[str] = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                dropped += 1
                continue
            good.append(stripped + "\n")
    if dropped:
        tmp = path + ".sanitize.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(good)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"[sanitize_jsonl] dropped {dropped} malformed line(s) from {path}")
    return dropped


def load_api_key(var_name: str) -> str:
    """Retrieve an API key from environment variables or a ``.env`` file.

    Parameters
    ----------
    var_name:
        Name of the environment variable containing the API key.

    Returns
    -------
    str
        The API key string.

    Raises
    ------
    ValueError
        If the variable is not set in the environment or ``.env`` file.
    """
    load_dotenv()
    key = os.getenv(var_name)
    if not key:
        raise ValueError(f"{var_name} not found in environment or .env file")
    return key


def parse_date(s: Optional[str]) -> Optional[datetime]:
    """Parse *s* into a :class:`~datetime.datetime` if possible."""

    if not s:
        return None
    s_str = str(s).strip()

    def _fix_z(val: str) -> str:
        return val.replace("Z", "+00:00") if val.endswith("Z") else val

    try:
        return datetime.fromisoformat(_fix_z(s_str))
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s_str, fmt)
        except Exception:
            continue
    return None


def within_inclusive(ts: str, start_dt: datetime, end_dt: datetime) -> bool:
    """Return ``True`` if ``ts`` is within the inclusive range."""

    d = parse_date(ts)
    if not d:
        return False
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return start_dt <= d <= end_dt
