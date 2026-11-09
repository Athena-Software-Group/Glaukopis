#!/usr/bin/env python
"""Download a model into the Hugging Face cache before the server starts.

A stalled Hub download otherwise uses up the server's 30-minute start limit. The queue runs this under `timeout`
and retries, so a stall costs one attempt instead of the whole run.

Usage: prefetch.py REPO_ID
"""
import os
import sys
import time

from huggingface_hub import snapshot_download


def main():
    repo = sys.argv[1]
    start = time.time()
    path = snapshot_download(repo, token=os.environ["HF_TOKEN"])
    gb = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors")) / 1e9
    print(f"fetched {repo}: {gb:.1f} GB of weights in {time.time() - start:.0f} s", flush=True)
    if gb < 50:
        print(f"FAIL: only {gb:.1f} GB of weights", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
