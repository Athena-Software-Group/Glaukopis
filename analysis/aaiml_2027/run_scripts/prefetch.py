#!/usr/bin/env python
"""Download a model into the Hugging Face cache before the server starts.
A stalled Hub download otherwise uses up the server's 30-minute start limit. The queue runs this under `timeout`
and retries, so a stall costs one attempt instead of the whole run.
Exits 0 only if every weight file listed on the Hub is in the cache at its listed size.
Usage: prefetch.py REPO_ID
"""
import os
import sys
import time
from huggingface_hub import HfApi, snapshot_download


def main():
    repo = sys.argv[1]
    token = os.environ["HF_TOKEN"]
    start = time.time()
    path = snapshot_download(repo, token=token)
    info = HfApi().model_info(repo, token=token, files_metadata=True)
    want = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".safetensors")}
    if not want:
        print(f"FAIL: {repo} lists no safetensors files", flush=True)
        sys.exit(2)
    bad = [f for f, size in want.items() if not os.path.exists(os.path.join(path, f)) or os.path.getsize(os.path.join(path, f)) != size]
    if bad:
        print(f"FAIL: {repo}: {len(bad)} of {len(want)} weight files missing or the wrong size: {bad[:3]}", flush=True)
        sys.exit(2)
    print(f"fetched {repo}: {sum(want.values()) / 1e9:.1f} GB of weights in {len(want)} files, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
