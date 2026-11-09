#!/usr/bin/env python
"""Check a finished run's weights against its Hugging Face repo, and optionally remove the local run folder.

Usage: verify_hf.py OUTPUT_DIR REPO_ID [--prune]
Exit 0 only if every local *.safetensors file exists on the Hub with the same size and SHA-256.
--prune removes OUTPUT_DIR after that check passes, and only for this round's run folders.
"""
import glob
import hashlib
import os
import re
import shutil
import sys

from huggingface_hub import HfApi

BASE = "/home/Glaukopis/SFT/saves/Qwen_Qwen2.5-32B-Instruct/full/"
ALLOWED = re.compile(r"^(g2a_skiptaa_cse|g2b_skiptaa_recal|g1_single_stage|g4_seed[123])$")
MIN_BYTES = 60e9  # a complete 32B checkpoint is about 65.5 GB


def fail(msg):
    print("FAIL:", msg, flush=True)
    sys.exit(2)


def sha_of(info):
    lfs = info.lfs
    if lfs is None:
        return None
    return lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256


def main():
    out, repo = sys.argv[1].rstrip("/"), sys.argv[2]
    prune = "--prune" in sys.argv[3:]
    local = sorted(glob.glob(out + "/*.safetensors"))
    if not local:
        fail(f"no safetensors in {out}")
    total = sum(os.path.getsize(p) for p in local)
    if total < MIN_BYTES:
        fail(f"local weights only {total / 1e9:.1f} GB")
    api = HfApi(token=os.environ["HF_TOKEN"])
    hub = {s.rfilename: s for s in api.model_info(repo, files_metadata=True).siblings}
    for p in local:
        name = os.path.basename(p)
        remote = hub.get(name)
        if remote is None:
            fail(f"{name} missing from {repo}")
        if remote.size != os.path.getsize(p):
            fail(f"{name} size differs")
        digest = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        if sha_of(remote) != digest.hexdigest():
            fail(f"{name} sha256 differs")
    if len(local) > 1 and "model.safetensors.index.json" not in hub:
        fail("index json missing from the Hub repo")
    print(f"OK: {repo} matches {len(local)} local file(s), {total / 1e9:.1f} GB", flush=True)
    if prune:
        if not out.startswith(BASE) or not ALLOWED.match(os.path.basename(out)):
            fail(f"refusing to remove {out}")
        shutil.rmtree(out)
        print("removed", out, flush=True)


if __name__ == "__main__":
    main()
