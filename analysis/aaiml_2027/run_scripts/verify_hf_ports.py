#!/usr/bin/env python
"""Check a finished single-pass run's weights against its Hugging Face repo, and optionally remove the local run folder.

Usage: verify_hf_ports.py OUTPUT_DIR REPO_ID --min-gb N [--prune]
Exit 0 only if every local *.safetensors file exists on the Hub with the same size and SHA-256.
--prune removes OUTPUT_DIR after that check passes, and only for this round's run folders.
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import sys

from huggingface_hub import HfApi

ALLOWED = re.compile(r"^/home/Glaukopis/SFT/saves/[^/]+/full/(port_single_stage_(foundation_8b|llama31_8b|qwen25_14b|qwen3_30b_thinking)|ctrl2x_(foundation_8b|llama31_8b)/(phase_b|taa|cse|recalibrate)|ctrl4x_(qwen25_14b|qwen25_32b)/(phase_b|taa|cse|recalibrate|recal_32b))$")


def fail(msg):
    print("FAIL:", msg, flush=True)
    sys.exit(2)


def sha_of(info):
    lfs = info.lfs
    if lfs is None:
        return None
    return lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("repo")
    ap.add_argument("--min-gb", type=float, required=True)
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args()
    out = args.out.rstrip("/")
    local = sorted(glob.glob(out + "/*.safetensors"))
    if not local:
        fail(f"no safetensors in {out}")
    total = sum(os.path.getsize(p) for p in local)
    if total < args.min_gb * 1e9:
        fail(f"local weights only {total / 1e9:.1f} GB")
    api = HfApi(token=os.environ["HF_TOKEN"])
    hub = {s.rfilename: s for s in api.model_info(args.repo, files_metadata=True).siblings}
    for p in local:
        name = os.path.basename(p)
        remote = hub.get(name)
        if remote is None:
            fail(f"{name} missing from {args.repo}")
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
    print(f"OK: {args.repo} matches {len(local)} local file(s), {total / 1e9:.1f} GB", flush=True)
    if args.prune:
        if not ALLOWED.match(out):
            fail(f"refusing to remove {out}")
        shutil.rmtree(out)
        print("removed", out, flush=True)


if __name__ == "__main__":
    main()
