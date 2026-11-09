#!/usr/bin/env python
"""List every run folder under SFT/saves and say whether its final weights are on Hugging Face, compared by SHA-256."""
import glob
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

ROOT = "/home/Glaukopis/SFT/saves"
PREFIX = {"fdtn-ai_Foundation-Sec-8B-Instruct": "foundation-8b", "meta-llama_Llama-3.1-8B-Instruct": "llama31-8b"}
STAGE = [(r"^v21_core_phase_b_", "core"), (r"^v21_plus_taa_", "taa"), (r"^v21_cse_", "cse"), (r"^v21_recalibrate_", "recalibrate")]
api = HfApi(token=os.environ["HF_TOKEN"])


def size_of(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs if not os.path.islink(os.path.join(r, f)))


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


def repo_for(model, run):
    if model not in PREFIX:
        return None
    for pattern, stage in STAGE:
        if re.match(pattern, run):
            return f"asg-ai/athena-cti-sft-{PREFIX[model]}-v21-{stage}"
    return None


def check(run_dir, repo):
    local = sorted(glob.glob(run_dir + "/*.safetensors"))
    if not local:
        return "no final weights in folder"
    hub = {s.rfilename: s for s in api.model_info(repo, files_metadata=True).siblings}
    for p in local:
        name = os.path.basename(p)
        remote = hub.get(name)
        if remote is None or remote.size != os.path.getsize(p):
            return f"DIFFERS from {repo} ({name})"
        lfs = remote.lfs
        want = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
        if sha(p) != want:
            return f"HASH DIFFERS from {repo} ({name})"
    return f"on Hugging Face, hash match: {repo}"


def describe(run_dir):
    parts = run_dir.split("/")
    model, kind, run = parts[-3], parts[-2], parts[-1]
    size = size_of(run_dir) / 1e9
    ckpts = len(glob.glob(run_dir + "/checkpoint-*"))
    finals = glob.glob(run_dir + "/*.safetensors")
    repo = repo_for(model, run)
    status = check(run_dir, repo) if repo else ("final weights, no published repo mapped" if finals else "no final weights")
    return size, f"{model}/{kind}/{run}", ckpts, status


runs = sorted(glob.glob(ROOT + "/*/*/*"))
runs = [r for r in runs if os.path.isdir(r)]
with ThreadPoolExecutor(6) as pool:
    rows = list(pool.map(describe, runs))
rows.sort(reverse=True)
print(f"{len(rows)} run folders, {sum(r[0] for r in rows):.0f} GB in total")
for size, name, ckpts, status in rows:
    print(f"{size:8.1f} GB  ckpts={ckpts:<2d} {name:80s} {status}")
