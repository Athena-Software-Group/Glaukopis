#!/usr/bin/env python3
"""Register a built CPT corpus with LLaMA-Factory's dataset_info.json.

LLaMA-Factory pretraining (--stage pt) consumes a raw-text dataset with
a single column mapped to `prompt`. For a JSONL of {"text": "..."} lines
the entry is:

    "<name>": {
      "file_name": "<path-from-SFT/data>",
      "columns": {"prompt": "text"}
    }

We place the corpus inside SFT/data/ via a relative-path symlink so
LlamaFactory's SFT_DIR/data dataset_dir sees it and no per-host path
editing is required. On Windows or filesystems without symlinks, pass
--copy to hard-copy the file instead.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SFT_DATA = REPO_ROOT / "SFT" / "data"
DATASET_INFO = SFT_DATA / "dataset_info.json"


def register(name: str, corpus_file: Path, copy: bool) -> int:
    if not corpus_file.exists():
        print(f"[register] corpus file not found: {corpus_file}", file=sys.stderr)
        return 2
    SFT_DATA.mkdir(parents=True, exist_ok=True)

    dst = SFT_DATA / corpus_file.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        print(f"[register] copy  {corpus_file} -> {dst}")
        shutil.copy2(corpus_file, dst)
    else:
        # relative symlink so the registration is portable across hosts
        rel = os.path.relpath(corpus_file.resolve(), SFT_DATA)
        print(f"[register] link  {dst} -> {rel}")
        os.symlink(rel, dst)

    info = {}
    if DATASET_INFO.exists():
        info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    info[name] = {"file_name": corpus_file.name, "columns": {"prompt": "text"}}
    DATASET_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"[register] updated {DATASET_INFO} with key '{name}'")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Register a CPT corpus with LLaMA-Factory")
    p.add_argument("--name", required=True, help="Dataset key to register (e.g. cti_corpus_v1)")
    p.add_argument("--file", required=True, help="Path to the built corpus JSONL")
    p.add_argument("--copy", action="store_true",
                   help="Copy instead of symlinking (e.g. on Windows or cross-filesystem)")
    args = p.parse_args(argv)
    return register(args.name, Path(args.file).resolve(), copy=args.copy)


if __name__ == "__main__":
    raise SystemExit(main())
