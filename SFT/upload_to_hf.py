#!/usr/bin/env python3
"""Merge a LoRA adapter (optionally) and push the resulting model to HF Hub.

Two modes:

1. Adapter mode (default): given a LoRA run directory written by
   llamafactory-cli train, merge it with the base model via
   `llamafactory-cli export` and upload the merged folder.

2. Merged mode: point at an already-merged directory with --merged-dir
   and only perform the upload.

Token resolution order:
    --token > $HF_TOKEN > $HUGGINGFACE_TOKEN > .env files > cached hf auth login

.env files are searched in order and first-match wins for each variable:
    <SFT>/.env  <SFT>/.env.local  <repo-root>/.env  <repo-root>/.env.local

Examples:
    # merge + push (private repo)
    python upload_to_hf.py \\
        --adapter-dir saves/meta-llama_Llama-3.1-8B-Instruct/lora/train_2026-04-21-08-33-05 \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --template llama3 \\
        --repo-id pworth1971/athena-cti-sft-llama31-8b

    # merge only, no upload (useful for local inspection)
    python upload_to_hf.py --adapter-dir <dir> --base-model <id> --repo-id x/y --skip-upload

    # upload a pre-merged folder
    python upload_to_hf.py --merged-dir merged/athena-cti-sft-llama31-8b \\
                           --repo-id pworth1971/athena-cti-sft-llama31-8b
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _load_dotenv_files(script_dir):
    """Populate os.environ from .env files without overriding existing vars.

    Silently no-ops if python-dotenv is not installed. Search order is
    SFT/.env, SFT/.env.local, <repo-root>/.env, <repo-root>/.env.local.
    First-write-wins to honour anything the caller already exported.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return []
    repo_root = script_dir.parent
    candidates = [
        script_dir / ".env",
        script_dir / ".env.local",
        repo_root / ".env",
        repo_root / ".env.local",
    ]
    loaded = []
    for path in candidates:
        if path.is_file():
            load_dotenv(dotenv_path=path, override=False)
            loaded.append(str(path))
    return loaded


def _resolve_token(cli_token):
    """cli > HF_TOKEN > HUGGINGFACE_TOKEN > huggingface_hub cached login."""
    if cli_token:
        return cli_token
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.getenv(var)
        if val:
            return val
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge a LoRA adapter and push the result to Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter-dir", help="LoRA run dir (llamafactory train output).")
    src.add_argument("--merged-dir", help="Already-merged model dir; skips the merge step.")

    p.add_argument("--base-model", help="Base model id for the merge step (required with --adapter-dir).")
    p.add_argument("--template", default="llama3",
                   help="LlamaFactory chat template to stamp into the merged tokenizer (default: llama3).")
    p.add_argument("--repo-id", required=True, help="Destination HF repo, e.g. org/name.")
    p.add_argument("--export-dir", default=None,
                   help="Where the merged model is written (default: ./merged/<repo-basename>).")
    p.add_argument("--export-size", type=int, default=5,
                   help="Max size per safetensors shard in GB (default: 5).")
    p.add_argument("--public", action="store_true", help="Create a public repo (default: private).")
    p.add_argument("--skip-merge", action="store_true",
                   help="Reuse an existing --export-dir instead of re-running the merge.")
    p.add_argument("--skip-upload", action="store_true", help="Merge only; do not push.")
    p.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN or $HUGGINGFACE_TOKEN).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p.parse_args()


def _default_export_dir(script_dir, repo_id):
    return script_dir / "merged" / repo_id.split("/")[-1]


def merge_lora(adapter_dir, base_model, template, export_dir, export_size, dry_run):
    if not base_model:
        sys.exit("--base-model is required when merging from --adapter-dir.")
    cmd = [
        "llamafactory-cli", "export",
        "--model_name_or_path", base_model,
        "--adapter_name_or_path", str(adapter_dir),
        "--template", template,
        "--finetuning_type", "lora",
        "--export_dir", str(export_dir),
        "--export_size", str(export_size),
        "--export_legacy_format", "False",
    ]
    print("[merge]", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)


def upload(folder, repo_id, token, private, dry_run):
    if not token:
        sys.exit(
            "No HF token found. Provide one of:\n"
            "  --token <tok>\n"
            "  export HF_TOKEN=<tok>\n"
            "  export HUGGINGFACE_TOKEN=<tok>\n"
            "  add HF_TOKEN=<tok> to SFT/.env or SFT/.env.local (install python-dotenv)\n"
            "  run 'hf auth login' once to cache credentials at ~/.cache/huggingface/token"
        )
    print(f"[upload] {folder} -> {repo_id} (private={private})")
    if dry_run:
        return
    from huggingface_hub import HfApi, login
    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    loaded = _load_dotenv_files(script_dir)
    if loaded:
        print(f"[env] loaded: {', '.join(loaded)}")

    if args.merged_dir:
        upload_folder_path = Path(args.merged_dir).resolve()
        if not upload_folder_path.is_dir():
            sys.exit(f"--merged-dir does not exist: {upload_folder_path}")
    else:
        adapter_dir = Path(args.adapter_dir).resolve()
        if not adapter_dir.is_dir():
            sys.exit(f"--adapter-dir does not exist: {adapter_dir}")
        export_dir = Path(args.export_dir).resolve() if args.export_dir \
            else _default_export_dir(script_dir, args.repo_id)

        already_merged = export_dir.is_dir() and any(export_dir.iterdir())
        if args.skip_merge:
            if not already_merged:
                sys.exit(f"--skip-merge given but export dir is empty: {export_dir}")
            print(f"[merge] skipped; reusing {export_dir}")
        else:
            if already_merged:
                print(f"[merge] export dir already populated: {export_dir} (pass --skip-merge to silence)")
            else:
                export_dir.mkdir(parents=True, exist_ok=True)
                merge_lora(adapter_dir, args.base_model, args.template,
                           export_dir, args.export_size, args.dry_run)
        upload_folder_path = export_dir

    if args.skip_upload:
        print(f"[done] skip-upload requested; merged model at {upload_folder_path}")
        return

    token = _resolve_token(args.token)
    upload(upload_folder_path, args.repo_id, token, private=not args.public, dry_run=args.dry_run)
    print(f"[done] https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()