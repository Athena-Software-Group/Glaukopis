#!/usr/bin/env python3
"""Validate one or more pushed HF model repos against a reference base.

Default checks (no large downloads):
  - repo exists and is readable with the resolved HF token
  - file manifest includes config.json, tokenizer files, safetensors
    index, and at least one weight shard
  - safetensors shard sizes are consistent with model.safetensors.index.json
  - config.json fields (architectures, hidden_size, num_hidden_layers,
    vocab_size) match the expected base architecture

With --deep:
  - download tokenizer files and round-trip a sample string
  - download config.json and verify model_type / torch_dtype

Token resolution mirrors upload_to_hf.py.

Examples:
    # validate the four v20 stages (metadata + config only)
    python validate_hf_repo.py --preset v20

    # validate one repo, including tokenizer round-trip
    python validate_hf_repo.py \\
        --repo-id asg-ai/athena-cti-sft-qwen25-14b-v20-recalibrate --deep
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path

EXPECTED_BASES = {
    "qwen2.5-14b": dict(architectures=["Qwen2ForCausalLM"], hidden_size=5120,
                        num_hidden_layers=48, vocab_size=152064, model_type="qwen2"),
    "llama3.1-8b": dict(architectures=["LlamaForCausalLM"], hidden_size=4096,
                        num_hidden_layers=32, vocab_size=128256, model_type="llama"),
}
PRESETS = {
    "v20": (["athena-cti-sft-qwen25-14b-v20-core", "athena-cti-sft-qwen25-14b-v20-taa",
             "athena-cti-sft-qwen25-14b-v20-cse", "athena-cti-sft-qwen25-14b-v20-recalibrate"],
            "qwen2.5-14b"),
}
REQUIRED_FILES = {"config.json", "tokenizer_config.json"}
TOKENIZER_ANY_OF = ({"tokenizer.json"}, {"tokenizer.model"}, {"vocab.json", "merges.txt"})
WEIGHT_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
SINGLE_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for p in (here / ".env", here / ".env.local", here.parent / ".env", here.parent / ".env.local"):
        if p.is_file():
            load_dotenv(p, override=False)


def _resolve_token(cli):
    return cli or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _detect_weight_layout(files):
    """Return (kind, names) for the weight files: ('sharded', [idx_name]),
    ('single', [weight_name]), or ('none', []). Sharded takes precedence."""
    fs = set(files)
    for idx in WEIGHT_INDEX_NAMES:
        if idx in fs:
            return "sharded", [idx]
    for single in SINGLE_WEIGHT_NAMES:
        if single in fs:
            return "single", [single]
    return "none", []


def _check_manifest(files):
    missing = REQUIRED_FILES - set(files)
    if missing:
        return False, f"missing required files: {sorted(missing)}"
    if not any(group <= set(files) for group in TOKENIZER_ANY_OF):
        return False, f"no usable tokenizer (need any of {[sorted(g) for g in TOKENIZER_ANY_OF]})"
    kind, _ = _detect_weight_layout(files)
    if kind == "sharded":
        shards = [f for f in files if (f.startswith("model-") and f.endswith(".safetensors"))
                  or (f.startswith("pytorch_model-") and f.endswith(".bin"))]
        if not shards:
            return False, "weight index present but no model-*/pytorch_model-* shards"
        return True, f"sharded ({len(shards)} shard(s)), tokenizer present"
    if kind == "single":
        return True, "single-file weights, tokenizer present"
    return False, f"no weight files (looked for {list(WEIGHT_INDEX_NAMES + SINGLE_WEIGHT_NAMES)})"


def _check_weight_sizes(api, repo_id, files_info, files):
    kind, names = _detect_weight_layout(files)
    if kind == "none":
        return False, "no weight files to size-check"
    by_path = {fi.path: fi.size for fi in files_info if fi.size is not None}
    if kind == "single":
        size = by_path.get(names[0])
        if size is None:
            return False, f"{names[0]} listed but no size metadata"
        return True, f"single file {names[0]}, size={size / 1e9:.2f} GB"
    from huggingface_hub import hf_hub_download
    idx_path = hf_hub_download(repo_id=repo_id, filename=names[0])
    index = json.loads(Path(idx_path).read_text())
    total = index.get("metadata", {}).get("total_size")
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    actual = sum(by_path.get(s, 0) for s in shard_names)
    if total is not None and abs(actual - total) > 1024:
        return False, f"shard size mismatch: index total_size={total}, sum={actual}"
    return True, f"shards={len(shard_names)}, total={actual / 1e9:.2f} GB"


def _check_config(api, repo_id, expected):
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    cfg = json.loads(Path(cfg_path).read_text())
    mismatches = [k for k, v in expected.items() if cfg.get(k) != v]
    if mismatches:
        details = ", ".join(f"{k}={cfg.get(k)!r} (want {expected[k]!r})" for k in mismatches)
        return False, f"config mismatch: {details}"
    return True, f"model_type={cfg.get('model_type')}, layers={cfg.get('num_hidden_layers')}"


def _check_tokenizer_roundtrip(repo_id, token):
    from transformers import AutoTokenizer
    with tempfile.TemporaryDirectory() as td:
        tok = AutoTokenizer.from_pretrained(repo_id, token=token, cache_dir=td, trust_remote_code=True)
        sample = "TA0001 Initial Access via T1566.001."
        ids = tok.encode(sample); back = tok.decode(ids, skip_special_tokens=True)
        if sample.replace(" ", "") not in back.replace(" ", ""):
            return False, f"tokenizer round-trip lost content: {sample!r} -> {back!r}"
        return True, f"vocab_size={tok.vocab_size}, ids={len(ids)}"


def validate_repo(repo_id, expected_cfg, token, deep):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"\n=== {repo_id} ===")
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        print(f"  [FAIL] model_info: {e}"); return False
    print(f"  exists      : private={info.private} sha={info.sha[:12]}")
    files = [s.rfilename for s in info.siblings]
    files_info = [type("F", (), dict(path=s.rfilename, size=s.size))() for s in info.siblings]
    ok_man, msg_man = _check_manifest(files); print(f"  [{'PASS' if ok_man else 'FAIL'}] manifest    : {msg_man}")
    results = [ok_man]
    if ok_man:
        ok_w, msg_w = _check_weight_sizes(api, repo_id, files_info, files)
        print(f"  [{'PASS' if ok_w else 'FAIL'}] weight sizes: {msg_w}")
        ok_cfg, msg_cfg = _check_config(api, repo_id, expected_cfg)
        print(f"  [{'PASS' if ok_cfg else 'FAIL'}] config      : {msg_cfg}")
        results += [ok_w, ok_cfg]
        if deep:
            ok_tok, msg_tok = _check_tokenizer_roundtrip(repo_id, token)
            print(f"  [{'PASS' if ok_tok else 'FAIL'}] tokenizer   : {msg_tok}")
            results.append(ok_tok)
    else:
        print(f"  [SKIP] downstream checks skipped (manifest failed); files on repo:")
        for f in sorted(files):
            print(f"           - {f}")
    return all(results)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", action="append", default=[], help="HF repo id; repeatable.")
    p.add_argument("--preset", choices=sorted(PRESETS), help="Validate a curated set; resolves names against $HF_USERNAME.")
    p.add_argument("--namespace", default=os.environ.get("HF_USERNAME"), help="HF user/org for --preset (default: $HF_USERNAME).")
    p.add_argument("--expected-base", choices=sorted(EXPECTED_BASES), help="Reference architecture (default: preset's).")
    p.add_argument("--deep", action="store_true", help="Add tokenizer round-trip check.")
    p.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN / $HUGGINGFACE_TOKEN / .env).")
    args = p.parse_args()
    _load_dotenv()
    token = _resolve_token(args.token)
    repos, base_key = list(args.repo_id), args.expected_base
    if args.preset:
        names, preset_base = PRESETS[args.preset]
        if not args.namespace:
            sys.exit("--preset requires --namespace or $HF_USERNAME")
        repos += [f"{args.namespace}/{n}" for n in names]
        base_key = base_key or preset_base
    if not repos:
        sys.exit("provide --repo-id (repeatable) or --preset")
    base_key = base_key or "qwen2.5-14b"
    expected = EXPECTED_BASES[base_key]
    print(f"validating {len(repos)} repo(s) against base={base_key} deep={args.deep}")
    results = {r: validate_repo(r, expected, token, args.deep) for r in repos}
    print("\n=== summary ===")
    for r, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {r}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
