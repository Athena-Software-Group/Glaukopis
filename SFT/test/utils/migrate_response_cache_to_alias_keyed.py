#!/usr/bin/env python3
"""One-shot migration: HF-repo-id-keyed -> alias-keyed response cache.

Background
----------
Before this migration every benchmark wrote responses to
``responses/<hf_repo_sanitized>/<task>/<task>_..._<hf_repo_sanitized>_response.<ext>``
where ``<hf_repo_sanitized>`` was ``model_mapping[alias].replace('/','_')``.
Two aliases pointing to the same HF repo (e.g.
``qwen3-30b-a3b-thinking-2507-vllm`` and the matching ``-no-think`` alias
both on ``Qwen/Qwen3-30B-A3B-Thinking-2507``) collided on the same cache
slot, so the second alias to run silently re-scored the first alias's CSV
on resume. After the migration the convention is alias-keyed everywhere
(see ``pipelines/models.alias_to_safe_name``); this script renames the
existing on-disk caches so resume keeps working.

Strategy
--------
For each ``responses/<dir>/`` directory whose name matches a sanitized HF
repo id with EXACTLY ONE alias pointing to it, rename:
    responses/<hf_sanitized>/         -> responses/<alias_sanitized>/
and inside, rewrite every embedded ``_<hf_sanitized>_`` substring in
filenames to ``_<alias_sanitized>_``. Summary JSON ``display_name`` /
``model`` fields are left untouched (still human-readable HF repo id).

For HF repos with multiple aliases (the collision case) the directory is
left in place and reported -- the operator must decide which alias
inherits the existing cache.

Dry-run by default. Pass ``--apply`` to actually rename. Pass
``--responses-root DIR`` to override the default ``SFT/test/responses``.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys


def _load_model_mapping(models_py: pathlib.Path) -> dict[str, str]:
    src = models_py.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "model_mapping":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"model_mapping not found in {models_py}")


def _hf_safe(hf_id: str) -> str:
    return hf_id.replace("/", "_")


def _alias_safe(alias: str) -> str:
    return alias.replace("/", "_")


def _build_reverse_index(mapping: dict[str, str]) -> dict[str, list[str]]:
    """sanitized HF id -> list of aliases that resolve to it."""
    idx: dict[str, list[str]] = {}
    for alias, hf_id in mapping.items():
        idx.setdefault(_hf_safe(hf_id), []).append(alias)
    return idx


def _migrate_one(
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    hf_safe: str,
    alias_safe: str,
    apply: bool,
) -> int:
    """Rename src_dir -> dst_dir and rewrite filenames inside. Returns count."""
    renames: list[tuple[pathlib.Path, pathlib.Path]] = []
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src_dir)
        new_name = p.name.replace(f"_{hf_safe}_", f"_{alias_safe}_")
        new_rel = rel.with_name(new_name)
        renames.append((p, dst_dir / new_rel))

    print(f"  {src_dir.name}/  ->  {dst_dir.name}/  ({len(renames)} files)")
    if not apply:
        return len(renames)

    if dst_dir.exists():
        print(f"  [SKIP] destination already exists: {dst_dir}", file=sys.stderr)
        return 0
    for _, dst in renames:
        dst.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in renames:
        src.rename(dst)
    # Drop the (now empty) source tree.
    for sub in sorted(src_dir.rglob("*"), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()
            except OSError:
                pass
    try:
        src_dir.rmdir()
    except OSError:
        print(f"  [WARN] {src_dir} not empty after move; leaving in place",
              file=sys.stderr)
    return len(renames)


def main(argv: list[str]) -> int:
    here = pathlib.Path(__file__).resolve().parent
    bench_dir = here.parent  # SFT/test
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--responses-root", default=str(bench_dir / "responses"))
    p.add_argument("--models-py", default=str(bench_dir / "pipelines/models.py"))
    p.add_argument("--apply", action="store_true",
                   help="Actually rename. Default is dry-run.")
    args = p.parse_args(argv)

    responses = pathlib.Path(args.responses_root)
    if not responses.is_dir():
        print(f"[FAIL] {responses} is not a directory", file=sys.stderr)
        return 1

    mapping = _load_model_mapping(pathlib.Path(args.models_py))
    aliases_by_hf = _build_reverse_index(mapping)
    aliases_safe = {_alias_safe(a) for a in mapping}

    print(f"[migrate] responses root: {responses}")
    print(f"[migrate] mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    migrated = ambiguous = already = 0
    for sub in sorted(responses.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in aliases_safe:
            already += 1
            continue  # already alias-keyed
        candidates = aliases_by_hf.get(sub.name, [])
        if len(candidates) == 1:
            alias = candidates[0]
            _migrate_one(sub, responses / _alias_safe(alias), sub.name,
                         _alias_safe(alias), args.apply)
            migrated += 1
        elif len(candidates) > 1:
            print(f"  [AMBIGUOUS] {sub.name}/  aliases: {candidates}")
            ambiguous += 1
        # else: not in mapping -> standalone dir (e.g. external model). Leave it.

    print()
    print(f"[migrate] migrated={migrated} ambiguous={ambiguous} already_alias_keyed={already}")
    if ambiguous:
        print("[migrate] resolve ambiguities by manually mv-ing each ambiguous dir "
              "to the alias-sanitized name you want it credited to; re-run --apply.")
    if not args.apply:
        print("[migrate] dry-run; re-invoke with --apply to perform renames.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
