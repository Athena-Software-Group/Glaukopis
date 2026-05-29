#!/usr/bin/env python3
"""Pair up per-row responses from a transformers (HF) run and a vLLM run of
the same model + task, and print the side-by-side deltas.

Purpose: when an aggregate metric diverges between transports (e.g. v3 RMS
45.9 F1 on HF vs 4.5 F1 on vLLM) we need to see whether the drop is driven
by truncation, stop-token asymmetry, prompt-template drift or something
else. Reads the two JSONL response files, joins on ``id``, and prints
length + prediction + truncated raw text for the first N rows where the
per-row verdict differs (HF correct / vLLM wrong by default, or any
per-row length delta > threshold).

Usage
-----

    python diff_hf_vllm_responses.py \
        --hf   responses/<model-dir>/<task>/<task>_..._<model>_response.jsonl \
        --vllm responses/<model-dir>/<task>/<task>_..._<model>-vllm_response.jsonl \
        [--n 20] [--mode diverge|length|all]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return {row["id"]: row for row in (json.loads(l) for l in f)}


def truncate(text: str, n: int = 220) -> str:
    if text is None:
        return ""
    t = text.replace("\n", "\\n")
    return t if len(t) <= n else t[:n] + "…"


def correct(row: dict) -> bool:
    score = row.get("score")
    if score is None:
        # Some tasks store binary correct/incorrect in a different field;
        # fall back to prediction==answer string match.
        return str(row.get("prediction", "")).strip() == str(
            row.get("answer", "")
        ).strip()
    try:
        return float(score) > 0
    except (TypeError, ValueError):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True, type=Path, help="HF-path JSONL")
    p.add_argument("--vllm", required=True, type=Path, help="vLLM-path JSONL")
    p.add_argument("--n", type=int, default=20, help="rows to print")
    p.add_argument(
        "--mode",
        choices=("diverge", "length", "all"),
        default="diverge",
        help="diverge=HF-correct but vLLM-wrong; length=|len(HF)-len(vLLM)|>LENGTH_THRESH; all=first N common ids",
    )
    p.add_argument("--length-thresh", type=int, default=100)
    args = p.parse_args()

    hf, vl = load(args.hf), load(args.vllm)
    common = sorted(set(hf) & set(vl))
    print(f"HF rows   : {len(hf)}")
    print(f"vLLM rows : {len(vl)}")
    print(f"common ids: {len(common)}\n")

    # aggregate correctness counts for orientation
    hf_ok = sum(1 for i in common if correct(hf[i]))
    vl_ok = sum(1 for i in common if correct(vl[i]))
    print(f"HF   correct on common: {hf_ok}/{len(common)} ({100*hf_ok/len(common):.1f}%)")
    print(f"vLLM correct on common: {vl_ok}/{len(common)} ({100*vl_ok/len(common):.1f}%)\n")

    hf_resp_lens = [len(hf[i].get("response") or "") for i in common]
    vl_resp_lens = [len(vl[i].get("response") or "") for i in common]
    def mean(xs): return sum(xs)/len(xs) if xs else 0
    print(f"mean response len  HF={mean(hf_resp_lens):.0f}  vLLM={mean(vl_resp_lens):.0f}\n")

    picked = []
    for i in common:
        h, v = hf[i], vl[i]
        hc, vc = correct(h), correct(v)
        if args.mode == "diverge" and hc and not vc:
            picked.append(i)
        elif args.mode == "length":
            dl = abs(len(h.get("response") or "") - len(v.get("response") or ""))
            if dl > args.length_thresh:
                picked.append(i)
        elif args.mode == "all":
            picked.append(i)
        if len(picked) >= args.n:
            break

    print(f"printing {len(picked)} row(s) (mode={args.mode})\n")

    for i in picked:
        h, v = hf[i], vl[i]
        gt = h.get("answer")
        hr = h.get("response") or ""
        vr = v.get("response") or ""
        print(f"=== id={i}  gt={gt!r} ===")
        print(f"  HF   pred={h.get('prediction')!r}  score={h.get('score')!r}  len={len(hr)}")
        print(f"       {truncate(hr)}")
        print(f"  VLLM pred={v.get('prediction')!r}  score={v.get('score')!r}  len={len(vr)}")
        print(f"       {truncate(vr)}")
        print()


if __name__ == "__main__":
    main()
