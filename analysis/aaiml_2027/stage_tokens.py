#!/usr/bin/env python3
"""Examples and tokens consumed by each training stage of the v21 32B chain (reviewer comment 3).

Reproduces LlamaFactory's supervised encoding for the `qwen` template on CPU:
  - Sophia shards map instruction -> system prompt, input -> user turn, output -> response
    (SFT/data/dataset_info.json); alpaca_en_demo maps instruction+"\\n"+input -> user turn under
    the default system prompt; Tulu-3 rows are ShareGPT conversations (an optional leading
    system message replaces the default; user/assistant turns must alternate, else the row is
    dropped, as LlamaFactory does).
  - Turns are rendered with the qwen template and truncated with LlamaFactory's infer_seqlen
    rule against the stage's cutoff_len, turn by turn.
  - `--max-samples` keeps the first N rows of each dataset (LlamaFactory loader.select(range(N))).
Reported per stage: examples, tokens consumed (prompt + response after truncation), response
(loss) tokens, and examples truncated. Stage 4 draws ~6,000 rows by interleave_under from the
first 3,600 rows of Core-A/Core-B/TAA at probabilities 0.15/0.60/0.25; its figures are the
expectation (900/3,600/1,500 draws times each shard's mean over its first 3,600 rows).

Usage (needs the `tokenizers` and `pyarrow` packages):
  python stage_tokens.py --tokenizer Qwen2.5-32B-Instruct/tokenizer.json \\
      --tulu train-00000-of-00006.parquet train-00001-of-00006.parquet
"""
import argparse, json, os, sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

R = os.environ.get("GLAUKOPIS_ROOT", "/Users/pietro/code/Glaukopis")
D = f"{R}/SFT/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stage_tokens.json")
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def infer_seqlen(source_len, target_len, cutoff_len):  # LlamaFactory processor_utils.infer_seqlen
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    return min(max_source_len, source_len), new_target_len

def render_pairs(system, turns):
    """turns: [(user, assistant), ...] -> [(source_text, target_text), ...] in the qwen template."""
    out = []
    for k, (u, a) in enumerate(turns):
        src = (f"<|im_start|>system\n{system}<|im_end|>\n" if k == 0 and system else "")
        src += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
        out.append((src, f"{a}<|im_end|>\n"))
    return out

def account(tok, conversations, cutoff, batch=2048):
    """conversations: list of turn-pair lists. Returns totals with LlamaFactory truncation."""
    stats = {"examples": 0, "tokens_consumed": 0, "response_tokens": 0, "truncated": 0}
    for i in range(0, len(conversations), batch):
        chunk = conversations[i:i + batch]
        flat = [t for conv in chunk for pair in conv for t in pair]
        lens = [len(e.ids) for e in tok.encode_batch(flat, add_special_tokens=False)]
        j = 0
        for conv in chunk:
            total = 0; resp = 0; cut = False
            for _ in conv:
                s, t = lens[j], lens[j + 1]; j += 2
                if total >= cutoff:
                    cut = True; continue
                ns, nt = infer_seqlen(s, t, cutoff - total)
                cut |= (ns, nt) != (s, t)
                total += ns + nt; resp += nt
            stats["examples"] += 1; stats["tokens_consumed"] += total
            stats["response_tokens"] += resp; stats["truncated"] += cut
    return stats

def sophia(name, limit=None):
    rows = json.load(open(f"{D}/{name}.json", encoding="utf-8"))[:limit]
    return [render_pairs(r.get("instruction") or DEFAULT_SYSTEM, [(r.get("input", ""), r.get("output", ""))]) for r in rows]

def alpaca(name):
    rows = json.load(open(f"{D}/{name}.json", encoding="utf-8"))
    return [render_pairs(DEFAULT_SYSTEM, [("\n".join(x for x in (r.get("instruction"), r.get("input")) if x), r.get("output", ""))])
            for r in rows]

def tulu(paths, limit):
    convs, dropped, seen = [], 0, 0
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(columns=["messages"], batch_size=4096):
            for msgs in batch.column(0).to_pylist():
                if seen >= limit:
                    return convs, dropped
                seen += 1
                system = DEFAULT_SYSTEM
                if msgs and msgs[0]["role"] == "system":
                    system, msgs = msgs[0]["content"], msgs[1:]
                ok = len(msgs) >= 2 and len(msgs) % 2 == 0 and all(
                    m["role"] == ("user" if k % 2 == 0 else "assistant") for k, m in enumerate(msgs))
                if not ok:
                    dropped += 1; continue
                convs.append(render_pairs(system, [(msgs[k]["content"], msgs[k + 1]["content"]) for k in range(0, len(msgs), 2)]))
    sys.exit(f"Tulu shards hold only {seen} rows; {limit} needed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tulu", nargs="+", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    res = {}
    tconv, tdrop = tulu(a.tulu, 240000)
    stage1a = {"Core-A": account(tok, sophia("ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn"), 8192),
               "Tulu-3 SFT (first 240,000 rows)": {**account(tok, tconv, 8192), "dropped_invalid_roles": tdrop},
               "Alpaca demo": account(tok, alpaca("alpaca_en_demo"), 8192)}
    res["1A (cutoff 8,192, packed)"] = stage1a
    res["1B (cutoff 16,384)"] = {"Core-B": account(tok, sophia("ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm"), 16384)}
    res["2 (cutoff 4,096, packed)"] = {"TAA": account(tok, sophia("ift_data_2026_05_18_v21_taa"), 4096)}
    res["3 (cutoff 4,096, packed)"] = {"CSE": account(tok, sophia("ift_data_2026_05_18_v21_cse"), 4096)}
    s4 = {}
    for lab, name, draws in [("Core-A", "ift_data_2026_05_18_v21_core_a_kb_mcq_taa_soc_cm_ms_yn", 900),
                             ("Core-B", "ift_data_2026_05_18_v21_core_b_rms_ate_vsp_rcm", 3600),
                             ("TAA", "ift_data_2026_05_18_v21_taa", 1500)]:
        st = account(tok, sophia(name, 3600), 16384)
        s4[lab] = {"expected_draws": draws,
                   "expected_tokens_consumed": round(draws * st["tokens_consumed"] / st["examples"]),
                   "expected_response_tokens": round(draws * st["response_tokens"] / st["examples"])}
    res["4 (cutoff 16,384, interleaved; expectation)"] = s4
    for stage, parts in res.items():
        ex = sum(p.get("examples", p.get("expected_draws", 0)) for p in parts.values())
        tc = sum(p.get("tokens_consumed", p.get("expected_tokens_consumed", 0)) for p in parts.values())
        rt = sum(p.get("response_tokens", p.get("expected_response_tokens", 0)) for p in parts.values())
        parts["_total"] = {"examples": ex, "tokens_consumed": tc, "response_tokens": rt}
        print(f"{stage:45s} examples={ex:>9,}  tokens={tc:>13,}  response tokens={rt:>12,}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", os.path.normpath(OUT))

if __name__ == "__main__":
    main()
