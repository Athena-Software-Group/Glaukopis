#!/usr/bin/env python
"""Post-fix v8.1 SFT outputs that absorbed a section-header banner from the
template file (parser bug, fixed in tmpl_docx2json.py from this rev forward).

For every shortname:
  1. Look up the OLD (buggy) compiled-template Answer text and the FIXED
     compiled-template Answer text.
  2. The buggy text equals (fixed_text + trailing_junk).  Extract trailing_junk.
  3. For every row whose `output` ends with trailing_junk (after stripping
     {force ...} wrappers and surrounding whitespace), strip it.

Operates in place on SFT/data/ift_data_2026_04_30_v81.json (a backup is
written to SFT/data/ift_data_2026_04_30_v81.pre_strip.json before edit).
"""
import json, re, shutil, sys
from pathlib import Path

DATA_PATH      = Path("SFT/data/ift_data_2026_04_30_v81.json")
BACKUP_PATH    = Path("SFT/data/ift_data_2026_04_30_v81.pre_strip.json")
BUGGY_TMPLS    = Path("tmpl_gen/data_generation/Sophia-CTI-Templates-v8_1.json")
FIXED_TMPLS    = Path("_v81_build/Sophia-CTI-Templates-v8_1.fixed.json")

# Strip {force ...} clauses, <* ... *> blocks, and surrounding whitespace from
# both texts so the diff reflects only renderable content.
_FORCE_BLOCK = re.compile(r"<\*.*?\*>", flags=re.DOTALL)
_FORCE_INLINE = re.compile(r"\{force[^}]*\}")

def _normalise_answer(text: str) -> str:
    if "Answer:" not in text:
        return text.strip()
    ans = text.split("Answer:", 1)[1]
    # AB.RMS.3* has TWO Answer: blocks; for diffing we want the FULL trailing
    # text (everything after the first Answer:).
    ans = _FORCE_BLOCK.sub("", ans)
    ans = _FORCE_INLINE.sub("", ans)
    return ans.rstrip()

def main():
    buggy = {t["shortname"]: t["text"] for t in json.load(open(BUGGY_TMPLS))}
    fixed = {t["shortname"]: t["text"] for t in json.load(open(FIXED_TMPLS))}

    junk_by_sn: dict[str, str] = {}
    for sn, btext in buggy.items():
        ftext = fixed.get(sn)
        if ftext is None:
            continue
        b_ans = _normalise_answer(btext)
        f_ans = _normalise_answer(ftext)
        if b_ans == f_ans:
            continue
        # Buggy must end with extra text on top of fixed; sanity-check
        if not b_ans.startswith(f_ans):
            print(f"  SKIP {sn}: fixed-answer is not a prefix of buggy-answer")
            print(f"    fixed[{len(f_ans)}]: {f_ans[-100:]!r}")
            print(f"    buggy[{len(b_ans)}]: {b_ans[-100:]!r}")
            continue
        junk = b_ans[len(f_ans):].strip("\n").strip()
        if not junk:
            continue
        junk_by_sn[sn] = junk

    print(f"templates with stripable trailing junk: {len(junk_by_sn)}")
    for sn, junk in junk_by_sn.items():
        print(f"  {sn:14s} -> strip {len(junk):4d} chars: {junk[:80]!r}")
    print()

    if "--apply" not in sys.argv:
        print("(dry run; pass --apply to write changes)")
        return

    if not BACKUP_PATH.exists():
        shutil.copy(DATA_PATH, BACKUP_PATH)
        print(f"backup written: {BACKUP_PATH}")

    data = json.load(open(DATA_PATH))
    n_in = len(data)
    fixed_rows = 0
    untouched_per_sn: dict[str, int] = {}
    for r in data:
        sn = r.get("shortname")
        junk = junk_by_sn.get(sn)
        if not junk:
            continue
        out = r.get("output", "") or ""
        # strip trailing whitespace, then if output ends with junk (allowing
        # internal whitespace differences), drop it.
        # Look for the literal junk substring; if present, truncate at its
        # earliest occurrence after the rendered content.
        idx = out.find(junk)
        if idx == -1:
            untouched_per_sn[sn] = untouched_per_sn.get(sn, 0) + 1
            continue
        new_out = out[:idx].rstrip()
        if new_out != out:
            r["output"] = new_out
            fixed_rows += 1

    print(f"rows fixed: {fixed_rows} / {n_in}")
    if untouched_per_sn:
        print("untouched (junk substring not found) per template:")
        for sn, c in sorted(untouched_per_sn.items()):
            print(f"  {sn:14s} {c:4d}")

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"wrote: {DATA_PATH}")

if __name__ == "__main__":
    main()
