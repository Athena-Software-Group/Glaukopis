#!/usr/bin/env python3
"""Shuffle MCQ options and rewrite outputs to use "Answer: X" format.

The upstream MCQ templates in Sophia-CTI-Templates-AthenaBench-aligned.txt
and Sophia-CTI-Templates-04022026-benchmark-addendum.txt bind the correct
answer to position A and append "Therefore, A." to every rendered output.
Training on that data teaches the model to always answer A, in a
"<justification>. Therefore, A." cadence, without emitting the "Answer: X"
form the AthenaBench MCQ prompt asks for.

This is a one-shot post-processing fix applied to the already-generated
Alpaca JSON dataset:
  * Each MCQ row's options are shuffled (correct answer moves to a
    uniformly random letter, distractors reshuffled among remaining slots).
  * The output's trailing ". Therefore, A." (or any single-letter variant)
    is stripped and replaced with "\nAnswer: <new_letter>" on its own line,
    preserving the justification text.
  * The original option count per row (4 from Q.* templates, 5 from
    AB.MCQ.* templates) is preserved so the model sees both formats.

Usage:
  python mcq_shuffle_and_reformat.py \
      --input  SFT/data/ift_data_2026_04_20.json \
      --output SFT/data/ift_data_2026_04_21_mcq_fixed.json

Run with --self-test to execute inline unit tests instead.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

MCQ_SHORTNAMES = {
    "AB.MCQ.1", "AB.MCQ.2", "AB.MCQ.3", "AB.MCQ.4", "AB.MCQ.5", "AB.MCQ.6",
    "Q.GM.1", "Q.VW.1", "Q.MN.1", "Q.WA.1", "Q.AM.1", "Q.MT.1",
}
LETTERS = "ABCDE"
OPTION_LINE_RE = re.compile(r"^([A-E])\)\s*(.*)$")
THEREFORE_RE = re.compile(r"\s*\.?\s*Therefore,?\s+[A-E]\.?\s*$", re.IGNORECASE)
CORRECT_LETTER_RE = re.compile(r"Therefore,?\s+([A-E])\.?\s*$", re.IGNORECASE)


def is_mcq_row(rec: dict) -> bool:
    if rec.get("shortname", "") in MCQ_SHORTNAMES:
        return True
    return bool(re.search(r"^[A-E]\)", rec.get("input", ""), re.MULTILINE))


def parse_options(text: str):
    """Split *text* into (stem, [(letter, content), ...]).

    Supports multi-line option content: a line starting with "X) " begins
    a new option; subsequent lines until the next option marker are
    appended to the current option's content.
    """
    lines = text.splitlines()
    stem_lines, options = [], []
    current_letter, current_content = None, []
    for line in lines:
        m = OPTION_LINE_RE.match(line)
        if m:
            if current_letter is not None:
                options.append((current_letter, "\n".join(current_content).rstrip()))
            current_letter = m.group(1)
            current_content = [m.group(2)]
        elif current_letter is None:
            stem_lines.append(line)
        else:
            current_content.append(line)
    if current_letter is not None:
        options.append((current_letter, "\n".join(current_content).rstrip()))
    stem = "\n".join(stem_lines).rstrip()
    return stem, options


def shuffle_and_rewrite(rec: dict, rng: random.Random):
    """Return (new_rec, changed, new_letter). *changed* is False when the
    row is not MCQ-shaped or has too few parseable options.
    """
    if not is_mcq_row(rec):
        return rec, False, None

    stem, options = parse_options(rec.get("input", ""))
    if len(options) < 2:
        return rec, False, None

    # Determine the current correct option. Legacy templates anchor the
    # correct answer at position A; templates that emit Shuffle: mcq at
    # render time encode the correct letter in the trailing "Therefore, X."
    # of the output. Honour that letter when present so already-shuffled
    # rows are re-shuffled correctly instead of corrupted.
    correct_idx_in = 0
    m_correct = CORRECT_LETTER_RE.search(rec.get("output", "").rstrip())
    if m_correct:
        idx_from_tail = LETTERS.index(m_correct.group(1).upper())
        if idx_from_tail < len(options):
            correct_idx_in = idx_from_tail
    correct_text = options[correct_idx_in][1]
    distractors = [c for i, (_l, c) in enumerate(options) if i != correct_idx_in]

    n = len(options)
    new_letters = list(LETTERS[:n])
    correct_idx = rng.randrange(n)
    new_correct_letter = new_letters[correct_idx]
    rng.shuffle(distractors)

    slotted = [None] * n
    slotted[correct_idx] = correct_text
    di = 0
    for i in range(n):
        if slotted[i] is None:
            slotted[i] = distractors[di]
            di += 1

    new_input = stem + "\n" + "\n".join(
        f"{new_letters[i]}) {slotted[i]}" for i in range(n)
    )

    out = rec.get("output", "")
    stripped = THEREFORE_RE.sub("", out).rstrip(" \n.")
    if stripped:
        stripped += "."
    new_output = f"{stripped}\nAnswer: {new_correct_letter}" if stripped else f"Answer: {new_correct_letter}"

    new_rec = dict(rec)
    new_rec["input"] = new_input
    new_rec["output"] = new_output
    return new_rec, True, new_correct_letter


def process_file(input_path: Path, output_path: Path, seed: int = 42):
    rng = random.Random(seed)
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))

    out_data, letter_counts, opt_counts = [], Counter(), Counter()
    n_mcq, n_changed = 0, 0
    for rec in data:
        new_rec, changed, letter = shuffle_and_rewrite(rec, rng)
        out_data.append(new_rec)
        if is_mcq_row(rec):
            n_mcq += 1
        if changed:
            n_changed += 1
            letter_counts[letter] += 1
            opt_counts[len(parse_options(new_rec["input"])[1])] += 1

    Path(output_path).write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"input         : {input_path} ({len(data)} rows)")
    print(f"output        : {output_path}")
    print(f"mcq detected  : {n_mcq}")
    print(f"mcq rewritten : {n_changed}")
    print(f"letter dist   : {dict(sorted(letter_counts.items()))}")
    print(f"option counts : {dict(sorted(opt_counts.items()))}")
    return out_data


def _self_test():
    rng = random.Random(0)
    rec5 = {
        "instruction": "You are a cyber analyst.",
        "input": "Which technique does X use?\nA) T0001 Correct\nB) T0002 Wrong\nC) T0003 Wrong\nD) T0004 Wrong\nE) T0005 Wrong",
        "output": "X uses technique T0001 Correct per MITRE. Therefore, A.",
        "shortname": "AB.MCQ.1",
    }
    new, changed, letter = shuffle_and_rewrite(rec5, rng)
    assert changed and letter in "ABCDE"
    stem, opts = parse_options(new["input"])
    assert len(opts) == 5, opts
    correct_line = next(l for l, c in opts if c == "T0001 Correct")
    assert correct_line == letter, (correct_line, letter)
    assert new["output"].endswith(f"\nAnswer: {letter}")
    assert "Therefore" not in new["output"]
    assert "X uses technique T0001 Correct per MITRE." in new["output"]

    rec4 = {"input": "Q?\nA) A_correct\nB) B_d\nC) C_d\nD) D_d",
            "output": "A_correct is right. Therefore, A.", "shortname": "Q.GM.1"}
    new4, changed4, _ = shuffle_and_rewrite(rec4, rng)
    assert changed4
    assert len(parse_options(new4["input"])[1]) == 4

    not_mcq = {"input": "Just a normal question", "output": "A normal answer", "shortname": "S.1"}
    same, changed_nm, _ = shuffle_and_rewrite(not_mcq, rng)
    assert not changed_nm and same is not_mcq

    counts = Counter()
    for i in range(2000):
        _, _, lt = shuffle_and_rewrite(rec5, random.Random(i))
        counts[lt] += 1
    for lt in "ABCDE":
        assert counts[lt] > 300, (counts, "distribution should be roughly uniform")

    # Pre-shuffled input: correct letter encoded in the Therefore tail (D),
    # not in position A. The rewriter must follow the tail, not assume A.
    rec_preshuffled = {
        "input": "Q?\nA) wrong1\nB) wrong2\nC) wrong3\nD) actually_correct\nE) wrong4",
        "output": "actually_correct is the answer per the data. Therefore, D.",
        "shortname": "Q.MSR.1",
    }
    new_ps, changed_ps, lt_ps = shuffle_and_rewrite(rec_preshuffled, random.Random(7))
    assert changed_ps and lt_ps in "ABCDE"
    _stem, opts_ps = parse_options(new_ps["input"])
    correct_line_ps = next(l for l, c in opts_ps if c == "actually_correct")
    assert correct_line_ps == lt_ps, (correct_line_ps, lt_ps)
    assert new_ps["output"].endswith(f"\nAnswer: {lt_ps}")
    print("self-test OK:", dict(counts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="Alpaca JSON to post-process")
    p.add_argument("--output", type=Path, help="Destination Alpaca JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true", help="Run unit tests and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.input or not args.output:
        p.error("--input and --output are required unless --self-test is set")
    process_file(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
