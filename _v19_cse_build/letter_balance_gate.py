#!/usr/bin/env python3
"""v19 letter-balance gate -- carried verbatim from v17.1, the build-time
check that catches the v17 mode-collapse defect at the corpus level before
training spend.

For each row in the v19 clean corpus, parse the rendered output for the
correct_answers letter list (both wrapped <json_object>{...}</json_object>
and bare {...} shapes are supported, mirroring the two CSE output shapes
in the v19_cse manifest). Aggregate three diagnostic distributions:

  (a) per-letter histogram across A-H (sum over all correct slots).
  (b) per-(sorted) combo histogram (e.g. ("A","B"), ("C",), (), ...).
  (c) unique combo count.

The gate FAILS the build (non-zero exit) if any of:

  - any of A-E (letters required by all 5-option templates) carries
    <MIN_LETTER_PCT or >MAX_LETTER_PCT of total correct slots;
  - any single (sorted) correct_answers tuple carries
    >=MAX_COMBO_PCT of corpus rows;
  - the count of distinct correct_answers tuples is <MIN_UNIQUE_COMBOS.

Defaults (lifted from v17_1_plan.txt §5 / v19_plan.txt and Sophia-CTI-
Templates-v19_cse.txt "Diagnostic expectation under Shuffle: mcq_multi"):

  MIN_LETTER_PCT     = 8.0
  MAX_LETTER_PCT     = 32.0
  MAX_COMBO_PCT      = 15.0
  MIN_UNIQUE_COMBOS  = 20

The combinatorial ceiling for the v19_cse manifest is 26 distinct combos:
the 14 templates emit shapes with 0/1/2/3 correct answers across 5 option
positions A-E, so the maximum reachable combos are
1 (zero correct) + C(5,1) + C(5,2) + C(5,3) = 1 + 5 + 10 + 10 = 26.
The 20-combo floor leaves headroom (~77% of the ceiling) so a future
manifest that drops a shape (e.g. removes the 0-correct NEG.* templates)
still passes provided the remaining shapes shuffle uniformly.

Reads from the post-dedup clean corpus by default
(SFT/data/ift_data_2026_05_15_v19_cse.clean.json) and writes a JSON
report to _v19_cse_build/letter_balance_gate_report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


WRAPPED_RE = re.compile(
    r'<json_object>\s*\{\s*"correct_answers"\s*:\s*\[([^\]]*)\]\s*\}\s*</json_object>'
)
BARE_RE = re.compile(
    r'\{\s*"correct_answers"\s*:\s*\[([^\]]*)\]\s*\}'
)
LETTER_RE = re.compile(r'"([A-H])"')


def parse_letters(text: str) -> tuple[str, ...] | None:
    m = WRAPPED_RE.search(text)
    if m is None:
        m = BARE_RE.search(text)
    if m is None:
        return None
    inside = m.group(1).strip()
    if inside == "":
        return ()
    return tuple(sorted(LETTER_RE.findall(inside)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("SFT/data/ift_data_2026_05_15_v19_cse.clean.json"))
    p.add_argument("--report", type=Path,
                   default=Path("_v19_cse_build/letter_balance_gate_report.json"))
    p.add_argument("--min-letter-pct", type=float, default=8.0)
    p.add_argument("--max-letter-pct", type=float, default=32.0)
    p.add_argument("--max-combo-pct", type=float, default=15.0)
    p.add_argument("--min-unique-combos", type=int, default=20)
    args = p.parse_args()

    rows = json.loads(args.input.read_text())
    print(f"[load] {len(rows):,} rows from {args.input}", file=sys.stderr)

    letter_hist: Counter[str] = Counter()
    combo_hist: Counter[tuple[str, ...]] = Counter()
    unparsed = 0
    for r in rows:
        text = r.get("output", "")
        combo = parse_letters(text)
        if combo is None:
            unparsed += 1
            continue
        combo_hist[combo] += 1
        letter_hist.update(combo)

    n_rows = len(rows) - unparsed
    n_slots = sum(letter_hist.values())
    failures: list[str] = []

    letter_pcts = {L: 100.0 * letter_hist.get(L, 0) / max(n_slots, 1)
                   for L in "ABCDE"}
    for L, pct in letter_pcts.items():
        if pct < args.min_letter_pct or pct > args.max_letter_pct:
            failures.append(
                f"letter {L} carries {pct:.2f}% of correct slots "
                f"(allowed band [{args.min_letter_pct:.1f}, {args.max_letter_pct:.1f}])"
            )

    top_combo, top_combo_n = combo_hist.most_common(1)[0]
    top_combo_pct = 100.0 * top_combo_n / max(n_rows, 1)
    if top_combo_pct >= args.max_combo_pct:
        failures.append(
            f"combo {list(top_combo)} carries {top_combo_pct:.2f}% of rows "
            f"(allowed <{args.max_combo_pct:.1f}%)"
        )

    if len(combo_hist) < args.min_unique_combos:
        failures.append(
            f"only {len(combo_hist)} distinct correct_answers combos "
            f"(allowed >={args.min_unique_combos})"
        )

    report = {
        "input": str(args.input),
        "rows": len(rows),
        "parsed_rows": n_rows,
        "unparsed_rows": unparsed,
        "correct_slots": n_slots,
        "letter_histogram": dict(sorted(letter_hist.items())),
        "letter_pcts_AE": letter_pcts,
        "unique_combos": len(combo_hist),
        "top10_combos": [
            {"combo": list(k), "n": v, "pct": round(100.0 * v / max(n_rows, 1), 3)}
            for k, v in combo_hist.most_common(10)
        ],
        "thresholds": {
            "min_letter_pct": args.min_letter_pct,
            "max_letter_pct": args.max_letter_pct,
            "max_combo_pct": args.max_combo_pct,
            "min_unique_combos": args.min_unique_combos,
        },
        "failures": failures,
        "outcome": "ok" if not failures else "fail",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
