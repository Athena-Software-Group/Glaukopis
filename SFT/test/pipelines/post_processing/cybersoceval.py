"""Answer extraction for CyberSOCEval (CrowdStrike + Meta) tasks.

Mirrors the upstream ``process_judge_prompt`` extraction order from
PurpleLlama's ``crwd_meta`` benchmarks so scores stay comparable to the
public leaderboard:

  1. ``<json_object>...</json_object>`` block (threat_intel_reasoning).
  2. ``$\\boxed{A,B,C}$`` LaTeX-style fallback (threat_intel_reasoning).
  3. Bare ``{...}`` JSON object via balanced-brace extraction
     (malware_analysis primary path).

The returned string is a canonical ``A,B,C`` form (sorted, upper-cased,
deduplicated) so the evaluator can split on comma without re-parsing.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, List


_JSON_OBJECT_RE = re.compile(r"<json_object>(.*?)</json_object>", re.DOTALL)
_BOXED_RE = re.compile(r"\$\\boxed\{([^}]*)\}\$")
_LETTER_RE = re.compile(r"\b([A-J])\b")


def _extract_json_object(text: str):
    """Port of PurpleLlama's ``benchmark_utils.extract_json``.

    Returns the first balanced JSON object parsed from ``text`` (dict),
    or ``None`` if no balanced object is found / parse fails.
    """
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        start = text.index("{")
    except ValueError:
        return None
    brace_count = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _canonicalize(letters: Iterable[object]) -> str:
    """Return canonical ``A,B,C`` form: sorted, deduplicated, A-J only."""
    out: List[str] = []
    seen = set()
    for item in letters or []:
        if item is None:
            continue
        s = str(item).strip().upper()
        if not s:
            continue
        # An entry like "A" or "A. ..." should reduce to its first letter.
        m = _LETTER_RE.search(s)
        if not m:
            continue
        L = m.group(1)
        if L not in seen:
            seen.add(L)
            out.append(L)
    out.sort()
    return ",".join(out)


def _extract_letters_from_response(response: str) -> str:
    """Apply the upstream extraction pipeline and return canonical letters.

    Returns ``""`` when no parseable answer was found (logged as a parse
    error by the evaluator, matching ``response_parsing_error_count`` in
    PurpleLlama's stats).
    """
    if not response:
        return ""
    flat = response.replace("\r", "").replace("\n", " ")

    m = _JSON_OBJECT_RE.search(flat)
    if m:
        parsed = _extract_json_object(m.group(1))
        if isinstance(parsed, dict):
            ans = parsed.get("correct_answers")
            if isinstance(ans, list):
                return _canonicalize(ans)

    m = _BOXED_RE.search(flat)
    if m:
        body = m.group(1).replace(" ", "")
        return _canonicalize(body.split(","))

    parsed = _extract_json_object(flat)
    if isinstance(parsed, dict):
        ans = parsed.get("correct_answers")
        if isinstance(ans, list):
            return _canonicalize(ans)

    return ""


class cybersoceval_postprocessing:
    """Single-method extractor mirroring ``athena_cti_postprocessing``."""

    TASKS = ("cybersoceval-malware", "cybersoceval-ti")

    def extract_answer(self, task: str, text: str, prompt: str = "") -> str:
        if task not in self.TASKS:
            return ""
        return _extract_letters_from_response(text or "")


def canonicalize_answer_list(letters: Iterable[object]) -> str:
    """Public helper used by the benchmark to canonicalize ground-truth."""
    return _canonicalize(letters)
