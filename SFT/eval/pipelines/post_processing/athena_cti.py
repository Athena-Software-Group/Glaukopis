"""Utility functions to extract answers from model outputs for each task."""

from __future__ import annotations
import re
from typing import Callable, Dict

class athena_cti_postprocessing:
    # Strip common prefixes like "Answer:", "Final Answer:", etc.
    _PREFIX_RE = re.compile(
        r'^\s*(?:final\s+answer|answer|prediction|output|result)\s*[:\-–—]?\s*',
        re.IGNORECASE,
    )

    # Explicit-commitment pattern for MCQ answers: matches
    # "answer is B", "the correct answer: C", "option would be D", "choice = E",
    # including optional qualifiers (final/correct/best/right) and a leading "("
    # before the letter. Used as tier 1 of the MCQ extractor.
    _MCQ_ANSWER_IS_X = re.compile(
        r'(?:final\s+|correct\s+|best\s+|right\s+)*'
        r'(?:answer|choice|option)\s*(?:is|would\s+be|:|=)\s*\(?([A-E])\b',
        re.IGNORECASE,
    )

    # Parse option lines from a formatted MCQ prompt, e.g. "A) T0822 External..."
    # Accepts ')', '.', '-', or ':' as the letter/body delimiter.
    _MCQ_OPTION_LINE = re.compile(r'^\s*\(?([A-E])[\)\.\-:]\s*(.+?)\s*$')

    def _strip_prefix(self, s: str) -> str:
        return self._PREFIX_RE.sub("", s).strip()

    @staticmethod
    def _last_match(pattern: str, line: str):
        """Return the last regex match on *line*, or None.

        Model outputs commit to their final answer at the end of the line
        ("...first-stage reasoning... Therefore, B."), so the last match is
        the correct one to return. Using ``re.search`` (first match) causes
        incidental tokens earlier in the line -- articles like "a", option
        letters quoted from the question body ("E) T1041 ..."), or midway
        reasoning -- to hijack the extracted answer.
        """
        matches = re.findall(pattern, line, re.IGNORECASE)
        return matches[-1] if matches else None

    def _extract_from_lines(self, text: str, pattern: str, transform=lambda x: x) -> str:
        """Search *text* from bottom to top and return the last regex match."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        for i in range(len(lines) - 1, -1, -1):
            raw = lines[i]
            line = self._strip_prefix(raw)

            m = self._last_match(pattern, line)
            if m is not None:
                return transform(m)

            # Check neighbors if "Answer" label is present
            if re.search(r'\banswer\b', raw, re.IGNORECASE):
                if i + 1 < len(lines):
                    nxt = self._strip_prefix(lines[i + 1])
                    m = self._last_match(pattern, nxt)
                    if m is not None:
                        return transform(m)
                if i > 0:
                    prv = self._strip_prefix(lines[i - 1])
                    m = self._last_match(pattern, prv)
                    if m is not None:
                        return transform(m)
        return ""

    def _clean_freeform(self, s: str) -> str:
        s = s.strip().strip('"\'')   # trim & remove quotes
        return re.sub(r"\s+", " ", s)

    def athena_rcm_answer(self, text: str) -> str:
        return self._extract_from_lines(text, r"(CWE-\d+)", lambda s: s.upper())

    def athena_vsp_answer(self, text: str) -> str:
        # Character class restricts to legal CVSS v3.1 vector chars so a
        # trailing period / punctuation from the trained conclusion tail
        # ("... is CVSS:3.1/AV:N/.../A:H.") is not captured by the greedy
        # match and therefore does not break the CVSS3(...) parse downstream.
        return self._extract_from_lines(text, r"(CVSS:3\.1/[A-Z:/]+)", lambda s: s.strip())

    def athena_taa_answer(self, text: str) -> str:
        return self._extract_from_lines(text, r"(.+)", self._clean_freeform)

    def athena_taa_canonical_answer(self, text: str) -> str:
        # The canonical-resolution prompt instructs the model to commit on a
        # final 'Answer:' line. Mirror the TAA extractor for the freeform
        # tail so the scorer can substring-match against canonical names
        # and G-codes (G####) in either order.
        return self._extract_from_lines(text, r"(.+)", self._clean_freeform)

    def athena_rms_answer(self, text: str) -> str:
        # Mitigation IDs may appear anywhere in the response (preamble,
        # body, or a closing summary). Verbose SFT outputs commonly cite
        # IDs in early lines and explain each one in following paragraphs,
        # leaving the bottom-most non-empty line ID-free; the previous
        # last-line-only scan dropped those IDs entirely. The scorer
        # collapses to set(re.findall(M\d{4}, pred)) regardless of
        # position, so the extractor mirrors that scope: scan the whole
        # response, dedupe in first-seen order, comma-join.
        if not text:
            return ""
        seen: list[str] = []
        for m in re.findall(r"M\d{4}", text.upper()):
            if m not in seen:
                seen.append(m)
        return ", ".join(seen)

    def athena_ate_answer(self, text: str) -> str:
        tid = self._extract_from_lines(text, r"(T\d{4}(?:\.\d{3})?)", lambda s: s.upper())
        return tid.split(".")[0]

    def _parse_mcq_options(self, prompt: str) -> Dict[str, str]:
        """Extract {letter: option_text} from a formatted MCQ prompt."""
        opts: Dict[str, str] = {}
        for ln in (prompt or "").splitlines():
            m = self._MCQ_OPTION_LINE.match(ln)
            if m:
                L = m.group(1).upper()
                if L not in opts:
                    opts[L] = m.group(2).strip()
        return opts

    def athena_mcq_answer(self, text: str, prompt: str = "") -> str:
        """Extract an A-E MCQ letter using a three-tier strategy.

        Tier 1 -- explicit commitment. Take the last global match of
        "(answer|choice|option) (is|:|=) X"; the model's final stated
        choice wins over stray letters that appear in reasoning text.

        Tier 2 -- last bare \\b[A-E]\\b per line, bottom-up. The previous
        behavior; handles "Answer: C" and trailing "...is D." cleanly.

        Tier 3 -- verbatim option-text match against the prompt. Only
        fires when the model echoed option content without committing to
        a letter (e.g. "...is T1583.006 Acquire Infrastructure: Web
        Services."). Requires >=8 chars to avoid false positives; picks
        the longest matching option when several overlap.
        """
        if not text:
            return ""

        explicit = self._MCQ_ANSWER_IS_X.findall(text)
        if explicit:
            return explicit[-1].upper()

        letter = self._extract_from_lines(text, r"\b([A-E])\b", lambda s: s.upper())
        if letter:
            return letter

        if prompt:
            options = self._parse_mcq_options(prompt)
            if options:
                lower = text.lower()
                hits = []
                for L, opt in options.items():
                    opt = (opt or "").strip()
                    if len(opt) >= 8 and opt.lower() in lower:
                        hits.append((len(opt), L))
                if hits:
                    hits.sort(reverse=True)
                    return hits[0][1]

        return ""

    def extract_answer(self, task: str, text: str, prompt: str = "") -> str:
        """Return the parsed answer for *task* from *text*.

        *prompt* is optional and currently only consumed by the MCQ
        extractor's option-text fallback; other tasks ignore it.
        """
        if task == "athena-mcq":
            return self.athena_mcq_answer(text, prompt)

        extractors: Dict[str, Callable[[str], str]] = {
            "athena-rcm": self.athena_rcm_answer,
            "athena-vsp": self.athena_vsp_answer,
            "athena-taa": self.athena_taa_answer,
            "athena-taa-canonical": self.athena_taa_canonical_answer,
            "athena-rms": self.athena_rms_answer,
            "athena-ate": self.athena_ate_answer,
        }
        func = extractors.get(task)
        return func(text) if func else ""
