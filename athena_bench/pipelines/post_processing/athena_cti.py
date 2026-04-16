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

    def _strip_prefix(self, s: str) -> str:
        return self._PREFIX_RE.sub("", s).strip()

    def _extract_from_lines(self, text: str, pattern: str, transform=lambda x: x) -> str:
        """Search *text* from bottom to top and return the last regex match."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        for i in range(len(lines) - 1, -1, -1):
            raw = lines[i]
            line = self._strip_prefix(raw)

            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return transform(match.group(1))

            # Check neighbors if "Answer" label is present
            if re.search(r'\banswer\b', raw, re.IGNORECASE):
                if i + 1 < len(lines):
                    nxt = self._strip_prefix(lines[i + 1])
                    match = re.search(pattern, nxt, re.IGNORECASE)
                    if match:
                        return transform(match.group(1))
                if i > 0:
                    prv = self._strip_prefix(lines[i - 1])
                    match = re.search(pattern, prv, re.IGNORECASE)
                    if match:
                        return transform(match.group(1))
        return ""

    def _clean_freeform(self, s: str) -> str:
        s = s.strip().strip('"\'')   # trim & remove quotes
        return re.sub(r"\s+", " ", s)

    def athena_rcm_answer(self, text: str) -> str:
        return self._extract_from_lines(text, r"(CWE-\d+)", lambda s: s.upper())

    def athena_vsp_answer(self, text: str) -> str:
        return self._extract_from_lines(text, r"(CVSS:3\.1/[^\s]+)", lambda s: s.strip())

    def athena_taa_answer(self, text: str) -> str:
        return self._extract_from_lines(text, r"(.+)", self._clean_freeform)

    def athena_rms_answer(self, text: str) -> str:
        line = self._extract_from_lines(text, r"(.+)", self._clean_freeform).upper()
        ids = re.findall(r"M\d{4}", line)
        return ", ".join(ids)

    def athena_ate_answer(self, text: str) -> str:
        tid = self._extract_from_lines(text, r"(T\d{4}(?:\.\d{3})?)", lambda s: s.upper())
        return tid.split(".")[0]

    def extract_answer(self, task: str, text: str) -> str:
        """Return the parsed answer for *task* from *text*."""
        extractors: Dict[str, Callable[[str], str]] = {
            "athena-rcm": self.athena_rcm_answer,
            "athena-vsp": self.athena_vsp_answer,
            "athena-taa": self.athena_taa_answer,
            "athena-rms": self.athena_rms_answer,
            "athena-ate": self.athena_ate_answer,
            "athena-mcq": lambda text: self._extract_from_lines(text, r"\b([A-E])\b", lambda s: s.upper()),
        }
        func = extractors.get(task)
        return func(text) if func else ""
