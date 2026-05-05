"""SOC.*.GEN.* knowledge tables and generators for v12 soc_generator.py.

Coverage is complementary to the template-driven SOC families: templates
bind to SigmaHQ rule and malware-family seeds whose v12 yields saturate
at ~5K rows; the generator covers the structural and procedural SOC
knowledge those seeds do not exercise (sigma rule structure, malware
behaviour catalogues, NIST 800-61 phase actions, alert triage rubric).
"""

from __future__ import annotations

import random


def pick_distractors(rng: random.Random, pool: list, correct, k: int = 4) -> list:
    """Return k distinct distractors drawn from pool, excluding correct."""
    cands = [c for c in pool if c != correct]
    rng.shuffle(cands)
    return cands[:k]
