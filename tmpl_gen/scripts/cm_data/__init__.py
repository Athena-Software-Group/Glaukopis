"""CM.* knowledge tables and generators for v12 cm_generator.py.

Each module exports a generate(rng, target_rows, instruction, shortname,
make_mcq) -> list[dict] function. Knowledge is encoded as structured
Python literals so the generator is fully reproducible and free of any
external (potentially-contaminating) MCQ banks.
"""

from __future__ import annotations

import random


def pick_distractors(rng: random.Random, pool: list, correct, k: int = 4) -> list:
    """Return k distinct distractors drawn from pool, excluding correct."""
    cands = [c for c in pool if c != correct]
    rng.shuffle(cands)
    return cands[:k]
