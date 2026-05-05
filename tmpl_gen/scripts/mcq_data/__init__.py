"""MCQ.EXT.* knowledge tables and generators for v12 mcq_generator.py.

Each module exports a generate(rng, target_rows, instruction, shortname,
make_mcq) -> list[dict] function. Coverage is complementary to the
template-driven AB.MCQ.* / JS.MCQ.* families: the templates traverse
MITRE technique<->tactic|mitigation|group|software|data-source edges,
the generator covers the edges they do NOT traverse (technique<->platform,
sub-tech<->parent, kill-chain phase) plus general security concepts
(OWASP, networking, OS) outside the MITRE ATT&CK ontology.
"""

from __future__ import annotations

import random


def pick_distractors(rng: random.Random, pool: list, correct, k: int = 4) -> list:
    """Return k distinct distractors drawn from pool, excluding correct."""
    cands = [c for c in pool if c != correct]
    rng.shuffle(cands)
    return cands[:k]
