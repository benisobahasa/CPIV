"""Benchmark specification: which conditions go into which split."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..config import CAUSES, TRAIN_PAIRS, UNKNOWN_CAUSES


@dataclass
class Condition:
    causes: list[str]          # ordered treatment chain; [] means identity (no change)
    severity: float
    split: str                 # train | val | test_seen | test_severity | test_combo | test_unknown
    unknown: bool = False


def _all_pairs() -> list[tuple[str, str]]:
    return [p for p in itertools.combinations(CAUSES, 2)]


def heldout_pairs() -> list[tuple[str, str]]:
    seen = {tuple(sorted(p)) for p in TRAIN_PAIRS}
    return [p for p in _all_pairs() if tuple(sorted(p)) not in seen]


def conditions_for_source(split_group: str, rng: np.random.Generator, per_cause: int = 2,
                          n_identity: int = 1, n_pairs: int = 2, n_unknown: int = 1) -> list[Condition]:
    """Conditions generated for one clean source utterance.

    ``split_group`` is the *source-level* partition (train / val / test), which keeps
    train and test utterances (and speakers, when grouped by speaker) disjoint.
    Severity regimes: training uses [0.2, 0.65]; test_severity uses [0.8, 1.0].
    """
    out: list[Condition] = []
    if split_group in ("train", "val"):
        for _ in range(n_identity):
            out.append(Condition([], 0.0, split_group))
        for c in CAUSES:
            for _ in range(per_cause):
                out.append(Condition([c], float(rng.uniform(0.2, 0.65)), split_group))
        for a, b in TRAIN_PAIRS:
            for _ in range(max(1, n_pairs // 2)):
                out.append(Condition([a, b], float(rng.uniform(0.2, 0.65)), split_group))
    else:  # test sources feed four test splits
        for _ in range(n_identity):
            out.append(Condition([], 0.0, "test_seen"))
        for c in CAUSES:
            for _ in range(per_cause):
                out.append(Condition([c], float(rng.uniform(0.2, 0.65)), "test_seen"))
            for _ in range(per_cause):
                out.append(Condition([c], float(rng.uniform(0.8, 1.0)), "test_severity"))
        hp = heldout_pairs()
        for i in rng.permutation(len(hp))[:n_pairs * 3]:
            out.append(Condition(list(hp[i]), float(rng.uniform(0.2, 0.65)), "test_combo"))
        for _ in range(n_unknown * len(UNKNOWN_CAUSES)):
            u = UNKNOWN_CAUSES[int(rng.integers(len(UNKNOWN_CAUSES)))]
            out.append(Condition([u], float(rng.uniform(0.3, 0.9)), "test_unknown", unknown=True))
    return out
