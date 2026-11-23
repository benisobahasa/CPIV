"""Base class for controlled interventions.

Every intervention applies exactly one known treatment to a clean signal, so the
ground-truth cause is known by construction. ``severity`` in [0, 1] scales how strong
the treatment is, which lets the benchmark test generalisation to unseen strengths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..audio import match_length


class Intervention(ABC):
    cause: str = "base"

    @abstractmethod
    def sample_params(self, rng: np.random.Generator, severity: float) -> dict[str, Any]:
        """Draw treatment parameters for a given severity in [0, 1]."""

    @abstractmethod
    def _apply(self, y: np.ndarray, sr: int, params: dict[str, Any],
               rng: np.random.Generator) -> np.ndarray:
        ...

    def __call__(self, y: np.ndarray, sr: int, rng: np.random.Generator | None = None,
                 severity: float = 0.5, params: dict[str, Any] | None = None):
        rng = rng or np.random.default_rng()
        params = params if params is not None else self.sample_params(rng, severity)
        out = self._apply(y.astype(np.float64), sr, params, rng)
        out = match_length(np.nan_to_num(out), len(y)).astype(np.float32)
        return out, {"cause": self.cause, **params}


def lerp(lo: float, hi: float, s: float) -> float:
    return float(lo + (hi - lo) * s)


def jitter_severity(rng: np.random.Generator, severity: float, width: float = 0.15) -> float:
    return float(np.clip(severity + rng.uniform(-width, width), 0.0, 1.0))
