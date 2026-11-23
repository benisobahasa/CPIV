"""High-level convenience API."""

from __future__ import annotations

import numpy as np

from .attribution import AttributionResult, CausalAttributor, explain
from .audio import load_audio
from .config import SAMPLE_RATE
from .features import feature_vector


def attribute_pair(model: CausalAttributor | str, test, reference=None, sr: int = SAMPLE_RATE,
                   top_k: int = 5) -> AttributionResult:
    """Attribute the change between ``reference`` and ``test`` (arrays or file paths).

    The model must have been trained in the same mode: paired (reference given) or blind.
    """
    if isinstance(model, str):
        model = CausalAttributor.load(model)
    t = load_audio(test, sr) if isinstance(test, str) else np.asarray(test, np.float32)
    r = None if reference is None else (load_audio(reference, sr) if isinstance(reference, str)
                                        else np.asarray(reference, np.float32))
    n = min(len(t), len(r)) if r is not None else len(t)
    x, names = feature_vector(t[:n], None if r is None else r[:n], sr)
    if names != model.feature_names:
        raise ValueError("Feature layout mismatch: model was trained in "
                         f"{'paired' if model.feature_names[0].startswith('delta') else 'blind'} mode.")
    return explain(model, x, top_k=top_k)
