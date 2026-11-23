"""Minimal Python API walk-through (runs on synthetic voices, no downloads)."""

import numpy as np

from cpiv import CAUSES
from cpiv.api import attribute_pair
from cpiv.audio import synth_voice
from cpiv.benchmark import build_dataset, synthetic_sources, train_and_evaluate
from cpiv.interventions import apply_chain

# 1) Build a small benchmark (replace synthetic_sources with collect_sources("path/to/corpus"))
data = build_dataset(synthetic_sources(10, 2, seed=0, duration=2.5), per_cause=1, seed=0)

# 2) Train + evaluate on all protocols
res = train_and_evaluate(data, mode="paired", backend="rf", n_estimators=150)
model = res["_model"]
print({k: round(v["micro_f1"], 3) for k, v in res["protocols"].items() if "micro_f1" in v})

# 3) Explain a new pair: reference vs. a telephone + noise version of it
ref = synth_voice(seed=999, duration=2.5)
test, _ = apply_chain(ref, 16000, ["telephone_channel", "additive_noise"], np.random.default_rng(1), 0.6)
print(attribute_pair(model, test, ref).text())
print("known causes:", CAUSES)
