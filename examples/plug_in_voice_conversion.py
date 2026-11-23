"""Plug a real voice-conversion / voice-cloning system into the benchmark.

CPIV ships only a lightweight proxy for voice conversion. For scientific claims, wrap your
own system as ``fn(y, sr) -> y2`` and register it so it replaces the proxy.
"""

import numpy as np

from cpiv.audio import synth_voice
from cpiv.interventions import REGISTRY, ExternalVoiceConversion


def my_vc(y: np.ndarray, sr: int) -> np.ndarray:
    # Call your model here (e.g. a Python wrapper around an open-source VC system).
    return y[::-1].copy()  # placeholder


REGISTRY["voice_conversion"] = lambda: ExternalVoiceConversion(my_vc, name="my_vc_v1")

from cpiv.interventions import get_intervention  # noqa: E402

out, params = get_intervention("voice_conversion")(synth_voice(1), 16000)
print(params)
