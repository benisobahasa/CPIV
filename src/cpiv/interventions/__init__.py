"""Controlled interventions with known ground-truth cause."""

from __future__ import annotations

import numpy as np

from .acoustic import Replay, Reverberation
from .base import Intervention
from .channel import AdditiveNoise, CodecCompression, TelephoneChannel
from .unknown import BitCrush, PhaseNoise, SpectralHole, TimeWarp
from .voice import (
    ExternalVoiceConversion,
    NaturalVariation,
    PitchManipulation,
    VoiceConversionProxy,
)

REGISTRY: dict[str, type[Intervention]] = {
    "telephone_channel": TelephoneChannel,
    "codec_compression": CodecCompression,
    "replay": Replay,
    "additive_noise": AdditiveNoise,
    "reverberation": Reverberation,
    "voice_conversion": VoiceConversionProxy,
    "pitch_manipulation": PitchManipulation,
    "natural_variation": NaturalVariation,
    "unk_bitcrush": BitCrush,
    "unk_spectral_hole": SpectralHole,
    "unk_time_warp": TimeWarp,
    "unk_phase_noise": PhaseNoise,
}


def get_intervention(name: str, **kwargs) -> Intervention:
    if name not in REGISTRY:
        raise KeyError(f"Unknown intervention '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def apply_chain(y: np.ndarray, sr: int, causes: list[str], rng: np.random.Generator,
                severity: float = 0.5, **kwargs):
    """Apply several interventions in sequence. Returns (audio, list_of_param_dicts)."""
    out, log = y, []
    for c in causes:
        out, p = get_intervention(c, **kwargs.get(c, {}))(out, sr, rng, severity)
        log.append(p)
    return out, log


__all__ = ["REGISTRY", "Intervention", "get_intervention", "apply_chain",
           "ExternalVoiceConversion", "TelephoneChannel", "CodecCompression", "Replay",
           "AdditiveNoise", "Reverberation", "VoiceConversionProxy", "PitchManipulation",
           "NaturalVariation", "BitCrush", "SpectralHole", "TimeWarp", "PhaseNoise"]
